import cv2
import os
import logging
import numpy as np
from app.services.analyzer import AnalysisService
from app.services.scoring import ScoringService

class VideoProcessor:
    def __init__(self):
        self.analyzer = AnalysisService()
        self.scorer = ScoringService()

    def process(self, input_path, original_filename=None):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0
        overall_stats = {
            "total_frames": total_frames,
            "people_detected": 0,
            "scores": {
                "attention": [],
                "posture": [],
                "engagement": [],
                "confidence": []
            }
        }
        
        # DeepFace is slow, so we can't run it every frame efficiently without massive lag or skipping.
        # For uploaded video, we can afford some time, but let's limit it.
        emotion_skip = 30 # Check emotion every 30 frames (approx 1 sec)
        current_emotions = {} # track_id -> emotion

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Tracking
            results = self.analyzer.track(frame)
            
            if results and results[0].boxes:
                boxes = results[0].boxes
                keypoints = results[0].keypoints
                overall_stats["people_detected"] = max(overall_stats["people_detected"], len(boxes))
                
                for i, box in enumerate(boxes):
                    track_id = int(box.id[0]) if box.id is not None else i
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Emotion Analysis (Periodic)
                    if frame_count % emotion_skip == 0:
                         # Crop face
                         # Heuristic: Upper 20% of person box or use keypoints if available
                         # Using box:
                         face_h = (y2 - y1) // 4
                         face_roi = frame[y1:y1+face_h, x1:x2]
                         if face_roi.size > 0:
                             dom, _ = self.analyzer.analyze_emotion(face_roi)
                             if dom:
                                 current_emotions[track_id] = dom
                    
                    current_emotion = current_emotions.get(track_id, "neutral")

                    # Calculate Scores
                    att_score = 0
                    pos_score = 0
                    conf_score = 0
                    eng_score = 0
                    
                    if keypoints is not None and len(keypoints) > i:
                        kps = keypoints[i].xy[0].cpu().numpy()
                        if len(kps) >= 7:
                            # 0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar
                            nose = kps[0]
                            l_eye = kps[1]
                            r_eye = kps[2]
                            l_ear = kps[3]
                            r_ear = kps[4]
                            
                            yaw = self.analyzer.calculate_yaw(nose, l_eye, r_eye, l_ear, r_ear)
                            pitch = self.analyzer.calculate_pitch(nose, l_eye, r_eye)
                            
                            att_score = self.scorer.calculate_attention(yaw, pitch)
                            pos_score = self.scorer.calculate_posture(kps)
                            conf_score = self.scorer.calculate_confidence(kps, current_emotion)
                            eng_score = self.scorer.calculate_engagement(att_score, current_emotion)
                            
                            # Accumulate
                            overall_stats["scores"]["attention"].append(att_score)
                            overall_stats["scores"]["posture"].append(pos_score)
                            overall_stats["scores"]["engagement"].append(eng_score)
                            overall_stats["scores"]["confidence"].append(conf_score)
            
        cap.release()
        
        # Average Scores
        final_scores = {}
        total_score_sum = 0
        count = 0
        for k, v in overall_stats["scores"].items():
            avg = sum(v) / len(v) if v else 0
            final_scores[k] = round(avg, 2)
            total_score_sum += avg
            count += 1
            
        overall_score = round(total_score_sum / count, 2) if count > 0 else 0
        
        methodology = {
            "attention": "Derived from head yaw and pitch. Closer to center = higher score.",
            "posture": "Derived from shoulder levelness and spine alignment.",
            "engagement": "Weighted combination of Attention score and detected Emotion (Happy/Neutral adds bonus).",
            "confidence": "Based on open posture and lack of negative emotions (Sad/Fear).",
            "overall_score": "Average of Attention, Posture, Engagement, and Confidence."
        }
            
        return {
            "input_file": original_filename if original_filename else os.path.basename(input_path),
            "total_frames": total_frames,
            "people_count": overall_stats["people_detected"],
            "overall_score": overall_score,
            "average_scores": final_scores,
            "scoring_methodology": methodology
        }

