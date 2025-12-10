import cv2
import os
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.analyzer import AnalysisService
from app.services.scoring import ScoringService
from app.services.whisper_service import WhisperService
# from app.services.groq_service import GroqService

class VideoProcessor:
    def __init__(self):
        self.analyzer = AnalysisService()
        self.scorer = ScoringService()
        
        # Initialize Whisper service (lazy loading to avoid startup delay)
        self._whisper = None
    
    @property
    def whisper(self):
        """Lazy load Whisper service."""
        if self._whisper is None:
            try:
                self._whisper = WhisperService()
                # self._whisper = GroqService()  # Use Groq for temporary testing
            except Exception as e:
                logging.error(f"Failed to initialize Whisper service: {e}")
                self._whisper = None
        return self._whisper

    def process(self, input_path, original_filename=None):
        """
        Process video for activity recognition and transcription in parallel.
        
        Args:
            input_path: Path to video file
            original_filename: Original filename for reporting
            
        Returns:
            Dictionary with activity analysis and transcription results
        """
        start_time = time.time()
        
        # Run activity analysis and transcription in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            activity_future = executor.submit(self._analyze_activity, input_path)
            transcription_future = executor.submit(self._transcribe_audio, input_path)
            
            # Wait for both to complete
            activity_result = activity_future.result()
            transcription_result = transcription_future.result()
            logging.info(f"Transcription Result: {transcription_result}")
        
        processing_time = round(time.time() - start_time, 2)
        
        # Combine results
        return {
            "input_file": original_filename if original_filename else os.path.basename(input_path),
            "processing_time_seconds": processing_time,
            "activity_analysis": activity_result,
            "transcription": transcription_result
        }
    
    def _transcribe_audio(self, video_path):
        """
        Transcribe audio from video using Whisper.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Transcription results or error message
        """
        try:
            if self.whisper is None:
                return {
                    "error": "Whisper service not available",
                    "text": "",
                    "segments": []
                }
            
            return self.whisper.transcribe(video_path)
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return {
                "error": str(e),
                "text": "",
                "segments": []
            }
    
    def _analyze_activity(self, input_path):
        """
        Analyze video for activity recognition with optimized frame sampling.
        
        Args:
            input_path: Path to video file
            
        Returns:
            Activity analysis results
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Optimize: Process every 2nd frame for speed (still smooth analysis)
        frame_skip = 2
        
        # Optimize: Check emotion less frequently (every 2 seconds instead of 1)
        emotion_skip = int(fps * 2) if fps > 0 else 60
        
        frame_count = 0
        frames_analyzed = 0
        overall_stats = {
            "total_frames": total_frames,
            "people_detected": 0,
            "scores": {
                "attention": [],
                "posture": [],
                "engagement": [],
                "confidence": []
            },
            "temporal_data": {
                "attention_timeline": [],
                "movement_detected": []
            }
        }
        
        current_emotions = {}  # track_id -> emotion
        previous_positions = {}  # track_id -> (x, y) for movement detection

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Frame sampling optimization
            if frame_count % frame_skip != 0:
                continue
            
            frames_analyzed += 1
            
            # Tracking
            results = self.analyzer.track(frame)
            
            if results and results[0].boxes:
                boxes = results[0].boxes
                keypoints = results[0].keypoints
                overall_stats["people_detected"] = max(overall_stats["people_detected"], len(boxes))
                
                for i, box in enumerate(boxes):
                    track_id = int(box.id[0]) if box.id is not None else i
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Movement detection
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    if track_id in previous_positions:
                        prev_x, prev_y = previous_positions[track_id]
                        movement = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                        overall_stats["temporal_data"]["movement_detected"].append(movement > 20)
                    previous_positions[track_id] = (center_x, center_y)
                    
                    # Emotion Analysis (Less frequent for speed)
                    if frame_count % emotion_skip == 0:
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
                            
                            # Temporal data
                            overall_stats["temporal_data"]["attention_timeline"].append({
                                "frame": frame_count,
                                "score": att_score
                            })
            
        cap.release()
        
        # Calculate statistics
        final_scores = {}
        total_score_sum = 0
        count = 0
        for k, v in overall_stats["scores"].items():
            avg = sum(v) / len(v) if v else 0
            final_scores[k] = round(avg, 2)
            total_score_sum += avg
            count += 1
            
        overall_score = round(total_score_sum / count, 2) if count > 0 else 0
        
        # Enhanced temporal metrics
        movement_percentage = (sum(overall_stats["temporal_data"]["movement_detected"]) / 
                             len(overall_stats["temporal_data"]["movement_detected"]) * 100) if overall_stats["temporal_data"]["movement_detected"] else 0
        
        # Calculate attention consistency (variance)
        attention_scores = overall_stats["scores"]["attention"]
        attention_variance = np.var(attention_scores) if attention_scores else 0
        attention_consistency = max(0, 10 - attention_variance)  # Lower variance = higher consistency
        
        methodology = {
            "attention": "Derived from head yaw and pitch. Closer to center = higher score.",
            "posture": "Derived from shoulder levelness and spine alignment.",
            "engagement": "Weighted combination of Attention score and detected Emotion (Happy/Neutral adds bonus).",
            "confidence": "Based on open posture and lack of negative emotions (Sad/Fear).",
            "overall_score": "Average of Attention, Posture, Engagement, and Confidence.",
            "temporal_metrics": "Movement detection and attention consistency over time."
        }
            
        return {
            "total_frames": total_frames,
            "frames_analyzed": frames_analyzed,
            "people_count": overall_stats["people_detected"],
            "overall_score": overall_score,
            "average_scores": final_scores,
            "temporal_metrics": {
                "movement_percentage": round(movement_percentage, 2),
                "attention_consistency": round(attention_consistency, 2),
                "attention_timeline_sample": overall_stats["temporal_data"]["attention_timeline"][:10]  # First 10 samples
            },
            "scoring_methodology": methodology
        }
