import cv2
import math
import time
from ultralytics import YOLO
from deepface import DeepFace
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_yaw(nose, left_eye, right_eye, left_ear, right_ear):
    """
    Estimate head yaw (left/right).
    """
    nx = nose[0]
    lex = left_eye[0]
    rex = right_eye[0]
    leax = left_ear[0]
    reax = right_ear[0]

    if leax > 0 and reax > 0:
        face_width = abs(leax - reax)
        mid_face = (leax + reax) / 2
    else:
        face_width = abs(lex - rex) * 2
        mid_face = (lex + rex) / 2
    
    if face_width == 0:
        return 0

    deviation = nx - mid_face
    yaw_ratio = deviation / (face_width / 2)
    return yaw_ratio

def calculate_pitch(nose, left_eye, right_eye):
    """
    Estimate head pitch (up/down).
    Returns ratio. Positive = Looking Down (Nose below eyes). Negative = Looking Up.
    """
    ny = nose[1]
    ley = left_eye[1]
    rey = right_eye[1]
    
    # Midpoint of eyes (vertical)
    mid_eye_y = (ley + rey) / 2
    
    # Distance between eyes (scale reference)
    eye_dist = abs(left_eye[0] - right_eye[0])
    
    if eye_dist == 0:
        return 0
        
    # Vertical distance from eye line to nose
    # Normal face: Nose is below eyes (positive diff)
    # Looking down: Nose gets further from eyes (larger positive)
    # Looking up: Nose gets closer to eyes (smaller positive or negative)
    
    # Heuristic: 
    # Normal ratio ~ 0.5 to 0.8 depending on face structure
    # Looking down > 1.0
    # Looking up < 0.2
    
    diff_y = ny - mid_eye_y
    ratio = diff_y / eye_dist
    return ratio

class AttentionStateTracker:
    def __init__(self):
        self.states = {} # track_id -> {pose, start_time, status}
        self.threshold_time = 3.0 # seconds

    def update(self, track_id, current_pose):
        now = time.time()
        
        if track_id not in self.states:
            self.states[track_id] = {
                'pose': current_pose,
                'start_time': now,
                'status': 'Attentive' if current_pose == 'Center' else 'Unknown'
            }
        
        state = self.states[track_id]
        
        # Check if pose changed
        if state['pose'] != current_pose:
            state['pose'] = current_pose
            state['start_time'] = now
            # Immediate reset to Attentive if back to center? 
            # Or wait? Let's say Center is immediate Attentive.
            if current_pose == 'Center':
                state['status'] = 'Attentive'
            else:
                state['status'] = 'Analyzing...' # Waiting for threshold
        else:
            # Pose is same, check duration
            duration = now - state['start_time']
            if duration > self.threshold_time:
                if current_pose == 'Side':
                    state['status'] = 'Distracted (Side)'
                elif current_pose == 'Up':
                    state['status'] = 'Distracted (Up)'
                elif current_pose == 'Down':
                    state['status'] = 'Writing/Activity'
                elif current_pose == 'Center':
                    state['status'] = 'Attentive'
            else:
                # Under threshold
                if current_pose != 'Center' and state['status'] == 'Attentive':
                     state['status'] = 'Analyzing...' # Transitioning
        
        return state['status'], state['pose'], (now - state['start_time'])

def analyze_emotion(face_img, track_id):
    """
    Background task to analyze emotion.
    Returns (track_id, emotion)
    """
    try:
        objs = DeepFace.analyze(img_path=face_img, 
                                actions=['emotion'], 
                                enforce_detection=False, 
                                silent=True)
        if objs:
            return track_id, objs[0]['dominant_emotion']
    except Exception as e:
        pass
    return track_id, None

def main():
    # 1. Load Models
    logging.info("Loading YOLOv8-pose model...")
    model = YOLO('yolov8n-pose.pt')
    
    logging.info("Initializing DeepFace (this might take a moment on first run)...")
    # Warmup DeepFace
    try:
        DeepFace.analyze(img_path=np.zeros((100, 100, 3), dtype=np.uint8), actions=['emotion'], enforce_detection=False, silent=True)
    except:
        pass

    # 2. Open Video Source
    cap = cv2.VideoCapture(0) # Default webcam
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return
    
    # Set resolution and FPS for better performance
    # Balanced resolution for quality and smoothness
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height
    cap.set(cv2.CAP_PROP_FPS, 60)  # Target FPS
    
    logging.info(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    logging.info(f"Camera FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")

    # Threading for emotion detection
    import concurrent.futures
    import numpy as np
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = None
    
    # Frame skipping for performance
    frame_count = 0
    frame_skip = 5 # Run YOLO every 5 frames (smoother display)
    last_results = None
    
    cached_emotions = {} # (track_id) -> emotion_text
    last_emotion_update = {} # (track_id) -> timestamp (frame_count)
    
    # Attention Tracker
    attention_tracker = AttentionStateTracker()

    logging.info("Starting video loop. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 3. Run YOLOv8 Inference (with tracking)
        # Run only every N frames
        if frame_count % frame_skip == 0 or last_results is None:
            # Increased confidence to 0.6 to reduce false positives
            results = model.track(frame, persist=True, verbose=False, conf=0.6)
            last_results = results
        else:
            results = last_results
        
        # Check for completed emotion analysis
        # if future is not None and future.done():
        #     try:
        #         tid, emotion = future.result()
        #         if emotion:
        #             cached_emotions[tid] = emotion
        #             last_emotion_update[tid] = frame_count
        #     except Exception as e:
        #         logging.error(f"Emotion analysis error: {e}")
        #     future = None

        # Visualize
        annotated_frame = frame.copy()
        
        if results and results[0].boxes:
            boxes = results[0].boxes
            keypoints = results[0].keypoints
            
            # Person Count
            person_count = len(boxes)
            cv2.putText(annotated_frame, f"Person Count: {person_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Candidates for mood analysis
            candidates = []

            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else i
                
                # Draw Box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # --- Attention Estimation (Head Pose: Yaw + Pitch) ---
                attention_status = "Unknown"
                pose_label = "Unknown"
                duration = 0.0
                
                if keypoints is not None and len(keypoints) > i:
                    kps = keypoints[i].xy[0].cpu().numpy() # shape (17, 2)
                    
                    if len(kps) >= 5:
                        nose = kps[0]
                        l_eye = kps[1]
                        r_eye = kps[2]
                        l_ear = kps[3]
                        r_ear = kps[4]
                        
                        if nose[0] > 0 and l_eye[0] > 0 and r_eye[0] > 0:
                            yaw = calculate_yaw(nose, l_eye, r_eye, l_ear, r_ear)
                            pitch = calculate_pitch(nose, l_eye, r_eye)
                            
                            # Determine Raw Pose
                            if abs(yaw) > 0.6:
                                pose_label = "Side"
                            elif pitch > 1.0:
                                pose_label = "Down"
                            elif pitch < 0.2:
                                pose_label = "Up"
                            else:
                                pose_label = "Center"
                            
                            # Update Tracker
                            attention_status, _, duration = attention_tracker.update(track_id, pose_label)
                            
                            # Color coding
                            if "Attentive" in attention_status:
                                color = (0, 255, 0) # Green
                            elif "Writing" in attention_status:
                                color = (0, 255, 255) # Yellow
                            elif "Distracted" in attention_status:
                                color = (0, 0, 255) # Red
                            else:
                                color = (200, 200, 200) # Gray
                                
                            cv2.putText(annotated_frame, f"{attention_status} ({duration:.1f}s)", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- Mood Display ---
                # current_emotion = cached_emotions.get(track_id, "Analyzing...")
                # cv2.putText(annotated_frame, f"Mood: {current_emotion}", (x1, y1 - 30), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Collect candidate for scheduling
                # Only consider if we have a valid face crop area
                # face_x1, face_y1, face_x2, face_y2 = x1, y1, x2, y1 + (y2-y1)//3
                # if face_x2 > face_x1 and face_y2 > face_y1:
                #      candidates.append({
                #          'id': track_id,
                #          'box': (face_x1, face_y1, face_x2, face_y2),
                #          'last_update': last_emotion_update.get(track_id, 0)
                #      })

            # Scheduler: Pick the candidate with the oldest update time
            # if frame_count % frame_skip == 0 and future is None and candidates:
            #     # Sort by last_update (ascending -> oldest first)
            #     candidates.sort(key=lambda c: c['last_update'])
            #     best_candidate = candidates[0]
            #     
            #     # Extract face image
            #     fx1, fy1, fx2, fy2 = best_candidate['box']
            #     face_img = frame[fy1:fy2, fx1:fx2].copy()
            #     
            #     future = executor.submit(analyze_emotion, face_img, best_candidate['id'])

        cv2.imshow('Activity Recognition POC', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)

if __name__ == "__main__":
    main()
