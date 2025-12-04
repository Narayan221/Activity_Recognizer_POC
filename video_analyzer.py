import cv2
import time
import sys
import os
from ultralytics import YOLO
import logging
import csv
from datetime import datetime

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
    """
    ny = nose[1]
    ley = left_eye[1]
    rey = right_eye[1]
    
    mid_eye_y = (ley + rey) / 2
    eye_dist = abs(left_eye[0] - right_eye[0])
    
    if eye_dist == 0:
        return 0
        
    diff_y = ny - mid_eye_y
    ratio = diff_y / eye_dist
    return ratio

class AttentionStateTracker:
    def __init__(self):
        self.states = {} # track_id -> {pose, start_time, status}
        self.threshold_time = 3.0 # seconds

    def update(self, track_id, current_pose, timestamp):
        if track_id not in self.states:
            self.states[track_id] = {
                'pose': current_pose,
                'start_time': timestamp,
                'status': 'Attentive' if current_pose == 'Center' else 'Unknown'
            }
        
        state = self.states[track_id]
        
        if state['pose'] != current_pose:
            state['pose'] = current_pose
            state['start_time'] = timestamp
            if current_pose == 'Center':
                state['status'] = 'Attentive'
            else:
                state['status'] = 'Analyzing...'
        else:
            duration = timestamp - state['start_time']
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
                if current_pose != 'Center' and state['status'] == 'Attentive':
                     state['status'] = 'Analyzing...'
        
        return state['status'], state['pose'], (timestamp - state['start_time'])

def analyze_video(video_path, output_log='analysis_log.csv', save_video=True):
    """
    Analyze a video file and generate logs.
    """
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return
    
    # Load model
    logging.info("Loading YOLOv8-pose model...")
    model = YOLO('yolov8n-pose.pt')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Could not open video file.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logging.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Setup output video writer if needed
    output_video = None
    if save_video:
        output_path = video_path.rsplit('.', 1)[0] + '_analyzed.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logging.info(f"Output video will be saved to: {output_path}")
    
    # Setup CSV log
    csv_file = open(output_log, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp', 'Frame', 'Person_ID', 'Person_Count', 'Pose', 'Attention_Status', 'Duration_Sec'])
    
    # Tracking
    attention_tracker = AttentionStateTracker()
    frame_count = 0
    frame_skip = 2  # Process every 2nd frame for speed
    
    logging.info("Starting video analysis...")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        
        # Run detection
        if frame_count % frame_skip == 0 or frame_count == 1:
            results = model.track(frame, persist=True, verbose=False, conf=0.6)
        
        annotated_frame = frame.copy()
        
        if results and results[0].boxes:
            boxes = results[0].boxes
            keypoints = results[0].keypoints
            person_count = len(boxes)
            
            # Display person count
            cv2.putText(annotated_frame, f"Person Count: {person_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else i
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                pose_label = "Unknown"
                attention_status = "Unknown"
                duration = 0.0
                
                if keypoints is not None and len(keypoints) > i:
                    kps = keypoints[i].xy[0].cpu().numpy()
                    
                    if len(kps) >= 5:
                        nose = kps[0]
                        l_eye = kps[1]
                        r_eye = kps[2]
                        l_ear = kps[3]
                        r_ear = kps[4]
                        
                        if nose[0] > 0 and l_eye[0] > 0 and r_eye[0] > 0:
                            yaw = calculate_yaw(nose, l_eye, r_eye, l_ear, r_ear)
                            pitch = calculate_pitch(nose, l_eye, r_eye)
                            
                            if abs(yaw) > 0.6:
                                pose_label = "Side"
                            elif pitch > 1.0:
                                pose_label = "Down"
                            elif pitch < 0.2:
                                pose_label = "Up"
                            else:
                                pose_label = "Center"
                            
                            attention_status, _, duration = attention_tracker.update(track_id, pose_label, timestamp)
                            
                            if "Attentive" in attention_status:
                                color = (0, 255, 0)
                            elif "Writing" in attention_status:
                                color = (0, 255, 255)
                            elif "Distracted" in attention_status:
                                color = (0, 0, 255)
                            else:
                                color = (200, 200, 200)
                                
                            cv2.putText(annotated_frame, f"{attention_status} ({duration:.1f}s)", (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Log to CSV
                csv_writer.writerow([
                    f"{timestamp:.2f}",
                    frame_count,
                    track_id,
                    person_count,
                    pose_label,
                    attention_status,
                    f"{duration:.2f}"
                ])
        
        # Save annotated frame
        if output_video:
            output_video.write(annotated_frame)
        
        # Progress
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            logging.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    if output_video:
        output_video.release()
    csv_file.close()
    
    elapsed = time.time() - start_time
    logging.info(f"Analysis complete! Processed {frame_count} frames in {elapsed:.1f}s")
    logging.info(f"Log saved to: {output_log}")
    if save_video:
        logging.info(f"Annotated video saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_analyzer.py <video_path> [output_log.csv]")
        print("Example: python video_analyzer.py my_video.mp4")
        print("Example: python video_analyzer.py my_video.mp4 custom_log.csv")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_log = sys.argv[2] if len(sys.argv) > 2 else 'analysis_log.csv'
    
    analyze_video(video_path, output_log, save_video=True)
