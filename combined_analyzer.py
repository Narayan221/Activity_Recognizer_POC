import cv2
import time
import sys
import os
from ultralytics import YOLO
from groq import Groq
from dotenv import load_dotenv
import logging
from datetime import timedelta
import threading

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_yaw(nose, left_eye, right_eye, left_ear, right_ear):
    """Estimate head yaw (left/right)."""
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
    """Estimate head pitch (up/down)."""
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
        self.states = {}
        self.threshold_time = 3.0

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
                    state['status'] = 'Distracted'
                elif current_pose == 'Up':
                    state['status'] = 'Distracted'
                elif current_pose == 'Down':
                    state['status'] = 'Distracted'
                elif current_pose == 'Center':
                    state['status'] = 'Attentive'
            else:
                if current_pose != 'Center' and state['status'] == 'Attentive':
                     state['status'] = 'Analyzing...'
        
        return state['status'], state['pose'], (timestamp - state['start_time'])

def transcribe_audio(video_path, api_key):
    """Transcribe audio in background thread"""
    try:
        logging.info("Starting audio transcription in background...")
        client = Groq(api_key=api_key)
        
        with open(video_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(video_path), file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
                temperature=0.0
            )
        
        logging.info("Audio transcription complete!")
        return transcription
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return None

def analyze_video_combined(video_path, activity_log='activity_log.txt', transcript_log='transcript_log.txt', timestamped_transcript_log='timestamped_transcript_log.txt'):
    """
    Analyze video with both activity recognition and audio transcription.
    Saves simplified logs to separate files.
    """
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return
    
    # Get API key
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        logging.error("GROQ_API_KEY not found in .env file")
        return
    
    logging.info("=" * 80)
    logging.info("COMBINED ANALYSIS: Activity Recognition + Audio Transcription")
    logging.info("=" * 80)

    # Start transcription in background thread
    transcription_result = None
    transcription_thread = threading.Thread(target=lambda: globals().update({'transcription_result': transcribe_audio(video_path, api_key)}))
    transcription_thread.start()
    
    # Load YOLO model
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
    
    logging.info(f"Video: {fps} FPS, {total_frames} frames")
    logging.info(f"Activity log: {activity_log}")
    logging.info(f"Transcript log: {transcript_log}")
    
    # Setup activity log file
    activity_file = open(activity_log, 'w', encoding='utf-8')
    activity_file.write("ACTIVITY LOG\n")
    activity_file.write("=" * 80 + "\n")
    activity_file.write("Timestamp | Status\n")
    activity_file.write("-" * 80 + "\n")
    
    # Tracking
    attention_tracker = AttentionStateTracker()
    frame_count = 0
    frame_skip = 5
    last_logged_second = -1  # Track last second we logged
    
    logging.info("Starting video analysis...")
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = frame_count / fps
        current_second = int(timestamp)  # Get current second
        
        # Run detection
        if frame_count % frame_skip == 0 or frame_count == 1:
            results = model.track(frame, persist=True, verbose=False, conf=0.6)
        
        if results and results[0].boxes:
            boxes = results[0].boxes
            keypoints = results[0].keypoints
            
            for i, box in enumerate(boxes):
                track_id = int(box.id[0]) if box.id is not None else i
                
                pose_label = "Unknown"
                attention_status = "Unknown"
                
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
                
                # Write to activity log once per second (only Attentive/Distracted)
                if current_second != last_logged_second:
                    if "Attentive" in attention_status or "Distracted" in attention_status:
                        time_str = str(timedelta(seconds=current_second))
                        simple_status = "Attentive" if "Attentive" in attention_status else "Distracted"
                        activity_file.write(f"{time_str} | {simple_status}\n")
                        last_logged_second = current_second
        
        # Progress
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            logging.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    activity_file.close()
    
    # Wait for transcription to complete
    logging.info("Waiting for audio transcription to complete...")
    transcription_thread.join()
    
    # Save transcript
    if 'transcription_result' in globals() and globals()['transcription_result']:
        transcription = globals()['transcription_result']
        
        # Save full transcript (no timestamps)
        with open(transcript_log, 'w', encoding='utf-8') as f:
            f.write("TRANSCRIPT\n")
            f.write("=" * 80 + "\n\n")
            f.write(transcription.text + "\n")
        logging.info(f"Transcript saved to: {transcript_log}")
        
        # Save timestamped transcript
        with open(timestamped_transcript_log, 'w', encoding='utf-8') as f:
            f.write("TIMESTAMPED TRANSCRIPT\n")
            f.write("=" * 80 + "\n")
            f.write("Timestamp | Text\n")
            f.write("-" * 80 + "\n")
            
            if hasattr(transcription, 'segments') and transcription.segments:
                for segment in transcription.segments:
                    segment_start_time = str(timedelta(seconds=int(segment['start'])))
                    text = segment['text'].strip()
                    f.write(f"{segment_start_time} | {text}\n")
                
        logging.info(f"Timestamped transcript saved to: {timestamped_transcript_log}")
    
    elapsed = time.time() - start_time
    logging.info("\n" + "=" * 80)
    logging.info("ANALYSIS COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Processed {frame_count} frames in {elapsed:.1f}s")
    logging.info(f"Activity log: {activity_log}")
    logging.info(f"Transcript log: {transcript_log}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("=" * 80)
        print("COMBINED VIDEO ANALYZER")
        print("=" * 80)
        print("\nUsage:")
        print("  python combined_analyzer.py <video_file>")
        print("\nExample:")
        print("  python combined_analyzer.py video.mp4")
        print("\nOutputs:")
        print("  - video_activity_log.txt: Timestamp | Attentive/Distracted")
        print("  - video_transcript_log.txt: Full transcribed text")
        print("  - video_timestamped_transcript_log.txt: Timestamp | Transcribed text")
        print("\nRequires:")
        print("  - .env file with GROQ_API_KEY")
        print("=" * 80)
        sys.exit(1)
    
    video_path = sys.argv[1]
    base_name = os.path.splitext(video_path)[0]
    
    analyze_video_combined(
        video_path,
        activity_log=f"{base_name}_activity_log.txt",
        transcript_log=f"{base_name}_transcript_log.txt",
        timestamped_transcript_log=f"{base_name}_timestamped_transcript_log.txt"
    )
