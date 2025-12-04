import cv2
import logging
from ultralytics import YOLO
from deepface import DeepFace
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_yaw(nose, left_eye, right_eye, left_ear, right_ear):
    # Same logic as main.py
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

def test_image(image_path):
    if not os.path.exists(image_path):
        logging.error(f"Image not found: {image_path}")
        return

    logging.info(f"Processing {image_path}...")
    frame = cv2.imread(image_path)
    
    # 1. Load YOLO
    logging.info("Loading YOLOv8-pose...")
    model = YOLO('yolov8n-pose.pt')
    
    # 2. Inference
    results = model(frame, verbose=False)
    
    if results and results[0].boxes:
        boxes = results[0].boxes
        keypoints = results[0].keypoints
        
        person_count = len(boxes)
        logging.info(f"--- Results for {image_path} ---")
        logging.info(f"Person Count: {person_count}")

        annotated_frame = frame.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Attention
            attention_status = "Unknown"
            yaw_val = 0
            if keypoints is not None and len(keypoints) > i:
                kps = keypoints[i].xy[0].cpu().numpy()
                if len(kps) >= 5:
                    nose = kps[0]
                    l_eye = kps[1]
                    r_eye = kps[2]
                    l_ear = kps[3]
                    r_ear = kps[4]
                    
                    if nose[0] > 0 and l_eye[0] > 0 and r_eye[0] > 0:
                        yaw_val = calculate_yaw(nose, l_eye, r_eye, l_ear, r_ear)
                        if abs(yaw_val) < 0.5:
                            attention_status = "Attentive"
                        else:
                            attention_status = "Distracted"
            
            logging.info(f"Person {i+1}: Attention={attention_status} (Yaw={yaw_val:.2f})")

            # Mood (DeepFace)
            # Crop face
            face_x1, face_y1, face_x2, face_y2 = x1, y1, x2, y1 + (y2-y1)//3
            # Refine with keypoints if possible (simplified for test)
            
            face_img = frame[face_y1:face_y2, face_x1:face_x2]
            mood = "Unknown"
            try:
                if face_img.size > 0:
                    objs = DeepFace.analyze(img_path=face_img, actions=['emotion'], enforce_detection=False, silent=True)
                    if objs:
                        mood = objs[0]['dominant_emotion']
            except Exception as e:
                logging.error(f"DeepFace error for Person {i+1}: {e}")
            
            logging.info(f"Person {i+1}: Mood={mood}")
            
            # Draw
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, f"{attention_status}, {mood}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = "output_test.jpg"
        cv2.imwrite(output_path, annotated_frame)
        logging.info(f"Saved annotated image to {output_path}")
    else:
        logging.info("No persons detected.")

if __name__ == "__main__":
    test_image("bus.jpg")
