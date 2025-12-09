import cv2
import time
import logging
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace

class AnalysisService:
    def __init__(self):
        logging.info("Loading YOLOv8-pose model...")
        self.model = YOLO('yolov8n-pose.pt')
        # Warmup DeepFace
        try:
            logging.info("Warming up DeepFace...")
            DeepFace.analyze(img_path=np.zeros((100, 100, 3), dtype=np.uint8), actions=['emotion'], enforce_detection=False, silent=True)
        except Exception:
            pass

    def track(self, frame):
        """
        Run YOLOv8 tracking on the frame.
        """
        results = self.model.track(frame, persist=True, verbose=False, conf=0.6)
        return results

    def analyze_emotion(self, face_img):
        """
        Analyze emotion from a face crop.
        """
        try:
            objs = DeepFace.analyze(img_path=face_img, 
                                    actions=['emotion'], 
                                    enforce_detection=False, 
                                    silent=True)
            if objs:
                return objs[0]['dominant_emotion'], objs[0]['emotion'] # return dominant + dict
        except Exception:
            pass
        return None, None

    @staticmethod
    def calculate_yaw(nose, left_eye, right_eye, left_ear, right_ear):
        nx, _ = nose
        lex, _ = left_eye
        rex, _ = right_eye
        leax, _ = left_ear
        reax, _ = right_ear

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

    @staticmethod
    def calculate_pitch(nose, left_eye, right_eye):
        _, ny = nose
        _, ley = left_eye
        _, rey = right_eye
        
        mid_eye_y = (ley + rey) / 2
        eye_dist = abs(left_eye[0] - right_eye[0])
        
        if eye_dist == 0:
            return 0
            
        diff_y = ny - mid_eye_y
        ratio = diff_y / eye_dist
        return ratio
