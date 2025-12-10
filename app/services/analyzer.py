import cv2
import time
import logging
import numpy as np
import torch
from ultralytics import YOLO
from deepface import DeepFace

# Configure PyTorch to allow Ultralytics model classes for safe loading
# This is required for PyTorch 2.6+ which defaults to weights_only=True
try:
    from ultralytics.nn.tasks import PoseModel, DetectionModel, SegmentationModel, ClassificationModel
    import torch.nn as nn
    
    # Import Ultralytics custom modules - handle version differences gracefully
    ultralytics_modules = []
    
    # Import conv modules individually
    try:
        from ultralytics.nn.modules.conv import Conv
        ultralytics_modules.append(Conv)
    except ImportError:
        pass
    
    try:
        from ultralytics.nn.modules.conv import DWConv, GhostConv, RepConv, Concat
        ultralytics_modules.extend([DWConv, GhostConv, RepConv, Concat])
    except ImportError:
        pass
    
    # Import block modules individually
    try:
        from ultralytics.nn.modules.block import C2f, C3, SPPF, Bottleneck, DFL
        ultralytics_modules.extend([C2f, C3, SPPF, Bottleneck, DFL])
    except ImportError:
        pass
    
    try:
        from ultralytics.nn.modules.block import C2, C3x, C3TR, C3Ghost, SPP, Proto
        ultralytics_modules.extend([C2, C3x, C3TR, C3Ghost, SPP, Proto])
    except ImportError:
        pass
    
    try:
        from ultralytics.nn.modules.block import HGStem, HGBlock, RepC3, C3k2
        ultralytics_modules.extend([HGStem, HGBlock, RepC3, C3k2])
    except ImportError:
        pass
    
    # Import head modules individually
    try:
        from ultralytics.nn.modules.head import Detect, Pose, Segment, Classify
        ultralytics_modules.extend([Detect, Pose, Segment, Classify])
    except ImportError:
        pass
    
    try:
        from ultralytics.nn.modules.head import OBB, RTDETRDecoder
        ultralytics_modules.extend([OBB, RTDETRDecoder])
    except ImportError:
        pass
    
    # Comprehensive list of PyTorch classes commonly used in YOLO models
    safe_classes = [
        # Ultralytics model classes
        PoseModel, DetectionModel, SegmentationModel, ClassificationModel,
        # PyTorch container modules
        nn.Sequential, nn.ModuleList, nn.ModuleDict,
        # Common PyTorch layers
        nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LeakyReLU, nn.SiLU,
        nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Upsample,
        nn.Linear, nn.Dropout, nn.Identity,
        # Base module class
        nn.Module,
        # Python builtins needed for unpickling
        getattr, setattr, type, dict, list, tuple, set, frozenset,
    ]
    
    # Add Ultralytics custom modules
    safe_classes.extend(ultralytics_modules)
    
    torch.serialization.add_safe_globals(safe_classes)
    logging.info(f"Added {len(safe_classes)} classes to PyTorch safe globals")
except Exception as e:
    logging.warning(f"Could not add safe globals: {e}")

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
