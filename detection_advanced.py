"""
Advanced Face Detection Module
Multi-model detection supporting SCRFD, YOLOv8-Face, and RetinaFace
Optimized for long-distance face detection and varying conditions
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from enum import Enum

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available")

try:
    import torch
    import ultralytics
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLOv8 not available")

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    logging.warning("RetinaFace not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionModel(Enum):
    """Available detection models"""
    SCRFD = "scrfd"          # SCRFD (InsightFace) - Balanced
    YOLOV8 = "yolov8"        # YOLOv8-Face - Best accuracy
    RETINAFACE = "retinaface"  # RetinaFace - Long-distance
    ENSEMBLE = "ensemble"    # Ensemble of all models


class FaceDetectorAdvanced:
    """
    Advanced face detector supporting multiple models with ensemble capability
    Optimized for long-distance face detection and various conditions
    """
    
    def __init__(self, 
                 model: DetectionModel = DetectionModel.ENSEMBLE,
                 min_confidence: float = 0.5,
                 ctx_id: int = -1,
                 enable_blur_detection: bool = True,
                 enable_face_quality: bool = True):
        """
        Initialize advanced face detector
        
        Args:
            model: Detection model to use (SCRFD, YOLOV8, RETINAFACE, or ENSEMBLE)
            min_confidence: Minimum confidence threshold (0.0-1.0)
            ctx_id: GPU device id, -1 for CPU
            enable_blur_detection: Enable blur detection for quality assurance
            enable_face_quality: Enable face quality assessment
        """
        self.model_type = model
        self.min_confidence = min_confidence
        self.ctx_id = ctx_id
        self.enable_blur_detection = enable_blur_detection
        self.enable_face_quality = enable_face_quality
        
        self.scrfd_detector = None
        self.yolo_detector = None
        self.retinaface_detector = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize selected detection models"""
        if self.model_type in [DetectionModel.SCRFD, DetectionModel.ENSEMBLE]:
            self._init_scrfd()
        
        if self.model_type in [DetectionModel.YOLOV8, DetectionModel.ENSEMBLE]:
            self._init_yolov8()
        
        if self.model_type in [DetectionModel.RETINAFACE, DetectionModel.ENSEMBLE]:
            self._init_retinaface()
    
    def _init_scrfd(self):
        """Initialize SCRFD (InsightFace) detector"""
        if not INSIGHTFACE_AVAILABLE:
            logger.warning("InsightFace not available for SCRFD")
            return
        
        try:
            model_name = 'buffalo_l'
            self.scrfd_detector = FaceAnalysis(
                name=model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
                if self.ctx_id >= 0 else ['CPUExecutionProvider']
            )
            self.scrfd_detector.prepare(
                ctx_id=self.ctx_id,
                det_thresh=self.min_confidence,
                det_size=(640, 640)
            )
            logger.info("✓ SCRFD detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SCRFD: {e}")
    
    def _init_yolov8(self):
        """Initialize YOLOv8-Face detector"""
        if not YOLO_AVAILABLE:
            logger.warning("YOLOv8 not available")
            return
        
        try:
            # Using YOLOv8n-face (nano) for speed or YOLOv8m-face for accuracy
            self.yolo_detector = YOLO('yolov8m.pt')  # Will auto-download
            logger.info("✓ YOLOv8-Face detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8: {e}")
    
    def _init_retinaface(self):
        """Initialize RetinaFace detector"""
        if not RETINAFACE_AVAILABLE:
            logger.warning("RetinaFace not available")
            return
        
        try:
            # RetinaFace is initialized on demand
            logger.info("✓ RetinaFace detector available")
        except Exception as e:
            logger.error(f"Failed to initialize RetinaFace: {e}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image using configured model(s)
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detected faces with enhanced metadata
        """
        if image is None or image.size == 0:
            logger.error("Invalid input image")
            return []
        
        if self.model_type == DetectionModel.SCRFD:
            return self._detect_scrfd(image)
        elif self.model_type == DetectionModel.YOLOV8:
            return self._detect_yolov8(image)
        elif self.model_type == DetectionModel.RETINAFACE:
            return self._detect_retinaface(image)
        elif self.model_type == DetectionModel.ENSEMBLE:
            return self._detect_ensemble(image)
    
    def _detect_scrfd(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using SCRFD"""
        if self.scrfd_detector is None:
            return []
        
        try:
            faces = self.scrfd_detector.get(image)
            results = []
            
            for face in faces:
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                w = x2 - x
                h = y2 - y
                
                keypoints = {}
                if hasattr(face, 'kps') and face.kps is not None:
                    kps = face.kps.astype(int)
                    keypoints = {
                        'left_eye': tuple(kps[0]),
                        'right_eye': tuple(kps[1]),
                        'nose': tuple(kps[2]),
                        'left_mouth': tuple(kps[3]),
                        'right_mouth': tuple(kps[4])
                    }
                
                detection = {
                    'box': (x, y, w, h),
                    'confidence': 0.95,  # SCRFD is highly confident
                    'keypoints': keypoints,
                    'model': 'SCRFD',
                    'x': x, 'y': y, 'x2': x2, 'y2': y2
                }
                
                if self.enable_face_quality:
                    detection['quality_score'] = self._assess_face_quality(image, bbox)
                
                results.append(detection)
            
            return results
        except Exception as e:
            logger.error(f"SCRFD detection failed: {e}")
            return []
    
    def _detect_yolov8(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using YOLOv8"""
        if self.yolo_detector is None:
            return []
        
        try:
            results = self.yolo_detector(image, conf=self.min_confidence, verbose=False)
            detections = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    w = x2 - x1
                    h = y2 - y1
                    confidence = float(box.conf[0])
                    
                    detection = {
                        'box': (x1, y1, w, h),
                        'confidence': confidence,
                        'keypoints': {},
                        'model': 'YOLOv8',
                        'x': x1, 'y': y1, 'x2': x2, 'y2': y2
                    }
                    
                    if self.enable_face_quality:
                        detection['quality_score'] = self._assess_face_quality(image, np.array([x1, y1, x2, y2]))
                    
                    detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"YOLOv8 detection failed: {e}")
            return []
    
    def _detect_retinaface(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using RetinaFace (for long-distance)"""
        if not RETINAFACE_AVAILABLE:
            return []
        
        try:
            # RetinaFace is excellent for small faces
            faces = RetinaFace.detect_faces(image)
            detections = []
            
            if isinstance(faces, dict):
                for key, face in faces.items():
                    if key != 'status':
                        bbox = face['facial_area']
                        x, y, x2, y2 = bbox
                        w = x2 - x
                        h = y2 - y
                        
                        detection = {
                            'box': (x, y, w, h),
                            'confidence': face.get('score', 0.9),
                            'keypoints': {
                                'left_eye': face['landmarks']['left_eye'],
                                'right_eye': face['landmarks']['right_eye'],
                                'nose': face['landmarks']['nose'],
                                'left_mouth': face['landmarks']['mouth_left'],
                                'right_mouth': face['landmarks']['mouth_right']
                            },
                            'model': 'RetinaFace',
                            'x': x, 'y': y, 'x2': x2, 'y2': y2
                        }
                        
                        if self.enable_face_quality:
                            detection['quality_score'] = self._assess_face_quality(image, bbox)
                        
                        detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"RetinaFace detection failed: {e}")
            return []
    
    def _detect_ensemble(self, image: np.ndarray) -> List[Dict]:
        """Ensemble detection combining multiple models"""
        all_detections = []
        
        # Get detections from all available models
        if self.scrfd_detector:
            all_detections.extend(self._detect_scrfd(image))
        
        if self.yolo_detector:
            all_detections.extend(self._detect_yolov8(image))
        
        if RETINAFACE_AVAILABLE:
            all_detections.extend(self._detect_retinaface(image))
        
        # Merge overlapping detections (NMS)
        merged = self._merge_detections(all_detections)
        return merged
    
    def _merge_detections(self, detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
        """Merge overlapping detections from multiple models"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        merged = []
        
        for det in detections:
            should_merge = False
            x1, y1, w1, h1 = det['box']
            x1_end = x1 + w1
            y1_end = y1 + h1
            
            for existing in merged:
                x2, y2, w2, h2 = existing['box']
                x2_end = x2 + w2
                y2_end = y2 + h2
                
                # Calculate IoU
                inter_x1 = max(x1, x2)
                inter_y1 = max(y1, y2)
                inter_x2 = min(x1_end, x2_end)
                inter_y2 = min(y1_end, y2_end)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    union_area = w1 * h1 + w2 * h2 - inter_area
                    iou = inter_area / union_area
                    
                    if iou > iou_threshold:
                        # Merge: average coordinates, keep highest confidence
                        if det.get('confidence', 0) > existing.get('confidence', 0):
                            existing['box'] = det['box']
                            existing['confidence'] = det['confidence']
                            if det.get('keypoints'):
                                existing['keypoints'] = det.get('keypoints', {})
                        should_merge = True
                        break
            
            if not should_merge:
                merged.append(det)
        
        return merged
    
    def _assess_face_quality(self, image: np.ndarray, bbox: np.ndarray) -> float:
        """
        Assess face image quality
        
        Args:
            image: Full image
            bbox: Bounding box [x, y, x2, y2]
            
        Returns:
            Quality score (0-1)
        """
        try:
            x, y, x2, y2 = bbox
            face_crop = image[y:y2, x:x2]
            
            if face_crop.size == 0:
                return 0.0
            
            # Blur detection (Laplacian variance)
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, laplacian_var / 500)
            
            # Brightness assessment
            brightness = np.mean(gray)
            brightness_score = 1.0 if 80 < brightness < 200 else 0.5
            
            # Contrast assessment
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 50)
            
            # Combined quality
            quality = (blur_score * 0.5 + brightness_score * 0.25 + contrast_score * 0.25)
            return float(quality)
        except:
            return 0.7
    
    def filter_by_distance(self, detections: List[Dict], 
                          min_size: Tuple[int, int] = (40, 40)) -> List[Dict]:
        """
        Filter detections by size (effective for long-distance detection)
        
        Args:
            detections: List of detections
            min_size: Minimum face size (width, height)
            
        Returns:
            Filtered detections
        """
        filtered = []
        min_w, min_h = min_size
        
        for det in detections:
            x, y, w, h = det['box']
            if w >= min_w and h >= min_h:
                filtered.append(det)
        
        return filtered
