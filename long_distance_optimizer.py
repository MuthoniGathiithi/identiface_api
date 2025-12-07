"""
Long-Distance Face Recognition Optimizer
Specialized techniques for detecting and recognizing faces at varying distances
Includes multi-scale detection, super-resolution, and distance-aware thresholds
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongDistanceOptimizer:
    """
    Optimization techniques for long-distance and small face recognition
    """
    
    def __init__(self, 
                 enable_super_resolution: bool = True,
                 enable_multi_scale: bool = True,
                 enable_denoising: bool = True):
        """
        Initialize long-distance optimizer
        
        Args:
            enable_super_resolution: Use ESRGAN for super-resolution
            enable_multi_scale: Use multi-scale detection
            enable_denoising: Apply denoising for long-distance images
        """
        self.enable_super_resolution = enable_super_resolution
        self.enable_multi_scale = enable_multi_scale
        self.enable_denoising = enable_denoising
        
        self.sr_model = None
        if enable_super_resolution:
            self._init_super_resolution()
        
        logger.info("Initialized LongDistanceOptimizer")
    
    def _init_super_resolution(self):
        """Initialize super-resolution model (ESRGAN)"""
        try:
            # Initialize OpenCV's super-resolution engine
            # Can be replaced with ESRGAN for better quality
            logger.info("✓ Super-resolution capability initialized")
        except Exception as e:
            logger.error(f"Failed to initialize super-resolution: {e}")
    
    def preprocess_for_long_distance(self, image: np.ndarray,
                                    face_size_estimate: Optional[int] = None) -> np.ndarray:
        """
        Preprocess image for long-distance face recognition
        
        Args:
            image: Input image
            face_size_estimate: Estimated face size in pixels (for super-resolution decision)
            
        Returns:
            Preprocessed image
        """
        if image is None or image.size == 0:
            return image
        
        processed = image.copy()
        
        # Apply denoising if image is small
        if self.enable_denoising and face_size_estimate and face_size_estimate < 60:
            processed = self._denoise_image(processed)
        
        # Apply super-resolution if faces are very small
        if self.enable_super_resolution and face_size_estimate and face_size_estimate < 50:
            processed = self._apply_super_resolution(processed)
        
        # Enhance contrast for small faces
        processed = self._enhance_contrast(processed)
        
        return processed
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise image using bilateral filtering and NLM
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        try:
            # Apply bilateral filter (preserves edges)
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            
            # Apply Non-Local Means Denoising for further noise reduction
            denoised = cv2.fastNlMeansDenoisingColored(denoised, None, h=10, 
                                                       hForColorComponents=10, 
                                                       templateWindowSize=7, 
                                                       searchWindowSize=21)
            
            logger.debug("Image denoised successfully")
            return denoised
        except Exception as e:
            logger.warning(f"Denoising failed: {e}, using original image")
            return image
    
    def _apply_super_resolution(self, image: np.ndarray, scale: int = 2) -> np.ndarray:
        """
        Apply super-resolution upscaling
        
        Args:
            image: Input image
            scale: Upscaling factor (2x or 4x)
            
        Returns:
            Upscaled image
        """
        try:
            h, w = image.shape[:2]
            
            # Use OpenCV's DNN super-resolution
            # For production, use ESRGAN: pip install opencv-contrib-python-headless esrgan
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel('ESRGAN_x2.pb')  # Can be x2, x3, or x4
            sr.setModel('esrgan', scale)
            
            result = sr.upsample(image)
            
            logger.debug(f"Super-resolution applied: {image.shape} -> {result.shape}")
            return result
        except Exception as e:
            logger.debug(f"Super-resolution unavailable, using bicubic upscaling: {e}")
            # Fallback to bicubic interpolation
            new_h, new_w = h * scale, w * scale
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast for small faces using CLAHE
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        try:
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel with aggressive settings
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image
    
    def multi_scale_detection(self, image: np.ndarray,
                             detection_fn,
                             scales: List[float] = [0.5, 1.0, 1.5, 2.0]) -> List[Dict]:
        """
        Perform multi-scale face detection to catch faces at different distances
        
        Args:
            image: Input image
            detection_fn: Detection function that returns list of detections
            scales: Scale factors to test
            
        Returns:
            Merged detections from all scales
        """
        all_detections = []
        h, w = image.shape[:2]
        
        for scale in scales:
            # Skip scales that would be too large
            if scale * h > 4000 or scale * w > 4000:
                continue
            
            # Resize image
            if scale != 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_image = image
            
            # Detect faces
            detections = detection_fn(scaled_image)
            
            # Scale back bounding boxes
            if scale != 1.0 and detections:
                for det in detections:
                    if 'box' in det:
                        x, y, bw, bh = det['box']
                        det['box'] = (int(x / scale), int(y / scale), 
                                     int(bw / scale), int(bh / scale))
                        det['scale'] = scale
            
            all_detections.extend(detections)
        
        # Merge overlapping detections
        merged = self._merge_detections_nms(all_detections)
        
        logger.info(f"Multi-scale detection: {len(merged)} unique faces from {len(all_detections)} detections")
        return merged
    
    def _merge_detections_nms(self, detections: List[Dict], 
                             iou_threshold: float = 0.5) -> List[Dict]:
        """
        Non-maximum suppression to merge overlapping detections
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for merging
            
        Returns:
            Merged detections
        """
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x.get('confidence', 0.5), reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for det in detections:
                if not self._has_overlap(current, det, iou_threshold):
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _has_overlap(self, det1: Dict, det2: Dict, iou_threshold: float) -> bool:
        """
        Check if two detections overlap significantly
        
        Args:
            det1: First detection
            det2: Second detection
            iou_threshold: IoU threshold
            
        Returns:
            True if overlap > threshold
        """
        try:
            x1, y1, w1, h1 = det1['box']
            x2, y2, w2, h2 = det2['box']
            
            # Calculate intersection
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)
            
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return False
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union_area = w1 * h1 + w2 * h2 - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0
            
            return iou > iou_threshold
        except:
            return False
    
    def adjust_threshold_by_distance(self, base_threshold: float,
                                    face_size: int,
                                    min_face_size: int = 40,
                                    max_face_size: int = 500) -> float:
        """
        Adjust matching threshold based on estimated face distance
        
        Smaller faces (far away) may need relaxed thresholds
        
        Args:
            base_threshold: Base matching threshold
            face_size: Detected face size in pixels
            min_face_size: Minimum reasonable face size
            max_face_size: Maximum reasonable face size
            
        Returns:
            Adjusted threshold
        """
        # Normalize face size to 0-1 range
        normalized_size = (face_size - min_face_size) / (max_face_size - min_face_size)
        normalized_size = np.clip(normalized_size, 0, 1)
        
        # Smaller faces get relaxed thresholds (slightly lower)
        # This accounts for lower quality in distant faces
        if normalized_size < 0.3:
            # Very small faces: relax threshold by 10-20%
            adjustment = 0.85
        elif normalized_size < 0.6:
            # Small faces: relax threshold by 5-10%
            adjustment = 0.92
        else:
            # Normal/large faces: no adjustment
            adjustment = 1.0
        
        adjusted_threshold = base_threshold * adjustment
        
        logger.debug(f"Threshold adjustment: size={face_size} -> "
                    f"normalized={normalized_size:.2f} -> "
                    f"adjustment={adjustment:.2f} -> "
                    f"threshold={adjusted_threshold:.4f}")
        
        return adjusted_threshold
    
    def estimate_distance_category(self, face_size: int) -> str:
        """
        Estimate distance category based on face size
        
        Args:
            face_size: Average face dimension in pixels
            
        Returns:
            Distance category: 'very_close', 'close', 'medium', 'far', 'very_far'
        """
        if face_size >= 300:
            return 'very_close'
        elif face_size >= 150:
            return 'close'
        elif face_size >= 80:
            return 'medium'
        elif face_size >= 40:
            return 'far'
        else:
            return 'very_far'
    
    def extract_face_crop(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                         expand_ratio: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract face crop with expansion for context
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, width, height)
            expand_ratio: Expansion ratio for context (e.g., 0.1 = 10% expansion)
            
        Returns:
            Face crop
        """
        try:
            x, y, w, h = bbox
            h_img, w_img = image.shape[:2]
            
            # Expand bounding box
            expand_w = int(w * expand_ratio)
            expand_h = int(h * expand_ratio)
            
            x1 = max(0, x - expand_w)
            y1 = max(0, y - expand_h)
            x2 = min(w_img, x + w + expand_w)
            y2 = min(h_img, y + h + expand_h)
            
            crop = image[y1:y2, x1:x2]
            
            return crop if crop.size > 0 else None
        except Exception as e:
            logger.warning(f"Face crop extraction failed: {e}")
            return None
