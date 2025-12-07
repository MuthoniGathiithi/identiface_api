"""
Advanced Face Normalization & Preprocessing Module
Multi-technique face alignment, normalization, and quality enhancement
Optimized for consistent feature extraction across varying conditions
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceNormalizerAdvanced:
    """
    Advanced face normalization with multiple alignment and enhancement techniques
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 enable_affine_alignment: bool = True,
                 enable_3d_alignment: bool = True,
                 normalize_color: bool = True):
        """
        Initialize advanced face normalizer
        
        Args:
            target_size: Target output size (width, height)
            enable_affine_alignment: Use 2D affine transformation
            enable_3d_alignment: Use 3D perspective alignment
            normalize_color: Normalize color channels
        """
        self.target_size = target_size
        self.enable_affine_alignment = enable_affine_alignment
        self.enable_3d_alignment = enable_3d_alignment
        self.normalize_color = normalize_color
        
        logger.info(f"Initialized FaceNormalizerAdvanced: target_size={target_size}")
    
    def normalize_complete(self, image: np.ndarray,
                          keypoints: Optional[Dict] = None,
                          bbox: Optional[Tuple] = None) -> np.ndarray:
        """
        Complete normalization pipeline
        
        Args:
            image: Input face image
            keypoints: Facial landmarks
            bbox: Bounding box of face
            
        Returns:
            Normalized face image
        """
        if image is None or image.size == 0:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        
        # Step 1: Align face
        if keypoints and len(keypoints) >= 2:
            aligned = self.align_face_advanced(image, keypoints)
        else:
            aligned = image
        
        # Step 2: Resize to target size
        normalized = cv2.resize(aligned, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        # Step 3: Enhance contrast
        normalized = self.enhance_contrast_adaptive(normalized)
        
        # Step 4: Normalize color
        if self.normalize_color:
            normalized = self.normalize_color_space(normalized)
        
        return normalized
    
    def align_face_advanced(self, image: np.ndarray, keypoints: Dict) -> np.ndarray:
        """
        Advanced face alignment using multiple keypoints
        
        Args:
            image: Input face image
            keypoints: Dictionary with facial landmarks
            
        Returns:
            Aligned face image
        """
        try:
            # Extract eye positions
            if not self._has_valid_keypoints(keypoints):
                return image
            
            left_eye = np.array(keypoints.get('left_eye', (0, 0)), dtype=np.float32)
            right_eye = np.array(keypoints.get('right_eye', (0, 0)), dtype=np.float32)
            
            if np.all(left_eye == 0) or np.all(right_eye == 0):
                return image
            
            # Calculate alignment based on eyes
            aligned = self._align_by_eyes(image, left_eye, right_eye)
            
            return aligned
        except Exception as e:
            logger.warning(f"Advanced alignment failed: {e}, returning original")
            return image
    
    def _has_valid_keypoints(self, keypoints: Dict) -> bool:
        """Check if keypoints dictionary has valid entries"""
        required_keys = ['left_eye', 'right_eye']
        return all(k in keypoints and keypoints[k] for k in required_keys)
    
    def _align_by_eyes(self, image: np.ndarray,
                      left_eye: np.ndarray,
                      right_eye: np.ndarray) -> np.ndarray:
        """
        Align face based on eye positions
        
        Args:
            image: Input image
            left_eye: Left eye coordinates
            right_eye: Right eye coordinates
            
        Returns:
            Aligned image
        """
        try:
            # Calculate angle between eyes
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))
            
            # Calculate center between eyes
            eye_center_x = (left_eye[0] + right_eye[0]) // 2
            eye_center_y = (left_eye[1] + right_eye[1]) // 2
            
            # Position eyes to desired location
            desired_left_eye = (0.35, 0.35)
            desired_right_eye = (0.65, 0.35)
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D((eye_center_x, eye_center_y), angle, 1.0)
            
            # Apply rotation
            h, w = image.shape[:2]
            rotated = cv2.warpAffine(image, M, (w, h), 
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            
            logger.debug(f"Face aligned by eyes: angle={angle:.2f}°")
            return rotated
        except Exception as e:
            logger.warning(f"Eye-based alignment failed: {e}")
            return image
    
    def enhance_contrast_adaptive(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptively enhance contrast using CLAHE
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image
    
    def normalize_color_space(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize color space using multiple techniques
        
        Args:
            image: Input image
            
        Returns:
            Color-normalized image
        """
        try:
            # Technique 1: Histogram equalization on YCrCb
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            from_ycrcb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            
            # Technique 2: Gamma correction for brightness normalization
            normalized = self._gamma_correction(from_ycrcb, gamma=1.2)
            
            return normalized
        except Exception as e:
            logger.warning(f"Color normalization failed: {e}")
            return image
    
    def _gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction for brightness normalization
        
        Args:
            image: Input image
            gamma: Gamma value (> 1.0 brightens, < 1.0 darkens)
            
        Returns:
            Gamma-corrected image
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def enhance_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance illumination consistency
        
        Args:
            image: Input image
            
        Returns:
            Illumination-enhanced image
        """
        try:
            # Convert to YCrCb
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y_channel = ycrcb[:, :, 0].astype(np.float32)
            
            # Calculate average illumination
            mean_y = np.mean(y_channel)
            
            # Normalize Y channel
            y_channel = (y_channel - np.mean(y_channel)) * (np.std(y_channel) ** -1)
            y_channel = y_channel * 100 + 128  # Scale back to 0-255
            y_channel = np.clip(y_channel, 0, 255).astype(np.uint8)
            
            # Reconstruct image
            ycrcb[:, :, 0] = y_channel
            enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            
            return enhanced
        except Exception as e:
            logger.warning(f"Illumination enhancement failed: {e}")
            return image
    
    def reduce_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply deblurring filter
        
        Args:
            image: Input image
            
        Returns:
            Deblurred image
        """
        try:
            # Apply unsharp mask (sharpening)
            gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
            sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            
            # Clip values
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
            
            return sharpened
        except Exception as e:
            logger.warning(f"Deblurring failed: {e}")
            return image
    
    def assess_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Assess normalized face image quality
        
        Args:
            image: Face image
            
        Returns:
            Quality metrics dictionary
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(1.0, laplacian_var / 500)
            
            # Brightness
            brightness = np.mean(gray)
            brightness_score = 1.0 if 50 < brightness < 200 else 0.5
            
            # Contrast
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 70)
            
            # Color balance
            mean_b = np.mean(image[:, :, 0])
            mean_g = np.mean(image[:, :, 1])
            mean_r = np.mean(image[:, :, 2])
            color_balance = 1.0 - np.std([mean_b, mean_g, mean_r]) / 100
            color_balance = np.clip(color_balance, 0, 1)
            
            # Overall quality
            overall_quality = (sharpness * 0.3 + brightness_score * 0.2 +
                              contrast_score * 0.3 + color_balance * 0.2)
            
            return {
                'overall': float(overall_quality),
                'sharpness': float(sharpness),
                'brightness': float(brightness_score),
                'contrast': float(contrast_score),
                'color_balance': float(color_balance),
                'brightness_value': float(brightness),
                'contrast_value': float(contrast)
            }
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return {'overall': 0.5}
    
    def preprocess_batch(self, images: List[np.ndarray],
                        keypoints_list: Optional[List[Dict]] = None) -> List[np.ndarray]:
        """
        Preprocess batch of faces
        
        Args:
            images: List of face images
            keypoints_list: Optional list of keypoint dictionaries
            
        Returns:
            List of normalized faces
        """
        normalized = []
        
        for i, img in enumerate(images):
            kpts = keypoints_list[i] if keypoints_list else None
            norm_img = self.normalize_complete(img, kpts)
            normalized.append(norm_img)
        
        return normalized
