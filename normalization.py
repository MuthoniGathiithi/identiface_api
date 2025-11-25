"""
Face Normalization Module
Handles face alignment, preprocessing, and standardization
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceNormalizer:
    """
    Face normalization and preprocessing for consistent feature extraction
    """
    
    def __init__(self, target_size: Tuple[int, int] = (160, 160)):
        """
        Initialize face normalizer
        
        Args:
            target_size: Target size for normalized faces (width, height)
        """
        self.target_size = target_size
        logger.info(f"Initialized FaceNormalizer with target size {target_size}")
    
    def align_face(self, image: np.ndarray, keypoints: Dict) -> Optional[np.ndarray]:
        """
        Align face based on eye positions
        
        Args:
            image: Input face image
            keypoints: Dictionary containing facial keypoints
            
        Returns:
            Aligned face image or None if alignment fails
        """
        if not keypoints or len(keypoints) < 2:
            logger.warning("Insufficient keypoints for alignment")
            return self.resize_face(image)
        
        # Try to find eye positions
        left_eye = None
        right_eye = None
        
        # MTCNN keypoint format
        if 'left_eye' in keypoints:
            left_eye = keypoints['left_eye']
            right_eye = keypoints['right_eye']
        # MediaPipe or generic format
        elif 'point_0' in keypoints and 'point_1' in keypoints:
            left_eye = keypoints['point_0']
            right_eye = keypoints['point_1']
        
        if left_eye is None or right_eye is None:
            logger.warning("Eye positions not found, skipping alignment")
            return self.resize_face(image)
        
        # Ensure keypoints are numeric
        try:
            left_x, left_y = float(left_eye[0]), float(left_eye[1])
            right_x, right_y = float(right_eye[0]), float(right_eye[1])
        except (TypeError, IndexError, ValueError) as e:
            logger.warning(f"Invalid keypoint format: {e}")
            return self.resize_face(image)
        
        # Calculate angle between eyes
        dx = right_x - left_x
        dy = right_y - left_y
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate center point between eyes
        center_x = int((left_x + right_x) / 2)
        center_y = int((left_y + right_y) / 2)
        eyes_center = (center_x, center_y)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Apply rotation
        h, w = image.shape[:2]
        aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        logger.debug(f"Face aligned with angle: {angle:.2f} degrees")
        return aligned
    
    def resize_face(self, image: np.ndarray) -> np.ndarray:
        """
        Resize face to target size
        
        Args:
            image: Input face image
            
        Returns:
            Resized face image
        """
        if image is None or image.size == 0:
            logger.error("Invalid input image for resizing")
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        return resized
    
    def normalize_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image histogram for consistent lighting
        
        Args:
            image: Input image
            
        Returns:
            Histogram-normalized image
        """
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Apply histogram equalization to Y channel
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        
        return normalized
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, 
                    tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            CLAHE-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to image
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return denoised
    
    def standardize_pixels(self, image: np.ndarray, method: str = 'mean_std') -> np.ndarray:
        """
        Standardize pixel values
        
        Args:
            image: Input image
            method: 'mean_std' or 'minmax'
            
        Returns:
            Standardized image as float array
        """
        image_float = image.astype(np.float32)
        
        if method == 'mean_std':
            # Zero mean, unit variance
            mean = np.mean(image_float, axis=(0, 1), keepdims=True)
            std = np.std(image_float, axis=(0, 1), keepdims=True)
            standardized = (image_float - mean) / (std + 1e-7)
        elif method == 'minmax':
            # Scale to [0, 1]
            standardized = image_float / 255.0
        else:
            # Scale to [-1, 1]
            standardized = (image_float / 127.5) - 1.0
        
        return standardized
    
    def preprocess_for_model(self, image: np.ndarray, keypoints: Optional[Dict] = None,
                            enhance: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for model input
        
        Args:
            image: Input face image
            keypoints: Optional facial keypoints for alignment
            enhance: Whether to apply enhancement
            
        Returns:
            Preprocessed face image ready for feature extraction
        """
        # Step 1: Align face if keypoints available
        if keypoints:
            aligned = self.align_face(image, keypoints)
        else:
            aligned = image
        
        # Step 2: Resize to target size
        resized = self.resize_face(aligned)
        
        # Step 3: Apply enhancement if requested
        if enhance:
            enhanced = self.apply_clahe(resized)
        else:
            enhanced = resized
        
        # Step 4: Standardize pixels
        standardized = self.standardize_pixels(enhanced, method='minmax')
        
        logger.debug("Face preprocessing completed")
        return standardized
    
    def augment_face(self, image: np.ndarray) -> list:
        """
        Generate augmented versions of face for robust enrollment
        
        Args:
            image: Input face image
            
        Returns:
            List of augmented face images
        """
        augmented = [image]
        
        # Slight rotation variations
        for angle in [-5, 5]:
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            augmented.append(rotated)
        
        # Brightness variations
        for beta in [-15, 15]:
            adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
            augmented.append(adjusted)
        
        # Horizontal flip
        flipped = cv2.flip(image, 1)
        augmented.append(flipped)
        
        logger.debug(f"Generated {len(augmented)} augmented versions")
        return augmented
    
    def check_face_quality(self, image: np.ndarray) -> Dict:
        """
        Check if face meets quality requirements
        
        Args:
            image: Input face image
            
        Returns:
            Quality assessment dictionary
        """
        if image is None or image.size == 0:
            return {'passed': False, 'reason': 'Invalid image'}
        
        h, w = image.shape[:2]
        
        # Check minimum size
        if h < 80 or w < 80:
            return {'passed': False, 'reason': f'Face too small: {w}x{h}'}
        
        # Check brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 30 or brightness > 230:
            return {'passed': False, 'reason': f'Poor lighting: {brightness:.1f}'}
        
        # Check sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:
            return {'passed': False, 'reason': f'Too blurry: {laplacian_var:.1f}'}
        
        return {
            'passed': True,
            'brightness': brightness,
            'sharpness': laplacian_var,
            'size': (w, h)
        }


# TEST CODE - Add your image path here
if __name__ == "__main__":
    import os
    from detection import FaceDetector
    
    def test_normalization():
        """Test face normalization with static images"""
        print("=== FACE NORMALIZATION TEST ===")
        
        # Initialize components
        detector = FaceDetector(model_name='buffalo_l', min_confidence=0.5, ctx_id=-1)
        normalizer = FaceNormalizer(target_size=(160, 160))
        
        # ADD YOUR IMAGE PATH HERE
        test_image_path = "/home/muthoni/Downloads/tets.jpg"  # CHANGE THIS PATH
        
        # Check if image exists
        if not os.path.exists(test_image_path):
            print(f"❌ Image not found: {test_image_path}")
            print("Please update the test_image_path variable with your actual image path")
            return
        
        # Load image
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"❌ Failed to load image: {test_image_path}")
            return
        
        print(f"✅ Loaded image: {test_image_path}")
        print(f"   Original size: {image.shape}")
        
        # First detect faces to get face region and keypoints
        print("\n--- Detecting Face for Normalization ---")
        faces = detector.detect_faces(image)
        
        if not faces:
            print("❌ No faces detected. Cannot test normalization.")
            return
        
        face = faces[0]  # Use first detected face
        face_box = face['box']
        keypoints = face['keypoints']
        
        print(f"✅ Using face: {face_box}")
        print(f"   Keypoints available: {len(keypoints) > 0}")
        
        # Extract face ROI
        face_roi = detector.extract_face_roi(image, face_box, margin=0.2)
        print(f"   Face ROI size: {face_roi.shape}")
        
        # Test 1: Basic resizing
        print("\n--- Test 1: Basic Resizing ---")
        resized = normalizer.resize_face(face_roi)
        print(f"   Resized to: {resized.shape}")
        
        # Test 2: Face alignment (if keypoints available)
        print("\n--- Test 2: Face Alignment ---")
        if keypoints and len(keypoints) >= 2:
            aligned = normalizer.align_face(face_roi, keypoints)
            print(f"   Aligned face size: {aligned.shape}")
        else:
            aligned = resized
            print("   No keypoints for alignment, using resized face")
        
        # Test 3: Histogram normalization
        print("\n--- Test 3: Histogram Normalization ---")
        hist_normalized = normalizer.normalize_histogram(aligned)
        print(f"   Histogram normalized size: {hist_normalized.shape}")
        
        # Test 4: CLAHE enhancement
        print("\n--- Test 4: CLAHE Enhancement ---")
        clahe_enhanced = normalizer.apply_clahe(aligned)
        print(f"   CLAHE enhanced size: {clahe_enhanced.shape}")
        
        # Test 5: Denoising
        print("\n--- Test 5: Denoising ---")
        denoised = normalizer.denoise(aligned)
        print(f"   Denoised size: {denoised.shape}")
        
        # Test 6: Pixel standardization
        print("\n--- Test 6: Pixel Standardization ---")
        standardized_minmax = normalizer.standardize_pixels(aligned, method='minmax')
        standardized_meanstd = normalizer.standardize_pixels(aligned, method='mean_std')
        print(f"   MinMax standardized: {standardized_minmax.shape}, range: [{standardized_minmax.min():.3f}, {standardized_minmax.max():.3f}]")
        print(f"   MeanStd standardized: {standardized_meanstd.shape}, range: [{standardized_meanstd.min():.3f}, {standardized_meanstd.max():.3f}]")
        
        # Test 7: Complete preprocessing pipeline
        print("\n--- Test 7: Complete Preprocessing Pipeline ---")
        preprocessed = normalizer.preprocess_for_model(face_roi, keypoints, enhance=True)
        print(f"   Preprocessed size: {preprocessed.shape}")
        print(f"   Preprocessed range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
        
        # Test 8: Quality check
        print("\n--- Test 8: Quality Check ---")
        quality_check = normalizer.check_face_quality(face_roi)
        print(f"   Quality passed: {quality_check['passed']}")
        if quality_check['passed']:
            print(f"   Brightness: {quality_check['brightness']:.1f}")
            print(f"   Sharpness: {quality_check['sharpness']:.1f}")
            print(f"   Size: {quality_check['size']}")
        else:
            print(f"   Reason: {quality_check['reason']}")
        
        # Test 9: Face augmentation
        print("\n--- Test 9: Face Augmentation ---")
        augmented_faces = normalizer.augment_face(aligned)
        print(f"   Generated {len(augmented_faces)} augmented versions")
        
        # Save results
        print("\n--- Saving Results ---")
        base_path = test_image_path.replace('.jpg', '').replace('.png', '')
        
        # Save different processing stages
        cv2.imwrite(f"{base_path}_original_roi.jpg", face_roi)
        cv2.imwrite(f"{base_path}_resized.jpg", resized)
        cv2.imwrite(f"{base_path}_aligned.jpg", aligned)
        cv2.imwrite(f"{base_path}_hist_norm.jpg", hist_normalized)
        cv2.imwrite(f"{base_path}_clahe.jpg", clahe_enhanced)
        cv2.imwrite(f"{base_path}_denoised.jpg", denoised)
        
        # Convert standardized back to uint8 for saving
        standardized_save = ((standardized_minmax * 255).astype(np.uint8))
        cv2.imwrite(f"{base_path}_standardized.jpg", standardized_save)
        
        print(f"💾 Saved normalization results with prefix: {base_path}_")
        print(f"\n✅ Normalization test completed!")
    
    # Run the test
    test_normalization()
