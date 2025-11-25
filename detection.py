"""
Face Detection Module
Handles face detection using InsightFace SCRFD (Sample and Computation Redistribution for Face Detection)
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available, install with: pip install insightface")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detector using InsightFace SCRFD with quality assessment
    """
    
    def __init__(self, model_name='buffalo_l', min_confidence=0.5, ctx_id=-1):
        """
        Initialize face detector with InsightFace
        
        Args:
            model_name: InsightFace model pack ('buffalo_l', 'buffalo_s', 'antelopev2')
            min_confidence: Minimum confidence threshold for detection (0.0-1.0)
            ctx_id: GPU device id, -1 for CPU
        """
        self.min_confidence = min_confidence
        self.ctx_id = ctx_id
        self.detector = None
        
        if INSIGHTFACE_AVAILABLE:
            try:
                # Initialize FaceAnalysis with SCRFD detector
                self.detector = FaceAnalysis(
                    name=model_name,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']
                )
                self.detector.prepare(ctx_id=ctx_id, det_thresh=min_confidence, det_size=(640, 640))
                logger.info(f"Initialized InsightFace SCRFD detector with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize InsightFace: {e}")
                self._init_fallback()
        else:
            logger.warning("InsightFace not available, using OpenCV fallback")
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback OpenCV detector"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        logger.info("Initialized OpenCV Haar Cascade detector (fallback)")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using InsightFace SCRFD
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of detected faces with bounding boxes, confidence scores, and landmarks
            Format: [{'box': (x, y, w, h), 'confidence': float, 'keypoints': dict, 'embedding': np.ndarray}]
        """
        if image is None or image.size == 0:
            logger.error("Invalid input image")
            return []
        
        if INSIGHTFACE_AVAILABLE and hasattr(self.detector, 'get'):
            return self._detect_insightface(image)
        else:
            return self._detect_opencv(image)
    
    def _detect_insightface(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using InsightFace SCRFD"""
        try:
            # InsightFace expects RGB format
            faces = self.detector.get(image)
            
            results = []
            for face in faces:
                # Get bounding box
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                w = x2 - x
                h = y2 - y
                
                # Get facial landmarks (5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth)
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
                
                # Get detection score
                confidence = float(face.det_score) if hasattr(face, 'det_score') else 1.0
                
                # Get embedding if available (from ArcFace)
                embedding = face.embedding if hasattr(face, 'embedding') else None
                
                # Get pose information
                pose = None
                if hasattr(face, 'pose'):
                    pose = face.pose
                
                results.append({
                    'box': (x, y, w, h),
                    'confidence': confidence,
                    'keypoints': keypoints,
                    'embedding': embedding,
                    'pose': pose,
                    'age': face.age if hasattr(face, 'age') else None,
                    'gender': face.gender if hasattr(face, 'gender') else None
                })
            
            logger.info(f"InsightFace SCRFD detected {len(results)} faces")
            return results
            
        except Exception as e:
            logger.error(f"InsightFace detection failed: {e}")
            return self._detect_opencv(image)
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using OpenCV Haar Cascades (fallback)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'box': (x, y, w, h),
                'confidence': 0.95,
                'keypoints': {},
                'embedding': None,
                'pose': None
            })
        
        logger.info(f"OpenCV detected {len(results)} faces (fallback)")
        return results
    
    def get_largest_face(self, detections: List[Dict]) -> Optional[Dict]:
        """
        Get the largest detected face
        
        Args:
            detections: List of face detections
            
        Returns:
            Largest face detection or None
        """
        if not detections:
            return None
        
        largest = max(detections, key=lambda d: d['box'][2] * d['box'][3])
        return largest
    
    def assess_face_quality(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict:
        """
        Assess the quality of a detected face
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, w, h)
            
        Returns:
            Quality metrics dictionary
        """
        x, y, w, h = face_box
        face_roi = image[y:y+h, x:x+w]
        
        if face_roi.size == 0:
            return {'quality_score': 0.0, 'issues': ['Invalid ROI']}
        
        issues = []
        
        # Check size
        min_size = 80
        if w < min_size or h < min_size:
            issues.append(f'Face too small ({w}x{h})')
        
        # Check brightness
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 40:
            issues.append('Too dark')
        elif brightness > 220:
            issues.append('Too bright')
        
        # Check blur (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            issues.append(f'Blurry (score: {laplacian_var:.1f})')
        
        # Calculate overall quality score
        quality_score = 1.0
        if w < min_size or h < min_size:
            quality_score *= 0.5
        if brightness < 40 or brightness > 220:
            quality_score *= 0.7
        if laplacian_var < 100:
            quality_score *= 0.6
        
        return {
            'quality_score': quality_score,
            'brightness': brightness,
            'sharpness': laplacian_var,
            'size': (w, h),
            'issues': issues
        }
    
    def extract_face_roi(self, image: np.ndarray, face_box: Tuple[int, int, int, int], 
                        margin: float = 0.2) -> np.ndarray:
        """
        Extract face region of interest with margin
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, w, h)
            margin: Margin to add around face (percentage)
            
        Returns:
            Extracted face ROI
        """
        x, y, w, h = face_box
        h_img, w_img = image.shape[:2]
        
        # Add margin
        margin_w = int(w * margin)
        margin_h = int(h * margin)
        
        x1 = max(0, x - margin_w)
        y1 = max(0, y - margin_h)
        x2 = min(w_img, x + w + margin_w)
        y2 = min(h_img, y + h + margin_h)
        
        face_roi = image[y1:y2, x1:x2]
        return face_roi
    
    def __del__(self):
        """Cleanup resources"""
        pass


# TEST CODE - Add your image path here
if __name__ == "__main__":
    import os
    
    def test_detection():
        """Test face detection with static images"""
        print("=== FACE DETECTION TEST ===")
        
        # Initialize detector
        detector = FaceDetector(model_name='buffalo_l', min_confidence=0.5, ctx_id=-1)
        
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
        print(f"   Image size: {image.shape}")
        
        # Detect faces
        print("\n--- Running Face Detection ---")
        faces = detector.detect_faces(image)
        
        print(f"🔍 Detected {len(faces)} face(s)")
        
        # Process each detected face
        for i, face in enumerate(faces):
            print(f"\n--- Face {i+1} ---")
            print(f"   Bounding box: {face['box']}")
            print(f"   Confidence: {face['confidence']:.3f}")
            print(f"   Has keypoints: {len(face['keypoints']) > 0}")
            print(f"   Has embedding: {face['embedding'] is not None}")
            
            # Assess face quality
            quality = detector.assess_face_quality(image, face['box'])
            print(f"   Quality score: {quality['quality_score']:.3f}")
            print(f"   Brightness: {quality['brightness']:.1f}")
            print(f"   Sharpness: {quality['sharpness']:.1f}")
            print(f"   Size: {quality['size']}")
            if quality['issues']:
                print(f"   Issues: {', '.join(quality['issues'])}")
            
            # Extract face ROI
            face_roi = detector.extract_face_roi(image, face['box'])
            print(f"   Face ROI size: {face_roi.shape}")
        
        # Test with largest face if multiple detected
        if faces:
            largest_face = detector.get_largest_face(faces)
            print(f"\n--- Largest Face ---")
            print(f"   Box: {largest_face['box']}")
            print(f"   Area: {largest_face['box'][2] * largest_face['box'][3]} pixels")
        
        print(f"\n✅ Detection test completed!")
        
        # Optional: Save result image with bounding boxes
        result_image = image.copy()
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_image, f"{face['confidence']:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save result
        output_path = test_image_path.replace('.jpg', '_detection_result.jpg').replace('.png', '_detection_result.png')
        cv2.imwrite(output_path, result_image)
        print(f"💾 Saved result image: {output_path}")
    
    # Run the test
    test_detection()
