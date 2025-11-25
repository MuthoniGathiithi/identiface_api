"""
Video Capture Module
Handles video capture with quality checks and face detection
Ensures clear, unobstructed face capture for enrollment
"""
import cv2
import numpy as np
from typing import Optional, Dict, Callable
import logging
import time
from queue import Queue
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Video capture with quality assessment and face detection
    """
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize video capture
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)
        self.capture_thread = None
        
        logger.info(f"Initialized VideoCapture (camera_id={camera_id}, {width}x{height}@{fps}fps)")
    
    def start(self) -> bool:
        """
        Start video capture
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            logger.info("Video capture started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start video capture: {e}")
            return False
    
    def _capture_loop(self):
        """Background thread for capturing frames"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Update queue (drop old frames if full)
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except:
                            pass
                    try:
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
            time.sleep(1.0 / self.fps)
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read latest frame from camera
        
        Returns:
            Frame as numpy array or None
        """
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
            return None
        except:
            return None
    
    def stop(self):
        """Stop video capture"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Video capture stopped")
    
    def check_frame_quality(self, frame: np.ndarray) -> Dict:
        """
        Check if frame meets quality requirements
        
        Args:
            frame: Input frame
            
        Returns:
            Quality assessment dictionary
        """
        if frame is None or frame.size == 0:
            return {
                'passed': False,
                'score': 0.0,
                'issues': ['Invalid frame']
            }
        
        issues = []
        score = 1.0
        
        # Check brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 50:
            issues.append('Too dark - improve lighting')
            score *= 0.5
        elif brightness > 220:
            issues.append('Too bright - reduce lighting')
            score *= 0.7
        
        # Check sharpness (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            issues.append('Image is blurry - hold camera steady')
            score *= 0.6
        
        # Check contrast
        contrast = gray.std()
        if contrast < 30:
            issues.append('Low contrast - improve lighting')
            score *= 0.7
        
        # Check for motion blur (compare with previous frame if available)
        # This would require storing previous frame
        
        passed = score >= 0.6 and len(issues) == 0
        
        return {
            'passed': passed,
            'score': float(score),
            'brightness': float(brightness),
            'sharpness': float(laplacian_var),
            'contrast': float(contrast),
            'issues': issues
        }
    
    def check_face_visibility(self, frame: np.ndarray, face_box: tuple) -> Dict:
        """
        Check if face is clearly visible and unobstructed
        
        Args:
            frame: Input frame
            face_box: Face bounding box (x, y, w, h)
            
        Returns:
            Visibility assessment dictionary
        """
        x, y, w, h = face_box
        h_img, w_img = frame.shape[:2]
        
        issues = []
        score = 1.0
        
        # Check if face is too close to edges
        margin = 20
        if x < margin or y < margin or x + w > w_img - margin or y + h > h_img - margin:
            issues.append('Face too close to edge - center your face')
            score *= 0.7
        
        # Check face size (should be reasonable portion of frame)
        face_area = w * h
        frame_area = w_img * h_img
        face_ratio = face_area / frame_area
        
        if face_ratio < 0.05:
            issues.append('Face too small - move closer to camera')
            score *= 0.5
        elif face_ratio > 0.6:
            issues.append('Face too large - move back from camera')
            score *= 0.7
        
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Check for obstructions using edge detection
        edges = cv2.Canny(face_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Too many edges might indicate obstructions (glasses, hands, etc.)
        if edge_density > 0.3:
            issues.append('Possible obstruction detected - ensure face is clear')
            score *= 0.8
        
        # Check face region brightness uniformity
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness_std = np.std(gray_face)
        
        if brightness_std > 60:
            issues.append('Uneven lighting on face')
            score *= 0.9
        
        passed = score >= 0.7 and len(issues) == 0
        
        return {
            'passed': passed,
            'score': float(score),
            'face_ratio': float(face_ratio),
            'edge_density': float(edge_density),
            'issues': issues
        }
    
    def capture_best_frame(self, detector, duration: float = 5.0, 
                          quality_threshold: float = 0.7) -> Optional[Dict]:
        """
        Capture the best quality frame over a duration
        
        Args:
            detector: Face detector instance
            duration: Duration to capture in seconds
            quality_threshold: Minimum quality score
            
        Returns:
            Dictionary with best frame and metadata or None
        """
        start_time = time.time()
        best_frame = None
        best_score = 0.0
        best_face = None
        
        logger.info(f"Capturing best frame over {duration} seconds...")
        
        while time.time() - start_time < duration:
            frame = self.read_frame()
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Check frame quality
            frame_quality = self.check_frame_quality(frame)
            
            if not frame_quality['passed']:
                continue
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            if not faces:
                continue
            
            # Get largest face
            face = detector.get_largest_face(faces)
            
            # Check face visibility
            face_visibility = self.check_face_visibility(frame, face['box'])
            
            # Calculate combined score
            combined_score = (frame_quality['score'] + face_visibility['score']) / 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_frame = frame.copy()
                best_face = face
            
            time.sleep(0.1)
        
        if best_frame is not None and best_score >= quality_threshold:
            logger.info(f"Captured best frame with quality score: {best_score:.2f}")
            return {
                'frame': best_frame,
                'face': best_face,
                'quality_score': best_score
            }
        else:
            logger.warning(f"Failed to capture frame meeting quality threshold ({best_score:.2f} < {quality_threshold})")
            return None
    
    def capture_with_feedback(self, detector, callback: Optional[Callable] = None) -> Optional[np.ndarray]:
        """
        Capture frame with real-time feedback
        
        Args:
            detector: Face detector instance
            callback: Optional callback function for feedback (receives frame and status dict)
            
        Returns:
            Captured frame or None
        """
        while True:
            frame = self.read_frame()
            
            if frame is None:
                time.sleep(0.1)
                continue
            
            # Check quality
            frame_quality = self.check_frame_quality(frame)
            
            # Detect face
            faces = detector.detect_faces(frame)
            
            status = {
                'frame_quality': frame_quality,
                'faces_detected': len(faces),
                'ready': False
            }
            
            if faces:
                face = detector.get_largest_face(faces)
                face_visibility = self.check_face_visibility(frame, face['box'])
                status['face_visibility'] = face_visibility
                status['face'] = face
                
                # Check if ready to capture
                if frame_quality['passed'] and face_visibility['passed']:
                    status['ready'] = True
            
            # Call feedback callback
            if callback:
                should_capture = callback(frame, status)
                if should_capture and status['ready']:
                    return frame.copy()
            
            time.sleep(0.1)
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
    
    def __del__(self):
        """Cleanup"""
        self.stop()
