"""
Face Enrollment Module
Handles multi-pose face enrollment process with advanced models
"""
import cv2
import numpy as np
from typing import Dict, List, Optional
import logging
import time
from datetime import datetime

from detection_advanced import FaceDetectorAdvanced
from feature_extraction_advanced import FeatureExtractorAdvanced
from normalization_advanced import FaceNormalizerAdvanced
from pose_estimation import PoseEstimator, FacePose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnrollmentSession:
    """
    Manages a complete face enrollment session with multiple poses
    """
    
    def __init__(self, user_id: str, detector: FaceDetectorAdvanced, 
                 extractor: FeatureExtractorAdvanced, normalizer: FaceNormalizerAdvanced,
                 pose_estimator: PoseEstimator):
        """
        Initialize enrollment session
        
        Args:
            user_id: Unique identifier for the user
            detector: Advanced face detector instance
            extractor: Advanced feature extractor instance
            normalizer: Advanced face normalizer instance
            pose_estimator: Pose estimator instance
        """
        self.user_id = user_id
        self.detector = detector
        self.extractor = extractor
        self.normalizer = normalizer
        self.pose_estimator = pose_estimator
        
        self.required_poses = [FacePose.FRONT, FacePose.LEFT, FacePose.RIGHT, FacePose.DOWN]
        self.captured_data = {}
        self.current_pose_index = 0
        self.session_start_time = datetime.now()
        
        logger.info(f"Started enrollment session for user: {user_id}")
    
    def get_current_required_pose(self) -> Optional[FacePose]:
        """Get the current required pose"""
        if self.current_pose_index < len(self.required_poses):
            return self.required_poses[self.current_pose_index]
        return None
    
    def get_progress(self) -> Dict:
        """Get enrollment progress"""
        # Create poses_captured dict for JavaScript compatibility
        poses_captured = {
            'front': FacePose.FRONT in self.captured_data,
            'left': FacePose.LEFT in self.captured_data,
            'right': FacePose.RIGHT in self.captured_data,
            'down': FacePose.DOWN in self.captured_data
        }
        
        return {
            'user_id': self.user_id,
            'total_poses_required': len(self.required_poses),
            'poses_captured_count': len(self.captured_data),
            'poses_captured': poses_captured,
            'current_pose': self.get_current_required_pose().value if self.get_current_required_pose() else None,
            'complete': self.is_complete(),
            'captured_poses': [pose.value for pose in self.captured_data.keys()]
        }
    
    def is_complete(self) -> bool:
        """Check if enrollment is complete"""
        return len(self.captured_data) == len(self.required_poses)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a frame for enrollment
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processing result with feedback
        """
        if self.is_complete():
            return {
                'status': 'complete',
                'message': 'Enrollment complete!',
                'progress': self.get_progress()
            }
        
        # Detect faces
        faces = self.detector.detect_faces(frame)
        
        if not faces:
            return {
                'status': 'no_face',
                'message': 'No face detected. Please position your face in the frame.',
                'progress': self.get_progress()
            }
        
        if len(faces) > 1:
            return {
                'status': 'multiple_faces',
                'message': 'Multiple faces detected. Please ensure only one person is in frame.',
                'progress': self.get_progress()
            }
        
        # Get the detected face
        face = faces[0]
        face_box = face['box']
        keypoints = face['keypoints']
        
        # Validate keypoints are present (required for pose estimation)
        # If keypoints missing, use fallback based on bounding box
        if not keypoints or len(keypoints) < 3:
            logger.warning(f"Insufficient keypoints detected: {len(keypoints) if keypoints else 0}. Using fallback.")
            # Generate fallback keypoints from bounding box
            x, y, w, h = face_box
            keypoints = {
                'nose': (int(x + w/2), int(y + h/2)),
                'left_eye': (int(x + w*0.3), int(y + h*0.35)),
                'right_eye': (int(x + w*0.7), int(y + h*0.35)),
                'left_mouth': (int(x + w*0.35), int(y + h*0.75)),
                'right_mouth': (int(x + w*0.65), int(y + h*0.75))
            }
            logger.info(f"Using fallback keypoints based on bounding box for face {face_box}")
        
        # Check face quality (lowered threshold to 0.4 for better usability)
        quality = self.detector.assess_face_quality(frame, face_box)
        
        if quality['quality_score'] < 0.4:
            issues_str = ', '.join(quality['issues']) if quality['issues'] else 'unknown quality issue'
            feedback_msg = f"Please improve: {issues_str}. Try moving closer or adjusting lighting."
            return {
                'status': 'poor_quality',
                'message': feedback_msg,
                'feedback': feedback_msg,
                'quality': quality,
                'progress': self.get_progress()
            }
        
        # Get current required pose
        required_pose = self.get_current_required_pose()
        
        # Estimate pose
        h, w = frame.shape[:2]
        pose_validation = self.pose_estimator.validate_pose_for_enrollment(
            keypoints, (h, w), required_pose
        )
        
        if not pose_validation['valid']:
            return {
                'status': 'waiting',
                'message': pose_validation['message'],
                'feedback': pose_validation['message'],
                'pose_info': pose_validation,
                'progress': self.get_progress()
            }
        
        # All checks passed - ready to capture, auto-capture this pose
        capture_result = self.capture_pose(frame, face, quality)
        
        return {
            'status': 'ready',
            'message': f"✓ Captured {required_pose.value} pose!",
            'feedback': f"✓ {required_pose.value.upper()} pose captured",
            'captured': capture_result,
            'face': face,
            'quality': quality,
            'pose_info': pose_validation,
            'progress': self.get_progress()
        }
    
    def capture_pose(self, frame: np.ndarray, face_data: Dict, quality: Dict = None) -> bool:
        """
        Capture and store data for current pose
        
        Args:
            frame: Input frame
            face_data: Face detection data
            quality: Face quality assessment data
            
        Returns:
            True if capture successful
        """
        if self.is_complete():
            return False
        
        required_pose = self.get_current_required_pose()
        
        try:
            # Extract face ROI
            face_box = face_data['box']
            keypoints = face_data['keypoints']
            
            face_roi = self.detector.extract_face_roi(frame, face_box, margin=0.2)
            
            # Normalize face
            normalized_face = self.normalizer.preprocess_for_model(face_roi, keypoints, enhance=True)
            
            # Extract embedding
            embedding = self.extractor.extract_embedding_from_detection(frame, face_box, keypoints)
            
            if embedding is None:
                logger.error("Failed to extract embedding")
                return False
            
            # Store captured data
            self.captured_data[required_pose] = {
                'frame': frame.copy(),
                'face_roi': face_roi,
                'normalized_face': normalized_face,
                'embedding': embedding,
                'keypoints': keypoints,
                'face_box': face_box,
                'timestamp': datetime.now(),
                'quality_score': quality.get('quality_score', 0.0) if quality else 0.0
            }
            
            logger.info(f"Captured {required_pose.value} pose for user {self.user_id}")
            
            # Move to next pose
            self.current_pose_index += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to capture pose: {e}")
            return False
    
    def get_enrollment_data(self) -> Optional[Dict]:
        """
        Get complete enrollment data
        
        Returns:
            Dictionary with all enrollment data or None if incomplete
        """
        if not self.is_complete():
            logger.warning("Enrollment not complete")
            return None
        
        # Calculate average embedding from all poses
        embeddings = [data['embedding'] for data in self.captured_data.values()]
        average_embedding = np.mean(embeddings, axis=0)
        average_embedding = average_embedding / np.linalg.norm(average_embedding)
        
        return {
            'user_id': self.user_id,
            'enrollment_date': self.session_start_time.isoformat(),
            'completion_date': datetime.now().isoformat(),
            'poses': {
                pose.value: {
                    'embedding': data['embedding'].tolist(),
                    'keypoints': data['keypoints'],
                    'face_box': data['face_box'],
                    'quality_score': data['quality_score'],
                    'timestamp': data['timestamp'].isoformat()
                }
                for pose, data in self.captured_data.items()
            },
            'average_embedding': average_embedding.tolist(),
            'embedding_size': len(average_embedding)
        }
    
    def save_enrollment_images(self, output_dir: str) -> bool:
        """
        Save enrollment images to disk
        
        Args:
            output_dir: Directory to save images
            
        Returns:
            True if successful
        """
        import os
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for pose, data in self.captured_data.items():
                # Save original frame
                frame_path = os.path.join(output_dir, f"{self.user_id}_{pose.value}_frame.jpg")
                cv2.imwrite(frame_path, data['frame'])
                
                # Save face ROI
                roi_path = os.path.join(output_dir, f"{self.user_id}_{pose.value}_face.jpg")
                cv2.imwrite(roi_path, data['face_roi'])
            
            logger.info(f"Saved enrollment images to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save enrollment images: {e}")
            return False
    
    def reset(self):
        """Reset enrollment session"""
        self.captured_data = {}
        self.current_pose_index = 0
        self.session_start_time = datetime.now()
        logger.info(f"Reset enrollment session for user {self.user_id}")


class EnrollmentManager:
    """
    Manages multiple enrollment sessions
    """
    
    def __init__(self):
        """Initialize enrollment manager"""
        self.sessions = {}
        self.detector = None
        self.extractor = None
        self.normalizer = None
        self.pose_estimator = None
        
        logger.info("Initialized EnrollmentManager")
    
    def initialize_components(self, model_name: str = 'buffalo_l', ctx_id: int = -1):
        """
        Initialize all required components
        
        Args:
            model_name: InsightFace model name
            ctx_id: GPU device id (-1 for CPU)
        """
        self.detector = FaceDetector(model_name=model_name, min_confidence=0.5, ctx_id=ctx_id)
        self.extractor = FeatureExtractor(model_name=model_name, ctx_id=ctx_id)
        self.normalizer = FaceNormalizer(target_size=(160, 160))
        self.pose_estimator = PoseEstimator(yaw_threshold=20.0, pitch_threshold=15.0)
        
        logger.info("Initialized all enrollment components")
    
    def start_session(self, user_id: str) -> EnrollmentSession:
        """
        Start a new enrollment session
        
        Args:
            user_id: User identifier
            
        Returns:
            EnrollmentSession instance
        """
        if not all([self.detector, self.extractor, self.normalizer, self.pose_estimator]):
            raise RuntimeError("Components not initialized. Call initialize_components() first.")
        
        session = EnrollmentSession(
            user_id=user_id,
            detector=self.detector,
            extractor=self.extractor,
            normalizer=self.normalizer,
            pose_estimator=self.pose_estimator
        )
        
        self.sessions[user_id] = session
        return session
    
    def get_session(self, user_id: str) -> Optional[EnrollmentSession]:
        """Get existing enrollment session"""
        return self.sessions.get(user_id)
    
    def end_session(self, user_id: str) -> Optional[Dict]:
        """
        End enrollment session and return data
        
        Args:
            user_id: User identifier
            
        Returns:
            Enrollment data or None
        """
        session = self.sessions.get(user_id)
        if not session:
            return None
        
        enrollment_data = session.get_enrollment_data()
        
        if enrollment_data:
            del self.sessions[user_id]
            logger.info(f"Ended enrollment session for user {user_id}")
        
        return enrollment_data
    
    def cancel_session(self, user_id: str):
        """Cancel enrollment session"""
        if user_id in self.sessions:
            del self.sessions[user_id]
            logger.info(f"Cancelled enrollment session for user {user_id}")
