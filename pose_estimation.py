"""
Pose Estimation Module
Detects and validates face poses (front, left, right, down) for enrollment
"""
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacePose(Enum):
    """Face pose types"""
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    DOWN = "down"
    UNKNOWN = "unknown"


class PoseEstimator:
    """
    Estimate face pose using facial landmarks
    """
    
    def __init__(self, yaw_threshold: float = 20.0, pitch_threshold: float = 15.0):
        """
        Initialize pose estimator
        
        Args:
            yaw_threshold: Threshold for left/right pose detection (degrees)
            pitch_threshold: Threshold for down pose detection (degrees)
        """
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        
        # 3D model points of a generic face
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        logger.info(f"Initialized PoseEstimator (yaw_threshold={yaw_threshold}, pitch_threshold={pitch_threshold})")
    
    def estimate_pose_from_landmarks(self, keypoints: Dict, image_shape: Tuple[int, int]) -> Dict:
        """
        Estimate pose angles from facial landmarks
        
        Args:
            keypoints: Dictionary with facial landmarks
                      Expected keys: 'left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth'
            image_shape: Image shape (height, width)
            
        Returns:
            Dictionary with pose information
        """
        if not keypoints or len(keypoints) < 5:
            logger.warning("Insufficient keypoints for pose estimation")
            return {
                'pose': FacePose.UNKNOWN,
                'yaw': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
                'confidence': 0.0
            }
        
        try:
            # Extract 2D image points from keypoints
            image_points = np.array([
                keypoints.get('nose', (0, 0)),
                keypoints.get('left_mouth', (0, 0)),  # Using left_mouth as chin approximation
                keypoints.get('left_eye', (0, 0)),
                keypoints.get('right_eye', (0, 0)),
                keypoints.get('left_mouth', (0, 0)),
                keypoints.get('right_mouth', (0, 0))
            ], dtype=np.float64)
            
            # Camera internals
            height, width = image_shape
            focal_length = width
            center = (width / 2, height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Assuming no lens distortion
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return self._estimate_pose_simple(keypoints)
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Calculate Euler angles
            yaw, pitch, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
            
            # Determine pose
            pose = self._classify_pose(yaw, pitch)
            
            # Calculate confidence based on how clear the pose is
            confidence = self._calculate_pose_confidence(yaw, pitch)
            
            return {
                'pose': pose,
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return self._estimate_pose_simple(keypoints)
    
    def _estimate_pose_simple(self, keypoints: Dict) -> Dict:
        """
        Simple pose estimation using eye and nose positions
        
        Args:
            keypoints: Facial landmarks
            
        Returns:
            Pose information dictionary
        """
        if 'left_eye' not in keypoints or 'right_eye' not in keypoints:
            return {
                'pose': FacePose.UNKNOWN,
                'yaw': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
                'confidence': 0.0
            }
        
        left_eye = np.array(keypoints['left_eye'])
        right_eye = np.array(keypoints['right_eye'])
        
        # Calculate eye center
        eye_center = (left_eye + right_eye) / 2
        
        # Calculate eye distance
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        # Estimate yaw from eye positions
        eye_diff_x = right_eye[0] - left_eye[0]
        
        if 'nose' in keypoints:
            nose = np.array(keypoints['nose'])
            # Nose position relative to eye center
            nose_offset_x = nose[0] - eye_center[0]
            yaw = (nose_offset_x / eye_distance) * 45  # Rough approximation
        else:
            yaw = 0.0
        
        # Estimate pitch from vertical positions
        if 'nose' in keypoints and 'left_mouth' in keypoints:
            nose = np.array(keypoints['nose'])
            mouth = np.array(keypoints['left_mouth'])
            vertical_distance = mouth[1] - nose[1]
            pitch = (vertical_distance / eye_distance - 1.5) * 30  # Rough approximation
        else:
            pitch = 0.0
        
        # Estimate roll from eye angle
        roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], eye_diff_x))
        
        pose = self._classify_pose(yaw, pitch)
        confidence = 0.5  # Lower confidence for simple estimation
        
        return {
            'pose': pose,
            'yaw': float(yaw),
            'pitch': float(pitch),
            'roll': float(roll),
            'confidence': float(confidence)
        }
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (yaw, pitch, roll)
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Tuple of (yaw, pitch, roll) in degrees
        """
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        # Convert to degrees
        yaw = np.degrees(z)
        pitch = np.degrees(x)
        roll = np.degrees(y)
        
        return yaw, pitch, roll
    
    def _classify_pose(self, yaw: float, pitch: float) -> FacePose:
        """
        Classify pose based on yaw and pitch angles
        
        Args:
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
            
        Returns:
            FacePose enum value
        """
        # Check for down pose first
        if pitch > self.pitch_threshold:
            return FacePose.DOWN
        
        # Check for left/right poses
        if yaw < -self.yaw_threshold:
            return FacePose.LEFT
        elif yaw > self.yaw_threshold:
            return FacePose.RIGHT
        else:
            return FacePose.FRONT
    
    def _calculate_pose_confidence(self, yaw: float, pitch: float) -> float:
        """
        Calculate confidence score for pose classification
        
        Args:
            yaw: Yaw angle
            pitch: Pitch angle
            
        Returns:
            Confidence score (0-1)
        """
        # Higher confidence when angles are clearly in one category
        yaw_confidence = min(abs(yaw) / self.yaw_threshold, 1.0)
        pitch_confidence = min(abs(pitch) / self.pitch_threshold, 1.0)
        
        # For front pose, confidence is higher when angles are close to 0
        if abs(yaw) < self.yaw_threshold / 2 and abs(pitch) < self.pitch_threshold / 2:
            return 1.0 - (abs(yaw) / self.yaw_threshold + abs(pitch) / self.pitch_threshold) / 2
        
        # For other poses, confidence is higher when clearly in that direction
        return max(yaw_confidence, pitch_confidence)
    
    def validate_pose_for_enrollment(self, keypoints: Dict, image_shape: Tuple[int, int],
                                    required_pose: FacePose) -> Dict:
        """
        Validate if detected pose matches required pose for enrollment
        
        Args:
            keypoints: Facial landmarks
            image_shape: Image shape
            required_pose: Required pose for this enrollment step
            
        Returns:
            Validation result dictionary
        """
        pose_info = self.estimate_pose_from_landmarks(keypoints, image_shape)
        detected_pose = pose_info['pose']
        confidence = pose_info['confidence']
        
        is_valid = detected_pose == required_pose and confidence > 0.6
        
        return {
            'valid': is_valid,
            'detected_pose': detected_pose.value,
            'required_pose': required_pose.value,
            'confidence': confidence,
            'yaw': pose_info['yaw'],
            'pitch': pose_info['pitch'],
            'roll': pose_info['roll'],
            'message': self._get_pose_guidance(detected_pose, required_pose)
        }
    
    def _get_pose_guidance(self, detected: FacePose, required: FacePose) -> str:
        """Get guidance message for user"""
        if detected == required:
            return f"Perfect! {required.value.capitalize()} pose detected."
        elif required == FacePose.FRONT:
            return "Please face the camera directly."
        elif required == FacePose.LEFT:
            return "Please turn your head to the left."
        elif required == FacePose.RIGHT:
            return "Please turn your head to the right."
        elif required == FacePose.DOWN:
            return "Please tilt your head down slightly."
        else:
            return f"Please adjust to {required.value} pose."
    
    def get_required_poses(self) -> List[FacePose]:
        """Get list of required poses for enrollment"""
        return [FacePose.FRONT, FacePose.LEFT, FacePose.RIGHT, FacePose.DOWN]
    
    def draw_pose_axes(self, image: np.ndarray, keypoints: Dict, 
                      pose_info: Dict, axis_length: int = 100) -> np.ndarray:
        """
        Draw pose axes on image for visualization
        
        Args:
            image: Input image
            keypoints: Facial landmarks
            pose_info: Pose information from estimate_pose_from_landmarks
            axis_length: Length of axes to draw
            
        Returns:
            Image with pose axes drawn
        """
        if 'nose' not in keypoints:
            return image
        
        nose_point = keypoints['nose']
        yaw = pose_info['yaw']
        pitch = pose_info['pitch']
        roll = pose_info['roll']
        
        # Draw text with pose information
        text = f"Pose: {pose_info['pose'].value} (Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f})"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image
