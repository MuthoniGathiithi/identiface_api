"""
Input Validation and Sanitization Module
Ensures all API inputs are validated and safe
"""
from typing import Optional, Tuple
import logging
import cv2
import numpy as np
from pathlib import Path


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error"""
    pass


class InputValidator:
    """Validates and sanitizes API inputs"""
    
    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    MAX_FILE_SIZE_MB = 50
    MIN_FILE_SIZE_BYTES = 1000
    
    # Image dimensions
    MIN_IMAGE_SIZE = (50, 50)
    MAX_IMAGE_SIZE = (4096, 4096)
    
    @staticmethod
    def validate_file_upload(
        file_bytes: bytes,
        filename: str,
        allowed_extensions: set = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded file
        
        Returns:
            (is_valid, error_message)
        """
        if allowed_extensions is None:
            allowed_extensions = InputValidator.ALLOWED_IMAGE_EXTENSIONS
        
        # Check file size
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > InputValidator.MAX_FILE_SIZE_MB:
            return False, f"File size exceeds {InputValidator.MAX_FILE_SIZE_MB}MB limit"
        
        if len(file_bytes) < InputValidator.MIN_FILE_SIZE_BYTES:
            return False, "File is too small or empty"
        
        # Check extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            return False, f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"
        
        return True, None
    
    @staticmethod
    def validate_image_data(image: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Validate image numpy array
        
        Returns:
            (is_valid, error_message)
        """
        if image is None:
            return False, "Invalid or corrupted image"
        
        if not isinstance(image, np.ndarray):
            return False, "Image must be numpy array"
        
        if image.size == 0:
            return False, "Image is empty"
        
        # Check dimensions
        height, width = image.shape[:2]
        if (width, height) < InputValidator.MIN_IMAGE_SIZE:
            return False, f"Image too small. Minimum: {InputValidator.MIN_IMAGE_SIZE}"
        
        if (width, height) > InputValidator.MAX_IMAGE_SIZE:
            return False, f"Image too large. Maximum: {InputValidator.MAX_IMAGE_SIZE}"
        
        return True, None
    
    @staticmethod
    def validate_string_input(
        value: str,
        field_name: str,
        min_length: int = 1,
        max_length: int = 500,
        pattern: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate string input"""
        if not isinstance(value, str):
            return False, f"{field_name} must be a string"
        
        if len(value) < min_length or len(value) > max_length:
            return False, f"{field_name} length must be between {min_length} and {max_length}"
        
        if pattern:
            import re
            if not re.match(pattern, value):
                return False, f"{field_name} format is invalid"
        
        return True, None
    
    @staticmethod
    def validate_numeric_range(
        value: float,
        field_name: str,
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> Tuple[bool, Optional[str]]:
        """Validate numeric input within range"""
        if not isinstance(value, (int, float)):
            return False, f"{field_name} must be numeric"
        
        if value < min_val or value > max_val:
            return False, f"{field_name} must be between {min_val} and {max_val}"
        
        return True, None
    
    @staticmethod
    def validate_id_format(
        id_value: str,
        field_name: str = "ID"
    ) -> Tuple[bool, Optional[str]]:
        """Validate ID format (alphanumeric with dashes/underscores)"""
        if not id_value or not isinstance(id_value, str):
            return False, f"{field_name} is required and must be string"
        
        # Allow alphanumeric, dashes, underscores
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', id_value):
            return False, f"{field_name} contains invalid characters"
        
        if len(id_value) < 2 or len(id_value) > 100:
            return False, f"{field_name} must be 2-100 characters"
        
        return True, None


class OutputSanitizer:
    """Sanitizes outputs to prevent data leaks"""
    
    @staticmethod
    def sanitize_error_message(error: Exception, expose_details: bool = False) -> str:
        """
        Sanitize error message for API response
        
        Args:
            error: Exception object
            expose_details: Whether to expose technical details
        """
        error_str = str(error)
        
        if expose_details:
            return error_str
        
        # Hide sensitive details in production
        sensitive_keywords = ['database', 'connection', 'file', 'path', 'password']
        for keyword in sensitive_keywords:
            if keyword.lower() in error_str.lower():
                return "Internal server error. Please contact support."
        
        return error_str
    
    @staticmethod
    def sanitize_embedding(embedding: np.ndarray, max_decimals: int = 6) -> list:
        """Convert embedding to list with limited precision"""
        return np.round(embedding, max_decimals).tolist()
