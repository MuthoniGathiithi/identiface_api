"""
Configuration file for Face Recognition Service
"""
import os
from typing import Optional

class Config:
    """Configuration settings"""
    
    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "True").lower() == "true"
    
    # InsightFace Model Settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "buffalo_l")  # buffalo_l, buffalo_s, antelopev2
    CTX_ID: int = int(os.getenv("CTX_ID", "-1"))  # -1 for CPU, 0+ for GPU
    
    # Detection Settings
    DETECTION_CONFIDENCE: float = float(os.getenv("DETECTION_CONFIDENCE", "0.5"))
    DETECTION_SIZE: tuple = (640, 640)  # Detection input size
    
    # Matching Settings
    MATCHING_THRESHOLD: float = float(os.getenv("MATCHING_THRESHOLD", "0.4"))
    MATCHING_METRIC: str = os.getenv("MATCHING_METRIC", "cosine")  # cosine or euclidean
    
    # Pose Estimation Settings
    YAW_THRESHOLD: float = float(os.getenv("YAW_THRESHOLD", "20.0"))
    PITCH_THRESHOLD: float = float(os.getenv("PITCH_THRESHOLD", "15.0"))
    
    # Normalization Settings
    TARGET_FACE_SIZE: tuple = (160, 160)
    
    # Video Capture Settings
    CAMERA_ID: int = int(os.getenv("CAMERA_ID", "0"))
    CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "640"))
    CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "480"))
    CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", "30"))
    
    # Quality Thresholds
    MIN_FACE_SIZE: int = 80
    MIN_BRIGHTNESS: float = 40.0
    MAX_BRIGHTNESS: float = 220.0
    MIN_SHARPNESS: float = 100.0
    MIN_QUALITY_SCORE: float = 0.6
    
    # Database Settings
    DJANGO_SETTINGS_MODULE: Optional[str] = os.getenv("DJANGO_SETTINGS_MODULE")
    CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", "1000"))
    
    # Enrollment Settings
    REQUIRED_POSES: list = ["front", "left", "right", "down"]
    ENROLLMENT_TIMEOUT: int = 300  # seconds
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS Settings
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    @classmethod
    def get_model_path(cls) -> str:
        """Get InsightFace model path"""
        home = os.path.expanduser("~")
        return os.path.join(home, ".insightface", "models", cls.MODEL_NAME)
    
    @classmethod
    def is_gpu_available(cls) -> bool:
        """Check if GPU is available"""
        return cls.CTX_ID >= 0
    
    @classmethod
    def get_device_name(cls) -> str:
        """Get device name for logging"""
        if cls.is_gpu_available():
            return f"GPU:{cls.CTX_ID}"
        return "CPU"


# Development configuration
class DevelopmentConfig(Config):
    """Development environment configuration"""
    API_RELOAD = True
    LOG_LEVEL = "DEBUG"


# Production configuration
class ProductionConfig(Config):
    """Production environment configuration"""
    API_RELOAD = False
    LOG_LEVEL = "WARNING"
    CORS_ORIGINS = []  # Set specific origins


# Testing configuration
class TestingConfig(Config):
    """Testing environment configuration"""
    MODEL_NAME = "buffalo_s"  # Smaller model for faster tests
    CACHE_SIZE = 100


# Configuration selector
def get_config():
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()


# Export default config
config = get_config()
