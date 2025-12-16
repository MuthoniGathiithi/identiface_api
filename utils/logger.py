"""
Logging and Monitoring Module
Comprehensive logging for debugging and monitoring
"""
import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class StructuredLogger:
    """Enhanced logger with structured logging capabilities"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.logger = logging.getLogger(name)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # File handler - daily rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'app.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'error.log',
            maxBytes=10*1024*1024,
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def log_request(self, method: str, endpoint: str, user_id: Optional[str] = None, **kwargs):
        """Log API request"""
        self.logger.info(
            f"REQUEST | {method} {endpoint} | User: {user_id}",
            extra={"details": kwargs}
        )
    
    def log_response(self, endpoint: str, status_code: int, duration_ms: float, **kwargs):
        """Log API response"""
        self.logger.info(
            f"RESPONSE | {endpoint} | Status: {status_code} | Duration: {duration_ms:.2f}ms",
            extra={"details": kwargs}
        )
    
    def log_error(self, message: str, error: Exception, **kwargs):
        """Log error with traceback"""
        self.logger.error(
            message,
            exc_info=True,
            extra={"details": kwargs}
        )
    
    def log_detection(self, faces_count: int, confidence_scores: list, **kwargs):
        """Log face detection results"""
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        self.logger.info(
            f"DETECTION | Faces: {faces_count} | Avg Confidence: {avg_confidence:.3f}",
            extra={"details": kwargs}
        )
    
    def log_matching(self, similarity: float, confidence: float, threshold: float, **kwargs):
        """Log face matching results"""
        passed = "PASSED" if confidence >= threshold else "FAILED"
        self.logger.info(
            f"MATCHING | {passed} | Similarity: {similarity:.3f} | Confidence: {confidence:.3f} | Threshold: {threshold:.3f}",
            extra={"details": kwargs}
        )
    
    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics"""
        self.logger.info(
            f"PERFORMANCE | {operation} | Duration: {duration_ms:.2f}ms",
            extra={"details": kwargs}
        )


def create_logger(name: str) -> StructuredLogger:
    """Factory function to create logger instances"""
    return StructuredLogger(name)
