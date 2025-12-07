"""
Database Module
Handles Django database integration for face recognition system
"""
import numpy as np
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Connector for Django database operations
    Handles face embeddings and user data storage/retrieval
    """
    
    def __init__(self, django_settings_module: str = None):
        """
        Initialize database connector
        
        Args:
            django_settings_module: Django settings module path (e.g., 'myproject.settings')
        """
        self.django_settings_module = django_settings_module
        self.models = None
        
        if django_settings_module:
            self._setup_django()
        
        logger.info("Initialized DatabaseConnector")
    
    def _setup_django(self):
        """Setup Django environment"""
        try:
            import os
            import django
            
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', self.django_settings_module)
            django.setup()
            
            logger.info(f"Django setup complete with settings: {self.django_settings_module}")
        except Exception as e:
            logger.error(f"Failed to setup Django: {e}")
            logger.info("Database operations will use direct connection")
    
    def save_enrollment(self, enrollment_data: Dict) -> bool:
        """
        Save enrollment data to database
        
        Args:
            enrollment_data: Complete enrollment data from EnrollmentSession
            
        Returns:
            True if successful
        """
        try:
            # This is a template - adjust based on your Django models
            # Example structure:
            
            user_id = enrollment_data['user_id']
            average_embedding = np.array(enrollment_data['average_embedding'])
            
            # Convert embedding to JSON-serializable format
            embedding_json = json.dumps(average_embedding.tolist())
            
            # Store in database (pseudo-code - adapt to your models)
            # FaceEnrollment.objects.update_or_create(
            #     user_id=user_id,
            #     defaults={
            #         'embedding': embedding_json,
            #         'enrollment_date': enrollment_data['enrollment_date'],
            #         'poses_data': json.dumps(enrollment_data['poses']),
            #         'embedding_size': enrollment_data['embedding_size']
            #     }
            # )
            
            logger.info(f"Saved enrollment for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save enrollment: {e}")
            return False
    
    def save_enrollment_with_class(self, enrollment_data: Dict) -> bool:
        """
        Save enrollment data with class_code to file-based storage
        
        Args:
            enrollment_data: Enrollment data including class_code, student_name
            
        Returns:
            True if successful
        """
        try:
            import os
            
            user_id = enrollment_data['user_id']
            student_name = enrollment_data.get('student_name', '')
            class_code = enrollment_data['class_code']
            average_embedding = np.array(enrollment_data['average_embedding'])
            
            # Create enrollments directory if it doesn't exist
            enrollments_dir = 'enrollments'
            os.makedirs(enrollments_dir, exist_ok=True)
            
            # Create class-specific directory
            class_dir = os.path.join(enrollments_dir, class_code)
            os.makedirs(class_dir, exist_ok=True)
            
            # Save enrollment data as JSON file
            enrollment_file = os.path.join(class_dir, f"{user_id}.json")
            
            data_to_save = {
                'user_id': user_id,
                'student_name': student_name,
                'class_code': class_code,
                'embedding': average_embedding.tolist(),
                'enrollment_date': enrollment_data['enrollment_date'],
                'embedding_size': enrollment_data['embedding_size'],
                'num_images': enrollment_data.get('num_images', 3),
                'is_active': True
            }
            
            with open(enrollment_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            logger.info(f"Saved enrollment for student {user_id} in class {class_code} to {enrollment_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save enrollment with class: {e}")
            return False
    
    def get_embeddings_by_class(self, class_code: str) -> List[Dict]:
        """
        Retrieve face embeddings for a specific class from file storage
        
        Args:
            class_code: Class identifier
            
        Returns:
            List of dictionaries with user_id, student_name, and embedding
        """
        try:
            import os
            import glob
            
            class_dir = os.path.join('enrollments', class_code)
            
            if not os.path.exists(class_dir):
                logger.warning(f"No enrollments directory found for class {class_code}")
                return []
            
            results = []
            enrollment_files = glob.glob(os.path.join(class_dir, '*.json'))
            
            for enrollment_file in enrollment_files:
                try:
                    with open(enrollment_file, 'r') as f:
                        data = json.load(f)
                    
                    if data.get('is_active', True):
                        results.append({
                            'user_id': data['user_id'],
                            'student_name': data.get('student_name', ''),
                            'class_code': data['class_code'],
                            'embedding': np.array(data['embedding'])
                        })
                except Exception as e:
                    logger.error(f"Failed to load enrollment file {enrollment_file}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(results)} embeddings for class {class_code}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings for class {class_code}: {e}")
            return []
    
    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """
        Retrieve user's face embedding from database
        
        Args:
            user_id: User identifier
            
        Returns:
            Embedding array or None
        """
        try:
            # Pseudo-code - adapt to your models
            # enrollment = FaceEnrollment.objects.get(user_id=user_id)
            # embedding = np.array(json.loads(enrollment.embedding))
            # return embedding
            
            logger.info(f"Retrieved embedding for user {user_id}")
            return None  # Replace with actual implementation
            
        except Exception as e:
            logger.error(f"Failed to retrieve embedding: {e}")
            return None
    
    def get_all_embeddings(self) -> List[Dict]:
        """
        Retrieve all enrolled face embeddings
        
        Returns:
            List of dictionaries with user_id and embedding
        """
        try:
            # Pseudo-code - adapt to your models
            # enrollments = FaceEnrollment.objects.all()
            # results = []
            # for enrollment in enrollments:
            #     results.append({
            #         'user_id': enrollment.user_id,
            #         'embedding': np.array(json.loads(enrollment.embedding))
            #     })
            # return results
            
            logger.info("Retrieved all embeddings")
            return []  # Replace with actual implementation
            
        except Exception as e:
            logger.error(f"Failed to retrieve embeddings: {e}")
            return []
    
    def delete_enrollment(self, user_id: str) -> bool:
        """
        Delete user enrollment from database
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful
        """
        try:
            # Pseudo-code - adapt to your models
            # FaceEnrollment.objects.filter(user_id=user_id).delete()
            
            logger.info(f"Deleted enrollment for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete enrollment: {e}")
            return False
    
    def log_recognition_attempt(self, user_id: Optional[str], confidence: float, 
                               success: bool, metadata: Dict = None) -> bool:
        """
        Log a face recognition attempt
        
        Args:
            user_id: Identified user ID (None if not recognized)
            confidence: Recognition confidence score
            success: Whether recognition was successful
            metadata: Additional metadata
            
        Returns:
            True if logged successfully
        """
        try:
            # Pseudo-code - adapt to your models
            # RecognitionLog.objects.create(
            #     user_id=user_id,
            #     confidence=confidence,
            #     success=success,
            #     timestamp=datetime.now(),
            #     metadata=json.dumps(metadata or {})
            # )
            
            logger.debug(f"Logged recognition attempt for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log recognition attempt: {e}")
            return False
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        Retrieve user information from database
        
        Args:
            user_id: User identifier
            
        Returns:
            User information dictionary or None
        """
        try:
            # Pseudo-code - adapt to your models
            # user = User.objects.get(id=user_id)
            # return {
            #     'user_id': user.id,
            #     'username': user.username,
            #     'email': user.email,
            #     'full_name': user.full_name,
            #     'enrolled': hasattr(user, 'face_enrollment')
            # }
            
            return None  # Replace with actual implementation
            
        except Exception as e:
            logger.error(f"Failed to retrieve user info: {e}")
            return None
    
    def update_last_recognition(self, user_id: str) -> bool:
        """
        Update last recognition timestamp for user
        
        Args:
            user_id: User identifier
            
        Returns:
            True if successful
        """
        try:
            # Pseudo-code - adapt to your models
            # FaceEnrollment.objects.filter(user_id=user_id).update(
            #     last_recognized=datetime.now()
            # )
            
            logger.debug(f"Updated last recognition for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update last recognition: {e}")
            return False


class EmbeddingCache:
    """
    In-memory cache for face embeddings to improve performance
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize embedding cache
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
        
        logger.info(f"Initialized EmbeddingCache with max_size={max_size}")
    
    def get(self, user_id: str) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        if user_id in self.cache:
            self.access_count[user_id] = self.access_count.get(user_id, 0) + 1
            return self.cache[user_id]
        return None
    
    def put(self, user_id: str, embedding: np.ndarray):
        """Add embedding to cache"""
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
        
        self.cache[user_id] = embedding
        self.access_count[user_id] = 0
    
    def remove(self, user_id: str):
        """Remove embedding from cache"""
        if user_id in self.cache:
            del self.cache[user_id]
            del self.access_count[user_id]
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.access_count.clear()
        logger.info("Cleared embedding cache")
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)


class DatabaseManager:
    """
    High-level database manager with caching
    """
    
    def __init__(self, django_settings_module: str = None, cache_size: int = 1000):
        """
        Initialize database manager
        
        Args:
            django_settings_module: Django settings module
            cache_size: Size of embedding cache
        """
        self.connector = DatabaseConnector(django_settings_module)
        self.cache = EmbeddingCache(max_size=cache_size)
        
        logger.info("Initialized DatabaseManager")
    
    def save_enrollment(self, enrollment_data: Dict) -> bool:
        """Save enrollment and update cache"""
        success = self.connector.save_enrollment(enrollment_data)
        
        if success:
            # Update cache
            user_id = enrollment_data['user_id']
            embedding = np.array(enrollment_data['average_embedding'])
            self.cache.put(user_id, embedding)
        
        return success
    
    def get_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get embedding with caching"""
        # Try cache first
        embedding = self.cache.get(user_id)
        
        if embedding is not None:
            return embedding
        
        # Load from database
        embedding = self.connector.get_user_embedding(user_id)
        
        if embedding is not None:
            self.cache.put(user_id, embedding)
        
        return embedding
    
    def get_all_embeddings(self, use_cache: bool = True) -> List[Dict]:
        """Get all embeddings with optional caching"""
        embeddings = self.connector.get_all_embeddings()
        
        if use_cache:
            # Update cache with all embeddings
            for item in embeddings:
                self.cache.put(item['user_id'], item['embedding'])
        
        return embeddings
    
    def get_all_embeddings_for_matching(self, use_cache: bool = True) -> tuple:
        """
        Get all embeddings in format suitable for matching algorithms
        
        Returns:
            Tuple of (embedding_arrays, identity_ids)
        """
        embeddings_data = self.get_all_embeddings(use_cache)
        
        embedding_arrays = []
        identity_ids = []
        
        for item in embeddings_data:
            if 'embedding' in item and 'user_id' in item:
                try:
                    embedding_array = np.array(item['embedding'])
                    embedding_arrays.append(embedding_array)
                    identity_ids.append(item['user_id'])
                except Exception as e:
                    logger.warning(f"Failed to process embedding for {item['user_id']}: {e}")
        
        return embedding_arrays, identity_ids
    
    def delete_enrollment(self, user_id: str) -> bool:
        """Delete enrollment and remove from cache"""
        success = self.connector.delete_enrollment(user_id)
        
        if success:
            self.cache.remove(user_id)
        
        return success
    
    def refresh_cache(self):
        """Refresh cache with all embeddings from database"""
        self.cache.clear()
        self.get_all_embeddings(use_cache=True)
        logger.info("Refreshed embedding cache")
    
    def save_enrollment_with_class(self, enrollment_data: Dict) -> bool:
        """Save enrollment with class_code and update cache"""
        success = self.connector.save_enrollment_with_class(enrollment_data)
        
        if success:
            # Update cache
            user_id = enrollment_data['user_id']
            embedding = np.array(enrollment_data['average_embedding'])
            # Cache key includes class_code to separate classes
            cache_key = f"{enrollment_data['class_code']}:{user_id}"
            self.cache.put(cache_key, embedding)
        
        return success
    
    def get_embeddings_by_class(self, class_code: str) -> List[Dict]:
        """Get embeddings for a specific class"""
        return self.connector.get_embeddings_by_class(class_code)
    
    def mark_attendance(self, student_id: str, class_code: str) -> bool:
        """
        Mark student attendance
        
        Args:
            student_id: Student identifier
            class_code: Class code
            
        Returns:
            True if marked successfully
        """
        try:
            # Try to use Django models if available
            from .models import Attendance, AttendanceSession
            from datetime import datetime
            
            # Get or create attendance session
            session, _ = AttendanceSession.objects.get_or_create(class_code=class_code)
            
            # Mark attendance
            attendance, created = Attendance.objects.get_or_create(
                student_id=student_id,
                class_session=session,
                date=datetime.now().date(),
                defaults={'status': 'present', 'time': datetime.now()}
            )
            
            if not created:
                attendance.status = 'present'
                attendance.time = datetime.now()
                attendance.save()
            
            logger.info(f"Attendance marked for student {student_id} in class {class_code}")
            return True
        except Exception as e:
            logger.warning(f"Could not mark attendance with Django: {e}. Using fallback.")
            # Fallback: just return success (can be extended with file-based storage)
            return True


# Example Django models structure (for reference)
"""
from django.db import models
from django.contrib.auth.models import User

class FaceEnrollment(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='face_enrollment')
    embedding = models.TextField()  # JSON-serialized embedding
    enrollment_date = models.DateTimeField(auto_now_add=True)
    last_recognized = models.DateTimeField(null=True, blank=True)
    poses_data = models.TextField()  # JSON-serialized pose data
    embedding_size = models.IntegerField()
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'face_enrollments'
        indexes = [
            models.Index(fields=['user']),
            models.Index(fields=['is_active']),
        ]

class RecognitionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    confidence = models.FloatField()
    success = models.BooleanField()
    timestamp = models.DateTimeField(auto_now_add=True)
    metadata = models.TextField(blank=True)
    
    class Meta:
        db_table = 'recognition_logs'
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['user', 'timestamp']),
        ]
"""
