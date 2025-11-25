# Django Attendance System Integration

## Overview
Your FastAPI face recognition service has been modified to support Django attendance system integration with class-based student enrollment and attendance marking.

## New Endpoints Added

### 1. POST /api/enroll
Enroll a student with 3 face images for a specific class.

**Request (multipart/form-data):**
- `student_id` (string): Student identifier (e.g., "STU001")
- `student_name` (string): Student full name
- `class_code` (string): Class identifier (e.g., "CS101", "DB202")
- `image1` (file): First face image
- `image2` (file): Second face image
- `image3` (file): Third face image

**Response:**
```json
{
  "success": true,
  "student_id": "STU001",
  "student_name": "John Doe",
  "class_code": "CS101",
  "face_encoding": "base64_encoded_embedding...",
  "embedding_size": 512,
  "message": "Student John Doe enrolled successfully in CS101"
}
```

**Features:**
- Extracts face embeddings from all 3 images
- Averages the embeddings for robustness
- Stores with class_code to separate different classes
- Returns base64-encoded face encoding

### 2. POST /api/mark-attendance
Mark attendance for a class by detecting faces in classroom photos.

**Request (multipart/form-data):**
- `class_code` (string): Class identifier
- `classroom_image1` (file): First classroom photo
- `classroom_image2` (file): Second classroom photo
- `classroom_image3` (file): Third classroom photo

**Response:**
```json
{
  "success": true,
  "class_code": "CS101",
  "present_students": ["STU001", "STU002", "STU003"],
  "total_present": 3,
  "total_enrolled": 25,
  "detection_details": [
    {
      "image_number": 1,
      "face_number": 1,
      "student_id": "STU001",
      "confidence": 0.856,
      "similarity": 0.856
    }
  ],
  "message": "Attendance marked successfully for CS101"
}
```

**Features:**
- Loads ONLY enrolled students for the specified class_code
- Detects all faces in the 3 classroom images
- Matches detected faces against class-specific student database
- Returns unique list of present students (no duplicates)
- Provides detailed detection information

## Database Changes

### New Methods in DatabaseConnector

#### `save_enrollment_with_class(enrollment_data: Dict) -> bool`
Saves student enrollment with class_code support.

**Expected enrollment_data structure:**
```python
{
    'user_id': 'STU001',
    'student_name': 'John Doe',
    'class_code': 'CS101',
    'average_embedding': np.array([...]),  # 512-dim vector
    'enrollment_date': '2025-11-04T15:00:00',
    'embedding_size': 512,
    'num_images': 3
}
```

#### `get_embeddings_by_class(class_code: str) -> List[Dict]`
Retrieves all face embeddings for a specific class.

**Returns:**
```python
[
    {
        'user_id': 'STU001',
        'student_name': 'John Doe',
        'class_code': 'CS101',
        'embedding': np.array([...])  # 512-dim vector
    },
    ...
]
```

### New Methods in DatabaseManager

#### `save_enrollment_with_class(enrollment_data: Dict) -> bool`
High-level method that saves enrollment and updates cache.

#### `get_embeddings_by_class(class_code: str) -> List[Dict]`
High-level method to get class-specific embeddings.

## Django Model Recommendations

You'll need to update your Django models to include class_code. Here's a suggested structure:

```python
from django.db import models

class FaceEnrollment(models.Model):
    user_id = models.CharField(max_length=50, db_index=True)
    student_name = models.CharField(max_length=200)
    class_code = models.CharField(max_length=50, db_index=True)
    embedding = models.TextField()  # JSON-serialized embedding
    enrollment_date = models.DateTimeField(auto_now_add=True)
    embedding_size = models.IntegerField(default=512)
    num_images = models.IntegerField(default=3)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'face_enrollments'
        unique_together = ['user_id', 'class_code']  # Student can be in multiple classes
        indexes = [
            models.Index(fields=['class_code', 'is_active']),
            models.Index(fields=['user_id']),
        ]

class AttendanceRecord(models.Model):
    class_code = models.CharField(max_length=50, db_index=True)
    student_id = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)
    confidence = models.FloatField()
    
    class Meta:
        db_table = 'attendance_records'
        indexes = [
            models.Index(fields=['class_code', 'timestamp']),
            models.Index(fields=['student_id', 'timestamp']),
        ]
```

## Implementation Steps for Django Integration

### 1. Update database.py
Replace the pseudo-code in `DatabaseConnector.save_enrollment_with_class()` and `DatabaseConnector.get_embeddings_by_class()` with actual Django ORM calls:

```python
def save_enrollment_with_class(self, enrollment_data: Dict) -> bool:
    try:
        from your_app.models import FaceEnrollment
        
        user_id = enrollment_data['user_id']
        student_name = enrollment_data.get('student_name', '')
        class_code = enrollment_data['class_code']
        average_embedding = np.array(enrollment_data['average_embedding'])
        embedding_json = json.dumps(average_embedding.tolist())
        
        FaceEnrollment.objects.update_or_create(
            user_id=user_id,
            class_code=class_code,
            defaults={
                'student_name': student_name,
                'embedding': embedding_json,
                'enrollment_date': enrollment_data['enrollment_date'],
                'embedding_size': enrollment_data['embedding_size'],
                'num_images': enrollment_data.get('num_images', 3),
                'is_active': True
            }
        )
        
        logger.info(f"Saved enrollment for student {user_id} in class {class_code}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save enrollment with class: {e}")
        return False

def get_embeddings_by_class(self, class_code: str) -> List[Dict]:
    try:
        from your_app.models import FaceEnrollment
        
        enrollments = FaceEnrollment.objects.filter(
            class_code=class_code, 
            is_active=True
        )
        
        results = []
        for enrollment in enrollments:
            results.append({
                'user_id': enrollment.user_id,
                'student_name': enrollment.student_name,
                'class_code': enrollment.class_code,
                'embedding': np.array(json.loads(enrollment.embedding))
            })
        
        logger.info(f"Retrieved {len(results)} embeddings for class {class_code}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to retrieve embeddings for class {class_code}: {e}")
        return []
```

### 2. Configure Django Settings
In your FastAPI startup, set the Django settings module:

```python
# In main.py, update the startup_event function:
db_manager = DatabaseManager(
    django_settings_module='your_django_project.settings',
    cache_size=1000
)
```

### 3. Test the Endpoints

**Test Enrollment:**
```bash
curl -X POST "http://localhost:8000/api/enroll" \
  -F "student_id=STU001" \
  -F "student_name=John Doe" \
  -F "class_code=CS101" \
  -F "image1=@student_photo1.jpg" \
  -F "image2=@student_photo2.jpg" \
  -F "image3=@student_photo3.jpg"
```

**Test Attendance:**
```bash
curl -X POST "http://localhost:8000/api/mark-attendance" \
  -F "class_code=CS101" \
  -F "classroom_image1=@classroom1.jpg" \
  -F "classroom_image2=@classroom2.jpg" \
  -F "classroom_image3=@classroom3.jpg"
```

## Key Features

### Class Isolation
- Each class has its own set of enrolled students
- Database class students won't match against Networking class students
- Filtering happens at the database query level for efficiency

### Robust Enrollment
- Uses 3 images per student for better accuracy
- Averages embeddings to handle variations in lighting, angle, etc.
- Normalizes embeddings for consistent matching

### Efficient Attendance Marking
- Processes 3 classroom photos to catch all students
- Uses set to avoid duplicate detections
- Only loads embeddings for the specific class (not all students)
- Returns detailed detection information for debugging

### Caching
- DatabaseManager includes caching for performance
- Cache keys include class_code to separate classes
- Cache automatically updated on enrollment

## Error Handling

Both endpoints include comprehensive error handling:
- Invalid image files
- No face detected in enrollment images
- No enrolled students for a class
- Database connection issues
- Face detection/matching failures

## Next Steps

1. **Update Django Models**: Add the FaceEnrollment model with class_code field
2. **Configure Database**: Set django_settings_module in main.py
3. **Implement Database Methods**: Replace pseudo-code with actual Django ORM
4. **Test Integration**: Use curl or Postman to test both endpoints
5. **Connect to Django**: Call these endpoints from your Django views

## Notes

- The face recognition uses InsightFace with ArcFace (512-dim embeddings)
- Default similarity threshold is 0.4 (cosine similarity)
- All embeddings are L2-normalized
- The service runs on port 8000 by default
