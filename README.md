# Face Recognition API Service

A production-ready facial identification API using **InsightFace** (SCRFD for detection + ArcFace for recognition) with Django database integration.

## 🚀 Features

- **Advanced Face Detection**: SCRFD (Sample and Computation Redistribution for Face Detection)
- **High-Accuracy Recognition**: ArcFace embeddings (512-dimensional vectors)
- **Multi-Pose Enrollment**: Captures front, left, right, and down poses for robust recognition
- **Quality Assurance**: Automatic checks for lighting, blur, obstructions, and pose validation
- **Real-time Processing**: Optimized for both CPU and GPU inference
- **Django Integration**: Ready-to-connect with Django projects
- **RESTful API**: FastAPI-based endpoints with automatic documentation
- **Caching System**: In-memory embedding cache for fast lookups

## 📋 System Requirements

- Python 3.8+
- OpenCV compatible system
- 4GB+ RAM (8GB+ recommended)
- Optional: CUDA-compatible GPU for faster processing

## 🛠️ Installation

### 1. Clone or Setup Project

```bash
cd /home/muthoni/face_service
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download InsightFace Models

The models will be automatically downloaded on first run. They will be stored in `~/.insightface/models/`.

**Available models:**
- `buffalo_l` - High accuracy (recommended, ~600MB)
- `buffalo_s` - Balanced speed/accuracy (~200MB)
- `antelopev2` - Latest model

## 🏃 Quick Start

### Start the API Server

```bash
python main.py
```

The API will be available at: `http://localhost:8000`

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Endpoints

### Core Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Detect Faces
```bash
POST /detect
Content-Type: multipart/form-data
Body: file=<image_file>
```

#### 3. Extract Features
```bash
POST /extract-features
Content-Type: multipart/form-data
Body: file=<image_file>
```

#### 4. Verify Faces (1:1)
```bash
POST /verify
Content-Type: multipart/form-data
Body: file1=<image1>, file2=<image2>
```

#### 5. Identify Face (1:N)
```bash
POST /identify
Content-Type: multipart/form-data
Body: file=<image_file>
```

### Enrollment Endpoints

#### Start Enrollment
```bash
POST /enroll/start
Content-Type: application/json
Body: {"user_id": "user123"}
```

#### Process Enrollment Frame
```bash
POST /enroll/process-frame/{user_id}
Content-Type: multipart/form-data
Body: file=<image_file>
```

#### Complete Enrollment
```bash
POST /enroll/complete/{user_id}
```

#### Cancel Enrollment
```bash
POST /enroll/cancel/{user_id}
```

#### Get Enrollment Status
```bash
GET /enroll/status/{user_id}
```

### User Management

#### Delete User
```bash
DELETE /user/{user_id}
```

#### Get User Count
```bash
GET /users/count
```

## 🔧 Module Overview

### 1. `detection.py`
- **SCRFD-based face detection**
- Quality assessment (brightness, sharpness, size)
- Facial landmark extraction
- Fallback to OpenCV Haar Cascades

### 2. `feature_extraction.py`
- **ArcFace embedding extraction**
- 512-dimensional face vectors
- L2 normalization
- Batch processing support

### 3. `normalization.py`
- Face alignment using landmarks
- Histogram equalization
- CLAHE enhancement
- Standardization for model input

### 4. `matching.py`
- Cosine similarity / Euclidean distance
- 1:1 verification
- 1:N identification
- Face clustering and deduplication

### 5. `pose_estimation.py`
- Yaw, pitch, roll angle estimation
- Pose classification (front, left, right, down)
- Validation for enrollment
- Real-time feedback

### 6. `video_capture.py`
- Quality-checked frame capture
- Real-time feedback system
- Obstruction detection
- Optimal frame selection

### 7. `enrollment.py`
- Multi-pose enrollment sessions
- Progress tracking
- Quality validation per pose
- Average embedding calculation

### 8. `database.py`
- Django ORM integration
- Embedding cache system
- Recognition logging
- User management

## 🔌 Django Integration

### 1. Add to Django Project

Copy the `face_service` directory to your Django project:

```
your_django_project/
├── manage.py
├── your_app/
└── face_service/  # This directory
```

### 2. Create Django Models

Add to your Django app's `models.py`:

```python
from django.db import models
from django.contrib.auth.models import User

class FaceEnrollment(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    embedding = models.TextField()  # JSON-serialized
    enrollment_date = models.DateTimeField(auto_now_add=True)
    last_recognized = models.DateTimeField(null=True, blank=True)
    poses_data = models.TextField()
    embedding_size = models.IntegerField()
    is_active = models.BooleanField(default=True)

class RecognitionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    confidence = models.FloatField()
    success = models.BooleanField()
    timestamp = models.DateTimeField(auto_now_add=True)
    metadata = models.TextField(blank=True)
```

### 3. Update `database.py`

Modify the `DatabaseConnector` class to use your Django models:

```python
def __init__(self, django_settings_module: str = 'your_project.settings'):
    # ... existing code ...
```

### 4. Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

## 🎯 Usage Examples

### Python Client Example

```python
import requests

# 1. Start enrollment
response = requests.post(
    'http://localhost:8000/enroll/start',
    json={'user_id': 'john_doe'}
)
print(response.json())

# 2. Process frames (repeat for each pose)
with open('photo.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/enroll/process-frame/john_doe',
        files={'file': f}
    )
    print(response.json())

# 3. Complete enrollment
response = requests.post('http://localhost:8000/enroll/complete/john_doe')
print(response.json())

# 4. Identify a face
with open('test_photo.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/identify',
        files={'file': f}
    )
    print(response.json())
```

### cURL Examples

```bash
# Detect faces
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@photo.jpg"

# Verify two faces
curl -X POST "http://localhost:8000/verify" \
  -F "file1=@person1.jpg" \
  -F "file2=@person2.jpg"

# Identify face
curl -X POST "http://localhost:8000/identify" \
  -F "file=@unknown.jpg"
```

## ⚙️ Configuration

### GPU Support

To enable GPU acceleration, install:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu==1.16.3
```

Then update `main.py`:

```python
ctx_id = 0  # GPU device ID (0, 1, 2, ...)
```

### Adjust Thresholds

In `main.py`, modify:

```python
# Detection confidence
detector = FaceDetector(min_confidence=0.5)  # 0.0-1.0

# Matching threshold
matcher = FaceMatcher(threshold=0.4)  # Lower = stricter

# Pose thresholds
pose_estimator = PoseEstimator(
    yaw_threshold=20.0,    # degrees
    pitch_threshold=15.0   # degrees
)
```

## 📊 Performance

### Typical Performance (CPU - Intel i7)
- Face Detection: ~50-100ms per image
- Feature Extraction: ~30-50ms per face
- Matching: <1ms per comparison
- 1:N Identification: ~10ms for 1000 users (with cache)

### GPU Performance (NVIDIA RTX 3080)
- Face Detection: ~10-20ms per image
- Feature Extraction: ~5-10ms per face

## 🔒 Security Considerations

1. **API Authentication**: Add authentication middleware for production
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **CORS**: Configure `allow_origins` appropriately
4. **HTTPS**: Use HTTPS in production
5. **Input Validation**: All inputs are validated
6. **Embedding Storage**: Store embeddings securely in database

## 🐛 Troubleshooting

### Models Not Downloading
```bash
# Manually download models
mkdir -p ~/.insightface/models
# Download from: https://github.com/deepinsight/insightface/tree/master/model_zoo
```

### ONNX Runtime Issues
```bash
# Reinstall onnxruntime
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime==1.16.3
```

### OpenCV Issues
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install libgl1-mesa-glx libglib2.0-0
```

## 📝 License

This project uses InsightFace models which are subject to their respective licenses.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- Documentation is updated

## 📧 Support

For issues and questions, please check:
- API documentation: http://localhost:8000/docs
- InsightFace: https://github.com/deepinsight/insightface
- FastAPI: https://fastapi.tiangolo.com/

## 🎓 References

- **InsightFace**: https://github.com/deepinsight/insightface
- **SCRFD Paper**: https://arxiv.org/abs/2105.04714
- **ArcFace Paper**: https://arxiv.org/abs/1801.07698
- **FastAPI**: https://fastapi.tiangolo.com/
