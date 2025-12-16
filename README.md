# 🎯 Identiface - Advanced Face Recognition System v2.1

**Production-ready facial identification system with multi-model detection and threshold-based matching**

## ✨ Key Features

### 🔍 Face Detection
- Multi-model ensemble (SCRFD, YOLOv8, RetinaFace)
- Multi-face detection in single images
- Quality assessment (blur, brightness, contrast)
- Long-distance and small face detection
- Pose-aware face capture

### 👤 Face Recognition  
- ArcFace embeddings (512-dimensional vectors)
- Threshold-based matching (reduce false positives)
- Cosine distance metric
- Confidence scoring
- Top-K identification

### 📚 Student Management
- Multi-pose enrollment (front, left, right, down)
- Automatic face embedding storage
- Per-class student galleries
- Real-time face verification

### 📊 Attendance System
- Multi-face recognition in single photo
- Confidence-based filtering
- Automatic student marking
- Attendance session tracking
- Historical records

### 🌈 Modern UI
- Indigo color theme (changed from green)
- Responsive design
- Real-time feedback messages
- Progress indicators
- Mobile-friendly interface

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install all dependencies (one command)
pip3 install fastapi uvicorn pydantic python-multipart opencv-python numpy scipy Pillow django psycopg2-binary requests python-dotenv

# 2. Start the backend API (Terminal 1)
cd /home/muthoni/identiface_api && python3 main.py

# 3. Start the frontend Django server (Terminal 2)
cd /home/muthoni/identiface && python3 manage.py runserver 0.0.0.0:8001
```

**Access:**
- Frontend: http://localhost:8001
- Backend Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 🛠️ Full Installation & Setup

### Detection (Multi-Model Ensemble)
- **SCRFD** (InsightFace) - Primary detector
- **YOLOv8-Face** - Long-distance detection optimization
- **RetinaFace** - Small and distant face detection
- **Ensemble NMS** - Automatic detection merging for best accuracy
- **Quality Assessment** - Blur, brightness, contrast evaluation
- **Multi-scale Detection** - Handles faces at any distance

### Face Recognition
- **ArcFace Embeddings** - 512-dimensional vectors (InsightFace)
- **Advanced Normalization** - Alignment, preprocessing, enhancement
- **Multi-metric Matching** - Cosine, Euclidean, Manhattan, Angular
- **Adaptive Thresholds** - Gallery-size aware matching
- **Quality-Weighted Scoring** - Confidence based on face quality
- **Long-Distance Optimization** - Super-resolution, denoising

### Django Integration
- **Django ORM Support** - Seamless database integration
- **Attendance Marking** - Automatic student recognition
- **Student Enrollment** - Multi-pose face capture
- **Caching System** - Redis/in-memory embedding cache
- **Class-based Grouping** - Per-class gallery management

### API Features
- **FastAPI v2** - Modern async REST endpoints
- **Automatic Documentation** - Swagger UI + ReDoc
- **Health Monitoring** - Component status checks
- **Error Handling** - Detailed error messages
- **CORS Support** - Cross-origin requests enabled

## 📋 System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 10GB disk space (for models)
- CPU: 4 cores

### Recommended (Optimal Performance)
- Python 3.10+
- 16GB+ RAM
- 20GB SSD
- NVIDIA GPU with 6GB+ VRAM (optional but recommended)
- CPU: Intel i7/i9 or AMD Ryzen 5+

## 🛠️ Installation

### Quick Start (3 steps)

```bash
# 1. Navigate to project
cd /home/muthoni/identiface_api

# 2. Create virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

### Detailed Installation

See `INSTALLATION_GUIDE.md` for:
- Step-by-step setup
- GPU configuration
- Model downloads
- Troubleshooting
- Testing procedures

### Required Modules

See `MODULES_TO_INSTALL.md` for:
- Critical modules (InsightFace, OpenCV, FastAPI)
- Optional advanced modules (YOLOv8, RetinaFace, PyTorch)
- One-line installation
- Verification steps

## 🏃 Quick Start

### Start the API Server

```bash
# Activate virtual environment
source venv/bin/activate

# Run the server
python3 main.py
```

The API will be available at: **http://localhost:8000**

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API Info**: http://localhost:8000/

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "components": {
    "detector": true,
    "extractor": true,
    "matcher": true,
    "normalizer": true,
    "long_distance_optimizer": true,
    "database": true
  }
}
```

## 📚 API Endpoints (v2.0)

### 1. Health Check
**GET** `/health`

Check API and component health status.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "detector": true,
    "extractor": true,
    "matcher": true,
    "normalizer": true,
    "long_distance_optimizer": true,
    "database": true
  }
}
```

---

### 2. Detect Faces
**POST** `/detect`

Detect faces in an image using multi-model ensemble (SCRFD + YOLOv8 + RetinaFace).

**Request:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg" \
  -H "Accept: application/json"
```

**Response:**
```json
{
  "detections": [
    {
      "bbox": [100, 150, 200, 250],
      "confidence": 0.99,
      "landmarks": [[110, 160], [130, 165], ...],
      "quality": 0.95,
      "blur_score": 0.05
    }
  ],
  "num_faces": 1,
  "models_used": ["SCRFD", "YOLOV8", "RETINAFACE"]
}
```

---

### 3. Extract Features
**POST** `/extract-features`

Extract 512-dimensional ArcFace embeddings from detected faces.

**Request:**
```bash
curl -X POST "http://localhost:8000/extract-features" \
  -F "file=@image.jpg" \
  -H "Accept: application/json"
```

**Response:**
```json
{
  "embeddings": [[0.123, 0.456, ..., -0.789]],
  "bboxes": [[100, 150, 200, 250]],
  "qualities": [0.95],
  "embedding_dim": 512,
  "model": "ARCFACE"
}
```

---

### 4. Verify Faces (1:1)
**POST** `/verify`

Verify if two faces belong to the same person (1:1 matching).

**Request:**
```bash
curl -X POST "http://localhost:8000/verify" \
  -F "file1=@image1.jpg" \
  -F "file2=@image2.jpg" \
  -H "Accept: application/json"
```

**Response:**
```json
{
  "is_match": true,
  "similarity": 0.85,
  "distance": 0.42,
  "confidence": 0.95,
  "threshold_used": 0.6
}
```

---

### 5. Identify Face (1:N)
**POST** `/identify`

Identify a face against all enrolled students (1:N matching).

**Request:**
```bash
curl -X POST "http://localhost:8000/identify" \
  -F "file=@image.jpg" \
  -H "Accept: application/json"
```

**Response:**
```json
{
  "matches": [
    {
      "student_id": "S001",
      "student_name": "John Doe",
      "similarity": 0.89,
      "distance": 0.38,
      "confidence": 0.98
    },
    {
      "student_id": "S002",
      "student_name": "Jane Smith",
      "similarity": 0.72,
      "distance": 0.65,
      "confidence": 0.85
    }
  ],
  "best_match": {
    "student_id": "S001",
    "student_name": "John Doe",
    "similarity": 0.89
  },
  "found": true
}
```

---

### 6. Enroll Student
**POST** `/enroll-student`

Enroll a new student with facial embeddings.

**Request:**
```bash
curl -X POST "http://localhost:8000/enroll-student" \
  -F "file=@image.jpg" \
  -F "student_id=S001" \
  -F "student_name=John Doe" \
  -F "class_code=CLASS101" \
  -H "Accept: application/json"
```

**Response:**
```json
{
  "success": true,
  "student_id": "S001",
  "student_name": "John Doe",
  "class_code": "CLASS101",
  "message": "Student enrolled successfully",
  "embeddings_saved": 1,
  "avg_quality": 0.94
}
```

---

### 7. Mark Attendance
**POST** `/attendance`

Mark student attendance using facial recognition.

**Request:**
```bash
curl -X POST "http://localhost:8000/attendance" \
  -F "file=@image.jpg" \
  -F "class_code=CLASS101" \
  -H "Accept: application/json"
```

**Response:**
```json
{
  "success": true,
  "student_id": "S001",
  "student_name": "John Doe",
  "class_code": "CLASS101",
  "timestamp": "2024-01-15T10:30:00Z",
  "confidence": 0.95,
  "message": "Attendance marked successfully"
}
```

---

### 8. API Info
**GET** `/`

Get API version and capabilities information.

**Response:**
```json
{
  "name": "Facial Recognition API v2.0",
  "version": "2.0.0",
  "description": "Advanced multi-model facial recognition system",
  "capabilities": {
    "detection_models": ["SCRFD (Primary)", "YOLOv8", "RetinaFace"],
    "embedding_models": ["ArcFace (Primary)", "FaceNet", "VGGFace2"],
    "distance_metrics": ["Cosine", "Euclidean", "Manhattan", "Angular"],
    "features": ["multi_model_ensemble", "quality_assessment", "long_distance_optimization", "adaptive_thresholds"]
  }
}
```

## 🔧 Module Overview (v2.0 - Advanced)

### 1. `detection_advanced.py`
**Multi-Model Ensemble Face Detection**
- **Primary Model**: SCRFD (InsightFace) - 500x500 resolution
- **Secondary Models**: YOLOv8-Face, RetinaFace
- **Ensemble NMS** - Automatic detection merging for best accuracy
- **Quality Assessment** - Blur detection, brightness, contrast evaluation
- **Facial Landmarks** - 5 or 106 point facial landmarks
- **Multi-scale Detection** - Handles faces from 1m to 50m distance
- **Classes**: 
  - `FaceDetectorAdvanced` - Main detection engine
  - Methods: `detect_faces()`, `_merge_detections()`, `_assess_quality()`

### 2. `feature_extraction_advanced.py`
**Advanced Multi-Model Embedding Extraction**
- **Primary Model**: ArcFace (512-dimensional embeddings)
- **Optional Models**: FaceNet, VGGFace2
- **Ensemble Mode** - Combine multiple embedding models for robustness
- **Batch Processing** - Efficient processing of multiple faces
- **L2 Normalization** - Normalized embedding vectors
- **Classes**:
  - `FeatureExtractorAdvanced` - Embedding extraction engine
  - Methods: `extract_embedding()`, `extract_embeddings_batch()`, `_extract_arcface()`

### 3. `matching_advanced.py`
**Advanced Face Matching with Adaptive Thresholds**
- **Distance Metrics**: Cosine (default), Euclidean, Manhattan, Angular
- **Adaptive Thresholding** - Gallery-size aware thresholds
- **Quality-Weighted Scoring** - Confidence based on face quality
- **1:1 Verification** - Compare two faces
- **1:N Identification** - Match against gallery of enrolled students
- **Top-K Matching** - Return top K matches with scores
- **Classes**:
  - `FaceMatcherAdvanced` - Matching engine
  - Methods: `verify()`, `identify()`, `compute_distance()`, `batch_identify()`

### 4. `normalization_advanced.py`
**Advanced Face Preprocessing and Alignment**
- **2D/3D Face Alignment** - Using landmarks
- **CLAHE Enhancement** - Adaptive histogram equalization
- **Gamma Correction** - Illumination normalization
- **Deblurring** - Image sharpening
- **Color Space Normalization** - RGB standardization
- **Quality Assessment** - Face suitability scoring
- **Classes**:
  - `FaceNormalizerAdvanced` - Preprocessing engine
  - Methods: `normalize_complete()`, `align_face_advanced()`, `assess_quality()`

### 5. `long_distance_optimizer.py`
**Long-Distance and Small-Face Recognition Optimization**
- **Multi-scale Detection** - Detect faces at various distances
- **Denoising** - Image noise reduction for small faces
- **Super-Resolution** - Upscaling small face regions
- **Distance Estimation** - Automatic threshold adjustment
- **Multi-scale Preprocessing** - Resolution adaptation
- **Classes**:
  - `LongDistanceOptimizer` - Long-distance recognition engine
  - Methods: `preprocess_for_long_distance()`, `multi_scale_detection()`, `adjust_threshold_by_distance()`

### 6. `main.py`
**FastAPI v2.0 Server with All Advanced Modules**
- **8 RESTful Endpoints** - Health, Detection, Features, Verify, Identify, Enroll, Attendance, Info
- **Async Processing** - FastAPI async/await support
- **Automatic Documentation** - Swagger UI + ReDoc at /docs and /redoc
- **Global Components** - Detector, Extractor, Matcher, Normalizer, LongDistanceOptimizer
- **Error Handling** - Comprehensive error messages with HTTP status codes
- **CORS Enabled** - Cross-origin requests support

### 7. `database.py`
**Django Integration and Embedding Management**
- **Django ORM Support** - Integration with Django models
- **Embedding Cache** - In-memory and database-backed caching
- **Student Management** - Enrollment and retrieval
- **Attendance Logging** - Track attendance records
- **Methods**:
  - `mark_attendance(student_id, class_code)` - Mark attendance with timestamp
  - `get_all_embeddings_for_matching()` - Retrieve embeddings for 1:N matching
  - `save_enrollment(student_id, embedding, class_code)`
  - `get_embeddings_by_class(class_code)`

### 8. `enrollment.py`
**Student Enrollment with Pose-Based Capture**
- **Multi-pose Enrollment** - Capture faces at different angles
- **Quality Validation** - Per-pose quality checking
- **Pose Estimation** - Front, left, right, down pose detection
- **Progress Tracking** - Enrollment session management
- **Average Embedding** - Combine multiple embeddings per student
- **Updated to use advanced modules** - Uses FaceDetectorAdvanced, FeatureExtractorAdvanced, etc.

### 9. `pose_estimation.py`
**Pose and Angle Estimation**
- **Yaw, Pitch, Roll** - 3D head pose angles
- **Pose Classification** - Categorical pose labels
- **Real-time Feedback** - Guidance for enrollment
- **Validation** - Pose acceptance/rejection criteria

### 10. `video_capture.py`
**Real-time Video Frame Capture and Quality Assessment**
- **Quality Monitoring** - Frame-by-frame quality evaluation
- **Obstruction Detection** - Detect glasses, masks, occlusions
- **Optimal Frame Selection** - Auto-select best frames
- **Real-time Feedback** - User guidance system

## 🎯 Advanced Features (v2.0)

### Multi-Model Ensemble Detection
Combines SCRFD, YOLOv8, and RetinaFace for maximum accuracy:
```python
# Automatically uses all 3 models and merges results
detections = detector.detect_faces(image, detector_type="ENSEMBLE")
```

### Adaptive Thresholding
Matching thresholds automatically adjust based on gallery size:
```python
# Threshold automatically adjusted for gallery of 100 students
match = matcher.identify(embedding, student_id="S001", gallery_size=100)
```

### Quality-Aware Matching
Face quality scores influence final confidence:
```python
# Confidence includes both similarity AND face quality
confidence = similarity * face_quality
```

### Long-Distance Optimization
Handles faces from 1m to 50m distance:
```python
# Automatically applies denoising, super-resolution for small faces
optimized = long_distance_optimizer.preprocess_for_long_distance(image)
```

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

## ⚙️ Configuration (v2.0)

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=False
API_WORKERS=4

# Model Settings
DETECTION_MODEL=ENSEMBLE  # SCRFD, YOLOV8, RETINAFACE, ENSEMBLE
EMBEDDING_MODEL=ARCFACE   # ARCFACE, FACENET, VGGFACE2
DISTANCE_METRIC=COSINE    # COSINE, EUCLIDEAN, MANHATTAN, ANGULAR

# Thresholds
VERIFICATION_THRESHOLD=0.6       # 1:1 matching
IDENTIFICATION_THRESHOLD=0.6     # 1:N matching
MIN_FACE_SIZE=40                 # pixels
MIN_CONFIDENCE=0.5               # detection confidence

# GPU Settings
USE_GPU=True
GPU_DEVICE_ID=0

# Database (optional)
DATABASE_URL=postgresql://user:password@localhost/facedb
REDIS_URL=redis://localhost:6379/0
```

### GPU Support (NVIDIA CUDA)

**Enable GPU acceleration:**

```bash
# 1. Install CUDA 11.8+ (if not already installed)
# Visit: https://developer.nvidia.com/cuda-downloads

# 2. Install GPU-accelerated packages
pip uninstall onnxruntime
pip install onnxruntime-gpu==1.16.3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Verify GPU setup
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Detector Configuration

In `main.py`, modify detector initialization:

```python
# Multi-model ensemble (recommended)
detector = FaceDetectorAdvanced(
    detector_type="ENSEMBLE",
    min_confidence=0.5,
    device="cuda:0",  # GPU device or "cpu"
    enable_quality=True
)

# Single model (faster)
detector = FaceDetectorAdvanced(
    detector_type="SCRFD",
    min_confidence=0.6,
    device="cpu"
)
```

### Matcher Configuration

Adjust matching thresholds in `main.py`:

```python
# Adaptive thresholding
matcher = FaceMatcherAdvanced(
    distance_metric="COSINE",
    default_threshold=0.6,
    quality_weight=0.3,  # How much quality affects confidence
    gallery_size_adaptive=True
)
```

### Long-Distance Optimization

Enable for distant/small face recognition:

```python
# In main.py
long_distance_optimizer = LongDistanceOptimizer(
    enable_denoising=True,
    enable_super_resolution=False,  # Requires OpenCV contrib
    distance_thresholds={
        "close": 0.65,      # < 2m
        "medium": 0.55,     # 2m - 5m
        "far": 0.45         # > 5m
    }
)
```

## 📊 Performance Tuning

### CPU Optimization
- **Batch Processing**: Process multiple faces together for efficiency
- **Model Selection**: SCRFD is fastest, RetinaFace is most accurate
- **Threading**: Use async endpoints for concurrent requests

### GPU Optimization
- **Batch Size**: Increase batch size for better GPU utilization
- **CUDA Streams**: Multiple streams for parallel processing
- **Memory**: Monitor GPU memory with `nvidia-smi`

### Database Optimization
- **Embedding Cache**: Keep in-memory cache of frequently used embeddings
- **Indexing**: Add database indexes on student_id and class_code
- **Batch Operations**: Bulk insert/update for multiple students

### Typical Performance

**CPU Performance (Intel i7-8700K):**
- Face Detection (SCRFD): 50-100ms per image
- Feature Extraction (ArcFace): 30-50ms per face
- Matching (1:1 verification): <1ms
- 1:N Identification (1000 students): 10-20ms

**GPU Performance (NVIDIA RTX 3080):**
- Face Detection (SCRFD): 10-20ms per image
- Feature Extraction (ArcFace): 5-10ms per face
- Batch Identification (1000 students): 5-10ms

## 🔒 Security & Deployment

### Production Deployment Checklist

- [ ] Enable HTTPS with valid SSL certificate
- [ ] Add authentication/authorization (JWT, OAuth2)
- [ ] Implement rate limiting (FastAPI SlowAPI)
- [ ] Enable CORS appropriately (`allow_origins`)
- [ ] Add request logging for audit trail
- [ ] Encrypt embeddings in database
- [ ] Use environment variables for secrets
- [ ] Set up monitoring and alerts
- [ ] Regular security updates to dependencies

### Authentication Example

```python
from fastapi.security import HTTPBearer, HTTPAuthCredential

security = HTTPBearer()

@app.post("/identify")
async def identify(
    file: UploadFile,
    credentials: HTTPAuthCredential = Depends(security)
):
    # Verify token before processing
    verify_token(credentials.credentials)
    # ... rest of code
```

### Rate Limiting

```bash
pip install slowapi

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/identify")
@limiter.limit("10/minute")
async def identify(request: Request, ...):
    ...
```

## 🐛 Troubleshooting

### Common Issues & Solutions

**1. Models Not Found / Downloading**
```bash
# Check model cache directory
ls -la ~/.insightface/models/

# Manually set model path
export INSIGHTFACE_HOME=/path/to/models

# Clear cache and re-download
rm -rf ~/.insightface/models/*
```

**2. ONNX Runtime Errors**
```bash
# Verify ONNX installation
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Reinstall ONNX
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime==1.16.3
```

**3. CUDA/GPU Issues**
```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
nvidia-smi  # Check GPU status

# Reinstall PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**4. OpenCV Errors (Linux)**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6

# Or use opencv-python-headless
pip uninstall opencv-python
pip install opencv-python-headless
```

**5. Out of Memory Errors**
```bash
# Reduce batch size in normalization_advanced.py
batch_size = 1  # Instead of 8 or 16

# Reduce embedding cache size in database.py
max_cache_size = 500  # Instead of 10000

# Use CPU instead of GPU
device = "cpu"  # In detector initialization
```

**6. Low Accuracy / False Matches**
```python
# Increase thresholds (stricter matching)
matcher = FaceMatcherAdvanced(default_threshold=0.7)

# Enable quality weighting
matcher.quality_weight = 0.5

# Use ensemble detection for better input quality
detector_type = "ENSEMBLE"
```

**7. Slow Performance**
```bash
# Check if GPU is being used
nvidia-smi

# Profile code
pip install py-spy
py-spy record -o profile.svg -- python3 main.py

# Analyze bottlenecks
python3 -m cProfile -s cumtime main.py
```

**8. Database Connection Issues**
```python
# Test Django connection
python manage.py shell
>>> from identiface.face_recognition.database import DatabaseConnector
>>> db = DatabaseConnector()
>>> db.test_connection()
```

### Debug Mode

Enable debug logging:

```python
# In main.py, add at startup
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Monitor component health
@app.on_event("startup")
async def startup_event():
    logger.info("API starting up...")
    logger.info(f"Detection model: {detector.detector_type}")
    logger.info(f"Device: {detector.device}")
```

### Getting Help

- **API Documentation**: http://localhost:8000/docs
- **Issue Tracker**: Check project issues
- **InsightFace Docs**: https://github.com/deepinsight/insightface
- **FastAPI Docs**: https://fastapi.tiangolo.com/

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
