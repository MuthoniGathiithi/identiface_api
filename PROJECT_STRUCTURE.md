# Face Recognition API - Project Structure

## 📁 Directory Structure

```
face_service/
├── main.py                    # FastAPI application with REST endpoints
├── config.py                  # Configuration management
├── detection.py               # SCRFD face detection module
├── feature_extraction.py      # ArcFace embedding extraction
├── normalization.py           # Face preprocessing and alignment
├── matching.py                # Face comparison and identification
├── pose_estimation.py         # Pose detection (front/left/right/down)
├── video_capture.py           # Video capture with quality checks
├── enrollment.py              # Multi-pose enrollment system
├── database.py                # Django database integration
├── test_api.py                # API testing script
├── requirements.txt           # Python dependencies
├── README.md                  # Complete documentation
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── PROJECT_STRUCTURE.md       # This file
```

## 🔧 Module Dependencies

```
main.py
├── detection.py
├── feature_extraction.py
├── normalization.py
├── matching.py
├── pose_estimation.py
├── enrollment.py
│   ├── detection.py
│   ├── feature_extraction.py
│   ├── normalization.py
│   └── pose_estimation.py
└── database.py
```

## 📦 Core Components

### 1. **detection.py** - Face Detection
- **Technology**: InsightFace SCRFD
- **Key Classes**: `FaceDetector`
- **Features**:
  - Multi-face detection
  - Facial landmark extraction (5 points)
  - Quality assessment (brightness, sharpness, size)
  - Fallback to OpenCV Haar Cascades

### 2. **feature_extraction.py** - Face Recognition
- **Technology**: InsightFace ArcFace
- **Key Classes**: `FeatureExtractor`
- **Features**:
  - 512-dimensional embeddings
  - L2 normalization
  - Batch processing
  - Similarity computation

### 3. **normalization.py** - Preprocessing
- **Key Classes**: `FaceNormalizer`
- **Features**:
  - Face alignment using landmarks
  - Histogram equalization
  - CLAHE enhancement
  - Pixel standardization

### 4. **matching.py** - Face Matching
- **Key Classes**: `FaceMatcher`
- **Features**:
  - 1:1 verification
  - 1:N identification
  - Cosine/Euclidean distance metrics
  - Face clustering and deduplication

### 5. **pose_estimation.py** - Pose Detection
- **Key Classes**: `PoseEstimator`, `FacePose` (Enum)
- **Features**:
  - Yaw, pitch, roll angle estimation
  - Pose classification (front, left, right, down)
  - Real-time validation
  - User guidance messages

### 6. **video_capture.py** - Video Processing
- **Key Classes**: `VideoCapture`
- **Features**:
  - Threaded frame capture
  - Quality assessment
  - Obstruction detection
  - Best frame selection

### 7. **enrollment.py** - Enrollment Management
- **Key Classes**: `EnrollmentSession`, `EnrollmentManager`
- **Features**:
  - Multi-pose capture workflow
  - Progress tracking
  - Quality validation per pose
  - Average embedding calculation

### 8. **database.py** - Data Persistence
- **Key Classes**: `DatabaseConnector`, `DatabaseManager`, `EmbeddingCache`
- **Features**:
  - Django ORM integration
  - In-memory caching
  - Recognition logging
  - User management

### 9. **main.py** - REST API
- **Framework**: FastAPI
- **Key Endpoints**:
  - `/detect` - Face detection
  - `/extract-features` - Embedding extraction
  - `/verify` - 1:1 face verification
  - `/identify` - 1:N face identification
  - `/enroll/*` - Enrollment workflow
  - `/user/*` - User management

## 🔄 Data Flow

### Enrollment Flow
```
1. POST /enroll/start
   └─> EnrollmentManager.start_session()
       └─> Create EnrollmentSession

2. POST /enroll/process-frame/{user_id} (repeat for each pose)
   └─> EnrollmentSession.process_frame()
       ├─> FaceDetector.detect_faces()
       ├─> FaceDetector.assess_face_quality()
       ├─> PoseEstimator.validate_pose_for_enrollment()
       └─> If ready: EnrollmentSession.capture_pose()
           ├─> FaceNormalizer.preprocess_for_model()
           └─> FeatureExtractor.extract_embedding()

3. POST /enroll/complete/{user_id}
   └─> EnrollmentManager.end_session()
       ├─> Calculate average embedding
       └─> DatabaseManager.save_enrollment()
```

### Identification Flow
```
POST /identify
└─> Read image
    └─> FeatureExtractor.extract_embedding()
        └─> DatabaseManager.get_all_embeddings()
            └─> FaceMatcher.find_best_match()
                ├─> Compare with all enrolled embeddings
                ├─> Log recognition attempt
                └─> Return best match
```

## 🎯 Key Features

### Quality Assurance
- ✓ Brightness validation (40-220 range)
- ✓ Sharpness detection (Laplacian variance)
- ✓ Face size requirements (min 80x80)
- ✓ Obstruction detection
- ✓ Pose validation per enrollment step

### Multi-Pose Enrollment
- ✓ Front pose (straight ahead)
- ✓ Left pose (head turned left)
- ✓ Right pose (head turned right)
- ✓ Down pose (head tilted down)

### Performance Optimizations
- ✓ In-memory embedding cache
- ✓ Threaded video capture
- ✓ Batch processing support
- ✓ GPU acceleration support

### Security & Reliability
- ✓ Input validation
- ✓ Error handling
- ✓ Logging system
- ✓ CORS configuration
- ✓ Health check endpoint

## 🚀 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
python main.py

# Run tests
python test_api.py <image_path>

# View API documentation
# Open: http://localhost:8000/docs
```

## 🔌 Django Integration Points

### Required Django Models
```python
# In your Django app's models.py
- FaceEnrollment (user, embedding, poses_data, etc.)
- RecognitionLog (user, confidence, success, timestamp)
```

### Configuration
```python
# In database.py
DatabaseConnector(django_settings_module='your_project.settings')
```

## 📊 API Response Formats

### Detection Response
```json
{
  "success": true,
  "faces_detected": 1,
  "faces": [{
    "box": {"x": 100, "y": 150, "width": 200, "height": 250},
    "confidence": 0.98,
    "keypoints": {...}
  }]
}
```

### Identification Response
```json
{
  "success": true,
  "user_id": "john_doe",
  "confidence": 0.87,
  "similarity": 0.87,
  "distance": 0.13,
  "message": "User identified successfully"
}
```

### Enrollment Progress
```json
{
  "user_id": "john_doe",
  "total_poses": 4,
  "completed_poses": 2,
  "current_pose": "right",
  "is_complete": false,
  "captured_poses": ["front", "left"]
}
```

## 🔧 Configuration Options

### Environment Variables (.env)
- `MODEL_NAME`: buffalo_l, buffalo_s, antelopev2
- `CTX_ID`: -1 (CPU) or 0+ (GPU)
- `DETECTION_CONFIDENCE`: 0.0-1.0
- `MATCHING_THRESHOLD`: 0.3-0.5 (cosine)
- `YAW_THRESHOLD`: degrees for left/right pose
- `PITCH_THRESHOLD`: degrees for down pose

## 📈 Performance Metrics

### Typical Latencies (CPU)
- Face Detection: 50-100ms
- Feature Extraction: 30-50ms
- Face Matching: <1ms
- Full Enrollment: 10-15 seconds (4 poses)

### Accuracy
- Detection: >95% (well-lit conditions)
- Recognition: >99% (LFW benchmark)
- False Accept Rate: <0.1% (threshold 0.4)

## 🐛 Common Issues & Solutions

### Issue: Models not downloading
**Solution**: Check internet connection, manually download from InsightFace repo

### Issue: ONNX Runtime errors
**Solution**: Reinstall onnxruntime, ensure compatible version

### Issue: Poor detection in low light
**Solution**: Improve lighting, adjust DETECTION_CONFIDENCE

### Issue: False matches
**Solution**: Increase MATCHING_THRESHOLD (stricter)

## 📚 Additional Resources

- InsightFace: https://github.com/deepinsight/insightface
- FastAPI Docs: https://fastapi.tiangolo.com/
- SCRFD Paper: https://arxiv.org/abs/2105.04714
- ArcFace Paper: https://arxiv.org/abs/1801.07698
