"""
Face Recognition API Service - UPGRADED
FastAPI-based REST API for facial identification with advanced multi-model support
Features: Long-distance detection, Advanced feature extraction, Optimized matching
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import cv2
import numpy as np
import logging
import io
import base64
from datetime import datetime

# Import upgraded modules
from detection_advanced import FaceDetectorAdvanced, DetectionModel
from feature_extraction_advanced import FeatureExtractorAdvanced, EmbeddingModel
from normalization_advanced import FaceNormalizerAdvanced
from matching_advanced import FaceMatcherAdvanced, DistanceMetric
from long_distance_optimizer import LongDistanceOptimizer
from pose_estimation import PoseEstimator, FacePose
from enrollment import EnrollmentManager
from database import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Face Recognition API",
    description="State-of-the-art facial identification service with multi-model support",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
detector: Optional[FaceDetectorAdvanced] = None
extractor: Optional[FeatureExtractorAdvanced] = None
normalizer: Optional[FaceNormalizerAdvanced] = None
matcher: Optional[FaceMatcherAdvanced] = None
long_distance_optimizer: Optional[LongDistanceOptimizer] = None
pose_estimator: Optional[PoseEstimator] = None
enrollment_manager: Optional[EnrollmentManager] = None
db_manager: Optional[DatabaseManager] = None


# Pydantic models
class EnrollmentRequest(BaseModel):
    user_id: str
    
class VerificationRequest(BaseModel):
    user_id: str
    
class IdentificationResponse(BaseModel):
    success: bool
    user_id: Optional[str] = None
    confidence: float
    similarity: float
    message: str

class StudentEnrollmentRequest(BaseModel):
    student_id: str
    student_name: str
    class_code: str

class AttendanceRequest(BaseModel):
    class_code: str


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup with advanced models"""
    global detector, extractor, normalizer, matcher, long_distance_optimizer
    global pose_estimator, enrollment_manager, db_manager
    
    logger.info("Initializing Advanced Face Recognition Service...")
    
    try:
        ctx_id = -1  # CPU (-1) or GPU (0, 1, 2, ...)
        
        # Initialize detector with ensemble model for best performance
        logger.info("Initializing detector (multi-model ensemble)...")
        detector = FaceDetectorAdvanced(
            model=DetectionModel.ENSEMBLE,  # Ensemble of SCRFD, YOLOv8, RetinaFace
            min_confidence=0.5,
            ctx_id=ctx_id,
            enable_blur_detection=True,
            enable_face_quality=True
        )
        
        # Initialize feature extractor with ArcFace
        logger.info("Initializing feature extractor (ArcFace)...")
        extractor = FeatureExtractorAdvanced(
            model=EmbeddingModel.ARCFACE,
            ctx_id=ctx_id,
            use_ensemble=False,
            normalize_embeddings=True
        )
        
        # Initialize normalizer with advanced preprocessing
        logger.info("Initializing normalizer (advanced preprocessing)...")
        normalizer = FaceNormalizerAdvanced(
            target_size=(224, 224),
            enable_affine_alignment=True,
            enable_3d_alignment=False,
            normalize_color=True
        )
        
        # Initialize matcher with cosine distance
        logger.info("Initializing matcher (advanced multi-metric)...")
        matcher = FaceMatcherAdvanced(
            metric=DistanceMetric.COSINE,
            threshold=0.4,
            use_adaptive_threshold=True,
            enable_quality_weighting=True
        )
        
        # Initialize long-distance optimizer
        logger.info("Initializing long-distance optimizer...")
        long_distance_optimizer = LongDistanceOptimizer(
            enable_super_resolution=False,  # Set to True if needed
            enable_multi_scale=True,
            enable_denoising=True
        )
        
        # Initialize pose estimator
        logger.info("Initializing pose estimator...")
        pose_estimator = PoseEstimator(yaw_threshold=20.0, pitch_threshold=15.0)
        
        # Initialize enrollment manager
        logger.info("Initializing enrollment manager...")
        enrollment_manager = EnrollmentManager()
        
        # Initialize database manager
        logger.info("Initializing database manager...")
        db_manager = DatabaseManager(django_settings_module=None, cache_size=1000)
        
        logger.info("✓ Advanced Face Recognition Service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Face Recognition Service...")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Advanced Face Recognition API",
        "version": "2.0.0",
        "status": "running",
        "capabilities": {
            "detection": "Multi-model ensemble (SCRFD, YOLOv8, RetinaFace)",
            "recognition": "ArcFace embeddings (512-dim)",
            "long_distance": "Optimized for long-distance and small faces",
            "features": ["Multi-scale detection", "Blur detection", "Quality assessment", 
                        "Adaptive thresholds", "Distance-aware matching"]
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "detector": detector is not None,
            "extractor": extractor is not None,
            "matcher": matcher is not None,
            "normalizer": normalizer is not None,
            "long_distance_optimizer": long_distance_optimizer is not None,
            "database": db_manager is not None
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/detect")
async def detect_faces(file: UploadFile = File(...), enable_quality: bool = True):
    """
    Detect faces in an image with advanced multi-model ensemble
    
    Features:
    - Multi-scale detection
    - Blur detection
    - Quality assessment
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect faces with ensemble
        faces = detector.detect_faces(image)
        
        # Format response
        results = []
        for face in faces:
            face_data = {
                "box": {
                    "x": int(face['box'][0]),
                    "y": int(face['box'][1]),
                    "width": int(face['box'][2]),
                    "height": int(face['box'][3])
                },
                "confidence": float(face['confidence']),
                "model": face.get('model', 'ensemble'),
                "keypoints": {k: [int(v[0]), int(v[1])] for k, v in face.get('keypoints', {}).items()}
            }
            
            if enable_quality and 'quality_score' in face:
                face_data['quality_score'] = float(face['quality_score'])
            
            results.append(face_data)
        
        return {
            "success": True,
            "faces_detected": len(faces),
            "faces": results,
            "image_size": {"width": image.shape[1], "height": image.shape[0]}
        }
        
    except Exception as e:
        logger.error(f"Face detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-features")
async def extract_features(file: UploadFile = File(...)):
    """
    Extract face embeddings (512-dimensional ArcFace vectors)
    
    Features:
    - Advanced preprocessing and alignment
    - Quality assessment
    - Normalization
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect faces
        faces = detector.detect_faces(image)
        
        if not faces:
            return {
                "success": False,
                "message": "No faces detected",
                "embeddings": []
            }
        
        embeddings = []
        for face in faces:
            x, y, w, h = face['box']
            face_crop = image[y:y+h, x:x+w]
            
            # Normalize face
            normalized = normalizer.normalize_complete(face_crop, face.get('keypoints', {}))
            
            # Extract embedding
            embedding = extractor.extract_embedding(normalized)
            
            if embedding is not None:
                embeddings.append({
                    "embedding": embedding.tolist(),
                    "dimension": len(embedding),
                    "confidence": float(face['confidence'])
                })
        
        return {
            "success": len(embeddings) > 0,
            "embeddings_extracted": len(embeddings),
            "embeddings": embeddings
        }
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Verify if two faces belong to the same person (1:1 matching)
    
    Advanced features:
    - Cosine distance metric
    - Adaptive thresholds
    - Quality-weighted confidence
    """
    try:
        # Read images
        contents1 = await file1.read()
        nparr1 = np.frombuffer(contents1, np.uint8)
        image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        
        contents2 = await file2.read()
        nparr2 = np.frombuffer(contents2, np.uint8)
        image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
        
        if image1 is None or image2 is None:
            raise HTTPException(status_code=400, detail="Invalid image file(s)")
        
        # Detect and extract embeddings from both images
        faces1 = detector.detect_faces(image1)
        faces2 = detector.detect_faces(image2)
        
        if not faces1 or not faces2:
            return {
                "success": False,
                "match": False,
                "message": "Could not detect face(s) in one or both images",
                "similarity": 0.0
            }
        
        # Extract embeddings
        face1 = faces1[0]
        face2 = faces2[0]
        
        x1, y1, w1, h1 = face1['box']
        crop1 = image1[y1:y1+h1, x1:x1+w1]
        norm1 = normalizer.normalize_complete(crop1, face1.get('keypoints', {}))
        emb1 = extractor.extract_embedding(norm1)
        
        x2, y2, w2, h2 = face2['box']
        crop2 = image2[y2:y2+h2, x2:x2+w2]
        norm2 = normalizer.normalize_complete(crop2, face2.get('keypoints', {}))
        emb2 = extractor.extract_embedding(norm2)
        
        if emb1 is None or emb2 is None:
            return {
                "success": False,
                "match": False,
                "message": "Could not extract embeddings",
                "similarity": 0.0
            }
        
        # Verify
        result = matcher.verify(emb1, emb2,
                               quality1=face1.get('quality_score', 1.0),
                               quality2=face2.get('quality_score', 1.0))
        
        return {
            "success": True,
            "match": result['match'],
            "similarity": result['similarity'],
            "distance": result['distance'],
            "confidence": result['confidence'],
            "threshold": result['threshold'],
            "metric": result['model']
        }
        
    except Exception as e:
        logger.error(f"Verification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify")
async def identify_face(file: UploadFile = File(...), top_k: int = 5):
    """
    Identify a face from gallery (1:N matching)
    
    Advanced features:
    - Adaptive thresholds
    - Top-K matching
    - Quality-weighted ranking
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect face
        faces = detector.detect_faces(image)
        
        if not faces:
            return {
                "success": False,
                "identified": False,
                "message": "No face detected",
                "matches": []
            }
        
        # Extract embedding
        face = faces[0]
        x, y, w, h = face['box']
        crop = image[y:y+h, x:x+w]
        normalized = normalizer.normalize_complete(crop, face.get('keypoints', {}))
        embedding = extractor.extract_embedding(normalized)
        
        if embedding is None:
            return {
                "success": False,
                "identified": False,
                "message": "Could not extract embedding",
                "matches": []
            }
        
        # Get gallery embeddings
        gallery_embeddings, gallery_ids = db_manager.get_all_embeddings_for_matching()
        
        if not gallery_embeddings:
            return {
                "success": True,
                "identified": False,
                "message": "Gallery is empty",
                "matches": []
            }
        
        # Identify
        matches = matcher.identify(embedding, gallery_embeddings, gallery_ids, top_k=top_k)
        
        return {
            "success": True,
            "identified": len(matches) > 0 and matches[0]['passes_threshold'],
            "top_k": top_k,
            "matches": [
                {
                    "identity_id": m['identity_id'],
                    "similarity": m['similarity'],
                    "distance": m['distance'],
                    "confidence": m['confidence'],
                    "matches_threshold": m['passes_threshold']
                } for m in matches
            ]
        }
        
    except Exception as e:
        logger.error(f"Identification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enroll-student")
async def enroll_student(
    file: UploadFile = File(...),
    student_id: str = Form(...),
    student_name: str = Form(...),
    class_code: str = Form(...)
):
    """
    Enroll a student with face images
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process enrollment
        result = enrollment_manager.enroll_student(
            image=image,
            student_id=student_id,
            student_name=student_name,
            class_code=class_code,
            detector=detector,
            extractor=extractor,
            normalizer=normalizer,
            pose_estimator=pose_estimator,
            db_manager=db_manager
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Enrollment error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/attendance")
async def mark_attendance(
    file: UploadFile = File(...),
    class_code: str = Form(...)
):
    """
    Mark student attendance using face recognition (1:N matching - multiple faces)
    
    Process ALL faces in image and return list of recognized students
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect ALL faces in image
        faces = detector.detect_faces(image)
        
        if not faces:
            return {
                "success": True,
                "marked": False,
                "present_students": [],
                "count": 0,
                "message": "No faces detected in image"
            }
        
        # Get gallery for matching
        gallery_embeddings, gallery_ids = db_manager.get_all_embeddings_for_matching()
        present_students = []
        matched_details = []
        
        # Process EACH face in image
        for face in faces:
            try:
                x, y, w, h = face['box']
                crop = image[y:y+h, x:x+w]
                normalized = normalizer.normalize_complete(crop, face.get('keypoints', {}))
                embedding = extractor.extract_embedding(normalized)
                
                if embedding is None:
                    continue
                
                # Identify student for this face
                best_match = matcher.find_best_match(embedding, gallery_embeddings, gallery_ids)
                
                if best_match and best_match['confidence'] > 0.6:
                    student_id = best_match['identity_id']
                    
                    # Only add unique students (avoid duplicates if same person appears twice)
                    if student_id not in present_students:
                        present_students.append(student_id)
                        
                        # Mark attendance in database
                        db_result = db_manager.mark_attendance(student_id, class_code)
                        
                        matched_details.append({
                            "student_id": student_id,
                            "confidence": best_match['confidence'],
                            "marked": db_result
                        })
                        
                        logger.info(f"Marked attendance: {student_id} (confidence: {best_match['confidence']:.3f})")
            
            except Exception as face_error:
                logger.warning(f"Error processing face: {face_error}")
                continue
        
        return {
            "success": True,
            "marked": len(present_students) > 0,
            "present_students": present_students,  # List of recognized student IDs
            "count": len(present_students),
            "faces_detected": len(faces),
            "details": matched_details,
            "message": f"Attendance marked for {len(present_students)} students from {len(faces)} faces detected"
        }
        
    except Exception as e:
        logger.error(f"Attendance error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
