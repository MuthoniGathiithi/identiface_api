"""
Face Recognition API Service - REFACTORED & ENHANCED
FastAPI-based REST API with advanced error handling, validation, and logging
Version: 2.1.0
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional, List, Dict
import cv2
import numpy as np
import logging
import io
import base64
import time
from datetime import datetime
import traceback

# Import upgraded modules
from detection_advanced import FaceDetectorAdvanced, DetectionModel
from feature_extraction_advanced import FeatureExtractorAdvanced, EmbeddingModel
from normalization_advanced import FaceNormalizerAdvanced
from matching_advanced import FaceMatcherAdvanced, DistanceMetric
from long_distance_optimizer import LongDistanceOptimizer
from pose_estimation import PoseEstimator, FacePose
from enrollment import EnrollmentManager
from database import DatabaseManager

# Import utilities
from utils import APIResponse, InputValidator, OutputSanitizer, create_logger

# Configure logging
logger = create_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Face Recognition API",
    description="State-of-the-art facial identification service with multi-model support",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
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


# ==================== Pydantic Models ====================

class EnrollmentRequest(BaseModel):
    """Student enrollment request"""
    student_id: str
    student_name: str
    class_code: str
    
    @validator('student_id', 'student_name', 'class_code')
    def validate_required_fields(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field must be non-empty string")
        return v.strip()


class VerificationRequest(BaseModel):
    """Face verification request"""
    threshold: Optional[float] = 0.6
    
    @validator('threshold')
    def validate_threshold(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Threshold must be between 0 and 1")
        return v


class AttendanceRequest(BaseModel):
    """Attendance marking request"""
    class_code: str
    
    @validator('class_code')
    def validate_class_code(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Class code is required")
        return v.strip()


class IdentificationRequest(BaseModel):
    """Face identification request"""
    top_k: int = 5
    confidence_threshold: float = 0.6
    
    @validator('top_k')
    def validate_top_k(cls, v):
        if v < 1 or v > 20:
            raise ValueError("top_k must be between 1 and 20")
        return v
    
    @validator('confidence_threshold')
    def validate_confidence(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v


# ==================== Exception Handlers ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all uncaught exceptions"""
    error_id = datetime.now().timestamp()
    logger.log_error(f"Unhandled exception (ID: {error_id})", exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=APIResponse.error(
            message="Internal server error",
            error_code="INTERNAL_ERROR",
            status_code=500
        )
    )


# ==================== Lifecycle Events ====================

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global detector, extractor, normalizer, matcher
    global long_distance_optimizer, pose_estimator, enrollment_manager, db_manager
    
    logger.logger.info("=" * 60)
    logger.logger.info("Starting Face Recognition Service v2.1.0")
    logger.logger.info("=" * 60)
    
    try:
        ctx_id = -1  # CPU (-1) or GPU (0, 1, 2, ...)
        
        logger.logger.info("Initializing components...")
        
        # Initialize detector
        logger.logger.info("  → Detector (multi-model ensemble)...")
        detector = FaceDetectorAdvanced(
            model=DetectionModel.ENSEMBLE,
            min_confidence=0.5,
            ctx_id=ctx_id,
            enable_blur_detection=True,
            enable_face_quality=True
        )
        
        # Initialize extractor
        logger.logger.info("  → Feature extractor (ArcFace)...")
        extractor = FeatureExtractorAdvanced(
            model=EmbeddingModel.ARCFACE,
            ctx_id=ctx_id,
            use_ensemble=False,
            normalize_embeddings=True
        )
        
        # Initialize normalizer
        logger.logger.info("  → Face normalizer...")
        normalizer = FaceNormalizerAdvanced(
            target_size=(224, 224),
            enable_affine_alignment=True,
            enable_3d_alignment=False,
            normalize_color=True
        )
        
        # Initialize matcher
        logger.logger.info("  → Face matcher (cosine)...")
        matcher = FaceMatcherAdvanced(
            metric=DistanceMetric.COSINE,
            threshold=0.4,
            use_adaptive_threshold=True,
            enable_quality_weighting=True
        )
        
        # Initialize long-distance optimizer
        logger.logger.info("  → Long-distance optimizer...")
        long_distance_optimizer = LongDistanceOptimizer(
            enable_super_resolution=False,
            enable_multi_scale=True,
            enable_denoising=True
        )
        
        # Initialize pose estimator
        logger.logger.info("  → Pose estimator...")
        pose_estimator = PoseEstimator(yaw_threshold=20.0, pitch_threshold=15.0)
        
        # Initialize enrollment manager
        logger.logger.info("  → Enrollment manager...")
        enrollment_manager = EnrollmentManager()
        
        # Initialize database manager
        logger.logger.info("  → Database manager...")
        db_manager = DatabaseManager(django_settings_module=None, cache_size=1000)
        
        logger.logger.info("✓ Service initialized successfully")
        logger.logger.info("=" * 60)
        
    except Exception as e:
        logger.log_error("Failed to initialize service", e)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.logger.info("Shutting down Face Recognition Service")


# ==================== Health & Info Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint - service info"""
    return APIResponse.success(
        data={
            "service": "Advanced Face Recognition API",
            "version": "2.1.0",
            "status": "running",
            "capabilities": {
                "detection": "Multi-model ensemble (SCRFD, YOLOv8, RetinaFace)",
                "recognition": "ArcFace embeddings (512-dim)",
                "long_distance": "Optimized for long-distance and small faces",
                "multi_face": "Multiple faces per image support",
                "features": [
                    "Multi-scale detection",
                    "Blur detection",
                    "Quality assessment",
                    "Adaptive thresholds",
                    "Distance-aware matching",
                    "Threshold-based filtering",
                    "Confidence scoring"
                ]
            }
        },
        message="Service is running"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return APIResponse.success(
        data={
            "components": {
                "detector": detector is not None,
                "extractor": extractor is not None,
                "matcher": matcher is not None,
                "normalizer": normalizer is not None,
                "long_distance_optimizer": long_distance_optimizer is not None,
                "database": db_manager is not None
            },
            "ready": all([
                detector is not None,
                extractor is not None,
                matcher is not None,
                normalizer is not None,
                db_manager is not None
            ])
        },
        message="Health check passed"
    )


# ==================== Face Detection Endpoint ====================

@app.post("/detect")
async def detect_faces(
    file: UploadFile = File(...),
    enable_quality: bool = True
):
    """
    Detect faces in an image with multi-model ensemble
    
    Returns:
    - faces_detected: Number of faces found
    - faces: List of face detections with boxes, confidence, and keypoints
    - image_size: Dimensions of input image
    """
    start_time = time.time()
    
    try:
        # Validate file
        contents = await file.read()
        is_valid, error_msg = InputValidator.validate_file_upload(contents, file.filename)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("file", error_msg)
            )
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        is_valid, error_msg = InputValidator.validate_image_data(image)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("image", error_msg)
            )
        
        # Detect faces
        faces = detector.detect_faces(image)
        
        # Format response
        results = []
        confidence_scores = []
        
        for face in faces:
            confidence = float(face['confidence'])
            confidence_scores.append(confidence)
            
            face_data = {
                "box": {
                    "x": int(face['box'][0]),
                    "y": int(face['box'][1]),
                    "width": int(face['box'][2]),
                    "height": int(face['box'][3])
                },
                "confidence": confidence,
                "model": face.get('model', 'ensemble'),
                "keypoints": {
                    k: [int(v[0]), int(v[1])]
                    for k, v in face.get('keypoints', {}).items()
                }
            }
            
            if enable_quality and 'quality_score' in face:
                face_data['quality_score'] = float(face['quality_score'])
            
            results.append(face_data)
        
        # Log detection
        logger.log_detection(len(faces), confidence_scores)
        
        duration_ms = (time.time() - start_time) * 1000
        logger.log_performance("detect_faces", duration_ms, faces=len(faces))
        
        return APIResponse.success(
            data={
                "faces_detected": len(faces),
                "faces": results,
                "image_size": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            },
            message=f"Detected {len(faces)} face(s)" if faces else "No faces detected"
        )
        
    except Exception as e:
        logger.log_error("Face detection failed", e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse.error(
                message="Face detection failed",
                error_code="DETECTION_ERROR",
                status_code=500
            )
        )


# ==================== Feature Extraction Endpoint ====================

@app.post("/extract-features")
async def extract_features(file: UploadFile = File(...)):
    """
    Extract face embeddings (512-dimensional ArcFace vectors)
    
    Returns:
    - embeddings_extracted: Number of embeddings
    - embeddings: List of embedding vectors with dimension and confidence
    """
    start_time = time.time()
    
    try:
        # Validate file
        contents = await file.read()
        is_valid, error_msg = InputValidator.validate_file_upload(contents, file.filename)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("file", error_msg)
            )
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        is_valid, error_msg = InputValidator.validate_image_data(image)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("image", error_msg)
            )
        
        # Detect faces
        faces = detector.detect_faces(image)
        
        if not faces:
            return APIResponse.success(
                data={"embeddings": []},
                message="No faces detected in image"
            )
        
        embeddings = []
        for face in faces:
            try:
                x, y, w, h = face['box']
                face_crop = image[y:y+h, x:x+w]
                
                # Normalize face
                normalized = normalizer.normalize_complete(
                    face_crop,
                    face.get('keypoints', {})
                )
                
                # Extract embedding
                embedding = extractor.extract_embedding(normalized)
                
                if embedding is not None:
                    embeddings.append({
                        "embedding": OutputSanitizer.sanitize_embedding(embedding),
                        "dimension": len(embedding),
                        "confidence": float(face['confidence']),
                        "quality_score": float(face.get('quality_score', 1.0))
                    })
            except Exception as e:
                logger.logger.warning(f"Failed to extract embedding from face: {e}")
                continue
        
        duration_ms = (time.time() - start_time) * 1000
        logger.log_performance("extract_features", duration_ms, embeddings=len(embeddings))
        
        return APIResponse.success(
            data={
                "embeddings_extracted": len(embeddings),
                "embeddings": embeddings
            },
            message=f"Extracted {len(embeddings)} embedding(s)"
        )
        
    except Exception as e:
        logger.log_error("Feature extraction failed", e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse.error(
                message="Feature extraction failed",
                error_code="EXTRACTION_ERROR",
                status_code=500
            )
        )


# ==================== Verification Endpoint ====================

@app.post("/verify")
async def verify_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float = 0.6
):
    """
    Verify if two faces belong to the same person (1:1 matching)
    
    Returns:
    - match: Boolean indicating if faces match
    - similarity: Similarity score (0-1)
    - confidence: Confidence of match
    - threshold: Applied threshold
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        is_valid, error = InputValidator.validate_numeric_range(
            threshold, "threshold", 0.0, 1.0
        )
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("threshold", error)
            )
        
        # Read and validate images
        contents1 = await file1.read()
        is_valid, error = InputValidator.validate_file_upload(contents1, file1.filename)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("file1", error)
            )
        
        nparr1 = np.frombuffer(contents1, np.uint8)
        image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        is_valid, error = InputValidator.validate_image_data(image1)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("image1", error)
            )
        
        contents2 = await file2.read()
        is_valid, error = InputValidator.validate_file_upload(contents2, file2.filename)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("file2", error)
            )
        
        nparr2 = np.frombuffer(contents2, np.uint8)
        image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
        is_valid, error = InputValidator.validate_image_data(image2)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("image2", error)
            )
        
        # Detect and extract embeddings
        faces1 = detector.detect_faces(image1)
        faces2 = detector.detect_faces(image2)
        
        if not faces1 or not faces2:
            return APIResponse.success(
                data={
                    "match": False,
                    "similarity": 0.0,
                    "confidence": 0.0,
                    "reason": "Could not detect face(s) in one or both images"
                },
                message="No faces detected in one or both images"
            )
        
        # Extract embeddings from first face of each image
        try:
            face1 = faces1[0]
            x1, y1, w1, h1 = face1['box']
            crop1 = image1[y1:y1+h1, x1:x1+w1]
            norm1 = normalizer.normalize_complete(crop1, face1.get('keypoints', {}))
            emb1 = extractor.extract_embedding(norm1)
            
            face2 = faces2[0]
            x2, y2, w2, h2 = face2['box']
            crop2 = image2[y2:y2+h2, x2:x2+w2]
            norm2 = normalizer.normalize_complete(crop2, face2.get('keypoints', {}))
            emb2 = extractor.extract_embedding(norm2)
            
            if emb1 is None or emb2 is None:
                return APIResponse.success(
                    data={
                        "match": False,
                        "similarity": 0.0,
                        "confidence": 0.0,
                        "reason": "Could not extract embeddings"
                    },
                    message="Feature extraction failed"
                )
            
            # Verify
            result = matcher.verify(
                emb1, emb2,
                quality1=face1.get('quality_score', 1.0),
                quality2=face2.get('quality_score', 1.0)
            )
            
            # Log result
            logger.log_matching(result['similarity'], result['confidence'], threshold)
            
            duration_ms = (time.time() - start_time) * 1000
            logger.log_performance("verify_faces", duration_ms, match=result['match'])
            
            return APIResponse.success(
                data={
                    "match": result['match'],
                    "similarity": result['similarity'],
                    "distance": result['distance'],
                    "confidence": result['confidence'],
                    "threshold_applied": threshold,
                    "metric": result['model']
                },
                message=f"Faces {'match' if result['match'] else 'do not match'}"
            )
        
        except Exception as e:
            logger.log_error("Verification processing failed", e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=APIResponse.error(
                    message="Verification processing failed",
                    error_code="VERIFICATION_ERROR",
                    status_code=500
                )
            )
        
    except Exception as e:
        logger.log_error("Verification request failed", e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse.error(
                message="Verification failed",
                error_code="VERIFICATION_ERROR",
                status_code=500
            )
        )


# ==================== Identification Endpoint ====================

@app.post("/identify")
async def identify_face(
    file: UploadFile = File(...),
    top_k: int = 5,
    confidence_threshold: float = 0.6
):
    """
    Identify a face from gallery (1:N matching) - WITH THRESHOLD FILTERING
    
    Returns:
    - identified: Boolean indicating if face was identified
    - matches: Top-K matches with similarity and confidence
    - threshold_applied: Confidence threshold used
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        is_valid, error = InputValidator.validate_numeric_range(
            top_k, "top_k", 1, 20
        )
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("top_k", error)
            )
        
        is_valid, error = InputValidator.validate_numeric_range(
            confidence_threshold, "confidence_threshold", 0.0, 1.0
        )
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("confidence_threshold", error)
            )
        
        # Validate file
        contents = await file.read()
        is_valid, error_msg = InputValidator.validate_file_upload(contents, file.filename)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("file", error_msg)
            )
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        is_valid, error_msg = InputValidator.validate_image_data(image)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("image", error_msg)
            )
        
        # Detect face
        faces = detector.detect_faces(image)
        
        if not faces:
            return APIResponse.success(
                data={
                    "identified": False,
                    "matches": [],
                    "reason": "No face detected"
                },
                message="No face detected in image"
            )
        
        try:
            # Extract embedding
            face = faces[0]
            x, y, w, h = face['box']
            crop = image[y:y+h, x:x+w]
            normalized = normalizer.normalize_complete(crop, face.get('keypoints', {}))
            embedding = extractor.extract_embedding(normalized)
            
            if embedding is None:
                return APIResponse.success(
                    data={
                        "identified": False,
                        "matches": [],
                        "reason": "Could not extract embedding"
                    },
                    message="Feature extraction failed"
                )
            
            # Get gallery embeddings
            gallery_embeddings, gallery_ids = db_manager.get_all_embeddings_for_matching()
            
            if not gallery_embeddings:
                return APIResponse.success(
                    data={
                        "identified": False,
                        "matches": [],
                        "reason": "Gallery is empty"
                    },
                    message="No faces in gallery"
                )
            
            # Identify with threshold filtering
            matches = matcher.identify(
                embedding,
                gallery_embeddings,
                gallery_ids,
                top_k=top_k
            )
            
            # Filter by confidence threshold
            filtered_matches = [
                m for m in matches
                if m['confidence'] >= confidence_threshold
            ]
            
            is_identified = len(filtered_matches) > 0
            
            duration_ms = (time.time() - start_time) * 1000
            logger.log_performance(
                "identify_face",
                duration_ms,
                identified=is_identified,
                matches=len(filtered_matches)
            )
            
            return APIResponse.success(
                data={
                    "identified": is_identified,
                    "matches_found": len(filtered_matches),
                    "total_candidates": len(matches),
                    "threshold_applied": confidence_threshold,
                    "matches": [
                        {
                            "identity_id": m['identity_id'],
                            "similarity": m['similarity'],
                            "distance": m['distance'],
                            "confidence": m['confidence'],
                            "passed_threshold": m['confidence'] >= confidence_threshold
                        }
                        for m in filtered_matches
                    ]
                },
                message=f"Face identified with {len(filtered_matches)} match(es)" if is_identified else "No matches above threshold"
            )
        
        except Exception as e:
            logger.log_error("Identification processing failed", e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=APIResponse.error(
                    message="Identification processing failed",
                    error_code="IDENTIFICATION_ERROR",
                    status_code=500
                )
            )
        
    except Exception as e:
        logger.log_error("Identification request failed", e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse.error(
                message="Identification failed",
                error_code="IDENTIFICATION_ERROR",
                status_code=500
            )
        )


# ==================== Multi-Face Attendance Endpoint ====================

@app.post("/attendance")
async def mark_attendance(
    file: UploadFile = File(...),
    class_code: str = Form(...),
    confidence_threshold: float = Form(0.65)
):
    """
    Mark student attendance using multi-face recognition
    
    Process ALL faces in image with threshold-based filtering:
    - Detects multiple faces per image
    - Uses confidence threshold to reduce false positives
    - Returns list of recognized students
    
    Returns:
    - marked: Whether any attendance was marked
    - present_students: List of student IDs marked present
    - count: Number of students marked
    - faces_detected: Total faces detected in image
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        is_valid, error = InputValidator.validate_id_format(class_code, "class_code")
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("class_code", error)
            )
        
        is_valid, error = InputValidator.validate_numeric_range(
            confidence_threshold, "confidence_threshold", 0.5, 0.95
        )
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("confidence_threshold", error)
            )
        
        # Validate file
        contents = await file.read()
        is_valid, error_msg = InputValidator.validate_file_upload(contents, file.filename)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("file", error_msg)
            )
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        is_valid, error_msg = InputValidator.validate_image_data(image)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("image", error_msg)
            )
        
        # Detect ALL faces in image (multi-face support)
        faces = detector.detect_faces(image)
        
        if not faces:
            return APIResponse.success(
                data={
                    "marked": False,
                    "present_students": [],
                    "count": 0,
                    "faces_detected": 0,
                    "confidence_threshold": confidence_threshold
                },
                message="No faces detected in image"
            )
        
        try:
            # Get gallery for matching
            gallery_embeddings, gallery_ids = db_manager.get_all_embeddings_for_matching()
            
            present_students = []
            matched_details = []
            processed_faces = 0
            
            # Process EACH face in image (multi-face recognition)
            for face_idx, face in enumerate(faces):
                try:
                    x, y, w, h = face['box']
                    crop = image[y:y+h, x:x+w]
                    normalized = normalizer.normalize_complete(crop, face.get('keypoints', {}))
                    embedding = extractor.extract_embedding(normalized)
                    
                    if embedding is None:
                        continue
                    
                    processed_faces += 1
                    
                    # Identify student for this face with threshold filtering
                    best_match = matcher.find_best_match(
                        embedding,
                        gallery_embeddings,
                        gallery_ids
                    )
                    
                    # Apply confidence threshold to reduce false positives
                    if best_match and best_match['confidence'] >= confidence_threshold:
                        student_id = best_match['identity_id']
                        
                        # Only add unique students (avoid duplicates if same person appears twice)
                        if student_id not in present_students:
                            present_students.append(student_id)
                            
                            # Mark attendance in database
                            try:
                                db_result = db_manager.mark_attendance(student_id, class_code)
                                matched_details.append({
                                    "student_id": student_id,
                                    "confidence": best_match['confidence'],
                                    "marked": db_result,
                                    "face_index": face_idx
                                })
                                logger.logger.info(
                                    f"✓ Attendance marked: {student_id} "
                                    f"(confidence: {best_match['confidence']:.3f})"
                                )
                            except Exception as db_err:
                                logger.logger.error(f"Database error marking attendance: {db_err}")
                                matched_details.append({
                                    "student_id": student_id,
                                    "confidence": best_match['confidence'],
                                    "marked": False,
                                    "error": str(db_err),
                                    "face_index": face_idx
                                })
                        else:
                            # Student already marked from another face
                            matched_details.append({
                                "student_id": student_id,
                                "confidence": best_match['confidence'],
                                "marked": False,
                                "note": "Duplicate - same student already marked",
                                "face_index": face_idx
                            })
                    else:
                        # Below threshold
                        confidence = best_match['confidence'] if best_match else 0.0
                        matched_details.append({
                            "face_index": face_idx,
                            "confidence": confidence,
                            "marked": False,
                            "reason": f"Below threshold ({confidence:.3f} < {confidence_threshold})"
                        })
                
                except Exception as face_error:
                    logger.logger.warning(f"Error processing face {face_idx}: {face_error}")
                    continue
            
            duration_ms = (time.time() - start_time) * 1000
            logger.log_performance(
                "mark_attendance",
                duration_ms,
                students=len(present_students),
                faces=len(faces)
            )
            
            return APIResponse.success(
                data={
                    "marked": len(present_students) > 0,
                    "present_students": present_students,
                    "count": len(present_students),
                    "faces_detected": len(faces),
                    "faces_processed": processed_faces,
                    "confidence_threshold": confidence_threshold,
                    "details": matched_details
                },
                message=(
                    f"Attendance marked for {len(present_students)} student(s) "
                    f"from {len(faces)} face(s) detected"
                )
            )
        
        except Exception as e:
            logger.log_error("Attendance processing failed", e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=APIResponse.error(
                    message="Attendance processing failed",
                    error_code="ATTENDANCE_ERROR",
                    status_code=500
                )
            )
        
    except Exception as e:
        logger.log_error("Attendance request failed", e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse.error(
                message="Attendance marking failed",
                error_code="ATTENDANCE_ERROR",
                status_code=500
            )
        )


# ==================== Enrollment Endpoint ====================

@app.post("/enroll-student")
async def enroll_student(
    file: UploadFile = File(...),
    student_id: str = Form(...),
    student_name: str = Form(...),
    class_code: str = Form(...)
):
    """
    Enroll a student with face images
    
    Returns:
    - enrolled: Boolean indicating successful enrollment
    - embedding_dimension: Dimension of stored embedding
    - confidence: Detection confidence
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        is_valid, error = InputValidator.validate_id_format(student_id, "student_id")
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("student_id", error)
            )
        
        is_valid, error = InputValidator.validate_string_input(
            student_name, "student_name", min_length=2, max_length=100
        )
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("student_name", error)
            )
        
        is_valid, error = InputValidator.validate_id_format(class_code, "class_code")
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("class_code", error)
            )
        
        # Validate file
        contents = await file.read()
        is_valid, error_msg = InputValidator.validate_file_upload(contents, file.filename)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("file", error_msg)
            )
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        is_valid, error_msg = InputValidator.validate_image_data(image)
        if not is_valid:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=APIResponse.validation_error("image", error_msg)
            )
        
        try:
            # Detect face
            faces = detector.detect_faces(image)
            
            if not faces:
                return APIResponse.success(
                    data={"enrolled": False, "reason": "No face detected"},
                    message="Failed to enroll: No face detected"
                )
            
            # Extract embedding from first face
            face = faces[0]
            x, y, w, h = face['box']
            crop = image[y:y+h, x:x+w]
            normalized = normalizer.normalize_complete(crop, face.get('keypoints', {}))
            embedding = extractor.extract_embedding(normalized)
            
            if embedding is None:
                return APIResponse.success(
                    data={"enrolled": False, "reason": "Could not extract face features"},
                    message="Failed to enroll: Feature extraction failed"
                )
            
            # Store in database
            enrollment_result = db_manager.enroll_student(
                student_id=student_id,
                student_name=student_name,
                class_code=class_code,
                embedding=embedding
            )
            
            duration_ms = (time.time() - start_time) * 1000
            logger.log_performance(
                "enroll_student",
                duration_ms,
                student=student_id,
                success=enrollment_result
            )
            
            if enrollment_result:
                logger.logger.info(
                    f"✓ Student enrolled: {student_id} ({student_name}) "
                    f"in class {class_code}"
                )
                return APIResponse.success(
                    data={
                        "enrolled": True,
                        "student_id": student_id,
                        "embedding_dimension": len(embedding),
                        "confidence": float(face['confidence'])
                    },
                    message=f"Student {student_name} enrolled successfully"
                )
            else:
                return APIResponse.success(
                    data={"enrolled": False, "reason": "Database enrollment failed"},
                    message="Failed to save enrollment to database"
                )
        
        except Exception as e:
            logger.log_error("Enrollment processing failed", e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=APIResponse.error(
                    message="Enrollment processing failed",
                    error_code="ENROLLMENT_ERROR",
                    status_code=500
                )
            )
        
    except Exception as e:
        logger.log_error("Enrollment request failed", e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse.error(
                message="Enrollment failed",
                error_code="ENROLLMENT_ERROR",
                status_code=500
            )
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
