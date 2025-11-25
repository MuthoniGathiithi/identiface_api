"""
Face Recognition API Service
FastAPI-based REST API for facial identification with Django integration
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import cv2
import numpy as np
import logging
import io
import base64
from datetime import datetime

from detection import FaceDetector
from feature_extraction import FeatureExtractor
from normalization import FaceNormalizer
from matching import FaceMatcher
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
    title="Face Recognition API",
    description="Deep learning-based facial identification service using InsightFace (SCRFD + ArcFace)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
detector: Optional[FaceDetector] = None
extractor: Optional[FeatureExtractor] = None
normalizer: Optional[FaceNormalizer] = None
matcher: Optional[FaceMatcher] = None
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
    
class EnrollmentStatus(BaseModel):
    user_id: str
    status: str
    progress: Dict
    message: str

class StudentEnrollmentRequest(BaseModel):
    student_id: str
    student_name: str
    class_code: str

class AttendanceRequest(BaseModel):
    class_code: str


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global detector, extractor, normalizer, matcher, pose_estimator, enrollment_manager, db_manager
    
    logger.info("Initializing Face Recognition Service...")
    
    try:
        # Initialize components with InsightFace
        model_name = 'buffalo_l'  # High accuracy model
        ctx_id = -1  # CPU (-1) or GPU (0, 1, 2, ...)
        
        detector = FaceDetector(model_name=model_name, min_confidence=0.5, ctx_id=ctx_id)
        extractor = FeatureExtractor(model_name=model_name, ctx_id=ctx_id)
        normalizer = FaceNormalizer(target_size=(160, 160))
        matcher = FaceMatcher(threshold=0.4, metric='cosine')
        pose_estimator = PoseEstimator(yaw_threshold=20.0, pitch_threshold=15.0)
        
        # Initialize enrollment manager
        enrollment_manager = EnrollmentManager()
        enrollment_manager.initialize_components(model_name=model_name, ctx_id=ctx_id)
        
        # Initialize database manager (configure Django settings as needed)
        db_manager = DatabaseManager(django_settings_module=None, cache_size=1000)
        
        logger.info("✓ Face Recognition Service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Face Recognition API",
        "version": "1.0.0",
        "status": "running",
        "models": {
            "detector": "InsightFace SCRFD",
            "feature_extractor": "InsightFace ArcFace",
            "embedding_size": 512
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "detector": detector is not None,
            "extractor": extractor is not None and extractor.is_available(),
            "matcher": matcher is not None,
            "database": db_manager is not None
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """
    Detect faces in an image
    
    Returns face bounding boxes, confidence scores, and landmarks
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect faces
        faces = detector.detect_faces(image)
        
        # Format response
        results = []
        for face in faces:
            results.append({
                "box": {
                    "x": int(face['box'][0]),
                    "y": int(face['box'][1]),
                    "width": int(face['box'][2]),
                    "height": int(face['box'][3])
                },
                "confidence": float(face['confidence']),
                "keypoints": {k: [int(v[0]), int(v[1])] for k, v in face['keypoints'].items()}
            })
        
        return {
            "success": True,
            "faces_detected": len(faces),
            "faces": results
        }
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-features")
async def extract_features(file: UploadFile = File(...)):
    """
    Extract face embeddings from an image
    
    Returns 512-dimensional embedding vectors
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract embeddings for all faces
        results = extractor.extract_multiple_embeddings(image)
        
        if not results:
            return {
                "success": False,
                "message": "No faces detected in image",
                "embeddings": []
            }
        
        # Format response
        embeddings_data = []
        for result in results:
            embeddings_data.append({
                "box": {
                    "x": int(result['box'][0]),
                    "y": int(result['box'][1]),
                    "width": int(result['box'][2]),
                    "height": int(result['box'][3])
                },
                "confidence": float(result['confidence']),
                "embedding": result['embedding'].tolist(),
                "embedding_size": len(result['embedding'])
            })
        
        return {
            "success": True,
            "faces_found": len(results),
            "embeddings": embeddings_data
        }
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Verify if two images contain the same person (1:1 matching)
    """
    try:
        # Read first image
        contents1 = await file1.read()
        nparr1 = np.frombuffer(contents1, np.uint8)
        image1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        
        # Read second image
        contents2 = await file2.read()
        nparr2 = np.frombuffer(contents2, np.uint8)
        image2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
        
        if image1 is None or image2 is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract embeddings
        embedding1 = extractor.extract_embedding(image1)
        embedding2 = extractor.extract_embedding(image2)
        
        if embedding1 is None or embedding2 is None:
            return {
                "success": False,
                "message": "Failed to extract face embeddings",
                "match": False
            }
        
        # Verify
        result = matcher.verify(embedding1, embedding2)
        
        return {
            "success": True,
            "match": result['match'],
            "similarity": result['similarity'],
            "distance": result['distance'],
            "confidence": result['confidence'],
            "threshold": result['threshold']
        }
        
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    """
    Identify a person from the enrolled database (1:N matching)
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract embedding
        query_embedding = extractor.extract_embedding(image)
        
        if query_embedding is None:
            return {
                "success": False,
                "message": "No face detected in image",
                "user_id": None,
                "confidence": 0.0
            }
        
        # Get all enrolled embeddings from database
        enrolled = db_manager.get_all_embeddings()
        
        if not enrolled:
            return {
                "success": False,
                "message": "No enrolled users in database",
                "user_id": None,
                "confidence": 0.0
            }
        
        gallery_embeddings = [item['embedding'] for item in enrolled]
        gallery_ids = [item['user_id'] for item in enrolled]
        
        # Identify
        best_match = matcher.find_best_match(query_embedding, gallery_embeddings, gallery_ids)
        
        if best_match:
            # Log recognition attempt
            db_manager.connector.log_recognition_attempt(
                user_id=best_match['identity_id'],
                confidence=best_match['similarity'],
                success=True
            )
            
            # Update last recognition time
            db_manager.connector.update_last_recognition(best_match['identity_id'])
            
            return {
                "success": True,
                "user_id": best_match['identity_id'],
                "confidence": best_match['similarity'],
                "similarity": best_match['similarity'],
                "distance": best_match['distance'],
                "message": "User identified successfully"
            }
        else:
            # Log failed attempt
            db_manager.connector.log_recognition_attempt(
                user_id=None,
                confidence=0.0,
                success=False
            )
            
            return {
                "success": False,
                "user_id": None,
                "confidence": 0.0,
                "similarity": 0.0,
                "message": "No matching user found"
            }
        
    except Exception as e:
        logger.error(f"Identification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/enroll")
async def enroll_student(
    student_id: str = File(...),
    student_name: str = File(...),
    class_code: str = File(...),
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: UploadFile = File(...)
):
    """
    Enroll a student with 3 images for a specific class
    
    Args:
        student_id: Student identifier (e.g., STU001)
        student_name: Student full name
        class_code: Class identifier (e.g., CS101, DB202)
        image1, image2, image3: Three face images of the student
        
    Returns:
        success, face_encoding (base64), student_id, class_code
    """
    try:
        logger.info(f"Starting enrollment for student {student_id} in class {class_code}")
        
        embeddings = []
        
        # Process each of the 3 images
        for idx, image_file in enumerate([image1, image2, image3], 1):
            # Read image
            contents = await image_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid image file: image{idx}"
                )
            
            # Extract embedding
            embedding = extractor.extract_embedding(image)
            
            if embedding is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"No face detected in image{idx}. Please ensure face is clearly visible."
                )
            
            embeddings.append(embedding)
            logger.info(f"Extracted embedding from image {idx} for student {student_id}")
        
        # Average the 3 embeddings for robustness
        average_embedding = np.mean(embeddings, axis=0)
        
        # Normalize the averaged embedding
        average_embedding = average_embedding / np.linalg.norm(average_embedding)
        
        # Prepare enrollment data with class_code
        enrollment_data = {
            'user_id': student_id,
            'student_name': student_name,
            'class_code': class_code,
            'average_embedding': average_embedding,
            'individual_embeddings': embeddings,
            'enrollment_date': datetime.now().isoformat(),
            'embedding_size': len(average_embedding),
            'num_images': 3
        }
        
        # Save to database with class_code
        success = db_manager.save_enrollment_with_class(enrollment_data)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save enrollment to database"
            )
        
        # Convert embedding to base64 for response
        embedding_base64 = base64.b64encode(average_embedding.tobytes()).decode('utf-8')
        
        logger.info(f"Successfully enrolled student {student_id} in class {class_code}")
        
        return {
            "success": True,
            "student_id": student_id,
            "student_name": student_name,
            "class_code": class_code,
            "face_encoding": embedding_base64,
            "embedding_size": len(average_embedding),
            "message": f"Student {student_name} enrolled successfully in {class_code}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment error for student {student_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/mark-attendance")
async def mark_attendance(
    class_code: str = File(...),
    classroom_image1: UploadFile = File(...),
    classroom_image2: UploadFile = File(...),
    classroom_image3: UploadFile = File(...)
):
    """
    Mark attendance for a class by detecting faces in classroom images
    
    Args:
        class_code: Class identifier (e.g., CS101, DB202)
        classroom_image1, classroom_image2, classroom_image3: Three classroom photos
        
    Returns:
        success, present_students (list of student_ids), details
    """
    try:
        logger.info(f"Starting attendance marking for class {class_code}")
        
        # Get enrolled students for this specific class only
        enrolled_students = db_manager.get_embeddings_by_class(class_code)
        
        if not enrolled_students:
            return {
                "success": False,
                "message": f"No enrolled students found for class {class_code}",
                "present_students": [],
                "class_code": class_code
            }
        
        logger.info(f"Found {len(enrolled_students)} enrolled students in class {class_code}")
        
        # Prepare gallery for matching (only students in this class)
        gallery_embeddings = [student['embedding'] for student in enrolled_students]
        gallery_ids = [student['user_id'] for student in enrolled_students]
        
        # Track detected students (use set to avoid duplicates)
        detected_students = set()
        detection_details = []
        
        # Process each classroom image
        for idx, classroom_image in enumerate([classroom_image1, classroom_image2, classroom_image3], 1):
            # Read image
            contents = await classroom_image.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.warning(f"Invalid classroom image {idx}")
                continue
            
            # Detect all faces in the classroom image
            faces = detector.detect_faces(image)
            logger.info(f"Detected {len(faces)} faces in classroom image {idx}")
            
            # Extract embeddings for each detected face
            for face_idx, face in enumerate(faces):
                try:
                    # Extract embedding for this face
                    face_embedding = extractor.extract_embedding_from_face(image, face)
                    
                    if face_embedding is None:
                        continue
                    
                    # Match against enrolled students in this class
                    match_result = matcher.find_best_match(
                        face_embedding, 
                        gallery_embeddings, 
                        gallery_ids
                    )
                    
                    if match_result:
                        student_id = match_result['identity_id']
                        confidence = match_result['similarity']
                        
                        detected_students.add(student_id)
                        
                        detection_details.append({
                            "image_number": idx,
                            "face_number": face_idx + 1,
                            "student_id": student_id,
                            "confidence": float(confidence),
                            "similarity": float(match_result['similarity'])
                        })
                        
                        logger.info(f"Matched student {student_id} with confidence {confidence:.3f}")
                
                except Exception as e:
                    logger.error(f"Error processing face {face_idx} in image {idx}: {e}")
                    continue
        
        # Convert set to sorted list
        present_students = sorted(list(detected_students))
        
        logger.info(f"Attendance marking complete for {class_code}: {len(present_students)} students present")
        
        return {
            "success": True,
            "class_code": class_code,
            "present_students": present_students,
            "total_present": len(present_students),
            "total_enrolled": len(enrolled_students),
            "detection_details": detection_details,
            "message": f"Attendance marked successfully for {class_code}"
        }
        
    except Exception as e:
        logger.error(f"Attendance marking error for class {class_code}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enroll/start")
async def start_enrollment(request: EnrollmentRequest):
    """
    Start a new enrollment session for a user
    Requires capturing multiple poses: front, left, right, down
    """
    try:
        user_id = request.user_id
        
        # Check if session already exists
        existing_session = enrollment_manager.get_session(user_id)
        if existing_session:
            return {
                "success": False,
                "message": "Enrollment session already in progress",
                "progress": existing_session.get_progress()
            }
        
        # Start new session
        session = enrollment_manager.start_session(user_id)
        
        return {
            "success": True,
            "message": "Enrollment session started",
            "user_id": user_id,
            "required_poses": [pose.value for pose in session.required_poses],
            "progress": session.get_progress()
        }
        
    except Exception as e:
        logger.error(f"Enrollment start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enroll/process-frame/{user_id}")
async def process_enrollment_frame(user_id: str, file: UploadFile = File(...)):
    """
    Process a frame for enrollment
    Returns feedback on pose, quality, and readiness to capture
    """
    try:
        # Get session
        session = enrollment_manager.get_session(user_id)
        if not session:
            raise HTTPException(status_code=404, detail="No enrollment session found for user")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process frame
        result = session.process_frame(frame)
        
        # If ready and status is 'ready', auto-capture
        if result['status'] == 'ready':
            captured = session.capture_pose(frame, result['face'])
            if captured:
                result['captured'] = True
                result['progress'] = session.get_progress()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment frame processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enroll/complete/{user_id}")
async def complete_enrollment(user_id: str, background_tasks: BackgroundTasks):
    """
    Complete enrollment and save to database
    """
    try:
        # Get enrollment data
        enrollment_data = enrollment_manager.end_session(user_id)
        
        if not enrollment_data:
            raise HTTPException(status_code=400, detail="Enrollment not complete or session not found")
        
        # Save to database
        success = db_manager.save_enrollment(enrollment_data)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save enrollment to database")
        
        return {
            "success": True,
            "message": "Enrollment completed successfully",
            "user_id": user_id,
            "enrollment_date": enrollment_data['enrollment_date']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enroll/cancel/{user_id}")
async def cancel_enrollment(user_id: str):
    """Cancel enrollment session"""
    try:
        enrollment_manager.cancel_session(user_id)
        
        return {
            "success": True,
            "message": "Enrollment session cancelled",
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Enrollment cancellation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/enroll/status/{user_id}")
async def get_enrollment_status(user_id: str):
    """Get enrollment session status"""
    try:
        session = enrollment_manager.get_session(user_id)
        
        if not session:
            return {
                "success": False,
                "message": "No active enrollment session",
                "user_id": user_id
            }
        
        return {
            "success": True,
            "progress": session.get_progress(),
            "current_pose": session.get_current_required_pose().value if session.get_current_required_pose() else None
        }
        
    except Exception as e:
        logger.error(f"Enrollment status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/user/{user_id}")
async def delete_user(user_id: str):
    """Delete user enrollment from database"""
    try:
        success = db_manager.delete_enrollment(user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="User not found or deletion failed")
        
        return {
            "success": True,
            "message": "User enrollment deleted successfully",
            "user_id": user_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/count")
async def get_users_count():
    """Get count of enrolled users"""
    try:
        enrolled = db_manager.get_all_embeddings()
        
        return {
            "success": True,
            "count": len(enrolled)
        }
        
    except Exception as e:
        logger.error(f"User count error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/database/refresh-cache")
async def refresh_database_cache():
    """Refresh the embedding cache from database"""
    try:
        db_manager.refresh_cache()
        
        return {
            "success": True,
            "message": "Database cache refreshed",
            "cache_size": db_manager.cache.size()
        }
        
    except Exception as e:
        logger.error(f"Cache refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
