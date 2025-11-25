"""
Django Integration Example
==========================
Copy these views to your Django project to integrate with the FastAPI face recognition service.

This file shows how to call the FastAPI endpoints from Django.
"""

import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import logging

logger = logging.getLogger(__name__)

# FastAPI service URL - update this to match your setup
FACE_API_URL = "http://localhost:8000"


# ============================================================================
# ENROLLMENT VIEW
# ============================================================================

@csrf_exempt
@require_http_methods(["POST"])
def enroll_student_view(request):
    """
    Enroll a student with face recognition
    
    Expected POST data:
    - student_id: Student identifier (e.g., "STU001")
    - student_name: Student full name
    - class_code: Class code (e.g., "CS101")
    - image1, image2, image3: Three face images
    
    Returns:
    - JSON response with enrollment status
    """
    try:
        # Validate required fields
        required_fields = ['student_id', 'student_name', 'class_code']
        for field in required_fields:
            if field not in request.POST:
                return JsonResponse({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }, status=400)
        
        # Validate image files
        required_images = ['image1', 'image2', 'image3']
        for img_field in required_images:
            if img_field not in request.FILES:
                return JsonResponse({
                    'success': False,
                    'error': f'Missing required image: {img_field}'
                }, status=400)
        
        # Prepare files for FastAPI
        files = {
            'image1': (
                request.FILES['image1'].name,
                request.FILES['image1'].read(),
                request.FILES['image1'].content_type
            ),
            'image2': (
                request.FILES['image2'].name,
                request.FILES['image2'].read(),
                request.FILES['image2'].content_type
            ),
            'image3': (
                request.FILES['image3'].name,
                request.FILES['image3'].read(),
                request.FILES['image3'].content_type
            ),
        }
        
        # Prepare form data
        data = {
            'student_id': request.POST['student_id'],
            'student_name': request.POST['student_name'],
            'class_code': request.POST['class_code'],
        }
        
        # Call FastAPI enrollment endpoint
        logger.info(f"Enrolling student {data['student_id']} in class {data['class_code']}")
        
        response = requests.post(
            f"{FACE_API_URL}/api/enroll",
            files=files,
            data=data,
            timeout=30
        )
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                logger.info(f"Successfully enrolled student {data['student_id']}")
                
                # Optional: Save enrollment record to Django database
                # from your_app.models import StudentEnrollment
                # StudentEnrollment.objects.create(
                #     student_id=data['student_id'],
                #     student_name=data['student_name'],
                #     class_code=data['class_code'],
                #     enrollment_date=timezone.now()
                # )
                
                return JsonResponse(result)
            else:
                logger.error(f"Enrollment failed: {result.get('message', 'Unknown error')}")
                return JsonResponse(result, status=400)
        else:
            logger.error(f"FastAPI returned status {response.status_code}")
            return JsonResponse({
                'success': False,
                'error': f'Face recognition service error: {response.status_code}'
            }, status=500)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to face recognition service: {e}")
        return JsonResponse({
            'success': False,
            'error': 'Failed to connect to face recognition service'
        }, status=503)
    
    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


# ============================================================================
# ATTENDANCE MARKING VIEW
# ============================================================================

@csrf_exempt
@require_http_methods(["POST"])
def mark_attendance_view(request):
    """
    Mark attendance for a class using face recognition
    
    Expected POST data:
    - class_code: Class code (e.g., "CS101")
    - classroom_image1, classroom_image2, classroom_image3: Three classroom photos
    
    Returns:
    - JSON response with list of present students
    """
    try:
        # Validate required fields
        if 'class_code' not in request.POST:
            return JsonResponse({
                'success': False,
                'error': 'Missing required field: class_code'
            }, status=400)
        
        # Validate image files
        required_images = ['classroom_image1', 'classroom_image2', 'classroom_image3']
        for img_field in required_images:
            if img_field not in request.FILES:
                return JsonResponse({
                    'success': False,
                    'error': f'Missing required image: {img_field}'
                }, status=400)
        
        class_code = request.POST['class_code']
        
        # Prepare files for FastAPI
        files = {
            'classroom_image1': (
                request.FILES['classroom_image1'].name,
                request.FILES['classroom_image1'].read(),
                request.FILES['classroom_image1'].content_type
            ),
            'classroom_image2': (
                request.FILES['classroom_image2'].name,
                request.FILES['classroom_image2'].read(),
                request.FILES['classroom_image2'].content_type
            ),
            'classroom_image3': (
                request.FILES['classroom_image3'].name,
                request.FILES['classroom_image3'].read(),
                request.FILES['classroom_image3'].content_type
            ),
        }
        
        # Prepare form data
        data = {
            'class_code': class_code,
        }
        
        # Call FastAPI attendance endpoint
        logger.info(f"Marking attendance for class {class_code}")
        
        response = requests.post(
            f"{FACE_API_URL}/api/mark-attendance",
            files=files,
            data=data,
            timeout=60  # Longer timeout for processing multiple faces
        )
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            if result.get('success'):
                present_students = result.get('present_students', [])
                logger.info(f"Attendance marked: {len(present_students)} students present in {class_code}")
                
                # Save attendance records to Django database
                # from your_app.models import AttendanceRecord
                # from django.utils import timezone
                # 
                # for student_id in present_students:
                #     AttendanceRecord.objects.create(
                #         student_id=student_id,
                #         class_code=class_code,
                #         timestamp=timezone.now(),
                #         marked_by='face_recognition'
                #     )
                
                return JsonResponse(result)
            else:
                logger.warning(f"Attendance marking returned no results: {result.get('message')}")
                return JsonResponse(result)
        else:
            logger.error(f"FastAPI returned status {response.status_code}")
            return JsonResponse({
                'success': False,
                'error': f'Face recognition service error: {response.status_code}'
            }, status=500)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to face recognition service: {e}")
        return JsonResponse({
            'success': False,
            'error': 'Failed to connect to face recognition service'
        }, status=503)
    
    except Exception as e:
        logger.error(f"Attendance marking error: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


# ============================================================================
# OPTIONAL: GET ENROLLED STUDENTS FOR A CLASS
# ============================================================================

@require_http_methods(["GET"])
def get_enrolled_students_view(request, class_code):
    """
    Get list of enrolled students for a class
    
    This is optional - you can query the FastAPI service to see who's enrolled
    """
    try:
        # You could add an endpoint in FastAPI to list enrolled students
        # For now, this is just a placeholder
        
        # Example:
        # response = requests.get(f"{FACE_API_URL}/api/enrolled/{class_code}")
        # return JsonResponse(response.json())
        
        return JsonResponse({
            'success': False,
            'message': 'Not implemented yet'
        })
    
    except Exception as e:
        logger.error(f"Error getting enrolled students: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


# ============================================================================
# URL PATTERNS (add to your urls.py)
# ============================================================================

"""
from django.urls import path
from . import views

urlpatterns = [
    path('api/face/enroll/', views.enroll_student_view, name='enroll_student'),
    path('api/face/mark-attendance/', views.mark_attendance_view, name='mark_attendance'),
    path('api/face/enrolled/<str:class_code>/', views.get_enrolled_students_view, name='get_enrolled'),
]
"""


# ============================================================================
# DJANGO MODELS (optional - for storing attendance records)
# ============================================================================

"""
from django.db import models
from django.utils import timezone

class AttendanceRecord(models.Model):
    student_id = models.CharField(max_length=50, db_index=True)
    class_code = models.CharField(max_length=50, db_index=True)
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    marked_by = models.CharField(max_length=50, default='face_recognition')
    confidence = models.FloatField(null=True, blank=True)
    
    class Meta:
        db_table = 'attendance_records'
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['class_code', 'timestamp']),
            models.Index(fields=['student_id', 'timestamp']),
        ]
    
    def __str__(self):
        return f"{self.student_id} - {self.class_code} - {self.timestamp}"


class StudentEnrollment(models.Model):
    student_id = models.CharField(max_length=50, unique=True)
    student_name = models.CharField(max_length=200)
    class_code = models.CharField(max_length=50, db_index=True)
    enrollment_date = models.DateTimeField(default=timezone.now)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        db_table = 'student_enrollments'
        unique_together = ['student_id', 'class_code']
    
    def __str__(self):
        return f"{self.student_name} ({self.student_id}) - {self.class_code}"
"""
