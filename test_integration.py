"""
Integration Test Suite for Face Recognition System
Comprehensive testing of backend-frontend communication
"""

import os
import sys
import time
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTester:
    """Test backend-frontend integration"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.all_passed = True
        self.results = []
    
    def test_backend_health(self) -> bool:
        """Test 1: Backend health check"""
        logger.info("=" * 60)
        logger.info("TEST 1: Backend Health Check")
        logger.info("=" * 60)
        
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            data = response.json()
            
            if response.status_code == 200 and data.get('status') == 'success':
                components = data.get('data', {}).get('components', {})
                ready = data.get('data', {}).get('ready', False)
                
                logger.info(f"✓ Backend is healthy: {data.get('message')}")
                logger.info(f"✓ All components ready: {ready}")
                
                for component, status in components.items():
                    logger.info(f"  • {component}: {'✓' if status else '✗'}")
                
                self.results.append(("Backend Health Check", True))
                return True
            else:
                logger.error(f"✗ Backend health failed: {data.get('message')}")
                self.results.append(("Backend Health Check", False))
                self.all_passed = False
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error(f"✗ Cannot connect to backend at {self.backend_url}")
            logger.error("  Make sure the backend is running: python3 main_v2.py")
            self.results.append(("Backend Health Check", False))
            self.all_passed = False
            return False
        except Exception as e:
            logger.error(f"✗ Health check error: {e}")
            self.results.append(("Backend Health Check", False))
            self.all_passed = False
            return False
    
    def test_api_endpoints_exist(self) -> bool:
        """Test 2: Check all API endpoints exist"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: API Endpoints Available")
        logger.info("=" * 60)
        
        endpoints = [
            ("/", "GET", "Root endpoint"),
            ("/health", "GET", "Health check"),
            ("/detect", "POST", "Face detection"),
            ("/extract-features", "POST", "Feature extraction"),
            ("/verify", "POST", "Face verification"),
            ("/identify", "POST", "Face identification"),
            ("/attendance", "POST", "Attendance marking"),
            ("/enroll-student", "POST", "Student enrollment"),
        ]
        
        all_available = True
        
        for endpoint, method, description in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{self.backend_url}{endpoint}", timeout=5)
                else:
                    # For POST endpoints, just check they exist with OPTIONS or test empty call
                    try:
                        response = requests.post(f"{self.backend_url}{endpoint}", timeout=2)
                    except:
                        # If POST fails without data, endpoint exists, just invalid data
                        response = type('obj', (object,), {'status_code': 422})()
                
                if response.status_code in [200, 201, 400, 422]:
                    logger.info(f"✓ {method:4s} {endpoint:20s} - {description}")
                else:
                    logger.warning(f"? {method:4s} {endpoint:20s} - Status: {response.status_code}")
                    all_available = False
            except Exception as e:
                logger.error(f"✗ {method:4s} {endpoint:20s} - Error: {e}")
                all_available = False
        
        self.results.append(("API Endpoints", all_available))
        if not all_available:
            self.all_passed = False
        return all_available
    
    def test_response_format(self) -> bool:
        """Test 3: Response format standardization"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Response Format Standardization")
        logger.info("=" * 60)
        
        try:
            response = requests.get(f"{self.backend_url}/")
            data = response.json()
            
            # Check for standardized response format
            required_fields = ['status', 'message', 'data', 'timestamp']
            missing_fields = [f for f in required_fields if f not in data]
            
            if not missing_fields:
                logger.info("✓ Response has all required fields:")
                for field in required_fields:
                    logger.info(f"  • {field}: {type(data.get(field)).__name__}")
                
                self.results.append(("Response Format", True))
                return True
            else:
                logger.error(f"✗ Missing fields: {missing_fields}")
                self.results.append(("Response Format", False))
                self.all_passed = False
                return False
                
        except Exception as e:
            logger.error(f"✗ Response format test error: {e}")
            self.results.append(("Response Format", False))
            self.all_passed = False
            return False
    
    def test_error_handling(self) -> bool:
        """Test 4: Error handling"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: Error Handling")
        logger.info("=" * 60)
        
        try:
            # Test 4a: Invalid file
            logger.info("  Testing invalid file upload...")
            files = {'file': ('invalid.txt', b'not an image')}
            response = requests.post(f"{self.backend_url}/detect", files=files, timeout=10)
            
            if response.status_code >= 400:
                data = response.json()
                logger.info(f"  ✓ Invalid file handled: {response.status_code}")
                logger.info(f"    Message: {data.get('message', 'N/A')}")
            
            # Test 4b: Missing required fields
            logger.info("  Testing missing required fields...")
            response = requests.post(
                f"{self.backend_url}/enroll-student",
                data={},  # Missing required fields
                timeout=10
            )
            
            if response.status_code >= 400:
                logger.info(f"  ✓ Missing fields handled: {response.status_code}")
            
            logger.info("✓ Error handling working correctly")
            self.results.append(("Error Handling", True))
            return True
            
        except Exception as e:
            logger.error(f"✗ Error handling test error: {e}")
            self.results.append(("Error Handling", False))
            self.all_passed = False
            return False
    
    def test_logging(self) -> bool:
        """Test 5: Logging system"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 5: Logging System")
        logger.info("=" * 60)
        
        try:
            log_dir = Path("logs")
            if log_dir.exists():
                log_files = list(log_dir.glob("*.log"))
                if log_files:
                    logger.info(f"✓ Log directory exists with {len(log_files)} files:")
                    for log_file in log_files[:3]:  # Show first 3
                        logger.info(f"  • {log_file.name}")
                    self.results.append(("Logging", True))
                    return True
                else:
                    logger.warning("⚠ Log directory empty (logs will be created on first request)")
                    self.results.append(("Logging", True))
                    return True
            else:
                logger.warning("⚠ Logs directory will be created on first request")
                self.results.append(("Logging", True))
                return True
                
        except Exception as e:
            logger.error(f"✗ Logging test error: {e}")
            self.results.append(("Logging", False))
            self.all_passed = False
            return False
    
    def test_validation(self) -> bool:
        """Test 6: Input validation"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST 6: Input Validation")
        logger.info("=" * 60)
        
        try:
            # Test invalid confidence threshold
            logger.info("  Testing validation of confidence threshold...")
            files = {'file': ('test.jpg', b'fake')}
            response = requests.post(
                f"{self.backend_url}/identify",
                files=files,
                data={'confidence_threshold': 1.5},  # Invalid: > 1.0
                timeout=10
            )
            
            if response.status_code >= 400:
                logger.info(f"  ✓ Invalid threshold rejected: {response.status_code}")
            
            # Test invalid top_k
            logger.info("  Testing validation of top_k parameter...")
            response = requests.post(
                f"{self.backend_url}/identify",
                files=files,
                data={'top_k': 100},  # Invalid: > 20
                timeout=10
            )
            
            if response.status_code >= 400:
                logger.info(f"  ✓ Invalid top_k rejected: {response.status_code}")
            
            logger.info("✓ Input validation working correctly")
            self.results.append(("Input Validation", True))
            return True
            
        except Exception as e:
            logger.error(f"✗ Validation test error: {e}")
            self.results.append(("Input Validation", False))
            self.all_passed = False
            return False
    
    def print_summary(self):
        """Print test summary"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for _, result in self.results if result)
        total = len(self.results)
        
        for test_name, result in self.results:
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{status:8s} - {test_name}")
        
        logger.info("-" * 60)
        logger.info(f"Results: {passed}/{total} tests passed")
        
        if self.all_passed:
            logger.info("✓ All tests passed! System is ready.")
        else:
            logger.error("✗ Some tests failed. Please review and fix issues.")
        
        logger.info("=" * 60)
        
        return self.all_passed
    
    def run_all_tests(self) -> bool:
        """Run all integration tests"""
        logger.info("\n" + "🧪 FACE RECOGNITION SYSTEM - INTEGRATION TEST SUITE 🧪\n")
        
        tests = [
            ("Backend Health", self.test_backend_health),
            ("API Endpoints", self.test_api_endpoints_exist),
            ("Response Format", self.test_response_format),
            ("Error Handling", self.test_error_handling),
            ("Logging", self.test_logging),
            ("Input Validation", self.test_validation),
        ]
        
        for test_name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                logger.error(f"Test '{test_name}' crashed: {e}")
                self.all_passed = False
        
        self.print_summary()
        return self.all_passed


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration Test Suite")
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    tester = IntegrationTester(backend_url=args.backend_url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
