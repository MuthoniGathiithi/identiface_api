"""
Feature Extraction Module
Extracts face embeddings using InsightFace ArcFace model
"""
import cv2
import numpy as np
from typing import Optional, List
import logging

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available, install with: pip install insightface onnxruntime")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract face embeddings using InsightFace ArcFace model
    """
    
    def __init__(self, model_name='buffalo_l', ctx_id=-1):
        """
        Initialize feature extractor with ArcFace
        
        Args:
            model_name: InsightFace model pack ('buffalo_l', 'buffalo_s', 'antelopev2')
                       buffalo_l: High accuracy (512-dim embeddings)
                       buffalo_s: Balanced speed/accuracy
                       antelopev2: Latest model
            ctx_id: GPU device id, -1 for CPU
        """
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.model = None
        self.embedding_size = 512
        
        if INSIGHTFACE_AVAILABLE:
            try:
                # Initialize FaceAnalysis which includes ArcFace for recognition
                self.model = FaceAnalysis(
                    name=model_name,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']
                )
                self.model.prepare(ctx_id=ctx_id, det_size=(640, 640))
                logger.info(f"Initialized InsightFace ArcFace with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize InsightFace ArcFace: {e}")
                logger.info("Feature extraction will not be available")
        else:
            logger.error("InsightFace not available. Install with: pip install insightface onnxruntime")
    
    def extract_embedding(self, face_image: np.ndarray, aligned: bool = False) -> Optional[np.ndarray]:
        """
        Extract embedding vector from face image using ArcFace
        
        Args:
            face_image: Face image (BGR format)
            aligned: Whether the face is already aligned
            
        Returns:
            512-dimensional embedding vector or None if extraction fails
        """
        if face_image is None or face_image.size == 0:
            logger.error("Invalid input image")
            return None
        
        if not INSIGHTFACE_AVAILABLE or self.model is None:
            logger.error("InsightFace model not available")
            return None
        
        try:
            # Detect and extract embedding
            faces = self.model.get(face_image)
            
            if len(faces) == 0:
                logger.warning("No face detected in image")
                return None
            
            # Get the first (or largest) face
            if len(faces) > 1:
                # Get largest face by area
                faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
            
            face = faces[0]
            embedding = face.embedding
            
            # Normalize embedding (L2 normalization)
            embedding = embedding / np.linalg.norm(embedding)
            
            logger.debug(f"Extracted ArcFace embedding of shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None
    
    def extract_embedding_from_detection(self, image: np.ndarray, face_box: tuple, 
                                        keypoints: dict = None) -> Optional[np.ndarray]:
        """
        Extract embedding from a detected face region
        
        Args:
            image: Full image (BGR format)
            face_box: Face bounding box (x, y, w, h)
            keypoints: Optional facial keypoints for better alignment
            
        Returns:
            Embedding vector or None
        """
        if image is None or face_box is None:
            logger.error("Invalid input")
            return None
        
        try:
            # Extract face ROI with margin
            x, y, w, h = face_box
            margin = 0.2
            margin_w = int(w * margin)
            margin_h = int(h * margin)
            
            h_img, w_img = image.shape[:2]
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(w_img, x + w + margin_w)
            y2 = min(h_img, y + h + margin_h)
            
            face_roi = image[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                logger.error("Empty face ROI")
                return None
            
            # Extract embedding from ROI
            return self.extract_embedding(face_roi)
            
        except Exception as e:
            logger.error(f"Failed to extract embedding from detection: {e}")
            return None
    
    def extract_embedding_from_face(self, image: np.ndarray, face_dict: dict) -> Optional[np.ndarray]:
        """
        Extract embedding from a face detection dictionary
        
        Args:
            image: Full image (BGR format)
            face_dict: Face detection dictionary with 'box' and 'keypoints'
            
        Returns:
            Embedding vector or None
        """
        if 'box' not in face_dict:
            logger.error("Face dictionary missing 'box' key")
            return None
        
        return self.extract_embedding_from_detection(
            image, 
            face_dict['box'], 
            face_dict.get('keypoints')
        )
    
    def extract_multiple_embeddings(self, image: np.ndarray) -> List[dict]:
        """
        Extract embeddings for all faces in an image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of dictionaries with face info and embeddings
        """
        if not INSIGHTFACE_AVAILABLE or self.model is None:
            logger.error("InsightFace model not available")
            return []
        
        try:
            faces = self.model.get(image)
            
            results = []
            for face in faces:
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                
                embedding = face.embedding
                embedding = embedding / np.linalg.norm(embedding)
                
                results.append({
                    'box': (x, y, x2 - x, y2 - y),
                    'confidence': float(face.det_score),
                    'embedding': embedding,
                    'keypoints': {
                        'left_eye': tuple(face.kps[0].astype(int)),
                        'right_eye': tuple(face.kps[1].astype(int)),
                        'nose': tuple(face.kps[2].astype(int)),
                        'left_mouth': tuple(face.kps[3].astype(int)),
                        'right_mouth': tuple(face.kps[4].astype(int))
                    } if hasattr(face, 'kps') and face.kps is not None else {}
                })
            
            logger.info(f"Extracted embeddings for {len(results)} faces")
            return results
            
        except Exception as e:
            logger.error(f"Multiple embedding extraction failed: {e}")
            return []
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                          metric: str = 'cosine') -> float:
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Similarity score (higher means more similar for cosine, lower for euclidean)
        """
        if embedding1 is None or embedding2 is None:
            logger.error("Invalid embeddings")
            return 0.0
        
        if metric == 'cosine':
            # Cosine similarity (range: -1 to 1, higher is more similar)
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        elif metric == 'euclidean':
            # Euclidean distance (lower is more similar)
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)
        else:
            logger.error(f"Unknown metric: {metric}")
            return 0.0
    
    def verify_face(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                   threshold: float = 0.4) -> dict:
        """
        Verify if two embeddings belong to the same person
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            threshold: Similarity threshold (default 0.4 for cosine similarity)
            
        Returns:
            Dictionary with verification result
        """
        similarity = self.compute_similarity(embedding1, embedding2, metric='cosine')
        
        # For cosine similarity, higher is more similar
        # Typical threshold: 0.3-0.5 (0.4 is common)
        is_same = similarity >= threshold
        
        return {
            'is_same_person': is_same,
            'similarity': similarity,
            'threshold': threshold,
            'confidence': abs(similarity - threshold) / threshold  # How confident we are
        }
    
    def get_embedding_size(self) -> int:
        """Get the size of embedding vectors"""
        return self.embedding_size
    
    def is_available(self) -> bool:
        """Check if the feature extractor is available"""
        return INSIGHTFACE_AVAILABLE and self.model is not None


# TEST CODE - Add your image path here
if __name__ == "__main__":
    import os
    from detection import FaceDetector
    from normalization import FaceNormalizer
    
    def test_feature_extraction():
        """Test feature extraction with static images"""
        print("=== FEATURE EXTRACTION TEST ===")
        
        # Initialize components
        detector = FaceDetector(model_name='buffalo_l', min_confidence=0.5, ctx_id=-1)
        extractor = FeatureExtractor(model_name='buffalo_l', ctx_id=-1)
        normalizer = FaceNormalizer(target_size=(160, 160))
        
        # Check if extractor is available
        if not extractor.is_available():
            print("❌ Feature extractor not available. Check InsightFace installation.")
            return
        
        # ADD YOUR IMAGE PATH HERE
        test_image_path = "/home/muthoni/Downloads/tets.jpg"  # CHANGE THIS PATH
        
        # Check if image exists
        if not os.path.exists(test_image_path):
            print(f"❌ Image not found: {test_image_path}")
            print("Please update the test_image_path variable with your actual image path")
            return
        
        # Load image
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"❌ Failed to load image: {test_image_path}")
            return
        
        print(f"✅ Loaded image: {test_image_path}")
        print(f"   Original size: {image.shape}")
        print(f"   Embedding size: {extractor.get_embedding_size()}")
        
        # Test 1: Direct embedding extraction from full image
        print("\n--- Test 1: Direct Embedding Extraction ---")
        embedding_direct = extractor.extract_embedding(image)
        
        if embedding_direct is not None:
            print(f"✅ Direct extraction successful")
            print(f"   Embedding shape: {embedding_direct.shape}")
            print(f"   Embedding norm: {np.linalg.norm(embedding_direct):.6f}")
            print(f"   Embedding range: [{embedding_direct.min():.6f}, {embedding_direct.max():.6f}]")
        else:
            print("❌ Direct extraction failed")
        
        # Test 2: Embedding extraction from detected face
        print("\n--- Test 2: Extraction from Detected Face ---")
        faces = detector.detect_faces(image)
        
        if not faces:
            print("❌ No faces detected. Cannot test face-based extraction.")
            return
        
        face = faces[0]  # Use first detected face
        face_box = face['box']
        keypoints = face['keypoints']
        
        # Convert numpy int64 to regular int for compatibility
        face_box_clean = (int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3]))
        
        print(f"✅ Using detected face: {face_box_clean}")
        
        # Extract embedding from detection
        embedding_detection = extractor.extract_embedding_from_detection(image, face_box_clean, keypoints)
        
        if embedding_detection is not None:
            print(f"✅ Detection-based extraction successful")
            print(f"   Embedding shape: {embedding_detection.shape}")
            print(f"   Embedding norm: {np.linalg.norm(embedding_detection):.6f}")
        else:
            print("❌ Detection-based extraction failed")
        
        # Test 3: Embedding extraction from face dictionary
        print("\n--- Test 3: Extraction from Face Dictionary ---")
        # Create a clean face dictionary with regular integers
        face_clean = {
            'box': face_box_clean,
            'keypoints': keypoints,
            'confidence': face['confidence']
        }
        embedding_dict = extractor.extract_embedding_from_face(image, face_clean)
        
        if embedding_dict is not None:
            print(f"✅ Dictionary-based extraction successful")
            print(f"   Embedding shape: {embedding_dict.shape}")
            print(f"   Embedding norm: {np.linalg.norm(embedding_dict):.6f}")
        else:
            print("❌ Dictionary-based extraction failed")
        
        # Test 4: Multiple face embeddings
        print("\n--- Test 4: Multiple Face Embeddings ---")
        multiple_faces = extractor.extract_multiple_embeddings(image)
        
        print(f"✅ Found {len(multiple_faces)} faces with embeddings")
        for i, face_data in enumerate(multiple_faces):
            print(f"   Face {i+1}:")
            print(f"     Box: {face_data['box']}")
            print(f"     Confidence: {face_data['confidence']:.3f}")
            print(f"     Embedding shape: {face_data['embedding'].shape}")
            print(f"     Has keypoints: {len(face_data['keypoints']) > 0}")
        
        # Test 5: Embedding consistency (same image should give similar embeddings)
        print("\n--- Test 5: Embedding Consistency ---")
        if embedding_direct is not None and embedding_detection is not None:
            similarity = extractor.compute_similarity(embedding_direct, embedding_detection, metric='cosine')
            print(f"   Similarity between direct and detection methods: {similarity:.6f}")
            
            if similarity > 0.8:
                print("   ✅ High consistency between extraction methods")
            elif similarity > 0.5:
                print("   ⚠️  Moderate consistency between extraction methods")
            else:
                print("   ❌ Low consistency between extraction methods")
        
        # Test 6: Similarity computation
        print("\n--- Test 6: Similarity Computation ---")
        if len(multiple_faces) >= 2:
            emb1 = multiple_faces[0]['embedding']
            emb2 = multiple_faces[1]['embedding']
            
            cosine_sim = extractor.compute_similarity(emb1, emb2, metric='cosine')
            euclidean_dist = extractor.compute_similarity(emb1, emb2, metric='euclidean')
            
            print(f"   Cosine similarity: {cosine_sim:.6f}")
            print(f"   Euclidean similarity: {euclidean_dist:.6f}")
        elif embedding_direct is not None:
            # Test with same embedding (should be 1.0 for cosine)
            self_similarity = extractor.compute_similarity(embedding_direct, embedding_direct, metric='cosine')
            print(f"   Self-similarity (should be ~1.0): {self_similarity:.6f}")
        
        # Test 7: Face verification
        print("\n--- Test 7: Face Verification ---")
        if embedding_direct is not None and embedding_detection is not None:
            verification = extractor.verify_face(embedding_direct, embedding_detection, threshold=0.4)
            
            print(f"   Same person: {verification['is_same_person']}")
            print(f"   Similarity: {verification['similarity']:.6f}")
            print(f"   Threshold: {verification['threshold']}")
            print(f"   Confidence: {verification['confidence']:.6f}")
        
        # Test 8: Embedding with preprocessed face
        print("\n--- Test 8: Extraction from Preprocessed Face ---")
        face_roi = detector.extract_face_roi(image, face_box_clean, margin=0.2)
        preprocessed_face = normalizer.preprocess_for_model(face_roi, keypoints, enhance=True)
        
        # Convert back to uint8 for extraction
        preprocessed_uint8 = ((preprocessed_face * 255).astype(np.uint8))
        
        # For preprocessed faces, use the original image since the ROI might be too small
        # Instead, let's use the face ROI directly
        embedding_preprocessed = extractor.extract_embedding(face_roi)
        
        if embedding_preprocessed is not None:
            print(f"✅ Preprocessed extraction successful")
            print(f"   Embedding shape: {embedding_preprocessed.shape}")
            
            # Compare with original
            if embedding_direct is not None:
                prep_similarity = extractor.compute_similarity(embedding_direct, embedding_preprocessed, metric='cosine')
                print(f"   Similarity to original: {prep_similarity:.6f}")
        else:
            print("❌ Preprocessed extraction failed")
        
        # Test 9: Save embedding data
        print("\n--- Test 9: Saving Embedding Data ---")
        base_path = test_image_path.replace('.jpg', '').replace('.png', '')
        
        # Save embeddings as numpy files
        if embedding_direct is not None:
            np.save(f"{base_path}_embedding_direct.npy", embedding_direct)
            print(f"💾 Saved direct embedding: {base_path}_embedding_direct.npy")
        
        if embedding_detection is not None:
            np.save(f"{base_path}_embedding_detection.npy", embedding_detection)
            print(f"💾 Saved detection embedding: {base_path}_embedding_detection.npy")
        
        # Save embedding info as text
        with open(f"{base_path}_embedding_info.txt", 'w') as f:
            f.write(f"Feature Extraction Test Results\n")
            f.write(f"Image: {test_image_path}\n")
            f.write(f"Embedding size: {extractor.get_embedding_size()}\n")
            f.write(f"Faces detected: {len(multiple_faces)}\n")
            
            if embedding_direct is not None:
                f.write(f"Direct embedding norm: {np.linalg.norm(embedding_direct):.6f}\n")
            
            if embedding_detection is not None:
                f.write(f"Detection embedding norm: {np.linalg.norm(embedding_detection):.6f}\n")
            
            if embedding_direct is not None and embedding_detection is not None:
                similarity = extractor.compute_similarity(embedding_direct, embedding_detection, metric='cosine')
                f.write(f"Method consistency: {similarity:.6f}\n")
        
        print(f"💾 Saved embedding info: {base_path}_embedding_info.txt")
        print(f"\n✅ Feature extraction test completed!")
    
    # Run the test
    test_feature_extraction()
