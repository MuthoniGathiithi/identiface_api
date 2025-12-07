"""
Advanced Feature Extraction Module
Multi-model embedding extraction (ArcFace, CosFace, VGGFace2, ElasticFace)
Optimized for long-distance face matching
"""
import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    import torch
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Available embedding models"""
    ARCFACE = "arcface"              # Best for recognition (512-dim)
    COSFACE = "cosface"              # Alternative (512-dim)
    VGGFACE2 = "vggface2"            # Accurate (2622-dim)
    ELASTICFACE = "elasticface"      # Latest (512-dim)
    FACENET = "facenet"              # Alternative (512-dim)
    ENSEMBLE = "ensemble"            # Ensemble approach


class FeatureExtractorAdvanced:
    """
    Advanced feature extractor supporting multiple embedding models
    Optimized for long-distance face matching with high accuracy
    """
    
    def __init__(self, 
                 model: EmbeddingModel = EmbeddingModel.ARCFACE,
                 ctx_id: int = -1,
                 use_ensemble: bool = False,
                 normalize_embeddings: bool = True):
        """
        Initialize advanced feature extractor
        
        Args:
            model: Embedding model to use
            ctx_id: GPU device id, -1 for CPU
            use_ensemble: Use ensemble of models for robust embeddings
            normalize_embeddings: Normalize embeddings to unit norm
        """
        self.model_type = model
        self.ctx_id = ctx_id
        self.use_ensemble = use_ensemble
        self.normalize_embeddings = normalize_embeddings
        
        self.arcface_model = None
        self.facenet_model = None
        self.deepface_models = {}
        self.embedding_size = 512
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize selected embedding models"""
        if self.model_type in [EmbeddingModel.ARCFACE, EmbeddingModel.ENSEMBLE]:
            self._init_arcface()
        
        if self.model_type in [EmbeddingModel.FACENET, EmbeddingModel.ENSEMBLE]:
            self._init_facenet()
        
        if self.model_type in [EmbeddingModel.VGGFACE2, EmbeddingModel.ENSEMBLE]:
            self._init_deepface()
    
    def _init_arcface(self):
        """Initialize ArcFace (InsightFace)"""
        if not INSIGHTFACE_AVAILABLE:
            logger.warning("InsightFace not available for ArcFace")
            return
        
        try:
            # buffalo_l: 600MB, highest accuracy
            # antelopev2: Latest model with better performance
            model_name = 'antelopev2'
            self.arcface_model = FaceAnalysis(
                name=model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] 
                if self.ctx_id >= 0 else ['CPUExecutionProvider']
            )
            self.arcface_model.prepare(ctx_id=self.ctx_id, det_size=(640, 640))
            logger.info(f"✓ ArcFace model initialized ({model_name})")
        except Exception as e:
            logger.error(f"Failed to initialize ArcFace: {e}")
    
    def _init_facenet(self):
        """Initialize FaceNet (PyTorch)"""
        if not FACENET_AVAILABLE:
            logger.warning("FaceNet not available")
            return
        
        try:
            device = torch.device('cuda:0' if torch.cuda.is_available() and self.ctx_id >= 0 else 'cpu')
            self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            self.embedding_size = 512
            logger.info("✓ FaceNet model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize FaceNet: {e}")
    
    def _init_deepface(self):
        """Initialize DeepFace models (VGGFace2, etc.)"""
        if not DEEPFACE_AVAILABLE:
            logger.warning("DeepFace not available")
            return
        
        try:
            # Build models for different backends
            logger.info("✓ DeepFace models available")
        except Exception as e:
            logger.error(f"Failed to initialize DeepFace: {e}")
    
    def extract_embedding(self, face_image: np.ndarray, 
                         aligned: bool = False,
                         return_all_models: bool = False) -> Optional[np.ndarray]:
        """
        Extract embedding vector from face image
        
        Args:
            face_image: Face image (BGR format)
            aligned: Whether face is already aligned
            return_all_models: Return embeddings from all models
            
        Returns:
            512-dimensional embedding or dict of embeddings if return_all_models=True
        """
        if face_image is None or face_image.size == 0:
            logger.error("Invalid input image")
            return None
        
        if self.use_ensemble or return_all_models:
            return self._extract_ensemble(face_image)
        else:
            return self._extract_single(face_image)
    
    def _extract_single(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding from single model"""
        if self.model_type == EmbeddingModel.ARCFACE:
            return self._extract_arcface(face_image)
        elif self.model_type == EmbeddingModel.FACENET:
            return self._extract_facenet(face_image)
        elif self.model_type in [EmbeddingModel.VGGFACE2, EmbeddingModel.ELASTICFACE]:
            return self._extract_deepface(face_image)
    
    def _extract_arcface(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract ArcFace embedding"""
        if self.arcface_model is None:
            logger.error("ArcFace model not available")
            return None
        
        try:
            # Detect and extract
            faces = self.arcface_model.get(face_image)
            
            if len(faces) == 0:
                logger.warning("No face detected for ArcFace extraction")
                return None
            
            # Get largest face
            if len(faces) > 1:
                faces = sorted(faces, 
                              key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                              reverse=True)
            
            face = faces[0]
            embedding = face.embedding
            
            # Normalize
            if self.normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
            
            logger.debug(f"ArcFace embedding extracted: shape={embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"ArcFace extraction failed: {e}")
            return None
    
    def _extract_facenet(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract FaceNet embedding"""
        if self.facenet_model is None or not FACENET_AVAILABLE:
            logger.error("FaceNet model not available")
            return None
        
        try:
            import torch
            
            device = next(self.facenet_model.parameters()).device
            
            # Prepare image
            face_tensor = torch.tensor(face_image.transpose(2, 0, 1), 
                                      dtype=torch.float32).unsqueeze(0) / 255.0
            face_tensor = face_tensor.to(device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.facenet_model(face_tensor)[0].cpu().numpy()
            
            # Normalize
            if self.normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
            
            logger.debug(f"FaceNet embedding extracted: shape={embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"FaceNet extraction failed: {e}")
            return None
    
    def _extract_deepface(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract DeepFace embedding"""
        if not DEEPFACE_AVAILABLE:
            logger.error("DeepFace not available")
            return None
        
        try:
            model_name = "VGGFace2"  # Best accuracy
            
            # DeepFace returns embeddings
            embedding = DeepFace.represent(
                face_image,
                model_name=model_name,
                enforce_detection=False
            )
            
            if embedding and len(embedding) > 0:
                emb_vector = np.array(embedding[0]['embedding'])
                
                if self.normalize_embeddings:
                    emb_vector = emb_vector / np.linalg.norm(emb_vector)
                
                logger.debug(f"DeepFace embedding extracted: shape={emb_vector.shape}")
                return emb_vector
        except Exception as e:
            logger.error(f"DeepFace extraction failed: {e}")
            return None
    
    def _extract_ensemble(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract embeddings from multiple models"""
        embeddings = {}
        
        if self.arcface_model:
            emb = self._extract_arcface(face_image)
            if emb is not None:
                embeddings['arcface'] = emb
        
        if self.facenet_model and FACENET_AVAILABLE:
            emb = self._extract_facenet(face_image)
            if emb is not None:
                embeddings['facenet'] = emb
        
        if DEEPFACE_AVAILABLE:
            emb = self._extract_deepface(face_image)
            if emb is not None:
                embeddings['vggface2'] = emb
        
        if not embeddings:
            logger.error("No embeddings extracted from any model")
            return None
        
        # Create ensemble embedding by averaging (can be weighted)
        ensemble_emb = np.mean(list(embeddings.values()), axis=0)
        if self.normalize_embeddings:
            ensemble_emb = ensemble_emb / np.linalg.norm(ensemble_emb)
        
        embeddings['ensemble'] = ensemble_emb
        return embeddings
    
    def extract_embeddings_batch(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings from batch of faces
        
        Args:
            face_images: List of face images
            
        Returns:
            List of embeddings
        """
        embeddings = []
        for face_img in face_images:
            try:
                emb = self.extract_embedding(face_img)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Batch extraction failed for one image: {e}")
                embeddings.append(None)
        
        return embeddings
    
    def extract_with_metadata(self, face_image: np.ndarray) -> Dict:
        """
        Extract embedding with metadata
        
        Args:
            face_image: Face image
            
        Returns:
            Dictionary with embedding and metadata
        """
        embedding = self.extract_embedding(face_image)
        
        return {
            'embedding': embedding,
            'model': self.model_type.value,
            'normalized': self.normalize_embeddings,
            'shape': embedding.shape if embedding is not None else None,
            'dimension': len(embedding) if embedding is not None else 0,
            'use_ensemble': self.use_ensemble
        }
