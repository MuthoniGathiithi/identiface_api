"""
Advanced Face Matching Module
Multi-metric matching with adaptive thresholds and long-distance optimization
Supports: Cosine, Euclidean, Manhattan, and angular distance metrics
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from scipy.spatial.distance import cosine, euclidean, cityblock
from scipy.spatial.distance import pdist, squareform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistanceMetric:
    """Available distance metrics"""
    COSINE = "cosine"           # Recommended for high-dimensional embeddings
    EUCLIDEAN = "euclidean"     # Euclidean distance
    MANHATTAN = "manhattan"     # Manhattan distance
    ANGULAR = "angular"         # Angular distance (for normalized vectors)
    MAHALANOBIS = "mahalanobis" # Mahalanobis distance (requires covariance)


class FaceMatcherAdvanced:
    """
    Advanced face matching system with multiple distance metrics,
    adaptive thresholding, and long-distance optimization
    """
    
    def __init__(self, 
                 metric: DistanceMetric = DistanceMetric.COSINE,
                 threshold: float = 0.4,
                 use_adaptive_threshold: bool = True,
                 enable_quality_weighting: bool = True):
        """
        Initialize advanced face matcher
        
        Args:
            metric: Distance metric to use
            threshold: Base similarity threshold
            use_adaptive_threshold: Adapt threshold based on gallery size
            enable_quality_weighting: Weight matches by face quality
        """
        self.metric = metric
        self.base_threshold = threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.enable_quality_weighting = enable_quality_weighting
        
        # Adaptive threshold ranges based on metric
        self.threshold_ranges = {
            DistanceMetric.COSINE: (0.25, 0.60),        # Higher is stricter
            DistanceMetric.EUCLIDEAN: (0.5, 1.5),       # Lower is stricter
            DistanceMetric.MANHATTAN: (2.0, 10.0),      # Lower is stricter
            DistanceMetric.ANGULAR: (5.0, 30.0),        # Lower is stricter
        }
        
        logger.info(f"Initialized FaceMatcherAdvanced with {metric.value} metric, "
                   f"threshold={threshold}, adaptive={use_adaptive_threshold}")
    
    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute distance between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Distance score
        """
        if embedding1 is None or embedding2 is None:
            return float('inf')
        
        try:
            if self.metric == DistanceMetric.COSINE:
                return float(cosine(embedding1, embedding2))
            elif self.metric == DistanceMetric.EUCLIDEAN:
                return float(euclidean(embedding1, embedding2))
            elif self.metric == DistanceMetric.MANHATTAN:
                return float(cityblock(embedding1, embedding2))
            elif self.metric == DistanceMetric.ANGULAR:
                # Angular distance in degrees
                cos_sim = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                )
                cos_sim = np.clip(cos_sim, -1, 1)
                return float(np.arccos(cos_sim) * 180 / np.pi)
            else:
                logger.error(f"Unknown metric: {self.metric}")
                return float('inf')
        except Exception as e:
            logger.error(f"Distance computation failed: {e}")
            return float('inf')
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute similarity score between two embeddings (0-1, higher is more similar)
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        try:
            if self.metric == DistanceMetric.COSINE:
                # Cosine similarity: 1 - distance, but distance already in [0, 2]
                distance = self.compute_distance(embedding1, embedding2)
                return float(1 - distance / 2)
            elif self.metric == DistanceMetric.EUCLIDEAN:
                distance = self.compute_distance(embedding1, embedding2)
                return float(1 / (1 + distance))
            elif self.metric == DistanceMetric.MANHATTAN:
                distance = self.compute_distance(embedding1, embedding2)
                return float(1 / (1 + distance / 100))
            elif self.metric == DistanceMetric.ANGULAR:
                distance = self.compute_distance(embedding1, embedding2)
                return float(1 - distance / 180)
            else:
                return 0.0
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0
    
    def _get_adaptive_threshold(self, gallery_size: int) -> float:
        """
        Get adaptive threshold based on gallery size
        
        Larger galleries need stricter thresholds to reduce false positives
        
        Args:
            gallery_size: Number of identities in gallery
            
        Returns:
            Adjusted threshold
        """
        if not self.use_adaptive_threshold:
            return self.base_threshold
        
        min_thresh, max_thresh = self.threshold_ranges.get(
            self.metric, (self.base_threshold * 0.8, self.base_threshold * 1.2)
        )
        
        # Scale based on gallery size
        # Small gallery: more lenient, Large gallery: stricter
        scale_factor = min(1.5, 1.0 + np.log1p(gallery_size) / 10)
        
        if self.metric in [DistanceMetric.COSINE, DistanceMetric.ANGULAR]:
            # For these metrics, higher threshold is stricter
            threshold = min(max_thresh, self.base_threshold * scale_factor)
        else:
            # For these metrics, lower threshold is stricter
            threshold = max(min_thresh, self.base_threshold / scale_factor)
        
        logger.debug(f"Adaptive threshold for gallery_size={gallery_size}: {threshold:.4f}")
        return threshold
    
    def verify(self, embedding1: np.ndarray, embedding2: np.ndarray, 
               quality1: float = 1.0, quality2: float = 1.0) -> Dict:
        """
        Verify if two embeddings belong to the same person (1:1 matching)
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            quality1: Quality score of first face (0-1)
            quality2: Quality score of second face (0-1)
            
        Returns:
            Verification result dictionary
        """
        if embedding1 is None or embedding2 is None:
            return {
                'match': False,
                'similarity': 0.0,
                'distance': float('inf'),
                'threshold': self.base_threshold,
                'confidence': 0.0,
                'model': self.metric.value
            }
        
        distance = self.compute_distance(embedding1, embedding2)
        similarity = self.compute_similarity(embedding1, embedding2)
        
        threshold = self.base_threshold
        
        # Determine match based on metric
        if self.metric in [DistanceMetric.COSINE, DistanceMetric.ANGULAR]:
            match = similarity >= threshold
            confidence = max(0, similarity - threshold) / (1 - threshold) if threshold < 1 else 0
        else:
            match = distance <= threshold
            confidence = max(0, (threshold - distance) / threshold)
        
        # Apply quality weighting if enabled
        if self.enable_quality_weighting:
            quality_weight = (quality1 + quality2) / 2
            confidence *= quality_weight
        
        return {
            'match': match,
            'similarity': float(similarity),
            'distance': float(distance),
            'threshold': threshold,
            'confidence': float(np.clip(confidence, 0, 1)),
            'model': self.metric.value,
            'quality_weighted': self.enable_quality_weighting
        }
    
    def identify(self, query_embedding: np.ndarray,
                gallery_embeddings: List[np.ndarray],
                gallery_ids: List[str],
                gallery_qualities: Optional[List[float]] = None,
                top_k: int = 5) -> List[Dict]:
        """
        Identify a face from gallery (1:N matching)
        
        Args:
            query_embedding: Query face embedding
            gallery_embeddings: List of gallery face embeddings
            gallery_ids: List of corresponding identity IDs
            gallery_qualities: Optional quality scores for gallery faces
            top_k: Number of top matches to return
            
        Returns:
            List of top matches with scores
        """
        if query_embedding is None or not gallery_embeddings:
            return []
        
        # Use adaptive threshold
        threshold = self._get_adaptive_threshold(len(gallery_embeddings))
        
        matches = []
        
        for idx, (gallery_emb, identity_id) in enumerate(zip(gallery_embeddings, gallery_ids)):
            if gallery_emb is None:
                continue
            
            distance = self.compute_distance(query_embedding, gallery_emb)
            similarity = self.compute_similarity(query_embedding, gallery_emb)
            
            # Check if passes threshold
            if self.metric in [DistanceMetric.COSINE, DistanceMetric.ANGULAR]:
                passes_threshold = similarity >= threshold
            else:
                passes_threshold = distance <= threshold
            
            # Get quality
            quality = gallery_qualities[idx] if gallery_qualities else 1.0
            
            match_data = {
                'identity_id': identity_id,
                'gallery_index': idx,
                'similarity': float(similarity),
                'distance': float(distance),
                'passes_threshold': passes_threshold,
                'confidence': float(similarity) if passes_threshold else 0.0,
                'quality': float(quality)
            }
            
            # Apply quality weighting
            if self.enable_quality_weighting and quality < 1.0:
                match_data['weighted_confidence'] = match_data['confidence'] * quality
            else:
                match_data['weighted_confidence'] = match_data['confidence']
            
            matches.append(match_data)
        
        # Sort by weighted confidence
        matches.sort(key=lambda x: x['weighted_confidence'], reverse=True)
        
        # Return top K
        top_matches = matches[:top_k]
        
        logger.info(f"Identification: {len([m for m in top_matches if m['passes_threshold']])} "
                   f"matches above threshold from {len(gallery_embeddings)} gallery faces")
        return top_matches
    
    def find_best_match(self, query_embedding: np.ndarray,
                       gallery_embeddings: List[np.ndarray],
                       gallery_ids: List[str],
                       gallery_qualities: Optional[List[float]] = None) -> Optional[Dict]:
        """
        Find the best matching identity
        
        Args:
            query_embedding: Query face embedding
            gallery_embeddings: List of gallery embeddings
            gallery_ids: List of identity IDs
            gallery_qualities: Optional quality scores
            
        Returns:
            Best match dictionary or None if no match above threshold
        """
        matches = self.identify(
            query_embedding,
            gallery_embeddings,
            gallery_ids,
            gallery_qualities,
            top_k=1
        )
        
        if not matches:
            return None
        
        best_match = matches[0]
        
        if best_match['passes_threshold']:
            return best_match
        else:
            return None
    
    def batch_identify(self, query_embeddings: List[np.ndarray],
                      gallery_embeddings: List[np.ndarray],
                      gallery_ids: List[str]) -> List[List[Dict]]:
        """
        Identify multiple query faces against gallery
        
        Args:
            query_embeddings: List of query embeddings
            gallery_embeddings: List of gallery embeddings
            gallery_ids: List of identity IDs
            
        Returns:
            List of identification results for each query
        """
        results = []
        for query_emb in query_embeddings:
            matches = self.identify(query_emb, gallery_embeddings, gallery_ids)
            results.append(matches)
        
        return results
    
    def compute_distance_matrix(self, embeddings1: List[np.ndarray],
                               embeddings2: List[np.ndarray]) -> np.ndarray:
        """
        Compute pairwise distance matrix between two sets of embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Distance matrix of shape (len(embeddings1), len(embeddings2))
        """
        matrix = np.zeros((len(embeddings1), len(embeddings2)))
        
        for i, emb1 in enumerate(embeddings1):
            for j, emb2 in enumerate(embeddings2):
                matrix[i, j] = self.compute_distance(emb1, emb2)
        
        return matrix
    
    def compute_similarity_matrix(self, embeddings1: List[np.ndarray],
                                 embeddings2: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Compute pairwise similarity matrix
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set (if None, compute within embeddings1)
            
        Returns:
            Similarity matrix
        """
        if embeddings2 is None:
            embeddings2 = embeddings1
        
        matrix = np.zeros((len(embeddings1), len(embeddings2)))
        
        for i, emb1 in enumerate(embeddings1):
            for j, emb2 in enumerate(embeddings2):
                matrix[i, j] = self.compute_similarity(emb1, emb2)
        
        return matrix
