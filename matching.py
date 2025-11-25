"""
Face Matching Module
Handles face comparison, identification, and verification
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from scipy.spatial.distance import cosine, euclidean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceMatcher:
    """
    Face matching and identification system
    """
    
    def __init__(self, threshold: float = 0.4, metric: str = 'cosine'):
        """
        Initialize face matcher
        
        Args:
            threshold: Similarity threshold for matching
                      For cosine: 0.3-0.5 (higher = stricter)
                      For euclidean: 0.6-1.2 (lower = stricter)
            metric: Distance metric ('cosine' or 'euclidean')
        """
        self.threshold = threshold
        self.metric = metric
        logger.info(f"Initialized FaceMatcher with {metric} metric, threshold={threshold}")
    
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
        
        if self.metric == 'cosine':
            # Cosine distance (0 = identical, 2 = opposite)
            # Convert to similarity: 1 - distance
            distance = cosine(embedding1, embedding2)
            return distance
        elif self.metric == 'euclidean':
            # Euclidean distance
            distance = euclidean(embedding1, embedding2)
            return distance
        else:
            logger.error(f"Unknown metric: {self.metric}")
            return float('inf')
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute similarity score between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (higher = more similar)
        """
        if self.metric == 'cosine':
            # Cosine similarity (1 = identical, -1 = opposite)
            similarity = 1 - self.compute_distance(embedding1, embedding2)
            return similarity
        elif self.metric == 'euclidean':
            # Convert euclidean distance to similarity (inverse)
            distance = self.compute_distance(embedding1, embedding2)
            similarity = 1 / (1 + distance)
            return similarity
        else:
            return 0.0
    
    def verify(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Dict:
        """
        Verify if two embeddings belong to the same person (1:1 matching)
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Verification result dictionary
        """
        if embedding1 is None or embedding2 is None:
            return {
                'match': False,
                'similarity': 0.0,
                'distance': float('inf'),
                'threshold': self.threshold,
                'confidence': 0.0
            }
        
        distance = self.compute_distance(embedding1, embedding2)
        similarity = self.compute_similarity(embedding1, embedding2)
        
        # Determine match based on metric
        if self.metric == 'cosine':
            # For cosine similarity, higher is better
            match = similarity >= self.threshold
            confidence = abs(similarity - self.threshold)
        else:
            # For euclidean distance, lower is better
            match = distance <= self.threshold
            confidence = abs(self.threshold - distance) / self.threshold
        
        return {
            'match': match,
            'similarity': float(similarity),
            'distance': float(distance),
            'threshold': self.threshold,
            'confidence': float(confidence)
        }
    
    def identify(self, query_embedding: np.ndarray, 
                gallery_embeddings: List[np.ndarray],
                gallery_ids: List[str],
                top_k: int = 5) -> List[Dict]:
        """
        Identify a face from a gallery (1:N matching)
        
        Args:
            query_embedding: Query face embedding
            gallery_embeddings: List of gallery face embeddings
            gallery_ids: List of corresponding identity IDs
            top_k: Number of top matches to return
            
        Returns:
            List of top matches with scores
        """
        if query_embedding is None or not gallery_embeddings:
            return []
        
        matches = []
        for idx, (gallery_emb, identity_id) in enumerate(zip(gallery_embeddings, gallery_ids)):
            if gallery_emb is None:
                continue
            
            distance = self.compute_distance(query_embedding, gallery_emb)
            similarity = self.compute_similarity(query_embedding, gallery_emb)
            
            # Check if it passes threshold
            if self.metric == 'cosine':
                passes_threshold = similarity >= self.threshold
            else:
                passes_threshold = distance <= self.threshold
            
            matches.append({
                'identity_id': identity_id,
                'gallery_index': idx,
                'similarity': float(similarity),
                'distance': float(distance),
                'passes_threshold': passes_threshold
            })
        
        # Sort by similarity (descending) or distance (ascending)
        if self.metric == 'cosine':
            matches.sort(key=lambda x: x['similarity'], reverse=True)
        else:
            matches.sort(key=lambda x: x['distance'])
        
        # Return top K matches
        top_matches = matches[:top_k]
        
        logger.info(f"Identification: {len([m for m in top_matches if m['passes_threshold']])} matches above threshold")
        return top_matches
    
    def find_best_match(self, query_embedding: np.ndarray,
                       gallery_embeddings: List[np.ndarray],
                       gallery_ids: List[str]) -> Optional[Dict]:
        """
        Find the best matching identity
        
        Args:
            query_embedding: Query face embedding
            gallery_embeddings: List of gallery embeddings
            gallery_ids: List of identity IDs
            
        Returns:
            Best match dictionary or None if no match above threshold
        """
        matches = self.identify(query_embedding, gallery_embeddings, gallery_ids, top_k=1)
        
        if not matches:
            return None
        
        best_match = matches[0]
        
        if best_match['passes_threshold']:
            return best_match
        else:
            logger.info("No match found above threshold")
            return None
    
    def batch_verify(self, embeddings1: List[np.ndarray], 
                    embeddings2: List[np.ndarray]) -> List[Dict]:
        """
        Verify multiple face pairs in batch
        
        Args:
            embeddings1: List of first embeddings
            embeddings2: List of second embeddings
            
        Returns:
            List of verification results
        """
        if len(embeddings1) != len(embeddings2):
            logger.error("Embedding lists must have same length")
            return []
        
        results = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            result = self.verify(emb1, emb2)
            results.append(result)
        
        return results
    
    def deduplicate_faces(self, embeddings: List[np.ndarray], 
                         ids: List[str],
                         similarity_threshold: float = 0.9) -> List[int]:
        """
        Find duplicate faces in a collection
        
        Args:
            embeddings: List of face embeddings
            ids: List of identity IDs
            similarity_threshold: Threshold for considering faces as duplicates
            
        Returns:
            List of indices to keep (duplicates removed)
        """
        if not embeddings:
            return []
        
        keep_indices = []
        removed_indices = set()
        
        for i in range(len(embeddings)):
            if i in removed_indices:
                continue
            
            keep_indices.append(i)
            
            # Check for duplicates
            for j in range(i + 1, len(embeddings)):
                if j in removed_indices:
                    continue
                
                similarity = self.compute_similarity(embeddings[i], embeddings[j])
                
                if similarity >= similarity_threshold:
                    removed_indices.add(j)
                    logger.debug(f"Found duplicate: {ids[i]} and {ids[j]} (similarity: {similarity:.3f})")
        
        logger.info(f"Deduplication: kept {len(keep_indices)} out of {len(embeddings)} faces")
        return keep_indices
    
    def cluster_faces(self, embeddings: List[np.ndarray], 
                     distance_threshold: float = 0.6) -> List[List[int]]:
        """
        Cluster similar faces together
        
        Args:
            embeddings: List of face embeddings
            distance_threshold: Distance threshold for clustering
            
        Returns:
            List of clusters (each cluster is a list of indices)
        """
        if not embeddings:
            return []
        
        n = len(embeddings)
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Start new cluster
            cluster = [i]
            visited[i] = True
            
            # Find all faces similar to this one
            for j in range(i + 1, n):
                if visited[j]:
                    continue
                
                distance = self.compute_distance(embeddings[i], embeddings[j])
                
                if distance <= distance_threshold:
                    cluster.append(j)
                    visited[j] = True
            
            clusters.append(cluster)
        
        logger.info(f"Clustered {n} faces into {len(clusters)} groups")
        return clusters
    
    def update_threshold(self, new_threshold: float):
        """Update the matching threshold"""
        self.threshold = new_threshold
        logger.info(f"Updated threshold to {new_threshold}")
    
    def get_optimal_threshold(self, genuine_pairs: List[Tuple[np.ndarray, np.ndarray]],
                            impostor_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Calculate optimal threshold based on genuine and impostor pairs
        
        Args:
            genuine_pairs: List of (embedding1, embedding2) pairs from same person
            impostor_pairs: List of (embedding1, embedding2) pairs from different people
            
        Returns:
            Optimal threshold value
        """
        genuine_scores = []
        for emb1, emb2 in genuine_pairs:
            similarity = self.compute_similarity(emb1, emb2)
            genuine_scores.append(similarity)
        
        impostor_scores = []
        for emb1, emb2 in impostor_pairs:
            similarity = self.compute_similarity(emb1, emb2)
            impostor_scores.append(similarity)
        
        if not genuine_scores or not impostor_scores:
            logger.warning("Insufficient data for threshold optimization")
            return self.threshold
        
        # Find threshold that minimizes error
        # Simple approach: midpoint between mean genuine and mean impostor scores
        mean_genuine = np.mean(genuine_scores)
        mean_impostor = np.mean(impostor_scores)
        optimal_threshold = (mean_genuine + mean_impostor) / 2
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f} (genuine: {mean_genuine:.3f}, impostor: {mean_impostor:.3f})")
        return float(optimal_threshold)
