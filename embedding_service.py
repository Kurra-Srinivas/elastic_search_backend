"""Embedding service using sentence-transformers - loads model once"""
from sentence_transformers import SentenceTransformer
import logging
from typing import List

logger = logging.getLogger(__name__)

# Global model instance - loaded once
_model: SentenceTransformer = None
_model_name = "all-MiniLM-L6-v2"


def get_embedding_model() -> SentenceTransformer:
    """Get or load the embedding model (singleton pattern)"""
    global _model
    
    if _model is None:
        logger.info(f"Loading embedding model: {_model_name}")
        try:
            _model = SentenceTransformer(_model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    return _model


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    if not texts:
        return []
    
    model = get_embedding_model()
    
    try:
        # Generate embeddings in batch
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Convert numpy arrays to lists
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text
    
    Args:
        text: Text string to embed
        
    Returns:
        Embedding vector as a list of floats
    """
    if not text:
        return []
    
    embeddings = generate_embeddings([text])
    return embeddings[0] if embeddings else []

