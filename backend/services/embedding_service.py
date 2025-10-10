"""Text embedding service for semantic search and similarity.

This module provides text embedding capabilities using OpenAI's embedding
models. It handles both single text and batch text processing, converting
natural language into high-dimensional vectors for semantic similarity
comparisons.

The service is used for:
    - Converting product descriptions into searchable embeddings
    - Creating query embeddings for semantic search
    - Computing similarity scores between texts

Example:
    >>> from backend.services.embedding_service import EmbeddingService
    >>> service = EmbeddingService()
    >>> embedding = service.get_text_embedding("running shoes")
    >>> print(f"Embedding dimension: {len(embedding)}")
"""

from openai import OpenAI
from backend.config import get_settings
from typing import List
import numpy as np

settings = get_settings()
client = OpenAI(api_key=settings.openai_api_key)

class EmbeddingService:
    """Service for generating and managing text embeddings using OpenAI.
    
    Provides methods for converting text into high-dimensional vector
    representations that capture semantic meaning. Supports both single
    text and batch processing for efficient embedding generation.
    
    Attributes:
        model: The embedding model name configured in settings.
        
    Example:
        >>> service = EmbeddingService()
        >>> embedding = service.get_text_embedding("comfortable running shoes")
        >>> similarity = service.cosine_similarity(embedding1, embedding2)
    """
    
    def __init__(self):
        """Initialize the embedding service with configured model."""
        self.model = settings.embedding_model

    def get_text_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for a single text input.
        
        Converts the input text into a high-dimensional vector representation
        that captures semantic meaning using OpenAI's embedding models.
        
        Args:
            text: Input text to convert to embedding.
            
        Returns:
            List of floats representing the text embedding vector.
            
        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.get_text_embedding("laptop computer")
            >>> print(f"Vector dimension: {len(embedding)}")
        """
        response = client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in a single API call.
        
        More efficient than calling get_text_embedding multiple times
        when processing large batches of text.
        
        Args:
            texts: List of text strings to convert to embeddings.
            
        Returns:
            List of embedding vectors, one for each input text.
            
        Example:
            >>> service = EmbeddingService()
            >>> texts = ["laptop", "smartphone", "tablet"]
            >>> embeddings = service.get_batch_embeddings(texts)
            >>> print(f"Generated {len(embeddings)} embeddings")
        """
        response = client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two embedding vectors.
        
        Computes the cosine of the angle between two vectors, providing
        a similarity score between -1 and 1, where 1 indicates identical
        semantic meaning and 0 indicates no similarity.
        
        Args:
            a: First embedding vector.
            b: Second embedding vector.
            
        Returns:
            Cosine similarity score between -1 and 1.
            
        Example:
            >>> service = EmbeddingService()
            >>> emb1 = service.get_text_embedding("running shoes")
            >>> emb2 = service.get_text_embedding("athletic footwear")
            >>> similarity = service.cosine_similarity(emb1, emb2)
            >>> print(f"Similarity: {similarity:.3f}")
        """
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))