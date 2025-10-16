"""AI agent tools for product search and recommendations.

This module provides the function tools that the AI agent can call to
perform product searches and recommendations. It includes both text-based
and image-based search capabilities using embedding and vector similarity.

The module defines:
    - Tool functions for product search
    - OpenAI function calling schemas
    - Tool mapping for function dispatch

Tools:
    search_products_by_text: Find products using natural language queries.
    search_products_by_image: Find products similar to uploaded images.

Example:
    >>> from backend.agent.tools import search_products_by_text
    >>> products = search_products_by_text("running shoes", n_results=3)
    >>> print(f"Found {len(products)} products")
"""

from typing import List, Dict
from backend.services.embedding_service import EmbeddingService
from backend.services.image_service import ImageService
from backend.services.vector_store import VectorStore

embedding_service = EmbeddingService()
image_service = ImageService()
vector_store = VectorStore()  

def search_products_by_text(query: str, n_results: int = 5) -> List[Dict]:
    """Search for products based on text description using semantic similarity.
    
    Uses text embeddings to find products that semantically match the
    provided query. This enables natural language product search that
    understands intent and context beyond simple keyword matching.
    
    Args:
        query: The text query describing desired products (e.g., 
            "comfortable running shoes for marathons").
        n_results: Maximum number of products to return.
        
    Returns:
        List of product dictionaries matching the query, ordered by
        relevance score from the vector similarity search.
        
    Example:
        >>> products = search_products_by_text("wireless headphones")
        >>> for product in products:
        ...     print(f"{product['name']} - ${product['price']}")
    """
    query_embedding = embedding_service.get_text_embedding(query)
    products = vector_store.search_text(query_embedding, n_results)
    return products

def search_products_by_image(image_base64: str, n_results: int = 5) -> List[Dict]:
    """Search for products similar to an uploaded image using computer vision.
    
    Uses image embeddings to find products visually similar to the
    uploaded image. This enables "search by image" functionality where
    users can upload photos to find similar products.
    
    Args:
        image_base64: Base64 encoded image data (JPEG, PNG supported).
        n_results: Maximum number of similar products to return.
        
    Returns:
        List of product dictionaries visually similar to the input image,
        ordered by visual similarity score.
        
    Example:
        >>> with open("shoe_image.jpg", "rb") as f:
        ...     img_base64 = base64.b64encode(f.read()).decode()
        >>> products = search_products_by_image(img_base64, n_results=3)
        >>> for product in products:
        ...     print(f"Similar: {product['name']}")
    """
    query_embedding = image_service.encode_image_from_base64(image_base64)
    products = vector_store.search_image(query_embedding, n_results)
    return products

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products_by_text",
            "description": "Search for products in the catalog based on a text description. Use this when the user asks for product recommendations or wants to find specific items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The product search query based on user's description"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of products to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products_by_image",
            "description": "Search for products similar to an uploaded image. Use this when the user uploads an image and wants to find similar products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64 encoded image data"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of products to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["image_base64"]
            }
        }
    }
]

TOOL_MAP = {
    "search_products_by_text": search_products_by_text,
    "search_products_by_image": search_products_by_image
}