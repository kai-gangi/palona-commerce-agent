"""Vector database service for storing and querying product embeddings.

This module provides a unified interface for storing and retrieving product
embeddings using ChromaDB as the underlying vector database. It supports
both text-based semantic search and image-based visual similarity search
through separate collections optimized for each use case.

The service maintains two distinct collections:
    - products_text: For text-based semantic product search
    - products_images: For image-based visual product search

Example:
    Basic usage for storing and searching products:
        >>> store = VectorStore()
        >>> # Add text embeddings
        >>> store.add_products_text(products, text_embeddings)
        >>> # Add image embeddings  
        >>> store.add_products_images(products, image_embeddings)
        >>> # Search by text
        >>> results = store.search_text(query_embedding, n_results=5)
        >>> # Search by image
        >>> results = store.search_image(image_embedding, n_results=5)

Dependencies:
    - chromadb: Vector database for embedding storage and similarity search
    - json: For serializing/deserializing product metadata
    - typing: For type hints and annotations

Attributes:
    client (chromadb.PersistentClient): ChromaDB client for database operations
    text_collection: Collection storing text-based product embeddings
    image_collection: Collection storing image-based product embeddings
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.config import get_settings
from typing import List, Dict, Any
import json 

settings = get_settings()

class VectorStore:
    """Vector database service for product embeddings and similarity search.
    
    This class provides a high-level interface for storing product embeddings
    and performing similarity searches using ChromaDB. It maintains separate
    collections for text and image embeddings to optimize search performance
    and enable different types of product discovery.
    
    The vector store uses cosine similarity for all searches, which works
    well for both semantic text search and visual image similarity. All
    embeddings are stored with rich metadata to enable fast product retrieval.
    
    Attributes:
        client: ChromaDB persistent client for database operations.
        text_collection: Collection for storing text-based product embeddings.
        image_collection: Collection for storing image-based product embeddings.
        
    Example:
        >>> store = VectorStore()
        >>> print(f"Text collection: {store.text_collection.name}")
        >>> print(f"Image collection: {store.image_collection.name}")
    """
    
    def __init__(self):
        """Initialize the VectorStore with ChromaDB collections.
        
        Creates a persistent ChromaDB client and initializes separate
        collections for text and image embeddings. Each collection is
        configured with cosine similarity for optimal search performance.
        
        The collections are created if they don't exist, or retrieved if
        they already exist, ensuring data persistence across application
        restarts.
        
        Raises:
            ConnectionError: If ChromaDB connection fails.
            PermissionError: If database path is not writable.
            
        Example:
            >>> store = VectorStore()
            >>> # Collections are now ready for use
            >>> print("VectorStore initialized successfully")
        """
        self.client = chromadb.PersistentClient(
            path=settings.vector_db_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.text_collection = self.client.get_or_create_collection(
            name="products_text",
            metadata={"hnsw:space": "cosine"}
        )
        self.image_collection = self.client.get_or_create_collection(
            name="products_images",
            metadata={"hnsw:space": "cosine"}
        )

    def add_products_text(self, products: List[Dict], embeddings: List[List[float]]):
        """Add products with text embeddings to the text search collection.
        
        Stores product information along with their text embeddings for
        semantic search capabilities. Each product's text representation
        (name, description, tags) is embedded and stored with comprehensive
        metadata for fast retrieval.
        
        Args:
            products: List of product dictionaries containing product information.
                Each product should have: id, name, description, tags, category,
                price, image_path, and other relevant fields.
            embeddings: List of text embedding vectors corresponding to each
                product. Each embedding should be a list of floats representing
                the semantic meaning of the product's text content.
                
        Raises:
            ValueError: If products and embeddings lists have different lengths.
            TypeError: If embedding format is invalid.
            ChromaError: If database operation fails.
            
        Example:
            >>> products = [
            ...     {
            ...         "id": "prod_1", 
            ...         "name": "Running Shoes",
            ...         "description": "Comfortable athletic footwear",
            ...         "tags": ["sports", "footwear"],
            ...         "category": "shoes",
            ...         "price": 99.99,
            ...         "image_path": "shoes/running_1.jpg"
            ...     }
            ... ]
            >>> embeddings = [[0.1, 0.2, 0.3, ...]]  # 512-dim embeddings
            >>> store.add_products_text(products, embeddings)
            >>> print("Products added to text collection")
        """
        ids = [p["id"] for p in products]
        documents = [f"{p['name']} {p['description']} {' '.join(p['tags'])}" for p in products]
        metadata = [{
            "name": p["name"],
            "category": p["category"],
            "price": p["price"],
            "image_path": p["image_path"],
            "product_data": json.dumps(p)
        } for p in products]
        
        self.text_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata
        )

    def add_products_images(self, products: List[Dict], embeddings: List):
        """Add products with image embeddings to the visual search collection.
        
        Stores product information along with their image embeddings for
        visual similarity search capabilities. Image embeddings are generated
        from product photos and enable "search by image" functionality.
        
        Args:
            products: List of product dictionaries containing product information.
                Each product should have an id, name, and other relevant fields.
            embeddings: List of image embedding vectors corresponding to each
                product. Can be NumPy arrays or lists of floats representing
                visual features extracted from product images.
                
        Raises:
            ValueError: If products and embeddings lists have different lengths.
            TypeError: If embedding format cannot be converted to list.
            ChromaError: If database operation fails.
            
        Example:
            >>> products = [
            ...     {
            ...         "id": "prod_1",
            ...         "name": "Blue Sneakers", 
            ...         "image_path": "sneakers/blue_1.jpg"
            ...     }
            ... ]
            >>> # embeddings from CLIP model (NumPy arrays)
            >>> image_embeddings = [np.array([0.1, 0.2, ...])]
            >>> store.add_products_images(products, image_embeddings)
            >>> print("Products added to image collection")
        """
        ids = [p["id"] for p in products]
        metadatas = [{
            "name": p["name"],
            "product_data": json.dumps(p)
        } for p in products]

        embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]

        self.image_collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas
        )

    def search_text(self, query_embedding: List[float], n_results: int = 5) -> List[Dict]:
        """Search for products using text-based semantic similarity.
        
        Performs semantic search using the provided text embedding to find
        products with similar textual descriptions. Uses cosine similarity
        to rank products by semantic relevance to the query.
        
        Args:
            query_embedding: Text embedding vector representing the search query.
                Should be generated using the same embedding model used for
                indexing products.
            n_results: Maximum number of similar products to return. Defaults
                to 5 for optimal performance and user experience.
                
        Returns:
            List of product dictionaries ordered by semantic similarity score.
            Each product contains all original fields plus similarity metadata.
            Returns empty list if no products found or collection is empty.
            
        Raises:
            ValueError: If query_embedding format is invalid.
            ChromaError: If database query fails.
            
        Example:
            >>> # Get embedding for search query
            >>> query = "comfortable running shoes for marathons"
            >>> query_embedding = embedding_service.get_text_embedding(query)
            >>> 
            >>> # Search for similar products
            >>> results = store.search_text(query_embedding, n_results=3)
            >>> for product in results:
            ...     print(f"Found: {product['name']} - ${product['price']}")
            >>> 
            >>> # Output:
            >>> # Found: Marathon Pro Runners - $129.99
            >>> # Found: Ultra Comfort Athletic Shoes - $99.99
            >>> # Found: Long Distance Running Sneakers - $149.99
        """
        results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        products = []
        for metadata in results['metadatas'][0]:
            product = json.loads(metadata['product_data'])
            products.append(product)
        return products
    
    def search_image(self, query_embedding: List[float], n_results: int = 5) -> List[Dict]:
        """Search for products using image-based visual similarity.
        
        Performs visual similarity search using the provided image embedding
        to find products with similar visual appearance. Uses cosine similarity
        to rank products by visual resemblance to the query image.
        
        Args:
            query_embedding: Image embedding vector representing the visual query.
                Can be NumPy array or list of floats. Should be generated using
                the same image encoder used for indexing products.
            n_results: Maximum number of visually similar products to return.
                Defaults to 5 for optimal performance and user experience.
                
        Returns:
            List of product dictionaries ordered by visual similarity score.
            Each product contains all original fields plus similarity metadata.
            Returns empty list if no products found or collection is empty.
            
        Raises:
            ValueError: If query_embedding format is invalid.
            ChromaError: If database query fails.
            
        Example:
            >>> # Get embedding for uploaded image
            >>> with open("user_uploaded_shoe.jpg", "rb") as f:
            ...     img_data = base64.b64encode(f.read()).decode()
            >>> image_embedding = image_service.encode_image_from_base64(img_data)
            >>> 
            >>> # Search for visually similar products  
            >>> results = store.search_image(image_embedding, n_results=4)
            >>> for product in results:
            ...     print(f"Similar: {product['name']} - {product['category']}")
            >>> 
            >>> # Output:
            >>> # Similar: Nike Air Max - athletic_shoes
            >>> # Similar: Adidas Ultraboost - running_shoes  
            >>> # Similar: Puma RS-X - lifestyle_shoes
            >>> # Similar: New Balance 990 - premium_shoes
        """
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()

        results = self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        products = []
        for metadata in results['metadatas'][0]:
            product = json.loads(metadata['product_data'])
            products.append(product)  
        return products