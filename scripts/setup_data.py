"""Product catalog initialization and embedding generation script.

This script prepares the AI Commerce Agent's product catalog by loading
product data and generating the necessary embeddings for semantic search.
It must be run once before starting the application to populate the
vector database with searchable product embeddings.

The script performs:
    1. Loading product catalog from JSON file
    2. Generating text embeddings for product descriptions
    3. Creating image embeddings for visual search
    4. Initializing the vector store with all embeddings

Usage:
    Run from the project root directory:
        $ python scripts/setup_data.py
        
    Or with PYTHONPATH set:
        $ PYTHONPATH=. python scripts/setup_data.py

Example:
    $ cd ai-commerce-agent
    $ python scripts/setup_data.py
    Setting up AI Commerce Agent Data
    ==================================================
    Loaded 12 products from catalog
    Creating text embeddings...
    ✓ Created 12 text embeddings
    Setting up vector store...
    ✓ Added 12 products to vector store
    ✓ Setup complete!

Note:
    Requires valid OpenAI API key in .env file for embedding generation.
"""

import json
import os
from pathlib import Path
from backend.services.embedding_service import EmbeddingService
from backend.services.image_service import ImageService
from backend.services.vector_store import VectorStore
from backend.config import get_settings

settings = get_settings()

def load_products():
    """Load product catalog from JSON configuration file.
    
    Reads the product catalog from the configured JSON file path.
    The products file should contain an array of product objects
    with required fields for the commerce agent.
    
    Returns:
        List[Dict]: List of product dictionaries loaded from JSON.
        
    Raises:
        FileNotFoundError: If the products JSON file doesn't exist.
        JSONDecodeError: If the products file contains invalid JSON.
        
    Example:
        >>> products = load_products()
        >>> print(f"Loaded {len(products)} products")
        Loaded 12 products
    """
    with open(settings.products_path, 'r') as f:
        return json.load(f)

def setup_text_embeddings(products):
    """Create and store text embeddings for product search.
    
    Generates text embeddings for all products by combining their name,
    description, and tags into searchable vector representations.
    These embeddings enable semantic search capabilities.
    
    Args:
        products: List of product dictionaries to process.
        
    Returns:
        List[Dict]: Products with added text embeddings.
        
    Example:
        >>> products = load_products()
        >>> products_with_embeddings = setup_text_embeddings(products)
        Creating text embeddings...
        ✓ Created 12 text embeddings
    """
    print("Creating text embeddings...")
    embedding_service = EmbeddingService()
    
    # Create combined text for each product
    texts = [
        f"{p['name']} {p['description']} {' '.join(p['tags'])}"
        for p in products
    ]
    
    embeddings = embedding_service.get_batch_embeddings(texts)
    vector_store = VectorStore()
    vector_store.add_products_text(products, embeddings)
    
    print(f"✓ Created text embeddings for {len(products)} products")

def setup_image_embeddings(products):
    """Create and store image embeddings for all products"""
    print("Creating image embeddings...")
    image_service = ImageService()
    
    embeddings = []
    valid_products = []
    
    for product in products:
        image_path = product['image_path']
        if os.path.exists(image_path):
            try:
                embedding = image_service.encode_image(image_path)
                embeddings.append(embedding)
                valid_products.append(product)
            except Exception as e:
                print(f"Warning: Could not process image for {product['name']}: {e}")
        else:
            print(f"Warning: Image not found for {product['name']}: {image_path}")
    
    if valid_products:
        vector_store = VectorStore()
        vector_store.add_products_images(valid_products, embeddings)
        print(f"✓ Created image embeddings for {len(valid_products)} products")
    else:
        print("No valid product images found")

def main():
    """Main setup function"""
    print("=" * 50)
    print("Setting up AI Commerce Agent Data")
    print("=" * 50)
    
    Path(settings.vector_db_path).mkdir(parents=True, exist_ok=True)
    Path(settings.images_path).mkdir(parents=True, exist_ok=True)
    
    products = load_products()
    print(f"\nLoaded {len(products)} products from catalog")
    
    setup_text_embeddings(products)
    setup_image_embeddings(products)
    
    print("\n" + "=" * 50)
    print("✓ Setup complete! Ready to start the application.")
    print("=" * 50)

if __name__ == "__main__":
    main()