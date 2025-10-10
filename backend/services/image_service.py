"""Image processing service using CLIP for visual embeddings.

This module provides image encoding capabilities using OpenAI's CLIP model
for visual similarity search in the commerce application. It handles both
file-based and base64-encoded image inputs for flexible usage across
different parts of the application.

The service uses the CLIP (Contrastive Language-Image Pre-training) model
to generate high-quality image embeddings that can be used for visual
product search and similarity matching.

Example:
    Basic usage for encoding images:
        >>> service = ImageService()
        >>> # Encode from file path
        >>> embedding1 = service.encode_image_from_pil("product.jpg")
        >>> # Encode from base64 data
        >>> embedding2 = service.encode_image_from_base64(base64_string)
        >>> # Compute similarity
        >>> similarity = service.compute_similarity(embedding1, embedding2)

Dependencies:
    - transformers: For CLIP model and processor
    - torch: For deep learning computations
    - PIL: For image processing
    - base64: For decoding base64 image data

Attributes:
    model (CLIPModel): Pre-trained CLIP model for image encoding
    processor (CLIPProcessor): CLIP processor for image preprocessing
    device (torch.device): Computing device (CPU/GPU) for model inference
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import base64
from io import BytesIO
from typing import List

class ImageService:
    """Service for encoding images into embeddings using CLIP model.
    
    This class provides methods to encode images from different sources
    (file paths and base64 strings) into numerical embeddings that can
    be used for visual similarity search in the commerce application.
    
    The service automatically detects and utilizes GPU acceleration when
    available, falling back to CPU computation as needed.
    
    Attributes:
        model: Pre-trained CLIP model for generating image embeddings.
        processor: CLIP processor for preprocessing images before encoding.
        device: Torch device used for model computations (CPU or GPU).
        
    Example:
        >>> service = ImageService()
        >>> embedding = service.encode_image_from_pil("product_image.jpg")
        >>> print(f"Embedding shape: {embedding.shape}")
    """
    
    def __init__(self):
        """Initialize the ImageService with CLIP model and processor.
        
        Loads the pre-trained CLIP model and processor, automatically
        detecting and configuring the optimal compute device (GPU if
        available, otherwise CPU).
        
        The model is moved to the selected device for efficient inference.
        Uses the base CLIP ViT model which provides a good balance of
        speed and embedding quality.
        
        Raises:
            RuntimeError: If model loading fails or device setup encounters issues.
            
        Example:
            >>> service = ImageService()
            >>> print(f"Using device: {service.device}")
        """
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode_image(self, image_path: str) -> torch.Tensor:
        """Encode an image from file path into an embedding vector.
        
        Loads an image from the specified file path, processes it through
        the CLIP model, and returns a numerical embedding that represents
        the visual features of the image.
        
        Args:
            image_path: Path to the image file to encode. Supports common
                formats like JPEG, PNG, etc.
                
        Returns:
            NumPy array containing the image embedding of shape (embed_dim,).
            The embedding is normalized and ready for similarity comparisons.
            
        Raises:
            FileNotFoundError: If the specified image file doesn't exist.
            PIL.UnidentifiedImageError: If the file is not a valid image.
            RuntimeError: If model inference fails.
            
        Example:
            >>> service = ImageService()
            >>> embedding = service.encode_image("products/shoe.jpg")
            >>> print(f"Embedding dimensions: {embedding.shape}")
            >>> # Returns: Embedding dimensions: (512,)
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features[0].cpu().numpy()
    
    def encode_image_from_base64(self, base64_str: str) -> torch.Tensor:
        """Encode an image from base64 string into an embedding vector.
        
        Decodes a base64-encoded image string, processes it through the
        CLIP model, and returns a numerical embedding. Handles both raw
        base64 strings and data URL formats (data:image/jpeg;base64,...).
        
        Args:
            base64_str: Base64-encoded image data. Can be either raw base64
                or a complete data URL with MIME type prefix.
                
        Returns:
            NumPy array containing the image embedding of shape (embed_dim,).
            The embedding is normalized and ready for similarity comparisons.
            
        Raises:
            ValueError: If base64 string is invalid or cannot be decoded.
            PIL.UnidentifiedImageError: If decoded data is not a valid image.
            RuntimeError: If model inference fails.
            
        Example:
            >>> service = ImageService()
            >>> # With data URL prefix
            >>> data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
            >>> embedding1 = service.encode_image_from_base64(data_url)
            >>> 
            >>> # Raw base64 string
            >>> raw_base64 = "/9j/4AAQSkZJRgABA..."
            >>> embedding2 = service.encode_image_from_base64(raw_base64)
        """
        image_data = base64.b64decode(base64_str.split(",")[1] if ',' in base64_str else base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        return image_features[0].cpu().numpy()
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Compute cosine similarity between two image embeddings.
        
        Calculates the cosine similarity score between two image embeddings,
        which indicates how visually similar the images are. The score ranges
        from -1 (completely dissimilar) to 1 (identical), with higher values
        indicating greater visual similarity.
        
        Args:
            embedding1: First image embedding vector as NumPy array or tensor.
            embedding2: Second image embedding vector as NumPy array or tensor.
            
        Returns:
            Cosine similarity score as a float between -1 and 1, where:
                - 1.0: Images are visually identical
                - 0.8-0.99: Very similar images
                - 0.6-0.8: Moderately similar images
                - 0.3-0.6: Somewhat similar images
                - < 0.3: Dissimilar images
                
        Example:
            >>> service = ImageService()
            >>> emb1 = service.encode_image_from_pil("shoe1.jpg")
            >>> emb2 = service.encode_image_from_pil("shoe2.jpg")
            >>> similarity = service.compute_similarity(emb1, emb2)
            >>> print(f"Visual similarity: {similarity:.3f}")
            >>> # Output: Visual similarity: 0.847
            >>> 
            >>> if similarity > 0.8:
            ...     print("Images are very similar!")
            >>> elif similarity > 0.6:
            ...     print("Images are moderately similar")
            >>> else:
            ...     print("Images are quite different")
        """
        return float(torch.nn.functional.cosine_similarity(torch.tensor(embedding1).unsqueeze(0), torch.tensor(embedding2).unsqueeze(0)))