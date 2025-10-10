"""Pydantic models and schemas for the AI Commerce Agent API.

This module defines the data models used throughout the application for
API requests, responses, and internal data structures. All models use
Pydantic for validation, serialization, and automatic API documentation.

Models:
    Product: Represents a product in the commerce catalog.
    ChatMessage: Represents a single message in a conversation.
    ChatRequest: Request model for chat API endpoint.
    ChatResponse: Response model for chat API endpoint.

Example:
    >>> from backend.models.schemas import Product, ChatRequest
    >>> product = Product(
    ...     id="123", name="Running Shoes", category="Athletic",
    ...     description="Comfortable running shoes", price=99.99,
    ...     image_path="/images/shoes.jpg", tags=["running", "sports"]
    ... )
    >>> request = ChatRequest(message="Show me running shoes")
"""

from pydantic import BaseModel, Field
from typing import Optional, List

class Product(BaseModel):
    """Represents a product in the commerce catalog.
    
    Contains all essential product information including identification,
    categorization, pricing, and media. Used for product searches,
    recommendations, and API responses.
    
    Attributes:
        id: Unique identifier for the product.
        name: Display name of the product.
        category: Product category for classification.
        description: Detailed product description.
        price: Product price in USD.
        image_path: Path to product image file.
        tags: List of searchable tags for the product.
        
    Example:
        >>> product = Product(
        ...     id="shoe-001",
        ...     name="Running Shoes",
        ...     category="Athletic",
        ...     description="Lightweight running shoes for daily training",
        ...     price=129.99,
        ...     image_path="/images/running_shoes.jpg",
        ...     tags=["running", "athletic", "lightweight"]
        ... )
    """
    id: str = Field(..., description="Unique product identifier")
    name: str = Field(..., description="Product name")
    category: str = Field(..., description="Product category")
    description: str = Field(..., description="Product description")
    price: float = Field(..., gt=0, description="Product price in USD")
    image_path: str = Field(..., description="Path to product image")
    tags: List[str] = Field(default_factory=list, description="Product tags")

class ChatMessage(BaseModel):
    """Represents a single message in a conversation history.
    
    Used to maintain conversation context between user and agent.
    Follows the standard chat message format with role and content.
    
    Attributes:
        role: Message sender role ('user' or 'assistant').
        content: The text content of the message.
        
    Example:
        >>> user_msg = ChatMessage(role="user", content="Show me laptops")
        >>> agent_msg = ChatMessage(role="assistant", content="Here are some laptops...")
    """
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text content")

class ChatRequest(BaseModel):
    """Request model for the chat API endpoint.
    
    Contains the user's message and optional context for the AI agent.
    Supports both text-only conversations and image-based interactions.
    
    Attributes:
        message: User's text message or query.
        image: Optional base64 encoded image for visual search.
        history: Previous conversation messages for context.
        
    Example:
        Text request:
        >>> request = ChatRequest(message="Show me running shoes")
        
        Image request:
        >>> request = ChatRequest(
        ...     message="Find similar products",
        ...     image="base64_encoded_image_data"
        ... )
    """
    message: str = Field(..., description="User's text message")
    image: Optional[str] = Field(None, description="Base64 encoded image")
    history: List[ChatMessage] = Field(default_factory=list, description="Conversation history")

class ChatResponse(BaseModel):
    """Response model for the chat API endpoint.
    
    Contains the agent's response message and any recommended products.
    Provides transparency about which tools were used to generate the response.
    
    Attributes:
        message: Agent's response text.
        products: Optional list of recommended products.
        tool_used: Optional name of the tool/function used.
        
    Example:
        >>> response = ChatResponse(
        ...     message="Here are some great running shoes for you:",
        ...     products=[product1, product2],
        ...     tool_used="search_products_by_text"
        ... )
    """
    message: str = Field(..., description="Agent's response message")
    products: Optional[List[Product]] = Field(None, description="Recommended products")
    tool_used: Optional[str] = Field(None, description="Tool used for response")


  