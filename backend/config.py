"""Configuration management for AI Commerce Agent.

This module provides application configuration using Pydantic Settings.
Configuration values can be set via environment variables or .env file.
The module supports multiple LLM providers and embedding services.

Example:
    Basic usage:
        >>> from backend.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.app_name)
        AI Commerce Agent
        
    Environment variable override:
        >>> import os
        >>> os.environ["LLM_MODEL"] = "gpt-4"
        >>> settings = get_settings()
        >>> print(settings.llm_model)
        gpt-4

Note:
    API keys should be set via environment variables for security.
    Default values in code are for development only.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings and configuration.
    
    Manages configuration for the AI Commerce Agent including API keys,
    model settings, file paths, and application parameters. Values can
    be overridden via environment variables or .env file.
    
    Attributes:
        openai_api_key (str): OpenAI API key for LLM and embedding services.
        llm_model (str): Specific OpenAI model name to use for chat completions.
        embedding_model (str): OpenAI model name for text embeddings.
        app_name (str): Application name for API documentation.
        debug (bool): Enable debug mode with verbose logging.
        products_path (str): Relative path to products JSON file.
        images_path (str): Relative path to product images directory.
        vector_db_path (str): Relative path to vector database storage.
    
    Example:
        >>> settings = Settings()
        >>> print(f"Using OpenAI with model {settings.llm_model}")
        Using OpenAI with model gpt-4o-mini
    """
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    app_name: str = "AI Commerce Agent"
    debug: bool = True

    products_path: str = "data/products.json"
    images_path: str = "data/product_images"
    vector_db_path: str = "data/vector_db"

    class Config:
        env_file = ".env"
        extra = "allow"

@lru_cache()
def get_settings():
    """Get cached application settings instance."""
    return Settings()