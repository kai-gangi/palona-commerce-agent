"""Prompt templates and text formatting utilities for the AI Commerce Agent.

This module contains the system prompt that defines the agent's personality
and capabilities, as well as utility functions for formatting product data
for display to users and LLM processing.

The system prompt establishes the agent as "ShopBot" with three main capabilities:
    1. General conversation and assistance
    2. Text-based product search and recommendations
    3. Image-based product search and similarity matching

Functions:
    format_products_for_display: Formats product lists for user presentation.
"""

from typing import List, Dict

SYSTEM_PROMPT = """You are a helpful AI shopping assistant for an e-commerce platform. Your name is ShopBot.

Your capabilities:
1. Have general conversations with users
2. Help users find products based on text descriptions
3. Help users find products similar to images they upload

When users ask for product recommendations:
- Use the search_products_by_text function to find relevant items
- Be specific about what you're searching for

When users upload an image:
- Use the search_products_by_image function to find similar products

For general questions about yourself or casual conversation:
- Respond naturally without using any tools

Always be friendly, helpful, and concise. When showing products, highlight their key features and why they match the user's needs."""

def format_products_for_display(products: List[Dict]) -> str:
    """Format product list for LLM presentation to users.
    
    Converts a list of product dictionaries into a formatted string
    that the LLM can present to users in a readable, structured way.
    Includes product names, prices, descriptions, and categories.
    
    Args:
        products: List of product dictionaries containing product information.
            Each product should have 'name', 'price', 'description', and 
            'category' keys.
            
    Returns:
        Formatted string representation of the products, or a "not found"
        message if the product list is empty.
        
    Example:
        >>> products = [
        ...     {"name": "Running Shoes", "price": 99.99, 
        ...      "description": "Comfortable running shoes", "category": "Athletic"}
        ... ]
        >>> result = format_products_for_display(products)
        >>> print(result)
        Here are the products I found:
        
        1. **Running Shoes** - $99.99
           Comfortable running shoes...
           Category: Athletic
    """
    if not products:
        return "No matching products found."
    
    formatted = "Here are the products I found:\n\n"
    for i, product in enumerate(products, 1):
        formatted += f"{i}. **{product['name']}** - ${product['price']}\n"
        formatted += f"   {product['description'][:100]}...\n"
        formatted += f"   Category: {product['category']}\n\n"
    return formatted