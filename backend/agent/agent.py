"""AI Commerce Agent for handling product recommendations and search.

This module implements the main CommerceAgent class that serves as the
intelligent interface for product recommendations, image-based search,
and general commerce-related conversations. The agent uses OpenAI's
GPT models and function calling for tool integration.

The agent handles three main use cases:
    1. General conversation with the agent
    2. Text-based product recommendations 
    3. Image-based product search 

Example:
    Basic agent usage:
        >>> from backend.agent.agent import CommerceAgent
        >>> agent = CommerceAgent()
        >>> response, products, tool = agent.chat("Show me running shoes")
        >>> print(f"Found {len(products)} products")
        
    Image-based search:
        >>> response, products, tool = agent.chat(
        ...     "Find similar products",
        ...     image_base64="base64_encoded_image"
        ... )

Classes:
    CommerceAgent: Main agent class for handling commerce interactions.
"""

from openai import OpenAI
from backend.config import get_settings
from backend.agent.tools import TOOLS, TOOL_MAP
from backend.agent.prompts import SYSTEM_PROMPT, format_products_for_display 
from backend.models.schemas import ChatMessage
from typing import List, Dict, Optional, Tuple, Generator
import json

settings = get_settings()

class CommerceAgent:
    """AI-powered commerce agent for product recommendations and search.
    
    The CommerceAgent serves as the main interface for all commerce-related
    AI interactions using OpenAI's GPT models with function calling to 
    integrate with product search and recommendation tools.
    
    Attributes:
        settings: Application configuration settings.
        client: OpenAI client for API communication.
        model: OpenAI model name for chat completions.
    
    Example:
        >>> agent = CommerceAgent()
        >>> response, products, tool = agent.chat("Show me sports shoes")
        >>> print(f"Agent used {tool} and found {len(products)} products")
    """
    
    def __init__(self):
        """Initialize the commerce agent with OpenAI.
        
        Sets up the OpenAI client using the configured API key and model.
        This simplified version only supports OpenAI as the LLM provider.
        
        Raises:
            ValueError: If OpenAI API key is not configured.
        """
        self.settings = settings
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required but not configured")
        
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
    
    def _format_history(self, history: List[ChatMessage]) -> List[Dict]:
        """Convert ChatMessage list to API format.
        
        Transforms the internal ChatMessage objects into the dictionary
        format expected by the LLM APIs.
        
        Args:
            history: List of ChatMessage objects from previous conversation.
            
        Returns:
            List of dictionaries with 'role' and 'content' keys suitable
            for LLM API calls.
        """
        return [{"role": msg.role, "content": msg.content} for msg in history]
    
    def chat(
        self,
        message: str,
        history: List[ChatMessage] = [],
        image_base64: Optional[str] = None,
        stream: bool = False
    ) -> Tuple[str, Optional[List[Dict]], Optional[str]]:
        """Main chat interface using OpenAI for all interactions.
        
        This is the primary entry point for all agent interactions using
        OpenAI's chat completions API with function calling support.
        
        Args:
            message: User's text message or query.
            history: Previous conversation history for context.
            image_base64: Base64 encoded image for image-based search.
            stream: Whether to stream the response or return complete response.
            
        Returns:
            Tuple containing:
                - str: Agent's response message
                - Optional[List[Dict]]: Product recommendations if applicable  
                - Optional[str]: Name of tool used for the response
                
        Example:
            >>> agent = CommerceAgent()
            >>> response, products, tool = agent.chat("Show me laptops")
            >>> print(f"Found {len(products or [])} products using {tool}")
        """
        if stream:
            return self._chat_stream(message, history, image_base64)
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._format_history(history))

        if image_base64:
            user_message = {
                "role": "user", 
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        else:
            user_message = {"role": "user", "content": message}
        
        messages.append(user_message)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            return response_message.content, None, None
        
        messages.append(response_message)

        products = None
        tool_used = None

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "search_products_by_image" and image_base64:
                function_args["image_base64"] = image_base64

            if function_name in TOOL_MAP:
                tool_used = function_name
                function_response = TOOL_MAP[function_name](**function_args)
                products = function_response

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": format_products_for_display(function_response)
                })

        final_response = self.client.chat.completions.create(
            model=self.model, 
            messages=messages
        )

        return final_response.choices[0].message.content, products, tool_used

    def _chat_stream(
        self,
        message: str,
        history: List[ChatMessage] = [],
        image_base64: Optional[str] = None
    ) -> Generator[Dict, None, None]:
        """Streaming version of chat method."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._format_history(history))

        if image_base64:
            user_message = {
                "role": "user", 
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            }
        else:
            user_message = {"role": "user", "content": message}
        
        messages.append(user_message)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            stream=False  # First call to check for tool calls
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        products = None
        tool_used = None

        # Handle tool calls first if present
        if tool_calls:
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "search_products_by_image" and image_base64:
                    function_args["image_base64"] = image_base64

                if function_name in TOOL_MAP:
                    tool_used = function_name
                    function_response = TOOL_MAP[function_name](**function_args)
                    products = function_response

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": format_products_for_display(function_response)
                    })

            # Stream the final response
            stream_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )

            for chunk in stream_response:
                if chunk.choices[0].delta.content:
                    yield {
                        "type": "content",
                        "content": chunk.choices[0].delta.content,
                        "products": None,
                        "tool_used": None
                    }

            # Send final data with products
            yield {
                "type": "complete",
                "content": "",
                "products": products,
                "tool_used": tool_used
            }
        else:
            # No tool calls, stream directly
            stream_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )

            for chunk in stream_response:
                if chunk.choices[0].delta.content:
                    yield {
                        "type": "content", 
                        "content": chunk.choices[0].delta.content,
                        "products": None,
                        "tool_used": None
                    }

            yield {
                "type": "complete",
                "content": "",
                "products": None,
                "tool_used": None
            }
