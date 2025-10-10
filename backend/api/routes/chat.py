"""Chat endpoints for AI agent interactions."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
import logging
import json

from backend.models.schemas import ChatRequest, ChatResponse
from backend.agent.agent import CommerceAgent

router = APIRouter()
agent = CommerceAgent()
logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for AI agent interactions.
    
    Handles all types of interactions with the AI commerce agent including:
    - General conversation and questions about capabilities
    - Text-based product recommendations using natural language
    - Image-based product search and similar product finding
    
    Args:
        request (ChatRequest): The chat request containing message, history, and optional image.
    
    Returns:
        ChatResponse: Agent response with message, products, and tool information.
    
    Raises:
        HTTPException: 500 error if agent processing fails.
    """
    try:
        response_text, products, tool_used = agent.chat(
            message=request.message,
            history=request.history,
            image_base64=request.image
        )
        
        return ChatResponse(
            message=response_text,
            products=products,
            tool_used=tool_used
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint for real-time AI agent responses.
    
    Provides server-sent events (SSE) streaming for real-time response delivery.
    This improves user experience for longer queries by showing partial responses
    as they are generated.
    
    Args:
        request (ChatRequest): The chat request containing message, history, and optional image.
    
    Returns:
        StreamingResponse: SSE stream with incremental response chunks.
    
    Raises:
        HTTPException: 500 error if agent processing fails.
    """
    try:
        def generate():
            try:
                for chunk in agent._chat_stream(
                    message=request.message,
                    history=request.history,
                    image_base64=request.image
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}")
                error_chunk = {
                    "type": "error",
                    "content": "An error occurred while processing your request",
                    "products": None,
                    "tool_used": None
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    
    except Exception as e:
        logger.error(f"Error in chat stream endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )
