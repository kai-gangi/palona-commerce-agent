"""AI Commerce Agent FastAPI Backend Server.

This module provides the main FastAPI application for the AI Commerce Agent.
It handles HTTP requests for chat interactions, product recommendations,
and image-based product search through a unified agent interface.

The server provides endpoints for:
    - General conversation with the AI agent
    - Text-based product recommendations
    - Image-based product search
    - Health checks and API status

Example:
    Run the server using uvicorn:
        $ uvicorn backend.main:app --reload
        
    Or run directly:
        $ python -m backend.main

Attributes:
    app (FastAPI): The main FastAPI application instance.
    agent (CommerceAgent): The AI agent instance for handling requests.
    settings (Settings): Application configuration settings.
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import get_settings
from backend.api.routes import chat, health
import uvicorn

settings = get_settings()

def setup_data():
    """Initialize data by running setup scripts if needed."""
    try:
        products_file = Path("data/products.json")
        vector_db_path = Path("data/vector_store")
        
        if products_file.exists() and vector_db_path.exists():
            print("Data already initialized, skipping setup...")
            return
                    
        result = subprocess.run([
            sys.executable, "scripts/seed_products.py"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            print(f"Error seeding products: {result.stderr}")
            return
            
        result = subprocess.run([
            sys.executable, "scripts/setup_data.py"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            print(f"Error setting up data: {result.stderr}")
            return
            
        print("Data setup completed successfully!")
        
    except Exception as e:
        print(f"Error during data setup: {e}")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Factory function that creates the FastAPI app instance with all
    necessary middleware, routes, and configuration applied.
    
    Returns:
        FastAPI: Configured application instance ready for deployment.
    """
    app = FastAPI(
        title=settings.app_name,
        description="AI-powered commerce agent for product recommendations",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router, prefix="/api")
    app.include_router(health.router, prefix="/api")
    
    return app

app = create_app()

@app.get("/")
async def root():
    """Get API root information and available endpoints."""
    return {
        "message": "AI Commerce Agent API",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "health": "/api/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)