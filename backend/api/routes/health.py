"""Health check endpoints for service monitoring."""

from fastapi import APIRouter, HTTPException
from backend.config import get_settings
import time

router = APIRouter()

@router.get("/health")
async def health_check():
    """Simple health check with basic service validation.
    
    Performs lightweight checks on critical services to ensure
    the application is functioning properly. Tests configuration
    and basic connectivity without expensive operations.
    
    Returns:
        dict: Health status with timestamp and basic service checks.
        
    Raises:
        HTTPException: 503 if critical services are unavailable.
    """
    start_time = time.time()
    
    try:
        # Check 1: Configuration is loaded
        settings = get_settings()
        config_ok = bool(settings.openai_api_key and settings.llm_model)
        
        # Check 2: Can import critical services (basic validation)
        try:
            from backend.agent.agent import CommerceAgent
            from backend.services.vector_store import VectorStore
            services_ok = True
        except ImportError:
            services_ok = False
        
        response_time = int((time.time() - start_time) * 1000)
        
        if config_ok and services_ok:
            return {
                "status": "healthy",
                "timestamp": int(time.time()),
                "response_time_ms": response_time,
                "checks": {
                    "config": "ok",
                    "services": "ok"
                }
            }
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "timestamp": int(time.time()),
                    "response_time_ms": response_time,
                    "checks": {
                        "config": "ok" if config_ok else "failed",
                        "services": "ok" if services_ok else "failed"
                    }
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        response_time = int((time.time() - start_time) * 1000)
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "timestamp": int(time.time()),
                "response_time_ms": response_time,
                "error": str(e)
            }
        )