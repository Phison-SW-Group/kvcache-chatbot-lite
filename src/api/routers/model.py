"""
Model deployment router
Handles model startup, restart, and shutdown operations
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from services.model import model_server

router = APIRouter(prefix="/model", tags=["model"])


class ModelUpRequest(BaseModel):
    """Request model for model startup operations"""
    model_name: Optional[str] = None
    config: Optional[dict] = None


class ModelResponse(BaseModel):
    """Response model for model operations"""
    status: str
    message: str
    timestamp: datetime
    model_name: Optional[str] = None
    pid: Optional[int] = None
    port: Optional[int] = None
    command: Optional[str] = None
    details: Optional[dict] = None


@router.post("/up/without_reset", response_model=ModelResponse)
async def start_model_without_reset(request: ModelUpRequest):
    """
    Start model without resetting existing configuration
    """
    try:
        result = model_server.up(reset=False)
        
        return ModelResponse(
            status=result["status"],
            message=result["message"],
            timestamp=datetime.now(),
            model_name=request.model_name,
            pid=result.get("pid"),
            port=result.get("port"),
            command=result.get("command"),
            details=result.get("details")
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start model: {str(e)}"
        )


@router.post("/up/reset", response_model=ModelResponse)
async def start_model_with_reset(request: ModelUpRequest):
    """
    Start model with reset (restart with new configuration)
    """
    try:
        result = model_server.up(reset=True)
        
        return ModelResponse(
            status=result["status"],
            message=result["message"],
            timestamp=datetime.now(),
            model_name=request.model_name,
            pid=result.get("pid"),
            port=result.get("port"),
            command=result.get("command"),
            details=result.get("details")
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restart model: {str(e)}"
        )


@router.post("/down", response_model=ModelResponse)
async def stop_model():
    """
    Stop the currently running model
    """
    try:
        result = model_server.down()
        
        return ModelResponse(
            status=result["status"],
            message=result["message"],
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop model: {str(e)}"
        )


@router.get("/status")
async def get_model_status():
    """
    Get current model status
    """
    try:
        result = model_server.get_status()
        result["last_updated"] = datetime.now()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )
