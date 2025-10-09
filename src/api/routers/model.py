"""
Model deployment router
Handles model startup, restart, and shutdown operations
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from pathlib import Path

from services.model import model_server
from services.document_manager import document_manager

router = APIRouter(prefix="/model", tags=["model"])

# llama-server creates this subdirectory under cache_path
MAESTRO_CACHE_SUBDIR = "maestro_phison"


def get_prefix_tree_path(cache_path: Path) -> Path:
    """
    Get the expected path for prefix_tree.bin
    Returns the path in maestro_phison subdirectory
    """
    return cache_path / MAESTRO_CACHE_SUBDIR / "prefix_tree.bin"

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
    This will also clear all uploaded documents since KV cache is reset
    """
    try:
        # Clear all documents when model is reset
        cleared_count = document_manager.clear_all_documents()
        
        result = model_server.up(reset=True)
        
        # Update message to include document clearing info
        if cleared_count > 0:
            result["message"] += f" (Cleared {cleared_count} document(s) due to model reset)"
        
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
        result = await model_server.down()
        
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


@router.get("/check_cache_existence")
async def check_cache_existence():
    """
    Check if prefix_tree.bin exists in cache directory
    llama-server creates a maestro_phison subdirectory under the cache path
    """
    try:
        # TODO HARDCODED: Directly check R:\maestro_phison\prefix_tree.bin
        prefix_tree_file = Path("R:/maestro_phison/prefix_tree.bin")
        
        if prefix_tree_file.exists():
            return {
                "cache_path": "R:\\",
                "prefix_tree_exists": True,
                "prefix_tree_path": str(prefix_tree_file)
            }
        else:
            return {
                "cache_path": "R:\\",
                "prefix_tree_exists": False,
                "prefix_tree_path": str(prefix_tree_file)
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check cache: {str(e)}"
        )