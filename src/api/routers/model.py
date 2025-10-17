"""
Model deployment router
Handles model startup, restart, and shutdown operations
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from pathlib import Path

from services.model import model_server
from services.document_manager import document_manager
from services.tokenizer_manager import tokenizer_manager
from config import settings

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
    serving_name: Optional[str] = None
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


class ModelInfo(BaseModel):
    """Model information"""
    model_name_or_path: str
    serving_name: str


@router.get("/list")
async def list_models() -> List[ModelInfo]:
    """
    Get list of all configured models from settings
    """
    try:
        models = []
        for model_config in settings.models:
            models.append(ModelInfo(
                model_name_or_path=model_config.model_name_or_path,
                serving_name=model_config.serving_name
            ))
        return models
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )


@router.post("/up/without_reset", response_model=ModelResponse)
async def start_model_without_reset(request: ModelUpRequest):
    """
    Start model without resetting existing configuration
    Also loads tokenizer for the selected model
    """
    try:
        # Load tokenizer first
        tokenizer_result = tokenizer_manager.load_tokenizer(request.serving_name)
        print(f"🔧 Tokenizer: {tokenizer_result['message']}")

        # Start model server
        result = model_server.up(reset=False, serving_name=request.serving_name)

        model_message = result["message"]

        # Format tokenizer status
        tokenizer_message = ""
        if tokenizer_result["status"] == "loaded":
            tokenizer_message = f"✅ Tokenizer: {tokenizer_result['tokenizer_name']}"
        elif tokenizer_result["status"] == "not_loaded":
            tokenizer_message = "⚠️  Tokenizer: Not configured"
        else:
            tokenizer_message = f"❌ Tokenizer: Failed to load"

        # Combine messages with line breaks
        combined_message = '\n'.join([model_message, tokenizer_message])

        return ModelResponse(
            status=result["status"],
            message=combined_message,
            timestamp=datetime.now(),
            model_name=request.serving_name,
            pid=result.get("pid"),
            port=result.get("port"),
            command=result.get("command"),
            details={
                **(result.get("details") or {}),
                "tokenizer": tokenizer_result
            }
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
    Also loads tokenizer for the selected model
    """
    try:
        # Load tokenizer first
        tokenizer_result = tokenizer_manager.load_tokenizer(request.serving_name)
        print(f"🔧 Tokenizer: {tokenizer_result['message']}")

        # Clear all documents when model is reset
        cleared_count = document_manager.clear_all_documents()

        # Start model server with reset
        result = model_server.up(reset=True, serving_name=request.serving_name)

        # Update message to include document clearing info
        model_message = result["message"]
        if cleared_count > 0:
            model_message += f" (Cleared {cleared_count} document(s) due to model reset)"

        # Format model and tokenizer status separately
        tokenizer_message = ""

        # Format tokenizer status
        if tokenizer_result["status"] == "loaded":
            tokenizer_message = f"✅ Tokenizer: {tokenizer_result['tokenizer_name']}"
        elif tokenizer_result["status"] == "not_loaded":
            tokenizer_message = "⚠️  Tokenizer: Not configured"
        else:
            tokenizer_message = f"❌ Tokenizer: Failed to load"

        # Combine messages with line breaks
        combined_message = '\n'.join([model_message, tokenizer_message])

        return ModelResponse(
            status=result["status"],
            message=combined_message,
            timestamp=datetime.now(),
            model_name=request.serving_name,
            pid=result.get("pid"),
            port=result.get("port"),
            command=result.get("command"),
            details={
                **(result.get("details") or {}),
                "tokenizer": tokenizer_result
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restart model: {str(e)}"
        )


@router.post("/down", response_model=ModelResponse)
async def stop_model():
    """
    Stop the currently running model and unload tokenizer
    """
    try:
        result = await model_server.down()

        # Unload tokenizer
        tokenizer_manager.unload_tokenizer()
        print("🔧 Tokenizer: Unloaded")

        # Format status messages
        model_status = f"✅ Model: {result['message']}"
        tokenizer_status = "✅ Tokenizer: Unloaded"

        # Combine messages with line breaks
        combined_message = '\n'.join([model_status, tokenizer_status])

        return ModelResponse(
            status=result["status"],
            message=combined_message,
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
    Get current model and tokenizer status
    """
    try:
        model_status = model_server.get_status()
        tokenizer_status = tokenizer_manager.get_status()

        result = {
            **model_status,
            "tokenizer": tokenizer_status,
            "last_updated": datetime.now()
        }

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
