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
from services.llm_service import llm_service
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
    model_type: str  # "local" or "remote"
    provider: Optional[str] = None  # Only for remote models
    is_running: Optional[bool] = None  # Only for local models - indicates if server is running


@router.get("/list")
async def list_models() -> List[ModelInfo]:
    """
    Get list of all configured models from settings (both local and remote)
    Includes running status for local models
    """
    try:
        # Get current running model info
        current_config = llm_service.get_current_config()
        current_running_model = current_config.get("model")

        # Check if model server is actually running (for local models)
        is_server_running = model_server._is_running()

        models = []
        for model_config in settings.all_models:
            is_running = None

            # For local models, check if this specific model is running
            if model_config.model_type == "local":
                # A local model is running if:
                # 1. Model server is running AND
                # 2. This model is the currently configured one
                is_running = is_server_running and (current_running_model == model_config.serving_name)

            models.append(ModelInfo(
                model_name_or_path=model_config.model_name_or_path,
                serving_name=model_config.serving_name,
                model_type=model_config.model_type,
                provider=model_config.provider if model_config.model_type == "remote" else None,
                is_running=is_running
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
    Only works for local models - remote models don't need to be started
    """
    try:
        # Find the selected model configuration
        selected_model = None
        for model_config in settings.all_models:
            if model_config.serving_name == request.serving_name:
                selected_model = model_config
                break

        if not selected_model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.serving_name}' not found in configuration"
            )

        # Check if it's a remote model
        if selected_model.model_type == "remote":
            return ModelResponse(
                status="info",
                message=f"❌ Remote model '{selected_model.serving_name}' does not require startup. Remote models ({selected_model.provider}) are always ready to use.",
                timestamp=datetime.now(),
                model_name=request.serving_name,
                details={"model_type": "remote", "provider": selected_model.provider}
            )

        # Update LLM service configuration to match the selected model
        completion_params_dict = selected_model.completion_params.model_dump(exclude={'custom_params'})
        if selected_model.completion_params.custom_params:
            completion_params_dict.update(selected_model.completion_params.custom_params)

        llm_service.reconfigure(
            model=selected_model.serving_name,
            api_key=selected_model.api_key,
            base_url=selected_model.base_url,
            **completion_params_dict
        )
        print(f"🔄 LLM service reconfigured for model: {selected_model.serving_name}")

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
    This will reset cache status for all uploaded documents (files remain, cache flags reset)
    Users need to re-cache documents after reset to populate KV cache
    Also loads tokenizer for the selected model
    Only works for local models - remote models don't need to be started
    """
    try:
        # Find the selected model configuration
        selected_model = None
        for model_config in settings.all_models:
            if model_config.serving_name == request.serving_name:
                selected_model = model_config
                break

        if not selected_model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.serving_name}' not found in configuration"
            )

        # Check if it's a remote model
        if selected_model.model_type == "remote":
            return ModelResponse(
                status="info",
                message=f"❌ Remote model '{selected_model.serving_name}' does not require startup. Remote models ({selected_model.provider}) are always ready to use.",
                timestamp=datetime.now(),
                model_name=request.serving_name,
                details={"model_type": "remote", "provider": selected_model.provider}
            )

        # Update LLM service configuration to match the selected model
        completion_params_dict = selected_model.completion_params.model_dump(exclude={'custom_params'})
        if selected_model.completion_params.custom_params:
            completion_params_dict.update(selected_model.completion_params.custom_params)

        llm_service.reconfigure(
            model=selected_model.serving_name,
            api_key=selected_model.api_key,
            base_url=selected_model.base_url,
            **completion_params_dict
        )
        print(f"🔄 LLM service reconfigured for model: {selected_model.serving_name}")

        # Load tokenizer first
        tokenizer_result = tokenizer_manager.load_tokenizer(request.serving_name)
        print(f"🔧 Tokenizer: {tokenizer_result['message']}")

        # Reset cache status for this model's documents (without deleting files)
        reset_count = document_manager.reset_model_cache_status(request.serving_name)

        # Start model server with reset
        result = model_server.up(reset=True, serving_name=request.serving_name)

        # Update message to include cache reset info
        model_message = result["message"]
        if reset_count > 0:
            model_message += f" (Reset cache status for {reset_count} document(s) - please re-cache them)"

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
    Stop the currently running local model and unload tokenizer
    Remote models don't need to be stopped
    """
    try:
        # Check if current model is remote by checking llm_service configuration
        current_config = llm_service.get_current_config()
        current_model_name = current_config.get("model")

        # Find if current model is remote
        if current_model_name:
            current_model = settings.get_model_by_serving_name(current_model_name)
            if current_model and current_model.model_type == "remote":
                return ModelResponse(
                    status="info",
                    message=f"❌ Remote model '{current_model.serving_name}' does not need to be stopped. Remote models ({current_model.provider}) are always available.",
                    timestamp=datetime.now(),
                    model_name=current_model_name,
                    details={"model_type": "remote", "provider": current_model.provider}
                )

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
    Get current model, tokenizer, and LLM service status
    """
    try:
        model_status = model_server.get_status()
        tokenizer_status = tokenizer_manager.get_status()
        llm_config = llm_service.get_current_config()

        result = {
            **model_status,
            "tokenizer": tokenizer_status,
            "llm_service": llm_config,
            "last_updated": datetime.now()
        }

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model status: {str(e)}"
        )


@router.post("/switch", response_model=ModelResponse)
async def switch_model(request: ModelUpRequest):
    """
    Switch to a different model (local or remote)
    For local models: just reconfigure LLM service (model needs to be started separately)
    For remote models: reconfigure LLM service and ready to use immediately
    """
    try:
        # Find the selected model configuration
        selected_model = None
        for model_config in settings.all_models:
            if model_config.serving_name == request.serving_name:
                selected_model = model_config
                break

        if not selected_model:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.serving_name}' not found in configuration"
            )

        # Update LLM service configuration to match the selected model
        completion_params_dict = selected_model.completion_params.model_dump(exclude={'custom_params'})
        if selected_model.completion_params.custom_params:
            completion_params_dict.update(selected_model.completion_params.custom_params)

        llm_service.reconfigure(
            model=selected_model.serving_name,
            api_key=selected_model.api_key,
            base_url=selected_model.base_url,
            **completion_params_dict
        )
        print(f"🔄 LLM service reconfigured for model: {selected_model.serving_name}")

        # Load tokenizer if available
        tokenizer_result = tokenizer_manager.load_tokenizer(request.serving_name)
        print(f"🔧 Tokenizer: {tokenizer_result['message']}")

        if selected_model.model_type == "remote":
            message = f"✅ Switched to remote model '{selected_model.serving_name}' ({selected_model.provider}). Ready to use immediately!"
        else:
            message = f"✅ Switched to local model '{selected_model.serving_name}'. Please start the model before using."

        return ModelResponse(
            status="success",
            message=message,
            timestamp=datetime.now(),
            model_name=request.serving_name,
            details={
                "model_type": selected_model.model_type,
                "provider": selected_model.provider if selected_model.model_type == "remote" else None,
                "tokenizer": tokenizer_result
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to switch model: {str(e)}"
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
