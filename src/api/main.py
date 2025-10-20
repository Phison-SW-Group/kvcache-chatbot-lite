"""
FastAPI main application
"""
import argparse
import asyncio
from dataclasses import dataclass
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from routers import session, upload, document, model, logs
from services.session_service import session_manager
from services.llm_service import llm_service, configure_llm_service
from services.model import model_server


@dataclass
class ApiArgs:
    ip     : str  = "localhost"
    port   : int  = 8000
    reload : bool = False

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--ip', default=cls.ip)
        parser.add_argument('-p', '--port', type=int, default=cls.port)
        parser.add_argument('--no-reload', action="store_true")

        args = parser.parse_args()
        return cls(
            ip=args.ip,
            port=args.port,
            reload=not args.no_reload
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events"""
    # Startup
    import os
    os.makedirs(settings.documents.upload_dir, exist_ok=True)
    await session_manager.start_cleanup_task()

    # Create model log session with unique ID
    from services.model_log import model_log_service
    log_session = model_log_service.create_session()
    print(f"📝 Model log session created: {log_session.session_id}")
    print(f"   Log file: {log_session.log_file_path}")
    print(f"   llama-server will write logs to this file via --log-file parameter")

    # Initialize LLM service with the first configured model
    if not settings.models:
        error_msg = "No models configured in env.yaml"
        print(f"❌ {error_msg}")
        model_log_service.append_log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # Initialize LLM service with the first configured model
    first_model = settings.models[0]

    # Validate first model configuration
    if not first_model.api_key or first_model.api_key == "empty":
        error_msg = f"API key not configured for model '{first_model.serving_name}'. Please set 'api_key' in the model configuration."
        print(f"❌ {error_msg}")
        model_log_service.append_log(f"ERROR: {error_msg}")
        raise ValueError(error_msg)

    # Convert completion_params to dict for **kwargs
    completion_params_dict = first_model.completion_params.model_dump(exclude={'custom_params'})
    if first_model.completion_params.custom_params:
        completion_params_dict.update(first_model.completion_params.custom_params)

    # Configure LLM service with first model
    configure_llm_service(
        model=first_model.serving_name,
        api_key=first_model.api_key,
        base_url=first_model.base_url,
        **completion_params_dict
    )

    print(f"✅ LLM service initialized")
    print(f"   Model: {first_model.serving_name}")
    print(f"   Base URL: {first_model.base_url}")
    print(f"   Temperature: {first_model.completion_params.temperature}")
    print(f"   Max Tokens: {first_model.completion_params.max_tokens}")
    print(f"   ℹ️  LLM service will be reconfigured when switching models")

    model_log_service.append_log(f"LLM service initialized - Model: {first_model.serving_name}, Base URL: {first_model.base_url}")

    yield

    # Shutdown
    await session_manager.stop_cleanup_task()

    # Stop model server if running
    if model_server._is_running():
        print("🛑 Stopping model server...")
        model_log_service.append_log("Shutting down model server...")
        await model_server.down()
        print("✅ Model server stopped")
        model_log_service.append_log("Model server stopped successfully")


# Create FastAPI app
app = FastAPI(
    title=settings.api.title,
    version=settings.api.version,
    lifespan=lifespan
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(session.router, prefix=settings.api.prefix)
app.include_router(upload.router, prefix=settings.api.prefix)  # Legacy
app.include_router(document.router, prefix=settings.api.prefix)  # New independent documents
app.include_router(model.router, prefix=settings.api.prefix)  # Model deployment
app.include_router(logs.router, prefix=settings.api.prefix)  # Model logs


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "KVCache Chatbot API",
        "version": settings.api.version,
        "endpoints": {
            "docs": "/docs",
            "session": f"{settings.api.prefix}/session/{{session_id}}",
            "messages": f"{settings.api.prefix}/session/{{session_id}}/messages",
            "documents": f"{settings.api.prefix}/documents",
            "upload_document": f"{settings.api.prefix}/documents/upload",
            "cache_document": f"{settings.api.prefix}/documents/cache/{{doc_id}}",
            "model": f"{settings.api.prefix}/model",
            "logs": f"{settings.api.prefix}/logs/current",
            "logs_recent": f"{settings.api.prefix}/logs/recent",
            "logs_sessions": f"{settings.api.prefix}/logs/sessions"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    import atexit

    args = ApiArgs.from_args()

    # Register cleanup handler to ensure model server is stopped on exit
    def cleanup_model_server():
        """Ensure model server is stopped when backend exits"""

        if model_server._is_running():
            print("\n🛑 Stopping model server...")
            try:
                # Run the async down() method in a new event loop
                result = asyncio.run(model_server.down())
                if result.get("status") == "success":
                    print("✅ Model server stopped")
                else:
                    print(f"⚠️ {result.get('message')}")
            except Exception as e:
                print(f"❌ Error stopping model server: {e}")
                # Force kill as last resort
                try:
                    if model_server.process and model_server.process.poll() is None:
                        model_server.process.kill()
                        print("🔨 Force killed model server process")
                except:
                    pass

    atexit.register(cleanup_model_server)

    uvicorn.run(
        "main:app",
        host=args.ip,
        port=args.port,
        reload=args.reload
    )

