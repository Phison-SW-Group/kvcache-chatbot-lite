"""
FastAPI main application
"""
import argparse
from dataclasses import dataclass
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from routers import session, upload, document
from services.session_service import session_manager
from services.llm_service import llm_service


@dataclass
class ApiArgs:
    ip     : str  = "0.0.0.0"
    port   : int  = 8000
    reload : bool = True

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
    os.makedirs("uploads", exist_ok=True)
    await session_manager.start_cleanup_task()
    
    # Initialize LLM service with config
    if settings.LLM_API_KEY:
        from services.llm_service import LLMService
        global llm_service
        llm_service.__init__(
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS
        )
        print(f"✅ LLM service initialized with model: {settings.LLM_MODEL}")
    else:
        print("⚠️  No LLM API key provided, using mock responses")
    
    yield
    
    # Shutdown
    await session_manager.stop_cleanup_task()


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
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
app.include_router(session.router, prefix=settings.API_PREFIX)
app.include_router(upload.router, prefix=settings.API_PREFIX)  # Legacy
app.include_router(document.router, prefix=settings.API_PREFIX)  # New independent documents


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "KVCache Chatbot API",
        "version": settings.API_VERSION,
        "endpoints": {
            "docs": "/docs",
            "session": f"{settings.API_PREFIX}/session/{{session_id}}",
            "messages": f"{settings.API_PREFIX}/session/{{session_id}}/messages",
            "documents": f"{settings.API_PREFIX}/documents",
            "upload_document": f"{settings.API_PREFIX}/documents/upload"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    args = ApiArgs.from_args()
    uvicorn.run(
        "main:app",
        host=args.ip,
        port=args.port,
        reload=args.reload
    )

