"""
Configuration settings for the API
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    API_TITLE: str = "KVCache Chatbot API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # File upload settings
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: list[str] = [".txt"]  # Extensible for future formats
    
    # Session settings
    SESSION_TIMEOUT: int = 3600  # 1 hour in seconds
    
    # LLM settings - OpenAI compatible API
    LLM_MODEL: Optional[str] = None  # e.g., "gpt-3.5-turbo", "gpt-4"
    LLM_API_KEY: Optional[str] = None  # Your OpenAI API key
    LLM_BASE_URL: Optional[str] = None  # Optional: for Azure OpenAI or other compatible APIs
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2000

    LLM_SERVER_EXE: Optional[str] = None
    LLM_SERVER_MODEL_NAME_OR_PATH: Optional[str] = None
    LLM_SERVER_CACHE: Optional[str] = None
    LLM_SERVER_LOG: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

