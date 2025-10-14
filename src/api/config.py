"""
Configuration settings for the API
"""
from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


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

    LLM_SERVER_EXE: Optional[str] = None
    LLM_SERVER_CACHE: Optional[str] = None
    LLM_SERVER_LOG: Optional[str] = None

    MODEL_NAME_OR_PATH: Optional[str] = None
    MODEL_SERVING_NAME: Optional[str] = None  # Display name for deployed model. Falls back to MODEL_NAME_OR_PATH if None
    BASE_URL: Optional[str] = None
    API_KEY: Optional[str] = None
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000

    def model_post_init(self, __context) -> None:
        """Automatically compute model display name and validate config after initialization"""
        # Set the computed display name
        self.MODEL_SERVING_NAME = self.MODEL_SERVING_NAME or self.MODEL_NAME_OR_PATH

        # Optional: Log configuration status
        if bool(self.MODEL_NAME_OR_PATH or self.MODEL_SERVING_NAME):
            print(f"✅ Model configuration valid: {self.MODEL_SERVING_NAME}")
        else:
            print("❌ Model configuration incomplete: MODEL_NAME_OR_PATH or MODEL_SERVING_NAME required")

    class Config:
        # Use absolute path to .env file (relative to this config.py file)
        env_file = str(Path(__file__).parent / ".env")
        case_sensitive = True


settings = Settings()

