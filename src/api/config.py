"""
Configuration settings for the API with nested structure
"""
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional, List, Dict, Any
from pathlib import Path


class ServerSettings(BaseSettings):
    """Server configuration settings"""
    exe_path: Optional[str] = None
    log_path: Optional[str] = None
    cache_dir: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="SERVER_",
        case_sensitive=True,
    )


class CompletionParamsSettings(BaseSettings):
    """Completion parameters for a specific model"""
    # Basic generation parameters
    temperature: float = 0.7
    max_tokens: int = 2000
    # top_p: float = 1.0
    # top_k: int = 40
    # repeat_penalty: float = 1.1
    # frequency_penalty: float = 0.0
    # presence_penalty: float = 0.0

    # # Advanced parameters
    # min_p: float = 0.05
    # tfs_z: float = 1.0
    # typical_p: float = 1.0
    # mirostat: int = 0
    # mirostat_tau: float = 5.0
    # mirostat_eta: float = 0.1

    # # Token handling
    # repeat_last_n: int = 64
    # penalize_newline: bool = True
    # add_bos_token: bool = True
    # ban_eos_token: bool = False
    # skip_special_tokens: bool = True

    # # Streaming and seed
    # stream: bool = False
    # seed: int = -1

    # # Stop sequences
    # stop: Optional[List[str]] = None

    # Custom parameters
    custom_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra='allow',
    )


class ModelSettings(BaseSettings):
    """Individual model configuration with its own API and completion settings"""
    model_name_or_path: Optional[str] = None
    serving_name: Optional[str] = None
    tokenizer: Optional[str] = None  # HuggingFace tokenizer identifier

    # API settings for this model
    base_url: Optional[str] = None  # Default: None
    api_key: Optional[str] = "empty"  # Default: "empty" (must be configured for LLM service)

    # Completion parameters for this model
    completion_params: CompletionParamsSettings = Field(default_factory=CompletionParamsSettings)

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        case_sensitive=True,
    )

    def model_post_init(self, __context) -> None:
        """Auto-compute serving name if not provided"""
        if self.serving_name is None:
            self.serving_name = self.model_name_or_path

        # Convert None api_key to "empty"
        if self.api_key is None:
            self.api_key = "empty"



class APISettings(BaseSettings):
    """API general settings"""
    title: str = "KVCache Chatbot API"
    version: str = "1.0.0"
    prefix: str = "/api/v1"

    # Session settings
    session_timeout: int = 3600  # 1 hour in seconds

    model_config = SettingsConfigDict(
        env_prefix="API_",
        case_sensitive=True,
    )


class DocumentSettings(BaseSettings):
    """Document management settings"""
    upload_dir: str = "uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB (increased for PDF files)
    allowed_extensions: List[str] = [".pdf"]

    # Chunking settings for PDF documents
    chunk_size: int = 5000  # Maximum characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks for context continuity

    # Tokenizer settings
    # If set, will use this model's tokenizer to count tokens in chunks
    # Should match a serving_name or model_name_or_path from models config
    tokenizer_model: Optional[str] = None

    # First-stage grouping settings (token-based sequential merging)
    grouping: bool = True  # Enable token-based grouping when tokenizer available
    file_max_tokens: int = 3000  # Maximum tokens per merged group
    utilization_threshold: float = 0.8  # Minimum utilization for final group (0.0-1.0)

    # Stage-2 similarity grouping settings
    similarity_method: str = "bm25"  # bm25 | embedding
    similarity_min_score: float | None = None  # optional BM25 threshold

    model_config = SettingsConfigDict(
        env_prefix="DOCUMENT_",
        case_sensitive=True,
    )


class Settings(BaseSettings):
    """Main application settings with nested configuration"""

    # Nested settings
    api: APISettings = Field(default_factory=APISettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    models: List[ModelSettings] = Field(default_factory=list)
    documents: DocumentSettings = Field(default_factory=DocumentSettings)

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',  # Ignore extra environment variables
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_yaml_config()
        self._validate_config()

    def _load_yaml_config(self):
        """Load configuration from YAML file and merge with current settings"""
        yaml_file = Path(__file__).parent / "env.yaml"
        if not yaml_file.exists():
            print(f"⚠️  YAML config file not found: {yaml_file}")
            return

        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f) or {}

            # Load server settings
            if 'server' in yaml_data:
                server_data = yaml_data['server']
                self.server = ServerSettings(
                    exe_path=server_data.get('exe_path'),
                    log_path=server_data.get('log_path'),
                    cache_dir=server_data.get('cache_dir')
                )

            # Load models settings
            if 'models' in yaml_data:
                models_data = yaml_data['models']
                self.models = []
                for model_data in models_data:
                    # Load completion params for this model
                    completion_params_data = model_data.get('completion_params', {})

                    # Handle custom parameters
                    completion_kwargs = {}
                    custom_params = {}

                    # Get valid field names from CompletionParamsSettings
                    valid_fields = set(CompletionParamsSettings.model_fields.keys())

                    for key, value in completion_params_data.items():
                        if key in valid_fields:
                            completion_kwargs[key] = value
                        else:
                            custom_params[key] = value

                    if custom_params:
                        completion_kwargs['custom_params'] = custom_params

                    completion_params = CompletionParamsSettings(**completion_kwargs)

                    # Create model settings
                    model_settings = ModelSettings(
                        model_name_or_path=model_data.get('model_name_or_path'),
                        serving_name=model_data.get('serving_name'),
                        tokenizer=model_data.get('tokenizer'),
                        base_url=model_data.get('base_url'),
                        api_key=model_data.get('api_key'),
                        completion_params=completion_params
                    )
                    self.models.append(model_settings)


            print("✅ YAML configuration loaded successfully")

        except Exception as e:
            print(f"❌ Error loading YAML config: {e}")

    def _validate_config(self):
        """Validate the complete configuration"""
        # Validate at least one model is configured
        if not self.models:
            print("⚠️  No models configured")
        else:
            for i, model in enumerate(self.models):
                if not model.model_name_or_path:
                    print(f"⚠️  Model {i+1} missing model_name_or_path")
                else:
                    print(f"✅ Model {i+1} configured: {model.serving_name or model.model_name_or_path}")

        # Validate server settings
        if self.server.exe_path:
            print(f"✅ Server configured: {self.server.exe_path}")
        else:
            print("⚠️  Server exe_path not configured")

    def get_tokenizer_for_model(self, model_identifier: Optional[str] = None) -> Optional[str]:
        """
        Get the tokenizer string for a given model identifier

        Args:
            model_identifier: serving_name or model_name_or_path,
                            or None to automatically use the first model with tokenizer

        Returns:
            Tokenizer string (HuggingFace identifier) or None if not configured
        """
        # If specific model identifier provided, search for that model
        if model_identifier:
            for model in self.models:
                if model.serving_name == model_identifier or model.model_name_or_path == model_identifier:
                    return model.tokenizer
            return None

        # If tokenizer_model is set in documents config, use that
        if self.documents.tokenizer_model:
            for model in self.models:
                if model.serving_name == self.documents.tokenizer_model or model.model_name_or_path == self.documents.tokenizer_model:
                    return model.tokenizer

        # Otherwise, automatically use the first model that has a tokenizer configured
        for model in self.models:
            if model.tokenizer:
                return model.tokenizer

        return None


# Create global settings instance
settings = Settings()
