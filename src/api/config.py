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


class ModelSettings(BaseSettings):
    """Individual model configuration"""
    model_name_or_path: Optional[str] = None
    serving_name: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        case_sensitive=True,
    )

    def model_post_init(self, __context) -> None:
        """Auto-compute serving name if not provided"""
        if self.serving_name is None:
            self.serving_name = self.model_name_or_path


class CompletionSettings(BaseSettings):
    """Completion parameters for LLM requests - supports all llama-cpp parameters"""

    # Core API settings
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    # Common generation parameters
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    top_k: int = 40
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None

    # Advanced parameters
    min_p: float = 0.05
    tfs_z: float = 1.0
    typical_p: float = 1.0
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    repeat_last_n: int = 64
    penalize_newline: bool = True
    add_bos_token: bool = True
    ban_eos_token: bool = False
    skip_special_tokens: bool = True

    # Streaming
    stream: bool = False

    # Seed
    seed: int = -1

    # Custom parameters (for any additional llama-cpp parameters)
    custom_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_prefix="COMPLETION_",
        case_sensitive=True,
        extra='allow',  # Allow additional parameters
    )

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
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = [".txt"]

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
    completion_params: CompletionSettings = Field(default_factory=CompletionSettings)
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
                self.models = [
                    ModelSettings(
                        model_name_or_path=model.get('model_name_or_path'),
                        serving_name=model.get('serving_name')
                    ) for model in models_data
                ]

            # Load completion settings (support 'completion_params', 'completion', and legacy 'inference_params')
            completion_data = None
            if 'completion_params' in yaml_data:
                completion_data = yaml_data['completion_params']
                print("📝 Using 'completion_params' configuration section")
            elif 'completion' in yaml_data:
                completion_data = yaml_data['completion']
                print("📝 Using 'completion' configuration section (consider migrating to 'completion_params')")
            elif 'inference_params' in yaml_data:
                completion_data = yaml_data['inference_params']
                print("📝 Using legacy 'inference_params' configuration section (consider migrating to 'completion_params')")

            if completion_data:
                # Create completion settings with all parameters from YAML
                completion_kwargs = {}

                # Handle custom parameters
                custom_params = {}

                # Process each parameter in the YAML data
                for key, value in completion_data.items():
                    # Check if it's a known field in CompletionSettings
                    if hasattr(CompletionSettings, key):
                        completion_kwargs[key] = value
                    else:
                        # Store unknown parameters as custom parameters
                        custom_params[key] = value

                # Add custom parameters if any
                if custom_params:
                    completion_kwargs['custom_params'] = custom_params

                # Create the settings instance
                self.completion = CompletionSettings(**completion_kwargs)

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

    # Convenience properties for backward compatibility
    @property
    def MODEL_NAME_OR_PATH(self) -> Optional[str]:
        """Get the first model's path for backward compatibility"""
        return self.models[0].model_name_or_path if self.models else None

    @property
    def MODEL_SERVING_NAME(self) -> Optional[str]:
        """Get the first model's serving name for backward compatibility"""
        return self.models[0].serving_name if self.models else None

    @property
    def BASE_URL(self) -> Optional[str]:
        """Get completion base URL for backward compatibility"""
        return self.completion.base_url

    @property
    def API_KEY(self) -> Optional[str]:
        """Get completion API key for backward compatibility"""
        return self.completion.api_key

    @property
    def TEMPERATURE(self) -> float:
        """Get completion temperature for backward compatibility"""
        return self.completion.temperature

    @property
    def MAX_TOKENS(self) -> int:
        """Get completion max tokens for backward compatibility"""
        return self.completion.max_tokens

    # Document settings backward compatibility
    @property
    def UPLOAD_DIR(self) -> str:
        """Get upload directory for backward compatibility"""
        return self.documents.upload_dir

    @property
    def MAX_FILE_SIZE(self) -> int:
        """Get max file size for backward compatibility"""
        return self.documents.max_file_size

    @property
    def ALLOWED_EXTENSIONS(self) -> List[str]:
        """Get allowed extensions for backward compatibility"""
        return self.documents.allowed_extensions


# Create global settings instance
settings = Settings()
