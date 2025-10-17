"""
Tokenizer Manager Service
Manages HuggingFace tokenizers independently from model server
"""
from typing import Optional, Dict
from pathlib import Path
import logging
from config import settings


class TokenizerStatus:
    """Tokenizer status constants"""
    LOADED = "loaded"
    NOT_LOADED = "not_loaded"
    ERROR = "error"
    LOADING = "loading"


class TokenizerManager:
    """
    Manages tokenizer lifecycle independently from model server

    This allows tokenizer to be used even if model fails to load
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = None
        self.tokenizer_name: Optional[str] = None
        self.tokenizer_status = TokenizerStatus.NOT_LOADED
        self.tokenizer_error: Optional[str] = None
        self.current_model_name: Optional[str] = None

    def load_tokenizer(self, serving_name: Optional[str] = None) -> Dict:
        """
        Load tokenizer for specified model

        Args:
            serving_name: Model serving name to get tokenizer config

        Returns:
            Status dict with tokenizer loading result
        """
        self.tokenizer_status = TokenizerStatus.LOADING
        self.tokenizer_error = None

        try:
            # Find model configuration
            selected_model = None
            if serving_name:
                for model in settings.models:
                    if model.serving_name == serving_name:
                        selected_model = model
                        break
            else:
                selected_model = settings.models[0] if settings.models else None

            if not selected_model:
                self.tokenizer_status = TokenizerStatus.ERROR
                self.tokenizer_error = "No model configuration found"
                return {
                    "status": TokenizerStatus.ERROR,
                    "message": "No model configuration found",
                    "tokenizer_name": None
                }

            self.current_model_name = selected_model.serving_name

            # Check if model has tokenizer configured
            if not selected_model.tokenizer:
                self.tokenizer_status = TokenizerStatus.NOT_LOADED
                self.tokenizer_name = None
                self.tokenizer = None
                return {
                    "status": TokenizerStatus.NOT_LOADED,
                    "message": f"No tokenizer configured for model '{selected_model.serving_name}'",
                    "tokenizer_name": None,
                    "model_name": selected_model.serving_name
                }

            # Load tokenizer
            tokenizer_id = selected_model.tokenizer
            self.logger.info(f"Loading tokenizer: {tokenizer_id} for model: {selected_model.serving_name}")

            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            self.tokenizer_name = tokenizer_id
            self.tokenizer_status = TokenizerStatus.LOADED

            self.logger.info(f"✅ Tokenizer loaded successfully: {tokenizer_id}")

            return {
                "status": TokenizerStatus.LOADED,
                "message": f"Tokenizer '{tokenizer_id}' loaded successfully",
                "tokenizer_name": tokenizer_id,
                "model_name": selected_model.serving_name
            }

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"❌ Failed to load tokenizer: {error_msg}")
            self.tokenizer_status = TokenizerStatus.ERROR
            self.tokenizer_error = error_msg
            self.tokenizer = None
            self.tokenizer_name = None

            return {
                "status": TokenizerStatus.ERROR,
                "message": f"Failed to load tokenizer: {error_msg}",
                "tokenizer_name": None,
                "error": error_msg
            }

    def unload_tokenizer(self):
        """Unload current tokenizer"""
        self.tokenizer = None
        self.tokenizer_name = None
        self.tokenizer_status = TokenizerStatus.NOT_LOADED
        self.tokenizer_error = None
        self.current_model_name = None
        self.logger.info("Tokenizer unloaded")

    def get_status(self) -> Dict:
        """
        Get current tokenizer status

        Returns:
            Status dict with tokenizer information
        """
        return {
            "status": self.tokenizer_status,
            "tokenizer_name": self.tokenizer_name,
            "model_name": self.current_model_name,
            "is_loaded": self.tokenizer_status == TokenizerStatus.LOADED,
            "error": self.tokenizer_error
        }

    def is_loaded(self) -> bool:
        """Check if tokenizer is loaded and ready"""
        return self.tokenizer_status == TokenizerStatus.LOADED and self.tokenizer is not None

    def count_tokens(self, text: str) -> Optional[int]:
        """
        Count tokens in text using loaded tokenizer

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens, or None if tokenizer not loaded
        """
        if not self.is_loaded():
            return None

        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            self.logger.error(f"Token counting failed: {e}")
            return None


# Global tokenizer manager instance
tokenizer_manager = TokenizerManager()

