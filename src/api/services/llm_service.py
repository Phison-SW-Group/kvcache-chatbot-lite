"""
LLM service for generating chat responses
Supports OpenAI compatible APIs
"""
from typing import AsyncGenerator, List, Dict, Optional


class LLMService:
    """
    Service for LLM interactions
    Supports OpenAI compatible APIs (OpenAI, Azure OpenAI, vLLM, etc.)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: str = "empty",
        base_url: Optional[str] = None,
        **completion_params
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.completion_params = completion_params  # Store all completion parameters
        # Don't create client here - create dynamically on each call to support model switching

    def _get_client(self):
        """Get or create OpenAI client with current configuration"""
        from openai import AsyncOpenAI
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def reconfigure(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **completion_params
    ):
        """Reconfigure LLM service (for switching models)"""
        if model is not None:
            self.model = model
        if api_key is not None:
            self.api_key = api_key
        if base_url is not None:
            self.base_url = base_url
        if completion_params:
            self.completion_params.update(completion_params)

    def get_current_config(self) -> dict:
        """Get current LLM service configuration"""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key": bool(self.api_key and self.api_key != "empty"),
            "completion_params": self.completion_params
        }

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Generate response from LLM

        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response

        Yields:
            Response text chunks
        """
        # Validate configuration - if missing, attempt an automatic fallback
        if not self.api_key or (self.api_key == "empty"):
            try:
                # Try to auto-select a usable model from settings
                from config import settings
                candidate = None
                for m in getattr(settings, "all_models", []):
                    if (m.api_key and m.api_key != "empty") or (m.api_key == "not-needed"):
                        candidate = m
                        break
                if candidate is not None:
                    params = candidate.completion_params.model_dump(exclude={'custom_params'})
                    if candidate.completion_params.custom_params:
                        params.update(candidate.completion_params.custom_params)
                    self.reconfigure(model=candidate.serving_name, api_key=candidate.api_key, base_url=candidate.base_url, **params)
                else:
                    raise RuntimeError("LLM service auto-configuration failed: no valid model found")
            except Exception:
                error_msg = (
                    "LLM service is not properly configured. "
                    "api_key is not set. Please configure it in env.yaml"
                )
                raise RuntimeError(error_msg)

        async for chunk in self._openai_response(messages, stream):
            yield chunk

    async def _openai_response(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate response using OpenAI compatible API"""
        try:
            from services.model_log import model_log_service

            # Get client with current configuration (supports dynamic model switching)
            client = self._get_client()

            # Log API request
            user_message = messages[-1]['content'] if messages else ""
            model_log_service.append_log(f"API Request - Model: {self.model}, Messages: {len(messages)}, Stream: {stream}")
            model_log_service.append_log(f"User message: {user_message[:100]}..." if len(user_message) > 100 else f"User message: {user_message}")

            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                **self.completion_params  # Include all completion parameters
            }

            print(self.base_url)
            print(self.api_key)
            print(api_params)

            response = await client.chat.completions.create(**api_params)

            if stream:
                # Streaming response
                full_response = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content

                # Log complete response
                model_log_service.append_log(f"API Response (streaming) - Length: {len(full_response)} chars")
                model_log_service.append_log(f"Response content: {full_response[:200]}..." if len(full_response) > 200 else f"Response content: {full_response}")
            else:
                # Non-streaming response
                response_content = response.choices[0].message.content
                model_log_service.append_log(f"API Response - Length: {len(response_content)} chars")
                model_log_service.append_log(f"Response content: {response_content[:200]}..." if len(response_content) > 200 else f"Response content: {response_content}")
                yield response_content

        except Exception as e:
            error_message = f"LLM API Error: {str(e)}"
            print(error_message)
            model_log_service.append_log(f"API Error: {error_message}")
            yield error_message


    def _prepare_messages_with_document(
        self,
        messages: List[Dict[str, str]],
        document_content: Optional[str]
    ) -> List[Dict[str, str]]:
        """
        Prepare messages with document context
        Adds document as system message or prepends to first user message
        """
        if not document_content:
            return messages

        # Add document context as system message at the beginning
        document_message = {
            "role": "system",
            "content": f"Document content:\n\n{document_content}\n\nPlease answer questions based on this document."
        }

        return [document_message] + messages


# Global LLM service instance
llm_service = LLMService()


def configure_llm_service(
    model: Optional[str] = None,
    api_key: str = "empty",
    base_url: Optional[str] = None,
    **completion_params
):
    """
    Configure the global LLM service instance

    Args:
        model: Model name/identifier
        api_key: API key for authentication
        base_url: Base URL for API endpoint
        **completion_params: All completion parameters (temperature, max_tokens, top_p, etc.)
    """
    global llm_service
    llm_service = LLMService(
        model=model,
        api_key=api_key,
        base_url=base_url,
        **completion_params
    )
    return llm_service
