"""
LLM service for generating chat responses
Supports OpenAI compatible APIs
"""
from typing import AsyncGenerator, List, Dict, Optional
import asyncio


class LLMService:
    """
    Service for LLM interactions
    Supports OpenAI compatible APIs (OpenAI, Azure OpenAI, vLLM, etc.)
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_real_llm = api_key is not None
        
        # Initialize OpenAI client if API key is provided
        if self.use_real_llm:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
    
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
        if self.use_real_llm:
            # Use real OpenAI compatible API
            async for chunk in self._openai_response(messages, stream):
                yield chunk
        else:
            # Use mock response for testing
            if stream:
                async for chunk in self._mock_streaming_response(messages):
                    yield chunk
            else:
                response = await self._mock_complete_response(messages)
                yield response
    
    async def _openai_response(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate response using OpenAI compatible API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream
            )
            
            if stream:
                # Streaming response
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                # Non-streaming response
                yield response.choices[0].message.content
                
        except Exception as e:
            error_message = f"LLM API Error: {str(e)}"
            print(error_message)
            yield error_message
    
    async def _mock_streaming_response(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """Mock streaming response for testing"""
        user_message = messages[-1]['content'] if messages else ""
        
        # Check if document context is being used
        has_document_context = any("Document content:" in msg['content'] for msg in messages)
        
        if has_document_context:
            response = (
                f"Based on the uploaded document and your question '{user_message}', "
                f"I understand you're asking about the content in the document. "
                f"This is a mock response demonstrating multi-turn conversation with document context. "
                f"In production, a real LLM would analyze the document and provide a detailed answer."
            )
        else:
            response = (
                f"Thank you for your message: '{user_message}'. "
                f"This is a mock streaming response from the LLM service. "
                f"The conversation history has been preserved for multi-turn dialogue. "
                f"In production, this would be replaced with a real LLM API call."
            )
        
        # Simulate streaming by sending word by word
        words = response.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.05)  # Simulate network delay
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word
    
    async def _mock_complete_response(self, messages: List[Dict[str, str]]) -> str:
        """Mock complete response for testing"""
        user_message = messages[-1]['content'] if messages else ""
        
        has_document_context = any("Document content:" in msg['content'] for msg in messages)
        
        if has_document_context:
            return (
                f"Based on the uploaded document and your question '{user_message}', "
                f"I can provide an answer using the document context. "
                f"This is a mock response. In production, a real LLM would be used."
            )
        else:
            return (
                f"Thank you for your message: '{user_message}'. "
                f"This is a mock response with conversation history preserved. "
                f"In production, this would use a real LLM API."
            )
    
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

