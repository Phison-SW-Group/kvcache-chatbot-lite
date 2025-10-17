"""
Data models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


class Message(BaseModel):
    """Single message in a conversation"""
    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: Optional[datetime] = None


class MessageRequest(BaseModel):
    """Request body for message endpoint"""
    message: str = Field(..., description="User message")
    document_id: Optional[str] = Field(default=None, description="Document ID to use as context")


class ChatResponse(BaseModel):
    """Response body for chat endpoint"""
    session_id: str
    message: str
    timestamp: datetime


class DocumentUploadResponse(BaseModel):
    """Response body for document upload endpoint"""
    doc_id: str
    filename: str
    file_size: int
    total_pages: int
    total_chunks: int
    message: str


class ChunkInfo(BaseModel):
    """Chunk information"""
    chunk_id: int
    page_numbers: List[int]
    char_count: int
    token_count: Optional[int] = None
    tokenizer: Optional[str] = None


class DocumentInfo(BaseModel):
    """Document information"""
    doc_id: str
    filename: str
    file_size: int
    total_pages: int
    total_chunks: int
    uploaded_at: str


class UploadResponse(BaseModel):
    """Response body for file upload endpoint (legacy)"""
    session_id: str
    filename: str
    file_size: int
    message: str


class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    created_at: datetime
    last_active: datetime
    message_count: int
    has_document: bool

