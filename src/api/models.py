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
    use_rag: bool = Field(default=True, description="Whether to use RAG retrieval for document context")
    serving_name: Optional[str] = Field(default=None, description="Selected model serving name for this request")


class ChatResponse(BaseModel):
    """Response body for chat endpoint"""
    session_id: str
    message: str
    timestamp: datetime
    rag_info: Optional[dict] = None  # RAG retrieval information (group_id, similarity_score)


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
    total_groups: int = 0
    uploaded_at: str


class GroupInfo(BaseModel):
    """Group information for RAG"""
    group_id: str
    chunk_ids: List[int]
    total_tokens: int
    content_length: int


class CacheGroupRequest(BaseModel):
    """Request body for caching a specific group"""
    doc_id: str = Field(..., description="Document ID")
    group_id: str = Field(..., description="Group ID to cache")


class CacheAllGroupsRequest(BaseModel):
    """Request body for caching all groups in a document"""
    doc_id: str = Field(..., description="Document ID")


class CacheGroupResponse(BaseModel):
    """Response body for group caching operations"""
    doc_id: str
    group_id: Optional[str] = None
    cached_groups: int
    total_groups: int
    message: str


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

