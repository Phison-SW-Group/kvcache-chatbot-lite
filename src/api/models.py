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
    use_document: bool = Field(default=False, description="Whether to use uploaded document context")


class ChatResponse(BaseModel):
    """Response body for chat endpoint"""
    session_id: str
    message: str
    timestamp: datetime


class UploadResponse(BaseModel):
    """Response body for file upload endpoint"""
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

