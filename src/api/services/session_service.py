"""
Session management service for multi-turn conversations
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from models import Message, SessionInfo
import asyncio


class ConversationSession:
    """Represents a single conversation session"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Message] = []
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.document_content: Optional[str] = None
        self.document_filename: Optional[str] = None

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history"""
        message = Message(role=role, content=content, timestamp=datetime.now())
        self.messages.append(message)
        self.last_active = datetime.now()

    def get_messages(self) -> List[Message]:
        """Get all messages in the conversation"""
        return self.messages

    def get_messages_for_llm(self) -> List[dict]:
        """
        Get messages in format suitable for LLM API
        Returns list of dicts with 'role' and 'content' keys
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def set_document(self, content: str, filename: str) -> None:
        """Store uploaded document content"""
        self.document_content = content
        self.document_filename = filename
        self.last_active = datetime.now()

    def has_document(self) -> bool:
        """Check if session has an uploaded document"""
        return self.document_content is not None

    def get_info(self) -> SessionInfo:
        """Get session information"""
        return SessionInfo(
            session_id=self.session_id,
            created_at=self.created_at,
            last_active=self.last_active,
            message_count=len(self.messages),
            has_document=self.has_document()
        )


class SessionManager:
    """
    Manages all conversation sessions
    Currently uses in-memory storage, designed to be easily extended to Redis/Database
    """

    def __init__(self, timeout_seconds: int = 3600):
        self._sessions: Dict[str, ConversationSession] = {}
        self._timeout_seconds = timeout_seconds
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start_cleanup_task(self):
        """Start background task to clean up expired sessions"""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())

    async def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_expired_sessions(self):
        """Background task to remove expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                now = datetime.now()
                expired_sessions = [
                    session_id for session_id, session in self._sessions.items()
                    if (now - session.last_active).total_seconds() > self._timeout_seconds
                ]
                for session_id in expired_sessions:
                    del self._sessions[session_id]
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in cleanup task: {e}")

    def get_or_create_session(self, session_id: str) -> ConversationSession:
        """Get existing session or create new one"""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(session_id)
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get existing session, return None if not found"""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def get_all_sessions(self) -> List[SessionInfo]:
        """Get information about all active sessions"""
        return [session.get_info() for session in self._sessions.values()]


# Global session manager instance
session_manager = SessionManager()

