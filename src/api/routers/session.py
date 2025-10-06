"""
Session router with RESTful design
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime
import json

from models import MessageRequest, ChatResponse, SessionInfo
from services.session_service import session_manager
from services.llm_service import llm_service


router = APIRouter(prefix="/session", tags=["session"])


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get information about a session"""
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.get_info()


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a conversation session"""
    success = session_manager.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted successfully"}


@router.get("/{session_id}/messages")
async def get_messages(session_id: str):
    """Get all messages (chat history) for a session"""
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "messages": session.get_messages(),
        "has_document": session.has_document(),
        "document_filename": session.document_filename
    }


@router.post("/{session_id}/messages", response_model=ChatResponse)
async def send_message(session_id: str, request: MessageRequest):
    """
    Send a message in a conversation (non-streaming)
    Maintains conversation history for multi-turn dialogue
    """
    # Get or create session
    session = session_manager.get_or_create_session(session_id)
    
    # Add user message to history
    session.add_message("user", request.message)
    
    # Prepare messages for LLM
    messages = session.get_messages_for_llm()
    
    # Add document context if requested and available
    if request.use_document and session.has_document():
        # Insert document as system message
        document_msg = {
            "role": "system",
            "content": f"Document content:\n\n{session.document_content}\n\nPlease answer based on this document."
        }
        messages.insert(0, document_msg)
    
    # Generate response (collect full response)
    full_response = ""
    async for chunk in llm_service.generate_response(messages, stream=False):
        full_response += chunk
    
    # Add assistant response to history
    session.add_message("assistant", full_response)
    
    return ChatResponse(
        session_id=session_id,
        message=full_response,
        timestamp=datetime.now()
    )


@router.post("/{session_id}/messages/stream")
async def stream_message(session_id: str, request: MessageRequest):
    """
    Send a message with streaming response (SSE)
    Better UX for real-time LLM responses
    """
    # Get or create session
    session = session_manager.get_or_create_session(session_id)
    
    # Add user message to history
    session.add_message("user", request.message)
    
    # Prepare messages for LLM
    messages = session.get_messages_for_llm()
    
    # Add document context if requested and available
    if request.use_document and session.has_document():
        document_msg = {
            "role": "system",
            "content": f"Document content:\n\n{session.document_content}\n\nPlease answer based on this document."
        }
        messages.insert(0, document_msg)
    
    async def event_generator():
        """Generate SSE events"""
        full_response = ""
        
        try:
            async for chunk in llm_service.generate_response(messages, stream=True):
                full_response += chunk
                # Send chunk as SSE
                data = json.dumps({"chunk": chunk, "done": False})
                yield f"data: {data}\n\n"
            
            # Add complete response to history
            session.add_message("assistant", full_response)
            
            # Send completion signal
            data = json.dumps({"chunk": "", "done": True, "full_response": full_response})
            yield f"data: {data}\n\n"
            
        except Exception as e:
            error_data = json.dumps({"error": str(e), "done": True})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

