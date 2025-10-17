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
from services.document_manager import document_manager
from services.rag_service import rag_service


router = APIRouter(prefix="/session", tags=["session"])


def _prepare_document_context(request: MessageRequest):
    """
    Helper function to prepare document context using RAG or full document

    Returns:
        Tuple of (context_dict, rag_info_dict):
        - context_dict: Dictionary with 'role' and 'content' for system message, or None
        - rag_info_dict: RAG retrieval information, or None
    """
    if not request.document_id:
        return None, None

    doc = document_manager.get_document(request.document_id)
    if not doc:
        return None, None

    # Use RAG retrieval if enabled and groups are available
    if request.use_rag and doc.get_groups():
        # Prepare groups with merged_content (rebuild if missing for backward compatibility)
        groups = doc.get_groups()
        enriched_groups = []
        for group in groups:
            if not group.get('merged_content'):
                # Rebuild merged_content from chunks
                chunk_ids = group.get('chunk_ids', [])
                chunk_contents = []
                for chunk in doc.chunks:
                    if chunk.chunk_id in chunk_ids:
                        chunk_contents.append(chunk.content)
                group['merged_content'] = "\n\n".join(chunk_contents)
            enriched_groups.append(group)

        # Retrieve most similar group using RAG
        print(f"DEBUG: Total groups to search: {len(enriched_groups)}")
        for i, g in enumerate(enriched_groups):
            print(f"  Group {i}: {g.get('group_id', 'unknown')}, content_length: {len(g.get('merged_content', ''))}")

        results = rag_service.retrieve_most_similar_group(
            query=request.message,
            groups=enriched_groups,
            top_k=1
        )

        print(f"DEBUG: RAG returned {len(results)} result(s)")
        if results:
            for i, r in enumerate(results):
                print(f"  Result {i}: group_id={r.group_id}, score={r.similarity_score:.4f}")

        if results:
            # Use the most similar group as context
            best_match = results[0]
            print(f"🔍 RAG: Retrieved group '{best_match.group_id}' with similarity {best_match.similarity_score:.4f}")

            context = {
                "role": "system",
                "content": f"Relevant context from document '{doc.filename}':\n\n{best_match.content}"
            }

            rag_info = {
                "group_id": best_match.group_id,
                "similarity_score": float(best_match.similarity_score),
                "method": "rag",
                "filename": doc.filename,
                "content_preview": best_match.content[:500] if best_match.content else ""  # First 500 chars
            }

            return context, rag_info
        else:
            print("⚠️  RAG: No matching groups found, using full document")

    # Fallback to full document content (legacy behavior or when RAG disabled)
    if doc.full_text:
        context = {
            "role": "system",
            "content": doc.full_text
        }
        return context, {"method": "full_document", "filename": doc.filename}

    return None, None


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
    Supports RAG-based document retrieval
    """
    # Get or create session
    session = session_manager.get_or_create_session(session_id)

    # Add user message to history
    session.add_message("user", request.message)

    # Prepare messages for LLM
    messages = session.get_messages_for_llm()

    # Add document context using RAG if available
    document_msg, rag_info = _prepare_document_context(request)
    if document_msg:
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
        timestamp=datetime.now(),
        rag_info=rag_info
    )


@router.post("/{session_id}/messages/stream")
async def stream_message(session_id: str, request: MessageRequest):
    """
    Send a message with streaming response (SSE)
    Better UX for real-time LLM responses
    Supports RAG-based document retrieval
    """
    # Get or create session
    session = session_manager.get_or_create_session(session_id)

    # Add user message to history
    session.add_message("user", request.message)

    # Prepare messages for LLM
    messages = session.get_messages_for_llm()

    # Add document context using RAG if available
    document_msg, rag_info = _prepare_document_context(request)
    if document_msg:
        messages.insert(0, document_msg)

    async def event_generator():
        """Generate SSE events"""
        full_response = ""

        # Send RAG info first if available
        if rag_info:
            data = json.dumps({"rag_info": rag_info, "chunk": "", "done": False})
            yield f"data: {data}\n\n"

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

