"""
Document upload router (part of session resources)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import os
import aiofiles

from models import UploadResponse
from config import settings
from services.session_service import session_manager
from services.document_service import document_service


router = APIRouter(prefix="/session", tags=["document"])


# Ensure upload directory exists
os.makedirs(settings.documents.upload_dir, exist_ok=True)


@router.post("/{session_id}/document", response_model=UploadResponse)
async def upload_document(
    session_id: str,
    file: UploadFile = File(...)
):
    """
    Upload a document for a conversation session
    Currently supports: .txt
    Extensible for future formats: .pdf, .docx, .md, etc.
    """
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in settings.documents.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. "
                   f"Allowed types: {', '.join(settings.documents.allowed_extensions)}"
        )

    # Check file size
    file_content = await file.read()
    file_size = len(file_content)

    if file_size > settings.documents.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.documents.max_file_size / (1024*1024):.1f}MB"
        )

    # Save file temporarily
    file_path = Path(settings.documents.upload_dir) / f"{session_id}_{file.filename}"

    try:
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)

        # Process document
        document_content = await document_service.process_document(file_path)

        # Store document in session
        session = session_manager.get_or_create_session(session_id)
        session.set_document(document_content, file.filename)

        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            file_size=file_size,
            message=f"Document '{file.filename}' uploaded successfully"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.delete("/{session_id}/document")
async def delete_document(session_id: str):
    """Delete uploaded document for a session"""
    session = session_manager.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.has_document():
        raise HTTPException(status_code=404, detail="No document uploaded for this session")

    # Clear document from session
    session.document_content = None
    session.document_filename = None

    # Clean up file
    if session.document_filename:
        file_path = Path(settings.documents.upload_dir) / f"{session_id}_{session.document_filename}"
        if file_path.exists():
            file_path.unlink()

    return {"message": "Document deleted successfully"}

