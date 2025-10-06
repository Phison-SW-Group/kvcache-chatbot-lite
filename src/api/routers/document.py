"""
Independent document router
Documents are managed separately from sessions
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from typing import List
import os
import aiofiles

from models import DocumentUploadResponse, DocumentInfo
from config import settings
from services.document_manager import document_manager
from services.document_service import document_service


router = APIRouter(prefix="/documents", tags=["documents"])


# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (independent of any session)
    Currently supports: .txt
    Extensible for future formats: .pdf, .docx, .md, etc.
    """
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_extension}. "
                   f"Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    file_content = await file.read()
    file_size = len(file_content)
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Generate unique filename
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{unique_id}_{file.filename}"
    file_path = Path(settings.UPLOAD_DIR) / safe_filename
    
    try:
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        # Process document to extract text
        document_content = await document_service.process_document(file_path)
        
        # Store in document manager
        doc_id = document_manager.add_document(
            filename=file.filename,
            file_size=file_size,
            file_path=file_path,
            content=document_content
        )
        
        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            file_size=file_size,
            message=f"Document '{file.filename}' uploaded successfully"
        )
    
    except ValueError as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@router.get("/list", response_model=List[DocumentInfo])
async def list_documents():
    """Get list of all uploaded documents"""
    documents = document_manager.list_documents()
    return documents


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document_info(doc_id: str):
    """Get information about a specific document"""
    doc = document_manager.get_document(doc_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return doc.to_dict()


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document"""
    success = document_manager.delete_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully"}

