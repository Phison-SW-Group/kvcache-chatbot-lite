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
from services.llm_service import llm_service
from services.model import model_server


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


@router.get("/{doc_id}")
async def get_document_info(doc_id: str, include_preview: bool = True):
    """
    Get information about a specific document
    
    Args:
        doc_id: Document ID
        include_preview: Whether to include content preview (default: True)
    """
    doc = document_manager.get_document(doc_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return doc.to_dict(include_preview=include_preview, preview_lines=10)


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document"""
    success = document_manager.delete_document(doc_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully"}


@router.post("/upload_and_cache", response_model=DocumentUploadResponse)
async def upload_document_and_cache(file: UploadFile = File(...)):
    """
    Upload a document and pre-cache it in KV Cache
    
    This endpoint performs two steps:
    1. Upload and process the document (same as /upload)
    2. Run inference with document content as system message (max_tokens=2)
       - This populates the KV Cache in memory
       - The cache is immediately available for subsequent queries
       - Prefix matching will accelerate future requests with the same document
    
    No server restart is needed - the cache remains in memory and is ready for use.
    """
    # Step 1: Upload document (reuse existing logic)
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
        
        # Step 2: Pre-cache document content via inference
        # Prepare messages with only system role containing document content
        messages = [
            {
                "role": "system",
                "content": document_content
            }
        ]
        
        # Run inference with max_tokens=2 (we only care about caching the prefix)
        # Temporarily override max_tokens
        original_max_tokens = llm_service.max_tokens
        llm_service.max_tokens = 2
        
        try:
            # Run inference to populate KV Cache
            async for _ in llm_service.generate_response(messages, stream=False):
                pass  # We don't care about the response, just caching
        finally:
            # Restore original max_tokens
            llm_service.max_tokens = original_max_tokens
        
        # Step 3: KV Cache is now populated and ready for use
        # No need to restart - the cache is already in memory and will be reused
        # for subsequent queries with the same document prefix
        # 
        # Note: The model server uses --kv-cache-resume-policy based on its startup mode:
        # - If started with reset=True: cache is fresh for this session
        # - If started with reset=False: cache is loaded from previous sessions
        # Either way, the cache from Step 2 is now available for prefix matching

        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            file_size=file_size,
            message=f"Document '{file.filename}' uploaded and cached successfully. KV Cache ready for prefix matching."
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

