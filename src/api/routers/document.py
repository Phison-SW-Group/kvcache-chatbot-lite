"""
Independent document router
Documents are managed separately from sessions
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pathlib import Path
from typing import List, Optional
import os
import aiofiles
import json
import asyncio
import hashlib

from models import (
    DocumentUploadResponse,
    DocumentInfo,
    GroupInfo,
    CacheGroupRequest,
    CacheAllGroupsRequest,
    CacheGroupResponse
)
from config import settings
from services.document_manager import document_manager
from services.document_service import document_service
from services.llm_service import llm_service
from services.model import model_server


router = APIRouter(prefix="/documents", tags=["documents"])


# ============================================================================
# Cache Helper Functions
# ============================================================================

async def _cache_content_in_kv(content: str, description: str = "content") -> None:
    """
    Helper function to cache content in KV Cache by running inference

    This function:
    1. For local models: Validates model server is running and configuration matches
    2. For remote models: Validates LLM service is configured correctly
    3. Runs inference with content as system message (max_tokens=2)
    4. Logs operation details

    Args:
        content: Text content to cache
        description: Description for logging (e.g., "document", "group")

    Raises:
        HTTPException: If model server not running (local) or configuration mismatch
    """
    # Get current configuration
    current_config = llm_service.get_current_config()
    current_model_name = current_config.get('model')

    if not current_model_name:
        raise HTTPException(
            status_code=400,
            detail="No model configured. Please select a model first."
        )

    # Check if current model is remote
    current_model = settings.get_model_by_serving_name(current_model_name)
    is_remote = current_model and current_model.model_type == "remote"

    # For local models: validate model server is running and configuration matches
    if not is_remote:
        # Step 1: Check if model server is running
        if not model_server._is_running():
            raise HTTPException(
                status_code=503,
                detail="Model server is not running. Please start the model server first."
            )

        # Step 2: Validate LLM service configuration matches running model
        model_config = model_server.config

        if not current_config['model'] or current_config['model'] != model_config.alias:
            raise HTTPException(
                status_code=503,
                detail=f"Model configuration mismatch. LLM service is configured for '{current_config['model']}' but model server is running '{model_config.alias}'. Please restart the model."
            )

    # Step 3: Log current model and configuration
    from services.model_log import model_log_service
    if is_remote:
        model_log_service.append_log(f"Cache operation for {description} using remote model: {current_config['model']}")
    else:
        model_log_service.append_log(f"Cache operation for {description} using local model: {current_config['model']}")
    model_log_service.append_log(f"Model base_url: {current_config['base_url']}")

    # Step 4: Prepare messages and run inference
    messages = [
        {
            "role": "system",
            "content": content
        }
    ]

    # Temporarily override max_tokens to 2 (we only care about caching/processing content)
    original_max_tokens = llm_service.completion_params.get('max_tokens')
    llm_service.completion_params['max_tokens'] = 2

    try:
        # Run inference to populate KV Cache (local) or send to remote API (remote)
        async for _ in llm_service.generate_response(messages, stream=False):
            pass  # We don't care about the response, just caching/processing
    finally:
        # Restore original max_tokens
        if original_max_tokens is not None:
            llm_service.completion_params['max_tokens'] = original_max_tokens
        else:
            llm_service.completion_params.pop('max_tokens', None)


# Ensure upload directory exists
os.makedirs(settings.documents.upload_dir, exist_ok=True)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None)
):
    """
    Upload a PDF document with model-specific processing
    Currently supports: .pdf
    The document will be automatically chunked based on the specified model's tokenizer
    Different models will have different processing results stored separately

    Args:
        file: PDF file to upload
        model_name: Model name to use for processing (if not provided, uses current llm_service model)
    """
    # Get model name - prefer explicit parameter, fallback to llm_service
    from services.llm_service import llm_service
    from services.tokenizer_manager import tokenizer_manager

    # DEBUG: Print received model_name
    print(f"🔍 DEBUG: Received model_name parameter: {repr(model_name)}")

    if model_name:
        current_model_name = model_name
        print(f"📤 Upload with explicit model parameter: {model_name}")

        # Switch llm_service and tokenizer to this model
        selected_model = settings.get_model_by_serving_name(model_name)
        if not selected_model:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not found in configuration"
            )

        # Reconfigure LLM service
        completion_params_dict = selected_model.completion_params.model_dump(exclude={'custom_params'})
        if selected_model.completion_params.custom_params:
            completion_params_dict.update(selected_model.completion_params.custom_params)

        llm_service.reconfigure(
            model=selected_model.serving_name,
            api_key=selected_model.api_key,
            base_url=selected_model.base_url,
            **completion_params_dict
        )

        # Load tokenizer for this model
        tokenizer_manager.load_tokenizer(model_name)
    else:
        # Fallback to current llm_service model
        current_config = llm_service.get_current_config()
        current_model_name = current_config.get("model")
        print(f"📤 Upload using llm_service model: {current_model_name}")

    if not current_model_name:
        raise HTTPException(
            status_code=400,
            detail="No model specified. Please provide model_name parameter or select a model first."
        )

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

    # Generate consistent doc_id based on filename
    import hashlib

    # Create a consistent doc_id based on filename (same file = same doc_id)
    file_hash = hashlib.md5(file.filename.encode()).hexdigest()[:12]
    doc_id = file_hash

    # Use model-specific directory structure
    # Sanitize model name for filesystem
    safe_model_name = current_model_name.replace('/', '_').replace('\\', '_')
    model_dir = Path(settings.documents.upload_dir) / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save file in model directory with doc_id suffix: {filename_stem}_{doc_id}{extension}
    file_stem = Path(file.filename).stem
    file_ext = Path(file.filename).suffix
    file_path = model_dir / f"{file_stem}_{doc_id}{file_ext}"

    print(f"📁 Model directory: {safe_model_name}/")
    print(f"📄 File will be saved as: {file_stem}_{doc_id}{file_ext}")

    try:
        # Save file (overwrite if exists)
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)

        # Process document to extract text and create chunks
        processed_doc = await document_service.process_document(file_path)

        # Convert DocumentChunk objects to ChunkMetadata objects
        from services.document_manager import ChunkMetadata
        chunks = [
            ChunkMetadata(**chunk.to_dict())
            for chunk in processed_doc.chunks
        ]

        # Store in document manager with model name
        final_doc_id = document_manager.add_document(
            filename=file.filename,
            file_size=file_size,
            file_path=file_path,
            model_name=current_model_name,  # NEW: Model-specific processing
            full_text=processed_doc.full_text,
            chunks=chunks,
            total_pages=processed_doc.total_pages,
            tokenizer=processed_doc.tokenizer,
            doc_id=doc_id  # Use the generated unique doc_id
        )

        # Store groups if available
        if processed_doc.groups:
            doc = document_manager.get_document(final_doc_id, current_model_name)
            if doc:
                for group in processed_doc.groups:
                    doc.add_group(group.to_dict())
                document_manager._save_document_metadata(current_model_name, final_doc_id)  # Persist groups (note order!)

        return DocumentUploadResponse(
            doc_id=final_doc_id,
            filename=file.filename,
            file_size=file_size,
            total_pages=processed_doc.total_pages,
            total_chunks=processed_doc.total_chunks,
            message=f"Document '{file.filename}' processed successfully with model '{current_model_name}' ({processed_doc.total_chunks} chunks)"
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


@router.post("/upload_collection")
async def upload_collection(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = Form(None),
    collection_name: Optional[str] = Form(None)
):
    """
    Upload multiple PDF documents as a collection with cross-file two-stage grouping
    All documents will be processed together and grouped across files

    Args:
        files: List of PDF files to upload
        model_name: Model name to use for processing (if not provided, uses current llm_service model)
        collection_name: Optional collection name (auto-generated if not provided)
    """
    from services.llm_service import llm_service
    from services.tokenizer_manager import tokenizer_manager
    from services.document_service import document_service
    from services.document_manager import document_manager, ChunkMetadata
    import uuid
    from datetime import datetime

    print(f"\n{'='*80}")
    print(f"📤 Collection upload request:")
    print(f"   - Files: {[f.filename for f in files]}")
    print(f"   - Model: {model_name}")
    print(f"   - Collection: {collection_name}")
    print(f"{'='*80}\n")

    # Validate files
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")

    # Get model name - prefer explicit parameter, fallback to llm_service
    if model_name:
        current_model_name = model_name
        print(f"📤 Collection upload with explicit model parameter: {model_name}")

        # Switch llm_service and tokenizer to this model
        selected_model = settings.get_model_by_serving_name(model_name)
        if not selected_model:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' not found in configuration"
            )

        # Reconfigure LLM service (same as single upload)
        completion_params_dict = selected_model.completion_params.model_dump(exclude={'custom_params'})
        if selected_model.completion_params.custom_params:
            completion_params_dict.update(selected_model.completion_params.custom_params)

        llm_service.reconfigure(
            model=selected_model.serving_name,
            api_key=selected_model.api_key,
            base_url=selected_model.base_url,
            **completion_params_dict
        )

        # Load tokenizer for this model
        tokenizer_manager.load_tokenizer(model_name)
    else:
        # Fallback to current llm_service model
        current_config = llm_service.get_current_config()
        current_model_name = current_config.get("model")
        print(f"📤 Collection upload using llm_service model: {current_model_name}")

    if not current_model_name:
        raise HTTPException(
            status_code=400,
            detail="No model specified. Please provide model_name parameter or select a model first."
        )

    # Create model directory
    model_dir = document_manager._get_model_directory(current_model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save all uploaded files temporarily
    file_paths = []
    filenames = []
    total_file_size = 0

    try:
        for file in files:
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if not document_service.supports_extension(file_ext):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}. Only PDF files are supported."
                )

            # Save file to model directory temporarily (consistent with single upload)
            file_hash = hashlib.md5(file.filename.encode()).hexdigest()[:8]
            file_stem = Path(file.filename).stem
            file_ext = Path(file.filename).suffix
            temp_filename = f"{file_stem}_{file_hash}{file_ext}"
            file_path = model_dir / temp_filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
                total_file_size += len(content)

            file_paths.append(file_path)
            filenames.append(file.filename)
            print(f"✅ Saved temporarily: {file.filename} → {file_path.name}")

        # Generate collection name based on naming rule
        if not collection_name:
            if len(files) == 1:
                # Single file: use filename without extension
                collection_name = Path(filenames[0]).stem
            else:
                # Multiple files: join first 10 chars of each filename
                truncated_names = [Path(fn).stem[:10] for fn in filenames]
                collection_name = 'collection-[' + ']['.join(truncated_names) + ']'

        print(f"\n📦 Collection name: {collection_name}")

        # Generate collection doc_id
        collection_hash = hashlib.md5(collection_name.encode()).hexdigest()[:12]
        collection_doc_id = collection_hash

        # Process all documents as a collection
        print(f"\n🔄 Processing collection with {len(file_paths)} documents...")
        collection_result = await document_service.process_documents_as_collection(
            file_paths=file_paths,
            collection_id=collection_name
        )

        # Merge all processed documents into one
        merged_full_text = ""
        merged_chunks = []
        total_pages = 0

        for idx, processed_doc in enumerate(collection_result["processed_documents"]):
            # Accumulate full text
            merged_full_text += processed_doc.full_text
            if idx < len(collection_result["processed_documents"]) - 1:
                merged_full_text += "\n\n"  # Separator between documents

            total_pages += processed_doc.total_pages

        # Use all chunks from collection (already cross-file processed)
        for chunk in collection_result["all_chunks"]:
            chunk_meta = ChunkMetadata(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                page_numbers=chunk.page_numbers,
                char_count=chunk.char_count,
                token_count=chunk.token_count,
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index,
                group_id=chunk.group_id,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                page_key=chunk.page_key,
                collection_id=collection_name
            )
            merged_chunks.append(chunk_meta)

        # Get tokenizer name
        tokenizer_name = collection_result["processed_documents"][0].tokenizer if collection_result["processed_documents"] else current_model_name

        # Add single collection document to manager
        doc_id = document_manager.add_document(
            filename=collection_name,  # Use collection name as filename
            file_size=total_file_size,
            file_path=file_paths[0] if len(file_paths) == 1 else model_dir / f"{collection_doc_id}_collection",  # Virtual path for multi-file
            model_name=current_model_name,
            full_text=merged_full_text,
            chunks=merged_chunks,
            total_pages=total_pages,
            tokenizer=tokenizer_name,
            doc_id=collection_doc_id
        )

        # Update document with collection info and groups
        doc_metadata = document_manager.get_document(doc_id, current_model_name)
        if doc_metadata:
            # Add collection ID
            doc_metadata.collection_id = collection_name

            # Add source files info (custom field)
            doc_metadata.source_files = filenames

            # Add groups from collection result
            for group in collection_result["collection_groups"]:
                group_dict = group.to_dict()
                doc_metadata.add_group(group_dict)

            # Save updated metadata
            document_manager._save_document_metadata(current_model_name, doc_id)

        # Clean up temporary files (keep only if single file)
        if len(file_paths) > 1:
            for file_path in file_paths:
                if file_path.exists():
                    file_path.unlink()
            print(f"🧹 Cleaned up {len(file_paths)} temporary files")

        # Prepare response
        response = {
            "collection_id": collection_name,
            "collection_name": collection_name,
            "doc_id": collection_doc_id,
            "model_name": current_model_name,
            "total_documents": len(filenames),
            "total_chunks": collection_result["total_chunks"],
            "total_groups": collection_result["total_groups"],
            "source_files": filenames,
            "file_size": total_file_size
        }

        print(f"\n{'='*80}")
        print(f"✅ Collection upload completed:")
        print(f"   - Collection: {collection_name}")
        print(f"   - Doc ID: {collection_doc_id}")
        print(f"   - Source files: {len(filenames)}")
        print(f"   - Total chunks: {collection_result['total_chunks']}")
        print(f"   - Total groups: {collection_result['total_groups']}")
        print(f"{'='*80}\n")

        return response

    except Exception as e:
        # Clean up files on error
        for file_path in file_paths:
            if file_path.exists():
                file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error processing collection: {str(e)}")


@router.get("/list")
async def list_documents():
    """
    Get list of uploaded documents for the CURRENT MODEL only
    Each model has its own independent document list
    """
    # Get current model
    from services.llm_service import llm_service
    current_config = llm_service.get_current_config()
    current_model_name = current_config.get("model")

    print(f"📋 list_documents called - current model: {current_model_name}")

    if not current_model_name:
        print("⚠️  No model selected, returning empty list")
        return []  # No model selected, return empty list

    # Return only documents for this model
    documents = document_manager.list_documents(model_name=current_model_name)

    print(f"📋 Returning {len(documents)} document(s) for model '{current_model_name}'")
    for doc in documents:
        print(f"   - {doc['filename']} (doc_id: {doc['doc_id']})")

    return documents


@router.get("/{doc_id}")
async def get_document_info(doc_id: str, include_preview: bool = True, include_chunks: bool = False):
    """
    Get information about a specific document for the current model

    Args:
        doc_id: Document ID
        include_preview: Whether to include content preview (default: True)
        include_chunks: Whether to include chunk information (default: False)
    """
    # Get current model
    from services.llm_service import llm_service
    current_config = llm_service.get_current_config()
    current_model_name = current_config.get("model")

    if not current_model_name:
        raise HTTPException(
            status_code=400,
            detail="No model selected"
        )

    doc = document_manager.get_document(doc_id, current_model_name)

    if not doc:
        available_models = document_manager.get_available_models(doc_id)
        if available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Document not processed with current model '{current_model_name}'. Available models: {', '.join(available_models)}"
            )
        else:
            raise HTTPException(status_code=404, detail="Document not found")

    return doc.to_dict(include_preview=include_preview, include_chunks=include_chunks)


@router.get("/{doc_id}/chunks")
async def get_document_chunks(doc_id: str):
    """
    Get all chunks for a specific document (current model)

    Args:
        doc_id: Document ID

    Returns:
        List of chunks with metadata
    """
    # Get current model
    from services.llm_service import llm_service
    current_config = llm_service.get_current_config()
    current_model_name = current_config.get("model")

    if not current_model_name:
        raise HTTPException(status_code=400, detail="No model selected")

    chunks = document_manager.get_document_chunks(doc_id, current_model_name)

    if chunks is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found for model '{current_model_name}'"
        )

    return {
        "doc_id": doc_id,
        "model_name": current_model_name,
        "total_chunks": len(chunks),
        "chunks": [chunk.to_dict(include_content=False) for chunk in chunks]
    }


@router.get("/{doc_id}/chunks/{chunk_id}")
async def get_document_chunk(doc_id: str, chunk_id: int):
    """
    Get a specific chunk from a document (current model)

    Args:
        doc_id: Document ID
        chunk_id: Chunk ID

    Returns:
        Chunk content and metadata
    """
    # Get current model
    from services.llm_service import llm_service
    current_config = llm_service.get_current_config()
    current_model_name = current_config.get("model")

    if not current_model_name:
        raise HTTPException(status_code=400, detail="No model selected")

    chunk = document_manager.get_document_chunk(doc_id, current_model_name, chunk_id)

    if chunk is None:
        raise HTTPException(status_code=404, detail="Document or chunk not found")

    return {
        "doc_id": doc_id,
        "model_name": current_model_name,
        "chunk": chunk.to_dict(include_content=True)
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, model_name: Optional[str] = None):
    """
    Delete a document (optionally for specific model only)

    Args:
        doc_id: Document ID
        model_name: Optional model name. If provided, only delete that model's metadata.
                   If not provided, delete all model metadata for this document.
    """
    success = document_manager.delete_document(doc_id, model_name)

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    if model_name:
        return {"message": f"Document metadata deleted for model '{model_name}'"}
    else:
        return {"message": "Document deleted successfully (all models)"}


@router.post("/cache/{doc_id}")
async def cache_document(doc_id: str):
    """
    Cache an already uploaded document in KV Cache by groups (for current model)
    Returns SSE stream with real-time progress updates for each group

    This endpoint performs caching for an existing document by processing each group separately:
    1. Retrieve document from document manager (for current model)
    2. Get all groups from the document
    3. Run inference for each group individually (max_tokens=2)
       - For local models: This populates the KV Cache in memory for each group
       - For remote models: This is a simulated operation (no actual caching occurs)
       - Each group is cached separately for better granularity
       - The cache is immediately available for subsequent queries
       - Prefix matching will accelerate future requests with the same document groups

    No server restart is needed - the cache remains in memory and is ready for use.

    SSE Event Format:
    - event: progress
      data: {"group_id": str, "group_name": str, "status": "pending|in_progress|done|failed",
             "cached_count": int, "total_groups": int, "error": str (optional)}
    - event: complete
      data: {"doc_id": str, "filename": str, "file_size": int, "cached_groups": int,
             "total_groups": int, "failed_groups": list}
    """

    async def generate_progress():
        """Generator function for SSE stream"""
        try:
            # Check if current model is remote
            from services.llm_service import llm_service
            from config import settings

            current_config = llm_service.get_current_config()
            current_model_name = current_config.get("model")

            if not current_model_name:
                yield f"event: error\ndata: {json.dumps({'error': 'No model selected. Please select a model first.'})}\n\n"
                return

            current_model = settings.get_model_by_serving_name(current_model_name) if current_model_name else None
            is_remote = current_model and current_model.model_type == "remote"

            # Get document from document manager (for current model)
            document = document_manager.get_document(doc_id, current_model_name)
            if not document:
                available_models = document_manager.get_available_models(doc_id)
                if available_models:
                    error_msg = f"Document with ID '{doc_id}' not processed with model '{current_model_name}'. Available models: {', '.join(available_models)}. Please upload the document with this model first."
                else:
                    error_msg = f"Document with ID '{doc_id}' not found"
                yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"
                return

            # Get all groups from the document
            groups = document.get_groups()
            if not groups:
                yield f"event: error\ndata: {json.dumps({'error': 'Document has no groups to cache'})}\n\n"
                return

            # Log cache operation start
            from services.model_log import model_log_service

            model_log_service.append_log(f"🚀 Starting cache operation for document: {document.filename} (doc_id={doc_id})")
            if is_remote:
                model_log_service.append_log(f"ℹ️  Current model: remote ({current_model.provider})")
                model_log_service.append_log(f"📊 Document has {len(groups)} groups to process")
            else:
                model_log_service.append_log(f"ℹ️  Current model: local (KV cache enabled)")
                model_log_service.append_log(f"📊 Document has {len(groups)} groups to cache")
            model_log_service.append_log(f"📄 Document size: {document.file_size:,} bytes")
            model_log_service.append_log("=" * 50)

            # Send initial status for all groups (pending)
            yield f"event: init\ndata: {json.dumps({'filename': document.filename, 'file_size': document.file_size, 'total_groups': len(groups), 'groups': [{'group_id': g.get('group_id', ''), 'status': 'pending'} for g in groups]})}\n\n"

            # Cache each group individually
            cached_count = 0
            failed_groups = []
            failed_groups_detail = []  # Store detailed error info

            for idx, group in enumerate(groups):
                group_id = group.get('group_id', '')
                group_content = group.get('merged_content', '')

                if not group_content:
                    model_log_service.append_log(f"Skipping empty group: {group_id}")
                    failed_groups.append(group_id)
                    failed_groups_detail.append({"group_id": group_id, "error": "Empty content"})
                    yield f"event: progress\ndata: {json.dumps({'group_id': group_id, 'status': 'failed', 'cached_count': cached_count, 'total_groups': len(groups), 'error': 'Empty content'})}\n\n"
                    continue

                # Send in_progress status
                yield f"event: progress\ndata: {json.dumps({'group_id': group_id, 'status': 'in_progress', 'cached_count': cached_count, 'total_groups': len(groups)})}\n\n"

                try:
                    # Log group details before caching
                    content_length = len(group_content)
                    model_log_service.append_log(f"Starting cache for group {idx + 1}/{len(groups)}: {group_id} (content: {content_length} chars)")

                    # Both local and remote models need to call LLM inference
                    await _cache_content_in_kv(
                        content=group_content,
                        description=f"group '{group_id}' from document '{document.filename}'"
                    )

                    if is_remote:
                        model_log_service.append_log(f"✅ Successfully processed group {idx + 1}/{len(groups)} with remote model: {group_id}")
                    else:
                        model_log_service.append_log(f"✅ Successfully cached group {idx + 1}/{len(groups)}: {group_id}")

                    cached_count += 1

                    # Mark group as cached in metadata
                    document.mark_group_as_cached(group_id)

                    # Send done status
                    yield f"event: progress\ndata: {json.dumps({'group_id': group_id, 'status': 'done', 'cached_count': cached_count, 'total_groups': len(groups)})}\n\n"

                except Exception as e:
                    error_msg = str(e)
                    model_log_service.append_log(f"❌ Failed to cache group {group_id}: {error_msg}")
                    failed_groups.append(group_id)
                    failed_groups_detail.append({"group_id": group_id, "error": error_msg})

                    # Send failed status
                    yield f"event: progress\ndata: {json.dumps({'group_id': group_id, 'status': 'failed', 'cached_count': cached_count, 'total_groups': len(groups), 'error': error_msg})}\n\n"

            # Log completion
            model_log_service.append_log("=" * 50)
            if is_remote:
                model_log_service.append_log(f"✅ Cache operation completed for document: {document.filename}")
                model_log_service.append_log(f"📊 Successfully processed {cached_count}/{len(groups)} groups with remote model ({current_model.provider})")
                model_log_service.append_log(f"ℹ️  Note: Remote models don't use local KV cache, but content was sent to remote API")
            else:
                model_log_service.append_log(f"✅ Cache operation completed for document: {document.filename}")
                model_log_service.append_log(f"📊 Successfully cached {cached_count}/{len(groups)} groups")
                model_log_service.append_log("🚀 KV Cache is ready for prefix matching!")

            if failed_groups:
                model_log_service.append_log(f"❌ Failed groups: {', '.join(failed_groups)}")

            # Update document cache status based on group cache status
            document.update_document_cache_status()

            # Save updated metadata to persist cache status
            document_manager._save_document_metadata(current_model_name, doc_id)
            model_log_service.append_log(f"💾 Cache status saved to metadata")

            # Send completion event
            if is_remote:
                message = f"Document '{document.filename}' processed successfully with remote model. {cached_count}/{len(groups)} groups sent to {current_model.provider}."
            else:
                message = f"Document '{document.filename}' cached successfully. {cached_count}/{len(groups)} groups cached."

            if failed_groups:
                message += f" Failed groups: {', '.join(failed_groups)}"

            yield f"event: complete\ndata: {json.dumps({'doc_id': doc_id, 'filename': document.filename, 'file_size': document.file_size, 'cached_groups': cached_count, 'total_groups': len(groups), 'failed_groups': failed_groups_detail, 'message': message})}\n\n"

        except Exception as e:
            from services.model_log import model_log_service
            model_log_service.append_log(f"Cache operation failed: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'error': f'Error caching document: {str(e)}'})}\n\n"

    return StreamingResponse(generate_progress(), media_type="text/event-stream")


@router.get("/{doc_id}/cache-status")
async def get_document_cache_status(doc_id: str):
    """
    Get cache status for a document and its groups (for current model)

    Returns:
        - doc_id: Document ID
        - filename: Document filename
        - cached: Whether all groups are cached
        - last_cached_at: Last cache timestamp
        - total_groups: Total number of groups
        - cached_groups: Number of cached groups
        - cache_completion: Percentage of groups cached
        - groups: List of groups with individual cache status
    """
    # Get current model
    from services.llm_service import llm_service
    current_config = llm_service.get_current_config()
    current_model_name = current_config.get("model")

    if not current_model_name:
        raise HTTPException(
            status_code=400,
            detail="No model selected"
        )

    document = document_manager.get_document(doc_id, current_model_name)
    if not document:
        available_models = document_manager.get_available_models(doc_id)
        if available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Document not processed with model '{current_model_name}'. Available: {', '.join(available_models)}"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{doc_id}' not found"
            )

    # Get cache summary
    cache_summary = document.get_cache_summary()

    # Add group-level cache details
    groups_cache_status = []
    for group in document.get_groups():
        groups_cache_status.append({
            'group_id': group.get('group_id'),
            'cached': group.get('cached', False),
            'cached_at': group.get('cached_at'),
            'total_tokens': group.get('total_tokens', 0),
            'content_length': group.get('content_length', 0)
        })

    cache_summary['groups'] = groups_cache_status

    return cache_summary


@router.get("/{doc_id}/groups", response_model=List[GroupInfo])
async def get_document_groups(doc_id: str):
    """
    Get all groups for a document (for current model)

    Returns list of groups with metadata (without full content)
    """
    # Get current model
    from services.llm_service import llm_service
    current_config = llm_service.get_current_config()
    current_model_name = current_config.get("model")

    if not current_model_name:
        raise HTTPException(
            status_code=400,
            detail="No model selected"
        )

    document = document_manager.get_document(doc_id, current_model_name)
    if not document:
        available_models = document_manager.get_available_models(doc_id)
        if available_models:
            raise HTTPException(
                status_code=404,
                detail=f"Document not processed with model '{current_model_name}'. Available: {', '.join(available_models)}"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{doc_id}' not found"
            )

    groups = document.get_groups()
    if not groups:
        return []

    # Convert to GroupInfo (exclude merged_content for efficiency)
    group_infos = []
    for group in groups:
        group_infos.append(GroupInfo(
            group_id=group.get('group_id', ''),
            chunk_ids=group.get('chunk_ids', []),
            total_tokens=group.get('total_tokens', 0),
            content_length=len(group.get('merged_content', ''))
        ))

    return group_infos
