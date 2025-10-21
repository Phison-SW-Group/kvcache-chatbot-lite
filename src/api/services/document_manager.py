"""
Independent document management service
Documents are stored separately from sessions
Supports collections with chunks for PDF documents
"""
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import uuid
import json
import os

from config import settings


class ChunkMetadata:
    """Metadata for a document chunk"""

    def __init__(
        self,
        chunk_id: int,
        content: str,
        page_numbers: List[int],
        char_count: int,
        token_count: Optional[int] = None,
        source_file: Optional[str] = None,
        chunk_index: Optional[int] = None,
        group_id: Optional[str] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        page_key: Optional[str] = None
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.page_numbers = page_numbers
        self.char_count = char_count
        self.token_count = token_count
        self.source_file = source_file
        self.chunk_index = chunk_index
        self.group_id = group_id
        self.start_page = start_page
        self.end_page = end_page
        self.page_key = page_key

    def to_dict(self, include_content: bool = True) -> dict:
        """Convert to dictionary format"""
        result = {
            "chunk_id": self.chunk_id,
            "page_numbers": self.page_numbers,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "group_id": self.group_id,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "page_key": self.page_key
        }
        if include_content:
            result["content"] = self.content
        return result

    def to_persistent_dict(self) -> dict:
        """Convert to dictionary format for persistence"""
        return self.to_dict(include_content=True)

    @classmethod
    def from_persistent_dict(cls, data: dict) -> 'ChunkMetadata':
        """Create ChunkMetadata from persisted dictionary"""
        return cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            page_numbers=data["page_numbers"],
            char_count=data["char_count"],
            token_count=data.get("token_count"),
            source_file=data.get("source_file"),
            chunk_index=data.get("chunk_index"),
            group_id=data.get("group_id"),
            start_page=data.get("start_page"),
            end_page=data.get("end_page"),
            page_key=data.get("page_key")
        )


class DocumentMetadata:
    """
    Metadata for an uploaded document (collection)
    Each document is a collection that may contain multiple chunks
    Model-specific: Different models can have different chunking strategies
    """

    def __init__(
        self,
        doc_id: str,
        filename: str,
        file_size: int,
        file_path: Path,
        model_name: str,  # NEW: Model name for this processing
        total_pages: int = 0,
        uploaded_at: Optional[datetime] = None,
        tokenizer: Optional[str] = None,
        cached: bool = False,  # NEW: Whether document has been cached
        last_cached_at: Optional[datetime] = None  # NEW: Last cache timestamp
    ):
        self.doc_id = doc_id
        self.filename = filename
        self.file_size = file_size
        self.file_path = file_path
        self.model_name = model_name  # NEW: Which model this metadata belongs to
        self.total_pages = total_pages
        self.uploaded_at = uploaded_at if uploaded_at else datetime.now()
        self.tokenizer = tokenizer  # Tokenizer used for this document
        self.cached = cached  # NEW: Cache status
        self.last_cached_at = last_cached_at  # NEW: Last cache time
        self.full_text: Optional[str] = None
        self.chunks: List[ChunkMetadata] = []
        self.groups: List[dict] = []  # Merged groups from document processing

    def add_chunk(self, chunk: ChunkMetadata) -> None:
        """Add a chunk to this document collection"""
        self.chunks.append(chunk)

    def get_chunk(self, chunk_id: int) -> Optional[ChunkMetadata]:
        """Get a specific chunk by ID"""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def get_all_chunks(self) -> List[ChunkMetadata]:
        """Get all chunks in this collection"""
        return self.chunks

    def add_group(self, group: dict) -> None:
        """Add a merged group to this document"""
        # Initialize cache status for new group
        if 'cached' not in group:
            group['cached'] = False
        if 'cached_at' not in group:
            group['cached_at'] = None
        self.groups.append(group)

    def get_groups(self) -> List[dict]:
        """Get all merged groups"""
        return self.groups

    def get_group_by_id(self, group_id: str) -> Optional[dict]:
        """Get a specific group by ID"""
        for group in self.groups:
            if group.get('group_id') == group_id:
                return group
        return None

    def mark_group_as_cached(self, group_id: str) -> bool:
        """
        Mark a specific group as cached

        Args:
            group_id: ID of the group to mark as cached

        Returns:
            True if group was found and marked, False otherwise
        """
        for group in self.groups:
            if group.get('group_id') == group_id:
                group['cached'] = True
                group['cached_at'] = datetime.now().isoformat()
                return True
        return False

    def mark_all_groups_as_cached(self) -> None:
        """Mark all groups as cached"""
        now = datetime.now().isoformat()
        for group in self.groups:
            group['cached'] = True
            group['cached_at'] = now

    def update_document_cache_status(self) -> None:
        """
        Update document cache status based on group cache status
        Sets cached=True and last_cached_at if all groups are cached
        """
        if not self.groups:
            return

        all_cached = all(group.get('cached', False) for group in self.groups)
        if all_cached:
            self.cached = True
            self.last_cached_at = datetime.now()

    def get_cache_summary(self) -> dict:
        """
        Get cache status summary for this document

        Returns:
            Dictionary with cache statistics
        """
        total_groups = len(self.groups)
        cached_groups = sum(1 for group in self.groups if group.get('cached', False))

        return {
            'doc_id': self.doc_id,
            'filename': self.filename,
            'cached': self.cached,
            'last_cached_at': self.last_cached_at.isoformat() if self.last_cached_at else None,
            'total_groups': total_groups,
            'cached_groups': cached_groups,
            'cache_completion': (cached_groups / total_groups * 100) if total_groups > 0 else 0
        }

    def to_dict(self, include_preview: bool = False, include_chunks: bool = False) -> dict:
        """
        Convert to dictionary format (for API response)

        Args:
            include_preview: Whether to include content preview
            include_chunks: Whether to include chunk information
        """
        result = {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "model_name": self.model_name,  # NEW: Include model name
            "total_pages": self.total_pages,
            "total_chunks": len(self.chunks),
            "total_groups": len(self.groups),
            "uploaded_at": self.uploaded_at.isoformat(),
            "tokenizer": self.tokenizer,
            "cached": self.cached,  # NEW: Cache status
            "last_cached_at": self.last_cached_at.isoformat() if self.last_cached_at else None,  # NEW: Last cache time
            "cached_groups": sum(1 for g in self.groups if g.get('cached', False))  # NEW: Count of cached groups
        }

        if include_preview and self.full_text:
            preview_text = self.full_text[:500]
            result["content_preview"] = preview_text
            result["total_chars"] = len(self.full_text)

        if include_chunks:
            result["chunks"] = [chunk.to_dict(include_content=False) for chunk in self.chunks]

        return result

    def to_persistent_dict(self) -> dict:
        """Convert to dictionary format for persistence (includes file_path and chunks)"""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "file_path": str(self.file_path),
            "model_name": self.model_name,  # NEW: Include model name
            "total_pages": self.total_pages,
            "uploaded_at": self.uploaded_at.isoformat(),
            "tokenizer": self.tokenizer,
            "cached": self.cached,  # NEW: Cache status
            "last_cached_at": self.last_cached_at.isoformat() if self.last_cached_at else None,  # NEW: Last cache time
            "chunks": [chunk.to_persistent_dict() for chunk in self.chunks],
            "groups": self.groups  # Groups already include cached and cached_at fields
        }

    @classmethod
    def from_persistent_dict(cls, data: dict) -> 'DocumentMetadata':
        """Create DocumentMetadata from persisted dictionary"""
        # Parse last_cached_at if it exists
        last_cached_at = None
        if data.get("last_cached_at"):
            try:
                last_cached_at = datetime.fromisoformat(data["last_cached_at"])
            except (ValueError, TypeError):
                last_cached_at = None

        doc = cls(
            doc_id=data["doc_id"],
            filename=data["filename"],
            file_size=data["file_size"],
            file_path=Path(data["file_path"]),
            model_name=data.get("model_name", "unknown"),  # NEW: Get model name
            total_pages=data.get("total_pages", 0),
            uploaded_at=datetime.fromisoformat(data["uploaded_at"]),
            tokenizer=data.get("tokenizer"),
            cached=data.get("cached", False),  # NEW: Restore cache status
            last_cached_at=last_cached_at  # NEW: Restore cache time
        )

        # Restore chunks
        if "chunks" in data:
            for chunk_data in data["chunks"]:
                chunk = ChunkMetadata.from_persistent_dict(chunk_data)
                doc.add_chunk(chunk)

        # Restore groups (with cache status)
        if "groups" in data:
            for group in data["groups"]:
                # Ensure cache fields exist in group
                if 'cached' not in group:
                    group['cached'] = False
                if 'cached_at' not in group:
                    group['cached_at'] = None
            doc.groups = data["groups"]

        return doc


class DocumentManager:
    """
    Manages uploaded documents independently from sessions
    Model-specific document processing - each model has its own directory

    NEW Storage structure:
    uploads/
      ├── Meta-Llama-3.1-8B/        # Model directory
      │   ├── doc_abc123.pdf        # Original file
      │   ├── doc_abc123_metadata.json
      │   ├── doc_def456.pdf
      │   └── doc_def456_metadata.json
      └── gemini-2.0-flash/         # Another model directory
          ├── doc_abc123.pdf
          └── doc_abc123_metadata.json
    """

    def __init__(self, upload_dir: str = "uploads"):
        # Store documents by (model_name, doc_id) key (model first for easier filtering)
        self._documents: Dict[tuple[str, str], DocumentMetadata] = {}
        self.upload_dir = Path(upload_dir)

        # Ensure upload directory exists
        os.makedirs(upload_dir, exist_ok=True)

        # Load existing documents metadata
        self._load_all_documents()

    def _get_model_directory(self, model_name: str) -> Path:
        """Get the directory for a specific model"""
        # Sanitize model name for filesystem
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        return self.upload_dir / safe_model_name

    def _get_doc_file_path(self, model_name: str, doc_id: str, filename: str) -> Path:
        """Get the path for the original document file within model directory"""
        model_dir = self._get_model_directory(model_name)
        # Store as: {doc_id}_{filename} for uniqueness
        return model_dir / f"{doc_id}_{filename}"

    def _get_metadata_path(self, model_name: str, doc_id: str) -> Path:
        """Get the metadata file path for a specific document and model"""
        model_dir = self._get_model_directory(model_name)
        return model_dir / f"{doc_id}_metadata.json"

    def _save_document_metadata(self, model_name: str, doc_id: str):
        """Save metadata for a single document and model"""
        key = (model_name, doc_id)
        if key not in self._documents:
            return

        try:
            doc = self._documents[key]
            metadata_path = self._get_metadata_path(model_name, doc_id)

            # Ensure model directory exists
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(doc.to_persistent_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving document metadata for {doc_id} (model: {model_name}): {e}")

    def _save_index(self):
        """
        Save index is no longer needed with new structure
        Each model directory is self-contained
        We can scan directories to discover documents
        """
        pass  # No longer needed with new structure

    def _load_all_documents(self):
        """Load all documents by scanning model directories"""
        if not self.upload_dir.exists():
            print("Upload directory not found, starting fresh")
            return

        try:
            loaded_count = 0

            # Scan each subdirectory (each is a model directory)
            for model_dir in self.upload_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                # Extract model name from directory name
                model_name = model_dir.name
                print(f"📂 Scanning model directory: {model_name}")

                # Load all metadata files in this model directory
                for metadata_file in model_dir.glob("*_metadata.json"):
                    try:
                        # Extract doc_id from filename: {doc_id}_metadata.json
                        doc_id = metadata_file.stem.replace('_metadata', '')

                        print(f"   Loading: {metadata_file.name} (doc_id: {doc_id})")

                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)

                        # Recreate DocumentMetadata object
                        metadata = DocumentMetadata.from_persistent_dict(doc_data)

                        # Check if file still exists
                        if not metadata.file_path.exists():
                            print(f"⚠️  Document file not found: {metadata.file_path}")
                            continue

                        # Reconstruct full_text from chunks (for PDF documents)
                        if metadata.chunks:
                            metadata.full_text = "\n\n".join(chunk.content for chunk in metadata.chunks)

                        # Store with (model_name, doc_id) key (model first!)
                        self._documents[(model_name, doc_id)] = metadata
                        print(f"   ✅ Loaded as key: ({model_name}, {doc_id})")
                        loaded_count += 1

                    except Exception as e:
                        print(f"Error loading {metadata_file}: {e}")
                        continue

            print(f"📂 Loaded {loaded_count} document(s) from model directories")
            print(f"📂 Total documents in memory: {len(self._documents)}")
            for (model, doc_id), doc in self._documents.items():
                print(f"   - Model: {model}, DocID: {doc_id}, File: {doc.filename}")

        except Exception as e:
            print(f"Error scanning model directories: {e}")


    def add_document(
        self,
        filename: str,
        file_size: int,
        file_path: Path,
        model_name: str,  # NEW: Model name for this processing
        full_text: str,
        chunks: List['ChunkMetadata'],
        total_pages: int = 0,
        tokenizer: Optional[str] = None,
        doc_id: Optional[str] = None  # NEW: Optional doc_id for re-uploading
    ) -> str:
        """
        Add a new document collection to the manager (model-specific)

        Args:
            filename: Original filename
            file_size: Size in bytes
            file_path: Path where file is stored (should be in doc subdirectory)
            model_name: Model used for processing this document
            full_text: Complete extracted text content
            chunks: List of document chunks
            total_pages: Total number of pages in document
            tokenizer: Tokenizer used (if any)
            doc_id: Optional doc_id (reuse if re-uploading same file with different model)

        Returns:
            Document ID
        """
        # Generate or reuse doc_id
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_size=file_size,
            file_path=file_path,
            model_name=model_name,
            total_pages=total_pages,
            tokenizer=tokenizer
        )
        metadata.full_text = full_text

        # Add all chunks to the document
        for chunk in chunks:
            metadata.add_chunk(chunk)

        # Store with (model_name, doc_id) key
        self._documents[(model_name, doc_id)] = metadata

        # Save document metadata
        self._save_document_metadata(model_name, doc_id)

        return doc_id

    def get_document(self, doc_id: str, model_name: str) -> Optional[DocumentMetadata]:
        """Get document metadata by ID and model name"""
        return self._documents.get((model_name, doc_id))

    def get_document_content(self, doc_id: str, model_name: str) -> Optional[str]:
        """Get complete document text content by ID and model"""
        doc = self._documents.get((model_name, doc_id))
        return doc.full_text if doc else None

    def get_document_chunk(self, doc_id: str, model_name: str, chunk_id: int) -> Optional['ChunkMetadata']:
        """Get a specific chunk from a document"""
        doc = self._documents.get((model_name, doc_id))
        return doc.get_chunk(chunk_id) if doc else None

    def get_document_chunks(self, doc_id: str, model_name: str) -> Optional[List['ChunkMetadata']]:
        """Get all chunks from a document"""
        doc = self._documents.get((model_name, doc_id))
        return doc.get_all_chunks() if doc else None

    def get_available_models(self, doc_id: str) -> List[str]:
        """Get list of models that have processed this document"""
        models = []
        for (model, did), _ in self._documents.items():
            if did == doc_id:
                models.append(model)
        return models

    def has_model_metadata(self, doc_id: str, model_name: str) -> bool:
        """Check if a document has been processed by a specific model"""
        return (model_name, doc_id) in self._documents

    def list_documents(self, model_name: Optional[str] = None) -> List[dict]:
        """
        List all uploaded documents for a specific model

        Args:
            model_name: Model name to filter documents (now REQUIRED in practice)

        Returns:
            List of document info for the specified model (includes cache status)
        """
        if model_name:
            # Return only documents for this model
            docs = []
            for (model, doc_id), doc in self._documents.items():
                if model == model_name:
                    # Calculate cache statistics
                    total_groups = len(doc.groups)
                    cached_groups = sum(1 for g in doc.groups if g.get('cached', False))

                    docs.append({
                        "doc_id": doc_id,
                        "filename": doc.filename,
                        "file_size": doc.file_size,
                        "total_pages": doc.total_pages,
                        "uploaded_at": doc.uploaded_at.isoformat(),
                        "model_name": model,
                        "cached": doc.cached,
                        "total_groups": total_groups,
                        "cached_groups": cached_groups,
                        "last_cached_at": doc.last_cached_at.isoformat() if doc.last_cached_at else None
                    })
            return docs
        else:
            # Legacy support: return all documents (grouped by model)
            docs = []
            for (model, doc_id), doc in self._documents.items():
                # Calculate cache statistics
                total_groups = len(doc.groups)
                cached_groups = sum(1 for g in doc.groups if g.get('cached', False))

                docs.append({
                    "doc_id": doc_id,
                    "filename": doc.filename,
                    "file_size": doc.file_size,
                    "total_pages": doc.total_pages,
                    "uploaded_at": doc.uploaded_at.isoformat(),
                    "model_name": model,
                    "cached": doc.cached,
                    "total_groups": total_groups,
                    "cached_groups": cached_groups,
                    "last_cached_at": doc.last_cached_at.isoformat() if doc.last_cached_at else None
                })
            return docs

    def delete_document(self, doc_id: str, model_name: Optional[str] = None) -> bool:
        """
        Delete a document (specific model or all models)

        Args:
            doc_id: Document ID to delete
            model_name: Optional model name. If None, delete from all models

        Returns:
            True if deleted, False if not found
        """
        deleted = False

        if model_name:
            # Delete specific model metadata
            key = (model_name, doc_id)
            if key in self._documents:
                doc = self._documents[key]
                del self._documents[key]

                # Delete metadata file
                metadata_path = self._get_metadata_path(model_name, doc_id)
                if metadata_path.exists():
                    metadata_path.unlink()
                    print(f"🗑️  Deleted: {metadata_path}")

                # Delete document file
                if doc.file_path.exists():
                    doc.file_path.unlink()
                    print(f"🗑️  Deleted: {doc.file_path}")

                deleted = True
        else:
            # Delete all model metadata for this document
            to_delete = [(model, did) for (model, did) in self._documents.keys() if did == doc_id]

            for key in to_delete:
                model_name_val, doc_id_val = key
                doc = self._documents[key]
                del self._documents[key]

                # Delete metadata file
                metadata_path = self._get_metadata_path(model_name_val, doc_id_val)
                if metadata_path.exists():
                    metadata_path.unlink()

                # Delete document file
                if doc.file_path.exists():
                    doc.file_path.unlink()

                deleted = True

        return deleted

    def clear_all_documents(self) -> int:
        """
        Clear all documents from all models

        Returns:
            Number of document-model pairs cleared
        """
        count = len(self._documents)

        # Delete all model directories
        for model_dir in self.upload_dir.iterdir():
            if model_dir.is_dir():
                # Delete all files in model directory
                for file in model_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                # Delete model directory
                model_dir.rmdir()
                print(f"🗑️  Deleted model directory: {model_dir.name}")

        # Clear in-memory documents
        self._documents.clear()

        return count

    def clear_model_documents(self, model_name: str) -> int:
        """
        Clear all documents for a specific model (when model is reset)
        Simply deletes the entire model directory

        Args:
            model_name: Model serving name to clear documents for

        Returns:
            Number of documents cleared
        """
        model_dir = self._get_model_directory(model_name)

        if not model_dir.exists():
            print(f"📂 Model directory does not exist: {model_dir}")
            return 0

        # Count documents before deleting
        count = sum(1 for (model, _) in self._documents.keys() if model == model_name)

        # Delete all files in model directory
        for file in model_dir.iterdir():
            if file.is_file():
                file.unlink()
                print(f"🗑️  Deleted: {file.name}")

        # Delete model directory
        model_dir.rmdir()
        print(f"🗑️  Deleted model directory: {model_dir.name}")

        # Remove from in-memory storage
        to_delete = [(model, doc_id) for (model, doc_id) in self._documents.keys() if model == model_name]
        for key in to_delete:
            del self._documents[key]

        return count


# Global document manager instance
document_manager = DocumentManager(upload_dir=settings.documents.upload_dir)
