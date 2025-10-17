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
        token_count: Optional[int] = None
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.page_numbers = page_numbers
        self.char_count = char_count
        self.token_count = token_count  # Number of tokens in this chunk

    def to_dict(self, include_content: bool = True) -> dict:
        """Convert to dictionary format"""
        result = {
            "chunk_id": self.chunk_id,
            "page_numbers": self.page_numbers,
            "char_count": self.char_count,
            "token_count": self.token_count
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
            token_count=data.get("token_count")
        )


class DocumentMetadata:
    """
    Metadata for an uploaded document (collection)
    Each document is a collection that may contain multiple chunks
    """

    def __init__(
        self,
        doc_id: str,
        filename: str,
        file_size: int,
        file_path: Path,
        total_pages: int = 0,
        uploaded_at: Optional[datetime] = None,
        tokenizer: Optional[str] = None
    ):
        self.doc_id = doc_id
        self.filename = filename
        self.file_size = file_size
        self.file_path = file_path
        self.total_pages = total_pages
        self.uploaded_at = uploaded_at if uploaded_at else datetime.now()
        self.tokenizer = tokenizer  # Tokenizer used for this document
        self.full_text: Optional[str] = None
        self.chunks: List[ChunkMetadata] = []

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
            "total_pages": self.total_pages,
            "total_chunks": len(self.chunks),
            "uploaded_at": self.uploaded_at.isoformat(),
            "tokenizer": self.tokenizer
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
            "total_pages": self.total_pages,
            "uploaded_at": self.uploaded_at.isoformat(),
            "tokenizer": self.tokenizer,
            "chunks": [chunk.to_persistent_dict() for chunk in self.chunks]
        }

    @classmethod
    def from_persistent_dict(cls, data: dict) -> 'DocumentMetadata':
        """Create DocumentMetadata from persisted dictionary"""
        doc = cls(
            doc_id=data["doc_id"],
            filename=data["filename"],
            file_size=data["file_size"],
            file_path=Path(data["file_path"]),
            total_pages=data.get("total_pages", 0),
            uploaded_at=datetime.fromisoformat(data["uploaded_at"]),
            tokenizer=data.get("tokenizer")
        )

        # Restore chunks
        if "chunks" in data:
            for chunk_data in data["chunks"]:
                chunk = ChunkMetadata.from_persistent_dict(chunk_data)
                doc.add_chunk(chunk)

        return doc


class DocumentManager:
    """
    Manages uploaded documents independently from sessions
    Users can upload documents and select them when chatting

    Documents metadata is persisted to JSON file for persistence across restarts.
    """

    METADATA_FILENAME = "documents_metadata.json"

    def __init__(self, upload_dir: str = "uploads"):
        self._documents: Dict[str, DocumentMetadata] = {}
        self.upload_dir = upload_dir
        self.metadata_path = Path(upload_dir) / self.METADATA_FILENAME

        # Ensure upload directory exists
        os.makedirs(upload_dir, exist_ok=True)

        # Load existing documents metadata
        self._load_metadata()

    def _save_metadata(self):
        """Save documents metadata to JSON file"""
        try:
            metadata_dict = {
                doc_id: doc.to_persistent_dict()
                for doc_id, doc in self._documents.items()
            }

            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving document metadata: {e}")

    def _load_metadata(self):
        """Load documents metadata from JSON file"""
        if not self.metadata_path.exists():
            return

        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)

            for doc_id, doc_data in metadata_dict.items():
                try:
                    # Recreate DocumentMetadata object
                    metadata = DocumentMetadata.from_persistent_dict(doc_data)

                    # Check if file still exists
                    if not metadata.file_path.exists():
                        print(f"Warning: Document file not found, skipping: {metadata.file_path}")
                        continue

                    # Reconstruct full_text from chunks (for PDF documents)
                    if metadata.chunks:
                        metadata.full_text = "\n\n".join(chunk.content for chunk in metadata.chunks)

                    self._documents[doc_id] = metadata

                except Exception as e:
                    print(f"Error loading document {doc_id}: {e}")
                    continue

            print(f"Loaded {len(self._documents)} document(s) from metadata file")

        except Exception as e:
            print(f"Error loading document metadata: {e}")

    def add_document(
        self,
        filename: str,
        file_size: int,
        file_path: Path,
        full_text: str,
        chunks: List['ChunkMetadata'],
        total_pages: int = 0,
        tokenizer: Optional[str] = None
    ) -> str:
        """
        Add a new document collection to the manager

        Args:
            filename: Original filename
            file_size: Size in bytes
            file_path: Path where file is stored
            full_text: Complete extracted text content
            chunks: List of document chunks
            total_pages: Total number of pages in document

        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        metadata = DocumentMetadata(
            doc_id=doc_id,
            filename=filename,
            file_size=file_size,
            file_path=file_path,
            total_pages=total_pages,
            tokenizer=tokenizer
        )
        metadata.full_text = full_text

        # Add all chunks to the document
        for chunk in chunks:
            metadata.add_chunk(chunk)

        self._documents[doc_id] = metadata

        # Save metadata to persist changes
        self._save_metadata()

        return doc_id

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata by ID"""
        return self._documents.get(doc_id)

    def get_document_content(self, doc_id: str) -> Optional[str]:
        """Get complete document text content by ID"""
        doc = self._documents.get(doc_id)
        return doc.full_text if doc else None

    def get_document_chunk(self, doc_id: str, chunk_id: int) -> Optional['ChunkMetadata']:
        """Get a specific chunk from a document"""
        doc = self._documents.get(doc_id)
        return doc.get_chunk(chunk_id) if doc else None

    def get_document_chunks(self, doc_id: str) -> Optional[List['ChunkMetadata']]:
        """Get all chunks from a document"""
        doc = self._documents.get(doc_id)
        return doc.get_all_chunks() if doc else None

    def list_documents(self) -> List[dict]:
        """List all uploaded documents"""
        return [doc.to_dict() for doc in self._documents.values()]

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document

        Args:
            doc_id: Document ID to delete

        Returns:
            True if deleted, False if not found
        """
        if doc_id in self._documents:
            doc = self._documents[doc_id]
            # Only remove from metadata, keep original file
            del self._documents[doc_id]

            # Save metadata to persist changes
            self._save_metadata()

            return True
        return False

    def clear_all_documents(self) -> int:
        """
        Clear all documents (used when model is reset)

        Returns:
            Number of documents cleared
        """
        count = len(self._documents)

        # Delete all files
        for doc in self._documents.values():
            if doc.file_path.exists():
                doc.file_path.unlink()

        # Clear in-memory documents
        self._documents.clear()

        # Save empty metadata
        self._save_metadata()

        return count


# Global document manager instance
document_manager = DocumentManager(upload_dir=settings.documents.upload_dir)

