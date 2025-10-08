"""
Independent document management service
Documents are stored separately from sessions
"""
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import uuid
import json
import os

from config import settings


class DocumentMetadata:
    """Metadata for an uploaded document"""
    
    def __init__(self, doc_id: str, filename: str, file_size: int, file_path: Path, uploaded_at: Optional[datetime] = None):
        self.doc_id = doc_id
        self.filename = filename
        self.file_size = file_size
        self.file_path = file_path
        self.uploaded_at = uploaded_at if uploaded_at else datetime.now()
        self.content: Optional[str] = None
    
    def to_dict(self, include_preview: bool = False, preview_lines: int = 10) -> dict:
        """
        Convert to dictionary format (for API response)
        
        Args:
            include_preview: Whether to include content preview
            preview_lines: Number of lines to include in preview
        """
        result = {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "uploaded_at": self.uploaded_at.isoformat()
        }
        
        if include_preview and self.content:
            lines = self.content.split('\n')
            preview_text = '\n'.join(lines[:preview_lines])
            result["content_preview"] = preview_text
            result["total_lines"] = len(lines)
        
        return result
    
    def to_persistent_dict(self) -> dict:
        """Convert to dictionary format for persistence (includes file_path)"""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "file_path": str(self.file_path),
            "uploaded_at": self.uploaded_at.isoformat()
        }
    
    @classmethod
    def from_persistent_dict(cls, data: dict) -> 'DocumentMetadata':
        """Create DocumentMetadata from persisted dictionary"""
        return cls(
            doc_id=data["doc_id"],
            filename=data["filename"],
            file_size=data["file_size"],
            file_path=Path(data["file_path"]),
            uploaded_at=datetime.fromisoformat(data["uploaded_at"])
        )


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
                    
                    # Reload content from file
                    try:
                        with open(metadata.file_path, 'r', encoding='utf-8') as f:
                            metadata.content = f.read()
                    except Exception as e:
                        print(f"Warning: Could not read document content from {metadata.file_path}: {e}")
                        metadata.content = ""
                    
                    self._documents[doc_id] = metadata
                    
                except Exception as e:
                    print(f"Error loading document {doc_id}: {e}")
                    continue
            
            print(f"Loaded {len(self._documents)} document(s) from metadata file")
            
        except Exception as e:
            print(f"Error loading document metadata: {e}")
    
    def add_document(self, filename: str, file_size: int, file_path: Path, content: str) -> str:
        """
        Add a new document to the manager
        
        Args:
            filename: Original filename
            file_size: Size in bytes
            file_path: Path where file is stored
            content: Extracted text content
            
        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        metadata = DocumentMetadata(doc_id, filename, file_size, file_path)
        metadata.content = content
        self._documents[doc_id] = metadata
        
        # Save metadata to persist changes
        self._save_metadata()
        
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata by ID"""
        return self._documents.get(doc_id)
    
    def get_document_content(self, doc_id: str) -> Optional[str]:
        """Get document text content by ID"""
        doc = self._documents.get(doc_id)
        return doc.content if doc else None
    
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
            # Clean up file
            if doc.file_path.exists():
                doc.file_path.unlink()
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
document_manager = DocumentManager(upload_dir=settings.UPLOAD_DIR)

