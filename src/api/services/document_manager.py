"""
Independent document management service
Documents are stored separately from sessions
"""
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
import uuid


class DocumentMetadata:
    """Metadata for an uploaded document"""
    
    def __init__(self, doc_id: str, filename: str, file_size: int, file_path: Path):
        self.doc_id = doc_id
        self.filename = filename
        self.file_size = file_size
        self.file_path = file_path
        self.uploaded_at = datetime.now()
        self.content: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "uploaded_at": self.uploaded_at.isoformat()
        }


class DocumentManager:
    """
    Manages uploaded documents independently from sessions
    Users can upload documents and select them when chatting
    """
    
    def __init__(self):
        self._documents: Dict[str, DocumentMetadata] = {}
    
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
            return True
        return False


# Global document manager instance
document_manager = DocumentManager()

