"""
Document processing service
Designed to be extensible for multiple document formats
"""
from typing import Protocol, Dict, Type
from pathlib import Path
import aiofiles


class DocumentProcessor(Protocol):
    """Protocol for document processors"""
    
    async def process(self, file_path: Path) -> str:
        """Process document and return text content"""
        ...


class TxtProcessor:
    """Processor for plain text files"""
    
    async def process(self, file_path: Path) -> str:
        """Read and return text file content"""
        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            content = await f.read()
        return content


class DocumentService:
    """
    Service for processing uploaded documents
    Extensible design for adding new document formats
    """
    
    def __init__(self):
        # Registry of processors by file extension
        self._processors: Dict[str, DocumentProcessor] = {
            '.txt': TxtProcessor(),
            # Future extensions:
            # '.pdf': PdfProcessor(),
            # '.docx': DocxProcessor(),
            # '.md': MarkdownProcessor(),
        }
    
    def register_processor(self, extension: str, processor: DocumentProcessor) -> None:
        """Register a new document processor for an extension"""
        self._processors[extension.lower()] = processor
    
    def supports_extension(self, extension: str) -> bool:
        """Check if extension is supported"""
        return extension.lower() in self._processors
    
    def get_supported_extensions(self) -> list[str]:
        """Get list of supported extensions"""
        return list(self._processors.keys())
    
    async def process_document(self, file_path: Path) -> str:
        """
        Process a document and return its text content
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If file extension is not supported
        """
        extension = file_path.suffix.lower()
        
        if not self.supports_extension(extension):
            supported = ', '.join(self.get_supported_extensions())
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Supported extensions: {supported}"
            )
        
        processor = self._processors[extension]
        content = await processor.process(file_path)
        
        return content


# Global document service instance
document_service = DocumentService()

