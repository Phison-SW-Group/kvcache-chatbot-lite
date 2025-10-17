"""
Document processing service
Designed to be extensible for multiple document formats
Supports PDF with chunking using LangChain RecursiveCharacterTextSplitter
Supports token counting using HuggingFace tokenizers
"""
from typing import Protocol, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""
    chunk_id: int
    content: str
    page_numbers: List[int]
    char_count: int
    token_count: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "page_numbers": self.page_numbers,
            "char_count": self.char_count,
            "token_count": self.token_count
        }


@dataclass
class ProcessedDocument:
    """Represents a processed document with chunks"""
    full_text: str
    chunks: List[DocumentChunk]
    total_pages: int
    total_chunks: int
    tokenizer: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "chunks": [chunk.to_dict() for chunk in self.chunks]
        }


class DocumentProcessor(Protocol):
    """Protocol for document processors"""

    async def process(self, file_path: Path) -> ProcessedDocument:
        """Process document and return processed document with chunks"""
        ...


class PdfProcessor:
    """Processor for PDF files with chunking support"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer: Optional[str] = None
    ):
        """
        Initialize PDF processor with chunking parameters

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            tokenizer: HuggingFace tokenizer identifier (optional)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer_name = tokenizer
        self.tokenizer = None

        # Initialize tokenizer if specified
        if tokenizer:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
                print(f"✅ Loaded tokenizer: {tokenizer}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load tokenizer '{tokenizer}': {e}")
                self.tokenizer = None
                self.tokenizer = None

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def count_tokens(self, text: str) -> Optional[int]:
        """
        Count tokens in text using the configured tokenizer

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens, or None if no tokenizer configured
        """
        if not self.tokenizer:
            return None

        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception as e:
            print(f"⚠️  Warning: Token counting failed: {e}")
            return None

    async def process(self, file_path: Path) -> ProcessedDocument:
        """
        Extract text from PDF and split into chunks

        Args:
            file_path: Path to the PDF file

        Returns:
            ProcessedDocument with chunks
        """
        # Extract text from PDF
        full_text = ""
        page_texts = []

        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            total_pages = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                page_texts.append((page_num + 1, page_text))
                full_text += page_text + "\n\n"

        # Split text into chunks
        text_chunks = self.text_splitter.split_text(full_text)

        # Create DocumentChunk objects with metadata
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            # Determine which pages this chunk spans
            page_numbers = self._get_page_numbers_for_chunk(chunk_text, page_texts)

            # Count tokens if tokenizer is available
            token_count = self.count_tokens(chunk_text)

            chunk = DocumentChunk(
                chunk_id=idx,
                content=chunk_text,
                page_numbers=page_numbers,
                char_count=len(chunk_text),
                token_count=token_count
            )
            chunks.append(chunk)

        return ProcessedDocument(
            full_text=full_text,
            chunks=chunks,
            total_pages=total_pages,
            total_chunks=len(chunks),
            tokenizer=self.tokenizer_name
        )

    def _get_page_numbers_for_chunk(
        self,
        chunk_text: str,
        page_texts: List[tuple[int, str]]
    ) -> List[int]:
        """
        Determine which pages a chunk comes from

        Args:
            chunk_text: The chunk text to analyze
            page_texts: List of (page_number, page_text) tuples

        Returns:
            List of page numbers that contain parts of this chunk
        """
        pages = []
        # Simple heuristic: check if first 100 chars of chunk appear in page
        chunk_start = chunk_text[:min(100, len(chunk_text))]

        for page_num, page_text in page_texts:
            if chunk_start in page_text:
                pages.append(page_num)
                break

        # If no match found, return page 1 as fallback
        return pages if pages else [1]


class DocumentService:
    """
    Service for processing uploaded documents
    Currently supports PDF with chunking
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer: Optional[str] = None
    ):
        """
        Initialize document service

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            tokenizer: HuggingFace tokenizer identifier (optional)
        """
        # Registry of processors by file extension
        self._processors: Dict[str, DocumentProcessor] = {
            '.pdf': PdfProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                tokenizer=tokenizer
            ),
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

    async def process_document(self, file_path: Path) -> ProcessedDocument:
        """
        Process a document and return its content with chunks

        Args:
            file_path: Path to the document file

        Returns:
            ProcessedDocument with chunks

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
        processed_doc = await processor.process(file_path)

        return processed_doc


# Global document service instance
# Import settings to get chunking parameters
from config import settings

# Get tokenizer if configured
tokenizer = settings.get_tokenizer_for_model()

document_service = DocumentService(
    chunk_size=settings.documents.chunk_size,
    chunk_overlap=settings.documents.chunk_overlap,
    tokenizer=tokenizer
)

