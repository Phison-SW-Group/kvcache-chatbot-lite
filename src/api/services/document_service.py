"""
Document processing service
Designed to be extensible for multiple document formats
Supports PDF with chunking using LangChain RecursiveCharacterTextSplitter
Supports token counting using HuggingFace tokenizers
Supports first-stage sequential grouping based on token budget
"""
from typing import Protocol, Dict, List, Optional, Tuple, Iterable
from pathlib import Path
from dataclasses import dataclass
import pypdf
import math
import re
from collections import Counter, defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""
    chunk_id: int
    content: str
    page_numbers: List[int]
    char_count: int
    token_count: Optional[int] = None
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None
    group_id: Optional[str] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    page_key: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
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


@dataclass
class MergedGroup:
    """Represents a merged group of chunks"""
    group_id: str
    chunk_ids: List[int]
    total_tokens: int
    merged_content: str

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            "group_id": self.group_id,
            "chunk_ids": self.chunk_ids,
            "total_tokens": self.total_tokens,
            "content_length": len(self.merged_content),
            "merged_content": self.merged_content  # Include content for RAG retrieval
        }


@dataclass
class ProcessedDocument:
    """Represents a processed document with chunks"""
    full_text: str
    chunks: List[DocumentChunk]
    total_pages: int
    total_chunks: int
    tokenizer: Optional[str] = None
    groups: Optional[List[MergedGroup]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        result = {
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "chunks": [chunk.to_dict() for chunk in self.chunks]
        }
        if self.groups is not None:
            result["total_groups"] = len(self.groups)
            result["groups"] = [group.to_dict() for group in self.groups]
        return result


class DocumentProcessor(Protocol):
    """Protocol for document processors"""

    async def process(self, file_path: Path) -> ProcessedDocument:
        """Process document and return processed document with chunks"""
        ...


# ============================================================================
# Similarity Scoring Interfaces
# ============================================================================

class SimilarityScorer(Protocol):
    """Protocol for computing pairwise similarity among texts.

    Returns a symmetric square matrix where entry (i,j) is the similarity between
    texts[i] and texts[j] in [0, +inf) (BM25 is unbounded; consumers should only
    compare relative magnitudes).
    """

    def compute_similarity_matrix(self, texts: List[str]) -> List[List[float]]:
        ...


def default_tokenize(text: str) -> List[str]:
    """Simple tokenizer supporting English and CJK without heavy deps.

    - Lowercases
    - Splits on non-word characters while preserving CJK ranges
    - Removes empty tokens
    """
    if not text:
        return []
    text = text.lower()
    # Keep alphanumerics and CJK; replace others with space
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text)
    tokens = text.strip().split()
    return [t for t in tokens if t]


class BM25Scorer:
    """Minimal BM25 implementation for pairwise similarity.

    BM25 formula (Okapi):
        score(q, d) = sum_idf_over_terms_in_q( idf(t) * f(t,d) * (k1+1) / (f(t,d) + k1*(1-b + b*|d|/avgdl)) )

    For pairwise similarity matrix, we compute symmetric score as the average of
    BM25(query=doc_i, doc=doc_j) and BM25(query=doc_j, doc=doc_i).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenize: callable = default_tokenize) -> None:
        self.k1 = k1
        self.b = b
        self.tokenize = tokenize

    def _build_index(self, texts: List[str]) -> tuple[List[List[str]], List[Counter], dict, float]:
        token_lists: List[List[str]] = [self.tokenize(t) for t in texts]
        counters: List[Counter] = [Counter(toks) for toks in token_lists]
        df: dict[str, int] = defaultdict(int)
        for c in counters:
            for term in c.keys():
                df[term] += 1
        avgdl = sum(sum(c.values()) for c in counters) / max(len(counters), 1)
        self._idf_cache = self._compute_idf(df, len(counters))
        return token_lists, counters, self._idf_cache, avgdl

    @staticmethod
    def _compute_idf(df: dict, n_docs: int) -> dict:
        # BM25 idf with +0.5 smoothing
        idf = {}
        for term, df_t in df.items():
            idf[term] = math.log((n_docs - df_t + 0.5) / (df_t + 0.5) + 1e-12)
        return idf

    def _bm25(self, query_terms: Iterable[str], doc_counts: Counter, avgdl: float) -> float:
        dl = sum(doc_counts.values()) or 1
        score = 0.0
        for t in query_terms:
            if t not in self._idf_cache:
                continue
            f = doc_counts.get(t, 0)
            if f == 0:
                continue
            idf = self._idf_cache[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def compute_similarity_matrix(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        token_lists, counters, _idf, avgdl = self._build_index(texts)
        n = len(texts)
        sims: List[List[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    sims[i][j] = 1.0
                    continue
                s_ij = self._bm25(token_lists[i], counters[j], avgdl)
                s_ji = self._bm25(token_lists[j], counters[i], avgdl)
                s = 0.5 * (s_ij + s_ji)
                sims[i][j] = s
                sims[j][i] = s
        return sims

# ============================================================================
# Second-Stage Similarity Grouping
# ============================================================================

def group_remaining_by_similarity(
    remaining_chunks: List[DocumentChunk],
    file_max_tokens: int,
    scorer: SimilarityScorer | None = None,
    min_similarity: float | None = None
) -> Tuple[List[MergedGroup], List[DocumentChunk]]:
    """
    Group remaining chunks by similarity with token budget.

    Args:
        remaining_chunks: chunks without group_id after stage-1
        file_max_tokens: max tokens per group
        scorer: SimilarityScorer implementation (default BM25Scorer)
        min_similarity: optional minimum similarity to include a candidate

    Returns:
        (new_groups, leftover_chunks)
    """
    if not remaining_chunks:
        return [], []

    scorer = scorer or BM25Scorer()

    texts = [c.content for c in remaining_chunks]
    sims = scorer.compute_similarity_matrix(texts)

    n = len(remaining_chunks)
    used = [False] * n
    groups: List[MergedGroup] = []

    def capacity_ok(current_tokens: int, add_tokens: int) -> bool:
        return current_tokens + add_tokens <= file_max_tokens

    group_counter = 100000  # offset to avoid colliding with stage-1 ids

    for i in range(n):
        if used[i]:
            continue
        # seed group with i
        seed = remaining_chunks[i]
        current = [i]
        current_tokens = seed.token_count or 0
        used[i] = True

        # get candidates sorted by similarity desc
        candidates = sorted(
            [j for j in range(n) if not used[j] and j != i],
            key=lambda j: sims[i][j],
            reverse=True,
        )
        for j in candidates:
            sim = sims[i][j]
            if min_similarity is not None and sim < min_similarity:
                continue
            t = remaining_chunks[j].token_count or 0
            if capacity_ok(current_tokens, t):
                current.append(j)
                current_tokens += t
                used[j] = True

        # materialize group
        selected = [remaining_chunks[idx] for idx in current]
        group_id = f"similarity_group_{group_counter}"
        merged_content = "\n\n=== SIMILARITY SEPARATOR ===\n\n".join([c.content for c in selected])
        total_tokens = sum(c.token_count or 0 for c in selected)
        chunk_ids = [c.chunk_id for c in selected]
        # set group_id back
        for c in selected:
            c.group_id = group_id
        groups.append(MergedGroup(group_id=group_id, chunk_ids=chunk_ids, total_tokens=total_tokens, merged_content=merged_content))
        group_counter += 1

    # any ungrouped leftovers (should be none unless capacity 0)
    leftovers = [remaining_chunks[i] for i in range(n) if not used[i]]
    return groups, leftovers


# ============================================================================
# First-Stage Sequential Grouping (Token-based)
# ============================================================================

def group_chunks_by_token_budget(
    chunks: List[DocumentChunk],
    file_max_tokens: int,
    utilization_threshold: float = 0.8
) -> Tuple[List[MergedGroup], List[DocumentChunk]]:
    """
    First-stage grouping: merge chunks from same file sequentially within token budget.

    Algorithm:
    1. Group chunks by source_file
    2. For each file, accumulate chunks sequentially until hitting token limit
    3. If last group has utilization < threshold, leave chunks ungrouped for stage 2

    Args:
        chunks: List of DocumentChunk with token_count filled
        file_max_tokens: Maximum tokens per group
        utilization_threshold: Minimum utilization rate for final group (default 0.8)

    Returns:
        Tuple of (merged_groups, remaining_chunks_for_stage2)
    """
    if not chunks:
        return [], []

    # Group chunks by source_file
    file_chunks: Dict[str, List[DocumentChunk]] = {}
    for chunk in chunks:
        source = chunk.source_file or "unknown"
        if source not in file_chunks:
            file_chunks[source] = []
        file_chunks[source].append(chunk)

    # Sort chunks within each file by chunk_index
    for source in file_chunks:
        file_chunks[source].sort(key=lambda c: c.chunk_index if c.chunk_index is not None else c.chunk_id)

    merged_groups: List[MergedGroup] = []
    remaining_chunks: List[DocumentChunk] = []
    group_counter = 0

    for source_file, file_chunk_list in file_chunks.items():
        current_group_chunks: List[DocumentChunk] = []
        current_tokens = 0

        for chunk in file_chunk_list:
            chunk_tokens = chunk.token_count if chunk.token_count is not None else 0

            # Check if adding this chunk exceeds budget
            if current_tokens + chunk_tokens > file_max_tokens and current_group_chunks:
                # Save current group
                utilization = current_tokens / file_max_tokens
                if utilization >= utilization_threshold:
                    # High utilization: create group
                    group = _create_merged_group(current_group_chunks, group_counter)
                    merged_groups.append(group)
                    group_counter += 1
                else:
                    # Low utilization: add to remaining
                    remaining_chunks.extend(current_group_chunks)

                # Start new group
                current_group_chunks = [chunk]
                current_tokens = chunk_tokens
            else:
                # Add to current group
                current_group_chunks.append(chunk)
                current_tokens += chunk_tokens

        # Handle last group for this file
        if current_group_chunks:
            utilization = current_tokens / file_max_tokens
            if utilization >= utilization_threshold:
                group = _create_merged_group(current_group_chunks, group_counter)
                merged_groups.append(group)
                group_counter += 1
            else:
                # Low utilization: add to remaining for stage 2
                remaining_chunks.extend(current_group_chunks)

    return merged_groups, remaining_chunks


def _create_merged_group(chunks: List[DocumentChunk], group_id_num: int) -> MergedGroup:
    """
    Create a MergedGroup from a list of chunks

    Args:
        chunks: List of chunks to merge
        group_id_num: Numeric group identifier

    Returns:
        MergedGroup with merged content
    """
    if not chunks:
        raise ValueError("Cannot create group from empty chunk list")

    # Generate group_id
    first_chunk = chunks[0]
    source_file = first_chunk.source_file or "unknown"
    group_id = f"{source_file}_group{group_id_num}"

    # Merge content with separator
    merged_content = "\n\n=== CHUNK SEPARATOR ===\n\n".join([c.content for c in chunks])

    # Calculate total tokens
    total_tokens = sum(c.token_count for c in chunks if c.token_count is not None)

    # Extract chunk IDs
    chunk_ids = [c.chunk_id for c in chunks]

    # Update chunk.group_id for all chunks in this group
    for chunk in chunks:
        chunk.group_id = group_id

    return MergedGroup(
        group_id=group_id,
        chunk_ids=chunk_ids,
        total_tokens=total_tokens,
        merged_content=merged_content
    )

# ============================================================================
# Document Processing Classes
# ============================================================================

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
        # Don't initialize tokenizer here - use tokenizer_manager instead

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def count_tokens(self, text: str) -> Optional[int]:
        """
        Count tokens in text using tokenizer_manager

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens, or None if no tokenizer loaded
        """
        from services.tokenizer_manager import tokenizer_manager
        return tokenizer_manager.count_tokens(text)

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
        source_file = file_path.name

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

            # Compute start/end page from page_numbers
            start_page = min(page_numbers) if page_numbers else 1
            end_page = max(page_numbers) if page_numbers else 1
            page_key = f"{source_file}-page{start_page}"

            chunk = DocumentChunk(
                chunk_id=idx,
                content=chunk_text,
                page_numbers=page_numbers,
                char_count=len(chunk_text),
                token_count=token_count,
                source_file=source_file,
                chunk_index=idx,
                start_page=start_page,
                end_page=end_page,
                page_key=page_key
            )
            chunks.append(chunk)

        # Get current tokenizer name from tokenizer_manager
        from services.tokenizer_manager import tokenizer_manager
        current_tokenizer = tokenizer_manager.tokenizer_name if tokenizer_manager.is_loaded() else None

        return ProcessedDocument(
            full_text=full_text,
            chunks=chunks,
            total_pages=total_pages,
            total_chunks=len(chunks),
            tokenizer=current_tokenizer
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
    Supports first-stage token-based grouping when tokenizer is available
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer: Optional[str] = None,
        grouping: bool = True,
        file_max_tokens: int = 13000,
        utilization_threshold: float = 0.8
    ):
        """
        Initialize document service

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            tokenizer: HuggingFace tokenizer identifier (optional)
            grouping: Enable first-stage token-based grouping (default: True)
            file_max_tokens: Maximum tokens per group (default: 13000)
            utilization_threshold: Minimum utilization for final group (default: 0.8)
        """
        # Registry of processors by file extension
        self._processors: Dict[str, DocumentProcessor] = {
            '.pdf': PdfProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                tokenizer=tokenizer
            ),
        }
        self.grouping = grouping
        self.file_max_tokens = file_max_tokens
        self.utilization_threshold = utilization_threshold

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
        If tokenizer is available and grouping is enabled, performs first-stage grouping

        Args:
            file_path: Path to the document file

        Returns:
            ProcessedDocument with chunks and optional groups

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

        # Trigger grouping if enabled and tokenizer is available
        if self.grouping:
            if processed_doc.tokenizer is not None:
                print(f"✅ Tokenizer detected: {processed_doc.tokenizer}")
                print(f"🔄 Triggering first-stage token-based grouping (max_tokens={self.file_max_tokens})...")

                # Perform first-stage grouping
                groups, remaining = group_chunks_by_token_budget(
                    chunks=processed_doc.chunks,
                    file_max_tokens=self.file_max_tokens,
                    utilization_threshold=self.utilization_threshold
                )

                # Update processed_doc with groups
                processed_doc.groups = groups

                print(f"✅ First-stage grouping completed:")
                print(f"   - Total groups: {len(groups)}")
                print(f"   - Remaining chunks for stage 2: {len(remaining)}")

                if groups:
                    total_grouped_tokens = sum(g.total_tokens for g in groups)
                    avg_tokens = total_grouped_tokens / len(groups)
                    print(f"   - Total grouped tokens: {total_grouped_tokens:,}")
                    print(f"   - Average tokens per group: {avg_tokens:.0f}")

                # Stage-2 similarity grouping for remaining chunks (no group_id)
                from config import settings as _settings
                remaining = [c for c in processed_doc.chunks if not c.group_id]
                if remaining:
                    print(f"🔎 Stage-2 similarity grouping on {len(remaining)} remaining chunks...")
                    new_groups, leftovers = group_remaining_by_similarity(
                        remaining_chunks=remaining,
                        file_max_tokens=self.file_max_tokens,
                        scorer=None,  # default BM25
                        min_similarity=_settings.documents.similarity_min_score,
                    )
                    if processed_doc.groups is None:
                        processed_doc.groups = []
                    processed_doc.groups.extend(new_groups)
                    print(f"✅ Stage-2 created {len(new_groups)} groups; leftovers: {len(leftovers)}")
            else:
                print(f"⚠️  WARNING: Grouping is enabled but no tokenizer is configured")
                print(f"⚠️  WARNING: Skipping document grouping - chunks will remain ungrouped")
                print(f"⚠️  WARNING: To enable grouping, configure a tokenizer in your model settings")

        return processed_doc


# Global document service instance
# Import settings to get chunking parameters
from config import settings
from services.tokenizer_manager import tokenizer_manager

# Get tokenizer from tokenizer manager (dynamically loaded)
def _get_current_tokenizer():
    """Get tokenizer name from tokenizer manager"""
    return tokenizer_manager.tokenizer_name if tokenizer_manager.is_loaded() else None

# Create document service with dynamic tokenizer
# Note: tokenizer_name will be retrieved at runtime from tokenizer_manager
document_service = DocumentService(
    chunk_size=settings.documents.chunk_size,
    chunk_overlap=settings.documents.chunk_overlap,
    tokenizer=None,  # Will be set dynamically
    grouping=settings.documents.grouping,
    file_max_tokens=settings.documents.file_max_tokens,
    utilization_threshold=settings.documents.utilization_threshold
)
