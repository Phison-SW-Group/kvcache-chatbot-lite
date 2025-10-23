"""
RAG (Retrieval-Augmented Generation) Service
Handles similarity-based retrieval of document groups for query answering
"""
import re
import math
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from collections import Counter, defaultdict

from services.tokenizer import get_tokenizer, Tokenizer

# Set up logger
logger = logging.getLogger(__name__)

# Ensure logger level is set to INFO to show RAG logs
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class RetrievalResult:
    """Result of similarity-based retrieval"""
    group_id: str
    content: str
    similarity_score: float
    chunk_ids: List[int]
    total_tokens: int
    # New fields for chunk-based retrieval
    matched_chunk_id: Optional[int] = None  # The chunk that matched the query
    matched_chunk_content: Optional[str] = None  # Content of the matched chunk
    matched_chunk_metadata: Optional[Dict] = None  # Full metadata of matched chunk


class RAGService:
    """
    Service for RAG operations:
    - Similarity-based retrieval of document groups
    - Query-to-group matching using BM25
    """

    def __init__(self, filter_stopwords: bool = False):
        """
        Initialize RAG service

        Args:
            filter_stopwords: Whether to filter out stopwords (default: False)
        """
        self.bm25_k1 = 1.5
        self.bm25_b = 0.75

        # Initialize tokenizer
        self.tokenizer = get_tokenizer(filter_stopwords=filter_stopwords)

    def retrieve_most_similar_group(
        self,
        query: str,
        groups: List[Dict],
        top_k: int = 1,
        document_name: str = "Unknown"
    ) -> List[RetrievalResult]:
        """
        Retrieve the most similar group(s) to the query using BM25

        Args:
            query: User query text
            groups: List of group dictionaries with 'group_id', 'merged_content', etc.
            top_k: Number of top results to return
            document_name: Name of the document being searched (for logging)

        Returns:
            List of RetrievalResult sorted by similarity score (descending)
        """
        if not groups or not query.strip():
            logger.warning("RAG: Empty query or groups provided")
            return []

        # Log query and document info
        logger.info(f"RAG: Starting retrieval for document '{document_name}'")
        logger.info(f"RAG: Query: '{query}'")
        logger.info(f"RAG: Available groups: {len(groups)}")

        # Tokenize query and documents
        query_tokens = self._tokenize(query)
        doc_tokens_list = [self._tokenize(g.get('merged_content', '')) for g in groups]

        # Log tokenization results
        logger.info(f"RAG: Query tokens ({len(query_tokens)}): {query_tokens}")

        # Build BM25 index
        doc_counters = [Counter(tokens) for tokens in doc_tokens_list]
        df = self._compute_document_frequency(doc_counters)
        avgdl = sum(sum(c.values()) for c in doc_counters) / max(len(doc_counters), 1)
        idf = self._compute_idf(df, len(groups))

        # Compute BM25 scores for each group
        scores = []
        logger.info("RAG: Computing BM25 scores for each group:")
        logger.info("-" * 80)

        for idx, (group, doc_counter) in enumerate(zip(groups, doc_counters)):
            score = self._bm25_score(query_tokens, doc_counter, avgdl, idf)
            scores.append((idx, score))

            # Log detailed group information
            group_id = group.get('group_id', f'group_{idx}')
            chunk_ids = group.get('chunk_ids', [])
            total_tokens = group.get('total_tokens', 0)
            content_preview = group.get('merged_content', '')[:100] + "..." if len(group.get('merged_content', '')) > 100 else group.get('merged_content', '')

            logger.info(f"RAG: Group {idx+1}/{len(groups)}")
            logger.info(f"  Group ID: {group_id}")
            logger.info(f"  Chunk IDs: {chunk_ids}")
            logger.info(f"  Total Tokens: {total_tokens}")
            logger.info(f"  BM25 Score: {score:.6f}")
            logger.info(f"  Content Preview: {content_preview}")
            logger.info("-" * 40)

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Log ranking results
        logger.info("RAG: Final ranking (sorted by BM25 score):")
        for rank, (idx, score) in enumerate(scores, 1):
            group = groups[idx]
            group_id = group.get('group_id', f'group_{idx}')
            logger.info(f"  Rank {rank}: {group_id} (Score: {score:.6f})")

        # Return top_k results
        results = []
        for idx, score in scores[:top_k]:
            group = groups[idx]
            results.append(RetrievalResult(
                group_id=group.get('group_id', ''),
                content=group.get('merged_content', ''),
                similarity_score=score,
                chunk_ids=group.get('chunk_ids', []),
                total_tokens=group.get('total_tokens', 0)
            ))

        # Log final results
        logger.info(f"RAG: Returning top {len(results)} results")
        for i, result in enumerate(results, 1):
            logger.info(f"  Result {i}: {result.group_id} (Score: {result.similarity_score:.6f})")

        logger.info("RAG: Retrieval completed")
        logger.info("=" * 80)

        return results

    def retrieve_by_chunk_with_group(
        self,
        query: str,
        chunks: List[Dict],
        groups: List[Dict],
        top_k: int = 1,
        document_name: str = "Unknown"
    ) -> List[RetrievalResult]:
        """
        Retrieve by matching chunks first, then mapping to their parent groups

        This method:
        1. Performs BM25 similarity search on individual chunks
        2. Finds the top matching chunk(s)
        3. Maps each matched chunk to its parent group
        4. Returns the group content for LLM + matched chunk info for frontend preview

        Args:
            query: User query text
            chunks: List of chunk dictionaries with 'chunk_id', 'content', 'group_id', etc.
            groups: List of group dictionaries with 'group_id', 'merged_content', etc.
            top_k: Number of top results to return
            document_name: Name of the document being searched (for logging)

        Returns:
            List of RetrievalResult with both group content and matched chunk info
        """
        if not chunks or not groups or not query.strip():
            logger.warning("RAG: Empty query, chunks, or groups provided")
            return []

        # Log query and document info
        logger.info(f"RAG (Chunk-based): Starting retrieval for document '{document_name}'")
        logger.info(f"RAG (Chunk-based): Query: '{query}'")
        logger.info(f"RAG (Chunk-based): Available chunks: {len(chunks)}, groups: {len(groups)}")

        # Tokenize query and chunks
        query_tokens = self._tokenize(query)
        chunk_tokens_list = [self._tokenize(c.get('content', '')) for c in chunks]

        # Log tokenization results
        logger.info(f"RAG (Chunk-based): Query tokens ({len(query_tokens)}): {query_tokens}")

        # Build BM25 index for chunks
        chunk_counters = [Counter(tokens) for tokens in chunk_tokens_list]
        df = self._compute_document_frequency(chunk_counters)
        avgdl = sum(sum(c.values()) for c in chunk_counters) / max(len(chunk_counters), 1)
        idf = self._compute_idf(df, len(chunks))

        # Compute BM25 scores for each chunk
        scores = []
        logger.info("RAG (Chunk-based): Computing BM25 scores for each chunk:")
        logger.info("-" * 80)

        for idx, (chunk, chunk_counter) in enumerate(zip(chunks, chunk_counters)):
            score = self._bm25_score(query_tokens, chunk_counter, avgdl, idf)
            scores.append((idx, score))

            # Log detailed chunk information (only for top candidates)
            if idx < 10 or score > 0:  # Limit logging
                chunk_id = chunk.get('chunk_id', idx)
                group_id = chunk.get('group_id', 'unknown')
                content_preview = chunk.get('content', '')[:80] + "..." if len(chunk.get('content', '')) > 80 else chunk.get('content', '')

                logger.info(f"RAG (Chunk-based): Chunk {chunk_id}")
                logger.info(f"  Group ID: {group_id}")
                logger.info(f"  BM25 Score: {score:.6f}")
                logger.info(f"  Content Preview: {content_preview}")
                logger.info("-" * 40)

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Log ranking results
        logger.info("RAG (Chunk-based): Top 5 chunks by BM25 score:")
        for rank, (idx, score) in enumerate(scores[:5], 1):
            chunk = chunks[idx]
            chunk_id = chunk.get('chunk_id', idx)
            group_id = chunk.get('group_id', 'unknown')
            logger.info(f"  Rank {rank}: Chunk {chunk_id} in {group_id} (Score: {score:.6f})")

        # Create a mapping from group_id to group dict for quick lookup
        group_map = {g.get('group_id', ''): g for g in groups}

        # Return top_k results
        results = []
        for idx, score in scores[:top_k]:
            chunk = chunks[idx]
            chunk_group_id = chunk.get('group_id', '')
            chunk_id = chunk.get('chunk_id', idx)
            chunk_content = chunk.get('content', '')

            # Find the corresponding group
            parent_group = group_map.get(chunk_group_id)

            if not parent_group:
                logger.warning(f"RAG (Chunk-based): Chunk {chunk_id} has group_id '{chunk_group_id}' but no matching group found!")
                logger.warning(f"  Available group_ids: {list(group_map.keys())}")
                continue

            # Create result with both group content and matched chunk info
            result = RetrievalResult(
                group_id=parent_group.get('group_id', ''),
                content=parent_group.get('merged_content', ''),  # Full group content for LLM
                similarity_score=score,
                chunk_ids=parent_group.get('chunk_ids', []),
                total_tokens=parent_group.get('total_tokens', 0),
                # Matched chunk info for frontend preview
                matched_chunk_id=chunk_id,
                matched_chunk_content=chunk_content,
                matched_chunk_metadata=chunk
            )
            results.append(result)

            logger.info(f"RAG (Chunk-based): Mapped chunk {chunk_id} -> group {parent_group.get('group_id', '')}")

        # Log final results
        logger.info(f"RAG (Chunk-based): Returning {len(results)} results")
        for i, result in enumerate(results, 1):
            logger.info(f"  Result {i}: Chunk {result.matched_chunk_id} -> Group {result.group_id} (Score: {result.similarity_score:.6f})")

        logger.info("RAG (Chunk-based): Retrieval completed")
        logger.info("=" * 80)

        return results

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with support for mixed English and Chinese

        Uses advanced tokenizer (jieba + opencc) if available,
        falls back to basic regex-based tokenization otherwise.
        """
        if not text:
            return []

        # Use advanced tokenizer if available
        if self.tokenizer:
            return self.tokenizer.tokenize(text)

        # Fallback: basic tokenization
        text = text.lower()
        # Extract English words
        english_words = re.findall(r'[a-z]+', text)
        # Extract Chinese characters
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        # Extract numbers
        numbers = re.findall(r'\d+', text)
        # Combine all tokens
        tokens = english_words + chinese_chars + numbers
        return [t for t in tokens if t]

    def _compute_document_frequency(self, doc_counters: List[Counter]) -> Dict[str, int]:
        """Compute document frequency for each term"""
        df = defaultdict(int)
        for counter in doc_counters:
            for term in counter.keys():
                df[term] += 1
        return dict(df)

    def _compute_idf(self, df: Dict[str, int], n_docs: int) -> Dict[str, float]:
        """Compute IDF scores with smoothing (standard BM25 formula)"""
        idf = {}
        for term, df_t in df.items():
            idf[term] = math.log((n_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)
        return idf

    def _bm25_score(
        self,
        query_tokens: List[str],
        doc_counter: Counter,
        avgdl: float,
        idf: Dict[str, float]
    ) -> float:
        """Compute BM25 score for a query against a document"""
        dl = sum(doc_counter.values()) or 1
        score = 0.0

        for term in query_tokens:
            if term not in idf:
                continue

            f = doc_counter.get(term, 0)
            if f == 0:
                continue

            idf_score = idf[term]
            denom = f + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * dl / avgdl)
            score += idf_score * (f * (self.bm25_k1 + 1)) / denom

        return score


# Global RAG service instance
rag_service = RAGService()

