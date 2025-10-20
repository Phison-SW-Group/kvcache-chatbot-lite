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

