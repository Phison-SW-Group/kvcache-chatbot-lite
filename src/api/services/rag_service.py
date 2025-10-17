"""
RAG (Retrieval-Augmented Generation) Service
Handles similarity-based retrieval of document groups for query answering
"""
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import re
from collections import Counter, defaultdict
import math


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

    def __init__(self):
        self.bm25_k1 = 1.5
        self.bm25_b = 0.75

    def retrieve_most_similar_group(
        self,
        query: str,
        groups: List[Dict],
        top_k: int = 1
    ) -> List[RetrievalResult]:
        """
        Retrieve the most similar group(s) to the query using BM25

        Args:
            query: User query text
            groups: List of group dictionaries with 'group_id', 'merged_content', etc.
            top_k: Number of top results to return

        Returns:
            List of RetrievalResult sorted by similarity score (descending)
        """
        if not groups or not query.strip():
            return []

        # Tokenize query and documents
        query_tokens = self._tokenize(query)
        doc_tokens_list = [self._tokenize(g.get('merged_content', '')) for g in groups]

        # Build BM25 index
        doc_counters = [Counter(tokens) for tokens in doc_tokens_list]
        df = self._compute_document_frequency(doc_counters)
        avgdl = sum(sum(c.values()) for c in doc_counters) / max(len(doc_counters), 1)
        idf = self._compute_idf(df, len(groups))

        # Compute BM25 scores for each group
        scores = []
        for idx, (group, doc_counter) in enumerate(zip(groups, doc_counters)):
            score = self._bm25_score(query_tokens, doc_counter, avgdl, idf)
            scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

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

        return results

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenizer supporting English and CJK

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

