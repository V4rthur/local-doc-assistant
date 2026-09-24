"""BM25 lexical retrieval."""
from typing import Any

import numpy as np

from src.config import CFG
from src.indexing.bm25_store import BM25Store, _tokenize
from src.retrieval.vector import RetrievedChunk


class BM25Retriever:
    """Lexical search — strong on exact term/number matches."""

    def __init__(self) -> None:
        self._store = BM25Store()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top_k most lexically similar chunks.

        Args:
            query: The user's question.
            top_k: Number of results (default from config).
            where: Optional filter on metadata fields (supports {"field": value}
                   and {"field": {"$in": [...]}}). Applied AFTER scoring so BM25
                   scores are computed on the full corpus.
        """
        k = top_k or CFG.retrieval.top_k_bm25

        if self._store._bm25 is None or self._store.count() == 0:
            return []

        # 1. Tokenize the query with the same tokenizer used at index time
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # 2. Score every chunk in the corpus
        scores = self._store._bm25.get_scores(query_tokens)

        # 3. Apply the metadata `where` filter (post-scoring, pre-topk)
        if where:
            for i, meta in enumerate(self._store._chunks_meta):
                if not _match_where(meta, where):
                    scores[i] = -np.inf

        # 4. Grab top-k indices, but skip anything filtered out
        # np.argpartition is faster than argsort for large corpora
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            if scores[idx] == -np.inf:
                break  # sorted, so rest are all filtered
            meta = self._store._chunks_meta[idx]
            results.append(RetrievedChunk(
                chunk_id=meta["chunk_id"],
                content=meta["content"],
                metadata={k: v for k, v in meta.items() if k not in ("chunk_id", "content")},
                score=float(scores[idx]),
                source="bm25",
            ))
        return results


def _match_where(meta: dict[str, Any], where: dict[str, Any]) -> bool:
    """Minimal `where` matcher — supports equality and $in / $and."""
    # $and: every sub-clause must match
    if "$and" in where:
        return all(_match_where(meta, clause) for clause in where["$and"])
    # Otherwise treat as field constraints
    for field, constraint in where.items():
        value = meta.get(field)
        if isinstance(constraint, dict):
            if "$in" in constraint and value not in constraint["$in"]:
                return False
        elif value != constraint:
            return False
    return True