"""Cross-encoder reranker using bge-reranker-v2-m3.

Runs entirely locally via sentence-transformers — no API, no Ollama.
The model (~1.2 GB) downloads from HuggingFace on first use and is cached.

Cross-encoders read (query, passage) as a single sequence and output a
relevance score. That's slower than bi-encoder cosine (~20x per pair) but
qualitatively more accurate, so we use it only on a small pre-filtered set.
"""
from typing import Any

from sentence_transformers import CrossEncoder

from src.config import CFG
from src.retrieval.vector import RetrievedChunk


class Reranker:
    """Wraps bge-reranker-base for re-scoring retrieved chunks.

    Per-process singleton (model load is expensive) + per-session LRU cache
    on (query, chunk_id) so repeat queries within a session skip re-scoring.
    """

    _instance: "Reranker | None" = None
    _model: CrossEncoder | None = None
    # (query, chunk_id) -> score
    _score_cache: dict[tuple[str, str], float] = {}
    _CACHE_MAX = 2048

    def __new__(cls) -> "Reranker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> None:
        if Reranker._model is None:
            Reranker._model = CrossEncoder(
                CFG.models.reranker,
                max_length=512,
                trust_remote_code=True,
            )

    def _cache_key(self, query: str, chunk_id: str) -> tuple[str, str]:
        # Normalize the query so trivially-different phrasings share cache
        return (query.strip().lower(), chunk_id)

    def _evict_if_full(self) -> None:
        # Simple FIFO eviction — remove oldest 25% when full
        if len(Reranker._score_cache) >= Reranker._CACHE_MAX:
            keys_to_drop = list(Reranker._score_cache.keys())[: Reranker._CACHE_MAX // 4]
            for k in keys_to_drop:
                Reranker._score_cache.pop(k, None)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-score chunks against the query, return sorted top_k.

        Chunks with a cache hit skip the model. Only cache-miss chunks are
        sent to the cross-encoder in one batch.
        """
        if not chunks:
            return []

        self._ensure_loaded()
        k = top_k or CFG.retrieval.top_k_after_rerank

        # Partition into cache hits and misses
        cache_hits: dict[str, float] = {}
        miss_chunks: list[RetrievedChunk] = []

        for c in chunks:
            key = self._cache_key(query, c.chunk_id)
            if key in Reranker._score_cache:
                cache_hits[c.chunk_id] = Reranker._score_cache[key]
            else:
                miss_chunks.append(c)

        # Score misses in one batch
        if miss_chunks:
            pairs = [(query, c.content) for c in miss_chunks]
            new_scores = Reranker._model.predict(
                pairs,
                show_progress_bar=False,
                batch_size=8,
            )
            for c, score in zip(miss_chunks, new_scores):
                fscore = float(score)
                cache_hits[c.chunk_id] = fscore
                self._evict_if_full()
                Reranker._score_cache[self._cache_key(query, c.chunk_id)] = fscore

        # Build result list with scores
        reranked: list[RetrievedChunk] = []
        for chunk in chunks:
            new_meta = {**chunk.metadata, "_hybrid_score": chunk.score}
            r = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=new_meta,
                score=cache_hits[chunk.chunk_id],
                source="reranker",
            )
            reranked.append(r)

        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked[:k]