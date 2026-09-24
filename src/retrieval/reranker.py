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
    """Wraps bge-reranker-v2-m3 for re-scoring retrieved chunks."""

    _instance: "Reranker | None" = None
    _model: CrossEncoder | None = None

    def __new__(cls) -> "Reranker":
        # Singleton — loading the model takes seconds and eats ~2GB RAM.
        # We only want one copy in the process.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> None:
        if Reranker._model is None:
            # trust_remote_code needed for bge-reranker-v2-m3 architecture
            Reranker._model = CrossEncoder(
                CFG.models.reranker,
                max_length=512,
                trust_remote_code=True,
            )

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-score chunks against the query, return sorted top_k."""
        if not chunks:
            return []

        self._ensure_loaded()
        k = top_k or CFG.retrieval.top_k_after_rerank

        pairs = [(query, c.content) for c in chunks]

        # activation_fn=None → raw logits (default for CrossEncoder)
        # truncation is handled by the tokenizer via max_length in _ensure_loaded
        scores = Reranker._model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=8,
        )

        reranked: list[RetrievedChunk] = []
        for chunk, new_score in zip(chunks, scores):
            new_meta = {**chunk.metadata, "_hybrid_score": chunk.score}
            r = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=new_meta,
                score=float(new_score),
                source="reranker",
            )
            reranked.append(r)

        reranked.sort(key=lambda c: c.score, reverse=True)
        return reranked[:k]