"""Hybrid retrieval = vector + BM25 fused with Reciprocal Rank Fusion (RRF).

RRF is rank-based, not score-based, which sidesteps the incompatible scales
between cosine similarity (0-1) and BM25 (unbounded). Formula:

    fused_score(d) = Σ  1 / (k + rank_i(d))
                    i∈retrievers

where rank_i(d) is d's rank in retriever i (1-based). k=60 is the standard
constant from the original RRF paper — it dampens the contribution of very
top-ranked results just enough to let strong hits from other retrievers surface.
"""
from typing import Any

from src.config import CFG
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector import RetrievedChunk, VectorRetriever


class HybridRetriever:
    """Vector + BM25 with RRF fusion."""

    def __init__(self) -> None:
        self._vector = VectorRetriever()
        self._bm25 = BM25Retriever()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top_k fused results.

        Args:
            query: The user's question.
            top_k: Final result count (default from config).
            where: Metadata filter applied to BOTH retrievers.
        """
        final_k = top_k or CFG.retrieval.top_k_after_fusion
        rrf_k = CFG.retrieval.rrf_k

        # 1. Fan out — each retriever gets its own top_k from config
        vec_results = self._vector.search(query, where=where)
        bm25_results = self._bm25.search(query, where=where)

        # 2. Compute RRF scores.
        # For each doc: sum over retrievers of 1/(k + rank).
        fused: dict[str, dict[str, Any]] = {}  # chunk_id -> merged info

        for rank, chunk in enumerate(vec_results, start=1):
            entry = fused.setdefault(chunk.chunk_id, {
                "chunk": chunk, "rrf": 0.0, "sources": [],
            })
            entry["rrf"] += 1.0 / (rrf_k + rank)
            entry["sources"].append(("vector", rank, chunk.score))

        for rank, chunk in enumerate(bm25_results, start=1):
            entry = fused.setdefault(chunk.chunk_id, {
                "chunk": chunk, "rrf": 0.0, "sources": [],
            })
            entry["rrf"] += 1.0 / (rrf_k + rank)
            entry["sources"].append(("bm25", rank, chunk.score))

        # 3. Sort by fused RRF score, take top_k
        ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:final_k]

        # 4. Return as RetrievedChunk with the fused score
        out: list[RetrievedChunk] = []
        for entry in ranked:
            c = entry["chunk"]
            fused_chunk = RetrievedChunk(
                chunk_id=c.chunk_id,
                content=c.content,
                metadata=c.metadata,
                score=entry["rrf"],
                source="hybrid",
            )
            # Stash provenance for debugging
            fused_chunk.metadata = {**c.metadata, "_fusion": entry["sources"]}
            out.append(fused_chunk)

        return out