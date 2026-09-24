"""Vector-only retrieval from Chroma."""
from typing import Any

from langchain_ollama import OllamaEmbeddings

from src.config import CFG
from src.indexing.vector_store import VectorStore


class RetrievedChunk:
    """A retrieved chunk with its content, metadata, and per-retriever score."""
    __slots__ = ("chunk_id", "content", "metadata", "score", "source")

    def __init__(self, chunk_id: str, content: str, metadata: dict[str, Any],
                 score: float, source: str) -> None:
        self.chunk_id = chunk_id
        self.content = content
        self.metadata = metadata
        self.score = score      # Higher = more relevant. Cosine similarity for vector.
        self.source = source    # 'vector' | 'bm25' — useful for debugging

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"<RetrievedChunk {self.chunk_id} [{self.source}] {self.score:.3f} — {preview}…>"


class VectorRetriever:
    """Semantic search over Chroma using bge-m3 embeddings."""

    def __init__(self) -> None:
        # Reuse the same VectorStore singleton pattern
        self._store = VectorStore()
        self._collection = self._store._collection  # direct access for query
        self._embedder = OllamaEmbeddings(model=CFG.models.embedder)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top_k most semantically similar chunks.

        Args:
            query: The user's question (or a rewritten form of it).
            top_k: How many results (default from config).
            where: Optional Chroma `where` filter — used for access control.
        """
        k = top_k or CFG.retrieval.top_k_vector

        # 1. Embed the query
        query_emb = self._embedder.embed_query(query)

        # 2. Query Chroma
        raw = self._collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            where=where,
        )

        # 3. Chroma returns lists-of-lists (one per query); we sent one query
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0]   # cosine distance: lower = better

        # Convert distance → similarity score (higher = better) for uniform sorting
        results = [
            RetrievedChunk(
                chunk_id=cid,
                content=doc,
                metadata=meta,
                score=1.0 - dist,     # cosine similarity from cosine distance
                source="vector",
            )
            for cid, doc, meta, dist in zip(ids, docs, metas, dists)
        ]
        return results