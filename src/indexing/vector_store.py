"""Chroma vector store: persistent, embedding-based semantic index."""
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

from src.config import CFG
from src.ingestion.chunker import Chunk
from src.indexing.metadata import chunk_to_metadata


class VectorStore:
    """Wraps a Chroma collection with the project's embedder."""

    def __init__(self) -> None:
        chroma_dir = Path(CFG.indexing.chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)

        # Persistent client — data survives process restart
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        # get_or_create so this is idempotent
        self._collection = self._client.get_or_create_collection(
            name=CFG.indexing.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Ollama-backed embedder (bge-m3 for multilingual Uz/Ru/En)
        self._embedder = OllamaEmbeddings(model=CFG.models.embedder)

    def add_chunks(
        self,
        chunks: list[Chunk],
        department: str = "general",
        access_level: str = "internal",
        doc_version: str = "v1",
    ) -> None:
        """Embed and store a batch of chunks.

        Embedding is the slow step — we batch to reduce overhead.
        """
        if not chunks:
            return

        # 1. Embed content in bulk
        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed_documents(texts)

        # 2. Build parallel arrays Chroma expects
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            chunk_to_metadata(c, department, access_level, doc_version)
            for c in chunks
        ]

        # 3. Upsert — safe for re-indexing (same chunk_id overwrites)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def delete_by_file_hash(self, file_hash: str) -> int:
        """Remove every chunk that came from a file with this hash.

        Returns the number of chunks deleted.
        """
        # Chroma's delete accepts a where clause
        existing = self._collection.get(where={"file_hash": file_hash})
        n = len(existing.get("ids", []))
        if n:
            self._collection.delete(where={"file_hash": file_hash})
        return n

    def delete_by_source_path(self, source_path: str) -> int:
        """Remove every chunk whose source file has been deleted from data/raw/."""
        existing = self._collection.get(where={"source_path": source_path})
        n = len(existing.get("ids", []))
        if n:
            self._collection.delete(where={"source_path": source_path})
        return n

    def count(self) -> int:
        """Total number of chunks in the collection."""
        return self._collection.count()

    def stats(self) -> dict[str, Any]:
        """Quick summary for the CLI."""
        return {
            "collection": self._collection.name,
            "count": self._collection.count(),
            "path": str(self._client.get_settings().persist_directory),
        }