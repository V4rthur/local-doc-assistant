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
        batch_size: int = 32,
        progress: bool = True,
    ) -> None:
        """Embed and store chunks in batches with a progress bar.

        Batching serves three purposes:
            1. Progress visibility — we can report per-batch instead of one big wait.
            2. Fault tolerance — if Ollama hiccups mid-run, only the failing batch is lost.
            3. Memory ceiling — 4k+ chunk embedding lists shouldn't sit in RAM at once.
        """
        if not chunks:
            return

        from tqdm import tqdm

        total = len(chunks)
        iterator = range(0, total, batch_size)
        if progress:
            iterator = tqdm(
                iterator,
                desc=f"  Embedding ({department})",
                total=(total + batch_size - 1) // batch_size,
                unit="batch",
                leave=False,
            )

        for start in iterator:
            end = min(start + batch_size, total)
            batch = chunks[start:end]

            texts = [c.content for c in batch]
            embeddings = self._embedder.embed_documents(texts)

            ids = [c.chunk_id for c in batch]
            metadatas = [
                chunk_to_metadata(c, department, access_level, doc_version)
                for c in batch
            ]

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