"""BM25 lexical index — strong on exact term/number matches.

BM25 has no built-in persistence, so we pickle the whole index alongside a
parallel list of chunk metadata (for citation lookup after retrieval).
"""
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.config import CFG
from src.ingestion.chunker import Chunk
from src.indexing.metadata import chunk_to_metadata


# Tokenizer optimized for mixed Uz/Ru/En + article numbers.
# - Keeps numbers intact (so "Modda 45" tokenizes to ['modda', '45'])
# - Splits on any non-word char, treats apostrophes ' and ʻ as separators
# - Lowercases everything

_TOKEN_RE = re.compile(r"[\wʻ']+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Split into lowercased tokens. Simple, fast, language-agnostic."""
    return [t.lower() for t in _TOKEN_RE.findall(text) if t]


class BM25Store:
    """A picklable BM25 index over chunk content."""

    def __init__(self) -> None:
        self._path = Path(CFG.indexing.bm25_path)
        self._chunks_meta: list[dict[str, Any]] = []  # parallel to corpus
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._loaded = False

        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._path, "rb") as f:
            state = pickle.load(f)
        self._chunks_meta = state["chunks_meta"]
        self._corpus_tokens = state["corpus_tokens"]
        # Rebuild BM25 from the tokenized corpus (BM25 objects don't pickle cleanly across versions)
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        self._loaded = True

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "chunks_meta": self._chunks_meta,
            "corpus_tokens": self._corpus_tokens,
        }
        with open(self._path, "wb") as f:
            pickle.dump(state, f)

    def add_chunks(
        self,
        chunks: list[Chunk],
        department: str = "general",
        access_level: str = "internal",
        doc_version: str = "v1",
    ) -> None:
        """Append chunks and rebuild the BM25 index.

        BM25Okapi has no incremental add — we rebuild from the full corpus.
        For 10k–100k chunks this takes < 1 second, so it's a non-issue.
        """
        for c in chunks:
            self._corpus_tokens.append(_tokenize(c.content))
            self._chunks_meta.append({
                "chunk_id": c.chunk_id,
                "content": c.content,
                **chunk_to_metadata(c, department, access_level, doc_version),
            })
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def delete_by_file_hash(self, file_hash: str) -> int:
        """Drop chunks whose source file matches this hash. Returns count deleted."""
        keep_meta = []
        keep_tokens = []
        deleted = 0
        for meta, toks in zip(self._chunks_meta, self._corpus_tokens):
            if meta.get("file_hash") == file_hash:
                deleted += 1
            else:
                keep_meta.append(meta)
                keep_tokens.append(toks)
        self._chunks_meta = keep_meta
        self._corpus_tokens = keep_tokens
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        return deleted

    def delete_by_source_path(self, source_path: str) -> int:
        keep_meta = []
        keep_tokens = []
        deleted = 0
        for meta, toks in zip(self._chunks_meta, self._corpus_tokens):
            if meta.get("source_path") == source_path:
                deleted += 1
            else:
                keep_meta.append(meta)
                keep_tokens.append(toks)
        self._chunks_meta = keep_meta
        self._corpus_tokens = keep_tokens
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        return deleted

    def count(self) -> int:
        return len(self._chunks_meta)

    def stats(self) -> dict[str, Any]:
        return {
            "path": str(self._path),
            "count": self.count(),
            "loaded": self._loaded,
        }