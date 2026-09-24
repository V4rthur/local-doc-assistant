"""Tracks which files were indexed at which content hash.

Enables incremental re-indexing:
  - Same file, same hash → skip (already indexed).
  - Same file, new hash → drop old chunks, re-index.
  - Missing file → drop its chunks (document deleted from data/raw/).
"""
import json
from pathlib import Path
from typing import Any


class Manifest:
    """A simple JSON file that maps source_path → {file_hash, chunk_count}."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict[str, Any]] = {}
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def record(self, source_path: str, file_hash: str, chunk_count: int) -> None:
        self._data[source_path] = {
            "file_hash": file_hash,
            "chunk_count": chunk_count,
        }

    def forget(self, source_path: str) -> None:
        self._data.pop(source_path, None)

    def hash_of(self, source_path: str) -> str | None:
        entry = self._data.get(source_path)
        return entry["file_hash"] if entry else None

    def known_files(self) -> set[str]:
        return set(self._data.keys())

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )