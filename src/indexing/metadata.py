"""Metadata schema attached to every indexed chunk.

Chroma stores this as a flat dict alongside each vector. We use it for:
  - Citation (filename, page, article_label)
  - Access control filtering at query time (department, access_level)
  - Version tracking so old chunks are dropped on re-index (doc_version, file_hash)
"""
from typing import Any

from src.ingestion.chunker import Chunk


# Chroma metadata values must be primitive (str, int, float, bool).
# No nested dicts, no None. We convert None → "" and enforce types below.

def chunk_to_metadata(
    chunk: Chunk,
    department: str = "general",
    access_level: str = "internal",
    doc_version: str = "v1",
) -> dict[str, Any]:
    """Convert a Chunk into a Chroma-compatible metadata dict.

    Args:
        chunk: The chunk being indexed.
        department: Which department owns this document ('legal', 'hr', 'general', ...).
                    Used for filtering: users only see their own department's docs.
        access_level: 'public' | 'internal' | 'confidential'.
        doc_version: Semantic version of the document. Bump this when the source
                     document is meaningfully revised, so answers can cite which
                     version they came from.
    """
    return {
        # Identity
        "chunk_id": chunk.chunk_id,
        "file_hash": chunk.file_hash,
        # Citation
        "filename": chunk.filename,
        "source_path": chunk.source_path,
        "page": chunk.page,
        "article_label": chunk.article_label or "",  # Chroma rejects None
        "kind": chunk.kind,
        "language_hint": chunk.language_hint,
        # Access control
        "department": department,
        "access_level": access_level,
        # Versioning
        "doc_version": doc_version,
    }


# Convenience: build the "where" filter for retrieval based on user context
def build_access_filter(
    user_department: str,
    user_clearance: str,
) -> dict[str, Any]:
    """Return a Chroma `where` clause enforcing access control.

    A user sees a chunk only if:
      - the chunk's department == user's department (or is 'general'), AND
      - the chunk's access_level is at or below the user's clearance.
    """
    # Access levels form a hierarchy (public < internal < confidential)
    LEVELS = {"public": 0, "internal": 1, "confidential": 2}
    user_max = LEVELS.get(user_clearance, 0)
    allowed_levels = [lvl for lvl, rank in LEVELS.items() if rank <= user_max]

    return {
        "$and": [
            {"department": {"$in": [user_department, "general"]}},
            {"access_level": {"$in": allowed_levels}},
        ]
    }