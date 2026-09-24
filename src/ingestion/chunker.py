"""Article-aware chunking for legal / regulatory documents.

Strategy (in priority order):
  1. Try article boundaries — split at 'Modda N', 'Bob N', 'Статья N', etc.
     Each article becomes one chunk (or several, if it's very long).
  2. Fallback — for prose without articles, use recursive character splitting.
  3. Tables always stay as their own chunk (never split a table).

Every chunk carries rich metadata: source file, page, article/section label,
block kind, and a stable chunk_id for citation.
"""
import hashlib
import re
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from src.config import CFG
from src.ingestion.models import Block, Document


class Chunk(BaseModel):
    """A retrievable unit of content ready for indexing."""
    chunk_id: str = Field(description="Stable ID: {file_hash[:8]}_{index}")
    content: str
    source_path: str
    filename: str
    file_hash: str
    page: int
    kind: str  # 'text' | 'table'
    article_label: str | None = None  # e.g., 'Modda 12', 'Bob 3', None if no boundary
    language_hint: str
    token_estimate: int = Field(description="Rough token count (chars/4)")


# ---------- Article detection ----------

# Compile patterns once. They match at the START of a line (^ with MULTILINE).
# Each pattern captures the full label (e.g. "Modda 12") so we can attach it as metadata.
_ARTICLE_PATTERNS = [re.compile(p, re.MULTILINE | re.IGNORECASE)
                     for p in CFG.chunking.article_patterns]

# A combined pattern that finds any article boundary
_COMBINED_ARTICLE_RE = re.compile(
    r"^(?:" + "|".join(CFG.chunking.article_patterns) + r")",
    re.MULTILINE | re.IGNORECASE,
)


def _find_article_boundaries(text: str) -> list[tuple[int, str]]:
    """Return a list of (char_offset, label) for every article header found.

    An 'article boundary' is a line that starts with one of the configured
    patterns (Modda N, Bob N, Статья N, etc.).
    """
    boundaries: list[tuple[int, str]] = []
    for match in _COMBINED_ARTICLE_RE.finditer(text):
        # Extract just the label (e.g., "Modda 12") — grab the current line up to
        # the first punctuation or end-of-line.
        line_end = text.find("\n", match.start())
        if line_end == -1:
            line_end = len(text)
        line = text[match.start():line_end].strip()
        # Take up to the first period or ~40 chars, whichever comes first
        label = re.split(r"[.\-—:]", line, maxsplit=1)[0].strip()[:40]
        boundaries.append((match.start(), label))
    return boundaries


def _split_by_articles(
    text: str,
    initial_annex_context: str | None = None,
) -> tuple[list[tuple[str, str | None]], str | None]:
    """Split text into (segment, article_label) tuples using article boundaries.

    Args:
        text: Text of one block to split.
        initial_annex_context: If we're already inside an annex (carried over
            from a previous block), pass its label here.

    Returns:
        (segments, final_annex_context) — the final annex_context is returned
        so the caller can carry it into the next block.
    """
    boundaries = _find_article_boundaries(text)
    if not boundaries:
        # No boundaries in this block — carry the annex context through unchanged
        return [(text, None)], initial_annex_context

    segments: list[tuple[str, str | None]] = []
    if boundaries[0][0] > 0:
        preamble = text[:boundaries[0][0]].strip()
        if preamble:
            segments.append((preamble, None))

    annex_context = initial_annex_context
    ANNEX_MARKERS = ("ilova", "приложение", "annex")

    for i, (offset, label) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        segment = text[offset:end].strip()
        if not segment:
            continue

        is_annex_header = any(m in label.lower() for m in ANNEX_MARKERS)
        if is_annex_header:
            annex_context = label
            final_label = label
        elif annex_context:
            final_label = f"{annex_context} / {label}"
        else:
            final_label = label

        segments.append((segment, final_label))

    return segments, annex_context


# ---------- Recursive fallback splitter ----------

_recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CFG.chunking.chunk_size,
    chunk_overlap=CFG.chunking.chunk_overlap,
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    length_function=len,
)


def _split_long_segment(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split a too-long segment (usually a giant article) into overlapping pieces."""
    if len(text) <= max_chars:
        return [text]
    return _recursive_splitter.split_text(text)


# ---------- Main entry point ----------

def chunk_document(doc: Document) -> list[Chunk]:
    """Convert a Document into a list of retrievable Chunks."""
    chunks: list[Chunk] = []
    max_chars = CFG.chunking.chunk_size
    overlap = CFG.chunking.chunk_overlap
    chunk_idx = 0

    MIN_CHUNK_CHARS = 30

    def _append(content: str, block: Block, article_label: str | None) -> None:
        nonlocal chunk_idx
        stripped = content.strip()
        if len(stripped) < MIN_CHUNK_CHARS:
            return
        chunks.append(_make_chunk(doc, block, stripped, chunk_idx, article_label))
        chunk_idx += 1

    # Annex context persists ACROSS blocks in the same document
    annex_context: str | None = None

    for block in doc.blocks:
        if block.kind == "table":
            # If we're inside an annex, prefix the table label too
            label = annex_context if annex_context else None
            _append(block.content, block, article_label=label)
            continue

        segments, annex_context = _split_by_articles(
            block.content, initial_annex_context=annex_context
        )

        for segment_text, article_label in segments:
            if len(segment_text) <= max_chars:
                _append(segment_text, block, article_label)
            else:
                for piece in _split_long_segment(segment_text, max_chars, overlap):
                    _append(piece, block, article_label)

    return chunks


def _make_chunk(
    doc: Document,
    block: Block,
    content: str,
    idx: int,
    article_label: str | None,
) -> Chunk:
    """Assemble a Chunk with all metadata filled in."""
    chunk_id = f"{doc.file_hash[:8]}_{idx:04d}"
    return Chunk(
        chunk_id=chunk_id,
        content=content.strip(),
        source_path=doc.source_path,
        filename=doc.filename,
        file_hash=doc.file_hash,
        page=block.page,
        kind=block.kind,
        article_label=article_label,
        language_hint=doc.language_hint,
        token_estimate=max(1, len(content) // 4),  # rough: 4 chars ≈ 1 token
    )

def chunk_documents(docs: Iterable[Document]) -> list[Chunk]:
    """Convenience: chunk many documents at once."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks