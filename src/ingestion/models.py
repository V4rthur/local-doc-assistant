"""Data models for ingested document content."""
from typing import Literal
from pydantic import BaseModel, Field


class Block(BaseModel):
    """A single logical block extracted from a document.

    A block is either a paragraph of prose or a table rendered as markdown.
    Each block carries enough metadata to trace it back to its source page.
    """
    kind: Literal["text", "table"]
    content: str
    page: int = Field(ge=1, description="1-based page number")


class Document(BaseModel):
    """A fully extracted source document."""
    source_path: str
    filename: str
    file_hash: str = Field(description="SHA-256 of the raw file bytes")
    blocks: list[Block]
    total_pages: int
    language_hint: str = "mixed"  # 'uz-latn', 'uz-cyrl', 'ru', 'en', or 'mixed'