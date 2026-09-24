"""Diagnose why the DOCX files chunk into so many pieces."""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")
from src.ingestion.loaders import load_document
from src.ingestion.chunker import chunk_document

FILES = [
    "data/raw/finance/30.12.2019.docx",
    "data/raw/hr/28.10.2022.docx",
    "data/raw/general/8139250.pdf",
]


def main() -> int:
    for path_str in FILES:
        path = Path(path_str)
        if not path.exists():
            print(f"⚠️  Not found: {path}")
            continue

        print(f"\n{'=' * 70}")
        print(f"📄 {path.name}")
        print("=" * 70)

        doc = load_document(path)
        print(f"   Pages/blocks reported: {doc.total_pages} pages, {len(doc.blocks)} blocks")

        kinds = Counter(b.kind for b in doc.blocks)
        print(f"   Block kinds: text={kinds.get('text', 0)}, table={kinds.get('table', 0)}")

        # Block size distribution
        text_blocks = [b for b in doc.blocks if b.kind == "text"]
        if text_blocks:
            sizes = [len(b.content) for b in text_blocks]
            print(f"   Text block sizes: min={min(sizes)}, avg={sum(sizes)//len(sizes)}, max={max(sizes)}")
            tiny = sum(1 for s in sizes if s < 100)
            print(f"   Tiny text blocks (<100 chars): {tiny} / {len(text_blocks)}")

        # Chunk it
        chunks = chunk_document(doc)
        chunk_sizes = [len(c.content) for c in chunks]
        labeled = sum(1 for c in chunks if c.article_label)

        print(f"   Chunks: {len(chunks)}")
        if chunks:
            print(f"   Chunk sizes: min={min(chunk_sizes)}, avg={sum(chunk_sizes)//len(chunks)}, max={max(chunk_sizes)}")
            print(f"   Labeled (with Modda/Bob/etc): {labeled}")

            # How many chunks are very small (fragments) vs full-size?
            tiny_chunks = sum(1 for s in chunk_sizes if s < 100)
            small_chunks = sum(1 for s in chunk_sizes if 100 <= s < 300)
            full_chunks = sum(1 for s in chunk_sizes if s >= 300)
            print(f"   Chunk size buckets: tiny(<100)={tiny_chunks}, small(100-300)={small_chunks}, full(≥300)={full_chunks}")

            # Show first 3 chunks
            print("\n   First 3 chunks:")
            for i, c in enumerate(chunks[:3], 1):
                label = c.article_label or "-"
                preview = c.content[:100].replace("\n", " ")
                print(f"     [{i}] {len(c.content)} chars, «{label}»: {preview}…")

    return 0


if __name__ == "__main__":
    sys.exit(main())