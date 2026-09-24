"""Inspect block contents from a single ingested document."""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.ingestion.loaders import iter_raw_documents, load_document


def main() -> int:
    raw_dir = Path("data/raw")
    files = list(iter_raw_documents(raw_dir))
    if not files:
        print("No files in data/raw/")
        return 1

    # Just take the first file
    doc = load_document(files[0])
    print(f"📄 {doc.filename} — {len(doc.blocks)} blocks\n")

    # Show first 3 text blocks and first 5 table blocks
    text_blocks = [b for b in doc.blocks if b.kind == "text"]
    table_blocks = [b for b in doc.blocks if b.kind == "table"]

    print(f"{'=' * 60}\nFIRST 3 TEXT BLOCKS\n{'=' * 60}")
    for i, b in enumerate(text_blocks[:3], 1):
        print(f"\n--- Text block {i} (page {b.page}) ---")
        print(b.content[:400])
        print(f"... [total {len(b.content)} chars]")

    print(f"\n{'=' * 60}\nFIRST 5 'TABLE' BLOCKS\n{'=' * 60}")
    for i, b in enumerate(table_blocks[:5], 1):
        print(f"\n--- Table block {i} (page {b.page}) ---")
        print(b.content[:500])
        print(f"... [total {len(b.content)} chars]")

    return 0


if __name__ == "__main__":
    sys.exit(main())