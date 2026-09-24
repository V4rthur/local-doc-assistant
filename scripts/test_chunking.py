"""Sanity-test the chunker on ingested documents."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from src.ingestion.loaders import iter_raw_documents, load_document
from src.ingestion.chunker import chunk_document


def main() -> int:
    raw_dir = Path("data/raw")
    files = list(iter_raw_documents(raw_dir))
    if not files:
        print("No files in data/raw/")
        return 1

    for path in files:
        print(f"\n{'=' * 60}")
        print(f"📄 {path.name}")
        print("=" * 60)

        doc = load_document(path)
        chunks = chunk_document(doc)

        # Summary stats
        kinds = Counter(c.kind for c in chunks)
        with_label = sum(1 for c in chunks if c.article_label is not None)
        token_total = sum(c.token_estimate for c in chunks)
        char_sizes = [len(c.content) for c in chunks]

        print(f"   Chunks:            {len(chunks)}")
        print(f"     text:            {kinds.get('text', 0)}")
        print(f"     table:           {kinds.get('table', 0)}")
        print(f"   With article label: {with_label} / {len(chunks)}")
        print(f"   Total tokens (est): {token_total:,}")
        if char_sizes:
            print(f"   Size (chars): min={min(char_sizes)}, "
                  f"avg={sum(char_sizes) // len(char_sizes)}, "
                  f"max={max(char_sizes)}")

        # Show a few chunks with article labels
        labeled = [c for c in chunks if c.article_label]
        if labeled:
            print(f"\n   Sample labeled chunks (first 3):")
            for c in labeled[:3]:
                preview = c.content[:150].replace("\n", " ")
                print(f"     [{c.chunk_id}] page {c.page} — {c.article_label}")
                print(f"        {preview}…")

        # Show a chunk WITHOUT an article label (fallback splitter output)
        unlabeled = [c for c in chunks if c.article_label is None and c.kind == "text"]
        if unlabeled:
            c = unlabeled[0]
            preview = c.content[:150].replace("\n", " ")
            print(f"\n   Sample unlabeled chunk:")
            print(f"     [{c.chunk_id}] page {c.page}")
            print(f"        {preview}…")

    return 0


if __name__ == "__main__":
    sys.exit(main())