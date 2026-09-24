"""Show a wider sample of labeled chunks to spot regex misfires."""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")
from src.ingestion.loaders import iter_raw_documents, load_document
from src.ingestion.chunker import chunk_document


def main() -> int:
    raw_dir = Path("data/raw")
    files = list(iter_raw_documents(raw_dir))
    if not files:
        print("No files in data/raw/")
        return 1

    doc = load_document(files[0])
    chunks = chunk_document(doc)

    labeled = [c for c in chunks if c.article_label]

    # Distribution of labels
    label_counts = Counter(c.article_label for c in labeled)
    print(f"📊 Label distribution (top 15):")
    for label, count in label_counts.most_common(15):
        print(f"   {count:>3}x  {label}")

    # 10 labeled samples spread across the document
    print(f"\n📄 10 labeled samples (every ~{len(labeled) // 10 or 1} chunks):")
    step = max(1, len(labeled) // 10)
    for c in labeled[::step][:10]:
        preview = c.content[:120].replace("\n", " ")
        print(f"\n   [{c.chunk_id}] page {c.page} — «{c.article_label}»")
        print(f"      {preview}…")

    return 0


if __name__ == "__main__":
    sys.exit(main())