"""Sanity-test the document loaders on whatever is in data/raw/."""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src.ingestion.loaders import iter_raw_documents, load_document


def main() -> int:
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        print(f"❌ {raw_dir} does not exist.")
        return 1

    files = list(iter_raw_documents(raw_dir))
    if not files:
        print(f"⚠️  No supported files found in {raw_dir}/")
        print("   Drop a .pdf, .docx, or .txt file in there and re-run.")
        return 0

    for path in files:
        print(f"\n{'=' * 60}")
        print(f"📄 {path.name}")
        print("=" * 60)
        try:
            doc = load_document(path)
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue

        print(f"   Pages: {doc.total_pages}")
        print(f"   Blocks: {len(doc.blocks)}  "
              f"(text: {sum(1 for b in doc.blocks if b.kind == 'text')}, "
              f"table: {sum(1 for b in doc.blocks if b.kind == 'table')})")
        print(f"   Language: {doc.language_hint}")
        print(f"   Hash: {doc.file_hash[:12]}…")
        if doc.blocks:
            preview = doc.blocks[0].content[:200].replace("\n", " ")
            print(f"   First block preview: {preview}…")

    return 0


if __name__ == "__main__":
    sys.exit(main())