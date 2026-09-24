"""Build (or update) both indexes from documents in data/raw/.

Usage:
    python -m src.indexing.build_index                # incremental
    python -m src.indexing.build_index --rebuild      # from scratch
    python -m src.indexing.build_index --department hr --access confidential
"""
import argparse
import sys
from pathlib import Path

from src.config import CFG
from src.ingestion.chunker import chunk_document
from src.ingestion.loaders import iter_raw_documents, load_document
from src.indexing.bm25_store import BM25Store
from src.indexing.manifest import Manifest
from src.indexing.vector_store import VectorStore


MANIFEST_PATH = Path("data/indexes/manifest.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build or update the document indexes.")
    p.add_argument("--rebuild", action="store_true",
                   help="Ignore manifest and re-index everything from scratch.")
    p.add_argument("--department", default="general",
                   help="Tag ingested chunks with this department (default: general).")
    p.add_argument("--access", default="internal",
                   choices=["public", "internal", "confidential"],
                   help="Access level for ingested chunks (default: internal).")
    p.add_argument("--doc-version", default="v1",
                   help="Document version tag (default: v1).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = Path(CFG.ingestion.raw_dir)

    if not raw_dir.exists():
        print(f"❌ {raw_dir} does not exist. Create it and add documents.")
        return 1

    vector = VectorStore()
    bm25 = BM25Store()
    manifest = Manifest(MANIFEST_PATH)

    # ------------------------------------------------------------
    # Handle --rebuild: wipe manifest, delete old chunks per-file so
    # both stores drain cleanly (no dangling entries).
    # ------------------------------------------------------------
    if args.rebuild:
        print("🔄 --rebuild: dropping all existing chunks…")
        for source_path in list(manifest.known_files()):
            vector.delete_by_source_path(source_path)
            bm25.delete_by_source_path(source_path)
            manifest.forget(source_path)

    # ------------------------------------------------------------
    # Handle deletions: files in the manifest that no longer exist on disk
    # ------------------------------------------------------------
    on_disk = {str(p) for p in iter_raw_documents(raw_dir)}
    deleted_files = manifest.known_files() - on_disk
    for stale_path in deleted_files:
        removed_v = vector.delete_by_source_path(stale_path)
        removed_b = bm25.delete_by_source_path(stale_path)
        manifest.forget(stale_path)
        print(f"🗑️  Removed missing file: {stale_path} "
              f"(vector: -{removed_v}, bm25: -{removed_b})")

    # ------------------------------------------------------------
    # Ingest each file: skip if hash matches manifest, else re-index
    # ------------------------------------------------------------
    total_added = 0
    total_skipped = 0

    for path in sorted(iter_raw_documents(raw_dir)):
        source_path = str(path)
        print(f"\n📄 {path.name}")

        try:
            doc = load_document(path)
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            continue

        known_hash = manifest.hash_of(source_path)
        if known_hash == doc.file_hash and not args.rebuild:
            print(f"   ⏭️  Unchanged (hash matches) — skipping")
            total_skipped += 1
            continue

        # Hash changed → drop old chunks first
        if known_hash and known_hash != doc.file_hash:
            removed_v = vector.delete_by_file_hash(known_hash)
            removed_b = bm25.delete_by_file_hash(known_hash)
            print(f"   ♻️  Content changed — dropped old chunks "
                  f"(vector: -{removed_v}, bm25: -{removed_b})")

        # Chunk + index
        chunks = chunk_document(doc)
        print(f"   ✂️  Chunked into {len(chunks)} pieces")

        vector.add_chunks(chunks, args.department, args.access, args.doc_version)
        bm25.add_chunks(chunks, args.department, args.access, args.doc_version)
        manifest.record(source_path, doc.file_hash, len(chunks))
        total_added += len(chunks)
        print(f"   ✅ Indexed: department={args.department} "
              f"access={args.access} version={args.doc_version}")

    # ------------------------------------------------------------
    # Persist BM25 + manifest (Chroma auto-persists)
    # ------------------------------------------------------------
    bm25.save()
    manifest.save()

    print(f"\n{'=' * 60}")
    print("📊 Index summary")
    print("=" * 60)
    v_stats = vector.stats()
    b_stats = bm25.stats()
    print(f"   Vector store: {v_stats['count']} chunks at {v_stats['path']}")
    print(f"   BM25 store:   {b_stats['count']} chunks at {b_stats['path']}")
    print(f"   Added this run:   {total_added} chunks")
    print(f"   Skipped (cached): {total_skipped} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())