"""Interactive retrieval test. Now includes reranker as the final stage."""
import sys
import time

sys.path.insert(0, ".")
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector import VectorRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.reranker import Reranker


def print_results(title: str, results: list, show_fusion: bool = False) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}  ({len(results)} results)")
    print("=" * 70)
    for i, r in enumerate(results, 1):
        label = r.metadata.get("article_label") or "-"
        page = r.metadata.get("page", "?")
        preview = r.content[:180].replace("\n", " ")
        extra = ""
        if show_fusion and "_fusion" in r.metadata:
            extra = f"  fusion={r.metadata['_fusion']}"
        if "_hybrid_score" in r.metadata:
            extra = f"  hybrid_was={r.metadata['_hybrid_score']:.4f}"
        print(f"\n[{i}] score={r.score:.4f}  page={page}  «{label}»{extra}")
        print(f"    {preview}…")


def main() -> int:
    print("Retrieval sanity test.  Ctrl+C to quit.\n")
    print("Loading retrievers…", end=" ", flush=True)
    vec = VectorRetriever()
    bm25 = BM25Retriever()
    hybrid = HybridRetriever()
    reranker = Reranker()
    print("done.\n")

    while True:
        try:
            query = input("\n🔎 Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return 0
        if not query:
            continue

        t0 = time.time()
        vec_results = vec.search(query, top_k=5)
        t_vec = time.time() - t0

        t0 = time.time()
        bm25_results = bm25.search(query, top_k=5)
        t_bm25 = time.time() - t0

        t0 = time.time()
        hybrid_results = hybrid.search(query, top_k=10)  # keep 10 for reranker
        t_hybrid = time.time() - t0

        t0 = time.time()
        reranked = reranker.rerank(query, hybrid_results, top_k=5)
        t_rerank = time.time() - t0

        print_results(f"VECTOR ({t_vec*1000:.0f}ms)", vec_results)
        print_results(f"BM25 ({t_bm25*1000:.0f}ms)", bm25_results)
        print_results(f"HYBRID top-10 (RRF, {t_hybrid*1000:.0f}ms)", hybrid_results, show_fusion=True)
        print_results(f"RERANKED top-5 ({t_rerank*1000:.0f}ms)", reranked)


if __name__ == "__main__":
    sys.exit(main())