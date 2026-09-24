"""Verify access-control filtering works across departments."""
import sys
sys.path.insert(0, ".")
from src.retrieval.hybrid import HybridRetriever
from src.indexing.metadata import build_access_filter


QUERIES = [
    ("mehnat shartnomasi", ["hr", "general", "finance"]),
    ("soliq stavkasi", ["finance", "hr", "general"]),
    ("portlovchi materiallar saqlash", ["general", "hr", "finance"]),
]


def main() -> int:
    hybrid = HybridRetriever()

    for query, depts in QUERIES:
        print(f"\n{'=' * 70}")
        print(f"🔎 Query: {query!r}")
        print("=" * 70)
        for dept in depts:
            where = build_access_filter(user_department=dept, user_clearance="internal")
            results = hybrid.search(query, top_k=3, where=where)
            print(f"\n  [dept={dept}]  {len(results)} results")
            for r in results[:3]:
                m = r.metadata
                filename = m.get("filename", "?")
                label = m.get("article_label") or "-"
                page = m.get("page", "?")
                doc_dept = m.get("department", "?")
                print(f"    - {filename} p.{page} «{label}»  [dept={doc_dept}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())