"""Append-only JSONL audit log of every query.

Each line records:
  - timestamp
  - user context (department, clearance)
  - query, final answer
  - sources used (filename, page, article_label — NOT full content)
  - the agent trace (rewrites, grader verdicts)
  - flag status
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CFG


def log_query(
    query: str,
    answer: str,
    sources: list[dict],
    trace: list[str],
    department: str,
    clearance: str,
    flagged: bool = False,
) -> None:
    """Append one audit row."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "department": department,
        "clearance": clearance,
        "query": query,
        "answer": answer,
        "sources": [
            {
                "filename": s.get("filename"),
                "page": s.get("page"),
                "article_label": s.get("article_label"),
                "chunk_id": s.get("chunk_id"),
                "doc_version": s.get("doc_version"),
            }
            for s in sources
        ],
        "trace": trace,
        "flagged": flagged,
    }

    path = Path(CFG.audit.log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")