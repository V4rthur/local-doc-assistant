"""End-to-end agent test. Type a question, get a full trace."""
import sys
import time

sys.path.insert(0, ".")
from src.agent.graph import build_graph
from src.agent.state import initial_state


def print_trace(state: dict, elapsed: float) -> None:
    print(f"\n{'=' * 70}")
    print(f"  ANSWER  (took {elapsed:.1f}s, reason: {state.get('final_reason', '?')})")
    print("=" * 70)
    print(state.get("answer", "(no answer)"))

    print(f"\n{'-' * 70}")
    print(f"  Rewrites: {state.get('rewrite_count', 0)}   "
          f"Generations: {state.get('generation_count', 0)}   "
          f"Chunks used: {len(state.get('relevant', []))}")

    if state.get("relevant"):
        print("\n  Sources:")
        for i, c in enumerate(state["relevant"], 1):
            m = c.metadata
            print(f"    [{i}] {m.get('filename')} p.{m.get('page')} "
                  f"«{m.get('article_label') or '-'}»")

    print(f"\n{'-' * 70}")
    print("  Trace:")
    for note in state.get("grader_notes", []):
        print(f"    • {note}")


def main() -> int:
    print("Building agent…", end=" ", flush=True)
    graph = build_graph()
    print("done.")
    print("\nAgentic RAG test. Ctrl+C to quit.\n")

    while True:
        try:
            query = input("\n🔎 Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return 0
        if not query:
            continue

        t0 = time.time()
        final_state = graph.invoke(initial_state(query))
        elapsed = time.time() - t0

        print_trace(final_state, elapsed)


if __name__ == "__main__":
    sys.exit(main())