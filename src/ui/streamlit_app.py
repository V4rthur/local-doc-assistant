"""Streamlit chat UI for the Local Doc Assistant.

Run with:
    streamlit run src/ui/streamlit_app.py

Everything the agent produces (answer, sources, trace, timings) is surfaced
so a reviewer can see how the answer was arrived at.
"""
import sys
import time
from pathlib import Path

# Streamlit's launch context has an empty sys.path — inject project root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.config import CFG
from src.agent.graph import build_graph
from src.agent.state import initial_state
from src.audit.logger import log_query  # NEW — see below


# ---------------- Cache expensive singletons ----------------

@st.cache_resource(show_spinner="Loading agent (models + indexes)…")
def get_graph():
    """Build the compiled LangGraph once per Streamlit session."""
    return build_graph()


# ---------------- Sidebar: access + settings ----------------

@st.cache_data(ttl=300)
def _list_indexed_departments() -> list[str]:
    """Read distinct department values from the vector store.

    Cached for 5 min so we don't hit Chroma on every rerun. Reset the
    conversation or restart Streamlit to refresh after re-indexing.
    """
    try:
        from src.indexing.vector_store import VectorStore
        store = VectorStore()
        # Grab all metadata (no embeddings needed) — cheap on 4k chunks
        raw = store._collection.get(include=["metadatas"])
        depts = {m.get("department") for m in raw.get("metadatas", []) if m.get("department")}
        return sorted(depts) if depts else ["general"]
    except Exception:
        return ["general"]


def render_sidebar() -> dict:
    st.sidebar.title("⚙️ Sozlamalar / Settings")

    st.sidebar.subheader("👤 User")

    departments = _list_indexed_departments()
    dept = st.sidebar.selectbox(
        "Department",
        departments,
        index=0,
        help=f"Departments detected from your index: {', '.join(departments)}",
    )
    clearance = st.sidebar.selectbox(
        "Access clearance",
        ["public", "internal", "confidential"],
        index=1,
    )

    st.sidebar.divider()
    st.sidebar.subheader("🔧 Debug")
    show_trace = st.sidebar.checkbox("Show internal trace", value=True)
    show_relevance_notes = st.sidebar.checkbox("Show relevance grader notes", value=False)

    st.sidebar.divider()
    if st.sidebar.button("🗑️  Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(
        f"Models\n"
        f"• gen: `{CFG.models.generator}`\n"
        f"• emb: `{CFG.models.embedder}`\n"
        f"• rerank: `{CFG.models.reranker}`"
    )
    st.sidebar.caption(f"Indexed departments: {len(departments)}")

    return {
        "department": dept,
        "clearance": clearance,
        "show_trace": show_trace,
        "show_relevance_notes": show_relevance_notes,
    }


# ---------------- Message rendering ----------------

def _render_sources(sources: list) -> None:
    if not sources:
        st.info("No sources — the assistant couldn't ground an answer.")
        return
    for i, c in enumerate(sources, 1):
        m = c.metadata
        st.markdown(
            f"**[{i}]** `{m.get('filename', '?')}` · "
            f"p.{m.get('page', '?')} · «{m.get('article_label') or '-'}»"
        )
        with st.expander("Show passage", expanded=False):
            st.markdown(c.content)


def _render_trace(trace: list[str]) -> None:
    for line in trace:
        st.markdown(f"- {line}")


def _render_relevance_notes(notes: list[str]) -> None:
    if not notes:
        st.caption("(no notes captured)")
        return
    for line in notes:
        st.markdown(f"- {line}")


def render_assistant_message(msg: dict, settings: dict, idx: int) -> None:
    """Render a completed assistant message with all its expansions."""
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])
        cols = st.columns([1, 1, 1, 4])

        with cols[0]:
            st.caption(f"⏱️ {msg['elapsed']:.1f}s")
        with cols[1]:
            st.caption(f"↻ {msg['rewrites']} rewrites")
        with cols[2]:
            st.caption(f"✍️ {msg['generations']} gens")
        with cols[3]:
            # Flag button — feeds audit log
            if st.button("🚩 Flag", key=f"flag_{idx}"):
                log_query(
                    query=msg["query"],
                    answer=msg["answer"],
                    sources=[s.metadata for s in msg["sources"]],
                    trace=msg["trace"],
                    department=settings["department"],
                    clearance=settings["clearance"],
                    flagged=True,
                )
                st.success("Flagged for review — thanks.")

        st.divider()

        # Sources panel
        with st.expander(f"📚 Sources ({len(msg['sources'])})", expanded=False):
            _render_sources(msg["sources"])

        # Trace panel
        if settings["show_trace"]:
            with st.expander("🔍 Agent trace", expanded=False):
                _render_trace(msg["trace"])

        # Grader notes
        if settings["show_relevance_notes"]:
            with st.expander("📝 Relevance grader notes", expanded=False):
                _render_relevance_notes(msg["relevance_notes"])


# ---------------- Main flow ----------------

def main() -> None:
    st.set_page_config(
        page_title="Lokal Hujjat Yordamchisi",
        page_icon="📚",
        layout="wide",
    )
    st.title("📚 Lokal Hujjat Yordamchisi")
    st.caption(
        "Agentic RAG over your local documents. Fully offline. "
        "Multilingual (Uz / Ru / En)."
    )

    settings = render_sidebar()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            render_assistant_message(msg, settings, i)

    # Input
    query = st.chat_input("Savolingizni yozing / Ask your question…")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run the agent with a progress indicator
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤔 O'ylayapman… / Thinking…")

        t0 = time.time()
        graph = get_graph()
        try:
            final_state = graph.invoke(initial_state(
                query,
                user_department=settings["department"],
                user_clearance=settings["clearance"],
            ))
        except Exception as e:
            placeholder.error(f"Agent failed: {e}")
            return
        elapsed = time.time() - t0
        placeholder.empty()

    # Persist and render
    msg = {
        "role": "assistant",
        "query": query,
        "answer": final_state.get("answer", "(no answer)"),
        "sources": final_state.get("relevant", []),
        "trace": final_state.get("grader_notes", []),
        "relevance_notes": final_state.get("relevance_notes", []),
        "elapsed": elapsed,
        "rewrites": final_state.get("rewrite_count", 0),
        "generations": final_state.get("generation_count", 0),
        "final_reason": final_state.get("final_reason", "?"),
    }
    st.session_state.messages.append(msg)

    # Audit log (unflagged, but recorded)
    log_query(
        query=query,
        answer=msg["answer"],
        sources=[s.metadata for s in msg["sources"]],
        trace=msg["trace"],
        department=settings["department"],
        clearance=settings["clearance"],
        flagged=False,
    )

    render_assistant_message(msg, settings, len(st.session_state.messages) - 1)


if __name__ == "__main__":
    main()