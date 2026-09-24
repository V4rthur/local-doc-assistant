"""LangGraph nodes.

Each node is a function: AgentState -> partial AgentState (dict of updates).
Nodes don't loop or branch themselves — that's the graph's job.
"""
from langchain_ollama import ChatOllama

from src.config import CFG
from src.agent import graders, prompts
from src.agent.state import AgentState
from src.indexing.metadata import build_access_filter
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker


# ---------------- Lazy singletons ----------------
# We build these once and reuse across turns. Loading the reranker is expensive.

_hybrid: HybridRetriever | None = None
_reranker: Reranker | None = None


def _hybrid_retriever() -> HybridRetriever:
    global _hybrid
    if _hybrid is None:
        _hybrid = HybridRetriever()
    return _hybrid


def _reranker_instance() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def _generator_llm() -> ChatOllama:
    """The generator uses a higher token cap than graders — full answers, not verdicts.

    keep_alive matches the grader so both share Ollama's warm-model window.
    """
    return ChatOllama(
        model=CFG.models.generator,
        temperature=0.1,
        num_predict=1024,
        num_ctx=4096,
        keep_alive="10m",
    )


# ---------------- Nodes ----------------

def retrieve_node(state: AgentState) -> dict:
    """Hybrid retrieval + reranker, filtered by user's access rights."""
    query = state["current_query"]
    where = build_access_filter(
        user_department=state["user_department"],
        user_clearance=state["user_clearance"],
    )

    hybrid = _hybrid_retriever()
    fused = hybrid.search(query, where=where)

    reranker = _reranker_instance()
    reranked = reranker.rerank(query, fused)

    return {
        "retrieved": reranked,
        "grader_notes": state.get("grader_notes", []) + [
            f"retrieve: fused={len(fused)}, reranked={len(reranked)}"
        ],
    }


def grade_docs_node(state: AgentState) -> dict:
    """Grade retrieved chunks for relevance.

    Optimizations:
      1. Skip grading when reranker score >= accept_threshold (auto-accept).
      2. Skip grading when reranker score <= reject_threshold (auto-reject).
      3. Grade the middle band in parallel via asyncio.gather.
    """
    import asyncio
    from src.agent import graders as g

    query = state["current_query"]
    retrieved = state["retrieved"]

    # Only look at top 3 — reranker already sorted by relevance
    to_consider = retrieved[:3]

    accept_thr = CFG.agent.reranker_accept_threshold
    reject_thr = CFG.agent.reranker_reject_threshold

    accepted: list = []
    to_grade: list = []
    notes: list[str] = []

    for chunk in to_consider:
        score = chunk.score  # This IS the reranker score
        if score >= accept_thr:
            accepted.append(chunk)
            notes.append(f"[{chunk.chunk_id}] auto-accept (rerank {score:.3f} ≥ {accept_thr})")
        elif score <= reject_thr:
            notes.append(f"[{chunk.chunk_id}] auto-reject (rerank {score:.3f} ≤ {reject_thr})")
        else:
            to_grade.append(chunk)

    # Grade only the middle band, in parallel
    if to_grade:
        async def _grade_all():
            tasks = [g.grade_relevance_async(query, c.content) for c in to_grade]
            return await asyncio.gather(*tasks)

        verdicts = asyncio.run(_grade_all())

        for chunk, verdict in zip(to_grade, verdicts):
            notes.append(f"[{chunk.chunk_id}] LLM: {verdict.relevant} — {verdict.reason}")
            if verdict.relevant == "yes":
                accepted.append(chunk)

    return {
        "relevant": accepted,
        "relevance_notes": notes,
        "grader_notes": state.get("grader_notes", []) + [
            f"grade_docs: {len(accepted)} accepted "
            f"(auto: {len(to_consider) - len(to_grade)}, LLM-graded: {len(to_grade)})"
        ],
    }


def rewrite_query_node(state: AgentState) -> dict:
    """Try a different phrasing when retrieval didn't yield enough relevant chunks."""
    previous = [state["original_query"], state["current_query"]]
    verdict = graders.rewrite_query(state["original_query"], previous)

    return {
        "current_query": verdict.rewritten,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "grader_notes": state.get("grader_notes", []) + [
            f"rewrite: '{verdict.rewritten}' ({verdict.reason})"
        ],
    }


def generate_node(state: AgentState) -> dict:
    """Write the answer from the relevant chunks."""
    passages = prompts.format_passages(state["relevant"])
    prompt = prompts.GENERATE_ANSWER_PROMPT.format(
        query=state["original_query"],   # answer the ORIGINAL question, not the rewrite
        passages=passages,
    )
    response = _generator_llm().invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer.strip(),
        "generation_count": state.get("generation_count", 0) + 1,
        "grader_notes": state.get("grader_notes", []) + [
            f"generate: attempt {state.get('generation_count', 0) + 1}, {len(answer)} chars"
        ],
    }


def grade_hallucination_node(state: AgentState) -> dict:
    """Check whether the answer is grounded in the retrieved passages."""
    passages = prompts.format_passages(state["relevant"])
    verdict = graders.grade_grounded(passages, state["answer"])
    ok = verdict.grounded == "yes"
    return {
        "hallucination_ok": ok,
        "grader_notes": state.get("grader_notes", []) + [
            f"hallucination: {verdict.grounded} — {verdict.reason}"
        ],
    }


def grade_answer_node(state: AgentState) -> dict:
    """Check whether the answer actually addresses the question."""
    verdict = graders.grade_addresses(state["original_query"], state["answer"])
    ok = verdict.addresses == "yes"
    return {
        "answer_ok": ok,
        "grader_notes": state.get("grader_notes", []) + [
            f"answer_relevance: {verdict.addresses} — {verdict.reason}"
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Set a final_reason when we exit cleanly."""
    return {"final_reason": "ok"}


def finalize_no_docs(state: AgentState) -> dict:
    """No relevant chunks found even after rewrites."""
    lang_note = "Hujjatlarda so'rovingizga tegishli ma'lumot topilmadi."
    return {
        "answer": lang_note,
        "final_reason": "no_docs",
    }


def finalize_gen_limit(state: AgentState) -> dict:
    """Generation kept failing quality checks — ship whatever we have with a note."""
    return {"final_reason": "gen_limit"}