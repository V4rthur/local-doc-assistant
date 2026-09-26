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
      4. Set high_confidence=True when the top chunk is strong and the
         median is decent — triggers post-gen grader skip in the graph.
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
        score = chunk.score
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

    # If this round found no relevant chunks BUT we had some before,
    # keep the previous ones — don't lose our good context from an earlier retry.
    previously_relevant = state.get("relevant", [])
    if not accepted and previously_relevant:
        return {
            "relevant": previously_relevant,
            "relevance_notes": notes,
            "high_confidence": False,
            "grader_notes": state.get("grader_notes", []) + [
                f"grade_docs: 0 new accepted, keeping {len(previously_relevant)} from prior round"
            ],
        }

    # High confidence: at least 2 accepted chunks, top chunk very strong (≥0.9),
    # median chunk reasonably strong (≥0.6). One weaker chunk doesn't kill the shortcut.
    high_conf = False
    if len(accepted) >= 2:
        scores = sorted([c.score for c in accepted], reverse=True)
        top_score = scores[0]
        median_score = scores[len(scores) // 2]
        high_conf = top_score >= 0.9 and median_score >= 0.6

    return {
        "relevant": accepted,
        "relevance_notes": notes,
        "high_confidence": high_conf,
        "grader_notes": state.get("grader_notes", []) + [
            f"grade_docs: {len(accepted)} accepted "
            f"(auto: {len(to_consider) - len(to_grade)}, LLM-graded: {len(to_grade)})"
            + (f" [high_confidence: {high_conf}]" if high_conf else "")
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
    """Set a final_reason when we exit cleanly.

    If we exhausted rewrites and the answer never passed relevance check,
    replace the low-quality answer with the "please rephrase" message.
    """
    from src.config import CFG

    answer_passed = state.get("answer_ok", True)
    rewrites_exhausted = state.get("rewrite_count", 0) >= CFG.agent.max_query_rewrites

    if not answer_passed and rewrites_exhausted:
        original = state.get("original_query", "").strip()
        contextualized = state.get("contextualized_query", "").strip()
        was_contextualized = contextualized and contextualized != original

        if was_contextualized:
            msg = (
                "Kechirasiz, savolingizni aniq tushuna olmadim. "
                "Iltimos, savolni to'liqroq shakllantirib qayta so'rang.\n\n"
                "(Sorry, I couldn't understand your follow-up. Please rephrase with more detail.)"
            )
            return {"answer": msg, "final_reason": "contextualize_failed"}

    return {"final_reason": "ok"}


def finalize_no_docs(state: AgentState) -> dict:
    """No relevant chunks found in this round.

    Distinguish two failure modes:
      1. Preserve a good earlier answer if one exists.
      2. Otherwise, choose a fallback message based on WHY we failed:
         - If contextualization changed the query heavily, likely a follow-up
           the system couldn't understand → ask user to rephrase.
         - Otherwise, the question genuinely isn't in the docs.
    """
    existing_answer = state.get("answer", "").strip()
    existing_relevant = state.get("relevant", [])

    if existing_answer and existing_relevant:
        return {
            "final_reason": "ok_with_earlier_answer",
        }

    # Diagnose: did we contextualize?
    original = state.get("original_query", "").strip()
    contextualized = state.get("contextualized_query", "").strip()
    was_contextualized = contextualized and contextualized != original

    if was_contextualized:
        msg = (
            "Kechirasiz, savolingizni aniq tushuna olmadim. "
            "Iltimos, savolni to'liqroq shakllantirib qayta so'rang "
            "(masalan, mavzuni aniq ko'rsating).\n\n"
            "(Sorry, I couldn't understand your follow-up clearly. "
            "Please rephrase with more context — e.g., name the topic explicitly.)"
        )
        reason = "contextualize_failed"
    else:
        msg = (
            "Hujjatlarda so'rovingizga tegishli ma'lumot topilmadi.\n\n"
            "(No relevant information found in the documents for your query.)"
        )
        reason = "no_docs"

    return {
        "answer": msg,
        "final_reason": reason,
    }


def finalize_gen_limit(state: AgentState) -> dict:
    """Generation kept failing quality checks — ship whatever we have with a note."""
    return {"final_reason": "gen_limit"}

def contextualize_node(state: AgentState) -> dict:
    """Rewrite a follow-up query into a standalone form, if needed.

    Multi-layer safety:
      1. No history → pass through.
      2. Not a follow-up (heuristic) → pass through.
      3. LLM rewrites, but we sanity-check:
         - Verbatim refusal → manual join
         - Empty rewrite → manual join
         - Content drift (<20% word overlap with original) → manual join
         - Rewrite matches a previous user question verbatim → manual join
    """
    query = state["original_query"]
    history = state.get("conversation_history", [])

    if not history:
        return {
            "contextualized_query": query,
            "current_query": query,
        }

    if not _is_likely_followup(query):
        return {
            "contextualized_query": query,
            "current_query": query,
            "grader_notes": state.get("grader_notes", []) + [
                "contextualize: skipped (query looks self-contained)"
            ],
        }

    # Ask the LLM
    verdict = graders.contextualize_query(query, history)
    rewritten = verdict.standalone_query.strip()

    # Collect all prior user messages for the "collapse" check
    prior_user_turns = [
        turn.get("content", "").strip().lower()
        for turn in history
        if turn.get("role") == "user"
    ]

    original_words = set(query.lower().split())
    rewritten_words = set(rewritten.lower().split())

    is_verbatim = rewritten.lower() == query.strip().lower()
    is_too_short = len(rewritten) < 5

    if original_words:
        overlap = len(original_words & rewritten_words) / len(original_words)
    else:
        overlap = 1.0
    is_drifted = overlap < 0.2 and len(rewritten_words) > 3

    # NEW: rewrite matches a previous user question → LLM collapsed
    is_collapsed = rewritten.lower() in prior_user_turns

    if is_verbatim or is_too_short or is_drifted or is_collapsed:
        last_user_msg = ""
        for turn in reversed(history):
            if turn.get("role") == "user":
                last_user_msg = turn.get("content", "")
                break
        rewritten = f"{query} (avvalgi savol: {last_user_msg})" if last_user_msg else query
        if is_verbatim:
            why = "verbatim"
        elif is_too_short:
            why = "empty"
        elif is_collapsed:
            why = "collapsed to previous question"
        else:
            why = f"drifted (overlap={overlap:.0%})"
        note = f"contextualize: LLM {why}, forced manual join → '{rewritten}'"
    else:
        note = f"contextualize: yes — '{rewritten}' ({verdict.reason})"

    return {
        "contextualized_query": rewritten,
        "current_query": rewritten,
        "grader_notes": state.get("grader_notes", []) + [note],
    }


# Referential/dependent words in Uz / Ru / En that signal a likely follow-up
_FOLLOWUP_MARKERS = {
    # Uzbek
    "u", "ular", "shu", "bu", "o'sha", "haqida", "ham", "uni", "ularni",
    "unga", "ularga", "haqida-chi", "-chi",
    # Russian
    "он", "она", "они", "это", "тот", "того", "их", "ему", "им", "а",
    # English
    "it", "they", "them", "that", "those", "this", "these", "what about",
    "how about", "and", "also",
}


def _is_likely_followup(query: str) -> bool:
    """Heuristic: does this query look like a follow-up?

    Signals:
      1. Very short (< 6 words) — pronouns and stubs.
      2. Contains referential words in any of Uz/Ru/En.
      3. Starts with a conjunction (Va, И, And, But).
    """
    q = query.strip().lower()
    words = q.split()

    if len(words) < 6:
        return True

    # Any word matches a marker?
    if any(w in _FOLLOWUP_MARKERS for w in words):
        return True

    # Starts with a conjunction?
    starters = ("va ", "и ", "and ", "but ", "а ", "а что ", "ham ", "yana ")
    if any(q.startswith(s) for s in starters):
        return True

    return False