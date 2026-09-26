"""LangGraph state schema.

State is the single object that flows through every node. Each node reads
whatever fields it needs and returns whichever fields it wants to update —
LangGraph merges the update into the existing state.

Every retry counter has a hard cap in config.yaml so we can't loop forever.
"""
from typing import Annotated, TypedDict

from src.retrieval.vector import RetrievedChunk


class AgentState(TypedDict, total=False):
    """State passed through the LangGraph.

    `total=False` means every key is optional — this makes partial updates
    from individual nodes clean (a node can return only what it changed).
    """
    # --- Query ---
    original_query: str  # What the user asked
    current_query: str  # Possibly rewritten form
    rewrite_count: int  # How many times we've rewritten (cap: config.agent.max_query_rewrites)

    # --- Retrieval ---
    retrieved: list[RetrievedChunk]  # Raw hybrid+rerank output
    relevant: list[RetrievedChunk]  # After grading
    relevance_notes: list[str]  # Per-chunk grader reasoning (debug)

    # --- Generation ---
    answer: str  # Final or draft answer
    generation_count: int  # How many times we've generated (cap: config.agent.max_generation_retries)

    # --- Quality checks ---
    hallucination_ok: bool  # Answer grounded in retrieved chunks?
    answer_ok: bool  # Answer actually addresses the question?
    grader_notes: list[str]  # For audit + UI transparency

    # --- Access control (passed in from caller) ---
    user_department: str
    user_clearance: str  # 'public' | 'internal' | 'confidential'

    # --- Terminal state ---
    final_reason: str  # Why we stopped: 'ok' | 'no_docs' | 'rewrite_limit' | 'gen_limit'

    conversation_history: list[
        dict]  # [{"role": "user"|"assistant", "content": str}, ...]
    contextualized_query: str  # Standalone form after contextualization

    # --- Optimization flags ---
    high_confidence: bool  # Set by grade_docs; skips hallucination + answer graders


def initial_state(
    query: str,
    user_department: str = "general",
    user_clearance: str = "internal",
    conversation_history: list[dict] | None = None,
) -> AgentState:
    """Build a fresh state at the start of a query."""
    return AgentState(
        original_query=query,
        current_query=query,
        rewrite_count=0,
        retrieved=[],
        relevant=[],
        relevance_notes=[],
        answer="",
        generation_count=0,
        hallucination_ok=False,
        answer_ok=False,
        grader_notes=[],
        user_department=user_department,
        user_clearance=user_clearance,
        final_reason="",
        conversation_history=conversation_history or [],
        contextualized_query=query,
        high_confidence=False
    )
