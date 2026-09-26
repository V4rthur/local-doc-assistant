"""The LangGraph state machine.

Flow:
    START → retrieve → grade_docs → [decide]
                                     ├→ rewrite → retrieve  (if not enough relevant AND retries left)
                                     ├→ no_docs → finalize_no_docs → END
                                     └→ generate → hallucination → [gate]
                                                                    ├→ regenerate (if fail AND retries left)
                                                                    └→ answer_relevance → [gate]
                                                                                          ├→ rewrite (if fail AND rewrites left)
                                                                                          └→ finalize → END
"""
from langgraph.graph import StateGraph, START, END

from src.config import CFG
from src.agent import nodes
from src.agent.state import AgentState


# ---------------- Conditional edge functions ----------------

def _decide_after_grading(state: AgentState) -> str:
    """After grading docs: enough relevant → generate; not enough → rewrite or give up."""
    relevant_count = len(state.get("relevant", []))
    rewrites_used = state.get("rewrite_count", 0)

    if relevant_count >= CFG.agent.min_relevant_docs:
        return "generate"
    if rewrites_used < CFG.agent.max_query_rewrites:
        return "rewrite_query"
    return "finalize_no_docs"


def _decide_after_hallucination(state: AgentState) -> str:
    """After hallucination check: pass → check answer relevance; fail → regenerate or give up."""
    if state.get("hallucination_ok", False):
        return "grade_answer"
    gens_used = state.get("generation_count", 0)
    if gens_used < CFG.agent.max_generation_retries + 1:  # +1 because we count the first gen
        return "generate"
    return "finalize_gen_limit"


def _decide_after_answer(state: AgentState) -> str:
    """After answer-relevance check: pass → done; fail → rewrite query or give up."""
    if state.get("answer_ok", False):
        return "finalize"
    rewrites_used = state.get("rewrite_count", 0)
    if rewrites_used < CFG.agent.max_query_rewrites:
        return "rewrite_query"
    return "finalize"  # ship the answer even if imperfect — better than nothing

def _decide_after_generation(state: AgentState) -> str:
    """After generation: skip the two grader checks if we had high-confidence retrieval."""
    if state.get("high_confidence", False):
        return "finalize"
    return "grade_hallucination"


# ---------------- Build the graph ----------------


def build_graph():
    """Assemble and compile the state graph."""
    g = StateGraph(AgentState)

    # Register every node
    g.add_node("contextualize", nodes.contextualize_node)  # NEW
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("grade_docs", nodes.grade_docs_node)
    g.add_node("rewrite_query", nodes.rewrite_query_node)
    g.add_node("generate", nodes.generate_node)
    g.add_node("grade_hallucination", nodes.grade_hallucination_node)
    g.add_node("grade_answer", nodes.grade_answer_node)
    g.add_node("finalize", nodes.finalize_node)
    g.add_node("finalize_no_docs", nodes.finalize_no_docs)
    g.add_node("finalize_gen_limit", nodes.finalize_gen_limit)

    # Linear edges
    g.add_edge(START, "contextualize")  # CHANGED — was: START, "retrieve"
    g.add_edge("contextualize", "retrieve")  # NEW
    g.add_edge("retrieve", "grade_docs")
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("finalize", END)
    g.add_edge("finalize_no_docs", END)
    g.add_edge("finalize_gen_limit", END)

    # Conditional edges (unchanged)
    g.add_conditional_edges(
        "generate",
        _decide_after_generation,
        {
            "grade_hallucination": "grade_hallucination",
            "finalize": "finalize",
        },
    )

    g.add_conditional_edges(
        "grade_docs",
        _decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "finalize_no_docs": "finalize_no_docs",
        },
    )
    g.add_conditional_edges(
        "grade_hallucination",
        _decide_after_hallucination,
        {
            "grade_answer": "grade_answer",
            "generate": "generate",
            "finalize_gen_limit": "finalize_gen_limit",
        },
    )
    g.add_conditional_edges(
        "grade_answer",
        _decide_after_answer,
        {
            "finalize": "finalize",
            "rewrite_query": "rewrite_query",
        },
    )

    return g.compile()
