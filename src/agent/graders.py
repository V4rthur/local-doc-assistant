"""LLM graders with strict Pydantic schemas.

Each grader:
  1. Calls Ollama with a prompt.
  2. Parses the JSON response.
  3. Validates it with Pydantic.
  4. Retries once if the model returns malformed JSON.

If validation fails twice, we default to a safe answer (usually "no" — better
to retry a query than to accept unverified relevance/grounding).
"""
import json
import re
from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError
import asyncio

from src.config import CFG
from src.agent import prompts


# ---------------- Pydantic schemas ----------------

class RelevanceVerdict(BaseModel):
    relevant: Literal["yes", "no"]
    reason: str = Field(max_length=300)


class GroundedVerdict(BaseModel):
    grounded: Literal["yes", "no"]
    reason: str = Field(max_length=300)


class AddressesVerdict(BaseModel):
    addresses: Literal["yes", "no"]
    reason: str = Field(max_length=300)


class RewriteVerdict(BaseModel):
    rewritten: str = Field(min_length=1, max_length=500)
    reason: str = Field(max_length=300)


# ---------------- LLM handle ----------------

def _grader_llm() -> ChatOllama:
    """Small, fast Ollama chat handle used for all grading tasks.

    keep_alive keeps the model warm in Ollama's process for 10 min after
    the last call, so grader-to-grader hops don't pay the reload penalty.
    """
    return ChatOllama(
        model=CFG.models.grader,
        temperature=0.0,
        num_predict=200,
        num_ctx=4096,
        keep_alive="10m",
    )


# ---------------- JSON extraction ----------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Grab the first {...} block from the model output. LLMs sometimes wrap
    their JSON in prose or code fences — this yanks it out."""
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _call_and_validate(prompt: str, schema: type[BaseModel], max_attempts: int = 2):
    """Call the grader LLM and validate against a Pydantic schema.

    Returns the validated model or None if all attempts fail.
    """
    llm = _grader_llm()
    for attempt in range(max_attempts):
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        data = _extract_json(raw)
        if data is not None:
            try:
                return schema.model_validate(data)
            except ValidationError:
                pass  # try again
    return None


# ---------------- Public grader functions ----------------

def grade_relevance(query: str, passage: str) -> RelevanceVerdict:
    """Is this passage relevant to the query? Defaults to 'no' on parse failure
    (safer to drop a maybe-relevant chunk than to include noise)."""
    prompt = prompts.DOC_RELEVANCE_PROMPT.format(query=query, passage=passage)
    verdict = _call_and_validate(prompt, RelevanceVerdict)
    return verdict or RelevanceVerdict(relevant="no", reason="grader parse failure")


def grade_grounded(passages: str, answer: str) -> GroundedVerdict:
    """Is the answer supported by the passages? Defaults to 'no' on parse failure
    (safer to regenerate than to pass a possibly-hallucinated answer)."""
    prompt = prompts.HALLUCINATION_PROMPT.format(passages=passages, answer=answer)
    verdict = _call_and_validate(prompt, GroundedVerdict)
    return verdict or GroundedVerdict(grounded="no", reason="grader parse failure")


def grade_addresses(query: str, answer: str) -> AddressesVerdict:
    """Does the answer address the query? Defaults to 'yes' on parse failure
    (harmless to accept — the hallucination check already ensures groundedness)."""
    prompt = prompts.ANSWER_RELEVANCE_PROMPT.format(query=query, answer=answer)
    verdict = _call_and_validate(prompt, AddressesVerdict)
    return verdict or AddressesVerdict(addresses="yes", reason="grader parse failure")


def rewrite_query(query: str, previous_attempts: list[str]) -> RewriteVerdict:
    """Ask the LLM to rewrite the query."""
    attempts_str = "\n".join(f"- {q}" for q in previous_attempts) or "(none)"
    prompt = prompts.QUERY_REWRITE_PROMPT.format(
        query=query, previous_attempts=attempts_str
    )
    verdict = _call_and_validate(prompt, RewriteVerdict)
    return verdict or RewriteVerdict(rewritten=query, reason="rewrite parse failure")

# ---------------- Async version for parallel calls ----------------


async def _call_and_validate_async(prompt: str, schema: type[BaseModel], max_attempts: int = 2):
    """Async wrapper around _call_and_validate — runs the sync call in a thread.

    langchain-ollama doesn't ship a native async client that plays nicely with
    the ChatOllama we use, so we offload to a thread. Threads are fine here
    because we're I/O-bound on the HTTP call to Ollama, not CPU-bound.
    """
    return await asyncio.to_thread(_call_and_validate, prompt, schema, max_attempts)


async def grade_relevance_async(query: str, passage: str) -> RelevanceVerdict:
    prompt = prompts.DOC_RELEVANCE_PROMPT.format(query=query, passage=passage)
    verdict = await _call_and_validate_async(prompt, RelevanceVerdict)
    return verdict or RelevanceVerdict(relevant="no", reason="grader parse failure")