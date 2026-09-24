"""All prompt templates used by the agent, kept in one file for easy tuning.

Prompts are in English (structure/instructions) but explicitly tell the model
to work in the QUERY's language. Legal QA needs answers in the user's own
tongue, and Uzbek/Russian/English models handle this well when told plainly.
"""

# ---------------- Document relevance grader ----------------

DOC_RELEVANCE_PROMPT = """You are grading a document passage against a user's question.

QUESTION:
{query}

PASSAGE:
{passage}

The passage is RELEVANT if any of these are true:
- It directly answers the question
- It contains facts that would help answer the question
- It's about the same topic (even if not the exact answer)
- It's a related regulation, procedure, or definition

Only mark "no" if the passage is clearly about a completely different subject.
When in doubt, mark "yes" — retrieval already filtered obvious noise.

Respond with a JSON object matching this exact schema:
{{"relevant": "yes" or "no", "reason": "one short sentence"}}

Do not include any text outside the JSON object."""


# ---------------- Query rewriter ----------------

QUERY_REWRITE_PROMPT = """You are improving a user's search query for a document retrieval system.

The previous retrieval returned few relevant results. Rewrite the query to be more effective:
- Add synonyms or related terms from the domain (legal, regulatory, technical)
- Expand abbreviations
- Make it more specific if it's vague, more general if it's overly narrow
- Keep it in the SAME language as the original

ORIGINAL QUERY:
{query}

PREVIOUS ATTEMPTS (rewrite each time to avoid repetition):
{previous_attempts}

Respond with a JSON object:
{{"rewritten": "the new query", "reason": "one short sentence"}}

Do not include any text outside the JSON object."""


# ---------------- Answer generator ----------------

GENERATE_ANSWER_PROMPT = """You are a document assistant answering questions strictly from provided source passages.

STRICT RULES:
1. Answer ONLY from the passages below. If they do not contain the answer, say so plainly in the user's language.
2. Answer in the SAME language as the QUESTION.
3. Cite sources inline like [1], [2] using the numbers in the passages.
4. Be concise and factual. Do not add background the passages don't mention.
5. If passages contradict, note the disagreement and cite both.

QUESTION:
{query}

PASSAGES:
{passages}

Write your answer now. If the passages do not answer the question, say so."""


# ---------------- Hallucination grader ----------------

HALLUCINATION_PROMPT = """You are checking whether an answer is grounded in provided source passages.

PASSAGES:
{passages}

ANSWER:
{answer}

Question: is every factual claim in the answer supported by the passages?
- "yes" if the answer only states things found in the passages (paraphrase is fine)
- "no" if the answer adds facts, numbers, names, or claims not in the passages

Respond with a JSON object:
{{"grounded": "yes" or "no", "reason": "one short sentence"}}

Do not include any text outside the JSON object."""


# ---------------- Answer relevance grader ----------------

ANSWER_RELEVANCE_PROMPT = """You are checking whether an answer actually addresses the user's question.

QUESTION:
{query}

ANSWER:
{answer}

Does the answer respond to what was asked?
- "yes" if it directly addresses the question (even if the answer is "the documents don't cover this")
- "no" if it drifts to a different topic or dodges the question

Respond with a JSON object:
{{"addresses": "yes" or "no", "reason": "one short sentence"}}

Do not include any text outside the JSON object."""


def format_passages(chunks: list) -> str:
    """Render retrieved chunks for insertion into a prompt.

    Numbers each one and includes filename + page + article label for citation.
    """
    lines = []
    for i, c in enumerate(chunks, start=1):
        meta = c.metadata
        label = meta.get("article_label") or "-"
        loc = f"{meta.get('filename', '?')}, p.{meta.get('page', '?')}, «{label}»"
        lines.append(f"[{i}] ({loc})\n{c.content}\n")
    return "\n".join(lines)