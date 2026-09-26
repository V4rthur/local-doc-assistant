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

CRITICAL LANGUAGE RULE — THIS IS THE MOST IMPORTANT RULE:
- Detect the language of the QUESTION below.
- Your ENTIRE answer MUST be in that exact language.
- If the question is in Uzbek Latin script → answer entirely in Uzbek Latin script.
- If the question is in Russian → answer entirely in Russian.
- If the question is in English → answer entirely in English.
- NEVER include Chinese characters (中文) in your answer. NEVER.
- NEVER mix languages within the answer. If the question is Uzbek, every sentence, every bullet, every word must be Uzbek.
- If you feel tempted to write in Chinese or English when the question was Uzbek, STOP and rewrite in Uzbek.

STRICT CONTENT RULES:
1. Answer ONLY from the passages below. If they do not contain the answer, say so plainly in the user's language.
2. Cite sources inline like [1], [2] using the numbers shown next to each passage.
3. Be concise and factual. Do not add background the passages don't mention.
4. If passages contradict, note the disagreement and cite both.
5. Do NOT repeat the same heading multiple times. Structure your answer once, cleanly.

QUESTION:
{query}

PASSAGES:
{passages}

Now write your answer, IN THE SAME LANGUAGE AS THE QUESTION.
If the question is Uzbek, your first word must be Uzbek. Every following word must be Uzbek."""


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

# ---------------- Multi-turn query contextualization ----------------

CONTEXTUALIZE_PROMPT = """You are rewriting a short follow-up question into a standalone question.

The user asked something previously (in HISTORY). Now they've asked a follow-up (NEW QUESTION). The follow-up may skip repeating what they're asking about — your job is to figure out what they mean and produce a standalone version.

RULES:
1. You MUST produce a NEW question. It must differ from BOTH the follow-up AND every previous question in the history.
2. Identify what the follow-up is asking ABOUT. Usually it's a noun that replaces the subject of a previous question:
   - Previous: "What are employee rights?"
   - Follow-up: "And employer?" or "Employer haqida-chi?"
   - Rewrite: "What are employer rights?" — the NEW subject with the SAME question structure.
3. Keep the rewrite in the SAME language as the follow-up.
4. Keep it SHORT. Match the length of previous questions in history. Do NOT add invented specifics ("in the contract", "under which article", etc.) that weren't in the follow-up.
5. Do NOT invent context that isn't in the follow-up.

FORBIDDEN OUTPUTS (these all get rejected):
- Returning the follow-up unchanged
- Returning any previous question from HISTORY verbatim
- Returning a version longer than any question in HISTORY
- Adding new topics not mentioned in the follow-up

CONVERSATION HISTORY (most recent last):
{history}

FOLLOW-UP QUESTION (must rewrite by inserting the missing subject):
{query}

Respond with a JSON object:
{{"standalone_query": "the rewritten question — new subject, same shape as history question", "was_rewritten": "yes", "reason": "one short sentence"}}

Do not include any text outside the JSON object."""

def format_history(history: list[dict], max_turns: int = 4) -> str:
    """Render the last few turns for insertion into the contextualization prompt."""
    if not history:
        return "(no previous turns)"
    recent = history[-max_turns:]
    lines = []
    for msg in recent:
        role = msg.get("role", "?").upper()
        content = msg.get("content", "")[:400]  # cap length
        lines.append(f"{role}: {content}")
    return "\n".join(lines)