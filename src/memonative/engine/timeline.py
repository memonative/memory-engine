"""Open-vocabulary timeline retrieval for inventory/sequence questions.

The main retrieval pass embeds the full question, so a question like
"which airlines have I flown" pulls memories whose surface form matches
the question (often future-tense booking plans), and misses past flight
mentions phrased differently. This module runs a second, focused pass
that:

1. Asks the LLM whether the question requires enumerating, counting,
   ordering, or comparing instances of some entity across the user's
   history, AND names the entity + likely surface variants.
2. Re-queries vector + FTS using the entity terms (not the question
   wording), broadening recall to past instances.
3. Formats the results chronologically with Event dates so the reader
   can count, order, or compare without re-deriving the sequence.

The classifier is a single LLM call that combines detection and entity
extraction. Earlier versions used a regex pre-filter, but LongMemEval
phrasings are too varied for a regex to cover ("the order of X from
earliest to latest", "trace my history with Y", etc.) and an extra
gpt-4o-mini call is ~$0.0001 — cheaper than the recall we lose by
missing a timeline question.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from memonative.engine.retrieval import retrieve_memories, ScoredMemory


@dataclass
class TimelineClassification:
    is_timeline: bool
    entity: str = ""
    synonyms: list[str] = field(default_factory=list)


_CLASSIFY_PROMPT_TEMPLATE = """\
You will receive a question a user is asking about their own \
conversation history. Decide whether answering it requires \
enumerating, counting, ordering, or comparing multiple instances \
of some entity across the user's past — and if so, name the entity \
and likely surface variants.

Treat these as TIMELINE questions:
- counting ("how many X", "how often did I")
- ordering ("which came first", "in what order", "from earliest to latest")
- enumerating ("which X have I tried/done/visited", "list every X")
- comparing pairs by time ("did X happen before Y")

Treat these as NOT timeline:
- single-fact lookups ("what is my favorite color")
- single-event recall ("what did I have for dinner yesterday")
- preferences or recommendations ("what dish did the assistant suggest")
- general questions that don't depend on multiple instances

Return JSON with three fields:
  "is_timeline": boolean.
  "entity": short canonical noun phrase for the target entity \
            (e.g. "airline", "aquarium tank", "museum visit"). \
            Empty string when is_timeline is false.
  "synonyms": up to 6 alternate surface forms — category words, \
              related verbs, likely instance names. Empty list when \
              is_timeline is false.

Examples:
Q: "Which airlines have I flown with in the past year?"
A: {{"is_timeline": true, "entity": "airline", "synonyms": ["flight", \
"flew", "booked", "carrier", "ticket"]}}

Q: "How many tanks do I have in total?"
A: {{"is_timeline": true, "entity": "aquarium tank", "synonyms": \
["tank", "fish tank", "aquarium", "betta", "freshwater"]}}

Q: "What is the order of airlines I flew with from earliest to latest?"
A: {{"is_timeline": true, "entity": "airline", "synonyms": ["flight", \
"flew", "booked", "carrier", "departed"]}}

Q: "What is my favorite color?"
A: {{"is_timeline": false, "entity": "", "synonyms": []}}

Q: "What was the name of that Jamaican dish you recommended I try?"
A: {{"is_timeline": false, "entity": "", "synonyms": []}}

Q: "How many days had passed between Holi and the Sunday mass?"
A: {{"is_timeline": true, "entity": "religious event", "synonyms": \
["Holi", "mass", "church", "festival", "celebration"]}}

Question: {question}
"""


async def classify_timeline_question(
    llm, question: str,
) -> TimelineClassification:
    """Single LLM call: is this a timeline question, and if so what entity?

    Returns is_timeline=False on any parse failure — a false negative
    costs one missed timeline pass, a false positive would cost extra
    retrieval cycles. Erring toward False is the safer default.
    """
    try:
        raw = await llm.complete(_CLASSIFY_PROMPT_TEMPLATE.format(question=question))
    except Exception:
        return TimelineClassification(is_timeline=False)

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return TimelineClassification(is_timeline=False)

    is_timeline = bool(data.get("is_timeline"))
    if not is_timeline:
        return TimelineClassification(is_timeline=False)

    entity = (data.get("entity") or "").strip()
    if not entity:
        return TimelineClassification(is_timeline=False)

    synonyms_raw = data.get("synonyms") or []
    synonyms = [
        s.strip() for s in synonyms_raw
        if isinstance(s, str) and s.strip()
    ]
    return TimelineClassification(
        is_timeline=True, entity=entity, synonyms=synonyms,
    )


async def retrieve_timeline(
    db: AsyncSession,
    llm,
    *,
    tenant_id: UUID,
    user_id: UUID,
    entity: str,
    synonyms: list[str],
    top_k: int = 20,
) -> list[ScoredMemory]:
    """Focused retrieval keyed on entity + synonyms.

    Restricted to episodic + semantic (procedural is interaction style,
    never part of an inventory).
    """
    expanded = " ".join([entity] + synonyms).strip()
    if not expanded:
        return []
    embedding = await llm.embed(expanded)
    intent = {"primary": "factual", "mode": "lookup"}
    return await retrieve_memories(
        db,
        tenant_id=tenant_id,
        user_id=user_id,
        message=expanded,
        intent=intent,
        embedding=embedding,
        top_k=top_k,
        reinforce=False,
        type_override=["episodic", "semantic"],
        include_associations=False,
    )


_MIN_DATE = datetime(1900, 1, 1, tzinfo=timezone.utc)


def format_timeline_block(
    memories: list[ScoredMemory],
    entity: str,
) -> str:
    """Chronological list of distinct memory contents with Event dates.

    Sorts oldest → newest. Deduplicates by content (the FTS pass often
    surfaces near-duplicate phrasings). Returns "" when there's nothing
    to inject — caller can then skip the block entirely.
    """
    if not memories:
        return ""

    sorted_mems = sorted(
        memories,
        key=lambda sm: sm.memory.created_at or _MIN_DATE,
    )
    lines = [f"Timeline of memories related to '{entity}' (oldest first):"]
    seen: set[str] = set()
    for sm in sorted_mems:
        content = sm.memory.content
        if content in seen:
            continue
        seen.add(content)
        date = (
            sm.memory.created_at.strftime("%Y-%m-%d")
            if sm.memory.created_at
            else "?"
        )
        lines.append(f"  - {date}: {content}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)
