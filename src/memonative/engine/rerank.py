"""LLM-driven dedup for retrieved memories.

The retrieval scorer ranks memories by vector similarity, FTS rank,
trust, salience, and decay. It does NOT understand that two memories
phrased differently can describe the same single fact. This module
adds a pure dedup pass — drop memories that are paraphrases of another
already-kept memory, preserve everything else.

A previous version of this module also tried to rerank by relevance,
but that turned out to drop critical context: in v3-deepseek-k10-rerank-30
the rerank kept only 1 of 5 useful memories for `505af2f5` (creamer
question) because it judged the user's existing recipe "tangential" to
"new creamer recommendations". The retrieval scorer already does
relevance ranking — adding a second relevance layer here drowned good
context. Pure dedup only.

This module is opt-in — the engine's `retrieve_memories` stays
deterministic and LLM-free for production hot paths. The eval adapter
calls this explicitly because per-question latency is irrelevant in
batch evaluation.
"""
from __future__ import annotations

import json

from memonative.engine.retrieval import ScoredMemory


_DEDUP_PROMPT_TEMPLATE = """\
A user asked: "{question}"

Below are memories the retrieval system returned, each with the date \
the underlying event happened. Your ONLY job is to identify memories \
that are PURE PARAPHRASES of another listed memory — describing the \
same single fact about the same entity on the same date.

DO NOT filter by relevance to the question. The retrieval system has \
already ranked them; downstream code uses everything you keep. Your \
job is dedup, nothing else.

KEEP (do NOT drop) memories that:
- Describe different entities, even when surface-similar ("Amazonia \
community tank" vs "Tiger I tank diorama" → both kept)
- Describe the same entity at different dates ("Weighs 180 lbs" \
2023-01-15 vs "Weighs 165 lbs" 2023-06-15 → both kept)
- Are different specific values ("$400,000 mortgage pre-approval" vs \
"$350,000 pre-approval for a different house" → both kept)
- Provide context, background, or assistant recommendations relevant \
to the topic, even if not a direct answer
- Have different specificity ("User flew Delta" vs "User flew Delta to \
Miami Jan 5" → keep both, even though the second is more specific)

DROP (only) memories that are pure paraphrases of another listed \
memory:
- "User has 20-gallon community tank" (2023-05-23) + "User has set up \
community tank" (2023-05-23) → drop one
- "User is traveling to Boston" (2023-04-10) + "User has a Boston trip \
planned" (2023-04-10) → drop one

When in doubt, KEEP (do not drop). A duplicate kept is harmless noise; \
a distinct fact dropped can flip the answer.

Memories (1-indexed, with Event date):
{memory_list}

Return JSON with one field:
  "drop": list of memory indices (1-based) to remove because each is a \
pure paraphrase of another listed memory. Empty list if no duplicates.
"""


async def llm_dedup_rerank(
    llm,
    question: str,
    scored: list[ScoredMemory],
    target_k: int,
) -> list[ScoredMemory]:
    """Dedup-only LLM pass over `scored`, preserving original rank order.

    Returns up to `target_k` items. If the LLM drops so many that fewer
    than `target_k` remain, backfills from the dropped candidates in
    original score order — so an over-aggressive rerank can never strip
    the reader of context it would have seen without rerank.

    On any failure (LLM error, JSON parse failure, garbage output)
    falls back to the original score-ordered top_k slice.
    """
    if target_k <= 0 or not scored:
        return []
    if len(scored) <= 1:
        return scored[:target_k]

    lines = []
    for i, sm in enumerate(scored, start=1):
        date_str = (
            sm.memory.created_at.strftime("%Y-%m-%d")
            if sm.memory.created_at
            else "no date"
        )
        lines.append(f"[{i}] (Event date: {date_str}) {sm.memory.content}")
    memory_list = "\n".join(lines)

    prompt = _DEDUP_PROMPT_TEMPLATE.format(
        question=question,
        memory_list=memory_list,
    )

    try:
        raw = await llm.complete(prompt)
    except Exception:
        return scored[:target_k]

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return scored[:target_k]

    drop_raw = data.get("drop") or []
    drop_indices: set[int] = set()
    for idx in drop_raw:
        if isinstance(idx, int) and 1 <= idx <= len(scored):
            drop_indices.add(idx)

    # First pass: take everything NOT in drop, in original score order
    result: list[ScoredMemory] = []
    for i, sm in enumerate(scored, start=1):
        if i in drop_indices:
            continue
        result.append(sm)
        if len(result) >= target_k:
            break

    # Backfill safety net: if the rerank somehow left us short (e.g.
    # dropped more than expected), pull from the dropped pool in original
    # score order. This guarantees the reader gets the same volume of
    # context it would have had without the rerank.
    if len(result) < target_k:
        for i, sm in enumerate(scored, start=1):
            if sm in result:
                continue
            result.append(sm)
            if len(result) >= target_k:
                break

    return result
