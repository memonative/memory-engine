"""
Reconsolidation Engine for Memonative
=====================================
 
Implements memory self-editing inspired by human reconsolidation:
every time a memory is recalled in the presence of new information,
it enters a labile state where it can be reinforced, corrected, or enriched.
 
This replaces the need for explicit "edit_memory" tool calls.
The agent never consciously edits — editing is a side-effect of experience.
 
Neuroscience basis:
- Nader et al. (2000): Recalled memories require re-stabilisation
- Lee et al. (2017): Reconsolidation updates memories with current context
- Hupbach et al. (2007): New learning during recall modifies the original trace
"""

import json
import logging
from uuid import UUID
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from memonative.db.models import Memory, MemoryEdge, MemoryRevision
from memonative.db.enums import RevisionType
from memonative.engine.scoring import compute_vector_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class ReconsolidationOutcome(str, Enum):
    reinforced = "reinforced"     # new experience confirms old memory
    enriched = "enriched"         # new experience adds detail to old memory
    no_change = "no_change"       # no meaningful relationship


@dataclass
class ReconsolidationEvent:
    """Record of what happened to a memory during reconsolidation."""
    memory_id: UUID
    outcome: ReconsolidationOutcome
    original_content: str
    updated_content: str | None = None
    trust_delta: float = 0.0
    strength_delta: float = 0.0


# ---------------------------------------------------------------------------
# The reconsolidation prompt (updated — no "corrected" option)
# ---------------------------------------------------------------------------

RECONSOLIDATION_PROMPT = """\
You are analysing whether new information strengthens or enriches
existing memories. You are NOT responsible for corrections — those
are handled separately.

For each retrieved memory, determine if the new message reinforces
or enriches it.

Retrieved memories (currently in the user's memory store):
{retrieved_memories}

New message from user: {message}

Newly extracted facts from this message:
{new_facts}

For EACH retrieved memory, classify the relationship.
Return ONLY valid JSON:
{{
  "reconsolidations": [
    {{
      "memory_id": "<uuid of the retrieved memory>",
      "outcome": "reinforced|enriched|no_change",
      "reasoning": "brief explanation of why",
      "updated_content": "merged version (only for enriched, null otherwise)",
      "confidence": 0.0 to 1.0
    }}
  ]
}}

Rules:
- "reinforced": the new message confirms or repeats what the memory says.
  Example: memory says "user prefers Python", user says "I love coding in Python"
- "enriched": the new message adds meaningful detail to the memory without
  contradicting it.
  Example: memory says "user is learning guitar", user says "I've been practising
  fingerpicking for 3 months now"
  For enriched, updated_content should merge old + new into a single concise fact.
- "no_change": no meaningful relationship, OR the new message contradicts the
  memory (contradictions are handled by the write path, not here).
- Be conservative. Most memories will be "no_change" or "reinforced".
- Enrichments require moderate confidence (>= 0.6).
- If the new message CONTRADICTS a memory, classify as "no_change" — do NOT
  attempt to correct it here.
- For enrichments, updated_content MERGES old and new into one statement.\
"""


# ---------------------------------------------------------------------------
# Core reconsolidation logic
# ---------------------------------------------------------------------------

async def plan_reconsolidation(
    llm_client,
    message: str,
    retrieved_memories: list,
    new_facts: list[dict],
) -> list[dict]:
    """LLM-only phase: ask the model to classify each retrieved memory
    as reinforced/enriched/no_change.

    No DB I/O. Safe to run concurrently with other read-only LLM tasks
    (e.g. extract_consensus) via asyncio.gather. Returns the raw event
    dicts; mutations happen in apply_reconsolidation_plan.
    """
    if not retrieved_memories:
        return []

    formatted_retrieved = "\n".join([
        f"- [{sm.memory.id}] (type: {sm.memory.memory_type.value}, "
        f"trust: {sm.memory.trust_score:.2f}) {sm.memory.content}"
        for sm in retrieved_memories
    ])

    formatted_new = "\n".join([
        f"- {fact.get('content', '')}" for fact in new_facts
    ]) if new_facts else "None extracted."

    prompt = RECONSOLIDATION_PROMPT.format(
        retrieved_memories=formatted_retrieved,
        message=message,
        new_facts=formatted_new,
    )

    result_text = await llm_client.complete(prompt)

    try:
        parsed = json.loads(result_text)
        return parsed.get("reconsolidations", [])
    except Exception as e:
        snippet = (result_text or "")[:300]
        logger.warning(
            "reconsolidation parse failed: %s | payload=%r", e, snippet
        )
        return []


async def apply_reconsolidation_plan(
    db: AsyncSession,
    llm_client,
    raw_events: list[dict],
    retrieved_memories: list,
    min_enrichment_confidence: float = 0.6,
) -> list[ReconsolidationEvent]:
    """Apply the plan from plan_reconsolidation to the DB session.

    Mutates ORM Memory instances (and may call llm.embed for enrichments).
    Must run sequentially with respect to other code that mutates the
    same AsyncSession.
    """
    memory_lookup = {
        str(sm.memory.id): sm.memory for sm in retrieved_memories
    }

    events: list[ReconsolidationEvent] = []
    for raw in raw_events:
        try:
            memory_id = UUID(raw["memory_id"])
        except (KeyError, ValueError):
            continue

        outcome = raw.get("outcome", "no_change")
        confidence = raw.get("confidence", 0.0)
        updated_content = raw.get("updated_content")
        memory = memory_lookup.get(str(memory_id))

        if not memory:
            continue

        # SAFETY: If the LLM somehow returns "corrected", treat as no_change
        if outcome == "corrected":
            outcome = "no_change"

        event = ReconsolidationEvent(
            memory_id=memory_id,
            outcome=ReconsolidationOutcome(outcome) if outcome in ("reinforced", "enriched", "no_change") else ReconsolidationOutcome.no_change,
            original_content=memory.content,
        )

        if outcome == "reinforced":
            event = await _apply_reinforcement(db, memory, event)

        elif outcome == "enriched" and confidence >= min_enrichment_confidence:
            if updated_content:
                event = await _apply_enrichment(
                    db, llm_client, memory, updated_content, event
                )

        else:
            event.outcome = ReconsolidationOutcome.no_change

        events.append(event)

    return events


async def reconsolidate(
    db: AsyncSession,
    llm_client,
    message: str,
    retrieved_memories: list,
    new_facts: list[dict],
    min_enrichment_confidence: float = 0.6,
) -> list[ReconsolidationEvent]:
    """Backward-compatible wrapper: plan then apply, sequentially."""
    raw_events = await plan_reconsolidation(
        llm_client, message, retrieved_memories, new_facts
    )
    return await apply_reconsolidation_plan(
        db, llm_client, raw_events, retrieved_memories,
        min_enrichment_confidence=min_enrichment_confidence,
    )


# ---------------------------------------------------------------------------
# Outcome handlers (reinforcement and enrichment only)
# ---------------------------------------------------------------------------

async def _apply_reinforcement(
    db: AsyncSession,
    memory: Memory,
    event: ReconsolidationEvent,
) -> ReconsolidationEvent:
    """
    New experience confirms old memory.
    Like recalling a friend's name correctly — the trace strengthens.
    """
    trust_bump = 0.05
    memory.trust_score = min(1.0, memory.trust_score + trust_bump)
    memory.reinforcement_count += 1
    memory.strength = 1.0
    memory.decay_state = "active"

    event.trust_delta = trust_bump
    event.strength_delta = 1.0 - memory.strength
    return event


async def _apply_enrichment(
    db: AsyncSession,
    llm_client,
    memory: Memory,
    merged_content: str,
    event: ReconsolidationEvent,
) -> ReconsolidationEvent:
    """
    New experience adds detail to old memory.
    Like learning your friend who plays guitar now does fingerpicking —
    the memory becomes richer and more specific.

    For semantic memories with attribute_slots, enrichment creates a
    revision entry so the content expansion is visible in the audit trail.
    """
    old_content = memory.content
    memory.content = merged_content
    new_embedding = await llm_client.embed(merged_content)
    memory.embedding = new_embedding

    trust_bump = 0.05
    salience_bump = 0.05
    memory.trust_score = min(1.0, memory.trust_score + trust_bump)
    memory.salience = min(1.0, memory.salience + salience_bump)
    memory.strength = 1.0
    memory.decay_state = "active"
    memory.reinforcement_count += 1

    if memory.attribute_slot:
        now = datetime.now(timezone.utc)
        prev_result = await db.execute(
            select(MemoryRevision).where(
                MemoryRevision.memory_id == memory.id,
                MemoryRevision.attribute_slot == memory.attribute_slot,
                MemoryRevision.valid_until.is_(None),
            ).order_by(MemoryRevision.sequence_number.desc()).limit(1)
        )
        prev_rev = prev_result.scalar_one_or_none()
        if prev_rev:
            prev_rev.valid_until = now

        seq_result = await db.execute(
            select(func.coalesce(func.max(MemoryRevision.sequence_number), 0))
            .where(
                MemoryRevision.tenant_id == memory.tenant_id,
                MemoryRevision.user_id == memory.user_id,
                MemoryRevision.attribute_slot == memory.attribute_slot,
            )
        )
        next_seq = seq_result.scalar() + 1

        db.add(MemoryRevision(
            tenant_id=memory.tenant_id,
            memory_id=memory.id,
            user_id=memory.user_id,
            content=merged_content,
            embedding=new_embedding,
            revision_type=RevisionType.enrichment,
            reason=f"Detail added during reconsolidation (was: {old_content})",
            valid_from=now,
            valid_until=None,
            attribute_slot=memory.attribute_slot,
            sequence_number=next_seq,
        ))

    event.updated_content = merged_content
    event.trust_delta = trust_bump
    return event


# ---------------------------------------------------------------------------
# Interference (unchanged — still useful for related memories)
# ---------------------------------------------------------------------------

async def apply_interference(
    db: AsyncSession,
    corrected_memory_id: UUID,
    retrieved_memories: list,
    similarity_threshold: float = 0.8,
) -> None:
    """
    When a memory is corrected, other semantically similar memories
    should have their trust reduced. This models proactive interference.
    """
    corrected = None
    for sm in retrieved_memories:
        if sm.memory.id == corrected_memory_id:
            corrected = sm.memory
            break

    if not corrected:
        return

    for sm in retrieved_memories:
        if sm.memory.id == corrected_memory_id:
            continue

        # Skip biographical memories — they're historical, not interfering
        if "biographical" in (sm.memory.context_tags or []):
            continue

        similarity = compute_vector_similarity(
            corrected.embedding, sm.memory.embedding
        )

        if similarity >= similarity_threshold:
            penalty = 0.1 * similarity
            sm.memory.trust_score = max(0.1, sm.memory.trust_score - penalty)

