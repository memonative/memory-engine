"""
Episodic Consensus Engine
=========================
 
Solves the "Melbourne problem": when multiple episodic memories all
point to the same conclusion but individually are too noisy or
ambiguous to trigger a semantic correction.
 
This module compresses related episodic memories into a single
consensus statement with a confidence score, BEFORE the
reconsolidation engine evaluates semantic facts.
 
Human memory analogy:
  You don't remember individual moments of learning a friend moved.
  You remember: "Sarah lives in London now." That's a compressed
  consensus from multiple episodes (she mentioned it, you saw her
  London photos, she talked about the tube). Each episode alone might
  be ambiguous. Together, they're unambiguous.
 
Pipeline position:
  extract → write → retrieve → [CONSENSUS] → reconsolidate → inject
"""
 
import json
import logging
from uuid import UUID
from dataclasses import dataclass, field
from datetime import datetime
from sqlalchemy import select
from memonative.engine.write_path import (
    archive_stale_siblings,
    write_semantic_with_slot,
)

logger = logging.getLogger(__name__)
 
 
@dataclass
class EpisodicConsensus:
    """A compressed conclusion from multiple episodic memories."""
    statement: str                  # "User currently lives in Delhi"
    confidence: float               # 0.0–1.0
    supporting_memory_ids: list[UUID] = field(default_factory=list)
    contradicts_semantic_ids: list[UUID] = field(default_factory=list)
    earliest_evidence: datetime | None = None
    latest_evidence: datetime | None = None
 
    @property
    def evidence_count(self) -> int:
        return len(self.supporting_memory_ids)
 
    @property
    def temporal_span_hours(self) -> float:
        """How long the evidence has been accumulating."""
        if self.earliest_evidence and self.latest_evidence:
            delta = self.latest_evidence - self.earliest_evidence
            return delta.total_seconds() / 3600.0
        return 0.0
 
 
CONSENSUS_PROMPT = """\
You are analysing a user's episodic memories to extract their current
state of truth. These are time-ordered memories from oldest to newest.
 
Episodic memories:
{episodic_list}
 
Semantic facts currently stored:
{semantic_list}
 
Task: Identify any conclusions that the episodic evidence supports
which CONTRADICT or UPDATE the stored semantic facts.
 
Return ONLY valid JSON:
{{
  "consensuses": [
    {{
      "statement": "the current truth, stated as a present-tense fact",
      "confidence": 0.0 to 1.0,
      "supporting_memory_ids": ["id1", "id2", ...],
      "contradicts_semantic_ids": ["id of the semantic fact this replaces"],
      "reasoning": "brief chain: which episodes lead to this conclusion"
    }}
  ]
}}
 
Rules:
- Only include consensuses that CONTRADICT an existing semantic fact.
  If episodes just add colour but don't change any fact, skip them.
- Confidence reflects how many episodes agree and how unambiguous
  the conclusion is:
  - 1 episode mentioning it indirectly: 0.3–0.5
  - 2 episodes stating it clearly: 0.6–0.7
  - 3+ episodes from different angles: 0.8–0.95
- Use the MOST RECENT evidence to determine current state.
  "Moved to Delhi" + "Delhi pollution is bad" + "Melbourne was better"
  = user is NOW in Delhi, regardless of Melbourne references.
- statement must be a direct replacement for the semantic fact.
  If semantic says "User lives in Melbourne", statement should be
  "User lives in Delhi", not "User may have moved to Delhi".
- Be decisive. If 3+ episodes point one direction, confidence >= 0.85.\
"""
 
 
async def extract_consensus(
    llm_client,
    retrieved_memories: list,  # list of ScoredMemory
) -> list[EpisodicConsensus]:
    """
    Compress retrieved episodic memories into consensus statements
    that can challenge semantic facts.
 
    The key insight: we separate episodic from semantic memories,
    sort episodics by time, and ask the LLM to reason about what
    the ACCUMULATED evidence says — not what any single message says.
    """
    # Separate episodic and semantic memories
    episodics = []
    semantics = []
    for sm in retrieved_memories:
        mem = sm.memory
        if mem.memory_type.value == "episodic":
            episodics.append(mem)
        elif mem.memory_type.value == "semantic":
            semantics.append(mem)
 
    # Nothing to do if no episodics or no semantics to challenge
    if not episodics or not semantics:
        return []
 
    # Sort episodics by creation time (oldest first)
    # This gives the LLM a temporal narrative to reason over
    episodics.sort(key=lambda m: m.created_at)
 
    # Format with timestamps — this is what was missing
    episodic_list = "\n".join([
        f"- [{m.id}] ({m.created_at.strftime('%Y-%m-%d %H:%M')}) "
        f"{m.content}"
        for m in episodics
    ])
 
    semantic_list = "\n".join([
        f"- [{m.id}] (trust: {m.trust_score:.2f}) {m.content}"
        for m in semantics
    ])
 
    prompt = CONSENSUS_PROMPT.format(
        episodic_list=episodic_list,
        semantic_list=semantic_list,
    )
 
    result_text = await llm_client.complete(prompt)
 
    try:
        parsed = json.loads(result_text)
        raw_items = parsed.get("consensuses", [])
    except Exception as e:
        snippet = (result_text or "")[:300]
        logger.warning(
            "consensus parse failed: %s | payload=%r", e, snippet
        )
        return []
 
    consensuses = []
    for raw in raw_items:
        supporting_ids = []
        for id_str in raw.get("supporting_memory_ids", []):
            try:
                supporting_ids.append(UUID(id_str))
            except ValueError:
                pass
 
        contradicts_ids = []
        for id_str in raw.get("contradicts_semantic_ids", []):
            try:
                contradicts_ids.append(UUID(id_str))
            except ValueError:
                pass
 
        # Find temporal bounds from supporting episodes
        supporting_mems = [
            m for m in episodics if m.id in supporting_ids
        ]
        earliest = min(
            (m.created_at for m in supporting_mems),
            default=None,
        )
        latest = max(
            (m.created_at for m in supporting_mems),
            default=None,
        )
 
        consensuses.append(EpisodicConsensus(
            statement=raw.get("statement", ""),
            confidence=raw.get("confidence", 0.0),
            supporting_memory_ids=supporting_ids,
            contradicts_semantic_ids=contradicts_ids,
            earliest_evidence=earliest,
            latest_evidence=latest,
        ))
 
    return consensuses
 
 
async def apply_consensus_corrections(
    db,
    llm_client,
    consensuses: list[EpisodicConsensus],
    retrieved_memories: list,
    min_confidence: float = 0.75,
    min_evidence_count: int = 2,
):
    """
    Apply consensus-driven corrections to semantic facts.
 
    This is deterministic, not LLM-dependent. If the consensus
    meets the confidence and evidence thresholds, the semantic
    fact gets corrected. No second-guessing.
 
    The thresholds encode the principle:
    - A single episode shouldn't override a trusted fact
    - But 2+ episodes pointing the same direction SHOULD,
      even if each episode alone is ambiguous
    """
    # Build lookup of semantic memories by ID
    semantic_lookup = {}
    for sm in retrieved_memories:
        if sm.memory.memory_type.value == "semantic":
            semantic_lookup[sm.memory.id] = sm.memory
 
    corrections = []
 
    for consensus in consensuses:
        # Gate 1: confidence threshold
        if consensus.confidence < min_confidence:
            continue
 
        # Gate 2: evidence count threshold
        if consensus.evidence_count < min_evidence_count:
            continue
 
        # Gate 3: must actually target a semantic fact
        if not consensus.contradicts_semantic_ids:
            continue
 
        for semantic_id in consensus.contradicts_semantic_ids:
            semantic_mem = semantic_lookup.get(semantic_id)
            if not semantic_mem:
                continue

            old_content = semantic_mem.content
            old_trust = semantic_mem.trust_score
            new_embedding = await llm_client.embed(consensus.statement)

            # Trust adjustment policy: penalty for having been wrong,
            # partially recovered by how confident the correction is.
            base_penalty = -0.15
            recovery = consensus.confidence * 0.10
            target_trust = max(0.1, min(1.0, old_trust + base_penalty + recovery))

            if semantic_mem.attribute_slot:
                # Slotted: route through the revision chain so
                # MemoryRevision history stays consistent with
                # Memory.content.
                await write_semantic_with_slot(
                    db=db,
                    tenant_id=semantic_mem.tenant_id,
                    user_id=semantic_mem.user_id,
                    content=consensus.statement,
                    attribute_slot=semantic_mem.attribute_slot,
                    embedding=new_embedding,
                    llm_client=llm_client,
                    source="consolidated",
                    salience=semantic_mem.salience,
                    context_tags=list(semantic_mem.context_tags or []),
                    correction_type="temporal_update",
                    trigger_message=f"[consensus] {consensus.statement}",
                )
                # write_semantic_with_slot resets trust to its default
                # for corrections; override with the consensus policy.
                semantic_mem.trust_score = target_trust
            else:
                # No slot — direct mutation; nothing to revise.
                semantic_mem.content = consensus.statement
                semantic_mem.embedding = new_embedding
                semantic_mem.trust_score = target_trust
                semantic_mem.reinforcement_count += 1

            # Archive stale siblings to clear out duplicates/old versions
            await archive_stale_siblings(
                db=db,
                tenant_id=semantic_mem.tenant_id,
                user_id=semantic_mem.user_id,
                corrected_memory_id=semantic_id,
                new_embedding=new_embedding,
                similarity_threshold=0.75,
            )

            corrections.append({
                "semantic_id": semantic_id,
                "old_content": old_content,
                "new_content": consensus.statement,
                "confidence": consensus.confidence,
                "evidence_count": consensus.evidence_count,
            })

    return corrections
