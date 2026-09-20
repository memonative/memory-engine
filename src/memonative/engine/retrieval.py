import asyncio
from uuid import UUID
from dataclasses import dataclass
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from memonative.db.models import Memory, MemoryRevision
from memonative.db.enums import RevisionType
from memonative.engine.decay import compute_decay, reinforce_memories_bulk
from memonative.engine.edges import walk_associations
from memonative.engine.fusion import reciprocal_rank_fusion
from memonative.config import INTENT_TYPE_WEIGHTS, SCORING_WEIGHTS, settings
from memonative.engine.cross_encoder import cross_encoder_rerank


@dataclass
class ScoredMemory:
    memory: Memory
    score: float


async def get_slot_history(
    db: AsyncSession, tenant_id: UUID, user_id: UUID, memory_id: UUID,
    attribute_slot: str,
) -> list[MemoryRevision]:
    """Fetch the full revision chain for an attribute slot."""
    result = await db.execute(
        select(MemoryRevision).where(
            MemoryRevision.tenant_id == tenant_id,
            MemoryRevision.memory_id == memory_id,
            MemoryRevision.user_id == user_id,
            MemoryRevision.attribute_slot == attribute_slot,
        ).order_by(MemoryRevision.sequence_number.asc())
    )
    return result.scalars().all()


def format_revision_history(revisions: list[MemoryRevision]) -> str:
    """Format a slot's revision history into a concise string.

    Example output:
      "Was in Delhi (changed: relocated) · Paris was stated in error (retracted)"
    """
    if len(revisions) <= 1:
        return ""

    # Skip the current revision (last one with valid_until=None)
    past_revisions = [r for r in revisions if r.valid_until is not None]

    if not past_revisions:
        return ""

    parts = []
    for rev in reversed(past_revisions):  # Most recent history first
        if rev.revision_type == RevisionType.initial:
            # Only show if it was later retracted
            next_rev = next(
                (r for r in revisions if r.sequence_number == rev.sequence_number + 1),
                None,
            )
            if next_rev and next_rev.revision_type == RevisionType.retraction:
                parts.append(f"{rev.content} (was stated in error)")
            else:
                date_str = rev.valid_until.strftime('%b %Y') if rev.valid_until else ""
                parts.append(f"{rev.content} (until {date_str})")

        elif rev.revision_type == RevisionType.retraction:
            # This was a correction of an error — the content here is what REPLACED the error
            # The error itself is in the previous revision
            continue  # Handled by the initial/previous revision's annotation

        elif rev.revision_type == RevisionType.temporal_update:
            date_str = rev.valid_from.strftime('%b %Y') if rev.valid_from else ""
            end_str = rev.valid_until.strftime('%b %Y') if rev.valid_until else "present"
            parts.append(f"{rev.content} ({date_str}–{end_str})")

    return " · ".join(parts) if parts else ""


def _format_revision_block(revisions: list[MemoryRevision]) -> str:
    """Format revision chain into indented lines for LLM context."""
    if len(revisions) <= 1:
        return ""

    lines = ["  Revision history:"]
    for rev in revisions:
        valid_range = ""
        if rev.valid_from:
            end = rev.valid_until.strftime("%Y-%m-%d") if rev.valid_until else "present"
            valid_range = f" (valid {rev.valid_from.strftime('%Y-%m-%d')} to {end})"
        rev_type = rev.revision_type.value
        type_label = f" [{rev_type}]" if rev_type != "initial" else ""
        lines.append(f"    - \"{rev.content}\"{type_label}{valid_range}")

    return "\n".join(lines)


_EVIDENCE_SOURCES = frozenset({"user_stated", "user_repeated", "consolidated"})


async def format_for_injection(
    memories: list[ScoredMemory],
    db: AsyncSession | None = None,
) -> str:
    """Format memories for LLM context injection.

    Semantic memories are split by provenance:
      - user_stated / user_repeated / consolidated → "Known Facts"
      - system_inferred → "Inferences (unconfirmed)" with trust score

    When a database session is available, includes revision history
    for semantic memories with attribute slots.
    """
    sections = {
        "procedural": [],
        "known_facts": [],
        "inferences": [],
        "episodic": [],
    }

    seen_contents = set()
    for sm in memories:
        mem = sm.memory
        content = mem.content
        if content in seen_contents:
            continue
        seen_contents.add(content)

        if mem.memory_type == "procedural":
            sections["procedural"].append(f"[Style] {content}")

        elif mem.memory_type == "semantic":
            source = getattr(mem.source, "value", str(mem.source)) if mem.source else "user_stated"
            is_evidence = source in _EVIDENCE_SOURCES

            if is_evidence:
                fact_line = f"[Fact — slot: {mem.attribute_slot or 'general'}] {content} (trust: {mem.trust_score:.1f})"
            else:
                fact_line = f"[Inference — slot: {mem.attribute_slot or 'general'}] {content} (trust: {mem.trust_score:.1f})"

            if db and mem.attribute_slot:
                revisions = await get_slot_history(
                    db, mem.tenant_id, mem.user_id, mem.id, mem.attribute_slot
                )
                rev_block = _format_revision_block(revisions)
                if rev_block:
                    fact_line += "\n" + rev_block

            if is_evidence:
                sections["known_facts"].append(fact_line)
            else:
                sections["inferences"].append(fact_line)

        elif mem.memory_type == "episodic":
            age = mem.created_at.strftime('%Y-%m-%d')
            if "biographical" in (mem.context_tags or []):
                sections["episodic"].append(f"[{age} — biographical] {content}")
            else:
                sections["episodic"].append(f"[{age}] {content}")

    output = ""
    if sections["procedural"]:
        output += "## Interaction Preferences\n"
        output += "\n".join(sections["procedural"]) + "\n\n"
    if sections["known_facts"]:
        output += "## Known Facts\n"
        output += "\n".join(sections["known_facts"]) + "\n\n"
    if sections["inferences"]:
        output += "## Inferences (unconfirmed)\n"
        output += "\n".join(sections["inferences"]) + "\n\n"
    if sections["episodic"]:
        output += "## Recent Context\n"
        output += "\n".join(sections["episodic"])

    return output.strip()


def deduplicate(
    top_memories: list[ScoredMemory], associated: list[Memory]
) -> list[ScoredMemory]:
    seen = {sm.memory.id for sm in top_memories}
    res = list(top_memories)
    for mem in associated:
        if mem.id not in seen:
            seen.add(mem.id)
            res.append(ScoredMemory(memory=mem, score=0.0))
    return res


async def retrieve_memories(
    db: AsyncSession,
    tenant_id: UUID,
    user_id: UUID,
    message: str,
    intent: dict,
    embedding: list[float],
    top_k: int = 8,
    reinforce: bool = True,
    type_override: list[str] | None = None,
    include_associations: bool = True,
    llm_client=None,
) -> list[ScoredMemory]:
    """Retrieve relevant memories via multi-strategy fusion.

    Fuses up to four ranked lists via RRF:
      1. Vector (HNSW cosine)
      2. Full-text search (tsvector)
      3. Graph walk (edge-connected neighbours of vector hits)
      4. Temporal/entity retrieval (intent-gated, for enumerative queries)

    reinforce=False makes this a side-effect-free read (no decay reset,
    no last_accessed bump) — used by the agent-native /v1/memory/search
    endpoint where the caller is browsing, not interacting.

    type_override lets an agent restrict the search to specific memory
    types (e.g. only semantic facts) without faking an intent.
    """

    primary_intent = intent.get('primary', 'social')
    type_weights = INTENT_TYPE_WEIGHTS.get(
        primary_intent, INTENT_TYPE_WEIGHTS['social']
    )
    if type_override:
        # Uniform weight for explicit type filters — caller already
        # decided what's relevant; intent-derived weights would just
        # re-rank against their wishes.
        type_weights = {t: 1.0 for t in type_override}
        eligible_types = list(type_override)
    else:
        eligible_types = [t for t, w in type_weights.items() if w >= 0.2]

    # Push candidate generation into Postgres. The HNSW index on
    # memories.embedding (ix_memories_embedding_hnsw, vector_cosine_ops)
    # turns vector candidate generation into an approximate O(log N)
    # lookup. We also fire a parallel full-text query against the
    # ix_memories_content_tsv GIN index so exact-token queries
    # (order IDs, names, error codes) aren't drowned out by semantic
    # neighbours. The two candidate lists are then fused with RRF.
    candidate_k = max(50, top_k * 6)
    cosine_distance = Memory.embedding.cosine_distance(embedding)
    vector_stmt = (
        select(Memory, (1 - cosine_distance).label("vector_sim"))
        .where(
            Memory.tenant_id == tenant_id,
            Memory.user_id == user_id,
            Memory.memory_type.in_(eligible_types),
            Memory.decay_state.in_(["active", "fading"]),
            Memory.embedding.is_not(None),
        )
        .order_by(cosine_distance)
        .limit(candidate_k)
    )

    ts_query = func.plainto_tsquery('english', message)
    ts_rank = func.ts_rank_cd(Memory.content_tsv, ts_query)
    fts_stmt = (
        select(Memory, ts_rank.label("fts_rank"))
        .where(
            Memory.tenant_id == tenant_id,
            Memory.user_id == user_id,
            Memory.memory_type.in_(eligible_types),
            Memory.decay_state.in_(["active", "fading"]),
            Memory.content_tsv.op("@@")(ts_query),
        )
        .order_by(ts_rank.desc())
        .limit(50)
    )

    vector_res, fts_res = await asyncio.gather(
        db.execute(vector_stmt),
        db.execute(fts_stmt),
    )
    vector_rows = vector_res.all()
    fts_rows = fts_res.all()

    by_id: dict[UUID, Memory] = {}
    vector_sim_by_id: dict[UUID, float] = {}
    for mem, vector_sim in vector_rows:
        by_id[mem.id] = mem
        vector_sim_by_id[mem.id] = vector_sim or 0.0
    for mem, _ in fts_rows:
        by_id.setdefault(mem.id, mem)

    # ── Strategy 3: graph walk (promoted from post-hoc to fusion input) ──
    # Seed from vector results so graph-connected memories compete on rank
    # instead of being appended at score 0.0.
    graph_ids: list[UUID] = []
    if include_associations:
        vector_seed_ids = [mem.id for mem, _ in vector_rows[:top_k * 3]]
        if vector_seed_ids:
            graph_mems = await walk_associations(
                db=db,
                tenant_id=tenant_id,
                memory_ids=vector_seed_ids,
                max_per_memory=2,
                min_edge_weight=0.5,
            )
            for gm in graph_mems:
                by_id.setdefault(gm.id, gm)
                graph_ids.append(gm.id)

    # ── Strategy 4: temporal/entity retrieval (intent-gated, no LLM call) ──
    temporal_ids: list[UUID] = []
    if (
        settings.TIMELINE_RETRIEVAL_ENABLED
        and primary_intent in ("recall", "reflect")
        and not type_override
        and llm_client is not None
    ):
        from memonative.engine.timeline import retrieve_timeline
        temporal_scored = await retrieve_timeline(
            db, llm_client,
            tenant_id=tenant_id,
            user_id=user_id,
            entity=message,
            synonyms=[],
            top_k=20,
        )
        for ts in temporal_scored:
            by_id.setdefault(ts.memory.id, ts.memory)
            temporal_ids.append(ts.memory.id)

    # ── RRF fusion across all available strategies ──
    ranked_lists = [
        [mem.id for mem, _ in vector_rows],
        [mem.id for mem, _ in fts_rows],
    ]
    if graph_ids:
        ranked_lists.append(graph_ids)
    if temporal_ids:
        ranked_lists.append(temporal_ids)

    fused_ids = reciprocal_rank_fusion(ranked_lists, k=60)

    scored = []
    for mem_id in fused_ids:
        mem = by_id[mem_id]
        decay = compute_decay(
            half_life_hours=mem.half_life_hours,
            reinforcement_count=mem.reinforcement_count,
            salience=mem.salience,
            last_accessed_at=mem.last_accessed_at,
        )
        if decay.strength < 0.1:
            continue

        vector_sim = vector_sim_by_id.get(mem_id, 0.0)

        final_score = (
            SCORING_WEIGHTS['type_match'] * type_weights.get(mem.memory_type.value, 0.0)
            + SCORING_WEIGHTS['strength'] * decay.strength
            + SCORING_WEIGHTS['trust'] * mem.trust_score
            + SCORING_WEIGHTS['salience'] * mem.salience
            + SCORING_WEIGHTS['vector'] * vector_sim
        )

        scored.append(ScoredMemory(memory=mem, score=final_score))

    scored.sort(key=lambda s: s.score, reverse=True)

    if settings.RERANK_ENABLED and scored:
        scored = await cross_encoder_rerank(
            query=message,
            candidates=scored,
            top_k=top_k,
            backend=settings.RERANK_BACKEND,
            model_name=settings.RERANK_MODEL,
            api_key=settings.RERANK_API_KEY.get_secret_value() or None,
            api_provider=settings.RERANK_API_PROVIDER or None,
            api_model=settings.RERANK_API_MODEL or None,
            blend_weight=settings.RERANK_BLEND_WEIGHT,
            max_candidates=settings.RERANK_MAX_CANDIDATES,
        )
        top_memories = scored
    else:
        top_memories = scored[:top_k]

    # Graph results are now part of fusion — no post-hoc append needed.
    if reinforce:
        await reinforce_memories_bulk(
            db, [s.memory.id for s in top_memories]
        )

    return top_memories
