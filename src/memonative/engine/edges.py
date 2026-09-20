import json
import logging
from uuid import UUID
from datetime import timedelta
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from memonative.db.models import Memory, MemoryEdge

logger = logging.getLogger(__name__)

async def create_temporal_edges(
    db: AsyncSession, tenant_id: UUID, user_id: UUID, new_memory_id: UUID,
    window_minutes: int = 30,
) -> None:
    recent = await db.execute(
        select(Memory).where(
            Memory.tenant_id == tenant_id,
            Memory.user_id == user_id,
            Memory.id != new_memory_id,
            Memory.created_at >= func.now() - timedelta(minutes=window_minutes),
        )
    )
    for mem in recent.scalars():
        db.add(MemoryEdge(
            tenant_id=tenant_id,
            from_memory_id=new_memory_id,
            to_memory_id=mem.id,
            edge_type="temporal",
            weight=0.6,
        ))

async def check_contradictions(
    db: AsyncSession, tenant_id: UUID, user_id: UUID, new_memory: Memory, llm_client
) -> list[UUID]:
    candidates = await db.execute(
        select(Memory).where(
            Memory.tenant_id == tenant_id,
            Memory.user_id == user_id,
            Memory.memory_type == "semantic",
            Memory.decay_state.in_(["active", "fading"]),
        ).order_by(
            Memory.embedding.cosine_distance(new_memory.embedding)
        ).limit(5)
    )

    cands = candidates.scalars().all()
    if not cands:
        return []

    def format_memories(mems):
        return "\n".join([f"- [{m.id}] {m.content}" for m in mems])

    prompt = f"""Does the new fact contradict any existing facts?
    New: {new_memory.content}
    Existing: {format_memories(cands)}
    Return JSON: {{"contradicts": ["id1", "id2"] or []}}"""

    result_text = await llm_client.complete(prompt)
    try:
        parsed = json.loads(result_text)
        contradicted_ids = parsed.get('contradicts', [])
    except Exception as e:
        snippet = (result_text or "")[:300]
        logger.warning(
            "contradiction-check parse failed: %s | payload=%r", e, snippet
        )
        contradicted_ids = []

    valid_cids = []
    for cid in contradicted_ids:
        try:
            target_id = UUID(cid)
            db.add(MemoryEdge(
                tenant_id=tenant_id,
                from_memory_id=new_memory.id,
                to_memory_id=target_id,
                edge_type="contradicts",
                weight=1.0,
            ))
            valid_cids.append(target_id)
        except ValueError:
            pass

    return valid_cids

_CAUSAL_PROMPT_TEMPLATE = """\
Given a new fact and a list of existing facts about the same user, \
identify any CAUSAL relationships — where one fact led to, caused, \
enabled, or resulted from another.

New fact: {new_content}

Existing facts:
{existing_facts}

Return JSON with one field:
  "causal_links": [
    {{"from_id": "<id of the cause>", "to_id": "<id of the effect>", \
"direction": "new_caused_existing" or "existing_caused_new"}}
  ]

Rules:
- Only include clear causal links, not mere correlations or co-occurrence.
- Examples of causal: "moved to Sydney" caused "joined Sydney office"; \
"started keto diet" led to "lost 10kg".
- NOT causal: two facts about the same topic with no cause-effect chain.
- Return an empty list if no causal relationships exist.
- Use the memory IDs exactly as shown in brackets.
"""


async def create_causal_edges(
    db: AsyncSession,
    tenant_id: UUID,
    user_id: UUID,
    new_memory: Memory,
    llm_client,
) -> list[UUID]:
    """Detect causal relationships between a new memory and its neighbours.

    Runs OFF the hot path (deferred to Celery alongside contradiction
    detection). Creates EdgeType.causal edges with weight 0.7.
    """
    if new_memory.embedding is None:
        return []

    candidates = await db.execute(
        select(Memory).where(
            Memory.tenant_id == tenant_id,
            Memory.user_id == user_id,
            Memory.memory_type == "semantic",
            Memory.decay_state.in_(["active", "fading"]),
            Memory.id != new_memory.id,
        ).order_by(
            Memory.embedding.cosine_distance(new_memory.embedding)
        ).limit(5)
    )
    cands = candidates.scalars().all()
    if not cands:
        return []

    existing_facts = "\n".join([f"- [{m.id}] {m.content}" for m in cands])
    prompt = _CAUSAL_PROMPT_TEMPLATE.format(
        new_content=new_memory.content,
        existing_facts=existing_facts,
    )

    try:
        result_text = await llm_client.complete(prompt)
        parsed = json.loads(result_text)
        causal_links = parsed.get("causal_links", [])
    except Exception as e:
        snippet = (result_text if "result_text" in dir() else "")[:300]
        logger.warning("causal-edge parse failed: %s | payload=%r", e, snippet)
        return []

    valid_ids = {str(m.id) for m in cands}
    valid_ids.add(str(new_memory.id))
    created = []

    for link in causal_links:
        try:
            from_id_str = link.get("from_id", "")
            to_id_str = link.get("to_id", "")
            direction = link.get("direction", "")

            if direction == "new_caused_existing":
                from_id = new_memory.id
                to_id = UUID(to_id_str)
            elif direction == "existing_caused_new":
                from_id = UUID(from_id_str)
                to_id = new_memory.id
            else:
                from_id = UUID(from_id_str)
                to_id = UUID(to_id_str)

            if str(from_id) not in valid_ids or str(to_id) not in valid_ids:
                continue
            if from_id == to_id:
                continue

            existing = await db.execute(
                select(MemoryEdge).where(
                    MemoryEdge.tenant_id == tenant_id,
                    MemoryEdge.from_memory_id == from_id,
                    MemoryEdge.to_memory_id == to_id,
                    MemoryEdge.edge_type == "causal",
                ).limit(1)
            )
            if existing.scalar_one_or_none() is not None:
                continue

            db.add(MemoryEdge(
                tenant_id=tenant_id,
                from_memory_id=from_id,
                to_memory_id=to_id,
                edge_type="causal",
                weight=0.7,
            ))
            created.append(to_id if from_id == new_memory.id else from_id)
        except (ValueError, KeyError):
            continue

    return created


async def walk_associations(
    db: AsyncSession,
    tenant_id: UUID,
    memory_ids: list[UUID],
    max_per_memory: int = 2,
    min_edge_weight: float = 0.5,
) -> list[Memory]:
    if not memory_ids:
        return []

    # tenant_id filter is defence-in-depth: memory_ids should already
    # come from a tenant-scoped read, but enforce it on both edges and
    # neighbours so a leaked id can't pull cross-tenant data.
    result = await db.execute(
        select(Memory).join(
            MemoryEdge, MemoryEdge.to_memory_id == Memory.id
        ).where(
            MemoryEdge.tenant_id == tenant_id,
            MemoryEdge.from_memory_id.in_(memory_ids),
            MemoryEdge.weight >= min_edge_weight,
            Memory.tenant_id == tenant_id,
            Memory.decay_state.in_(["active", "fading"]),
            Memory.id.notin_(memory_ids),
        ).order_by(
            MemoryEdge.weight.desc()
        ).limit(max_per_memory * len(memory_ids))
    )
    return result.scalars().all()
