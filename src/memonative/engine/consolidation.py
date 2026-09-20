import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from uuid import UUID
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from memonative.db.models import Memory, MemoryEdge
from memonative.engine.write_path import find_near_duplicate
from memonative.engine.decay import reinforce_memory
from memonative.engine.scoring import compute_vector_similarity

logger = logging.getLogger(__name__)


@dataclass
class MemoryCluster:
    memories: list[Memory]
    label: str = "cluster"

    @property
    def user_id(self) -> UUID:
        return self.memories[0].user_id

    @property
    def tenant_id(self) -> UUID:
        return self.memories[0].tenant_id

    @property
    def shared_tags(self) -> list[str]:
        """Tags appearing in at least half the cluster's memories.

        Falls back to the cluster label when no tag is widely shared,
        so the consolidated semantic memory always has at least one
        meaningful tag for downstream retrieval.
        """
        if not self.memories:
            return [self.label]
        counter: Counter[str] = Counter()
        for m in self.memories:
            counter.update(m.context_tags or [])
        threshold = max(1, len(self.memories) // 2)
        shared = [t for t, c in counter.items() if c >= threshold]
        return shared or [self.label]


async def find_episodic_clusters(
    db: AsyncSession,
    tenant_id: UUID,
    user_id: UUID,
    min_cluster_size: int = 2,
    min_total_reinforcements: int = 3,
    similarity_threshold: float = 0.75,
) -> list[MemoryCluster]:
    """Group episodic memories by embedding similarity.

    Greedy single-seed clustering: walk episodes once, and for each
    unclustered memory start a new cluster containing every other
    unclustered memory whose embedding is within `similarity_threshold`
    of the seed. This is O(N^2) but consolidation runs on a 4-hour
    cadence and per-user counts are bounded.

    Tag-based grouping was the previous approach and missed any pair
    of related episodes that didn't happen to share a tag string.
    """
    result = await db.execute(
        select(Memory).where(
            Memory.tenant_id == tenant_id,
            Memory.user_id == user_id,
            Memory.memory_type == "episodic",
            Memory.decay_state.in_(["active", "fading"]),
            Memory.embedding.is_not(None),
        )
    )
    episodes = list(result.scalars())

    clustered_ids: set[UUID] = set()
    clusters: list[MemoryCluster] = []

    for i, seed in enumerate(episodes):
        if seed.id in clustered_ids:
            continue

        members = [seed]
        clustered_ids.add(seed.id)

        for other in episodes[i + 1:]:
            if other.id in clustered_ids:
                continue
            sim = compute_vector_similarity(seed.embedding, other.embedding)
            if sim >= similarity_threshold:
                members.append(other)
                clustered_ids.add(other.id)

        if len(members) < min_cluster_size:
            continue
        if sum(m.reinforcement_count for m in members) < min_total_reinforcements:
            continue

        # Label by the most common tag in the cluster, falling back to
        # the seed id so the cluster is always identifiable in logs.
        tag_counter: Counter[str] = Counter()
        for m in members:
            tag_counter.update(m.context_tags or [])
        label = (
            tag_counter.most_common(1)[0][0]
            if tag_counter
            else f"cluster:{seed.id}"
        )

        clusters.append(MemoryCluster(memories=members, label=label))

    return clusters

CONSOLIDATION_PROMPT = """
You are analysing a cluster of episodic memories from a user's interactions.

Episodes:
{formatted_episodes}

Extract stable facts, preferences, or knowledge that appears consistently
across 2+ episodes. Return ONLY valid JSON:
{{
  "facts": [
    {{
      "content": "the stable fact or preference",
      "confidence": 0.0 to 1.0,
      "source_episode_ids": ["id1", "id2"]
    }}
  ]
}}

Rules:
- Only include facts supported by 2+ episodes
- Confidence must reflect how consistently the fact appears
- Be conservative: when uncertain, exclude
- Do NOT include one-off mentions or transient states
"""

class ExtractedFact:
    def __init__(self, data: dict):
        self.content = data.get("content", "")
        self.confidence = data.get("confidence", 0.0)
        self.source_episode_ids = []
        for id_str in data.get("source_episode_ids", []):
            try:
                self.source_episode_ids.append(UUID(id_str))
            except ValueError:
                pass

async def extract_facts(cluster: MemoryCluster, llm_client) -> list[ExtractedFact]:
    formatted = "\n".join([f"- [{m.id}] {m.content}" for m in cluster.memories])
    prompt = CONSOLIDATION_PROMPT.format(formatted_episodes=formatted)
    
    result_text = await llm_client.complete(prompt)
    try:
        parsed = json.loads(result_text)
        return [ExtractedFact(f) for f in parsed.get("facts", [])]
    except Exception as e:
        snippet = (result_text or "")[:300]
        logger.warning(
            "consolidation parse failed: %s | payload=%r", e, snippet
        )
        return []

async def consolidate_cluster(db: AsyncSession, cluster: MemoryCluster, llm_client):
    facts = await extract_facts(cluster, llm_client)

    for fact in facts:
        if fact.confidence < 0.7 or not fact.content:
            continue

        embedding = await llm_client.embed(fact.content)

        existing = await find_near_duplicate(
            db=db,
            tenant_id=cluster.tenant_id,
            user_id=cluster.user_id,
            embedding=embedding,
            threshold=0.9,
        )

        if existing:
            await reinforce_memory(db, existing)
            existing.trust_score = min(1.0, existing.trust_score + 0.1)
        else:
            salience = max(ep.salience for ep in cluster.memories) if cluster.memories else 0.5
            semantic = Memory(
                tenant_id=cluster.tenant_id,
                user_id=cluster.user_id,
                memory_type="semantic",
                content=fact.content,
                embedding=embedding,
                source="consolidated",
                trust_score=fact.confidence,
                salience=salience,
                half_life_hours=720.0,
                context_tags=cluster.shared_tags,
            )
            db.add(semantic)
            await db.flush()

            for ep_id in fact.source_episode_ids:
                db.add(MemoryEdge(
                    tenant_id=cluster.tenant_id,
                    from_memory_id=ep_id,
                    to_memory_id=semantic.id,
                    edge_type="consolidates",
                    weight=fact.confidence,
                ))

    await db.commit()
