"""Memonative adapter for the LongMemEval eval.

Runs in-process against the same Postgres the API uses. We bypass the
HTTP surface so 250k+ requests aren't bottlenecked on uvicorn, but go
through the same engine modules (extraction → write_path → retrieval)
the API does, so what we measure is the real system.

Granularity: one extraction call per session (concatenated turns),
not per turn. At 50 questions × ~50 sessions, per-turn extraction is
~$30+ in OpenAI spend; per-session keeps a 100-question run under $5.
The trade-off: extraction may miss facts buried in a long session,
but mem0 sees the same multi-turn input so the comparison stays fair.

Each extracted memory is tagged with `lme_session:<id>` in
context_tags. The eval's recall@5 metric reads that tag back from
retrieved memories to check needle-session overlap.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid5, NAMESPACE_DNS

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from memonative.api.llm import LLMClient
from memonative.config import settings
from memonative.db.models import Goal, Interaction, Memory, MemoryEdge, MemoryRevision, Tenant
from memonative.engine import extraction, rerank, retrieval, timeline, write_path

from eval.longmemeval.adapter import MemoryAdapter, RetrievedMemory


# Stable, dedicated tenant for the eval. Picked outside any production
# UUID range so reset_user wipes never accidentally touch real data.
EVAL_TENANT_ID = UUID("00000000-0000-0000-0000-00000000eeee")
SESSION_TAG_PREFIX = "lme_session:"


def _user_uuid(user_id: str) -> UUID:
    """Deterministic LongMemEval user_id → UUID. uuid5 keeps reruns
    pointing at the same row so resume-from-JSONL works."""
    return uuid5(NAMESPACE_DNS, f"longmemeval:{user_id}")


def _parse_session_date(date_str: str) -> datetime:
    """Parse LongMemEval date strings like '2023/03/04 (Sat) 22:43'."""
    clean = date_str.split("(")[0].strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(clean, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.now(timezone.utc)


def _format_session(turns: list[dict]) -> str:
    """Concatenate (role, content) turns into one extraction input.

    Speaker tags matter: extraction's prompt assumes a single user
    message, so without the tag it would attribute assistant claims
    to the user. Tagging fixes that without retraining the prompt.
    """
    parts = []
    for t in turns:
        role = (t.get("role") or "").upper() or "USER"
        content = t.get("content") or ""
        parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


class MemonativeAdapter(MemoryAdapter):
    name = "memonative"

    def __init__(self):
        self._engine = create_async_engine(settings.DATABASE_URL)
        self._session_maker = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False,
        )
        self._llm = LLMClient()
        self._tenant_id = EVAL_TENANT_ID
        self._tenant_ensured = False

    async def aclose(self) -> None:
        await self._engine.dispose()

    async def _ensure_tenant(self) -> None:
        if self._tenant_ensured:
            return
        async with self._session_maker() as db:
            existing = await db.execute(
                select(Tenant).where(Tenant.id == self._tenant_id)
            )
            if existing.scalar_one_or_none() is None:
                db.add(Tenant(
                    id=self._tenant_id,
                    name="longmemeval-eval",
                    status="active",
                ))
                await db.commit()
        self._tenant_ensured = True

    async def ingest(
        self,
        user_id: str,
        session_id: str,
        session_date: str,
        turns: list[dict],
    ) -> None:
        await self._ensure_tenant()
        user_uuid = _user_uuid(user_id)

        session_text = _format_session(turns)
        if not session_text.strip():
            return

        # Pass the session date to extraction so relative phrases in
        # the conversation ("a month ago", "last week") get resolved
        # to absolute event_dates the reader can do math with.
        session_dt = _parse_session_date(session_date)
        extracted = await extraction.extract(
            self._llm, session_text, conversation_date=session_dt,
        )
        if not extracted.memories_to_write:
            return

        # Tag every candidate with its source session id so query() can
        # round-trip session provenance back to the runner.
        session_tag = f"{SESSION_TAG_PREFIX}{session_id}"
        for cand in extracted.memories_to_write:
            tags = list(cand.get("context_tags") or [])
            tags.append(session_tag)
            cand["context_tags"] = tags

        async with self._session_maker() as db:
            written_ids, _ = await write_path.write_memories(
                db,
                tenant_id=self._tenant_id,
                user_id=user_uuid,
                candidates_raw=extracted.memories_to_write,
                llm_client=self._llm,
                trigger_message=session_text[:500],
            )
            if written_ids:
                await db.execute(
                    update(Memory)
                    .where(Memory.id.in_(written_ids))
                    .values(created_at=session_dt)
                )
                await db.execute(
                    update(MemoryRevision)
                    .where(MemoryRevision.memory_id.in_(written_ids))
                    .values(valid_from=session_dt)
                )
            await db.commit()

    async def query(
        self,
        user_id: str,
        question: str,
        top_k: int = 5,
    ) -> list[RetrievedMemory]:
        user_uuid = _user_uuid(user_id)
        embedding = await self._llm.embed(question)

        intent = {"primary": "social", "mode": "conversational"}

        # Retrieve 2x the target so the LLM rerank has material to dedup
        # (a noisy top_k is what hurt v3-deepseek-k10-30 precise-value
        # lookups). Engine retrieval stays deterministic; the LLM dedup
        # is opt-in at this adapter boundary so production code paths
        # don't pay the latency.
        oversample_k = max(top_k * 2, top_k + 5)
        async with self._session_maker() as db:
            candidates = await retrieval.retrieve_memories(
                db,
                tenant_id=self._tenant_id,
                user_id=user_uuid,
                message=question,
                intent=intent,
                embedding=embedding,
                top_k=oversample_k,
                reinforce=False,
                include_associations=False,
            )
            scored = await rerank.llm_dedup_rerank(
                self._llm, question, candidates, target_k=top_k,
            )

            results: list[RetrievedMemory] = []
            for sm in scored:
                session_id = ""
                for tag in (sm.memory.context_tags or []):
                    if tag.startswith(SESSION_TAG_PREFIX):
                        session_id = tag[len(SESSION_TAG_PREFIX):]
                        break

                rev_history: list[dict] = []
                if sm.memory.attribute_slot and sm.memory.memory_type == "semantic":
                    revisions = await retrieval.get_slot_history(
                        db, self._tenant_id, user_uuid,
                        sm.memory.id, sm.memory.attribute_slot,
                    )
                    for rev in revisions:
                        rev_history.append({
                            "content": rev.content,
                            "revision_type": rev.revision_type.value,
                            "reason": rev.reason,
                            "valid_from": rev.valid_from.strftime("%Y-%m-%d") if rev.valid_from else None,
                            "valid_until": rev.valid_until.strftime("%Y-%m-%d") if rev.valid_until else None,
                        })

                mem_type = sm.memory.memory_type
                type_str = mem_type.value if hasattr(mem_type, "value") else str(mem_type)

                ev = sm.memory.event_type
                event_str = ev.value if hasattr(ev, "value") else str(ev or "none")

                ed = sm.memory.event_date
                event_date_str = ed.strftime("%Y-%m-%d") if ed else ""

                results.append(RetrievedMemory(
                    content=sm.memory.content,
                    source_session_id=session_id,
                    score=sm.score,
                    memory_type=type_str,
                    attribute_slot=sm.memory.attribute_slot,
                    trust_score=sm.memory.trust_score,
                    created_at=sm.memory.created_at.strftime("%Y-%m-%d") if sm.memory.created_at else "",
                    revision_history=rev_history,
                    event_type=event_str,
                    event_date=event_date_str,
                ))
        return results

    async def query_timeline_block(self, user_id: str, question: str) -> str:
        """Second-pass retrieval for inventory / sequence questions.

        Returns a chronological text block to inject alongside the main
        retrieved memories, or "" when the question isn't a timeline
        question or when the focused pass turns up nothing useful.

        Decoupled from `query` so the base MemoryAdapter contract stays
        portable: mem0 doesn't implement this, and the runner falls back
        to plain query when the adapter doesn't expose it.
        """
        cls = await timeline.classify_timeline_question(self._llm, question)
        if not cls.is_timeline:
            return ""

        user_uuid = _user_uuid(user_id)
        # Same oversample-then-rerank pattern as `query` — but bigger
        # since timeline questions explicitly want broad recall (counting,
        # ordering across history). 40 candidates → 20 distinct.
        target_k = 20
        oversample_k = target_k * 2
        async with self._session_maker() as db:
            candidates = await timeline.retrieve_timeline(
                db,
                self._llm,
                tenant_id=self._tenant_id,
                user_id=user_uuid,
                entity=cls.entity,
                synonyms=cls.synonyms,
                top_k=oversample_k,
            )
        scored = await rerank.llm_dedup_rerank(
            self._llm, question, candidates, target_k=target_k,
        )
        return timeline.format_timeline_block(scored, cls.entity)

    async def reset_user(self, user_id: str) -> None:
        user_uuid = _user_uuid(user_id)
        async with self._session_maker() as db:
            # ON DELETE CASCADE on memory_id covers edges, revisions,
            # and reconsolidation_log — those auto-cascade with memory
            # deletion. Goals and Interactions don't FK to memories,
            # so they need explicit deletes.
            await db.execute(
                delete(Memory).where(
                    Memory.tenant_id == self._tenant_id,
                    Memory.user_id == user_uuid,
                )
            )
            await db.execute(
                delete(Goal).where(
                    Goal.tenant_id == self._tenant_id,
                    Goal.user_id == user_uuid,
                )
            )
            await db.execute(
                delete(Interaction).where(
                    Interaction.tenant_id == self._tenant_id,
                    Interaction.user_id == user_uuid,
                )
            )
            await db.commit()
