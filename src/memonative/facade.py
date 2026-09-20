"""`MemoryEngine` — the engine as a library.

Everything the HTTP routes used to do inline lives here, so a caller can
`from memonative.facade import MemoryEngine` and drive the pipeline in-process
with no web framework installed and no HTTP round trip. `api/routes.py` is now
one of its callers rather than its owner.

Three things stayed behind in the route on purpose:

  - `emit_usage_event` — billing, not memory. It no-ops outside the cloud
    edition and needs a request-scoped timer the engine has no notion of.
  - Parsing path parameters from strings into UUIDs — transport-level
    concern. Every method here takes a real `UUID`.
  - `/v1/tools/{provider}` — no database, no tenant, no memory.

Errors are `memonative.errors` types carrying the status code their
`HTTPException` carried, so the API layer re-raises them mechanically.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from memonative.config import DEFAULT_TENANT_ID, get_decay_profile
from memonative.db.database import SessionLocal, set_rls_tenant
from memonative.db.enums import DecayState
from memonative.db.models import Interaction, Memory, MemoryRevision
from memonative.engine import consolidation, extraction, goals, retrieval, write_path
from memonative.engine.consensus import apply_consensus_corrections, extract_consensus
from memonative.engine.decay import compute_decay
from memonative.engine.reconsolidation import (
    apply_interference,
    apply_reconsolidation_plan,
    plan_reconsolidation,
)
from memonative.errors import NotFoundError
from memonative.llm import LLMProtocol
from memonative.observability import time_stage
from memonative.types import (
    AuditEntry,
    AuditTrailResponse,
    DecayedMemory,
    IntentResponse,
    MemoryDetail,
    MemoryFactsResponse,
    MemoryRevisionItem,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySlotItem,
    MemorySlotsResponse,
    MemoryWriteItem,
    MemoryWriteRequest,
    MemoryWriteResponse,
    ProcessRequest,
    ProcessResponse,
    RetrievedMemory,
)

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate. ~4 chars/token is the OpenAI rule of thumb;
    accurate enough for budget-trimming, and avoids loading tiktoken."""
    return max(1, (len(text) + 3) // 4)


def _to_memory_detail(mem: Memory, score: float = 0.0) -> MemoryDetail:
    return MemoryDetail(
        id=mem.id,
        content=mem.content,
        type=mem.memory_type.value,
        attribute_slot=mem.attribute_slot,
        score=score,
        strength=mem.strength,
        trust_score=mem.trust_score,
        salience=mem.salience,
        source=mem.source.value if hasattr(mem.source, "value") else str(mem.source),
        decay_state=mem.decay_state.value if hasattr(mem.decay_state, "value") else str(mem.decay_state),
        reinforcement_count=mem.reinforcement_count,
        context_tags=list(mem.context_tags or []),
        created_at=mem.created_at,
        last_accessed_at=mem.last_accessed_at,
        estimated_tokens=_estimate_tokens(mem.content),
    )


class _Default:
    """Sentinel for `set_retention`. Needed because `None` is already taken —
    it is the value that means "pinned" — so absence needs its own spelling."""


_DEFAULT = _Default()


def _to_decayed_memory(mem: Memory) -> DecayedMemory:
    decay = compute_decay(
        half_life_hours=mem.half_life_hours,
        reinforcement_count=mem.reinforcement_count,
        salience=mem.salience,
        last_accessed_at=mem.last_accessed_at,
    )
    pinned = mem.half_life_hours is None
    return DecayedMemory(
        id=mem.id,
        content=mem.content,
        type=mem.memory_type.value,
        attribute_slot=mem.attribute_slot,
        decay_state=(
            mem.decay_state.value
            if hasattr(mem.decay_state, "value")
            else str(mem.decay_state)
        ),
        strength=decay.strength,
        half_life_hours=mem.half_life_hours,
        # compute_decay reports infinity for a pin; the wire says None.
        effective_half_life_hours=(
            None if pinned else decay.effective_half_life_hours
        ),
        reinforcement_count=mem.reinforcement_count,
        salience=mem.salience,
        created_at=mem.created_at,
        last_accessed_at=mem.last_accessed_at,
    )


class MemoryEngine:
    """The memory pipeline, callable in-process.

    Construct once and reuse; every method takes optional `db`, `tenant_id`
    and `llm` overrides for callers that already have them (the HTTP layer
    resolves all three per request from `Depends`).
    """

    def __init__(
        self,
        *,
        tenant_id: UUID | None = None,
        llm: LLMProtocol | None = None,
        session_factory=SessionLocal,
    ):
        self._tenant_id = tenant_id if tenant_id is not None else DEFAULT_TENANT_ID
        self._llm = llm
        self._session_factory = session_factory

    def _resolve_llm(self, llm: LLMProtocol | None) -> LLMProtocol:
        if llm is not None:
            return llm
        if self._llm is None:
            # Built on first use, not in __init__, so constructing an engine
            # for read-only work never needs LLM credentials configured.
            from memonative.llm import LLMClient, require_env_llm_keys

            require_env_llm_keys()
            self._llm = LLMClient()
        return self._llm

    def _resolve_tenant(self, tenant_id: UUID | None) -> UUID:
        return self._tenant_id if tenant_id is None else tenant_id

    @asynccontextmanager
    async def _session(self, db: AsyncSession | None, tenant_id: UUID, *, write: bool):
        """Yield an RLS-scoped session, committing only for write paths.

        A caller-supplied `db` is borrowed, never closed — it belongs to
        whoever opened it. With `db=None` the engine owns the whole
        lifecycle.

        `write` preserves the read/write asymmetry the routes had: the three
        write paths commit and roll back, the read paths do neither.
        `set_rls_tenant` is inside the try because it was inside the routes'
        try — a failure there must roll back like any other.
        """
        async with AsyncExitStack() as stack:
            if db is None:
                db = await stack.enter_async_context(self._session_factory())
            try:
                await set_rls_tenant(db, tenant_id)
                yield db
                if write:
                    await db.commit()
            except Exception:
                if write:
                    await db.rollback()
                raise

    async def process(
        self,
        *,
        user_id: UUID,
        message: str,
        context: Sequence[str] = (),
        top_k: int = 8,
        session_id: UUID | None = None,
        model: str | None = None,
        conversation_date: datetime | None = None,
        db: AsyncSession | None = None,
        tenant_id: UUID | None = None,
        llm: LLMProtocol | None = None,
    ) -> ProcessResponse:
        """The hot path: extract, write, retract, retrieve, reconcile, inject.

        `conversation_date` is what relative phrases ("next Tuesday", "a month
        ago") resolve against. It defaults to now, but a caller that captured
        the message earlier must pass the capture time — a note written offline
        and synced six hours later means something different against the two
        clocks. It also makes extraction reproducible for evals.
        """
        request = ProcessRequest(
            user_id=user_id,
            message=message,
            context=list(context),
            top_k=top_k,
            session_id=session_id,
            model=model,
        )
        tenant_id = self._resolve_tenant(tenant_id)
        llm = self._resolve_llm(llm)

        async with self._session(db, tenant_id, write=True) as db:
            llm.with_model(request.model)

            # 1. Extract intent + memories from message. The conversation date
            # lets the extraction LLM resolve relative phrases ("a month ago",
            # "last week") to absolute dates on the memory row.
            if conversation_date is None:
                conversation_date = datetime.now(timezone.utc)

            with time_stage(logger, "extract"):
                extracted = await extraction.extract(
                    llm, request.message, request.context,
                    conversation_date=conversation_date,
                )

            # 2. Write new memories AND embed the query in parallel.
            # write_memories owns the DB session; llm.embed only does HTTP.
            # No session conflict, and the two slow OpenAI calls overlap.
            with time_stage(logger, "write_and_embed_query"):
                (written_ids, contradicted_ids), query_embedding = await asyncio.gather(
                    write_path.write_memories(
                        db, tenant_id, request.user_id, extracted.memories_to_write, llm,
                        trigger_message=request.message,
                    ),
                    llm.embed(request.message),
                )

            # 2b. Apply explicit retractions ("lol I was kidding about X").
            # Excluded ids ensure a freshly-written replacement isn't archived
            # as a near-neighbour of the thing being retracted.
            with time_stage(logger, "retract"):
                retracted_ids = await write_path.retract_memories(
                    db,
                    tenant_id,
                    request.user_id,
                    extracted.retractions,
                    llm,
                    exclude_ids=written_ids,
                )

            # 3. Retrieve relevant memories
            with time_stage(logger, "retrieve"):
                retrieved = await retrieval.retrieve_memories(
                    db, tenant_id, request.user_id, request.message,
                    extracted.intent, query_embedding, request.top_k,
                    llm_client=llm,
                )

            # 4. Plan consensus + reconsolidation in parallel.
            # Both are LLM-only reads of the already-loaded retrieved list,
            # so they don't touch the AsyncSession concurrently.
            with time_stage(logger, "plan_consensus_recon"):
                consensuses, recon_raw = await asyncio.gather(
                    extract_consensus(
                        llm_client=llm,
                        retrieved_memories=retrieved,
                    ),
                    plan_reconsolidation(
                        llm_client=llm,
                        message=request.message,
                        retrieved_memories=retrieved,
                        new_facts=extracted.memories_to_write,
                    ),
                )

            # 5. Apply consensus corrections (mutates session — must be sequential).
            with time_stage(logger, "apply_consensus"):
                consensus_corrections = await apply_consensus_corrections(
                    db=db,
                    llm_client=llm,
                    consensuses=consensuses,
                    retrieved_memories=retrieved,
                    min_confidence=0.75,
                    min_evidence_count=2,
                )

                for correction in consensus_corrections:
                    await apply_interference(
                        db=db,
                        corrected_memory_id=correction["semantic_id"],
                        retrieved_memories=retrieved,
                    )

            # 6. Apply reconsolidation plan (mutates session).
            with time_stage(logger, "apply_recon"):
                recon_events = await apply_reconsolidation_plan(
                    db=db,
                    llm_client=llm,
                    raw_events=recon_raw,
                    retrieved_memories=retrieved,
                )

            # 7. Process goal updates
            with time_stage(logger, "goals"):
                goal_id = await goals.process_goal_update(
                    db, tenant_id, request.user_id, extracted.goal_update,
                )

            # 8. Format injection text (with revision history)
            with time_stage(logger, "inject"):
                injection = await retrieval.format_for_injection(retrieved, db=db)

            mapped_retrieved = []
            for sm in retrieved:
                mapped_retrieved.append(RetrievedMemory(
                    id=sm.memory.id,
                    content=sm.memory.content,
                    type=sm.memory.memory_type.value,
                    score=sm.score,
                    strength=sm.memory.strength,
                    trust_score=sm.memory.trust_score,
                ))

            # Persist an interaction row so requests are auditable. The
            # response field is left null — this engine emits memory
            # context for an LLM elsewhere, not the final assistant reply.
            db.add(Interaction(
                tenant_id=tenant_id,
                user_id=request.user_id,
                session_id=request.session_id,
                message=request.message,
                response=None,
                intent=extracted.intent or {},
                memories_written=list(written_ids),
                memories_retrieved=[sm.memory.id for sm in retrieved],
            ))

            return ProcessResponse(
                intent=IntentResponse(
                    primary=extracted.intent.get('primary', 'social'),
                    mode=extracted.intent.get('mode', 'conversational'),
                ),
                memories_written=written_ids,
                memories_retrieved=mapped_retrieved,
                injection_text=injection,
                goal_updated=goal_id,
                contradictions_flagged=contradicted_ids,
                retracted_ids=retracted_ids,
            )

    async def search(
        self,
        *,
        user_id: UUID,
        query: str,
        top_k: int = 8,
        memory_types: list[str] | None = None,
        token_budget: int | None = None,
        include_associations: bool = True,
        db: AsyncSession | None = None,
        tenant_id: UUID | None = None,
        llm: LLMProtocol | None = None,
    ) -> MemorySearchResponse:
        """Read-only memory search.

        Does not run extraction, does not write, does not reinforce. Returns
        full provenance so the caller can decide what to inject and how much
        to trust each item.
        """
        request = MemorySearchRequest(
            user_id=user_id,
            query=query,
            top_k=top_k,
            memory_types=memory_types,
            token_budget=token_budget,
            include_associations=include_associations,
        )
        tenant_id = self._resolve_tenant(tenant_id)
        llm = self._resolve_llm(llm)

        async with self._session(db, tenant_id, write=True) as db:
            with time_stage(logger, "search_embed"):
                embedding = await llm.embed(request.query)

            with time_stage(logger, "search_retrieve"):
                scored = await retrieval.retrieve_memories(
                    db=db,
                    tenant_id=tenant_id,
                    user_id=request.user_id,
                    message=request.query,
                    intent={"primary": "recall"},
                    embedding=embedding,
                    top_k=request.top_k,
                    reinforce=False,
                    type_override=request.memory_types,
                    include_associations=request.include_associations,
                )

            details = [_to_memory_detail(s.memory, s.score) for s in scored]

            truncated = False
            if request.token_budget is not None:
                kept: list[MemoryDetail] = []
                running = 0
                for d in details:
                    if running + d.estimated_tokens > request.token_budget:
                        truncated = True
                        break
                    kept.append(d)
                    running += d.estimated_tokens
                details = kept

            return MemorySearchResponse(
                query=request.query,
                memories=details,
                total_estimated_tokens=sum(d.estimated_tokens for d in details),
                truncated=truncated,
            )

    async def write(
        self,
        *,
        user_id: UUID,
        memories: Iterable[MemoryWriteItem | dict],
        trigger_message: str | None = None,
        model: str | None = None,
        db: AsyncSession | None = None,
        tenant_id: UUID | None = None,
        llm: LLMProtocol | None = None,
    ) -> MemoryWriteResponse:
        """Write structured memories directly, bypassing the extraction LLM.

        Use this when the caller already knows the fact. The slot lifecycle,
        decay profile, and revision chain still apply.
        """
        request = MemoryWriteRequest(
            user_id=user_id,
            memories=list(memories),
            trigger_message=trigger_message,
            model=model,
        )
        tenant_id = self._resolve_tenant(tenant_id)
        llm = self._resolve_llm(llm)

        async with self._session(db, tenant_id, write=True) as db:
            llm.with_model(request.model)
            written, contradictions = await write_path.write_memories(
                db=db,
                tenant_id=tenant_id,
                user_id=request.user_id,
                candidates_raw=[m.model_dump() for m in request.memories],
                llm_client=llm,
                trigger_message=request.trigger_message,
            )
            return MemoryWriteResponse(
                written_ids=written,
                contradictions_flagged=contradictions,
            )

    async def recall(
        self,
        memory_id: UUID,
        *,
        db: AsyncSession | None = None,
        tenant_id: UUID | None = None,
    ) -> MemoryDetail:
        """Fetch a single memory's full provenance by id, scoped to tenant.

        Cross-tenant lookups raise `NotFoundError` just like a missing id —
        never reveal that the id exists under another tenant.
        """
        tenant_id = self._resolve_tenant(tenant_id)
        async with self._session(db, tenant_id, write=False) as db:
            result = await db.execute(
                select(Memory).where(
                    Memory.id == memory_id,
                    Memory.tenant_id == tenant_id,
                )
            )
            mem = result.scalar_one_or_none()
            if mem is None:
                raise NotFoundError("Memory not found")
            return _to_memory_detail(mem, score=0.0)

    async def slots(
        self,
        user_id: UUID,
        *,
        db: AsyncSession | None = None,
        tenant_id: UUID | None = None,
    ) -> MemorySlotsResponse:
        """List a user's attribute slots with their current values.

        Use this to discover what facts the system knows about a user before
        drilling into a specific slot's revision history.
        """
        tenant_id = self._resolve_tenant(tenant_id)
        async with self._session(db, tenant_id, write=False) as db:
            result = await db.execute(
                select(
                    MemoryRevision.attribute_slot,
                    func.count().label("revision_count"),
                    func.max(MemoryRevision.valid_from).label("last_updated"),
                )
                .where(
                    MemoryRevision.tenant_id == tenant_id,
                    MemoryRevision.user_id == user_id,
                    MemoryRevision.attribute_slot.is_not(None),
                )
                .group_by(MemoryRevision.attribute_slot)
                .order_by(MemoryRevision.attribute_slot)
            )
            rows = result.all()

            slots = []
            for slot_name, rev_count, last_updated in rows:
                current = await db.execute(
                    select(MemoryRevision.content)
                    .where(
                        MemoryRevision.tenant_id == tenant_id,
                        MemoryRevision.user_id == user_id,
                        MemoryRevision.attribute_slot == slot_name,
                        MemoryRevision.valid_until.is_(None),
                    )
                    .limit(1)
                )
                current_value = current.scalar_one_or_none() or "(retracted)"
                slots.append(MemorySlotItem(
                    attribute_slot=slot_name,
                    current_value=current_value,
                    revision_count=rev_count,
                    last_updated=last_updated,
                ))

            return MemorySlotsResponse(user_id=user_id, slots=slots)

    async def audit(
        self,
        user_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
        db: AsyncSession | None = None,
        tenant_id: UUID | None = None,
    ) -> AuditTrailResponse:
        """Chronological audit trail of every fact change for a user.

        All revision entries across all attribute slots, newest first —
        "what changed about this user, when, and why" in a single call.
        """
        tenant_id = self._resolve_tenant(tenant_id)
        async with self._session(db, tenant_id, write=False) as db:
            base_filter = [
                MemoryRevision.tenant_id == tenant_id,
                MemoryRevision.user_id == user_id,
                MemoryRevision.attribute_slot.is_not(None),
            ]

            count_result = await db.execute(
                select(func.count()).select_from(MemoryRevision).where(*base_filter)
            )
            total = count_result.scalar()

            result = await db.execute(
                select(MemoryRevision)
                .where(*base_filter)
                .order_by(MemoryRevision.valid_from.desc())
                .limit(min(limit, 500))
                .offset(offset)
            )
            revisions = result.scalars().all()

            return AuditTrailResponse(
                user_id=user_id,
                total=total,
                entries=[
                    AuditEntry(
                        attribute_slot=r.attribute_slot,
                        sequence=r.sequence_number,
                        content=r.content,
                        revision_type=r.revision_type.value,
                        reason=r.reason,
                        trigger_message=r.trigger_message,
                        valid_from=r.valid_from,
                        valid_until=r.valid_until,
                        is_current=r.valid_until is None,
                    )
                    for r in revisions
                ],
            )

    async def facts(
        self,
        user_id: UUID,
        attribute_slot: str,
        *,
        db: AsyncSession | None = None,
        tenant_id: UUID | None = None,
    ) -> MemoryFactsResponse:
        """Current value + full revision history for one attribute slot.

        Strongly-typed twin of `history()`, which returns the same data as a
        plain dict for the older route.
        """
        tenant_id = self._resolve_tenant(tenant_id)
        async with self._session(db, tenant_id, write=False) as db:
            revisions = await self._revisions(db, tenant_id, user_id, attribute_slot)
            return MemoryFactsResponse(
                user_id=user_id,
                attribute_slot=attribute_slot,
                current_value=_current_value(revisions),
                revisions=[
                    MemoryRevisionItem(
                        sequence=r.sequence_number,
                        content=r.content,
                        revision_type=r.revision_type.value,
                        reason=r.reason,
                        valid_from=r.valid_from,
                        valid_until=r.valid_until,
                        trigger_message=r.trigger_message,
                    )
                    for r in revisions
                ],
            )

    async def history(
        self,
        user_id: UUID,
        attribute_slot: str,
        *,
        db: AsyncSession | None = None,
        tenant_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Revision history for one attribute slot, as a plain dict.

        Returns a dict rather than a model because the route it backs always
        has — timestamps are pre-formatted strings. Prefer `facts()`.
        """
        tenant_id = self._resolve_tenant(tenant_id)
        async with self._session(db, tenant_id, write=False) as db:
            revisions = await self._revisions(db, tenant_id, user_id, attribute_slot)
            return {
                "attribute_slot": attribute_slot,
                "current_value": _current_value(revisions),
                "revisions": [
                    {
                        "sequence": r.sequence_number,
                        "content": r.content,
                        "revision_type": r.revision_type.value,
                        "reason": r.reason,
                        "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                        "valid_until": r.valid_until.isoformat() if r.valid_until else None,
                        "trigger_message": r.trigger_message,
                    }
                    for r in revisions
                ],
            }

    async def consolidate(
        self,
        user_id: UUID,
        *,
        db: AsyncSession | None = None,
        tenant_id: UUID | None = None,
        llm: LLMProtocol | None = None,
    ) -> int:
        """Merge episodic clusters into semantic memories. Returns cluster count.

        Deliberately does not commit — `consolidate_cluster` mutates the
        session and the route has always left the commit to whoever owns it.
        """
        tenant_id = self._resolve_tenant(tenant_id)
        llm = self._resolve_llm(llm)
        async with self._session(db, tenant_id, write=False) as db:
            clusters = await consolidation.find_episodic_clusters(db, tenant_id, user_id)
            for cluster in clusters:
                await consolidation.consolidate_cluster(db, cluster, llm)
            return len(clusters)

    async def set_retention(
        self,
        memory_id: UUID,
        half_life_hours: float | None | _Default = _DEFAULT,
        *,
        db: AsyncSession | None = None,
        tenant_id: UUID | None = None,
    ) -> DecayedMemory:
        """Pin a memory (`None`), set a half-life (a float), or restore the
        engine's default for its type and source (omit the argument).

        The third case is what unpinning normally wants. Pinning overwrites
        the old half-life and nothing keeps a copy, so "put it back" has to
        mean recomputing from the decay profile rather than restoring a
        remembered number — and the profile is engine knowledge that callers
        should not be hard-coding.

        Each direction carries a second change that assigning
        `half_life_hours` alone would miss.

        **Pinning also restores full strength and the active state.** A pin
        that froze a memory at `fading` would leave it pinned below
        retrieval's cutoff — never forgotten, never found either. "Never
        forget this" has to mean more than "never forget it *further*".

        **Unpinning also resets `last_accessed_at`.** Strength is recomputed
        from that timestamp on every sweep rather than decremented, so a
        memory pinned a year ago would otherwise be measured against a year
        of elapsed time the moment its half-life came back, and land in
        `archived` within the hour.

        Validation of the number belongs to the caller: the HTTP layer
        already parses and bounds it, the same split that leaves UUID
        parsing in the route.
        """
        tenant_id = self._resolve_tenant(tenant_id)
        async with self._session(db, tenant_id, write=True) as db:
            memory = (
                await db.execute(
                    select(Memory).where(
                        Memory.id == memory_id,
                        # Explicit rather than leaning on RLS. This method is
                        # reachable from a user-supplied id, so the tenant
                        # boundary is stated in the query that crosses it.
                        Memory.tenant_id == tenant_id,
                    )
                )
            ).scalar_one_or_none()
            if memory is None:
                raise NotFoundError(f"Memory {memory_id} not found")

            if isinstance(half_life_hours, _Default):
                source = (
                    memory.source.value
                    if hasattr(memory.source, "value")
                    else str(memory.source)
                )
                half_life_hours = get_decay_profile(
                    memory.memory_type.value, source
                ).half_life_hours

            memory.half_life_hours = half_life_hours
            if half_life_hours is None:
                memory.strength = 1.0
                memory.decay_state = DecayState.active
            else:
                memory.last_accessed_at = datetime.now(timezone.utc)

            await db.flush()
            return _to_decayed_memory(memory)

    async def list_by_decay_state(
        self,
        user_id: UUID,
        *,
        states: Sequence[str] = ("active", "fading", "dormant", "archived"),
        limit: int = 50,
        offset: int = 0,
        db: AsyncSession | None = None,
        tenant_id: UUID | None = None,
    ) -> list[DecayedMemory]:
        """Memories in the given lifecycle states, weakest first.

        This is the only way to reach `dormant` and `archived` memories.
        Retrieval filters to `('active', 'fading')`, so everything past
        `fading` is invisible to `search()` by design — listing a tier is a
        different question from ranking it, and needs no embedding call.

        Weakest first: tiers are requested one at a time, and the
        interesting end of a tier is the edge it is about to fall over.

        `decay_state` is the stored column, so a tier is accurate as of the
        last hourly sweep. `strength` on each result is recomputed live, so
        the two can disagree near a threshold. That is the same skew
        retrieval already runs with, and the live number is the honest one.
        """
        tenant_id = self._resolve_tenant(tenant_id)
        async with self._session(db, tenant_id, write=False) as db:
            result = await db.execute(
                select(Memory)
                .where(
                    Memory.tenant_id == tenant_id,
                    Memory.user_id == user_id,
                    Memory.decay_state.in_(list(states)),
                )
                .order_by(Memory.strength.asc(), Memory.id.asc())
                .limit(limit)
                .offset(offset)
            )
            return [_to_decayed_memory(m) for m in result.scalars().all()]

    @staticmethod
    async def _revisions(
        db: AsyncSession, tenant_id: UUID, user_id: UUID, attribute_slot: str
    ):
        result = await db.execute(
            select(MemoryRevision).where(
                MemoryRevision.tenant_id == tenant_id,
                MemoryRevision.user_id == user_id,
                MemoryRevision.attribute_slot == attribute_slot,
            ).order_by(MemoryRevision.sequence_number.asc())
        )
        revisions = result.scalars().all()
        if not revisions:
            raise NotFoundError(f"No history for slot '{attribute_slot}'")
        return revisions


def _current_value(revisions) -> str:
    """The open revision, or the newest one if the slot has been retracted."""
    return next(
        (r.content for r in revisions if r.valid_until is None),
        revisions[-1].content,
    )
