import asyncio
import logging
import time

import redis
from celery import Celery
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from memonative.config import settings
from memonative.db.models import Memory

from memonative.engine.consolidation import find_episodic_clusters, consolidate_cluster
from memonative.engine.edges import check_contradictions, create_causal_edges
from memonative.api.llm import LLMClient
from memonative.auth.credentials import resolve_credentials
from memonative.db.database import set_rls_tenant

logger = logging.getLogger(__name__)

_redis_pool = redis.ConnectionPool.from_url(settings.REDIS_URL)


def _get_redis() -> redis.Redis:
    return redis.Redis(connection_pool=_redis_pool)


def _acquire_beat_lock(task_name: str, timeout: int) -> bool:
    """Acquire a Redis lock to guarantee single-execution of beat tasks.

    Even with a single beat instance, this protects against the scenario
    where beat fires a task while a previous invocation is still running
    (e.g. consolidation taking longer than 4 hours). Returns True if the
    lock was acquired, False if another instance holds it.
    """
    try:
        r = _get_redis()
        return bool(r.set(f"beat_lock:{task_name}", "1", nx=True, ex=timeout))
    except Exception:
        logger.warning(
            "Redis unreachable for beat lock %s — proceeding without exclusivity",
            task_name,
        )
        return True


def _release_beat_lock(task_name: str) -> None:
    try:
        r = _get_redis()
        r.delete(f"beat_lock:{task_name}")
    except Exception:
        pass


async def _set_worker_bypass(session: AsyncSession) -> None:
    """Set the RLS bypass sentinel for cross-tenant worker operations."""
    await session.execute(
        text("SELECT set_config('app.current_tenant_id', '__worker__', true)")
    )

celery_app = Celery(
    "memonative_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


def _run_async(coro_factory):
    """Run an async task body inside a Celery task with a fresh event loop
    AND a fresh async engine.

    Celery tasks run synchronously and `asyncio.run` creates a new loop
    each time. asyncpg connections are bound to the loop they were
    created on, so reusing a module-level engine across Celery
    invocations triggers "Future attached to a different loop". We
    create + dispose the engine inside the loop to keep them aligned.

    `coro_factory` receives the per-task `Session` factory.
    """
    async def _runner():
        engine = create_async_engine(settings.DATABASE_URL)
        Session = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        try:
            return await coro_factory(Session)
        finally:
            await engine.dispose()

    return asyncio.run(_runner())


DECAY_BATCH_SIZE = 5000

DECAY_SQL = """
WITH computed AS (
    SELECT
        id,
        CASE
            WHEN (half_life_hours * (ln(reinforcement_count + 1) / ln(2)) * (1 + salience)) <= 0
            THEN 0.0
            ELSE exp(
                -0.693 * (EXTRACT(EPOCH FROM (now() - last_accessed_at)) / 3600.0)
                / (half_life_hours * (ln(reinforcement_count + 1) / ln(2)) * (1 + salience))
            )
        END AS new_strength
    FROM memories
    WHERE decay_state IN ('active', 'fading', 'dormant')
      -- Pinned. This is the clause that makes a pin hold; without it the
      -- next hourly sweep would decay the row like any other. Safe against
      -- the cursor pagination because LIMIT applies after the filter, so a
      -- window of entirely-pinned rows is scanned past rather than
      -- returning empty and ending the sweep early.
      AND half_life_hours IS NOT NULL
      AND id > :cursor_id
    ORDER BY id
    LIMIT :batch_size
)
UPDATE memories m
SET
    strength = GREATEST(0.0, LEAST(1.0, c.new_strength)),
    -- The cast is required, not decoration. A multi-branch CASE of string
    -- literals resolves to `text` on its own, before anything looks at the
    -- column, so the assignment fails outright with "column decay_state is
    -- of type decaystate but expression is of type text" and takes the
    -- whole sweep with it.
    decay_state = (CASE
        WHEN c.new_strength > 0.5 THEN 'active'
        WHEN c.new_strength > 0.2 THEN 'fading'
        WHEN c.new_strength > 0.05 THEN 'dormant'
        ELSE 'archived'
    END)::decaystate
FROM computed c
WHERE m.id = c.id
RETURNING m.id
"""


@celery_app.task(name="memonative.decay_job")
def apply_decay():
    """Recalculate Ebbinghaus decay for all non-archived memories.

    Pushes the full Ebbinghaus formula into a set-based SQL UPDATE,
    paginated by id in batches of DECAY_BATCH_SIZE. Never loads rows
    into Python — scales to millions of memories.

    Protected by a Redis lock (2x beat interval) so overlapping
    invocations don't double-update.
    """
    if not _acquire_beat_lock("decay", timeout=7200):
        return "skipped — previous decay run still holds the lock"

    logger.info("decay_job.start")
    t0 = time.monotonic()

    async def _run(Session):
        total = 0
        batches = 0
        cursor_id = UUID("00000000-0000-0000-0000-000000000000")

        async with Session() as db:
            await _set_worker_bypass(db)
            while True:
                result = await db.execute(
                    text(DECAY_SQL),
                    {"cursor_id": str(cursor_id), "batch_size": DECAY_BATCH_SIZE},
                )
                rows = result.fetchall()
                if not rows:
                    break
                total += len(rows)
                batches += 1
                # max(), not rows[-1]: the ORDER BY belongs to the CTE, and
                # RETURNING is not obliged to preserve it. Taking whichever
                # id happened to come back last can advance the cursor past
                # rows this batch never updated, and they are skipped for
                # good — the next sweep starts from zero and hits the same
                # reordering somewhere else.
                cursor_id = max(r[0] for r in rows)
                await db.commit()

        elapsed = time.monotonic() - t0
        logger.info(
            "decay_job.complete total=%d batches=%d duration_s=%.1f",
            total, batches, elapsed,
        )
        return f"Decayed {total} memories in {batches} batches."

    try:
        return _run_async(_run)
    finally:
        _release_beat_lock("decay")


@celery_app.task(name="memonative.consolidation_job")
def apply_consolidation():
    """Cluster episodic memories into generalized semantic knowledge.

    Chunked per-tenant: each tenant gets its own session and commit
    boundary so one tenant's failure doesn't block others, and memory
    usage stays bounded regardless of total tenant count.

    Protected by a Redis lock (2x beat interval = 28800s) so
    overlapping invocations don't double-consolidate.
    """
    if not _acquire_beat_lock("consolidation", timeout=28800):
        return "skipped — previous consolidation run still holds the lock"

    logger.info("consolidation_job.start")
    t0 = time.monotonic()

    async def _run(Session):
        async with Session() as db:
            await _set_worker_bypass(db)
            result = await db.execute(
                select(Memory.tenant_id).distinct()
            )
            tenant_ids = [row[0] for row in result.all()]

        total_clusters = 0
        total_pairs = 0

        for tid in tenant_ids:
            try:
                async with Session() as db:
                    await set_rls_tenant(db, tid)
                    creds = await resolve_credentials(db, tid)
                    if not creds.engine_api_key:
                        logger.debug("skipping tenant %s — no engine key configured", tid)
                        continue
                    llm = LLMClient.from_credentials(creds)

                    result = await db.execute(
                        select(Memory.user_id).where(
                            Memory.tenant_id == tid
                        ).distinct()
                    )
                    user_ids = [row[0] for row in result.all()]

                    for user_id in user_ids:
                        clusters = await find_episodic_clusters(db, tid, user_id)
                        for cluster in clusters:
                            try:
                                await consolidate_cluster(db, cluster, llm)
                                total_clusters += 1
                            except Exception as exc:
                                logger.error(
                                    "consolidation failed for cluster in tenant=%s user=%s: %s",
                                    tid, user_id, exc, exc_info=True,
                                )
                                await db.rollback()
                                await set_rls_tenant(db, tid)
                                continue
                        total_pairs += 1

                    await db.commit()
            except Exception as exc:
                logger.error(
                    "consolidation failed for tenant %s: %s", tid, exc,
                    exc_info=True,
                )
                continue

        elapsed = time.monotonic() - t0
        logger.info(
            "consolidation_job.complete clusters=%d pairs=%d tenants=%d duration_s=%.1f",
            total_clusters, total_pairs, len(tenant_ids), elapsed,
        )
        return (
            f"Consolidated {total_clusters} clusters across "
            f"{total_pairs} (tenant, user) pairs in {len(tenant_ids)} tenants."
        )

    try:
        return _run_async(_run)
    finally:
        _release_beat_lock("consolidation")


@celery_app.task(
    name="memonative.check_contradictions",
    acks_late=True,
    reject_on_worker_lost=True,
)
def check_contradictions_task(tenant_id: str, user_id: str, memory_id: str):
    """Run contradiction detection for a newly-written semantic memory.

    Deferred off the write path because the LLM call adds 500–2000ms
    of latency to user-facing requests. Resolves per-tenant LLM
    credentials so BYOK keys are used for the contradiction LLM call.
    """
    async def _run(Session):
        async with Session() as db:
            tid = UUID(tenant_id)
            await set_rls_tenant(db, tid)
            creds = await resolve_credentials(db, tid)
            if not creds.engine_api_key:
                return "skipped — tenant has no engine key configured"
            llm = LLMClient.from_credentials(creds)
            mem = await db.get(Memory, UUID(memory_id))
            if mem is None:
                return "memory gone — nothing to check"
            flagged = await check_contradictions(
                db=db,
                tenant_id=tid,
                user_id=UUID(user_id),
                new_memory=mem,
                llm_client=llm,
            )
            causal = await create_causal_edges(
                db=db,
                tenant_id=tid,
                user_id=UUID(user_id),
                new_memory=mem,
                llm_client=llm,
            )
            await db.commit()
            return (
                f"flagged {len(flagged)} contradictions, "
                f"created {len(causal)} causal edges for {memory_id}"
            )

    return _run_async(_run)


_beat_schedule = {
    'calculate-decay-every-hour': {
        'task': 'memonative.decay_job',
        'schedule': 3600.0,
    },
    'consolidate-memories-every-4-hours': {
        'task': 'memonative.consolidation_job',
        'schedule': 14400.0,
    },
}

celery_app.conf.beat_schedule = _beat_schedule
