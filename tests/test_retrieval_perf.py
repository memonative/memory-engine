"""Retrieval performance test.

Seeds 500 memories for one (tenant, user) pair and asserts the rewritten
retrieve_memories (HNSW + in-DB candidate generation) returns in well
under 500ms.

Marked ``perf`` so CI can skip it with ``-m 'not perf'``. Also auto-skips
when the local Postgres isn't reachable, so contributors without the
docker-compose stack running don't see a hard failure.
"""
from __future__ import annotations

import asyncio
import random
import time
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from memonative.config import settings
from memonative.db.models import Memory, Tenant
from memonative.engine.retrieval import retrieve_memories


pytestmark = pytest.mark.perf


def _random_unit_vector(dim: int = 1536) -> list[float]:
    # Uniform-ish embedding; actual distribution doesn't matter for the
    # timing test — pgvector only cares that it's a 1536-d float vector.
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]


async def _engine_or_skip():
    engine = create_async_engine(settings.DATABASE_URL)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"Postgres unavailable for perf test: {exc!r}")
    return engine


async def _seed_and_time() -> float:
    engine = await _engine_or_skip()
    Session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    tenant_id = uuid4()
    user_id = uuid4()
    try:
        async with Session() as db:
            db.add(Tenant(id=tenant_id, name=f"perf-{tenant_id}"))
            await db.flush()
            for _ in range(500):
                db.add(Memory(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    memory_type="semantic",
                    content="seeded memory for perf test",
                    embedding=_random_unit_vector(),
                    source="user_stated",
                    trust_score=0.7,
                    salience=0.5,
                    half_life_hours=720.0,
                ))
            await db.commit()

        query_embedding = _random_unit_vector()
        async with Session() as db:
            t0 = time.perf_counter()
            await retrieve_memories(
                db=db,
                tenant_id=tenant_id,
                user_id=user_id,
                message="does not matter — we only embed the query upstream",
                intent={"primary": "recall"},
                embedding=query_embedding,
                top_k=8,
                reinforce=False,
                include_associations=False,
            )
            elapsed = time.perf_counter() - t0
            return elapsed
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM memories WHERE tenant_id = :t"),
                {"t": str(tenant_id)},
            )
            await conn.execute(
                text("DELETE FROM tenants WHERE id = :t"),
                {"t": str(tenant_id)},
            )
        await engine.dispose()


def test_retrieval_fast_with_500_memories():
    elapsed = asyncio.run(_seed_and_time())
    # 500ms is the documented budget; comfortably under is the bar.
    assert elapsed < 0.5, f"retrieve_memories took {elapsed:.3f}s over 500 seeded memories"
