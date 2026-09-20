"""FTS-path integration test.

Seeds a rare token ("ORD-7731-X") in a memory's content and verifies
that retrieve_memories surfaces it when queried by that token alone.
Pure-vector retrieval tends to miss tokens like these; the FTS
candidate query + RRF fusion is what rescues them.

Marked ``dbintegration`` so DB-less CI can skip with ``-m 'not dbintegration'``.
Auto-skips if Postgres isn't reachable.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

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


pytestmark = pytest.mark.dbintegration


RARE_TOKEN = "ORD-7731-X"


def _embedding(seed: float, dim: int = 1536) -> list[float]:
    # Deterministic dummy embedding — we want the vector channel to NOT
    # surface the target memory for this test, so just use different
    # seeds for the noise memories vs the target.
    return [seed] * dim


async def _seed_and_query() -> list[str]:
    engine = create_async_engine(settings.DATABASE_URL)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"Postgres unavailable for FTS test: {exc!r}")

    Session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    tenant_id = uuid4()
    user_id = uuid4()
    try:
        async with Session() as db:
            db.add(Tenant(id=tenant_id, name=f"fts-{tenant_id}"))
            await db.flush()
            # Noise memories with a different embedding signature.
            for i in range(10):
                db.add(Memory(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    memory_type="semantic",
                    content=f"unrelated fact number {i}",
                    embedding=_embedding(0.001),
                    source="user_stated",
                    trust_score=0.7, salience=0.5, half_life_hours=720.0,
                ))
            # Target memory containing the rare token. Embedding is
            # intentionally near-orthogonal to the query embedding, so
            # only the FTS path can surface it.
            db.add(Memory(
                tenant_id=tenant_id,
                user_id=user_id,
                memory_type="semantic",
                content=f"User's outstanding order is {RARE_TOKEN}",
                embedding=_embedding(0.999),
                source="user_stated",
                trust_score=0.7, salience=0.5, half_life_hours=720.0,
            ))
            await db.commit()

        async with Session() as db:
            results = await retrieve_memories(
                db=db,
                tenant_id=tenant_id,
                user_id=user_id,
                message=RARE_TOKEN,
                intent={"primary": "recall"},
                embedding=_embedding(0.001),
                top_k=5,
                reinforce=False,
                include_associations=False,
            )
            return [sm.memory.content for sm in results]
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


def test_fts_surfaces_rare_token():
    contents = asyncio.run(_seed_and_query())
    assert any(RARE_TOKEN in c for c in contents), (
        f"expected FTS to surface memory containing {RARE_TOKEN}; got {contents!r}"
    )
