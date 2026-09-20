"""Task 5: retrieve_memories emits a single UPDATE for reinforcement.

Seeds N memories, calls retrieve_memories, and counts UPDATE statements
via a SQLAlchemy ``before_cursor_execute`` listener. The contract is:
one UPDATE covers every retrieved memory, rather than N round-trips.

Marked ``dbintegration`` so DB-less CI skips it.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from memonative.config import settings
from memonative.db.models import Memory, Tenant
from memonative.engine.retrieval import retrieve_memories


pytestmark = pytest.mark.dbintegration


def _embedding(dim: int = 1536) -> list[float]:
    return [0.01] * dim


async def _run() -> int:
    engine = create_async_engine(settings.DATABASE_URL)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"Postgres unavailable for bulk-reinforce test: {exc!r}")

    update_count = {"n": 0}

    # event.listen attaches to the sync engine under the AsyncEngine.
    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _count_updates(conn, cursor, statement, params, context, executemany):
        if statement.lstrip().upper().startswith("UPDATE MEMORIES"):
            update_count["n"] += 1

    Session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    tenant_id = uuid4()
    user_id = uuid4()
    try:
        async with Session() as db:
            db.add(Tenant(id=tenant_id, name=f"bulk-{tenant_id}"))
            await db.flush()
            for i in range(12):
                db.add(Memory(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    memory_type="semantic",
                    content=f"seeded fact {i}",
                    embedding=_embedding(),
                    source="user_stated",
                    trust_score=0.7, salience=0.5, half_life_hours=720.0,
                ))
            await db.commit()

        # Reset counter — only count UPDATEs from the retrieval call.
        update_count["n"] = 0

        async with Session() as db:
            results = await retrieve_memories(
                db=db,
                tenant_id=tenant_id,
                user_id=user_id,
                message="seeded fact",
                intent={"primary": "recall"},
                embedding=_embedding(),
                top_k=8,
                reinforce=True,
                include_associations=False,
            )
            await db.commit()
        assert results, "expected at least one retrieval hit for the test to be meaningful"

        return update_count["n"]
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


def test_retrieve_memories_issues_one_update_for_reinforcement():
    n_updates = asyncio.run(_run())
    assert n_updates == 1, (
        f"expected exactly one UPDATE memories ... for bulk reinforcement, "
        f"got {n_updates}"
    )
