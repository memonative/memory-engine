"""Pinning: `set_retention(id, None)` and the clauses that make it hold.

A pin is only as good as the decay job's willingness to skip the row, so the
load-bearing test here is the one that runs the real `DECAY_SQL` and checks a
pinned memory came out untouched — next to an unpinned control that did not.
Without the control the test would pass just as happily if the UPDATE matched
nothing at all.

Marked ``dbintegration`` so DB-less CI skips it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from memonative.config import settings
from memonative.db.models import Memory, Tenant
from memonative.facade import MemoryEngine
from memonative.worker import DECAY_SQL

pytestmark = [pytest.mark.dbintegration, pytest.mark.asyncio]

ANCIENT = timedelta(days=30)


def _embedding(dim: int = 1536) -> list[float]:
    return [0.01] * dim


@pytest_asyncio.fixture
async def world():
    """A tenant with one decaying memory and one that will be pinned."""
    engine = create_async_engine(settings.DATABASE_URL)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"Postgres unavailable: {exc!r}")

    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    tenant_id, user_id = uuid4(), uuid4()
    stale = datetime.now(timezone.utc) - ANCIENT
    ids = {}

    async with Session() as db:
        db.add(Tenant(id=tenant_id, name=f"retention-{tenant_id}"))
        await db.flush()
        for name in ("control", "subject"):
            mem = Memory(
                tenant_id=tenant_id, user_id=user_id, memory_type="semantic",
                content=f"{name} fact", embedding=_embedding(),
                source="user_stated", trust_score=0.7, salience=0.0,
                half_life_hours=72.0, strength=1.0, decay_state="active",
                reinforcement_count=1, last_accessed_at=stale,
            )
            db.add(mem)
            await db.flush()
            ids[name] = mem.id
        await db.commit()

    yield MemoryEngine(tenant_id=tenant_id, session_factory=Session), Session, user_id, ids

    async with Session() as db:
        await db.execute(
            text("DELETE FROM tenants WHERE id = :t"), {"t": str(tenant_id)}
        )
        await db.commit()
    await engine.dispose()


async def _run_decay(Session) -> None:
    """The real sweep, paginated exactly as `apply_decay` paginates it.

    Not a single batch: the batch is a window over the whole table ordered
    by id, and these memories have random uuids, so one execution would
    usually miss them and report a pass for the wrong reason.
    """
    cursor = "00000000-0000-0000-0000-000000000000"
    async with Session() as db:
        while True:
            rows = (
                await db.execute(
                    text(DECAY_SQL), {"cursor_id": cursor, "batch_size": 5000}
                )
            ).fetchall()
            if not rows:
                break
            cursor = str(max(r[0] for r in rows))
            await db.commit()


async def test_the_decay_job_skips_a_pinned_memory(world):
    """The clause the whole feature rests on.

    Both memories were last accessed 30 days ago with a 72h half-life, so
    both are overdue to be archived. The control proves the sweep ran.
    """
    engine, Session, _, ids = world
    await engine.set_retention(ids["subject"], None)

    await _run_decay(Session)

    async with Session() as db:
        control = await db.get(Memory, ids["control"])
        subject = await db.get(Memory, ids["subject"])

    assert control.decay_state.value == "archived", "sweep did not run"
    assert subject.decay_state.value == "active"
    assert subject.strength == 1.0


async def test_pinning_rescues_a_memory_that_was_already_fading(world):
    """A pin that froze the memory below retrieval's cutoff would leave it
    never forgotten and never found — which is not what the user asked for.
    """
    engine, Session, _, ids = world
    async with Session() as db:
        mem = await db.get(Memory, ids["subject"])
        mem.strength, mem.decay_state = 0.21, "fading"
        await db.commit()

    result = await engine.set_retention(ids["subject"], None)

    assert result.decay_state == "active"
    assert result.strength == 1.0
    assert result.half_life_hours is None
    assert result.effective_half_life_hours is None


async def test_unpinning_resets_the_access_clock(world):
    """Strength is recomputed from `last_accessed_at`, never decremented, so
    restoring a half-life without moving the clock would measure the memory
    against every hour it spent pinned and archive it on the next sweep.
    """
    engine, Session, _, ids = world
    await engine.set_retention(ids["subject"], None)

    await engine.set_retention(ids["subject"], 72.0)

    async with Session() as db:
        subject = await db.get(Memory, ids["subject"])
    assert datetime.now(timezone.utc) - subject.last_accessed_at < timedelta(minutes=5)

    # And it genuinely survives the sweep that archives the control.
    await _run_decay(Session)
    async with Session() as db:
        assert (await db.get(Memory, ids["control"])).decay_state.value == "archived"
        assert (await db.get(Memory, ids["subject"])).decay_state.value == "active"


async def test_listing_reaches_archived_memories_that_search_cannot(world):
    """Retrieval filters to ('active', 'fading'). If listing did the same,
    the Attic would render empty forever.
    """
    engine, Session, user_id, ids = world
    await _run_decay(Session)

    archived = await engine.list_by_decay_state(user_id, states=["archived"])

    assert {m.id for m in archived} == set(ids.values())
    assert all(m.decay_state == "archived" for m in archived)


async def test_listing_reports_the_effective_half_life_not_the_stored_one(world):
    """Reinforcement and salience extend the real lifetime well past the
    configured number, and the longer one is what a retention chip should say.
    """
    engine, Session, user_id, ids = world
    async with Session() as db:
        mem = await db.get(Memory, ids["subject"])
        mem.reinforcement_count, mem.salience = 7, 0.5
        await db.commit()

    listed = {m.id: m for m in await engine.list_by_decay_state(user_id)}

    subject = listed[ids["subject"]]
    assert subject.half_life_hours == 72.0
    assert subject.effective_half_life_hours == pytest.approx(72.0 * 3 * 1.5)


async def test_setting_retention_on_another_tenants_memory_is_not_found(world):
    """The tenant filter is stated in the query rather than left to RLS,
    because the id crossing this boundary comes from the caller.
    """
    from memonative.errors import NotFoundError

    _, Session, _, ids = world
    stranger = MemoryEngine(tenant_id=uuid4(), session_factory=Session)

    with pytest.raises(NotFoundError):
        await stranger.set_retention(ids["subject"], None)
