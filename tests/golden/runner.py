"""Drives the corpus through the real pipeline and normalizes the result.

Record and replay share this module on purpose: if the two passes ran
different code, a golden match would prove nothing. The record pass wraps
a live LLM, the replay pass wraps a cassette, and everything else — the
call sequence, the session handling, the response assembly — is identical.

Route handlers are invoked as plain async functions with their
dependencies passed explicitly. FastAPI's `Depends` defaults only bind
when the app resolves them, so a direct call runs the same orchestration
without an HTTP layer. After S1.5 the handlers delegate to the facade,
this runner is unchanged, and any diff is a bug in the move.
"""

from __future__ import annotations

import contextlib
import json
import random
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import UUID as PGUUID

from memonative.auth.deps import DEFAULT_TENANT_ID
from memonative.db.database import SessionLocal, set_rls_tenant

from tests.golden.corpus import CAPTURES, GOLDEN_USER_ID, SEARCH_QUERIES

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
# Bare dates are masked too, not just full timestamps: `injection_text`
# carries a `[YYYY-MM-DD]` prefix off the row's creation date, so goldens
# recorded today would fail tomorrow. Masking costs us nothing here — a
# refactor cannot change date resolution, and date correctness is the
# notes eval's job (spec §11), not the characterization harness's.
_TS_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)


def normalize(obj: Any, ids: dict[str, str] | None = None) -> Any:
    """Replace run-varying ids/timestamps with stable placeholders.

    Memory ids are freshly generated on every seed, so a literal byte
    comparison could never pass. Mapping them to `<id:1>`, `<id:2>` … in
    order of first appearance keeps the *relationships* checkable — the
    same id in two places still normalizes to the same token — while
    making the document stable across runs.
    """
    if ids is None:
        ids = {}

    def _str(s: str) -> str:
        def sub(m: re.Match) -> str:
            raw = m.group(0).lower()
            if raw not in ids:
                ids[raw] = f"<id:{len(ids) + 1}>"
            return ids[raw]

        return _TS_RE.sub("<ts>", _UUID_RE.sub(sub, s))

    if isinstance(obj, str):
        return _str(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            v = normalize(v, ids)
            # `write_path` builds these with `list(set(...))`, so their order
            # follows PYTHONHASHSEED — stable inside one process, different in
            # the next. Sorting is safe because tag order carries no meaning;
            # every other list here (retrieved memories, above all) is ranked,
            # so ordering is checked rather than normalized away.
            if k == "context_tags" and isinstance(v, list):
                v = sorted(v)
            out[k] = v
        return out
    if isinstance(obj, list):
        return [normalize(v, ids) for v in obj]
    if isinstance(obj, float):
        # Retrieval scores drift in the 6th decimal between identical replays
        # — pgvector sums float4 components in whatever order the plan gives
        # it. 4dp absorbs that and is still orders of magnitude tighter than
        # any behaviour change a refactor could cause. Note this masks values
        # only: if two memories ever score within 1e-4 of each other their
        # *ranking* could still flip, and no rounding would hide that.
        return round(obj, 4)
    return obj


def _jsonable(model) -> dict:
    return json.loads(model.model_dump_json())


_RNG = random.Random()
_installed = False


def _fake_uuid4(_ctx=None) -> uuid.UUID:
    return uuid.UUID(int=_RNG.getrandbits(128), version=4)


def install_uuid_defaults() -> None:
    """Point every UUID primary-key default at `_fake_uuid4`, once per process.

    Installing per-call and restoring afterwards looks tidier but is wrong:
    SQLAlchemy caches compiled INSERTs, and the cache captures whichever
    callable was installed the first time a statement compiled. A second
    `run_corpus` in the same process would re-patch, be ignored, and keep
    drawing from the *first* run's generator — so its ids continued the
    stream instead of restarting it, the cassette's `memory_id` references
    stopped resolving, and reconsolidation silently no-opped. One permanent
    callable plus a reseedable generator sidesteps the cache entirely.

    For the same reason this has to run before *any* insert compiles, which
    is why conftest calls it at session start: a test that wrote a memory
    first would pin the original `uuid4` into the cache and this would never
    take effect. Outside `deterministic_uuids` the generator is seeded from
    OS entropy, so every other test still gets ordinary random ids.
    """
    global _installed
    if _installed:
        return
    from memonative.db.models import Base

    for table in Base.metadata.tables.values():
        for col in table.columns:
            default = col.default
            if default is None or not default.is_callable:
                continue
            if not isinstance(col.type, PGUUID):
                continue  # created_at & friends are callable defaults too
            default.arg = _fake_uuid4
    _installed = True


@contextlib.contextmanager
def deterministic_uuids(seed: int = 20260828):
    """Make generated row ids reproducible for the duration of a run.

    Recorded LLM responses quote memory ids back at us — reconsolidation
    answers with `{"memory_id": "<uuid>", "outcome": "reinforced"}`. If
    replay regenerated fresh ids, every one of those references would fail
    to resolve, silently skipping the reinforcement and producing a
    *different* pipeline outcome than the one recorded. Pinning the id
    stream makes the cassette's references valid on replay.

    Rebinding `models.uuid4` does nothing: `mapped_column(default=uuid4)`
    captured the function object when the class body ran, so the column's
    `ColumnDefault` still holds the original. The callable has to be
    swapped on the default itself. Every persisted id in the engine comes
    from one of these column defaults — nothing calls `uuid4()` inline —
    so this covers the whole id stream.
    """
    install_uuid_defaults()
    _RNG.seed(seed)
    try:
        yield
    finally:
        # Reseed from OS entropy rather than restoring the original callable:
        # anything running after this block gets random ids again, but the
        # callable itself must stay installed (see `_install_uuid_defaults`).
        _RNG.seed()


async def purge_user(db, tenant_id: UUID, user_id: UUID) -> None:
    """Reset the golden user's slate. Edges/revisions/logs cascade off memories."""
    await set_rls_tenant(db, tenant_id)
    for stmt in (
        "DELETE FROM memories WHERE tenant_id = :t AND user_id = :u",
        "DELETE FROM goals WHERE tenant_id = :t AND user_id = :u",
        "DELETE FROM interactions WHERE tenant_id = :t AND user_id = :u",
        "DELETE FROM usage_events WHERE tenant_id = :t AND user_id = :u",
    ):
        await db.execute(text(stmt), {"t": str(tenant_id), "u": str(user_id)})
    await db.commit()


# The date the fixtures were recorded. The extraction prompt embeds the
# weekday of `conversation_date`, and cassette keys mask bare dates but not
# the weekday — so an unpinned clock makes replay miss on six days in seven.
GOLDEN_NOW = datetime(2026, 8, 29, 10, 55, tzinfo=timezone.utc)


@contextlib.contextmanager
def frozen_clock():
    """Pin the engine's idea of "now" to when the fixtures were recorded."""
    import memonative.facade as facade

    real = facade.datetime

    class _Frozen(real):
        @classmethod
        def now(cls, tz=None):
            return GOLDEN_NOW if tz is not None else GOLDEN_NOW.replace(tzinfo=None)

    facade.datetime = _Frozen
    try:
        yield
    finally:
        facade.datetime = real


async def run_corpus(
    llm, *, tenant_id: UUID = DEFAULT_TENANT_ID, verbose: bool = False
) -> dict[str, Any]:
    """Run the full capture script + read surfaces. Returns normalized goldens."""
    with deterministic_uuids(), frozen_clock():
        return await _run_corpus(llm, tenant_id=tenant_id, verbose=verbose)


async def _run_corpus(llm, *, tenant_id: UUID, verbose: bool) -> dict[str, Any]:
    # Imported here so the module is importable before the refactor lands.
    from memonative.api.routes import (
        get_attribute_history,
        get_audit_trail,
        list_slots,
        process_message,
        search_memory,
    )
    from memonative.api.schemas import MemorySearchRequest, ProcessRequest

    out: dict[str, Any] = {}

    async with SessionLocal() as db:
        await purge_user(db, tenant_id, GOLDEN_USER_ID)

    for label, message in CAPTURES:
        async with SessionLocal() as db:
            resp = await process_message(
                ProcessRequest(user_id=GOLDEN_USER_ID, message=message),
                tenant_id=tenant_id,
                db=db,
                llm=llm,
            )
        out[f"process__{label}"] = normalize(_jsonable(resp))
        if verbose:
            print(
                f"  [{label}] wrote={len(resp.memories_written)} "
                f"retrieved={len(resp.memories_retrieved)} "
                f"retracted={len(resp.retracted_ids)}"
            )

    for label, query in SEARCH_QUERIES:
        async with SessionLocal() as db:
            resp = await search_memory(
                MemorySearchRequest(user_id=GOLDEN_USER_ID, query=query, top_k=8),
                tenant_id=tenant_id,
                db=db,
                llm=llm,
            )
        out[f"search__{label}"] = normalize(_jsonable(resp))
        if verbose:
            print(f"  [{label}] matched={len(resp.memories)}")

    async with SessionLocal() as db:
        slots = await list_slots(str(GOLDEN_USER_ID), tenant_id=tenant_id, db=db)
    out["slots"] = normalize(_jsonable(slots))

    async with SessionLocal() as db:
        audit = await get_audit_trail(
            str(GOLDEN_USER_ID), tenant_id=tenant_id, db=db, limit=100, offset=0
        )
    out["audit"] = normalize(_jsonable(audit))

    # Revision chains are the differentiator; capture whatever slots exist.
    for slot in [s.attribute_slot for s in slots.slots]:
        async with SessionLocal() as db:
            hist = await get_attribute_history(
                str(GOLDEN_USER_ID), slot, tenant_id=tenant_id, db=db
            )
        out[f"history__{slot}"] = normalize(hist)

    if verbose:
        print(f"  slots={len(slots.slots)} audit_entries={audit.total}")

    return out
