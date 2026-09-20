"""Usage accounting seam.

The OSS engine does not meter requests. `emit_usage_event` exists so the
request handlers have a stable call site; hosted deployments replace this
module to record per-tenant usage.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession


@contextmanager
def request_timer():
    """Context manager that yields a callable returning elapsed ms."""
    t0 = time.monotonic()
    elapsed = lambda: int((time.monotonic() - t0) * 1000)  # noqa: E731
    yield elapsed


async def emit_usage_event(
    db: AsyncSession,
    *,
    tenant_id: UUID,
    endpoint: str,
    user_id: UUID | None = None,
    api_key_id: UUID | None = None,
    memories_written: int = 0,
    memories_retrieved: int = 0,
    response_time_ms: int = 0,
) -> None:
    return None
