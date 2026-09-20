"""App-startup bootstrap for the tenancy layer.

Idempotent. Safe to run on every boot.

Guarantees on completion:
  - The 'default' tenant row exists (id = DEFAULT_TENANT_ID).
  - If settings.API_KEY is set, an ApiKey row hashed from it exists,
    bound to the default tenant, label='master'.

Bootstrap intentionally does NOT mint a tenant key when API_KEY is
empty — that's dev mode and the auth dependency already short-circuits
without consulting the table.

Uses raw SQL to avoid ORM/RLS issues during startup.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from memonative.auth.deps import DEFAULT_TENANT_ID
from memonative.auth.keys import hash_key
from memonative.config import settings
from memonative.db.database import engine

logger = logging.getLogger(__name__)


async def ensure_default_tenant_and_master_key() -> None:
    async with engine.begin() as conn:
        result = await conn.execute(
            text("SELECT id FROM tenants WHERE id = :tid"),
            {"tid": str(DEFAULT_TENANT_ID)},
        )
        if result.first() is None:
            await conn.execute(
                text(
                    "INSERT INTO tenants (id, name, status) "
                    "VALUES (:tid, 'default', 'active') "
                    "ON CONFLICT (name) DO UPDATE SET id = :tid"
                ),
                {"tid": str(DEFAULT_TENANT_ID)},
            )
            logger.info("bootstrap: created default tenant %s", DEFAULT_TENANT_ID)

    if not settings.API_KEY.get_secret_value():
        return

    digest = hash_key(settings.API_KEY.get_secret_value())

    async with engine.begin() as conn:
        await conn.execute(
            text("SELECT set_config('app.current_tenant_id', :tid, true)"),
            {"tid": str(DEFAULT_TENANT_ID)},
        )
        result = await conn.execute(
            text("SELECT id FROM api_keys WHERE key_hash = :hash"),
            {"hash": digest},
        )
        if result.first() is None:
            await conn.execute(
                text(
                    "INSERT INTO api_keys (tenant_id, key_hash, label) "
                    "VALUES (:tid, :hash, 'master') "
                    "ON CONFLICT DO NOTHING"
                ),
                {"tid": str(DEFAULT_TENANT_ID), "hash": digest},
            )
            logger.info("bootstrap: registered master key for default tenant")
