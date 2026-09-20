"""FastAPI auth dependencies.

Two trust levels:

  require_tenant   — resolves the bearer token to a tenant_id. Used on
                     every /v1/* route. Empty settings.API_KEY = dev mode
                     and resolves to the implicit default tenant.

  require_admin    — guards /admin/* routes. Only the master API_KEY env
                     var works here; tenant-scoped keys are rejected even
                     if valid.

Plaintext bearer tokens are SHA-256 hashed and looked up by digest. The
master env-var key is itself stored as an ApiKey row tied to the default
tenant (created in the bootstrap step) so the lookup path is uniform.
"""

from __future__ import annotations

import hmac
from datetime import datetime, timezone
from uuid import UUID

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from memonative.auth.keys import hash_key
from memonative.config import DEFAULT_TENANT_ID, settings
from memonative.db.database import get_db
from memonative.db.models import ApiKey, Tenant


# Re-exported from `config` — importers here predate the move and the
# constant is not FastAPI-specific.
__all__ = ["DEFAULT_TENANT_ID", "require_admin", "require_tenant"]


# auto_error=False so /docs renders the Authorize button without
# 403'ing every unfilled request.
_bearer_scheme = HTTPBearer(auto_error=False)


def _extract_bearer(creds: HTTPAuthorizationCredentials | None) -> str | None:
    if creds is None:
        return None
    if creds.scheme.lower() != "bearer":
        return None
    return creds.credentials


async def require_tenant(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> UUID:
    """Resolve a bearer token to a tenant_id.

    Dev mode (`settings.API_KEY` empty AND no bearer sent): returns the
    default tenant. Anything else requires a valid, non-revoked key.
    """
    bearer = _extract_bearer(creds)
    master_key = settings.API_KEY.get_secret_value()

    if not master_key and bearer is None:
        return DEFAULT_TENANT_ID

    if bearer is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    if master_key and hmac.compare_digest(bearer, master_key):
        return DEFAULT_TENANT_ID

    digest = hash_key(bearer)
    row = await db.execute(
        select(ApiKey, Tenant)
        .join(Tenant, ApiKey.tenant_id == Tenant.id)
        .where(ApiKey.key_hash == digest)
    )
    pair = row.first()
    if pair is None:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    api_key, tenant = pair
    if api_key.revoked_at is not None:
        raise HTTPException(status_code=401, detail="Key revoked")
    if tenant.status != "active":
        raise HTTPException(status_code=403, detail="Tenant not active")

    # Best-effort touch. Failure here must not block the request, so we
    # let the surrounding transaction carry it (committed by the route).
    await db.execute(
        update(ApiKey)
        .where(ApiKey.id == api_key.id)
        .values(last_used_at=datetime.now(timezone.utc))
    )

    return tenant.id


async def require_admin(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """Guard /admin/* routes. Only the master env-var key opens these."""
    expected = settings.API_KEY.get_secret_value()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin API disabled (set API_KEY to enable)",
        )

    bearer = _extract_bearer(creds)
    if bearer is None:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if not hmac.compare_digest(bearer, expected):
        raise HTTPException(status_code=403, detail="Admin access required")
