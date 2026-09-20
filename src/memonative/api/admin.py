"""Admin surface for tenant + API key management.

Master-key gated (`require_admin`). Tenant-scoped keys are rejected
even if otherwise valid.

Plaintext bearer tokens are returned exactly once at creation; the
caller must capture them immediately. The DB only stores the SHA-256
digest — there is no recovery path for a lost key, only revoke + mint.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from memonative.auth.deps import require_admin
from memonative.auth.encryption import encrypt_secret
from memonative.auth.keys import generate_key, hash_key
from memonative.db.database import get_db
from memonative.db.models import ApiKey, Tenant, TenantCredential


# ---- Schemas ----------------------------------------------------------

class TenantCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class TenantOut(BaseModel):
    id: UUID
    name: str
    status: str
    created_at: datetime


class KeyCreate(BaseModel):
    label: str | None = Field(default=None, max_length=100)


class KeyOut(BaseModel):
    id: UUID
    tenant_id: UUID
    label: str | None
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class KeyWithPlaintext(BaseModel):
    """Plaintext is shown ONCE. Callers must store it immediately."""
    api_key: str
    key: KeyOut


class TenantWithInitialKey(BaseModel):
    tenant: TenantOut
    api_key: str
    key: KeyOut


class CredentialSet(BaseModel):
    """Set per-tenant BYOK credentials. All fields optional — only
    provided fields are updated."""
    engine_api_key: str | None = Field(default=None, min_length=1)
    engine_base_url: str | None = Field(default=None, max_length=500)
    embedding_api_key: str | None = Field(default=None, min_length=1)

    @field_validator("engine_base_url")
    @classmethod
    def _validate_url_scheme(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.startswith("https://"):
            raise ValueError("engine_base_url must use https://")
        return v


class CredentialOut(BaseModel):
    """Masked view of stored credentials. Never exposes full keys."""
    tenant_id: UUID
    has_engine_key: bool
    engine_base_url: str | None
    has_embedding_key: bool
    updated_at: datetime


# ---- Router -----------------------------------------------------------

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


@router.post("/tenants", response_model=TenantWithInitialKey, status_code=201)
async def create_tenant(
    request: TenantCreate,
    db: AsyncSession = Depends(get_db),
):
    tenant = Tenant(name=request.name, status="active")
    db.add(tenant)
    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Tenant name already exists")

    plaintext = generate_key()
    key_row = ApiKey(
        tenant_id=tenant.id,
        key_hash=hash_key(plaintext),
        label="initial",
    )
    db.add(key_row)
    await db.commit()
    await db.refresh(tenant)
    await db.refresh(key_row)

    return TenantWithInitialKey(
        tenant=TenantOut.model_validate(tenant, from_attributes=True),
        api_key=plaintext,
        key=KeyOut.model_validate(key_row, from_attributes=True),
    )


@router.get("/tenants", response_model=list[TenantOut])
async def list_tenants(db: AsyncSession = Depends(get_db)):
    rows = await db.execute(select(Tenant).order_by(Tenant.created_at.asc()))
    return [TenantOut.model_validate(t, from_attributes=True) for t in rows.scalars()]


@router.post(
    "/tenants/{tenant_id}/keys",
    response_model=KeyWithPlaintext,
    status_code=201,
)
async def mint_key(
    tenant_id: UUID,
    request: KeyCreate,
    db: AsyncSession = Depends(get_db),
):
    tenant = await db.get(Tenant, tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    plaintext = generate_key()
    key_row = ApiKey(
        tenant_id=tenant_id,
        key_hash=hash_key(plaintext),
        label=request.label,
    )
    db.add(key_row)
    await db.commit()
    await db.refresh(key_row)
    return KeyWithPlaintext(
        api_key=plaintext,
        key=KeyOut.model_validate(key_row, from_attributes=True),
    )


@router.delete("/keys/{key_id}", status_code=204)
async def revoke_key(
    key_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    key_row = await db.get(ApiKey, key_id)
    if key_row is None:
        raise HTTPException(status_code=404, detail="Key not found")
    if key_row.revoked_at is None:
        key_row.revoked_at = datetime.now(timezone.utc)
        await db.commit()
    return None


# ---- Tenant Credentials (BYOK) ----------------------------------------


@router.put(
    "/tenants/{tenant_id}/credentials",
    response_model=CredentialOut,
)
async def set_credentials(
    tenant_id: UUID,
    request: CredentialSet,
    db: AsyncSession = Depends(get_db),
):
    """Set or update per-tenant BYOK LLM credentials.

    Keys are envelope-encrypted at rest with MASTER_ENCRYPTION_KEY.
    Only provided (non-null) fields are updated; omitted fields keep
    their current value.
    """
    from memonative.config import settings as _settings
    if not _settings.MASTER_ENCRYPTION_KEY.get_secret_value():
        raise HTTPException(
            status_code=503,
            detail="BYOK not available (MASTER_ENCRYPTION_KEY not configured)",
        )

    tenant = await db.get(Tenant, tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    row = await db.execute(
        select(TenantCredential).where(
            TenantCredential.tenant_id == tenant_id
        )
    )
    cred = row.scalar_one_or_none()

    if cred is None:
        cred = TenantCredential(tenant_id=tenant_id)
        db.add(cred)

    if request.engine_api_key is not None:
        cred.engine_api_key_enc = encrypt_secret(request.engine_api_key)
    if request.engine_base_url is not None:
        cred.engine_base_url = request.engine_base_url
    if request.embedding_api_key is not None:
        cred.embedding_api_key_enc = encrypt_secret(request.embedding_api_key)

    cred.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(cred)

    return CredentialOut(
        tenant_id=cred.tenant_id,
        has_engine_key=cred.engine_api_key_enc is not None,
        engine_base_url=cred.engine_base_url,
        has_embedding_key=cred.embedding_api_key_enc is not None,
        updated_at=cred.updated_at,
    )


@router.get(
    "/tenants/{tenant_id}/credentials",
    response_model=CredentialOut,
)
async def get_credentials(
    tenant_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """View (masked) credential status for a tenant."""
    tenant = await db.get(Tenant, tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    row = await db.execute(
        select(TenantCredential).where(
            TenantCredential.tenant_id == tenant_id
        )
    )
    cred = row.scalar_one_or_none()

    if cred is None:
        return CredentialOut(
            tenant_id=tenant_id,
            has_engine_key=False,
            engine_base_url=None,
            has_embedding_key=False,
            updated_at=datetime.now(timezone.utc),
        )

    return CredentialOut(
        tenant_id=cred.tenant_id,
        has_engine_key=cred.engine_api_key_enc is not None,
        engine_base_url=cred.engine_base_url,
        has_embedding_key=cred.embedding_api_key_enc is not None,
        updated_at=cred.updated_at,
    )


@router.delete("/tenants/{tenant_id}/credentials", status_code=204)
async def delete_credentials(
    tenant_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Remove per-tenant credentials, reverting to global env vars."""
    row = await db.execute(
        select(TenantCredential).where(
            TenantCredential.tenant_id == tenant_id
        )
    )
    cred = row.scalar_one_or_none()
    if cred is not None:
        await db.delete(cred)
        await db.commit()
    return None
