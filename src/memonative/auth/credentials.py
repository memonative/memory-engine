"""Resolve per-tenant LLM credentials at request time.

Cloud edition: engine LLM key is compulsory BYOK — tenants must
provide their own key via the dashboard. Embeddings use the
platform's global OpenAI key (cost absorbed).

OSS edition: falls back to global env vars for everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from memonative.config import settings
from memonative.db.models import TenantCredential

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedCredentials:
    engine_api_key: str
    engine_base_url: str
    engine_model: str
    embedding_api_key: str


async def resolve_credentials(
    db: AsyncSession, tenant_id: UUID
) -> ResolvedCredentials:
    """Look up per-tenant keys, decrypt, and resolve engine + embedding keys.

    The global env keys are the baseline. A tenant row overrides them
    per-field, so a deployment can run one shared key, per-tenant keys,
    or a mix. Tenant rows are only read when `MASTER_ENCRYPTION_KEY` is
    set, since without it there is nothing to decrypt them with.
    """
    cred = None
    if settings.MASTER_ENCRYPTION_KEY.get_secret_value():
        try:
            async with db.begin_nested():
                row = await db.execute(
                    select(TenantCredential).where(
                        TenantCredential.tenant_id == tenant_id
                    )
                )
                cred = row.scalar_one_or_none()
        except ProgrammingError:
            logger.debug("tenant_credentials table missing — using global keys")

    engine_key = settings.DEEPSEEK_API_KEY.get_secret_value()
    engine_url = settings.DEEPSEEK_BASE_URL
    engine_model = ""

    if cred is not None:
        from memonative.auth.encryption import decrypt_secret

        if cred.engine_api_key_enc:
            engine_key = decrypt_secret(cred.engine_api_key_enc)
        if cred.engine_base_url:
            engine_url = cred.engine_base_url
        if cred.engine_model:
            engine_model = cred.engine_model

    # Embeddings: always platform-absorbed (global OpenAI key)
    embed_key = settings.OPENAI_API_KEY.get_secret_value()

    return ResolvedCredentials(
        engine_api_key=engine_key,
        engine_base_url=engine_url,
        engine_model=engine_model,
        embedding_api_key=embed_key,
    )
