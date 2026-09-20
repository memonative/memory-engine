"""HTTP surface over `MemoryEngine`.

Every handler now does the same four things: resolve tenant + session + LLM
from `Depends`, parse any path parameters, call the facade, emit a usage
event. The pipeline itself lives in `memonative.facade` so it can run
in-process without a web server.

Engine errors (`NotFoundError` and friends) carry their own status code and
are turned into responses by the `EngineError` handler in `memonative.main` —
handlers here raise `HTTPException` only for transport-level problems the
engine has no opinion about, namely unparseable path parameters.
"""

import logging
import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from memonative.agent_tools import build_tool_manifest
from memonative.api.llm import LLMClient
from memonative.llm import require_env_llm_keys
from memonative.api.schemas import (
    MemoryDetail, MemorySearchRequest, MemorySearchResponse,
    MemoryWriteRequest, MemoryWriteResponse,
    MemoryFactsResponse,
    MemorySlotsResponse,
    AuditTrailResponse,
    ProcessRequest, ProcessResponse,
)
from memonative.api.usage import emit_usage_event, request_timer
from memonative.auth.credentials import resolve_credentials
from memonative.auth.deps import require_tenant
from memonative.db.database import get_db, _set_rls_tenant
from memonative.facade import MemoryEngine

logger = logging.getLogger(__name__)

# Stateless — every call passes tenant, session and LLM explicitly, so one
# instance serves all requests.
engine = MemoryEngine()


async def get_llm(
    tenant_id: UUID = Depends(require_tenant),
    db: AsyncSession = Depends(get_db),
) -> LLMClient:
    from memonative.config import settings as _s
    if not _s.MASTER_ENCRYPTION_KEY.get_secret_value():
        require_env_llm_keys()
        return LLMClient()
    await _set_rls_tenant(db, tenant_id)
    creds = await resolve_credentials(db, tenant_id)
    if not creds.engine_api_key:
        raise HTTPException(
            status_code=422,
            detail=(
                "Engine LLM key not configured for this tenant. Set one with "
                "PUT /admin/tenants/{tenant_id}/credentials, or unset "
                "MASTER_ENCRYPTION_KEY to use the shared DEEPSEEK_API_KEY."
            ),
        )
    return LLMClient.from_credentials(creds)


def _parse_uuid(value: str, field: str) -> UUID:
    try:
        return uuid.UUID(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {field}")


async def _record_usage(db: AsyncSession, **kwargs) -> None:
    """Emit the usage row and commit it.

    Needs its own commit because the facade already committed the request's
    real work. Must not raise for the same reason: the memories are durable by
    now, so letting a billing-row failure surface would turn a request that
    fully succeeded into a 500. `usage.emit_usage_event` makes the same
    promise on its own behalf ("usage tracking must never break a request");
    this keeps it true across the commit.
    """
    try:
        await emit_usage_event(db, **kwargs)
        await db.commit()
    except Exception:
        logger.warning(
            "usage event not recorded for %s", kwargs.get("endpoint"), exc_info=True
        )
        await db.rollback()


router = APIRouter(prefix='/v1', dependencies=[Depends(require_tenant)])

# Health check stays unauth'd so liveness probes work without secrets.
health_router = APIRouter()


@health_router.get('/health')
async def health_check():
    return {"status": "ok"}


@router.post('/process', response_model=ProcessResponse)
async def process_message(
    request: ProcessRequest,
    tenant_id: UUID = Depends(require_tenant),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    with request_timer() as elapsed:
        result = await engine.process(
            user_id=request.user_id,
            message=request.message,
            context=request.context,
            top_k=request.top_k,
            session_id=request.session_id,
            model=request.model,
            db=db,
            tenant_id=tenant_id,
            llm=llm,
        )

        await _record_usage(
            db,
            tenant_id=tenant_id,
            endpoint="/v1/process",
            user_id=request.user_id,
            memories_written=len(result.memories_written),
            memories_retrieved=len(result.memories_retrieved),
            response_time_ms=elapsed(),
        )
    return result


@router.post('/memory/search', response_model=MemorySearchResponse)
async def search_memory(
    request: MemorySearchRequest,
    tenant_id: UUID = Depends(require_tenant),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Read-only memory search for agents.

    Does not run extraction, does not write, does not reinforce. Returns
    JSON-only matches with full provenance so the agent can decide what
    to inject and how much to trust each item.
    """
    with request_timer() as elapsed:
        result = await engine.search(
            user_id=request.user_id,
            query=request.query,
            top_k=request.top_k,
            memory_types=request.memory_types,
            token_budget=request.token_budget,
            include_associations=request.include_associations,
            db=db,
            tenant_id=tenant_id,
            llm=llm,
        )

        await _record_usage(
            db,
            tenant_id=tenant_id,
            endpoint="/v1/memory/search",
            user_id=request.user_id,
            memories_retrieved=len(result.memories),
            response_time_ms=elapsed(),
        )
    return result


@router.get('/memory/{memory_id}', response_model=MemoryDetail)
async def get_memory(
    memory_id: str,
    tenant_id: UUID = Depends(require_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Fetch a single memory's full provenance by id, scoped to tenant.

    Cross-tenant lookups 404 just like a missing id — never reveal that
    the id exists under another tenant.
    """
    return await engine.recall(
        _parse_uuid(memory_id, "memory_id"), db=db, tenant_id=tenant_id
    )


@router.post('/memory/write', response_model=MemoryWriteResponse)
async def write_memory(
    request: MemoryWriteRequest,
    tenant_id: UUID = Depends(require_tenant),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    """Write structured memories directly, bypassing the extraction LLM.

    Use this when the caller already knows the fact (e.g. synced from a
    CRM, asserted by an agent after a tool call). The slot lifecycle,
    decay profile, and revision chain still apply.
    """
    with request_timer() as elapsed:
        result = await engine.write(
            user_id=request.user_id,
            memories=request.memories,
            trigger_message=request.trigger_message,
            model=request.model,
            db=db,
            tenant_id=tenant_id,
            llm=llm,
        )

        await _record_usage(
            db,
            tenant_id=tenant_id,
            endpoint="/v1/memory/write",
            user_id=request.user_id,
            memories_written=len(result.written_ids),
            response_time_ms=elapsed(),
        )
    return result


@router.get('/memory/slots/{user_id}', response_model=MemorySlotsResponse)
async def list_slots(
    user_id: str,
    tenant_id: UUID = Depends(require_tenant),
    db: AsyncSession = Depends(get_db),
):
    """List all attribute slots for a user with their current values.

    Use this to discover what facts the system knows about a user
    before drilling into a specific slot's revision history.
    """
    return await engine.slots(
        _parse_uuid(user_id, "user_id"), db=db, tenant_id=tenant_id
    )


@router.get('/memory/audit/{user_id}', response_model=AuditTrailResponse)
async def get_audit_trail(
    user_id: str,
    tenant_id: UUID = Depends(require_tenant),
    db: AsyncSession = Depends(get_db),
    limit: int = 100,
    offset: int = 0,
):
    """Full chronological audit trail of every fact change for a user.

    Returns all revision entries across all attribute slots, newest
    first. Enterprise compliance teams use this to answer "what
    changed about this user, when, and why" in a single call.
    """
    return await engine.audit(
        _parse_uuid(user_id, "user_id"),
        limit=limit,
        offset=offset,
        db=db,
        tenant_id=tenant_id,
    )


@router.get('/memory/facts/{user_id}/{attribute_slot}', response_model=MemoryFactsResponse)
async def get_facts(
    user_id: str,
    attribute_slot: str,
    tenant_id: UUID = Depends(require_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Current value + full revision history for one attribute slot.

    Strongly-typed twin of /history/{user_id}/{attribute_slot}; the
    older route is kept for backward compatibility.
    """
    return await engine.facts(
        _parse_uuid(user_id, "user_id"),
        attribute_slot,
        db=db,
        tenant_id=tenant_id,
    )


@router.get('/tools/{provider}')
async def get_tool_manifest(provider: str):
    """Return ready-to-register tool schemas for an agent SDK.

    provider: "anthropic", "openai", or "deepseek".

    The returned tools (search_memory, recall_fact, save_memory,
    list_history) require the calling app to bind user_id server-side
    before forwarding to /v1/memory/*. The schemas leave user_id out so
    an agent can't read another user's memory by guessing a UUID.
    """
    try:
        return build_tool_manifest(provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post('/consolidate/{user_id}')
async def trigger_consolidation(
    user_id: str,
    tenant_id: UUID = Depends(require_tenant),
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm),
):
    processed = await engine.consolidate(
        _parse_uuid(user_id, "user_id"), db=db, tenant_id=tenant_id, llm=llm
    )
    return {"status": "success", "clusters_processed": processed}


@router.get('/history/{user_id}/{attribute_slot}')
async def get_attribute_history(
    user_id: str,
    attribute_slot: str,
    tenant_id: UUID = Depends(require_tenant),
    db: AsyncSession = Depends(get_db),
):
    """Query the full revision history for a specific attribute slot.

    Returns the chain of values with timestamps and reasons for each change.
    Useful for "where has the user lived?" or "what names has the user used?"
    """
    return await engine.history(
        _parse_uuid(user_id, "user_id"),
        attribute_slot,
        db=db,
        tenant_id=tenant_id,
    )
