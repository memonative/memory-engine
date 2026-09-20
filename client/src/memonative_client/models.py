"""Pydantic response models exposed by the SDK.

Deliberately not re-exported from the server's `memonative.api.schemas`
— that module pulls in SQLAlchemy and pgvector (via engine imports
downstream) which SDK users should not have to install. These models
are a hand-written subset matching the server's JSON shape.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# ---- Agent-native response shapes -----------------------------------


class MemoryDetail(BaseModel):
    id: UUID
    content: str
    type: str
    attribute_slot: str | None = None
    score: float = 0.0
    strength: float
    trust_score: float
    salience: float
    source: str
    decay_state: str
    reinforcement_count: int
    context_tags: list[str] = Field(default_factory=list)
    created_at: datetime
    last_accessed_at: datetime
    estimated_tokens: int


class MemorySearchResponse(BaseModel):
    query: str
    memories: list[MemoryDetail]
    total_estimated_tokens: int
    truncated: bool = False


class MemoryWriteResponse(BaseModel):
    written_ids: list[UUID]
    contradictions_flagged: list[UUID] = Field(default_factory=list)


class MemoryRevisionItem(BaseModel):
    sequence: int
    content: str
    revision_type: str
    reason: str | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    trigger_message: str | None = None


class MemoryFactsResponse(BaseModel):
    user_id: UUID
    attribute_slot: str
    current_value: str | None
    revisions: list[MemoryRevisionItem]


# ---- /v1/process ----------------------------------------------------


class IntentResponse(BaseModel):
    primary: str
    mode: str


class RetrievedMemory(BaseModel):
    id: UUID
    content: str
    type: str
    score: float
    strength: float
    trust_score: float


class ProcessResponse(BaseModel):
    intent: IntentResponse
    memories_written: list[UUID]
    memories_retrieved: list[RetrievedMemory]
    injection_text: str
    goal_updated: UUID | None = None
    contradictions_flagged: list[UUID] = Field(default_factory=list)
    retracted_ids: list[UUID] = Field(default_factory=list)


# ---- /admin shapes --------------------------------------------------


class TenantOut(BaseModel):
    id: UUID
    name: str
    status: str
    created_at: datetime


class KeyOut(BaseModel):
    id: UUID
    tenant_id: UUID
    label: str | None = None
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


class KeyWithPlaintext(BaseModel):
    api_key: str
    key: KeyOut


class TenantWithInitialKey(BaseModel):
    tenant: TenantOut
    api_key: str
    key: KeyOut
