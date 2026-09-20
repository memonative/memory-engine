"""The engine's public request/response vocabulary.

What you hand `MemoryEngine` and what it hands back. These live here rather
than under `api/` because a caller that imports the engine in-process and
never starts a web server still needs them, and needing them should not drag
FastAPI in. Nothing in this module may import beyond pydantic and the
standard library — that is the whole point of it.

`memonative.api.schemas` re-exports everything here for the HTTP layer.
"""

from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from typing import Any
from uuid import UUID

# Limits keep request payloads bounded so a single caller can't
# trigger arbitrarily large LLM/embedding workloads.
MAX_MESSAGE_CHARS = 8000
MAX_CONTEXT_ITEMS = 32
MAX_CONTEXT_ITEM_CHARS = 2000

ALLOWED_MEMORY_TYPES = {"episodic", "semantic", "procedural"}


class ProcessRequest(BaseModel):
    user_id: UUID
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: UUID | None = None
    context: list[str] = Field(
        default_factory=list,
        max_length=MAX_CONTEXT_ITEMS,
    )
    top_k: int = Field(default=8, ge=1, le=50)
    model: str | None = Field(default=None, max_length=200)

    @field_validator("context")
    @classmethod
    def _check_context_item_length(cls, v: list[str]) -> list[str]:
        for item in v:
            if len(item) > MAX_CONTEXT_ITEM_CHARS:
                raise ValueError(
                    f"context item exceeds {MAX_CONTEXT_ITEM_CHARS} chars"
                )
        return v

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
    goal_updated: UUID | None
    contradictions_flagged: list[UUID] = Field(default_factory=list)
    retracted_ids: list[UUID] = Field(default_factory=list)


# ---------------------------------------------------------------
# Agent-native primitives
#
# Designed for tool-calling agents (Anthropic / OpenAI / LangGraph):
# JSON-only output, full provenance, token-budget control,
# read-only retrieval that does NOT trigger writes.
# ---------------------------------------------------------------

class MemoryDetail(BaseModel):
    """Full provenance record for a single memory.

    Designed so an agent can cite a claim back to a memory id and
    judge how much to trust it (trust_score, source, strength,
    reinforcement_count, decay_state, age).
    """
    id: UUID
    content: str
    type: str
    attribute_slot: str | None = None
    score: float = 0.0  # relevance score from search; 0 for direct lookups
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


class DecayedMemory(BaseModel):
    """A memory listed by lifecycle state rather than by relevance.

    Deliberately not `MemoryDetail`. That model is a provenance record for an
    agent deciding how much to trust a claim; this one answers "how long will
    you keep this, and can I stop you forgetting it", which needs the two
    half-lives and needs neither a relevance score nor a token estimate.

    `half_life_hours` is the configured lifetime; `effective_half_life_hours`
    is the one actually in force once reinforcement and salience are folded
    in. The second is the number worth showing, since a much-repeated memory
    outlives its nominal setting by a wide margin. It is computed here rather
    than client-side to keep one implementation of the Ebbinghaus formula.

    **Both are None when the memory is pinned**, which is also how a caller
    detects a pin. None rather than infinity because infinity has no JSON
    spelling, and "no half-life" is the truer statement anyway.
    """
    id: UUID
    content: str
    type: str
    attribute_slot: str | None = None
    decay_state: str
    strength: float
    half_life_hours: float | None = None
    effective_half_life_hours: float | None = None
    reinforcement_count: int
    salience: float
    created_at: datetime
    last_accessed_at: datetime


class MemorySearchRequest(BaseModel):
    user_id: UUID
    query: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    top_k: int = Field(default=8, ge=1, le=50)
    memory_types: list[str] | None = None
    token_budget: int | None = Field(default=None, ge=1, le=200_000)
    include_associations: bool = True

    @field_validator("memory_types")
    @classmethod
    def _validate_types(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        bad = [t for t in v if t not in ALLOWED_MEMORY_TYPES]
        if bad:
            raise ValueError(
                f"unknown memory_types: {bad}. allowed: {sorted(ALLOWED_MEMORY_TYPES)}"
            )
        return v


class MemorySearchResponse(BaseModel):
    query: str
    memories: list[MemoryDetail]
    total_estimated_tokens: int
    truncated: bool = False


class MemoryWriteItem(BaseModel):
    """A structured memory the caller wants written directly.

    Bypasses the extraction LLM — use this when the caller already
    knows the fact (e.g., synced from a CRM, asserted by an agent
    after a tool call).
    """
    content: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    type: str = Field(default="episodic")
    attribute_slot: str | None = Field(default=None, max_length=50)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = "system_inferred"
    context_tags: list[str] = Field(default_factory=list)
    correction_type: str | None = None  # "retraction" | "temporal_update" | None

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in ALLOWED_MEMORY_TYPES:
            raise ValueError(
                f"unknown type {v!r}. allowed: {sorted(ALLOWED_MEMORY_TYPES)}"
            )
        return v


class MemoryWriteRequest(BaseModel):
    user_id: UUID
    memories: list[MemoryWriteItem] = Field(min_length=1, max_length=32)
    trigger_message: str | None = Field(default=None, max_length=MAX_MESSAGE_CHARS)
    model: str | None = Field(default=None, max_length=200)


class MemoryWriteResponse(BaseModel):
    written_ids: list[UUID]
    contradictions_flagged: list[UUID] = Field(default_factory=list)


class MemoryRevisionItem(BaseModel):
    sequence: int
    content: str
    revision_type: str
    reason: str | None
    valid_from: datetime | None
    valid_until: datetime | None
    trigger_message: str | None


class MemorySlotItem(BaseModel):
    attribute_slot: str
    current_value: str
    revision_count: int
    last_updated: datetime | None


class MemorySlotsResponse(BaseModel):
    user_id: UUID
    slots: list[MemorySlotItem]


class AuditEntry(BaseModel):
    attribute_slot: str
    sequence: int
    content: str
    revision_type: str
    reason: str | None
    trigger_message: str | None
    valid_from: datetime | None
    valid_until: datetime | None
    is_current: bool


class AuditTrailResponse(BaseModel):
    user_id: UUID
    total: int
    entries: list[AuditEntry]


class MemoryFactsResponse(BaseModel):
    user_id: UUID
    attribute_slot: str
    current_value: str | None
    revisions: list[MemoryRevisionItem]
