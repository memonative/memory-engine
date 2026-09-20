"""Backwards-compatible re-export of `memonative.types`.

The models moved out of the API layer so the engine can be imported without
FastAPI. This shim keeps `from memonative.api.schemas import ...` working for
the HTTP layer and for anything pinned to the old path.

Named explicitly rather than star-imported: deleting or renaming a model in
`memonative.types` should break here loudly, at import time, instead of
silently shrinking this module's surface.
"""

from memonative.types import (
    ALLOWED_MEMORY_TYPES,
    MAX_CONTEXT_ITEM_CHARS,
    MAX_CONTEXT_ITEMS,
    MAX_MESSAGE_CHARS,
    AuditEntry,
    AuditTrailResponse,
    IntentResponse,
    MemoryDetail,
    MemoryFactsResponse,
    MemoryRevisionItem,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySlotItem,
    MemorySlotsResponse,
    MemoryWriteItem,
    MemoryWriteRequest,
    MemoryWriteResponse,
    ProcessRequest,
    ProcessResponse,
    RetrievedMemory,
)

__all__ = [
    "ALLOWED_MEMORY_TYPES",
    "MAX_CONTEXT_ITEM_CHARS",
    "MAX_CONTEXT_ITEMS",
    "MAX_MESSAGE_CHARS",
    "AuditEntry",
    "AuditTrailResponse",
    "IntentResponse",
    "MemoryDetail",
    "MemoryFactsResponse",
    "MemoryRevisionItem",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "MemorySlotItem",
    "MemorySlotsResponse",
    "MemoryWriteItem",
    "MemoryWriteRequest",
    "MemoryWriteResponse",
    "ProcessRequest",
    "ProcessResponse",
    "RetrievedMemory",
]
