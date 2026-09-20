"""Memonative Python SDK.

Distributed as `memonative-client`, separate from the `memonative` engine and
MIT-licensed rather than BSL: an application that only calls the engine over
HTTP should not have to install asyncpg, celery and alembic to do it, nor
take a source-available licence into its own codebase.

    from memonative_client import Client, AsyncClient
    from memonative_client.errors import MemonativeError, AuthError
"""

from memonative_client.client import Client
from memonative_client.async_client import AsyncClient
from memonative_client.errors import (
    MemonativeError,
    AuthError,
    NotFoundError,
    ValidationError,
    ServerError,
)
from memonative_client.tool_dispatch import (
    dispatch_tool_call,
    async_dispatch_tool_call,
    UnknownToolError,
)
from memonative_client.models import (
    MemoryDetail,
    MemorySearchResponse,
    MemoryWriteResponse,
    MemoryFactsResponse,
    ProcessResponse,
    TenantOut,
    TenantWithInitialKey,
    KeyOut,
    KeyWithPlaintext,
)

__all__ = [
    "Client",
    "AsyncClient",
    "MemonativeError",
    "AuthError",
    "NotFoundError",
    "ValidationError",
    "ServerError",
    "UnknownToolError",
    "dispatch_tool_call",
    "async_dispatch_tool_call",
    "MemoryDetail",
    "MemorySearchResponse",
    "MemoryWriteResponse",
    "MemoryFactsResponse",
    "ProcessResponse",
    "TenantOut",
    "TenantWithInitialKey",
    "KeyOut",
    "KeyWithPlaintext",
]
