"""Synchronous Memonative client.

Design rules:
  - Thin wrappers: each method maps to one HTTP call. No retries, no
    caching, no hidden batching. Add those when a real user asks.
  - Typed returns: every method hands back a Pydantic model from
    `memonative_client.models`, not a raw dict.
  - Symmetric with AsyncClient — if you add a method here, mirror the
    signature there.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping
from uuid import UUID

import httpx

from memonative_client._transport import build_headers, raise_for_status
from memonative_client.models import (
    KeyOut,
    KeyWithPlaintext,
    MemoryDetail,
    MemoryFactsResponse,
    MemorySearchResponse,
    MemoryWriteResponse,
    ProcessResponse,
    TenantOut,
    TenantWithInitialKey,
)


class Client:
    """Sync client for the Memonative HTTP API.

    Example:
        with Client("http://localhost:8000", api_key="mn_…") as mn:
            mn.process(user_id=uid, message="I moved to Berlin")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
        )
        self.admin = AdminAPI(self)

    # ---- lifecycle --------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- internal ---------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        resp = self._http.request(
            method,
            path,
            json=json,
            params=params,
            headers=build_headers(self.api_key),
        )
        raise_for_status(resp)
        if not resp.content:
            return None
        return resp.json()

    # ---- /v1 agent primitives --------------------------------------

    def process(
        self,
        user_id: UUID | str,
        message: str,
        *,
        context: Iterable[str] | None = None,
        session_id: UUID | str | None = None,
        top_k: int = 8,
    ) -> ProcessResponse:
        body: dict[str, Any] = {
            "user_id": str(user_id),
            "message": message,
            "context": list(context) if context else [],
            "top_k": top_k,
        }
        if session_id is not None:
            body["session_id"] = str(session_id)
        data = self._request("POST", "/v1/process", json=body)
        return ProcessResponse.model_validate(data)

    def search(
        self,
        user_id: UUID | str,
        query: str,
        *,
        top_k: int = 8,
        memory_types: list[str] | None = None,
        token_budget: int | None = None,
        include_associations: bool = True,
    ) -> MemorySearchResponse:
        body: dict[str, Any] = {
            "user_id": str(user_id),
            "query": query,
            "top_k": top_k,
            "include_associations": include_associations,
        }
        if memory_types is not None:
            body["memory_types"] = memory_types
        if token_budget is not None:
            body["token_budget"] = token_budget
        data = self._request("POST", "/v1/memory/search", json=body)
        return MemorySearchResponse.model_validate(data)

    def recall(self, user_id: UUID | str, attribute_slot: str) -> MemoryFactsResponse:
        """Current value + revision chain for one attribute slot.

        Wraps GET /v1/memory/facts/{user_id}/{slot}. Use this for exact
        fact lookup; use search() for similarity search.
        """
        data = self._request(
            "GET", f"/v1/memory/facts/{user_id}/{attribute_slot}",
        )
        return MemoryFactsResponse.model_validate(data)

    # Alias — the tool manifest calls this "list_history" and the
    # legacy /history route covers the same shape.
    history = recall

    def write(
        self,
        user_id: UUID | str,
        memories: list[dict[str, Any]],
        *,
        trigger_message: str | None = None,
    ) -> MemoryWriteResponse:
        body: dict[str, Any] = {
            "user_id": str(user_id),
            "memories": memories,
        }
        if trigger_message is not None:
            body["trigger_message"] = trigger_message
        data = self._request("POST", "/v1/memory/write", json=body)
        return MemoryWriteResponse.model_validate(data)

    def get_memory(self, memory_id: UUID | str) -> MemoryDetail:
        data = self._request("GET", f"/v1/memory/{memory_id}")
        return MemoryDetail.model_validate(data)

    def consolidate(self, user_id: UUID | str) -> dict[str, Any]:
        return self._request("POST", f"/v1/consolidate/{user_id}")

    def tool_manifest(self, provider: str) -> dict[str, Any]:
        """Fetch the tool manifest for 'anthropic', 'openai', or 'deepseek'.

        DeepSeek's tool-calling schema is identical to OpenAI's (same
        SDK, base_url-overridden) — exposed as its own provider name
        purely for discoverability.
        """
        return self._request("GET", f"/v1/tools/{provider}")


class AdminAPI:
    """Thin namespace for /admin routes. Requires the master key."""

    def __init__(self, client: Client) -> None:
        self._c = client

    def create_tenant(self, name: str) -> TenantWithInitialKey:
        data = self._c._request(
            "POST", "/admin/tenants", json={"name": name},
        )
        return TenantWithInitialKey.model_validate(data)

    def list_tenants(self) -> list[TenantOut]:
        data = self._c._request("GET", "/admin/tenants")
        return [TenantOut.model_validate(t) for t in data]

    def mint_key(
        self, tenant_id: UUID | str, *, label: str | None = None,
    ) -> KeyWithPlaintext:
        data = self._c._request(
            "POST", f"/admin/tenants/{tenant_id}/keys",
            json={"label": label},
        )
        return KeyWithPlaintext.model_validate(data)

    def revoke_key(self, key_id: UUID | str) -> None:
        self._c._request("DELETE", f"/admin/keys/{key_id}")
