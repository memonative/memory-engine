"""SDK unit tests using httpx.MockTransport.

No network required: the transport replays canned responses and
records the outgoing request so we can assert the SDK built it
correctly (URL, headers, body).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import httpx
import pytest

from memonative_client import (
    AsyncClient,
    AuthError,
    Client,
    NotFoundError,
    ServerError,
    ValidationError,
    async_dispatch_tool_call,
    dispatch_tool_call,
)
from memonative_client.tool_dispatch import UnknownToolError


USER_ID = UUID("11111111-1111-1111-1111-111111111111")
API_KEY = "mn_test-key"


# ---- helpers --------------------------------------------------------


def _capturing_handler(responder):
    """MockTransport handler that records the request before replying."""
    captured = {"request": None}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        return responder(request)

    return handler, captured


def _json_body(request: httpx.Request) -> dict | list | None:
    if not request.content:
        return None
    return json.loads(request.content)


def _sample_memory_detail() -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": str(uuid4()),
        "content": "sample",
        "type": "semantic",
        "attribute_slot": "location",
        "score": 0.0,
        "strength": 1.0,
        "trust_score": 0.8,
        "salience": 0.5,
        "source": "user_stated",
        "decay_state": "active",
        "reinforcement_count": 1,
        "context_tags": [],
        "created_at": now,
        "last_accessed_at": now,
        "estimated_tokens": 2,
    }


# ---- auth + transport -----------------------------------------------


def test_client_sends_bearer_header():
    handler, cap = _capturing_handler(
        lambda r: httpx.Response(200, json={
            "query": "hi", "memories": [],
            "total_estimated_tokens": 0, "truncated": False,
        })
    )
    with Client("http://x", api_key=API_KEY, transport=httpx.MockTransport(handler)) as c:
        c.search(USER_ID, "hi")

    assert cap["request"].headers["Authorization"] == f"Bearer {API_KEY}"


def test_client_omits_bearer_when_no_key():
    handler, cap = _capturing_handler(
        lambda r: httpx.Response(200, json={
            "query": "hi", "memories": [],
            "total_estimated_tokens": 0, "truncated": False,
        })
    )
    with Client("http://x", api_key=None, transport=httpx.MockTransport(handler)) as c:
        c.search(USER_ID, "hi")

    assert "Authorization" not in cap["request"].headers


# ---- URL routing + payload shape ------------------------------------


def test_search_builds_correct_request():
    handler, cap = _capturing_handler(
        lambda r: httpx.Response(200, json={
            "query": "hi", "memories": [_sample_memory_detail()],
            "total_estimated_tokens": 2, "truncated": False,
        })
    )
    with Client("http://x", api_key=API_KEY, transport=httpx.MockTransport(handler)) as c:
        resp = c.search(USER_ID, "hi", top_k=5, memory_types=["semantic"], token_budget=500)

    req = cap["request"]
    assert req.method == "POST"
    assert req.url.path == "/v1/memory/search"
    body = _json_body(req)
    assert body["user_id"] == str(USER_ID)
    assert body["top_k"] == 5
    assert body["memory_types"] == ["semantic"]
    assert body["token_budget"] == 500
    assert resp.total_estimated_tokens == 2


def test_recall_uses_get_and_path_params():
    handler, cap = _capturing_handler(
        lambda r: httpx.Response(200, json={
            "user_id": str(USER_ID),
            "attribute_slot": "location",
            "current_value": "Berlin",
            "revisions": [],
        })
    )
    with Client("http://x", api_key=API_KEY, transport=httpx.MockTransport(handler)) as c:
        resp = c.recall(USER_ID, "location")

    assert cap["request"].method == "GET"
    assert cap["request"].url.path == f"/v1/memory/facts/{USER_ID}/location"
    assert resp.current_value == "Berlin"


def test_write_wraps_memories_list():
    handler, cap = _capturing_handler(
        lambda r: httpx.Response(200, json={
            "written_ids": [str(uuid4())],
            "contradictions_flagged": [],
        })
    )
    with Client("http://x", api_key=API_KEY, transport=httpx.MockTransport(handler)) as c:
        c.write(USER_ID, memories=[{"content": "hello"}], trigger_message="why")

    body = _json_body(cap["request"])
    assert body["user_id"] == str(USER_ID)
    assert body["memories"] == [{"content": "hello"}]
    assert body["trigger_message"] == "why"


# ---- error mapping ---------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, AuthError),
        (403, AuthError),
        (404, NotFoundError),
        (400, ValidationError),
        (422, ValidationError),
        (500, ServerError),
        (503, ServerError),
    ],
)
def test_error_status_maps_to_typed_exception(status, expected):
    handler = lambda r: httpx.Response(status, json={"detail": "nope"})
    with Client("http://x", api_key=API_KEY, transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(expected) as exc:
            c.recall(USER_ID, "anything")
        assert exc.value.status_code == status


# ---- admin wrappers --------------------------------------------------


def test_admin_create_tenant_parses_response():
    now = datetime.now(timezone.utc).isoformat()
    tenant_id = str(uuid4())
    key_id = str(uuid4())
    handler, cap = _capturing_handler(
        lambda r: httpx.Response(201, json={
            "tenant": {"id": tenant_id, "name": "acme", "status": "active", "created_at": now},
            "api_key": "mn_plaintext-once",
            "key": {
                "id": key_id, "tenant_id": tenant_id, "label": "initial",
                "created_at": now, "last_used_at": None, "revoked_at": None,
            },
        })
    )
    with Client("http://x", api_key="master", transport=httpx.MockTransport(handler)) as c:
        out = c.admin.create_tenant("acme")

    assert cap["request"].url.path == "/admin/tenants"
    assert out.tenant.name == "acme"
    assert out.api_key == "mn_plaintext-once"


# ---- tool dispatcher -------------------------------------------------


def test_dispatch_search_memory_maps_to_search():
    handler, cap = _capturing_handler(
        lambda r: httpx.Response(200, json={
            "query": "where", "memories": [],
            "total_estimated_tokens": 0, "truncated": False,
        })
    )
    with Client("http://x", api_key=API_KEY, transport=httpx.MockTransport(handler)) as c:
        result = dispatch_tool_call(
            c, user_id=USER_ID, name="search_memory",
            tool_input={"query": "where do I live"},
        )

    assert cap["request"].url.path == "/v1/memory/search"
    assert _json_body(cap["request"])["user_id"] == str(USER_ID)
    assert result == {"query": "where", "memories": [], "total_estimated_tokens": 0, "truncated": False}


def test_dispatch_save_memory_wraps_into_list():
    handler, cap = _capturing_handler(
        lambda r: httpx.Response(200, json={
            "written_ids": [str(uuid4())], "contradictions_flagged": [],
        })
    )
    with Client("http://x", api_key=API_KEY, transport=httpx.MockTransport(handler)) as c:
        dispatch_tool_call(
            c, user_id=USER_ID, name="save_memory",
            tool_input={
                "content": "user lives in Berlin",
                "type": "semantic",
                "attribute_slot": "location",
                "context_tags": ["biographical"],
            },
        )

    body = _json_body(cap["request"])
    assert body["memories"][0]["content"] == "user lives in Berlin"
    assert body["memories"][0]["attribute_slot"] == "location"


def test_dispatch_unknown_tool_raises():
    handler = lambda r: httpx.Response(200, json={})
    with Client("http://x", transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(UnknownToolError):
            dispatch_tool_call(c, user_id=USER_ID, name="nuke_everything", tool_input={})


# ---- async surface ---------------------------------------------------


@pytest.mark.asyncio
async def test_async_client_search_and_dispatcher():
    handler, cap = _capturing_handler(
        lambda r: httpx.Response(200, json={
            "query": "hi", "memories": [],
            "total_estimated_tokens": 0, "truncated": False,
        })
    )
    async with AsyncClient(
        "http://x", api_key=API_KEY,
        transport=httpx.MockTransport(handler),
    ) as c:
        result = await async_dispatch_tool_call(
            c, user_id=USER_ID, name="search_memory",
            tool_input={"query": "hi"},
        )

    assert cap["request"].url.path == "/v1/memory/search"
    assert cap["request"].headers["Authorization"] == f"Bearer {API_KEY}"
    assert result["query"] == "hi"
