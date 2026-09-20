"""HTTP-surface integration tests for the agent-native primitives.

Exercises the real ASGI app through TestClient: routing, middleware,
auth gating, tool-manifest endpoint, and request validation for the
/v1/memory/* routes. Does not require Postgres/Redis/OpenAI — routes
that hit the DB/LLM are covered via payload validation (422 before any
handler runs) rather than e2e calls.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from memonative.config import settings


def _fresh_app():
    """Reload the app so the API_KEY value picked up by the auth
    dependency reflects whatever the test just set on `settings`."""
    from memonative import main
    importlib.reload(main)
    return main.app


@pytest.fixture
def client():
    # MASTER_ENCRYPTION_KEY is cleared, not just API_KEY: when it is set, the
    # `get_llm` dependency resolves tenant credentials out of Postgres, and
    # FastAPI runs dependencies *before* it validates the request body. A
    # developer .env that sets it therefore turns the 422s asserted below into
    # connection errors, even though these tests are meant to be DB-less.
    #
    # The LLM keys are stubbed for the same reason: `get_llm` refuses to build
    # a keyless client, and that refusal would also land before validation.
    previous_master = settings.MASTER_ENCRYPTION_KEY
    previous_keys = (settings.OPENAI_API_KEY, settings.DEEPSEEK_API_KEY)
    settings.API_KEY = SecretStr("")
    settings.MASTER_ENCRYPTION_KEY = SecretStr("")
    settings.OPENAI_API_KEY = SecretStr("sk-test")
    settings.DEEPSEEK_API_KEY = SecretStr("sk-test")
    yield TestClient(_fresh_app())
    settings.MASTER_ENCRYPTION_KEY = previous_master
    settings.OPENAI_API_KEY, settings.DEEPSEEK_API_KEY = previous_keys


@pytest.fixture
def authed_client():
    settings.API_KEY = SecretStr("test-secret")
    yield TestClient(_fresh_app())
    settings.API_KEY = SecretStr("")


def test_health_is_unauthenticated(authed_client):
    # Liveness must work even when API_KEY is set, so probes don't
    # need secrets.
    r = authed_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_tool_manifest_anthropic(client):
    r = client.get("/v1/tools/anthropic")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "anthropic"
    names = {t["name"] for t in body["tools"]}
    assert names == {"search_memory", "recall_fact", "save_memory", "list_history"}
    # Anthropic schema uses input_schema, not parameters.
    for tool in body["tools"]:
        assert "input_schema" in tool
        # user_id must never be a tool parameter — see agent_tools.py.
        assert "user_id" not in tool["input_schema"].get("properties", {})


def test_tool_manifest_openai(client):
    r = client.get("/v1/tools/openai")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "openai"
    names = {t["function"]["name"] for t in body["tools"]}
    assert names == {"search_memory", "recall_fact", "save_memory", "list_history"}
    for tool in body["tools"]:
        assert tool["type"] == "function"
        props = tool["function"]["parameters"].get("properties", {})
        assert "user_id" not in props


def test_tool_manifest_deepseek(client):
    # DeepSeek's tool-calling schema is OpenAI-compatible; the adapter
    # exists for discoverability so /v1/tools/deepseek doesn't 400.
    r = client.get("/v1/tools/deepseek")
    assert r.status_code == 200
    body = r.json()
    assert body["provider"] == "deepseek"
    names = {t["function"]["name"] for t in body["tools"]}
    assert names == {"search_memory", "recall_fact", "save_memory", "list_history"}
    for tool in body["tools"]:
        assert tool["type"] == "function"
        props = tool["function"]["parameters"].get("properties", {})
        assert "user_id" not in props


def test_tool_manifest_rejects_unknown_provider(client):
    r = client.get("/v1/tools/langchain")
    assert r.status_code == 400


def test_auth_required_when_api_key_set(authed_client):
    # No bearer → 401. Raises before any DB roundtrip.
    r = authed_client.get("/v1/tools/openai")
    assert r.status_code == 401

    # Master env-var key has a string-equality shortcut to the default
    # tenant — also no DB roundtrip — so it works without Postgres.
    r = authed_client.get(
        "/v1/tools/openai",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert r.status_code == 200

    # Tenant-scoped keys go through DB lookup; that path is covered by
    # the live integration suite, not this DB-less test.


def test_memory_search_validates_payload(client):
    # Missing required fields (user_id, query) → 422 before the handler
    # touches the DB, so this runs without Postgres.
    r = client.post("/v1/memory/search", json={})
    assert r.status_code == 422

    # Unknown memory_type rejected by field_validator.
    r = client.post(
        "/v1/memory/search",
        json={
            "user_id": "00000000-0000-0000-0000-000000000000",
            "query": "hi",
            "memory_types": ["nonsense"],
        },
    )
    assert r.status_code == 422


def test_memory_write_validates_payload(client):
    r = client.post("/v1/memory/write", json={})
    assert r.status_code == 422

    # Empty memories list violates min_length=1.
    r = client.post(
        "/v1/memory/write",
        json={
            "user_id": "00000000-0000-0000-0000-000000000000",
            "memories": [],
        },
    )
    assert r.status_code == 422


def test_memory_facts_rejects_invalid_uuid(client):
    r = client.get("/v1/memory/facts/not-a-uuid/location")
    assert r.status_code == 400


def test_request_id_header_is_echoed(client):
    r = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers.get("X-Request-ID") == "abc-123"

    # Minted when absent.
    r = client.get("/health")
    assert r.headers.get("X-Request-ID")


# ---- Admin surface auth gating (DB-less; verifies require_admin only) ----

def test_admin_disabled_when_api_key_unset(client):
    # No API_KEY → /admin/* returns 503 even with a bearer token.
    r = client.get("/admin/tenants", headers={"Authorization": "Bearer anything"})
    assert r.status_code == 503

    r = client.post(
        "/admin/tenants",
        json={"name": "acme"},
        headers={"Authorization": "Bearer anything"},
    )
    assert r.status_code == 503


def test_admin_requires_master_bearer(authed_client):
    # API_KEY set but no bearer → 401.
    r = authed_client.get("/admin/tenants")
    assert r.status_code == 401

    # Wrong bearer → 403, NOT 401: distinguishes "missing" from "denied"
    # so callers know to escalate, not retry with the same token.
    r = authed_client.get(
        "/admin/tenants",
        headers={"Authorization": "Bearer not-the-master-key"},
    )
    assert r.status_code == 403

    # Tenant-scoped keys (which would resolve via DB) also can't reach
    # /admin — but that path needs Postgres; covered by live integration.


def test_admin_post_payload_validation_runs_after_auth(authed_client):
    # Master bearer + invalid body → 422 (means we got past require_admin
    # and into request validation, which is the contract we want).
    r = authed_client.post(
        "/admin/tenants",
        json={},
        headers={"Authorization": "Bearer test-secret"},
    )
    assert r.status_code == 422
