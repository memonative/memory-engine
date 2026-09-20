"""The LLM client's failure paths, which the golden cassette never reaches.

The cassette only ever replays successful responses, so the eight error
branches in `memonative.llm` are invisible to the characterization test. They
used to raise `fastapi.HTTPException`; they now raise `EngineLLMError`, and
the status code and message have to survive that move unchanged or the HTTP
contract quietly shifts.
"""

from __future__ import annotations

import openai
import pytest
from pydantic import SecretStr

from memonative.config import settings
from memonative.errors import EngineError, EngineLLMError
from memonative.llm import LLMClient
import memonative.llm as llm_module


@pytest.fixture(autouse=True)
def _stub_llm_keys():
    """Hand the openai SDK a non-empty key for every test in this module.

    These tests exercise error mapping and never reach the network, so the
    key's value is irrelevant — but openai 2.5+ refuses to *construct* a
    client with an empty `api_key`, which lands before the code under test
    runs. Older releases allowed it, so leaving the keys unset makes the
    module pass or fail depending on which SDK version the environment
    resolved, which is how this first reached CI green locally and red on a
    clean machine. Stubbing here keeps it version-independent.
    """
    previous = (settings.OPENAI_API_KEY, settings.DEEPSEEK_API_KEY)
    settings.OPENAI_API_KEY = SecretStr("sk-test")
    settings.DEEPSEEK_API_KEY = SecretStr("sk-test")
    yield
    settings.OPENAI_API_KEY, settings.DEEPSEEK_API_KEY = previous


def _openai_error(cls: type, **attrs):
    """Build an openai exception without satisfying its constructor.

    These take `response`/`body` objects we have no reason to fabricate;
    only the type (and `status_code`, for `APIError`) affects our branching.
    """
    exc = cls.__new__(cls)
    for k, v in attrs.items():
        setattr(exc, k, v)
    return exc


CASES = [
    (openai.RateLimitError, {}, 429, "quota/rate limit exceeded"),
    (openai.AuthenticationError, {}, 502, "authentication failed"),
    (openai.APIError, {"status_code": 503}, 502, "(status 503)"),
    (ValueError, {}, 500, "Internal"),
]


@pytest.fixture
def raising(monkeypatch):
    """Make the retry wrapper fail, so every attempt lands in the handler."""

    def _install(exc):
        async def _boom(call, **kwargs):
            raise exc

        monkeypatch.setattr(llm_module, "_with_retry", _boom)

    return _install


@pytest.mark.parametrize("exc_type,attrs,status,fragment", CASES)
@pytest.mark.asyncio
async def test_complete_maps_failures(raising, exc_type, attrs, status, fragment):
    raising(_openai_error(exc_type, **attrs))

    with pytest.raises(EngineLLMError) as caught:
        await LLMClient().complete("anything")

    assert caught.value.status_code == status
    assert fragment in caught.value.detail
    assert "Engine API" in caught.value.detail or status == 500


@pytest.mark.parametrize("exc_type,attrs,status,fragment", CASES)
@pytest.mark.asyncio
async def test_embed_batch_maps_failures(raising, exc_type, attrs, status, fragment):
    raising(_openai_error(exc_type, **attrs))

    with pytest.raises(EngineLLMError) as caught:
        await LLMClient().embed_batch(["anything"])

    assert caught.value.status_code == status
    assert fragment in caught.value.detail
    assert "Embedding" in caught.value.detail or status == 500


@pytest.mark.asyncio
async def test_empty_embed_batch_makes_no_call():
    assert await LLMClient().embed_batch([]) == []


def test_engine_llm_error_is_an_engine_error():
    # The API layer catches EngineError once; LLM failures must be caught by it.
    assert issubclass(EngineLLMError, EngineError)


def test_tenant_embedding_key_is_used():
    # Regression: the parameter was accepted and then ignored, so every
    # tenant's embeddings were billed to the platform key.
    assert LLMClient(embedding_api_key="sk-tenant").embed_client.api_key == "sk-tenant"


def test_embedding_key_falls_back_to_platform():
    client = LLMClient()
    assert client.embed_client.api_key == settings.OPENAI_API_KEY.get_secret_value()
