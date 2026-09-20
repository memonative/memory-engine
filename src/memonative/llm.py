"""LLM and embedding client for engine-internal work.

Lives outside `api/` so the engine can be imported in-process without a web
framework installed. Failures raise `EngineLLMError` carrying the status
code the old `HTTPException` carried; the API layer maps it back.
"""

import asyncio
import random
from typing import Protocol, runtime_checkable

import httpx
import openai
from openai import AsyncOpenAI

from memonative.config import settings
from memonative.errors import ConfigurationError, EngineLLMError


def require_env_llm_keys() -> None:
    """Fail loudly when the engine is about to build a keyless client.

    Callers that fall back to the environment must go through this first.
    Without it the openai SDK raises its own error deep in the request —
    the operator sees a bare 500 and the real cause only in the logs, which
    is the most common way a fresh deployment goes wrong.
    """
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", settings.OPENAI_API_KEY.get_secret_value()),
            ("DEEPSEEK_API_KEY", settings.DEEPSEEK_API_KEY.get_secret_value()),
        )
        if not value
    ]
    if missing:
        raise ConfigurationError(
            f"Missing {' and '.join(missing)}. Copy .env.example to .env, add "
            "your keys, and restart. OPENAI_API_KEY is used for embeddings and "
            "DEEPSEEK_API_KEY for the engine LLM."
        )

_LLM_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


# Errors worth retrying — transient infra/upstream blips. RateLimit is
# included since OpenAI sometimes returns it for short bursts.
_RETRY_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.RateLimitError,
)


async def _with_retry(call, *, max_attempts: int = 3, base_delay: float = 0.5):
    """Run an awaitable with exponential backoff on transient OpenAI errors.

    Re-raises non-retryable errors immediately. Final attempt's exception
    bubbles up unchanged so the caller's mapping (HTTPException, etc.)
    still applies.
    """
    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return await call()
        except _RETRY_ERRORS:
            if attempt == max_attempts:
                raise
            # Jitter avoids synchronized retries from concurrent requests
            await asyncio.sleep(delay + random.uniform(0, delay / 2))
            delay *= 2


@runtime_checkable
class LLMProtocol(Protocol):
    """What the engine actually needs from an LLM client.

    Narrow on purpose: four methods is the whole surface every engine module
    touches. Anything satisfying this can be passed to `MemoryEngine` — the
    record/replay wrappers in `tests/support/llm_capture.py` already do, and
    a caller wanting a different provider only has to implement these.
    """

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        model: str | None = None,
    ) -> str: ...

    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    def with_model(self, model: str | None) -> "LLMProtocol": ...


class LLMClient:
    """Wrapper for engine-internal LLM tasks.

    Two providers backstop two responsibilities:
      - DeepSeek (OpenAI-compatible API) handles `complete()` for
        extraction, consensus, edges, consolidation, reconsolidation,
        and the timeline classifier — all structured-JSON callers.
      - OpenAI handles `embed()` because DeepSeek doesn't offer an
        embeddings endpoint and pgvector indexes are dimension-locked
        to text-embedding-3-small.

    Memonative does NOT generate the customer's chat reply; the
    customer's own LLM does that downstream with the memories we
    return (see routes.py /v1/process: `response` field on Interaction
    is null).

    `complete()` defaults to temperature=0 because every internal use
    site wants the same input to produce the same JSON output. Callers
    that want sampling variance can pass `temperature=...` explicitly.
    """

    DEFAULT_MODEL = "deepseek-v4-flash"

    def __init__(
        self,
        engine_api_key: str | None = None,
        engine_base_url: str | None = None,
        engine_model: str | None = None,
        embedding_api_key: str | None = None,
    ):
        self.complete_client = AsyncOpenAI(
            api_key=engine_api_key or settings.DEEPSEEK_API_KEY.get_secret_value(),
            base_url=engine_base_url or settings.DEEPSEEK_BASE_URL,
            timeout=_LLM_TIMEOUT,
        )
        self._default_model = engine_model or self.DEFAULT_MODEL
        # A tenant that brings its own embedding key gets billed for its own
        # embeddings; otherwise the platform key absorbs the cost. The
        # parameter was accepted and then ignored here, so every tenant hit
        # the platform key regardless of what they configured.
        self.embed_client = AsyncOpenAI(
            api_key=embedding_api_key or settings.OPENAI_API_KEY.get_secret_value(),
            timeout=_LLM_TIMEOUT,
        )

    @classmethod
    def from_credentials(cls, creds) -> "LLMClient":
        """Construct from a ResolvedCredentials dataclass."""
        return cls(
            engine_api_key=creds.engine_api_key,
            engine_base_url=creds.engine_base_url,
            engine_model=creds.engine_model,
            embedding_api_key=creds.embedding_api_key,
        )

    def with_model(self, model: str | None) -> "LLMClient":
        """Return self with a per-request model override. No-op if model is None."""
        if model:
            self._default_model = model
        return self

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        model: str | None = None,
    ) -> str:
        # Default temperature is 0 because every engine caller produces
        # structured JSON (extraction, consensus, reconsolidation, edges,
        # consolidation) — creative variation only adds run-to-run noise.
        # The API default of 1.0 was causing extraction to miss facts on
        # rerun (e.g. 778164c6 lost "Grilled Snapper with Mango Salsa"
        # between runs of the same input).
        #
        # max_tokens defaults to 16384 — the documented ceiling for
        # DeepSeek v4-flash in standard (non-thinking) mode. Long
        # ChatGPT-style sessions with many list-item memories were
        # truncating mid-JSON at the previous 8K cap; silent truncation
        # drops the whole session because the parser fails. 16K gives
        # ~2x headroom over the worst case we've observed.
        resolved_model = model or self._default_model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async def _call():
            return await self.complete_client.chat.completions.create(
                model=resolved_model,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # DeepSeek defaults thinking ON for v4-flash, which (a)
                # silently ignores `temperature` (so extraction loses the
                # determinism we explicitly added after 778164c6 dropped
                # "Grilled Snapper" on rerun), and (b) burns thinking
                # tokens on what are really pattern-matching tasks
                # (extraction, consensus, edges, consolidation,
                # reconsolidation, timeline classifier). Disable.
                extra_body={"thinking": {"type": "disabled"}},
            )

        try:
            response = await _with_retry(_call)
            return response.choices[0].message.content
        except openai.RateLimitError:
            raise EngineLLMError(
                "Engine API quota/rate limit exceeded.", status_code=429
            )
        except openai.AuthenticationError:
            raise EngineLLMError(
                "Engine API authentication failed. Check configured credentials.",
                status_code=502,
            )
        except openai.APIError as e:
            raise EngineLLMError(
                f"Engine API error (status {e.status_code}).", status_code=502
            )
        except Exception:
            raise EngineLLMError("Internal LLM processing error.", status_code=500)

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async def _call():
            return await self.embed_client.embeddings.create(
                input=texts,
                model="text-embedding-3-small",
            )

        try:
            response = await _with_retry(_call)
            # OpenAI returns data in input order
            return [item.embedding for item in response.data]
        except openai.RateLimitError:
            raise EngineLLMError(
                "Embedding API quota/rate limit exceeded.", status_code=429
            )
        except openai.AuthenticationError:
            raise EngineLLMError(
                "Embedding API authentication failed. Check configured credentials.",
                status_code=502,
            )
        except openai.APIError as e:
            raise EngineLLMError(
                f"Embedding API error (status {e.status_code}).", status_code=502
            )
        except Exception:
            raise EngineLLMError(
                "Internal embedding processing error.", status_code=500
            )
