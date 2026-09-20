"""Record/replay wrappers around the engine LLM client.

The golden harness needs byte-identical responses across runs. Live LLM
calls can't give that — even at temperature=0 providers drift — so we
record every request/response once against the real API and replay from
a fixture afterwards. The golden test then exercises *orchestration*
(call ordering, session mutation, rollback, response assembly) rather
than the model's mood on a given afternoon.

Two details that are easy to get wrong and cost a re-record if you do:

1. Embeddings are rounded *during recording*, before the vector is handed
   back to the engine. Rounding afterwards would mean the live pass
   retrieved on full-precision vectors while replay retrieves on rounded
   ones — different neighbours, different prompts, and every downstream
   key misses.

2. Keys are content hashes of the normalized request, and normalization
   strips absolute timestamps and UUIDs. Both change on every run without
   changing meaning (the pipeline stamps `datetime.now()` into the
   extraction prompt; memory ids are fresh on every seed).

The full normalized request is stored next to the response so a miss can
show you *what* changed instead of just a hash that didn't match.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Matches anything a ranking could plausibly turn on; 6dp is far below it.
EMBED_PRECISION = 6

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
# ISO-8601-ish: 2026-08-28, 2026-08-28T14:03:11, with optional zone.
_TS_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)


def normalize_text(text: str) -> str:
    """Collapse run-varying tokens so the same logical request hashes alike."""
    return _TS_RE.sub("<TS>", _UUID_RE.sub("<UUID>", text))


def _request_text(method: str, payload: dict[str, Any]) -> str:
    blob = json.dumps(
        {"method": method, **payload}, sort_keys=True, ensure_ascii=False, default=str
    )
    return normalize_text(blob)


def _key(request_text: str) -> str:
    return hashlib.sha256(request_text.encode("utf-8")).hexdigest()[:32]


def _round(vectors: list[list[float]]) -> list[list[float]]:
    return [[round(f, EMBED_PRECISION) for f in v] for v in vectors]


class RecordingLLM:
    """Delegates to a real client, tee-ing every exchange into a cassette.

    Duplicate keys are appended in call order, so a prompt issued twice
    with different answers replays in the same sequence.
    """

    def __init__(self, inner):
        self._inner = inner
        self.entries: dict[str, dict[str, Any]] = {}

    @property
    def _default_model(self) -> str:
        return self._inner._default_model

    def _record(self, method: str, payload: dict[str, Any], response: Any) -> None:
        req = _request_text(method, payload)
        slot = self.entries.setdefault(
            _key(req), {"method": method, "request": req, "responses": []}
        )
        slot["responses"].append(response)

    def with_model(self, model: str | None):
        self._inner.with_model(model)
        return self

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        model: str | None = None,
    ) -> str:
        out = await self._inner.complete(
            prompt, system_prompt, temperature, max_tokens, model
        )
        self._record(
            "complete",
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": model or self._inner._default_model,
            },
            out,
        )
        return out

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Round before returning: the engine must see exactly what replay
        # will later serve, or retrieval diverges between the two passes.
        out = _round(await self._inner.embed_batch(texts))
        self._record("embed_batch", {"texts": texts}, out)
        return out

    def dump(self) -> dict[str, Any]:
        """Cassette plus the model it was recorded against.

        The model name is part of every request key, so replay has to know
        which one was live or it misses on the very first prompt.
        """
        return {"default_model": self._inner._default_model, "entries": self.entries}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.dump(), indent=1, sort_keys=True), encoding="utf-8"
        )


class CassetteMiss(KeyError):
    """Raised when replay sees a request that was never recorded."""


class ReplayLLM:
    """Serves recorded responses. No network, no API key, no cost."""

    def __init__(self, cassette: dict[str, Any]):
        self._default_model = cassette["default_model"]
        self._entries = {
            k: {**v, "responses": list(v["responses"])}
            for k, v in cassette["entries"].items()
        }
        self._cursor: dict[str, int] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> "ReplayLLM":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_model(self, model: str | None):
        if model:
            self._default_model = model
        return self

    def _explain_miss(self, method: str, request: str) -> str:
        """Show the closest recorded request and the first line that differs.

        A bare hash mismatch tells you nothing actionable; this points at
        the prompt fragment that drifted.
        """
        candidates = [
            (k, v) for k, v in self._entries.items() if v["method"] == method
        ]
        if not candidates:
            return "  (nothing of this method recorded at all)"
        best_key, best = max(
            candidates,
            key=lambda kv: difflib.SequenceMatcher(
                None, kv[1]["request"], request
            ).ratio(),
        )
        ratio = difflib.SequenceMatcher(None, best["request"], request).ratio()
        diff = list(
            difflib.unified_diff(
                best["request"].splitlines(),
                request.splitlines(),
                fromfile=f"recorded[{best_key[:8]}]",
                tofile="requested",
                lineterm="",
                n=1,
            )
        )
        return (
            f"  closest recorded request: {ratio:.1%} similar\n"
            + "\n".join(f"  {line}" for line in diff[:40])
        )

    def _take(self, method: str, request: str):
        entry = self._entries.get(_key(request))
        if entry is None:
            raise CassetteMiss(
                f"No recorded {method} response for this request. The prompt "
                f"changed since the cassette was recorded — re-record with "
                f"scripts/record_golden.py.\n"
                + self._explain_miss(method, request)
            )
        k = _key(request)
        i = self._cursor.get(k, 0)
        self._cursor[k] = i + 1
        responses = entry["responses"]
        # Last recorded answer repeats if replay asks more times than we recorded.
        return responses[min(i, len(responses) - 1)]

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        model: str | None = None,
    ) -> str:
        return self._take(
            "complete",
            _request_text(
                "complete",
                {
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "model": model or self._default_model,
                },
            ),
        )

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._take("embed_batch", _request_text("embed_batch", {"texts": texts}))
