"""Cost tracking for the eval — counts OpenAI tokens across BOTH adapters.

Patches `openai.resources.chat.completions.Completions.create` and the
async + embeddings equivalents at module level. Every call records its
`response.usage` against a shared `CostTracker`.

Why the patch instead of wrapping each adapter's client:
  - mem0 instantiates its own openai.OpenAI internally and we don't get
    a hook into it without forking mem0
  - Memonative's LLMClient could be wrapped, but then the tracker would
    only see one side and we'd have no apples-to-apples spend visibility

Token prices below are OpenAI's published `gpt-4o-mini` and
`text-embedding-3-small` rates as of 2026-04. Cost estimates are
order-of-magnitude correct, not penny-perfect — good enough for a
$15 kill-switch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# USD per token. Source: OpenAI pricing page, 2026-04.
# DeepSeek entries are placeholders (zero) so calls register call/token
# counts without inflating the $15 kill-switch — DeepSeek pricing varies
# and we're optimising for benchmark accuracy, not exact dollar tracking.
_PRICING = {
    "gpt-4o-mini":             (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4o":                  (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-5-mini":              (0.25 / 1_000_000, 2.00 / 1_000_000),
    "text-embedding-3-small":  (0.02 / 1_000_000, 0.0),
    "text-embedding-3-large":  (0.13 / 1_000_000, 0.0),
    "deepseek-v4-flash":       (0.0, 0.0),
    "deepseek-v4-pro":         (0.0, 0.0),
}


@dataclass
class ModelStats:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class CostTracker:
    by_model: dict[str, ModelStats] = field(default_factory=dict)
    total_cost: float = 0.0

    def _stats(self, model: str) -> ModelStats:
        if model not in self.by_model:
            self.by_model[model] = ModelStats()
        return self.by_model[model]

    def record(self, model: str | None, usage) -> None:
        if not model or usage is None:
            return
        # OpenAI usage objects expose prompt_tokens / completion_tokens
        # for chat, total_tokens for embeddings. Cover both.
        prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
        if prompt == 0 and completion == 0:
            prompt = getattr(usage, "total_tokens", 0) or 0

        in_rate, out_rate = _PRICING.get(model, (0.0, 0.0))
        cost = prompt * in_rate + completion * out_rate

        s = self._stats(model)
        s.calls += 1
        s.input_tokens += prompt
        s.output_tokens += completion
        s.cost += cost

        self.total_cost += cost


_TRACKER = CostTracker()
_INSTALLED = False


def get_tracker() -> CostTracker:
    return _TRACKER


def install_tracker() -> None:
    """Patch openai SDK so every chat/embed call updates the tracker.

    Idempotent — second call is a no-op. Must be called BEFORE any
    adapter constructs its OpenAI client; the patch is on class methods,
    so existing instances pick it up too, but this lets the patch be
    obvious at the entry point.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from openai.resources.chat.completions import (
        Completions, AsyncCompletions,
    )
    from openai.resources.embeddings import Embeddings, AsyncEmbeddings

    _orig_chat = Completions.create
    _orig_async_chat = AsyncCompletions.create
    _orig_embed = Embeddings.create
    _orig_async_embed = AsyncEmbeddings.create

    def _patched_chat(self, *args, **kwargs):
        resp = _orig_chat(self, *args, **kwargs)
        _TRACKER.record(kwargs.get("model"), getattr(resp, "usage", None))
        return resp

    async def _patched_async_chat(self, *args, **kwargs):
        resp = await _orig_async_chat(self, *args, **kwargs)
        _TRACKER.record(kwargs.get("model"), getattr(resp, "usage", None))
        return resp

    def _patched_embed(self, *args, **kwargs):
        resp = _orig_embed(self, *args, **kwargs)
        _TRACKER.record(kwargs.get("model"), getattr(resp, "usage", None))
        return resp

    async def _patched_async_embed(self, *args, **kwargs):
        resp = await _orig_async_embed(self, *args, **kwargs)
        _TRACKER.record(kwargs.get("model"), getattr(resp, "usage", None))
        return resp

    Completions.create = _patched_chat
    AsyncCompletions.create = _patched_async_chat
    Embeddings.create = _patched_embed
    AsyncEmbeddings.create = _patched_async_embed

    _INSTALLED = True
