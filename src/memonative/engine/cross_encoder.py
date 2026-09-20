"""Cross-encoder reranking for retrieved memories.

Applies a neural cross-encoder after RRF fusion + decay scoring to
reorder candidates by query-passage relevance. The existing
decay/trust/salience weighting is preserved as a tiebreaker — decay-aware
ranking is a differentiator we don't want to lose.

Two backends:
  - LOCAL: sentence-transformers CrossEncoder, runs on CPU via
    asyncio.to_thread() so FastAPI stays non-blocking.
  - API: hosted reranker (Cohere, Jina, Voyage) for cloud deployments
    that don't want to ship a model.

Contract: REORDER ONLY. The reranker never reduces how many memories the
caller receives. On any failure, falls back to original score order.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memonative.engine.retrieval import ScoredMemory

logger = logging.getLogger(__name__)

_cross_encoder = None
_cross_encoder_model_name: str | None = None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


try:
    import importlib
    importlib.import_module("sentence_transformers")
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


def _get_cross_encoder(model_name: str):
    """Lazy-load the CrossEncoder singleton. Thread-safe via GIL."""
    global _cross_encoder, _cross_encoder_model_name
    if _cross_encoder is not None and _cross_encoder_model_name == model_name:
        return _cross_encoder
    from sentence_transformers import CrossEncoder as _CE
    logger.info("Loading cross-encoder model: %s", model_name)
    _cross_encoder = _CE(model_name)
    _cross_encoder_model_name = model_name
    return _cross_encoder


def _score_pairs_local(
    model_name: str,
    query: str,
    passages: list[str],
) -> list[float]:
    """Synchronous cross-encoder scoring (runs in thread)."""
    ce = _get_cross_encoder(model_name)
    pairs = [[query, p] for p in passages]
    scores = ce.predict(pairs)
    return [float(s) for s in scores]


async def _score_pairs_api(
    provider: str,
    api_key: str,
    query: str,
    passages: list[str],
    model: str | None = None,
) -> list[float]:
    """Score via hosted reranker API (Cohere, Jina, Voyage)."""
    import httpx

    if provider == "cohere":
        url = "https://api.cohere.com/v2/rerank"
        body = {
            "model": model or "rerank-v3.5",
            "query": query,
            "documents": passages,
            "top_n": len(passages),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        scores = [0.0] * len(passages)
        for result in data["results"]:
            scores[result["index"]] = result["relevance_score"]
        return scores

    elif provider == "jina":
        url = "https://api.jina.ai/v1/rerank"
        body = {
            "model": model or "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": passages,
            "top_n": len(passages),
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        scores = [0.0] * len(passages)
        for result in data["results"]:
            scores[result["index"]] = result["relevance_score"]
        return scores

    else:
        raise ValueError(f"Unsupported rerank provider: {provider}")


async def cross_encoder_rerank(
    query: str,
    candidates: list["ScoredMemory"],
    top_k: int,
    *,
    backend: str = "local",
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    api_key: str | None = None,
    api_provider: str | None = None,
    api_model: str | None = None,
    blend_weight: float = 0.7,
    max_candidates: int = 30,
) -> list["ScoredMemory"]:
    """Rerank candidates using a cross-encoder, then take top_k.

    Returns exactly min(top_k, len(candidates)) items. On any failure,
    returns the original score-ordered slice unchanged.

    blend_weight controls the mix: higher = more cross-encoder influence.
      rerank_score = blend_weight * sigmoid(ce_score) + (1 - blend_weight) * final_score
    """
    if not candidates or top_k <= 0:
        return candidates[:top_k]

    if backend == "local" and not _ST_AVAILABLE:
        logger.debug("sentence-transformers not installed — skipping rerank")
        return candidates[:top_k]

    pool = candidates[:max_candidates]
    remainder = candidates[max_candidates:]

    passages = [sm.memory.content for sm in pool]

    try:
        if backend == "api" and api_key and api_provider:
            ce_scores = await _score_pairs_api(
                api_provider, api_key, query, passages, model=api_model
            )
        else:
            ce_scores = await asyncio.to_thread(
                _score_pairs_local, model_name, query, passages
            )

        blended = []
        for sm, ce_score in zip(pool, ce_scores):
            rerank_score = (
                blend_weight * _sigmoid(ce_score)
                + (1 - blend_weight) * sm.score
            )
            blended.append((sm, rerank_score))

        blended.sort(key=lambda pair: pair[1], reverse=True)
        reranked = [sm for sm, _ in blended]

        result = reranked + remainder
        return result[:top_k]

    except Exception:
        logger.warning("Cross-encoder rerank failed — falling back to original order", exc_info=True)
        return candidates[:top_k]
