"""Unit tests for cross-encoder reranking."""
import math
import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from memonative.engine.cross_encoder import cross_encoder_rerank, _sigmoid


def _make_scored(content: str, score: float):
    from memonative.engine.retrieval import ScoredMemory
    mem = MagicMock()
    mem.id = uuid4()
    mem.content = content
    return ScoredMemory(memory=mem, score=score)


def test_sigmoid_midpoint():
    assert _sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_positive():
    assert _sigmoid(5.0) > 0.99


def test_sigmoid_negative():
    assert _sigmoid(-5.0) < 0.01


@pytest.mark.asyncio
async def test_skips_when_library_unavailable(monkeypatch):
    """When sentence-transformers is not installed, returns original order."""
    monkeypatch.setattr("memonative.engine.cross_encoder._ST_AVAILABLE", False)
    candidates = [
        _make_scored("first", 0.9),
        _make_scored("second", 0.7),
    ]
    result = await cross_encoder_rerank("query", candidates, top_k=2, backend="local")
    assert len(result) == 2
    assert result[0].memory.content == "first"


@pytest.mark.asyncio
async def test_empty_candidates_returns_empty():
    result = await cross_encoder_rerank("query", [], top_k=5, backend="local")
    assert result == []


@pytest.mark.asyncio
async def test_single_candidate_returns_it():
    c = _make_scored("only one", 0.9)
    result = await cross_encoder_rerank("query", [c], top_k=5, backend="local")
    assert len(result) == 1
    assert result[0] is c


@pytest.mark.asyncio
async def test_fallback_on_failure(monkeypatch):
    """On model load failure, returns original order unchanged."""
    monkeypatch.setattr("memonative.engine.cross_encoder._ST_AVAILABLE", True)

    def _fail(*a, **kw):
        raise RuntimeError("model not found")

    monkeypatch.setattr(
        "memonative.engine.cross_encoder._score_pairs_local", _fail
    )

    candidates = [
        _make_scored("first", 0.9),
        _make_scored("second", 0.7),
        _make_scored("third", 0.5),
    ]
    result = await cross_encoder_rerank(
        "query", candidates, top_k=3, backend="local"
    )
    assert len(result) == 3
    assert result[0].memory.content == "first"
    assert result[1].memory.content == "second"
    assert result[2].memory.content == "third"


@pytest.mark.asyncio
async def test_reranker_reorders_by_ce_score(monkeypatch):
    """Cross-encoder scores override original order."""
    monkeypatch.setattr("memonative.engine.cross_encoder._ST_AVAILABLE", True)

    def _mock_score(model_name, query, passages):
        scores = []
        for p in passages:
            if "gold" in p:
                scores.append(5.0)
            else:
                scores.append(-2.0)
        return scores

    monkeypatch.setattr(
        "memonative.engine.cross_encoder._score_pairs_local", _mock_score
    )

    candidates = [
        _make_scored("noise first", 0.95),
        _make_scored("noise second", 0.90),
        _make_scored("the gold answer", 0.10),
    ]
    result = await cross_encoder_rerank(
        "what is the answer?", candidates, top_k=3, backend="local"
    )
    assert result[0].memory.content == "the gold answer"


@pytest.mark.asyncio
async def test_count_never_reduced(monkeypatch):
    """Reranker must not reduce returned count vs rerank-off."""
    monkeypatch.setattr("memonative.engine.cross_encoder._ST_AVAILABLE", True)

    def _mock_score(model_name, query, passages):
        return [0.0] * len(passages)

    monkeypatch.setattr(
        "memonative.engine.cross_encoder._score_pairs_local", _mock_score
    )

    candidates = [_make_scored(f"mem {i}", 1.0 - i * 0.1) for i in range(10)]
    result = await cross_encoder_rerank(
        "query", candidates, top_k=5, backend="local"
    )
    assert len(result) == 5


@pytest.mark.asyncio
async def test_top_k_respected(monkeypatch):
    """top_k limits the output even when more candidates exist."""
    monkeypatch.setattr("memonative.engine.cross_encoder._ST_AVAILABLE", True)

    def _mock_score(model_name, query, passages):
        return [1.0] * len(passages)

    monkeypatch.setattr(
        "memonative.engine.cross_encoder._score_pairs_local", _mock_score
    )

    candidates = [_make_scored(f"mem {i}", 0.5) for i in range(20)]
    result = await cross_encoder_rerank(
        "query", candidates, top_k=3, backend="local"
    )
    assert len(result) == 3
