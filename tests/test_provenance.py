"""Unit tests for evidence vs inference provenance (Change 5).

Verifies that format_for_injection() correctly splits semantic memories
into Known Facts vs Inferences based on source type.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from memonative.engine.retrieval import format_for_injection, ScoredMemory


def _make_memory(
    content: str,
    memory_type: str = "semantic",
    source: str = "user_stated",
    trust_score: float = 0.8,
    attribute_slot: str | None = "general",
    context_tags: list | None = None,
):
    mem = MagicMock()
    mem.id = uuid4()
    mem.content = content
    mem.memory_type = memory_type
    mem.source = MagicMock()
    mem.source.value = source
    mem.trust_score = trust_score
    mem.attribute_slot = attribute_slot
    mem.context_tags = context_tags or []
    mem.tenant_id = uuid4()
    mem.user_id = uuid4()
    mem.created_at = datetime(2025, 6, 1, tzinfo=timezone.utc)
    return mem


@pytest.mark.asyncio
async def test_user_stated_renders_under_known_facts():
    mem = _make_memory("Lives in Sydney", source="user_stated")
    scored = [ScoredMemory(memory=mem, score=0.9)]
    output = await format_for_injection(scored)
    assert "## Known Facts" in output
    assert "[Fact" in output
    assert "## Inferences" not in output


@pytest.mark.asyncio
async def test_system_inferred_renders_under_inferences():
    mem = _make_memory("Probably likes coffee", source="system_inferred", trust_score=0.4)
    scored = [ScoredMemory(memory=mem, score=0.5)]
    output = await format_for_injection(scored)
    assert "## Inferences (unconfirmed)" in output
    assert "[Inference" in output
    assert "## Known Facts" not in output


@pytest.mark.asyncio
async def test_consolidated_renders_under_known_facts():
    mem = _make_memory("Has three cats", source="consolidated")
    scored = [ScoredMemory(memory=mem, score=0.8)]
    output = await format_for_injection(scored)
    assert "## Known Facts" in output
    assert "[Fact" in output


@pytest.mark.asyncio
async def test_user_repeated_renders_under_known_facts():
    mem = _make_memory("Prefers Python", source="user_repeated")
    scored = [ScoredMemory(memory=mem, score=0.9)]
    output = await format_for_injection(scored)
    assert "## Known Facts" in output


@pytest.mark.asyncio
async def test_mixed_provenance_produces_both_sections():
    fact = _make_memory("Lives in Sydney", source="user_stated")
    inference = _make_memory("Might enjoy surfing", source="system_inferred", trust_score=0.3)
    scored = [
        ScoredMemory(memory=fact, score=0.9),
        ScoredMemory(memory=inference, score=0.4),
    ]
    output = await format_for_injection(scored)
    assert "## Known Facts" in output
    assert "## Inferences (unconfirmed)" in output
    assert "Lives in Sydney" in output.split("## Known Facts")[1].split("##")[0]
    assert "Might enjoy surfing" in output.split("## Inferences")[1]


@pytest.mark.asyncio
async def test_trust_score_shown_in_inferences():
    mem = _make_memory("Likes hiking", source="system_inferred", trust_score=0.35)
    scored = [ScoredMemory(memory=mem, score=0.5)]
    output = await format_for_injection(scored)
    assert "(trust: 0.4)" in output or "(trust: 0.3)" in output


@pytest.mark.asyncio
async def test_episodic_still_in_recent_context():
    mem = _make_memory("Went to the beach", memory_type="episodic")
    scored = [ScoredMemory(memory=mem, score=0.5)]
    output = await format_for_injection(scored)
    assert "## Recent Context" in output


@pytest.mark.asyncio
async def test_procedural_still_in_preferences():
    mem = _make_memory("Be concise", memory_type="procedural")
    scored = [ScoredMemory(memory=mem, score=0.5)]
    output = await format_for_injection(scored)
    assert "## Interaction Preferences" in output


@pytest.mark.asyncio
async def test_dedup_across_provenance():
    """Same content from different sources should only appear once."""
    mem1 = _make_memory("Lives in Sydney", source="user_stated")
    mem2 = _make_memory("Lives in Sydney", source="system_inferred")
    scored = [
        ScoredMemory(memory=mem1, score=0.9),
        ScoredMemory(memory=mem2, score=0.5),
    ]
    output = await format_for_injection(scored)
    assert output.count("Lives in Sydney") == 1
