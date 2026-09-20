"""Unit tests for causal edge detection (Change 3).

Tests the LLM prompt parsing and edge creation logic without a database
by mocking the async session and LLM client.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID

from memonative.engine.edges import create_causal_edges


def _make_memory(content: str, tenant_id: UUID, user_id: UUID, mem_id: UUID | None = None):
    mem = MagicMock()
    mem.id = mem_id or uuid4()
    mem.content = content
    mem.tenant_id = tenant_id
    mem.user_id = user_id
    mem.memory_type = "semantic"
    mem.decay_state = "active"
    mem.embedding = [0.1] * 10
    return mem


@pytest.mark.asyncio
async def test_no_candidates_returns_empty():
    """No neighbours → no causal edges."""
    db = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    db.execute.return_value = result_mock

    tid = uuid4()
    uid = uuid4()
    new_mem = _make_memory("started keto diet", tid, uid)

    result = await create_causal_edges(db, tid, uid, new_mem, llm_client=AsyncMock())
    assert result == []


@pytest.mark.asyncio
async def test_no_embedding_returns_empty():
    """Memory without embedding → skip."""
    db = AsyncMock()
    tid = uuid4()
    uid = uuid4()
    new_mem = _make_memory("test", tid, uid)
    new_mem.embedding = None

    result = await create_causal_edges(db, tid, uid, new_mem, llm_client=AsyncMock())
    assert result == []


@pytest.mark.asyncio
async def test_llm_failure_returns_empty():
    """LLM error → graceful fallback, no edges created."""
    db = AsyncMock()
    tid = uuid4()
    uid = uuid4()
    existing = _make_memory("lives in Sydney", tid, uid)

    candidates_result = MagicMock()
    candidates_result.scalars.return_value.all.return_value = [existing]
    db.execute.return_value = candidates_result

    llm = AsyncMock()
    llm.complete.side_effect = RuntimeError("LLM down")

    new_mem = _make_memory("joined Sydney office", tid, uid)
    result = await create_causal_edges(db, tid, uid, new_mem, llm_client=llm)
    assert result == []


@pytest.mark.asyncio
async def test_causal_link_creates_edge():
    """Valid causal link from LLM → edge created."""
    tid = uuid4()
    uid = uuid4()
    existing = _make_memory("moved to Sydney", tid, uid)
    new_mem = _make_memory("joined Sydney office", tid, uid)

    llm = AsyncMock()
    llm.complete.return_value = json.dumps({
        "causal_links": [{
            "from_id": str(existing.id),
            "to_id": str(new_mem.id),
            "direction": "existing_caused_new",
        }]
    })

    db = AsyncMock()
    call_count = 0

    async def _mock_execute(stmt):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        if call_count == 1:
            result.scalars.return_value.all.return_value = [existing]
        else:
            result.scalar_one_or_none.return_value = None
        return result

    db.execute = _mock_execute

    result = await create_causal_edges(db, tid, uid, new_mem, llm_client=llm)
    assert len(result) == 1
    assert result[0] == existing.id
    db.add.assert_called_once()

    edge = db.add.call_args[0][0]
    assert edge.edge_type == "causal"
    assert edge.weight == 0.7
    assert edge.from_memory_id == existing.id
    assert edge.to_memory_id == new_mem.id


@pytest.mark.asyncio
async def test_duplicate_edge_not_created():
    """If a causal edge already exists, don't create another."""
    tid = uuid4()
    uid = uuid4()
    existing = _make_memory("moved to Sydney", tid, uid)
    new_mem = _make_memory("joined Sydney office", tid, uid)

    llm = AsyncMock()
    llm.complete.return_value = json.dumps({
        "causal_links": [{
            "from_id": str(existing.id),
            "to_id": str(new_mem.id),
            "direction": "existing_caused_new",
        }]
    })

    db = AsyncMock()
    call_count = 0

    async def _mock_execute(stmt):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        if call_count == 1:
            result.scalars.return_value.all.return_value = [existing]
        else:
            existing_edge = MagicMock()
            result.scalar_one_or_none.return_value = existing_edge
        return result

    db.execute = _mock_execute

    result = await create_causal_edges(db, tid, uid, new_mem, llm_client=llm)
    assert result == []
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_self_referential_edge_rejected():
    """A causal edge from a memory to itself is invalid."""
    tid = uuid4()
    uid = uuid4()
    new_mem = _make_memory("test", tid, uid)

    llm = AsyncMock()
    llm.complete.return_value = json.dumps({
        "causal_links": [{
            "from_id": str(new_mem.id),
            "to_id": str(new_mem.id),
            "direction": "new_caused_existing",
        }]
    })

    db = AsyncMock()
    candidates_result = MagicMock()
    candidates_result.scalars.return_value.all.return_value = [
        _make_memory("other", tid, uid)
    ]
    db.execute.return_value = candidates_result

    result = await create_causal_edges(db, tid, uid, new_mem, llm_client=llm)
    assert result == []
