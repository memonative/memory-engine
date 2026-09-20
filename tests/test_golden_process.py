"""Characterization test for the /v1/process pipeline.

This is the safety net for the M0 facade extraction. It replays recorded
LLM responses through the real orchestration and asserts the assembled
output is byte-identical to what the pre-refactor code produced. No API
key, no cost, no nondeterminism — a diff here means the refactor changed
behaviour.

Needs a local Postgres (marked `dbintegration`). No fixtures ship with the
repo — recording them costs a live LLM run, so the test skips until you
record your own:

    PYTHONPATH=src python scripts/record_golden.py

Re-record only when changing engine behaviour on purpose.

A cassette is tied to the id stream of the tree that recorded it. The
recorded responses quote memory ids back at us, so any change that adds or
removes a uuid-defaulted insert on the request path shifts every later id,
the `memory_id` references stop resolving, and reinforcements silently
no-op. That shows up as drifted trust scores rather than an obvious error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"
CASSETTE = GOLDEN_DIR / "cassette.json"
EXPECTED = GOLDEN_DIR / "expected.json"

pytestmark = [
    pytest.mark.dbintegration,
    pytest.mark.skipif(
        not (CASSETTE.exists() and EXPECTED.exists()),
        reason="golden fixtures not recorded; run scripts/record_golden.py",
    ),
]


@pytest.mark.asyncio
async def test_pipeline_matches_golden_responses():
    from memonative.db.database import engine

    from tests.golden.runner import run_corpus
    from tests.support.llm_capture import ReplayLLM

    # The engine's pool is module-level but pytest-asyncio gives each test its
    # own event loop, so connections opened by an earlier test surface here as
    # "another operation is in progress". Dispose first to force fresh ones.
    await engine.dispose()

    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    actual = await run_corpus(ReplayLLM(json.loads(CASSETTE.read_text(encoding="utf-8"))))

    assert set(actual) == set(expected), (
        "golden document set changed: "
        f"added={sorted(set(actual) - set(expected))} "
        f"removed={sorted(set(expected) - set(actual))}"
    )

    drifted = [k for k in expected if actual[k] != expected[k]]
    if drifted:
        k = drifted[0]
        pytest.fail(
            f"{len(drifted)} golden document(s) changed: {drifted}\n\n"
            f"first diff — {k}:\n"
            f"  expected: {json.dumps(expected[k], indent=1)[:1500]}\n"
            f"  actual  : {json.dumps(actual[k], indent=1)[:1500]}"
        )
