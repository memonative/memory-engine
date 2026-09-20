"""Record the golden-response fixtures for the M0 facade refactor.

Run this ONCE against a local Postgres before touching engine code, and
again only when you deliberately change engine behaviour.

    docker compose up -d db
    DATABASE_URL=postgresql+asyncpg://memonative:memonative@localhost:5432/memonative \
        PYTHONPATH=src python scripts/record_golden.py

Three passes, in order:

  1. LIVE     — real LLM, real cost. Tees every request/response into a cassette.
  2. REPLAY   — same corpus, served from the cassette. *This* pass produces the
                goldens, so fixtures and goldens are consistent by construction
                rather than by hope.
  3. VERIFY   — a second replay, asserted equal to pass 2. Proves the harness is
                a fixed point before anyone trusts it as a refactor safety net.

Costs roughly 20 messages x ~5 LLM round trips plus embeddings. It writes
only under the fixed golden user id and purges that user first, so it is
safe to re-run.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

GOLDEN_DIR = ROOT / "tests" / "golden"
CASSETTE = GOLDEN_DIR / "cassette.json"
EXPECTED = GOLDEN_DIR / "expected.json"


async def main() -> int:
    db_url = os.environ.get("DATABASE_URL", "")
    if "localhost" not in db_url and "127.0.0.1" not in db_url:
        print(
            "REFUSING TO RUN: DATABASE_URL is not local.\n"
            f"  got: {db_url.split('@')[-1] or '(unset — would use .env)'}\n"
            "This script writes memories, revisions and usage rows. Point it at "
            "the docker-compose Postgres, not your cloud instance.",
            file=sys.stderr,
        )
        return 2

    # The write path fires `check_contradictions_task.delay(...)`. Left on the
    # .env value that would enqueue twenty corpus runs onto the production
    # queue, so pin it local before config is imported. No worker consumes it.
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"

    from memonative.auth.bootstrap import ensure_default_tenant_and_master_key
    from memonative.api.llm import LLMClient

    from tests.golden.runner import run_corpus
    from tests.support.llm_capture import RecordingLLM, ReplayLLM

    await ensure_default_tenant_and_master_key()

    # --skip-live regenerates goldens from the existing cassette. Use it when
    # the harness itself changes (normalization, corpus ordering) rather than
    # engine behaviour — no reason to pay for the same answers twice.
    if "--skip-live" in sys.argv:
        print("\n=== pass 1/3: LIVE — skipped, reusing cassette ===")
        cassette = json.loads(CASSETTE.read_text(encoding="utf-8"))
        print(f"  cassette: {len(cassette['entries'])} unique requests")
    else:
        print("\n=== pass 1/3: LIVE (real LLM calls, costs money) ===")
        recorder = RecordingLLM(LLMClient())
        await run_corpus(recorder, verbose=True)
        cassette = recorder.dump()
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        CASSETTE.write_text(json.dumps(cassette, indent=1, sort_keys=True), encoding="utf-8")
        print(f"  cassette: {len(cassette['entries'])} unique requests -> {CASSETTE.name} "
              f"({CASSETTE.stat().st_size / 1_000_000:.1f} MB)")

    print("\n=== pass 2/3: REPLAY (generates goldens) ===")
    goldens = await run_corpus(ReplayLLM(cassette), verbose=True)
    EXPECTED.write_text(json.dumps(goldens, indent=1, sort_keys=True), encoding="utf-8")
    print(f"  goldens: {len(goldens)} documents -> {EXPECTED.name}")

    print("\n=== pass 3/3: VERIFY (replay must be a fixed point) ===")
    again = await run_corpus(ReplayLLM(cassette))
    if again != goldens:
        drifted = [k for k in goldens if again.get(k) != goldens[k]]
        print(f"  FAILED — {len(drifted)} document(s) differ between two "
              f"identical replays: {drifted[:5]}", file=sys.stderr)
        print("  The harness is not deterministic; do not refactor against it.",
              file=sys.stderr)
        return 1

    print(f"  OK — {len(goldens)} documents byte-identical across replays.")
    print("\nGolden harness ready. Refactor away.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
