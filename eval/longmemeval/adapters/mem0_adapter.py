"""mem0 self-hosted adapter for the LongMemEval eval.

Pinned to the same models Memonative uses (gpt-4o-mini for extraction,
text-embedding-3-small for vectors). mem0's default v2.0.1 LLM is
gpt-5-mini, which fails on the deprecated `max_tokens` parameter mem0
still emits — so we override regardless of what the comparison demands.

Storage: embedded Qdrant at `eval/longmemeval/data/qdrant_mem0/`. No
docker container needed; mem0 manages its own Qdrant via local files.

mem0's API is sync. We push calls onto a thread executor so the async
adapter contract works. The runner ingests sequentially anyway, so
single-threaded is fine.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from mem0 import Memory
from mem0.configs.base import MemoryConfig

from memonative.config import settings
from eval.longmemeval.adapter import MemoryAdapter, RetrievedMemory


logger = logging.getLogger(__name__)


_QDRANT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "qdrant_mem0"
)


def _build_memory() -> Memory:
    # mem0 reads OPENAI_API_KEY directly from os.environ; Memonative
    # reads it from a pydantic-settings .env file. Bridge the two so
    # users don't have to export it twice.
    if not os.environ.get("OPENAI_API_KEY") and settings.OPENAI_API_KEY.get_secret_value():
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

    cfg = MemoryConfig()
    # Pin LLM and embedder so we're measuring memory architecture, not
    # model choice asymmetry against Memonative (which uses these too).
    cfg.llm.config = {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 2000,
    }
    cfg.embedder.config = {
        "model": "text-embedding-3-small",
    }
    cfg.vector_store.config.path = str(_QDRANT_PATH)
    cfg.vector_store.config.embedding_model_dims = 1536
    return Memory(cfg)


class Mem0Adapter(MemoryAdapter):
    name = "mem0"

    def __init__(self) -> None:
        self._memory = _build_memory()

    async def aclose(self) -> None:
        # mem0 holds a Qdrant local-mode lock; releasing it before
        # process exit avoids the noisy msvcrt-on-shutdown warning.
        await asyncio.to_thread(self._memory.close)

    async def ingest(
        self,
        user_id: str,
        session_id: str,
        session_date: str,
        turns: list[dict],
    ) -> None:
        if not turns:
            return

        # mem0 takes the {role, content} list natively — no concat needed.
        # session_id rides on metadata so query() can read it back.
        # Per-session fail-soft: mem0 raises BadRequestError on sessions
        # that exceed text-embedding-3-small's 8192-token limit (it doesn't
        # pre-chunk). Without this catch, one bad session zeros out the
        # whole question's haystack — which represents mem0's OOTB
        # behavior under realistic input far more harshly than reality.
        # We log + continue, matching Memonative's per-session resilience.
        try:
            await asyncio.to_thread(
                self._memory.add,
                messages=turns,
                user_id=user_id,
                metadata={"session_id": session_id, "session_date": session_date},
            )
        except Exception as e:
            logger.warning(
                "mem0 ingest failed for user=%s session=%s: %s",
                user_id, session_id, e,
            )

    async def query(
        self,
        user_id: str,
        question: str,
        top_k: int = 5,
    ) -> list[RetrievedMemory]:
        raw = await asyncio.to_thread(
            self._memory.search,
            query=question,
            top_k=top_k,
            filters={"user_id": user_id},
        )

        # mem0 v2 returns {"results": [{"id", "memory", "score",
        # "metadata", ...}]} for search.
        results_list = raw.get("results", []) if isinstance(raw, dict) else raw

        out: list[RetrievedMemory] = []
        for item in results_list:
            metadata = item.get("metadata") or {}
            out.append(RetrievedMemory(
                content=item.get("memory") or item.get("text") or "",
                source_session_id=metadata.get("session_id", ""),
                score=float(item.get("score", 0.0) or 0.0),
            ))
        return out

    async def reset_user(self, user_id: str) -> None:
        await asyncio.to_thread(
            self._memory.delete_all,
            user_id=user_id,
        )
