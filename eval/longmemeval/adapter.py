"""Adapter contract for swapping memory backends in the LongMemEval run.

Both adapters implement the same async interface so the runner doesn't
care which system it's driving. The runner ingests sessions one at a
time per question, queries once, then resets the user.

`source_session_id` on the returned memories is the load-bearing field:
LongMemEval ground truth is "which session(s) contain the answer", so
recall@k checks whether the top-k retrieved memories include any tagged
with one of the question's needle sessions. Each adapter is responsible
for round-tripping the session id through whatever its native tagging
mechanism is (Memonative context_tags, mem0 metadata, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedMemory:
    content: str
    source_session_id: str
    score: float
    memory_type: str = "episodic"
    attribute_slot: str | None = None
    trust_score: float = 0.5
    created_at: str = ""
    revision_history: list[dict] = field(default_factory=list)
    # Per-memory event semantics — one of "none", "add", "update",
    # "delete". Lets the reader distinguish state changes from static
    # facts without inferring from prose. Defaults to "none" so
    # non-Memonative adapters (mem0) stay compatible.
    event_type: str = "none"
    # Resolved absolute date of the event described in `content`,
    # distinct from `created_at` (when the user told the assistant).
    # Empty string = same as created_at / not set.
    event_date: str = ""


class MemoryAdapter(ABC):
    """Both Memonative and mem0 implementations conform to this."""

    name: str

    @abstractmethod
    async def ingest(
        self,
        user_id: str,
        session_id: str,
        session_date: str,
        turns: list[dict],
    ) -> None:
        """Persist one conversation session for `user_id`.

        `turns` is a list of {"role": "user"|"assistant", "content": str}.
        `session_date` is an ISO-8601 string from the dataset; adapters
        that don't expose timestamp control should ignore it.
        """

    @abstractmethod
    async def query(
        self,
        user_id: str,
        question: str,
        top_k: int = 5,
    ) -> list[RetrievedMemory]:
        """Return the top-k memories for `question`, best-first."""

    @abstractmethod
    async def reset_user(self, user_id: str) -> None:
        """Wipe all memories for `user_id`. Each LongMemEval question
        gets a fresh user so haystacks don't bleed into each other."""
