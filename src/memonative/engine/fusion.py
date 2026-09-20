"""Reciprocal Rank Fusion.

Combines multiple ranked candidate lists (e.g. vector-similarity and
full-text hits) into a single ranking. Pure function, no DB, no ORM —
keeps fusion easy to unit-test and trivially reusable if we add a
third retriever later.

Formula (standard RRF):

    score(id) = sum over lists L of  1 / (k + rank_L(id))

where rank_L is 1-indexed. Items missing from a list contribute nothing
from that list. k = 60 is the common default from the original RRF
paper and is robust across list sizes.
"""
from __future__ import annotations

from typing import Hashable, Iterable, Sequence, TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[T]],
    k: int = 60,
) -> list[T]:
    """Fuse several ranked lists into one, ordered best-first.

    Each inner list is ordered best-first (rank 1 is the strongest hit
    from that retriever). Duplicates inside a single list are ignored
    after the first occurrence — the same id at rank 5 and rank 20 of
    the same list only counts once, at its best rank.
    """
    scores: dict[T, float] = {}
    for ranked in ranked_lists:
        seen: set[T] = set()
        for rank, item in enumerate(ranked, start=1):
            if item in seen:
                # Only the best rank from a list should contribute.
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)

    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
