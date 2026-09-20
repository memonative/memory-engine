"""Temporal query expansion — regex-based detection of time markers.

Detects temporal phrases in queries and returns structured context
that the reading LLM can use to reason about time. No LLM calls —
pure regex, zero latency cost.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal


@dataclass
class TemporalContext:
    has_temporal_marker: bool
    marker_type: Literal[
        "relative_ago",       # "X days/weeks/months ago"
        "duration_between",   # "how many days between X and Y"
        "ordering",           # "which happened first", "order of"
        "relative_period",    # "last month", "last week"
        "absolute_date",      # "in 2024", "on March 5"
        "before_after",       # "before I moved", "after the wedding"
        "none",
    ]
    matched_phrase: str | None
    reference_date: datetime | None

    @property
    def needs_date_context(self) -> bool:
        return self.marker_type in (
            "relative_ago", "duration_between", "relative_period",
        )


_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("relative_ago", re.compile(
        r"\b(?:how many|how much)\b.+?\b(?:days?|weeks?|months?|years?)\b.+?\b(?:ago|since|passed)\b",
        re.IGNORECASE,
    )),
    ("relative_ago", re.compile(
        r"\b(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple of)\s+(?:days?|weeks?|months?|years?)\s+ago\b",
        re.IGNORECASE,
    )),
    ("duration_between", re.compile(
        r"\b(?:how many|how much|how long)\b.+?\b(?:days?|weeks?|months?|years?)\b.+?\b(?:between|from .+ to|passed between)\b",
        re.IGNORECASE,
    )),
    ("duration_between", re.compile(
        r"\b(?:how long|how many days|how many weeks)\b.+?\b(?:did|had|take|took|spent|spend|been)\b",
        re.IGNORECASE,
    )),
    ("ordering", re.compile(
        r"\b(?:which .+ (?:first|last|earlier|later)|(?:order|sequence) of|from (?:first|earliest) to (?:last|latest))\b",
        re.IGNORECASE,
    )),
    ("ordering", re.compile(
        r"\bwho\b.+\b(?:first|most recently|before|earlier)\b",
        re.IGNORECASE,
    )),
    ("ordering", re.compile(
        r"\bmost recently\b",
        re.IGNORECASE,
    )),
    ("relative_period", re.compile(
        r"\b(?:last|past|previous|this)\s+(?:week|month|year|quarter|semester|weekend|saturday|sunday|monday|tuesday|wednesday|thursday|friday)\b",
        re.IGNORECASE,
    )),
    ("relative_period", re.compile(
        r"\b(?:in|during)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+and\s+(?:january|february|march|april|may|june|july|august|september|october|november|december))?\b",
        re.IGNORECASE,
    )),
    ("absolute_date", re.compile(
        r"\b(?:in\s+20\d{2}|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})\b",
        re.IGNORECASE,
    )),
    ("absolute_date", re.compile(
        r"\b(?:on\s+)?(?:valentine'?s\s+day|christmas|new\s+year|thanksgiving|easter|halloween)\b",
        re.IGNORECASE,
    )),
    ("before_after", re.compile(
        r"\b(?:before|after|prior to|since|following|when)\s+(?:I|my|the|we|it)\b",
        re.IGNORECASE,
    )),
    ("before_after", re.compile(
        r"\bhow old was I when\b",
        re.IGNORECASE,
    )),
    ("before_after", re.compile(
        r"\bwhen did I\b",
        re.IGNORECASE,
    )),
]


def detect_temporal_context(
    query: str,
    reference_date: datetime | None = None,
) -> TemporalContext:
    for marker_type, pattern in _PATTERNS:
        match = pattern.search(query)
        if match:
            return TemporalContext(
                has_temporal_marker=True,
                marker_type=marker_type,
                matched_phrase=match.group(0),
                reference_date=reference_date,
            )

    return TemporalContext(
        has_temporal_marker=False,
        marker_type="none",
        matched_phrase=None,
        reference_date=reference_date,
    )
