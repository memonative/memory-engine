from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from memonative.db.enums import RevisionType
from memonative.engine.retrieval import format_revision_history


@dataclass
class FakeRevision:
    sequence_number: int
    content: str
    revision_type: RevisionType
    valid_from: datetime
    valid_until: datetime | None


def _ts(days_ago: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


def test_empty_for_single_revision():
    only = FakeRevision(1, "User lives in Mumbai", RevisionType.initial, _ts(10), None)
    assert format_revision_history([only]) == ""


def test_temporal_update_chain_shows_history():
    revisions = [
        FakeRevision(1, "User lives in Delhi", RevisionType.initial, _ts(60), _ts(30)),
        FakeRevision(2, "User lives in Mumbai", RevisionType.temporal_update, _ts(30), None),
    ]
    out = format_revision_history(revisions)
    assert "Delhi" in out


def test_retraction_marks_initial_as_error():
    revisions = [
        FakeRevision(1, "Paris", RevisionType.initial, _ts(20), _ts(10)),
        FakeRevision(2, "Mumbai", RevisionType.retraction, _ts(10), None),
    ]
    out = format_revision_history(revisions)
    assert "Paris" in out
    assert "error" in out.lower()


def test_current_revision_is_omitted():
    # The current value (valid_until=None) should not appear in history.
    revisions = [
        FakeRevision(1, "Old City", RevisionType.initial, _ts(60), _ts(30)),
        FakeRevision(2, "Current City", RevisionType.temporal_update, _ts(30), None),
    ]
    out = format_revision_history(revisions)
    assert "Current City" not in out
