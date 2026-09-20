from datetime import datetime, timedelta, timezone

from memonative.engine.decay import compute_decay


def test_fresh_memory_is_full_strength():
    # Just-accessed memory should be at strength ~1.0 and active.
    res = compute_decay(
        half_life_hours=72.0,
        reinforcement_count=1,
        salience=0.5,
        last_accessed_at=datetime.now(timezone.utc),
    )
    assert res.strength > 0.99
    assert res.decay_state == "active"


def test_old_memory_decays_to_archived():
    # 30 days ago with default half-life — should be archived.
    long_ago = datetime.now(timezone.utc) - timedelta(days=30)
    res = compute_decay(
        half_life_hours=72.0,
        reinforcement_count=1,
        salience=0.0,
        last_accessed_at=long_ago,
    )
    assert res.strength < 0.05
    assert res.decay_state == "archived"


def test_a_pinned_memory_reports_full_strength_however_old():
    """A NULL half-life means pinned. Full strength is what keeps it above
    retrieval's floor — a pin that let its memory drop out of search would
    not be a pin.
    """
    res = compute_decay(
        half_life_hours=None,
        reinforcement_count=1,
        salience=0.0,
        last_accessed_at=datetime.now(timezone.utc) - timedelta(days=3650),
    )
    assert res.strength == 1.0
    assert res.decay_state == "active"


def test_reinforcement_extends_half_life():
    # Same elapsed time, more reinforcements → higher strength.
    elapsed = datetime.now(timezone.utc) - timedelta(hours=72)
    weak = compute_decay(72.0, 1, 0.5, elapsed)
    strong = compute_decay(72.0, 16, 0.5, elapsed)
    assert strong.strength > weak.strength


def test_salience_extends_half_life():
    elapsed = datetime.now(timezone.utc) - timedelta(hours=72)
    low = compute_decay(72.0, 1, 0.0, elapsed)
    high = compute_decay(72.0, 1, 1.0, elapsed)
    assert high.strength > low.strength


def test_naive_timestamp_is_treated_as_utc():
    # last_accessed_at without tzinfo should not crash.
    naive = datetime.utcnow() - timedelta(hours=1)
    res = compute_decay(72.0, 1, 0.5, naive)
    assert 0.0 <= res.strength <= 1.0
