from dataclasses import dataclass

from memonative.engine.goals import find_best_goal_match


@dataclass
class FakeGoal:
    description: str


def test_returns_none_when_no_goals():
    assert find_best_goal_match([], "anything") is None


def test_returns_none_when_description_only_stopwords():
    goals = [FakeGoal("learn python and ship the project")]
    # description is all stopwords — no signal
    assert find_best_goal_match(goals, "i to a the and") is None


def test_returns_none_below_overlap_threshold():
    # Single overlapping non-stopword shouldn't trigger a match.
    goals = [FakeGoal("learn rust this quarter")]
    assert find_best_goal_match(goals, "learn french") is None


def test_picks_strongest_overlap():
    goals = [
        FakeGoal("learn rust this quarter"),
        FakeGoal("ship the memonative mvp by friday"),
        FakeGoal("write blog post about embeddings"),
    ]
    chosen = find_best_goal_match(
        goals, "complete the memonative mvp"
    )
    assert chosen is not None
    assert chosen.description.startswith("ship")


def test_does_not_fall_back_to_first_goal():
    # Old behavior: returned goals[0] when nothing matched. New behavior:
    # returns None so the caller doesn't silently mutate the wrong goal.
    goals = [
        FakeGoal("learn rust this quarter"),
        FakeGoal("write blog post"),
    ]
    assert find_best_goal_match(goals, "play guitar") is None
