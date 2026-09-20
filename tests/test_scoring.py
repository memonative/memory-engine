from memonative.config import SCORING_WEIGHTS
from memonative.engine.scoring import compute_vector_similarity


def _weighted_score(
    vector_sim: float,
    strength: float,
    trust: float,
    salience: float,
    type_match: float,
) -> float:
    """Mirror of the weighted sum used in retrieve_memories.

    Kept here so the scoring-weight contract is pinned by unit tests
    without spinning up Postgres. If retrieve_memories changes how it
    combines these terms, update both sides.
    """
    return (
        SCORING_WEIGHTS["vector"] * vector_sim
        + SCORING_WEIGHTS["strength"] * strength
        + SCORING_WEIGHTS["trust"] * trust
        + SCORING_WEIGHTS["salience"] * salience
        + SCORING_WEIGHTS["type_match"] * type_match
    )


def test_scoring_weights_sum_to_one():
    # Not strictly required — but the current formula is a convex
    # combination and drifting off 1.0 almost always means someone
    # rebalanced one weight and forgot the others.
    assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 1e-9


def test_vector_is_the_dominant_factor():
    # The whole point of Task 2: vector similarity must outweigh the
    # other terms combined so semantic relevance actually drives ranking.
    assert SCORING_WEIGHTS["vector"] >= 0.5
    other = sum(v for k, v in SCORING_WEIGHTS.items() if k != "vector")
    assert SCORING_WEIGHTS["vector"] >= other


def test_semantically_matching_memory_wins_over_wrong_type():
    # Query is semantically very close to memory A (vector_sim=0.95)
    # but memory A is an "off-type" relative to the intent (type_match=0.0).
    # Memory B is perfectly on-type (type_match=1.0) but semantically far
    # from the query (vector_sim=0.05). A should still win because
    # semantic relevance is now the dominant term.
    a = _weighted_score(
        vector_sim=0.95, strength=0.6, trust=0.6, salience=0.5,
        type_match=0.0,
    )
    b = _weighted_score(
        vector_sim=0.05, strength=0.6, trust=0.6, salience=0.5,
        type_match=1.0,
    )
    assert a > b, f"expected semantic match to win; got a={a:.3f}, b={b:.3f}"


def test_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert compute_vector_similarity(v, v) == 1.0


def test_orthogonal_vectors():
    assert compute_vector_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_opposite_vectors():
    assert compute_vector_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_handles_none():
    assert compute_vector_similarity(None, [1.0, 2.0]) == 0.0
    assert compute_vector_similarity([1.0, 2.0], None) == 0.0


def test_handles_empty():
    assert compute_vector_similarity([], [1.0, 2.0]) == 0.0


def test_handles_zero_norm():
    assert compute_vector_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_handles_dimension_mismatch():
    # Different lengths shouldn't crash; returns 0.
    assert compute_vector_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
