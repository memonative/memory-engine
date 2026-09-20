"""Unit tests for multi-strategy fusion (Change 2).

Verifies that RRF works correctly with 3 and 4 input lists, and
that graph-connected memories now compete on rank rather than being
appended at score 0.0.
"""
from memonative.engine.fusion import reciprocal_rank_fusion


def test_three_lists_shared_id_ranks_highest():
    """An id present in all three lists outranks singletons."""
    fused = reciprocal_rank_fusion([
        ["a", "b", "c"],
        ["a", "d", "e"],
        ["a", "f", "g"],
    ])
    assert fused[0] == "a"


def test_four_lists_shared_ids_above_singletons():
    """With 4 strategies, ids appearing in multiple lists dominate."""
    fused = reciprocal_rank_fusion([
        ["a", "b", "c"],
        ["d", "a", "e"],
        ["f", "a", "g"],
        ["h", "a", "i"],
    ])
    assert fused[0] == "a"


def test_graph_list_boosts_connected_memories():
    """A memory ranked low in vector but present in graph list gets boosted."""
    vector = ["v1", "v2", "v3", "v4", "v5", "graph_mem"]
    fts = ["fts1", "fts2"]
    graph = ["graph_mem", "graph_only"]

    fused = reciprocal_rank_fusion([vector, fts, graph])
    graph_mem_rank = fused.index("graph_mem")
    assert graph_mem_rank < 5, (
        f"graph_mem should be boosted into top 5, got rank {graph_mem_rank}"
    )


def test_empty_strategies_ignored():
    """Empty lists contribute nothing and don't break fusion."""
    fused = reciprocal_rank_fusion([
        ["a", "b"],
        [],
        ["c", "a"],
        [],
    ])
    assert fused[0] == "a"
    assert set(fused) == {"a", "b", "c"}


def test_temporal_list_surfaces_entity_matches():
    """Temporal strategy surfaces memories that appear in entity-expanded retrieval."""
    vector = ["v1", "v2", "v3"]
    fts = ["f1", "f2"]
    graph = []
    temporal = ["t1", "t2", "v2"]

    fused = reciprocal_rank_fusion([vector, fts, graph, temporal])
    v2_rank = fused.index("v2")
    assert v2_rank <= 2, (
        f"v2 appears in vector AND temporal, should rank high, got {v2_rank}"
    )


def test_fusion_count_not_reduced_by_extra_strategies():
    """Adding more strategies never reduces the total candidate count."""
    two = reciprocal_rank_fusion([["a", "b", "c"], ["d", "e"]])
    four = reciprocal_rank_fusion([["a", "b", "c"], ["d", "e"], ["f"], ["g"]])
    assert len(four) >= len(two)
