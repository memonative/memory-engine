"""Unit tests for reciprocal rank fusion."""

from memonative.engine.fusion import reciprocal_rank_fusion


def test_empty_inputs_return_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order():
    # Only one retriever → fused order is the same order.
    assert reciprocal_rank_fusion([["a", "b", "c"]]) == ["a", "b", "c"]


def test_item_in_both_lists_wins_over_singletons():
    # 'a' appears at rank 1 in both lists; 'b' and 'c' each only in one.
    # The sum of reciprocal ranks makes 'a' strictly dominant.
    fused = reciprocal_rank_fusion([
        ["a", "b"],
        ["a", "c"],
    ])
    assert fused[0] == "a"
    assert set(fused[1:]) == {"b", "c"}


def test_duplicate_within_a_list_only_counts_best_rank():
    # 'a' appears at ranks 1 and 5 in the same list — the second
    # occurrence must not double-count, otherwise fusion would reward
    # noisy retrievers that return the same item twice.
    fused = reciprocal_rank_fusion([
        ["a", "b", "c", "d", "a"],
    ])
    assert fused == ["a", "b", "c", "d"]


def test_k_constant_changes_ranking_compression():
    # With a very small k, top-of-list items weigh much more heavily:
    # one retriever's rank-1 can outrank another retriever's rank-3
    # that's also at rank-2 elsewhere. This test just pins that the
    # helper accepts the k argument and produces a deterministic order.
    fused_small_k = reciprocal_rank_fusion(
        [["x", "y"], ["z", "y"]],
        k=1,
    )
    # 'y' appears at ranks 2 and 2; 'x' and 'z' at rank 1 in one list each.
    # With k=1: score(y) = 2/(1+2) ≈ 0.667; score(x)=score(z)=1/2 = 0.5.
    assert fused_small_k[0] == "y"


def test_tuple_keys_are_supported():
    # The helper is generic over Hashable; callers frequently pass
    # composite keys. Just make sure tuples still work.
    fused = reciprocal_rank_fusion([
        [(1, "a"), (1, "b")],
        [(1, "b"), (1, "a")],
    ])
    assert set(fused) == {(1, "a"), (1, "b")}
