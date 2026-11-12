"""Tests for ``ruvector.GnnLayer`` and ``ruvector.AttentionReranker``
(ADR-352 capability-expansion slice): GNN forward-pass rerank and
attention-based rerank bindings.

``GnnLayer`` is random-init (Xavier/Glorot) with no training step in this
binding, so its forward pass is only checked for basic sanity (right
shape, finite, not all zeros) — the exact values are not meaningful to
pin down. ``AttentionReranker`` is trainless and deterministic
(``softmax(QK^T/sqrt(d))V``), so its weight distribution over an
orthogonal candidate set *is* a meaningful, checkable property.
"""

from __future__ import annotations

import numpy as np
import pytest

from ruvector import AttentionReranker, GnnLayer, RuVectorError


# ---------------------------------------------------------------------------
# GnnLayer
# ---------------------------------------------------------------------------


def test_gnn_layer_construction() -> None:
    layer = GnnLayer(input_dim=4, hidden_dim=8, heads=2, dropout=0.1)
    assert layer.input_dim == 4
    assert layer.hidden_dim == 8
    assert layer.heads == 2
    assert layer.dropout == pytest.approx(0.1)


def test_gnn_layer_forward_shape_and_sanity() -> None:
    layer = GnnLayer(input_dim=4, hidden_dim=8, heads=2, dropout=0.0)
    node = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    neighbors = np.array(
        [[0.5, 1.0, 1.5, 2.0], [2.0, 3.0, 4.0, 5.0]], dtype=np.float32
    )
    weights = np.array([0.3, 0.7], dtype=np.float32)

    output = layer.forward(node, neighbors, weights)

    assert output.shape == (8,)
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))
    assert not np.all(output == 0.0)


def test_gnn_layer_forward_no_neighbors() -> None:
    layer = GnnLayer(input_dim=4, hidden_dim=8, heads=2, dropout=0.0)
    node = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    neighbors = np.zeros((0, 4), dtype=np.float32)

    output = layer.forward(node, neighbors)

    assert output.shape == (8,)
    assert np.all(np.isfinite(output))


def test_gnn_layer_forward_default_weights_matches_uniform() -> None:
    # weights=None (default) must behave identically to an explicit
    # uniform array, since RuvectorLayer normalizes weights internally.
    layer = GnnLayer(input_dim=4, hidden_dim=8, heads=2, dropout=0.0)
    node = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    neighbors = np.array(
        [[0.5, 1.0, 1.5, 2.0], [2.0, 3.0, 4.0, 5.0]], dtype=np.float32
    )

    out_default = layer.forward(node, neighbors)
    out_uniform = layer.forward(node, neighbors, np.array([1.0, 1.0], dtype=np.float32))

    assert np.array_equal(out_default, out_uniform)


def test_gnn_layer_to_json_from_json_round_trip() -> None:
    layer = GnnLayer(input_dim=4, hidden_dim=8, heads=2, dropout=0.1)
    node = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    neighbors = np.array(
        [[0.5, 1.0, 1.5, 2.0], [2.0, 3.0, 4.0, 5.0]], dtype=np.float32
    )
    weights = np.array([0.3, 0.7], dtype=np.float32)

    before = layer.forward(node, neighbors, weights)

    data = layer.to_json()
    restored = GnnLayer.from_json(data)

    assert restored.input_dim == layer.input_dim
    assert restored.hidden_dim == layer.hidden_dim
    assert restored.heads == layer.heads
    assert restored.dropout == pytest.approx(layer.dropout)

    after = restored.forward(node, neighbors, weights)
    # forward() has no RNG of its own, so a round-tripped layer must
    # reproduce the exact same output.
    assert np.array_equal(before, after)


def test_gnn_layer_node_dimension_mismatch_raises() -> None:
    layer = GnnLayer(input_dim=4, hidden_dim=8, heads=2, dropout=0.0)
    bad_node = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # wrong length
    neighbors = np.zeros((0, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        layer.forward(bad_node, neighbors)


def test_gnn_layer_neighbors_dimension_mismatch_raises() -> None:
    layer = GnnLayer(input_dim=4, hidden_dim=8, heads=2, dropout=0.0)
    node = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    bad_neighbors = np.zeros((2, 3), dtype=np.float32)  # wrong inner dim

    with pytest.raises(ValueError):
        layer.forward(node, bad_neighbors)


def test_gnn_layer_weights_length_mismatch_raises() -> None:
    layer = GnnLayer(input_dim=4, hidden_dim=8, heads=2, dropout=0.0)
    node = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    neighbors = np.array([[0.5, 1.0, 1.5, 2.0]], dtype=np.float32)
    bad_weights = np.array([0.3, 0.7], dtype=np.float32)  # length 2, n=1

    with pytest.raises(ValueError):
        layer.forward(node, neighbors, bad_weights)


def test_gnn_layer_invalid_heads_hidden_dim_raises_ruvector_error() -> None:
    # hidden_dim=7 is not divisible by heads=3 — this is the one path that
    # exercises the real RuvectorLayer::new error -> to_pyerr_gnn mapper,
    # as opposed to this binding's own boundary ValueErrors above.
    with pytest.raises(RuVectorError):
        GnnLayer(input_dim=4, hidden_dim=7, heads=3, dropout=0.1)


def test_gnn_layer_invalid_dropout_raises_ruvector_error() -> None:
    with pytest.raises(RuVectorError):
        GnnLayer(input_dim=4, hidden_dim=8, heads=2, dropout=1.5)


def test_gnn_layer_zero_dims_raise_value_error() -> None:
    with pytest.raises(ValueError):
        GnnLayer(input_dim=0, hidden_dim=8, heads=2, dropout=0.0)
    with pytest.raises(ValueError):
        GnnLayer(input_dim=4, hidden_dim=0, heads=2, dropout=0.0)
    with pytest.raises(ValueError):
        GnnLayer(input_dim=4, hidden_dim=8, heads=0, dropout=0.0)


# ---------------------------------------------------------------------------
# AttentionReranker
# ---------------------------------------------------------------------------


def test_attention_reranker_construction() -> None:
    reranker = AttentionReranker(dim=4)
    assert reranker.dim == 4


def test_attention_reranker_picks_closest_candidate() -> None:
    # Orthogonal-ish candidate set, unambiguous nearest answer — same
    # style as test_collection.py's search tests.
    reranker = AttentionReranker(dim=4)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    candidates = np.eye(4, dtype=np.float32)[:3]  # rows: e0, e1, e2

    blended, weights = reranker.rerank(query, candidates)

    assert weights.shape == (3,)
    assert blended.shape == (4,)
    assert np.all(np.isfinite(weights))
    assert weights.sum() == pytest.approx(1.0, abs=1e-5)
    # e0 is exactly aligned with the query -> highest attention weight.
    assert int(np.argmax(weights)) == 0
    assert weights[0] > weights[1]
    assert weights[0] > weights[2]


def test_attention_reranker_blended_equals_weights_matmul_candidates() -> None:
    # Pins the relationship between the authoritative `compute()` output
    # and the locally-recomputed weights (see gnn.rs's rerank() doc
    # comment) — if the duplicated softmax formula ever drifts from the
    # upstream crate's private implementation, this test catches it.
    reranker = AttentionReranker(dim=4)
    query = np.array([0.3, -0.1, 0.9, 0.2], dtype=np.float32)
    candidates = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.2, 0.2, 0.2, 0.2],
        ],
        dtype=np.float32,
    )

    blended, weights = reranker.rerank(query, candidates)
    expected = weights @ candidates

    np.testing.assert_allclose(blended, expected, atol=1e-5)


def test_attention_reranker_uniform_weights_when_query_orthogonal_to_all() -> None:
    reranker = AttentionReranker(dim=4)
    query = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    candidates = np.eye(4, dtype=np.float32)[:3]

    _, weights = reranker.rerank(query, candidates)

    # Zero query -> all scores are 0 -> uniform softmax.
    np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0), atol=1e-6)


def test_attention_reranker_query_dimension_mismatch_raises() -> None:
    reranker = AttentionReranker(dim=4)
    bad_query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidates = np.eye(4, dtype=np.float32)[:2]

    with pytest.raises(ValueError):
        reranker.rerank(bad_query, candidates)


def test_attention_reranker_candidates_dimension_mismatch_raises() -> None:
    reranker = AttentionReranker(dim=4)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    bad_candidates = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        reranker.rerank(query, bad_candidates)


def test_attention_reranker_empty_candidates_raises() -> None:
    reranker = AttentionReranker(dim=4)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    empty_candidates = np.zeros((0, 4), dtype=np.float32)

    with pytest.raises(ValueError):
        reranker.rerank(query, empty_candidates)


def test_attention_reranker_zero_dim_raises() -> None:
    with pytest.raises(ValueError):
        AttentionReranker(dim=0)
