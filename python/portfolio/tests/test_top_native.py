import numpy as np
import pytest
from ruvector_core import VectorStore
from ruvector_attention import attend
from ruvector_gnn import GraphEncoder
from ruvector_sona import LoraTransform


def test_core_real_search_metadata_and_failure_atomicity():
    store = VectorStore(2)
    ids = store.add([[1, 0], [0, 1]], metadata=[{'kind':'a'}, {'kind':'b'}])
    assert store.search([1, 0], k=1)[0].id == ids[0]
    assert store.search([0, 1], k=1)[0].metadata == {'kind':'b'}
    with pytest.raises(ValueError):
        store.add([[1, 0], [float('nan'), 1]])
    assert len(store) == 2


def test_attention_matches_independent_math():
    q = np.array([1., 0.], dtype=np.float32)
    candidates = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    result = attend(q, candidates)
    logits = candidates @ q / np.sqrt(2)
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    np.testing.assert_allclose(result.weights, expected, rtol=1e-6)
    np.testing.assert_allclose(result.blended, expected @ candidates, rtol=1e-6)


def test_gnn_native_forward_and_weight_roundtrip():
    from ruvector_gnn import GnnLayer
    encoder = GraphEncoder(2, 4, heads=2)
    result = encoder.encode([1, 0], [[0, 1], [1, 1]])
    assert result.shape == (4,)
    assert np.isfinite(result).all()
    restored = GnnLayer.from_json(encoder.layer.to_json())
    np.testing.assert_allclose(result, restored.forward(np.array([1,0], dtype=np.float32), np.array([[0,1],[1,1]], dtype=np.float32)))


def test_sona_fresh_identity_both_paths():
    adapter = LoraTransform(2)
    data = [[1, 0], [0, 1]]
    np.testing.assert_array_equal(adapter.transform(data), data)
    np.testing.assert_array_equal(adapter.transform(data, layer=0), data)


@pytest.mark.parametrize('call', [
    lambda: VectorStore(True),
    lambda: VectorStore(65537),
    lambda: VectorStore(2).add([[1]]),
    lambda: VectorStore(2).search([1, 0], k=True),
    lambda: attend([float('nan')], [[1]]),
    lambda: attend([1, 0], [[1]]),
    lambda: GraphEncoder(2, 4, heads=3),
    lambda: GraphEncoder(2, 4).encode([1, 0], [[1, float('inf')]]),
    lambda: GraphEncoder(2, 4).encode([1, 0], [[1, 0]], weights=[-1]),
    lambda: LoraTransform(4097),
    lambda: LoraTransform(2).transform([[1, 0]], layer=-1),
    lambda: LoraTransform(2).transform([[float('inf'), 0]]),
])
def test_invalid_inputs_fail_closed(call):
    with pytest.raises(ValueError):
        call()


def test_gnn_aggregate_parameter_limit():
    with pytest.raises(ValueError, match="64 MiB"):
        GraphEncoder(4096, 4096)


def test_gnn_weight_scale_invariance():
    encoder = GraphEncoder(2, 4)
    node, neighbors = [1, 0], [[0, 1], [1, 1]]
    expected = encoder.encode(node, neighbors, weights=[1, 1])
    for magnitude in [3e38, 1e308]:
        np.testing.assert_allclose(encoder.encode(node, neighbors, weights=[magnitude, magnitude]), expected)


def test_gnn_large_finite_input_never_returns_nonfinite():
    encoder = GraphEncoder(2, 4)
    try:
        result = encoder.encode([3e38, 3e38], [[3e38, 3e38]])
    except ValueError as exc:
        assert "overflow" in str(exc)
    else:
        assert np.isfinite(result).all()


@pytest.mark.parametrize('metric', ['cosine', 'euclidean', 'dot', 'manhattan'])
def test_core_rejects_native_metric_overflow(metric):
    store = VectorStore(2, metric=metric)
    with pytest.raises(ValueError, match="magnitude"):
        store.add([[1e30, 1e30]])
    assert len(store) == 0
    store.add([[1, 0]])
    with pytest.raises(ValueError, match="magnitude"):
        store.search([1e30, 0])


@pytest.mark.parametrize('scale', [1e-20, 1e-300])
def test_core_cosine_tiny_opposite_vectors(scale):
    store = VectorStore(2)
    ids = store.add([[scale, 0], [-scale, 0]])
    hits = store.search([scale, 0], k=2)
    assert hits[0].id == ids[0]
    assert hits[0].score == pytest.approx(0, abs=1e-6)
    assert hits[1].score == pytest.approx(2, abs=1e-6)


def test_core_cosine_zero_rejected():
    store = VectorStore(2)
    with pytest.raises(ValueError, match="nonzero"):
        store.add([[0, 0]])
    with pytest.raises(ValueError, match="nonzero"):
        store.search([0, 0])
