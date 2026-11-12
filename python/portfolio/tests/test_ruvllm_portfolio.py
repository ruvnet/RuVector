import numpy as np
import pytest
from ruvector_ruvllm import KVCache, TokenCache


def test_matrix_cache_attention_reference():
    cache = TokenCache(2, 3)
    keys = np.eye(2)
    values = np.array([[2, 4], [8, 10]])
    cache.append(keys, values)
    scores = keys @ [1, 0] / np.sqrt(2)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    np.testing.assert_allclose(cache.attend([1, 0]), weights @ values, rtol=1e-6)
    assert cache.attend([1, 0]).dtype == np.float32
    stored_keys, stored_values = cache.get_all()
    np.testing.assert_array_equal(stored_keys, keys)
    np.testing.assert_array_equal(stored_values, values)
    stored_keys[:] = 99
    np.testing.assert_array_equal(cache.get_all()[0], keys)
    cache.clear()
    assert len(cache) == 0
    assert cache.get_all()[0].shape == (0, 2)
    assert isinstance(KVCache(2), KVCache)


@pytest.mark.parametrize("keys,values", [([1, 2], [3, 4]), ([[1]], [[2]]), ([[1, 2]], [[3]]), ([[float('nan'), 0]], [[1, 2]]), ([[1j, 0]], [[1, 2]]), ([[True, False]], [[1, 2]])])
def test_bad_matrices_do_not_mutate(keys, values):
    cache = TokenCache(2)
    with pytest.raises(ValueError):
        cache.append(keys, values)
    assert len(cache) == 0


def test_overflow_is_atomic():
    cache = TokenCache(2, 1)
    cache.append([[1, 2]], [[3, 4]])
    with pytest.raises(ValueError, match="capacity"):
        cache.append([[5, 6]], [[7, 8]])
    assert len(cache) == 1
    np.testing.assert_array_equal(cache.attend([1, 0]), [3, 4])
