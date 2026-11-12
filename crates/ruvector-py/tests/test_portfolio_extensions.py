"""Native LLM cache acceptance tests against independent attention calculations."""
import math

import pytest
from ruvector import KVCache


def test_native_attention_matches_reference():
    cache = KVCache(2, max_tokens=3)
    cache.append([1, 0, 0, 1], [2, 4, 8, 10])
    scale = 1 / math.sqrt(2)
    weight = math.exp(scale) / (math.exp(scale) + 1)
    assert cache.attend([1, 0]) == pytest.approx(
        [weight * 2 + (1 - weight) * 8, weight * 4 + (1 - weight) * 10]
    )
    assert len(cache) == 2
    assert cache.get_all() == ([1, 0, 0, 1], [2, 4, 8, 10])
    assert cache.dimensions == 2
    assert cache.max_tokens == 3


def test_capacity_and_rejected_appends_are_atomic():
    cache = KVCache(2, max_tokens=1)
    cache.append([1, 2], [3, 4])
    with pytest.raises(ValueError, match="capacity"):
        cache.append([5, 6], [7, 8])
    assert cache.get_all() == ([1, 2], [3, 4])
    cache.clear()
    assert len(cache) == 0
    assert cache.attend([1, 0]) == [0, 0]
    cache.append([5, 6], [7, 8])
    assert cache.attend([1, 0]) == pytest.approx([7, 8])


@pytest.mark.parametrize("dimensions,max_tokens", [(0, 1), (4097, 1), (1, 0), (1, 65537), (4096, 65536)])
def test_configuration_bounds(dimensions, max_tokens):
    with pytest.raises(ValueError):
        KVCache(dimensions, max_tokens)


@pytest.mark.parametrize("keys,values", [([], []), ([1], [2]), ([1, 2], [3]), ([math.nan, 0], [1, 2]), ([1, 2], [math.inf, 0]), ([1e11, 0], [1, 2])])
def test_invalid_data_cannot_mutate_cache(keys, values):
    cache = KVCache(2)
    with pytest.raises(ValueError):
        cache.append(keys, values)
    assert len(cache) == 0


@pytest.mark.parametrize("query,scale", [([1], 1), ([math.nan, 0], 1), ([1, 0], 0), ([1, 0], -1), ([1, 0], math.inf), ([1, 0], 1e11)])
def test_attention_rejects_invalid_arguments(query, scale):
    with pytest.raises(ValueError):
        KVCache(2).attend(query, scale)
