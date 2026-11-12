"""Bounded full precision token caches backed by native ruvllm Rust code."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from ruvector import KVCache

__all__ = ["KVCache", "TokenCache"]


class TokenCache:
    """Row-shaped interface to the native KVCache.

    append accepts (tokens, dimensions) arrays. Attention is a single
    flattened head, not a model inference or text generation API.
    """

    def __init__(self, dimensions: int, max_tokens: int = 4096) -> None:
        if isinstance(dimensions, bool) or isinstance(max_tokens, bool):
            raise ValueError("dimensions and max_tokens must be integers, not booleans")
        self._cache = KVCache(dimensions, max_tokens)

    @property
    def dimensions(self) -> int:
        return self._cache.dimensions

    @property
    def max_tokens(self) -> int:
        return self._cache.max_tokens

    def __len__(self) -> int:
        return len(self._cache)

    def append(self, keys: ArrayLike, values: ArrayLike) -> None:
        """Append matching token matrices, without eviction or quantization.

        Finite values must have magnitude at most 1e10. Conversion to
        float32 may copy input arrays. Invalid input never changes the cache.
        """
        k = np.asarray(keys)
        v = np.asarray(values)
        if k.ndim != 2 or v.shape != k.shape or k.shape[1] != self.dimensions:
            raise ValueError("keys and values must share shape (tokens, dimensions)")
        if k.shape[0] == 0 or k.shape[0] > self.max_tokens - len(self):
            raise ValueError("empty batch or cache capacity exceeded")
        for data in (k, v):
            if data.dtype.kind not in "iuf" or not np.isfinite(data).all() or (np.abs(data.astype(np.float64)) > 1e10).any():
                raise ValueError("coordinates must be finite real numbers with magnitude <= 1e10")
        self._cache.append(k.astype(np.float32).ravel().tolist(), v.astype(np.float32).ravel().tolist())

    def attend(self, query: ArrayLike, scale: float | None = None) -> NDArray[np.float32]:
        """Return softmax attention output; default scale = 1/sqrt(dimensions)."""
        q = np.asarray(query)
        if q.shape != (self.dimensions,) or q.dtype.kind not in "iuf":
            raise ValueError("query must be a real vector with shape (dimensions,)")
        if not np.isfinite(q).all() or (np.abs(q.astype(np.float64)) > 1e10).any():
            raise ValueError("query must be finite with magnitude <= 1e10")
        return np.asarray(self._cache.attend(q.astype(np.float32).tolist(), scale), dtype=np.float32)

    def get_all(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return copied (keys, values) matrices in insertion order."""
        keys, values = self._cache.get_all()
        return (np.asarray(keys, dtype=np.float32).reshape(-1, self.dimensions),
                np.asarray(values, dtype=np.float32).reshape(-1, self.dimensions))

    def clear(self) -> None:
        """Discard all cached tokens."""
        self._cache.clear()
