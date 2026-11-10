"""Validated Python convenience APIs backed by RuVector Rust."""
import numpy as np

_MAX_DIM = 65536

def _integer(value, name, upper=_MAX_DIM):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
        raise ValueError(f"{name} must be an integer in [1, {upper}]")
    return value

def _array(value, ndim, name, metric):
    data = np.ascontiguousarray(value, dtype=np.float64)
    if data.ndim != ndim or not data.size or data.shape[-1] > _MAX_DIM:
        raise ValueError(f"{name} has an invalid or empty shape")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} must be finite")
    if np.any(np.abs(data) > 1e10):
        raise ValueError("vector magnitude must not exceed 1e10 for stable native metrics")
    if metric == "cosine":
        scale = np.max(np.abs(data), axis=-1, keepdims=True)
        if np.any(scale == 0):
            raise ValueError("cosine vectors must have nonzero norm")
        data = data / scale
        data = data / np.linalg.norm(data, axis=-1, keepdims=True)
    return np.ascontiguousarray(data, dtype=np.float32)

from ruvector import Collection, HnswIndex, RabitqIndex, SearchHit
__all__ = ["VectorStore", "Collection", "HnswIndex", "RabitqIndex", "SearchHit"]

class VectorStore:
    """Validated batch ingestion and search over the native HNSW Collection."""
    def __init__(self, dimensions: int, *, metric: str = "cosine"):
        self.dimensions = _integer(dimensions, "dimensions")
        if metric not in {"cosine", "euclidean", "dot", "manhattan"}:
            raise ValueError("unsupported metric")
        self.metric = metric
        self.collection = Collection.create(dimensions, metric=metric)

    def add(self, vectors, *, metadata=None):
        data = _array(vectors, 2, "vectors", self.metric)
        if data.shape[1] != self.dimensions:
            raise ValueError("vector dimension mismatch")
        if metadata is not None and (len(metadata) != len(data) or any(m is not None and not isinstance(m, dict) for m in metadata)):
            raise ValueError("metadata must contain one dict or None per vector")
        return self.collection.insert_batch(data, metadatas=metadata)

    def search(self, query, *, k: int = 10, filter=None):
        q = _array(query, 1, "query", self.metric)
        if len(q) != self.dimensions:
            raise ValueError("query dimension mismatch")
        _integer(k, "k", 100000)
        return self.collection.search(q, k, filter=filter)

    def __len__(self):
        return len(self.collection)
