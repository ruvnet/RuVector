"""Embedding based routing over native RuVector HNSW, without network calls."""
from dataclasses import dataclass
import math
import numpy as np
from ruvector import HnswIndex

__all__ = ['SemanticRouter', 'RouteMatch']

@dataclass(frozen=True)
class RouteMatch:
    name: str
    score: float

class SemanticRouter:
    """One representative embedding per route; cosine similarity threshold.

    This is a Python routing policy over native core HNSW, not a binding to
    the separate neural router crate. Embeddings must be supplied by callers.
    """
    def __init__(self, dimensions: int, *, threshold: float = 0.7):
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or not 1 <= dimensions <= 65536:
            raise ValueError('dimensions must be an integer in [1, 65536]')
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError('threshold must be finite in [-1, 1]')
        self.dimensions, self.threshold = dimensions, float(threshold)
        self._vectors = {}
        self._index = HnswIndex.create(dimensions)

    def _vector(self, vector):
        data = np.ascontiguousarray(vector, dtype=np.float32)
        if data.ndim != 1 or len(data) != self.dimensions or not np.isfinite(data).all():
            raise ValueError('embedding must be a finite vector matching dimensions')
        # Reject zero norm; normalize in float64 to avoid float32 overflow.
        norm = np.linalg.norm(data.astype(np.float64))
        if not norm > 0:
            raise ValueError('cosine routing requires nonzero embeddings')
        return np.ascontiguousarray(data.astype(np.float64) / norm, dtype=np.float32)

    def _replace(self, vectors):
        # Build first, then commit. Replacement/deletion cannot retain stale HNSW
        # graph nodes and invalid inserts leave the prior router intact.
        index = HnswIndex.create(self.dimensions)
        for name, vector in sorted(vectors.items()):
            index.insert(name, vector)
        self._index, self._vectors = index, vectors

    def add(self, name: str, embedding):
        if not isinstance(name, str) or not name.strip() or len(name) > 1024:
            raise ValueError('name must be a nonempty string of at most 1024 characters')
        vector = self._vector(embedding)
        self._replace({**self._vectors, name: vector})

    def remove(self, name: str) -> bool:
        if name not in self._vectors:
            return False
        vectors = self._vectors.copy()
        del vectors[name]
        self._replace(vectors)
        return True

    def route(self, embedding, *, k: int = 1) -> list[RouteMatch]:
        vector = self._vector(embedding)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100000:
            raise ValueError('k must be an integer in [1, 100000]')
        if not self._vectors:
            return []
        results = self._index.search(vector, min(k, len(self._vectors)))
        matches = [RouteMatch(name, max(-1., min(1., 1. - distance))) for name, distance, _ in results]
        return sorted((m for m in matches if m.score >= self.threshold), key=lambda m: (-m.score, m.name))

    def __len__(self):
        return len(self._vectors)
