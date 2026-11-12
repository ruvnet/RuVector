"""Embedding clustering backed by RuVector Rust bindings."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from ruvector import kmeans

__all__ = ["ClusterResult", "cluster"]

@dataclass(frozen=True)
class ClusterResult:
    labels: NDArray[np.int64]
    centroids: NDArray[np.float32]
    cohesion: NDArray[np.float32]
    sizes: NDArray[np.int64]

def cluster(vectors, k: int, *, iterations: int = 20) -> ClusterResult:
    """Cluster embeddings with the native Rust Lloyd implementation.

    Converts numeric array inputs to contiguous float32. Cohesion is cosine
    similarity, not Euclidean inertia. Labels are not stable identifiers.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 10000:
        raise ValueError("iterations must be between 1 and 10000")
    data = np.ascontiguousarray(vectors, dtype=np.float32)
    if data.ndim != 2 or not all(data.shape) or k > data.shape[0]:
        raise ValueError("vectors must be nonempty rows by dimensions and k <= rows")
    if not np.isfinite(data).all():
        raise ValueError("vectors must be finite")
    return ClusterResult(*kmeans(data, k, iters=iterations))
