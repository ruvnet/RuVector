"""Attention ranking for RAG backed by RuVector Rust bindings."""
from dataclasses import dataclass
import numpy as np
from ruvector import AttentionReranker

__all__ = ["RankedCandidate", "rank"]

@dataclass(frozen=True)
class RankedCandidate:
    index: int
    weight: float

def rank(query, candidates, *, top_k: int | None = None) -> list[RankedCandidate]:
    """Rank by native scaled dot product attention weights.

    Weights sum to one before truncation. This is not a cross encoder:
    order is equivalent to dot product for fixed query and dimension.
    Ties retain the original candidate order.
    """
    q = np.ascontiguousarray(query, dtype=np.float32)
    data = np.ascontiguousarray(candidates, dtype=np.float32)
    if q.ndim != 1 or not q.size or data.ndim != 2 or data.shape[1] != q.size or not data.shape[0]:
        raise ValueError("query and nonempty candidates must have matching dimensions")
    if not np.isfinite(q).all() or not np.isfinite(data).all():
        raise ValueError("inputs must be finite")
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer or None")
    _, weights = AttentionReranker(q.size).rerank(q, data)
    if not np.isfinite(weights).all():
        raise ValueError("attention overflow; rescale input vectors")
    order = np.argsort(-weights, kind="stable")
    return [RankedCandidate(int(i), float(weights[i])) for i in order[:top_k]]
