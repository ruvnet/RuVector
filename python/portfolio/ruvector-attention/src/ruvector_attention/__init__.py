"""Validated Python convenience APIs backed by RuVector Rust."""
import numpy as np

_MAX_DIM = 65536

def _integer(value, name, upper=_MAX_DIM):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
        raise ValueError(f"{name} must be an integer in [1, {upper}]")
    return value

def _array(value, ndim, name):
    data = np.ascontiguousarray(value, dtype=np.float32)
    if data.ndim != ndim or not data.size or data.shape[-1] > _MAX_DIM:
        raise ValueError(f"{name} has an invalid or empty shape")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} must be finite")
    return data

from dataclasses import dataclass
from ruvector import AttentionReranker
__all__ = ["AttentionResult", "attend", "AttentionReranker"]

@dataclass(frozen=True)
class AttentionResult:
    blended: np.ndarray
    weights: np.ndarray

def attend(query, candidates) -> AttentionResult:
    """Native scaled dot product attention; keys and values are candidates."""
    q = _array(query, 1, "query")
    data = _array(candidates, 2, "candidates")
    if data.shape[1] != len(q):
        raise ValueError("query and candidate dimensions must match")
    blended, weights = AttentionReranker(len(q)).rerank(q, data)
    if not np.isfinite(blended).all() or not np.isfinite(weights).all():
        raise ValueError("attention overflow; rescale input")
    return AttentionResult(blended, weights)
