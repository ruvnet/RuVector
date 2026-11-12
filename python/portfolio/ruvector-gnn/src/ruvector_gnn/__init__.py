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

from ruvector import GnnLayer
__all__ = ["GraphEncoder", "GnnLayer"]

class GraphEncoder:
    """Validated native GNN inference. Fresh weights are random, not trained."""
    def __init__(self, input_dim: int, hidden_dim: int, *, heads: int = 1):
        _integer(input_dim, "input_dim", 4096)
        _integer(hidden_dim, "hidden_dim", 4096)
        _integer(heads, "heads", hidden_dim)
        if hidden_dim % heads:
            raise ValueError("heads must divide hidden_dim")
        # One input projection, eleven hidden projections, twelve biases,
        # and LayerNorm gamma/beta. Bound persistent f32 storage to 64 MiB.
        parameter_count = input_dim * hidden_dim + 11 * hidden_dim ** 2 + 14 * hidden_dim
        if parameter_count > 16 * 1024 * 1024:
            raise ValueError("GNN parameters exceed the 64 MiB allocation limit")
        self.layer = GnnLayer(input_dim, hidden_dim, heads)

    def encode(self, node, neighbors, *, weights=None):
        node = _array(node, 1, "node")
        data = np.ascontiguousarray(neighbors, dtype=np.float32)
        if len(node) != self.layer.input_dim or data.ndim != 2 or data.shape[1] != len(node) or not np.isfinite(data).all():
            raise ValueError("neighbors must be finite with shape (n, input_dim)")
        if weights is not None:
            weights = np.ascontiguousarray(weights, dtype=np.float64)
            if weights.ndim != 1 or len(weights) != len(data) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
                raise ValueError("weights must match neighbors and be finite nonnegative with positive sum")
            # Scale before summing, avoiding overflow even near float64 max.
            weights = weights / weights.max()
            weights = np.ascontiguousarray(weights / weights.sum(dtype=np.float64), dtype=np.float32)
        result = self.layer.forward(node, data, weights)
        if not np.isfinite(result).all():
            raise ValueError("GNN overflow; rescale input")
        return result
