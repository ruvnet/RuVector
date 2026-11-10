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

from ruvector import SonaEngine
__all__ = ["LoraTransform", "SonaEngine"]

class LoraTransform:
    """Inference adapter; fresh native engines apply an identity transform."""
    def __init__(self, dimensions: int):
        self.dimensions = _integer(dimensions, "dimensions", 4096)
        self.engine = SonaEngine(dimensions)

    def transform(self, vectors, *, layer: int | None = None):
        data = _array(vectors, 2, "vectors")
        if data.shape[1] != self.dimensions:
            raise ValueError("vector dimension mismatch")
        if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < self.engine.num_layers):
            raise ValueError("layer must be a valid zero-based layer index")
        result = np.stack([self.engine.apply_micro_lora(row) if layer is None else self.engine.apply_base_lora(layer, row) for row in data])
        if not np.isfinite(result).all():
            raise ValueError("LoRA overflow; rescale input")
        return result
