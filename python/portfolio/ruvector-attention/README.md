# ruvector-attention

![RuVector attention animated overview](https://raw.githubusercontent.com/ruvnet/RuVector/main/docs/python-portfolio/attention.svg)

Native scaled dot product attention for embedding aggregation.

## Getting started

Requires Python 3.10 or later and a platform supported by the `ruvector==0.1.4` native wheel.

```sh
python -m pip install ruvector-attention==0.2.0
```

```python
from ruvector_attention import attend
result = attend([1, 0], [[1, 0], [0, 1]])
assert result.weights[0] > result.weights[1]
assert result.blended.shape == (2,)
```

## How it works

Python sequences → finite contiguous float32 validation → shared RuVector native Rust runtime → typed Python or NumPy results. No Node.js process is used.

## Scope and limits

This exposes scaled dot product attention with candidates as both keys and values. It does not expose every npm attention mechanism or a trained cross encoder. Candidate ordering is equivalent to dot product.

This is a Python convenience API backed by the shared native runtime, not complete npm API parity. Invalid dimensions and nonfinite inputs are rejected.

## Documentation

[Portfolio and architecture](https://github.com/ruvnet/RuVector/tree/main/python/portfolio) · [Source](https://github.com/ruvnet/RuVector/tree/main/python/portfolio/ruvector-attention) · [Issues](https://github.com/ruvnet/RuVector/issues)
