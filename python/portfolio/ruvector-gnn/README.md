# ruvector-gnn

![RuVector gnn animated overview](https://raw.githubusercontent.com/ruvnet/RuVector/main/docs/python-portfolio/gnn.svg)

Native graph neural network neighborhood encoding.

## Getting started

Requires Python 3.10 or later and a platform supported by the `ruvector==0.1.4` native wheel.

```sh
python -m pip install ruvector-gnn==0.2.0
```

```python
from ruvector_gnn import GraphEncoder
encoder = GraphEncoder(2, 4, heads=2)
encoded = encoder.encode([1, 0], [[0, 1], [1, 1]])
assert encoded.shape == (4,)
```

## How it works

Python sequences → finite contiguous float32 validation → shared RuVector native Rust runtime → typed Python or NumPy results. No Node.js process is used.

## Scope and limits

Fresh weights are random. This is inference only, not training or a demonstrated ranking improvement. The exposed native GnnLayer supports JSON weight serialization; load only trusted model files.

This is a Python convenience API backed by the shared native runtime, not complete npm API parity. Invalid dimensions and nonfinite inputs are rejected.

## Documentation

[Portfolio and architecture](https://github.com/ruvnet/RuVector/tree/main/python/portfolio) · [Source](https://github.com/ruvnet/RuVector/tree/main/python/portfolio/ruvector-gnn) · [Issues](https://github.com/ruvnet/RuVector/issues)

`GraphEncoder` caps persistent native parameter storage at 64 MiB (temporary inference allocations are additional). Neighbor weights are normalized with float64 accumulation before conversion to float32; overflowing outputs raise `ValueError`.
