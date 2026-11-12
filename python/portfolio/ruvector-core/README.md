# ruvector-core

![RuVector core animated overview](https://raw.githubusercontent.com/ruvnet/RuVector/main/docs/python-portfolio/core.svg)

Native vector search with ergonomic batch ingestion.

## Getting started

Requires Python 3.10 or later and a platform supported by the `ruvector==0.1.4` native wheel.

```sh
python -m pip install ruvector-core==0.2.0
```

```python
from ruvector_core import VectorStore
store = VectorStore(2)
store.add([[1, 0], [0, 1]])
assert store.search([1, 0], k=1)[0].id == 0
```

## How it works

Python sequences → finite contiguous float32 validation → shared RuVector native Rust runtime → typed Python or NumPy results. No Node.js process is used.

## Scope and limits

HNSW is approximate. Metadata filters are applied after ANN retrieval, with Collection overfetch retries. No embedding model is included.

This is a Python convenience API backed by the shared native runtime, not complete npm API parity. Invalid dimensions and nonfinite inputs are rejected.

## Documentation

[Portfolio and architecture](https://github.com/ruvnet/RuVector/tree/main/python/portfolio) · [Source](https://github.com/ruvnet/RuVector/tree/main/python/portfolio/ruvector-core) · [Issues](https://github.com/ruvnet/RuVector/issues)

Coordinates are limited to absolute values of `1e10` and dimensions to 65,536, preventing overflow in native float32 distance accumulation. Rescale larger vectors before ingestion or querying.

Cosine vectors are normalized in float64 before float32 conversion to preserve tiny nonzero directions. Zero vectors are rejected for cosine; other metrics retain their original magnitudes.
