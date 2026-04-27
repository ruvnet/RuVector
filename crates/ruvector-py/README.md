# ruvector — Python SDK (M1)

Vector similarity search via RaBitQ 1-bit quantization, implemented in Rust
with native NumPy interop. M1 ships exactly one index class —
`RabitqIndex` — backed by `ruvector_rabitq::RabitqPlusIndex` (symmetric
1-bit scan + exact f32 rerank).

This crate is the Python wheel half of the ruvector workspace; the
underlying algorithms live in `crates/ruvector-rabitq/` and are unchanged
by this binding. The full SDK plan (M1 → M4) is in
[`docs/sdk/`](../../docs/sdk/).

## Install

Once published to PyPI:

```sh
pip install ruvector
```

For local development from a checkout:

```sh
cd crates/ruvector-py
maturin develop --release
pytest tests/
```

`maturin develop` builds the Rust cdylib in-place and links it as
`ruvector._native` so `import ruvector` works from any Python interpreter
in the active virtualenv. The `--release` flag matters: a debug build is
~30× slower on the search loop and will fail the latency acceptance test.

## 30-second example

```python
import numpy as np
import ruvector

# Build an index over 100k random D=128 vectors.
rng = np.random.default_rng(42)
vectors = rng.standard_normal((100_000, 128), dtype=np.float32)
idx = ruvector.RabitqIndex.build(vectors, rerank_factor=20)

# Search the 10 nearest neighbours of a query.
query = vectors[0]
hits = idx.search(query, k=10)
for vid, score in hits:
    print(vid, score)
# 0 0.0
# 12345 0.0023
# ...

# Persist and reload.
idx.save("idx.rbpx")
idx2 = ruvector.RabitqIndex.load("idx.rbpx")
assert idx2.search(query, k=10) == hits
```

## API summary

| Call | Returns | Notes |
|---|---|---|
| `RabitqIndex.build(vectors, *, rerank_factor=20, seed=42)` | `RabitqIndex` | `vectors`: `(n, dim)` C-contig `float32` |
| `idx.search(query, k, *, rerank_factor=None)` | `list[(int, float)]` | `(id, score²)` ascending; `rerank_factor=None` uses the build value |
| `idx.save(path)` / `RabitqIndex.load(path)` | `None` / `RabitqIndex` | `.rbpx` v1 format |
| `len(idx)` / `idx.dim` / `idx.memory_bytes` / `idx.rerank_factor` | `int` | diagnostics |
| `ruvector.RuVectorError` | exception | base of the (future) error tree |
| `ruvector.__version__` | `str` | mirrors `Cargo.toml` |

Non-contiguous or wrong-dtype inputs raise `TypeError` at the boundary
rather than silently copying — predictable beats fast.

## Acceptance gates (M1)

Per `docs/sdk/04-milestones.md`:

  1. `pip install ruvector` (or `maturin develop`) succeeds in <10 s
  2. 100k-vector D=128 search returns in <10 ms (p99 over 100 queries)
  3. Type stubs validate with `mypy --strict`

## Links

  - [SDK plan and milestones](../../docs/sdk/) — M1 through M4 roadmap
  - [Binding strategy](../../docs/sdk/02-strategy.md) — why PyO3 + maturin
  - [API surface sketch](../../docs/sdk/03-api-surface.md) — full Python surface
  - [`ruvector-rabitq`](../ruvector-rabitq/) — the Rust crate this wraps

## License

Dual MIT / Apache-2.0, matching the rest of the ruvector workspace.
