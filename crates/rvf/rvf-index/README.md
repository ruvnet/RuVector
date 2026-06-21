# rvf-index

Progressive HNSW indexing with tiered Layer A/B/C search for RuVector Format.

## Overview

`rvf-index` implements a Hierarchical Navigable Small World (HNSW) index optimized for the RVF storage model:

- **Layer A** -- hot vectors, full-precision, in-memory graph
- **Layer B** -- warm vectors, quantized, memory-mapped
- **Layer C** -- cold vectors, compressed, on-disk with lazy loading
- **Progressive build** -- index grows incrementally without full rebuilds
- **Vamana alpha-pruning** -- diversity-aware neighbor selection during build
  (recall@10 0.986 -> 0.996 at ef_search=30, measured at 100k x 64-dim,
  Windows x64 criterion release)
- **Hardened codec** -- INDEX_SEG decoding validates lengths and counts
  against the payload size before allocating (crafted-file DoS resistance)
- **Deterministic search** -- `(distance, id)` tie-breaking for stable result
  ordering across runs

## Usage

```toml
[dependencies]
rvf-index = "0.2"
```

## Features

- `std` (default) -- enable `std` support
- `simd` -- enable SIMD-accelerated distance computations

## License

MIT OR Apache-2.0
