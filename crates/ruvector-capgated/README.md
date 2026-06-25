# ruvector-capgated

**Capability-gated approximate nearest-neighbour search** — per-vector read access control baked into the retrieval engine, using 64-bit bitset capability tokens. Pure Rust, zero external dependencies, WASM-safe.

Most vector databases enforce access control at the *collection* level: everyone sharing an index can retrieve everyone else's vectors. For multi-tenant agent memory (thousands of agents on one index) that is a design gap, and one-collection-per-tenant does not scale. `ruvector-capgated` makes the access check part of the search itself: each vector carries a required `CapMask`, each query presents a held `CapMask`, and only vectors the querier is authorised for are ever returned.

This is the read-side complement to RuVector's proof-gated writes (ADR-227); see ADR-268.

## Capability model

```rust
use ruvector_capgated::CapMask;

let required = CapMask::single(3);       // vector needs capability bit 3
let holder   = CapMask::single(3).union(CapMask::single(7));
assert!(holder.satisfies(required));      // (holder & required) == required
```

A capability check is one 64-bit bitwise AND — orders of magnitude cheaper than the f32 distance computation it guards.

## Variants

All implement the `CapGatedIndex` trait:

| Variant | Strategy | Recall | Notes |
|---------|----------|--------|-------|
| `PostFilter` | Score all vectors, filter after distance | 100% | Baseline; equivalent to current post-filtering SOTA |
| `EagerMask` | Build authorised bitset first, skip distance for unauthorised | 100% | Latency scales with the *authorised fraction*, not corpus size |
| `CapGraph` | k-NN graph walk with `ef`-bounded exploration | ~90% | Sub-linear node visits; traverses bridge nodes for connectivity |

## Measured results

`cargo run --release -p ruvector-capgated --bin benchmark` (5,000 × 64-dim, 200 queries, x86_64 Linux):

- **Low-access (12.5% authorised):** EagerMask 17,548 QPS / 100% recall@10 — **7.9× faster than PostFilter**.
- **High-access (37.5% authorised):** EagerMask 5,728 QPS / 100% recall@10 — 2.8× faster than PostFilter.

EagerMask latency tracks `authorised_fraction × full-scan latency`, because unauthorised vectors never enter the distance loop.

## Usage

```rust
use ruvector_capgated::{CapGatedIndex, CapMask};
// build an EagerMask index, insert (id, vector, required_mask), then:
// index.search(&query, k, holder_mask) -> Vec<SearchResult>
```

```bash
cargo test  -p ruvector-capgated          # 22 tests
cargo run   --release -p ruvector-capgated --bin benchmark
```

## License

MIT
