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
| `HierarchicalCapGraph` | `CapGraph` + sparse top layer for entry-point seeding | ~58-100%, selectivity- and budget-dependent | Isolates seeding quality from degree; see nightly research below |

## Nightly research: low-selectivity audit (2026-10-08)

At the 12.5%-37.5% selectivity measured above, `CapGraph` looks solid. It was never tested near the 1-2% selectivity this crate's own "thousands of agents on one index" use case implies. A nightly run tested whether ACORN's two levers (γ-augmented degree, hierarchical seeding) fix that: **both were rejected** at this crate's production-facing `ef_multiplier=30` default — denser graphs *reduce* recall at that budget, not improve it, because the visited-node cap gets consumed on a near-field that is almost entirely unauthorised at low selectivity before the search is forced outward. γ-augmentation does win, even reaching perfect recall, but only once the budget allows visiting ≥~25% of the corpus. A second finding: `CapGraph`'s recall ceiling is <100% even visiting the *entire* graph, because its k-NN adjacency is directed and a small fraction of nodes are nobody's near neighbour — a pre-existing connectivity gap, not introduced by this research. Full report, raw numbers, and root-cause analysis: `docs/research/nightly/2026-10-08-acorn-capgated-selectivity/README.md` (ADR-352).

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
cargo test  -p ruvector-capgated          # 28 tests
cargo run   --release -p ruvector-capgated --bin benchmark
cargo run   --release -p ruvector-capgated --example selectivity_acorn_bench
```

## License

MIT
