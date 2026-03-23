# ADR-117: Pseudo-Deterministic Canonical Minimum Cut

**Status**: Accepted
**Date**: 2026-03-23
**Author**: Claude (ruvnet)
**Crates**: `ruvector-mincut` (canonical module), `rvf-types`
**References**: Yotam Kenneth-Mordoch, "Faster Pseudo-Deterministic Minimum Cut" (arXiv, 2026)

## Context

The existing `ruvector-mincut` canonical module (`src/canonical/mod.rs`) uses a cactus representation with lexicographic root selection and leftmost branch traversal to produce reproducible minimum cuts. This approach works well but has two limitations:

1. **Tie-breaking is structural (cactus-based), not source-anchored.** The cactus encodes *all* minimum cuts and then picks among them by sorting partitions lexicographically. This is correct but does not match the stronger uniqueness guarantee from recent theoretical work: given a fixed source vertex and vertex ordering, there is a *unique* canonical minimum cut defined by lexicographic tie-breaking on `(λ, first_separable_vertex, |S|, π(S))`.

2. **No dynamic maintenance.** The current implementation rebuilds the cactus from scratch on every query. The 2026 paper gives a fully-dynamic pseudo-deterministic maintainer for unweighted graphs with polylogarithmic update and Õ(n) query time, enabling canonical cuts to be maintained across insertions and deletions without full recomputation.

3. **No integration with RVF witness chains.** The `CanonicalCutResult` struct returns `canonical_key: [u8; 32]` but this hash is not structurally compatible with RVF witness headers (`rvf-types/src/witness.rs`), making cut receipts ad-hoc rather than first-class audit artifacts.

### What changed in the literature

The Kenneth-Mordoch 2026 paper introduces:

- A pseudo-deterministic algorithm for weighted graphs running in O(m log² n) time — matching the best randomized bounds.
- The first fully-dynamic pseudo-deterministic maintainer for unweighted graphs with polylogarithmic updates and Õ(n) queries.
- A lexicographic tie-breaking mechanism: fix a source vertex v₀ and a total order on vertices. Among all minimum cuts, choose the one where:
  1. The first separable vertex (smallest in the ordering that participates in any minimum cut) anchors the cut.
  2. Among cuts involving that vertex, choose the one with fewest vertices on the source side.
  3. Among ties, minimize a priority sum π(S) over the source side.

This yields a *unique* canonical minimum cut that any correct randomized algorithm will agree on with high probability.

## Decision

Extend the `ruvector-mincut` canonical module with a source-anchored pseudo-deterministic canonical min-cut algorithm, integrated with RVF witness hashing. Ship in three tiers.

### Tier 1: Exact Canonical Engine (Ship Now)

Add a new `CanonicalMinCut` struct and `canonical_mincut()` function alongside the existing cactus-based implementation.

#### Algorithm

```
CANONICAL_MINCUT(G, source=0, order, priority):
    λ* = GLOBAL_MINCUT_VALUE(G)          // existing Stoer-Wagner

    for v in order excluding source:
        (value, S) = MIN_ST_CUT(G, source, v)
        if value ≠ λ*:
            continue

        side = NORMALIZE_SIDE(G, S, source, v)

        return CanonicalMinCut {
            lambda: λ*,
            source_vertex: source,
            first_separable_vertex: v,
            side_vertices: sorted(side),
            side_size: |side|,
            priority_sum: Σ priority[u] for u in side,
            cut_hash: SHA-256(λ* ‖ source ‖ v ‖ priority_sum ‖ side_vertices),
        }

    return None  // disconnected or trivial
```

#### Complexity

- T_global + O(n · T_st) where T_st is the cost of an exact max-flow / min s-t cut.
- Not optimal, but exact and auditable.

#### Minimum-cardinality side selection

To ensure the source side S has minimum cardinality among all minimum s-t cuts with the same value, use **capacity perturbation**:

```
C'(e) = C(e) · M + w_vertex
```

where M > Σ P(v). A single max-flow on transformed capacities simultaneously minimizes:
1. Primary cut weight.
2. Source-side priority sum (secondary).

This avoids needing residual graph condensation or brute-force enumeration.

#### Output struct

```rust
pub struct CanonicalMinCut {
    pub lambda: u64,                       // cut weight (fixed-point)
    pub source_vertex: u32,                // designated source
    pub first_separable_vertex: u32,       // uniqueness anchor
    pub side_vertices: Vec<u32>,           // sorted canonical side
    pub side_size: usize,
    pub priority_sum: u64,                 // contracted mass / vertex priority
    pub cut_edges: Vec<(u32, u32)>,        // optional witness edges
    pub cut_hash: [u8; 32],               // SHA-256 for RVF receipts
}
```

#### RVF integration

The `cut_hash` field uses the same SHA-256 from `rvf-types/src/sha256.rs` (pure `no_std` FIPS 180-4). Hash inputs are serialized in little-endian, matching existing RVF witness conventions:

```
SHA-256(lambda_le8 ‖ source_le4 ‖ first_v_le4 ‖ priority_sum_le8 ‖ side_vertices_le4...)
```

This hash can be embedded directly in `WitnessHeader.policy_hash` (truncated to 8 bytes) or stored as a full 32-byte cut witness in a `WIT_HAS_TRACE` TLV section.

### Tier 2: Fast Path (Near-Optimal)

Swap the inner loop with:

- **Tree packing** for O(m log² n) global min-cut computation.
- **2-respecting cut search** over packed trees.
- **Lex-aware tie-breaking** integrated into the tree search, avoiding redundant s-t cut calls.

This matches the paper's O(m log² n) randomized weighted min-cut bound.

#### Prerequisites

- `ruvector-mincut` already has tree operations (`src/tree/`) and sparsification (`src/sparsify/`).
- Requires adding a tree packing module and 2-respecting cut enumeration.

### Tier 3: Dynamic Path

Maintain the canonical cut across graph mutations using:

- **NMC / weak-NMC style sparsifiers** (building on `src/sparsify/`).
- **Contracted mass as vertex priority** — each sparsified node carries the count of original vertices it represents.
- **Trivial cut comparison** — compare the canonical cut from the sparsifier with the minimum-degree vertex's trivial cut.
- **Stable vertex ordering** — use persistent global IDs from `ruvector-core`.

#### Query path

1. Materialize current sparsifier.
2. Assign each sparsified node: priority = contracted vertex count, lex number = minimum original vertex ID inside contraction.
3. Run `canonical_mincut()` on the sparsifier.
4. Compare with trivial cut of minimum-degree vertex.
5. Return lexicographically smaller result.

#### Use cases

- RVF witness chains with delta cut tracking.
- kHz coherence loops in `ruvector-mincut` coherence gate (ADR-001).
- Live graph memory pruning in `mcp-brain-server`.
- Shard boundary agreement in distributed swarms.

## Relationship to Existing Canonical Module

The existing cactus-based `CactusGraph::canonical_cut()` remains available and is not deprecated. It serves a different purpose:

| Property | Cactus-based (existing) | Source-anchored (new) |
|----------|------------------------|----------------------|
| Uniqueness anchor | Lex-smallest original vertex in cactus root | Fixed source + vertex ordering |
| Tie-breaking | Partition lexicographic order | (λ, first_separable_vertex, \|S\|, π(S)) |
| Dynamic support | None (rebuild from scratch) | Tier 3 via sparsifier |
| RVF witness hash | `canonical_key` (ad-hoc) | `cut_hash` (SHA-256, RVF-compatible) |
| Scope | All minimum cuts via cactus enumeration | Single canonical cut via s-t probing |
| Paper basis | Classical cactus / Dinitz-Karzanov-Lomonosov | Kenneth-Mordoch 2026 |

Users choose based on their needs:
- **Cactus**: when you need to enumerate all minimum cuts or inspect the cactus structure.
- **Source-anchored**: when you need a single reproducible cut with a stable hash for receipts, witnesses, and regression benchmarks.

## Failure Modes and Mitigations

| Failure | Impact | Mitigation |
|---------|--------|------------|
| Unstable vertex IDs | Canonicality breaks across runs | Use persistent global IDs; include ordering version in witness |
| Floating-point weights | Non-deterministic comparison | Quantize to `FixedWeight` (32.32 fixed-point, already in crate) before canonical computation |
| Implicit side orientation | Hash mismatch for same cut | Always orient side to contain designated source vertex |
| Randomized max-flow internals | Replay breaks | Fixed seed in receipt mode; or deterministic backend for audit |

## Acceptance Criteria

### Invariance test

```
For 1000 random seeds:
    canonical_mincut(G) must return:
    - same lambda
    - same first_separable_vertex
    - same sorted side_vertices
    - same cut_hash
```

### Adversarial cases

- Cycles (all cuts equal).
- Complete graphs with uniform weights.
- Ladder graphs with many equal minimum cuts.
- Cactus-style graphs with exponentially many minimum cuts.
- RMAT and LDBC SNB graphs for regression benchmarks.

### Benchmark target

Run the same graph 1,000 times → `cut_hash` is invariant across all runs.

## File Layout

```
crates/ruvector-mincut/
├── src/
│   ├── canonical/
│   │   ├── mod.rs              # existing cactus-based canonical cut
│   │   ├── source_anchored.rs  # NEW: source-anchored pseudo-deterministic cut
│   │   └── tests.rs            # extended with new test cases
│   └── ...
```

## Consequences

### Positive

- **Reproducible cut receipts**: Same graph + same ordering → same `cut_hash`, usable as RVF witness evidence.
- **Theoretical grounding**: Algorithm directly implements the 2026 paper's uniqueness guarantee.
- **Incremental path**: Tier 1 ships immediately using existing Stoer-Wagner; Tier 2/3 can be swapped in without API changes.
- **Composable with sparsifier**: ADR-116's spectral sparsifier can feed the dynamic Tier 3 path directly.

### Negative

- Tier 1 is O(n · T_st), which is slower than cactus-based enumeration for small graphs with many equal cuts.
- Adds a second canonical cut API surface alongside the existing cactus-based one.

### Neutral

- No changes to `ruvector-mincut` public API — new functionality is additive.
- `FixedWeight` and existing graph types are reused without modification.
- Feature-gated behind `canonical` feature flag (already exists).
