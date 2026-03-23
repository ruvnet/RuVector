# ADR-117: Pseudo-Deterministic Canonical Minimum Cut

**Status**: Accepted
**Date**: 2026-03-23
**Authors**: rUv / Claude
**Crates**: `ruvector-mincut` (canonical module), `rvf-types`
**Depends on**: existing `ruvector-mincut`, `canonical` feature, RVF witness layout, current Stoer-Wagner and cactus enumeration support
**References**: Yotam Kenneth-Mordoch, "Faster Pseudo-Deterministic Minimum Cut", arXiv:2602.14550, 2026; Aryan Agarwala and Nithin Varma, "Pseudodeterministic Algorithms for Minimum Cut Problems", arXiv:2512.23468, 2025.

## Context

RuVector already treats minimum cut as a first-class coherence signal across vector memory, graph reasoning, witness generation, and compute gating. Today the stack has two useful but distinct capabilities:

1. **Exact global min-cut computation** for value and partitioning.
2. **Cactus-based enumeration** for cases where all or many minimum cuts matter.

What is still missing is a single **canonical minimum cut** that is stable across runs, replayable in witness logs, and suitable for proof-gated mutation and deterministic control. The current cactus-based canonicalization chooses among enumerated cuts by partition sort. That is useful for enumeration workflows but it does not define a source-anchored, lexicographically unique canonical cut with a graph-theoretic tie-break.

Recent work by Yotam Kenneth-Mordoch introduces a natural pseudo-deterministic tie-breaking rule for minimum cut, yielding a unique canonical minimum cut under a fixed source and vertex ordering while preserving the best known randomized weighted running time in the fast path. This directly matches RuVector's needs for auditability, replay, and structural identity.

## Decision

Add a new source-anchored canonical minimum cut API under `ruvector-mincut/src/canonical/` gated behind the existing `canonical` feature flag.

The canonical cut is defined by the lexicographic tuple:

```
(λ, first_separable_vertex, |S|, π(S))
```

Where:

- **λ** is the global minimum cut value
- **first_separable_vertex** is the first vertex in the fixed global ordering that can be separated from the designated source by some minimum cut
- **|S|** is the cardinality of the source side
- **π(S)** is a secondary priority sum over vertices on the source side, used for contracted graphs, sparsifiers, and dynamic maintenance

The side S is always oriented to contain the designated source vertex.

This new source-anchored canonical API is additive. The existing `CactusGraph::canonical_cut()` remains unchanged for cut enumeration and partition browsing use cases.

### Why this decision

This gives RuVector a canonical cut that is:

- Unique under fixed source and vertex ordering
- Stable across runs
- Hashable into receipts and witnesses
- Compatible with dynamic sparsifiers and contracted node priorities
- Stronger than cactus tie-breaking for replay and governance

This is the right fit for:

- RVF witness chains
- Mincut-gated transformer control
- Coherence-based rollback
- Swarm shard boundary agreement
- Structural delta tracking

### Scope

This ADR introduces:

- A canonical source-anchored min-cut definition
- A ship-now exact implementation
- A fast-path architecture target
- A dynamic maintenance architecture target
- RVF serialization and hashing guidance

This ADR does **not** remove or alter cactus enumeration semantics.

## API

### New public types

```rust
pub struct SourceAnchoredCut {
    pub lambda: FixedWeight,          // cut value (32.32 fixed-point)
    pub source_vertex: VertexId,
    pub first_separable_vertex: VertexId,
    pub side_vertices: Vec<VertexId>, // sorted, source side only
    pub side_size: usize,
    pub priority_sum: u64,
    pub cut_edges: Vec<(VertexId, VertexId)>,
    pub cut_hash: [u8; 32],          // SHA-256
}

pub struct SourceAnchoredConfig {
    pub source: Option<VertexId>,
    pub vertex_order: Option<Vec<VertexId>>,
    pub vertex_priorities: Option<Vec<(VertexId, u64)>>,
}
```

### New public entry points

```rust
pub fn canonical_mincut(
    graph: &DynamicGraph,
    config: &SourceAnchoredConfig,
) -> Option<SourceAnchoredCut>;
```

### Stateful wrapper

```rust
pub struct SourceAnchoredMinCut { /* ... */ }

impl SourceAnchoredMinCut {
    pub fn with_edges(edges: Vec<(VertexId, VertexId, Weight)>,
                      config: SourceAnchoredConfig) -> Result<Self>;
    pub fn canonical_cut(&mut self) -> Option<SourceAnchoredCut>;
    pub fn receipt(&mut self) -> Option<SourceAnchoredReceipt>;
    pub fn insert_edge(&mut self, u: VertexId, v: VertexId, w: Weight) -> Result<f64>;
    pub fn delete_edge(&mut self, u: VertexId, v: VertexId) -> Result<f64>;
}
```

### WASM FFI

```c
int32_t canonical_init(uint32_t num_vertices);
int32_t canonical_add_edge(uint64_t u, uint64_t v, uint64_t weight_fixed);
int32_t canonical_compute(uint64_t source);
const CanonicalMinCutResult* canonical_get_result(void);
int32_t canonical_get_hash(uint8_t* out_buf);
int32_t canonical_get_side(uint64_t* out_buf, uint32_t buf_len);
int32_t canonical_get_cut_edges(uint64_t* out_buf, uint32_t buf_len);
void canonical_free(void);
int32_t canonical_hashes_equal(const uint8_t* a, const uint8_t* b);
```

## Algorithm

### Tier 1: Ship now

Use the exact engine built from existing components.

**Inputs:**
- Connected undirected weighted graph
- Designated source vertex
- Stable global vertex ordering
- Integer vertex priorities, default all ones

**Steps:**

**Step 1: compute the global min-cut value.**
Use the existing exact Stoer-Wagner engine:

```
λ* = global_mincut_value(G)
```

**Step 2: find the first separable vertex.**
Scan vertices in the fixed ordering, skipping the source.

For each vertex v:
- Compute an exact minimum s,t cut between source and v
- If the cut value equals λ*, then v is the `first_separable_vertex`
- Stop at the first such vertex

**Step 3: choose the canonical side.**
Among all minimum s,t cuts for source and `first_separable_vertex`, choose the source side minimizing:

```
(|S|, π(S))
```

**Implementation mechanism:**
- Use capacity perturbation so primary cut value dominates, while secondary cost minimizes cardinality and then priority
- **Vertex side penalties require vertex splitting or equivalent source-side accounting** — perturbation on edge capacities alone does not correctly penalize side membership
- Orient the chosen side to always contain source
- Sort `side_vertices` before materialization

**Practical perturbation rule:**

Transform capacities so every cut minimizes:
1. Original cut value
2. Side cardinality
3. Priority sum

```
C'(e) = M · C(e) + Δ(e)
```

Where M is chosen so no possible sum of secondary costs can outweigh a unit difference in primary cut value.

Safe bound:

```
M > n + Σ π(v) for all v ∈ V
```

For vertex side penalties, use vertex splitting or equivalent side accounting so the cut cost reflects membership on the source side.

**Tier 1 complexity:**

```
T_SW + O(n · T_st)
```

Not the asymptotically best algorithm, but exact, auditable, and easy to test.

**Why Tier 1 first:** Immediate production value for witness determinism, replay, test stability, deterministic gating, and regression baselines for later fast and dynamic engines.

### Tier 2: Fast path

Target the Kenneth-Mordoch fast path:
- Tree packing
- 2-respecting cut search
- Canonical tie-breaking during candidate selection

Target running time: `O(m log² n)` for weighted graphs, matching the best randomized global min-cut bounds while returning the canonical cut.

**Tier 2 implementation goal:** Swap out the repeated exact s,t scans with a specialized canonical search pipeline while preserving the same output struct and hash semantics. This keeps public API stable, witness format stable, and correctness baseline stable.

### Tier 3: Dynamic maintenance

Maintain the canonical cut under insertions and deletions using non-trivial minimum cut sparsifiers and contracted mass priorities.

**Core idea:** Maintain dynamic graph, trivial cut candidate from minimum degree, NMC sparsifier, contracted vertex masses as priorities, and stable inherited vertex order.

At query time:
1. Materialize or query the maintained sparsifier
2. Run canonical source-anchored min-cut on the sparsifier
3. Compare against the trivial minimum-degree cut candidate
4. Lift the result back to original vertices
5. Emit the canonical cut and hash

**Why this matters in RuVector:** Streaming coherence on live graph updates, shard boundary stability in swarms, kHz control loops in Cognitum, incremental structural deltas for RVF.

## RVF Integration

`cut_hash` is a SHA-256 digest over canonical fields using the existing pure `no_std` implementation in `rvf-types`.

### Canonical hash input

Hash the ordered tuple:

```
version ‖ lambda ‖ source_vertex ‖ first_separable_vertex ‖
side_size ‖ priority_sum ‖ sorted_side_vertices
```

**Encoding rules:**
- All integers little-endian
- Source side only
- Vertices sorted ascending by stable global ID
- Include a version tag to allow future format upgrades

**Storage options:**
1. Embed directly into `WitnessHeader.policy_hash` (truncated to 8 bytes)
2. Store as a full 32-byte witness payload in a TLV witness section
3. Optionally store both when policy and structural identity need to be bound together

**Recommended witness label:**

```
WITNESS_KIND_CANONICAL_MINCUT = 0x43 // 'C' for Canonical
```

This should be bound into receipts whenever a mutation, routing decision, attention gate, or rollback depends on structural coherence.

## Coexistence with Cactus Canonicalization

The current `CactusGraph::canonical_cut()` remains.

| Property | Cactus-based (existing) | Source-anchored (new) |
|----------|------------------------|----------------------|
| **Use case** | Enumerate all minimum cuts | Single canonical cut for hashing/replay |
| **Uniqueness anchor** | Lex-smallest original vertex in cactus root | Fixed source + vertex ordering |
| **Tie-breaking** | Partition lexicographic order | `(λ, first_separable_vertex, \|S\|, π(S))` |
| **Dynamic support** | None (rebuild from scratch) | Tier 3 via sparsifier |
| **RVF witness hash** | `canonical_key` (ad-hoc SipHash) | `cut_hash` (SHA-256, RVF-compatible) |
| **Paper basis** | Classical cactus / Dinitz-Karzanov-Lomonosov | Kenneth-Mordoch 2026 |

**Doc note:** `CactusGraph::canonical_cut()` returns a canonical representative for enumeration workflows. `canonical_mincut()` returns a source-anchored canonical cut suitable for hashing, replay, and deterministic control.

## Invariants

The implementation must guarantee:

1. `lambda` equals the true global minimum cut value
2. `side_vertices` always contains `source_vertex`
3. `first_separable_vertex` is the earliest vertex in the fixed order separable by a minimum cut
4. Among all such cuts, `side_vertices` minimizes `(|S|, π(S))`
5. `cut_hash` is stable across runs given identical graph, source, ordering, and priorities
6. All behavior is integer-exact, with no floating-point dependence

## Failure Modes and Mitigations

| Failure | Impact | Mitigation |
|---------|--------|------------|
| Unstable vertex ordering | Canonicality breaks across runs | Require stable global IDs; include ordering version in hash preimage |
| Randomized subroutines | Replay breaks | Provide audit_mode with fixed seed or exact deterministic fallback to Tier 1 |
| Capacity overflow in perturbation | Incorrect cut selection | Use checked `u128` during transformed capacity construction; reject or rescale if exceeded |
| Complement ambiguity | Hash mismatch for same cut | Always orient to side containing `source_vertex` |
| Contracted graph ambiguity | Inconsistent canonical cut in dynamic mode | Use inherited minimum original vertex ID as stable representative; contracted mass as π |

## Implementation Plan

### Phase 1 (complete)

Source-anchored API and exact Tier 1 engine.

**Files:**
- `ruvector-mincut/src/canonical/source_anchored/mod.rs` — core algorithm
- `ruvector-mincut/src/canonical/source_anchored/tests.rs` — 33 tests
- `ruvector-mincut/src/wasm/canonical.rs` — WASM FFI with 7 tests
- `ruvector-mincut/benches/canonical_bench.rs` — criterion benchmarks

### Phase 2

Wire RVF integration.

**Files:**
- `rvf-types/src/witness/canonical_mincut.rs`
- `rvf-types/src/hash/mincut.rs`

**Tests:**
- `no_std` hash parity
- TLV round-trip
- Witness binding compatibility

### Phase 3

Add fast-path engine behind an internal strategy enum.

```rust
pub enum CanonicalEngineKind {
    ExactTier1,
    FastTier2,
    DynamicTier3,
}
```

### Phase 4

Add dynamic maintenance on top of sparsifier infrastructure.

## Acceptance Tests

### Determinism

Run the same graph 1,000 times. Expected: same `lambda`, same `first_separable_vertex`, same `side_vertices`, same `priority_sum`, same `cut_hash`. **Implemented:** `test_hash_stability_1000_iterations`, `test_determinism_100_runs`.

### Correctness

For small graphs where cactus enumeration is feasible: enumerate all minimum cuts, filter to those separating source from `first_separable_vertex`, confirm chosen cut minimizes `(|S|, π(S))`.

### RVF stability

Serialize and deserialize witness payloads across platforms and confirm identical hash bytes.

### Dynamic regression

Under insertion and deletion traces, canonical cut changes only when graph structure changes, not due to engine nondeterminism.

### SHA-256 NIST vectors

Verify against FIPS 180-4 test vectors for empty string and "abc". **Implemented:** `test_sha256_empty`, `test_sha256_abc`.

### Security

- Source always appears on source side (`test_source_always_on_source_side`)
- Different graphs produce different hashes (`test_different_graphs_different_hashes`)
- Constant-time hash comparison in WASM FFI (`canonical_hashes_equal`)
- Null pointer rejection in all FFI functions (`test_wasm_null_safety`)
- Graph size limits enforced (`test_wasm_init_too_large`)

## Consequences

### Positive

- Deterministic structural identity
- Stronger witness semantics
- Stable mincut-based control
- Clean path from software to dynamic and hardware loops
- Additive change with no breakage to cactus users

### Negative

- Tier 1 can be expensive on large graphs (`O(n · T_st)`)
- Capacity perturbation with vertex splitting adds implementation complexity
- Two notions of canonical cut now coexist and must be documented clearly

### Neutral

- Public API surface grows modestly
- Existing cactus code remains fully valid
- Feature-gated behind existing `canonical` flag
