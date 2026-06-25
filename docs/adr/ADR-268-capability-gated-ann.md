# ADR-268: Capability-Gated ANN Search

**Status**: Proposed  
**Date**: 2026-06-25  
**Author**: Nightly Research Agent  
**Branch**: `research/nightly/2026-06-25-capability-gated-ann`  
**Crate**: `crates/ruvector-capgated`  
**Related**: ADR-227 (Proof-Gated Writes), ADR-240 (Coherence-HNSW), ADR-256 (Hybrid Search)

---

## Context

RuVector stores vectors that represent memory, knowledge, and agent state.  In
multi-agent deployments — and increasingly in enterprise RAG pipelines — the
**right to retrieve** is not uniform.  An agent managing user A's memory should
not be able to recall user B's embeddings, even if they are geometrically close.
A security-cleared retrieval pipeline should not surface classified embeddings to
uncleared queriers.

Current state of the ecosystem:

- **ADR-227 (proof-gated writes)**: vectors can only be *written* if the writer
  presents a valid proof token.  There is no symmetric read-time access control.
- **ACORN / Milvus RBAC / Qdrant payload-filter**: these systems filter *after*
  ANN search — meaning the ANN engine first retrieves candidates from the full
  corpus, then drops unauthorised ones.  This wastes distance computations and,
  more importantly, leaks the existence of nearby vectors to any observer who
  can count results or measure latency.
- **pgvector / Weaviate class-level permissions**: access control is at the
  collection or class level, not per-vector.

None of these systems embed access control *into the retrieval index* at the
per-vector granularity.  For agent memory, this matters: a single index may hold
memories belonging to hundreds of different agent sessions, each of which should
be isolated.

---

## Decision

Introduce `crates/ruvector-capgated` as a standalone Rust crate implementing
**capability-gated ANN search**: each stored vector carries a `CapMask` (64-bit
bitset of required capabilities) and each query presents a `CapMask` (held
capabilities).  A result is returned only if the querier holds all required bits.

Three retrieval variants are implemented and benchmarked:

| Variant | Strategy | Recall@10 | Mean Latency | QPS |
|---------|----------|-----------|--------------|-----|
| **PostFilter** | Compute all distances, discard unauthorised | 100% | 494 μs | 2,023 |
| **EagerMask** | Skip distance computation for unauthorised | 100% | 175 μs | 5,728 |
| **CapGraph** | k-NN graph walk, ef-bounded exploration | 90.6% | 289 μs | 3,466 |

*(Numbers from n=5,000 × 64-dim, 37.5% access ratio, 200 queries, release build on x86_64.)*

The public API is the `CapGatedIndex` trait:

```rust
pub trait CapGatedIndex {
    fn insert(&mut self, id: usize, vector: Vec<f32>, required: CapMask);
    fn search(&self, query: &[f32], k: usize, holder: CapMask) -> Vec<SearchResult>;
    fn name(&self) -> &'static str;
}
```

This trait shape should survive into production.

---

## Consequences

### Positive

- **Per-vector read access control**: the only RuVector system to enforce
  capability checks at vector granularity during retrieval.
- **EagerMask is strictly better than PostFilter** at low access ratios: at
  12.5% access ratio, EagerMask is 7.9× faster than PostFilter while
  achieving identical recall.
- **CapGraph achieves sub-linear recall potential**: graph-based traversal
  visits only ef=300 nodes out of 5,000 to find 90.6% of ground-truth results.
- **No external dependencies**: zero-dep pure Rust, WASM-safe.
- **Composable with existing RuVector primitives**: `CapMask` can wrap any
  inner ANN backend (HNSW, LSM-ANN, PQ-ADC).

### Negative / Risks

- **CapGraph build is O(n²·d)**: the brute-force k-NN graph build in this PoC
  takes 2.3 s for n=5,000.  Production requires incremental insertion (HNSW-
  style) or an approximate k-NN graph (RNG, random projection).
- **Graph traversal leaks access patterns**: in the "transparent traversal"
  model, unauthorised node coordinates are implicitly traversed (but not
  returned).  A timing side-channel could reveal existence.  A "strict
  isolation" model (only expand authorised nodes) closes this but degrades
  recall further.
- **CapGraph recall degrades with low access ratio**: at 12.5% access,
  recall@10 is 0.869; at 37.5% it is 0.906.  This is expected — the graph
  is built on the full corpus and the authorised subgraph may be disconnected.
- **64-bit cap space only**: the current `CapMask` is a fixed 64-bit integer.
  For >64 roles, a variable-length bitset or hierarchical capability tree
  is needed.

---

## Alternatives Considered

### 1. Post-hoc SQL-style WHERE filter (status quo)

All major vector databases (Qdrant, pgvector, Milvus, Weaviate) perform
access control as a post-retrieval filter.  This is simple but: (a) wastes
distance computation on unauthorised vectors, (b) requires k' >> k candidates
to guarantee k authorised results, (c) leaks the existence of unauthorised
neighbours via timing.

**Rejected**: misses the EagerMask speedup at low access ratios; no index-level
isolation.

### 2. Separate index per access group

Maintain one HNSW/LSM index per capability group.  A querier with 3 caps
holds 3 separate indices.

**Rejected**: memory cost is O(n × 2^caps) in the worst case; search requires
merging multiple indices; impractical for fine-grained capabilities.

### 3. Encrypted vector retrieval (homomorphic / SEAL)

Encrypt vectors before storing; query against encrypted corpus.

**Rejected**: 3–6 orders of magnitude slower; no practical Rust SEAL binding
with HNSW; out of scope for this nightly.  Worth a separate ADR as a long-term
research direction.

### 4. Capability-aware quantization (cap bits in PQ codes)

Embed capability bits directly into PQ codes; filter at the codebook-lookup
stage.

**Interesting future direction**: would enable compressed + access-controlled
retrieval in one pass.  Documented as a "next" item.

---

## Implementation Plan

### Phase 1 (this PR)

- `crates/ruvector-capgated`: trait, CapMask, Oracle, PostFilter, EagerMask,
  CapGraph with batch_build.
- Benchmark binary with acceptance tests.
- ADR, research document, gist.

### Phase 2 (production hardening)

- Replace O(n²) graph build with incremental HNSW-style insertion.
- Variable-length CapMask for >64 capabilities.
- Integrate EagerMask as a filter layer in `ruvector-core`'s query pipeline.
- MCP tool wrapper: `vector_memory_search(query, capability_token)`.
- Proof-gate integration: writes require proof, reads require capability token.

### Phase 3 (research)

- Capability-aware PQ quantization.
- Strict isolation mode with recall characterisation.
- Timing side-channel analysis and mitigation.
- Distributed capability-gated search with ruFlo workflow.

---

## Benchmark Evidence

Hardware: x86_64 Linux  
Rust: 1.94.1  
Command: `cargo run --release -p ruvector-capgated --bin benchmark`

**Scenario: high-access (37.5% authorised)**

| Variant | Mean(μs) | p50(μs) | p95(μs) | QPS | Recall@10 | Pass |
|---------|----------|---------|---------|-----|-----------|------|
| PostFilter | 494 | 493 | 552 | 2,023 | 1.000 | ✓ |
| EagerMask | 175 | 167 | 206 | 5,728 | 1.000 | ✓ |
| CapGraph | 289 | 281 | 329 | 3,466 | 0.906 | ✓ |

**Scenario: low-access (12.5% authorised)**

| Variant | Mean(μs) | p50(μs) | p95(μs) | QPS | Recall@10 | Pass |
|---------|----------|---------|---------|-----|-----------|------|
| PostFilter | 450 | 456 | 487 | 2,221 | 1.000 | ✓ |
| EagerMask | 57 | 54 | 74 | 17,548 | 1.000 | ✓ |
| CapGraph | 295 | 288 | 358 | 3,396 | 0.869 | ✓ |

**Key observation**: EagerMask provides the best latency when access ratio is
low, as it skips distance computation for unauthorised vectors entirely.

---

## Failure Modes

1. **Access ratio → 0%**: all variants degrade gracefully; PostFilter and
   EagerMask return empty results immediately; CapGraph returns empty after
   exhausting ef budget.
2. **Capability token forgery**: `CapMask` is a plain `u64` in this PoC.
   Production requires cryptographic signing (integrate with ADR-227
   proof-gate) to prevent queriers from self-asserting capabilities.
3. **Graph disconnection**: if authorised vectors form isolated clusters in
   the proximity graph, CapGraph may miss them entirely.  Mitigation: increase
   ef_multiplier; ensure n_entry_points covers the authorised subgraph.
4. **Capability inflation**: if a querier holds all 64 bits (`CapMask::ALL`),
   all vectors are accessible — equivalent to no access control.  Applications
   must not hand out ALL-capable tokens unless intentional.

---

## Security Considerations

- **This PoC does not provide cryptographic access control**.  Any caller can
  construct any `CapMask` and pass it to `search`.  Production must gate token
  creation behind a proof-gate (ADR-227) or a signed JWT / capability cert.
- **Timing side-channels**: EagerMask and PostFilter latency is proportional to
  the authorised fraction, leaking the fraction to a timing observer.  For high-
  sensitivity deployments, constant-time execution is required.
- **CapGraph traversal leakage**: the graph walk visits unauthorised nodes as
  bridge nodes.  Strict isolation (expand only authorised nodes) prevents this
  at a recall cost.

---

## Migration Path

1. Add `ruvector-capgated` as a dependency in `ruvector-core`.
2. Wrap existing `HnswIndex` with a `CapGatedWrapper<HnswIndex>` using the
   EagerMask strategy.
3. Expose via `ruvector-server` as `POST /search` with `capability_token` field.
4. Wire into MCP tool surface as `vector_memory_search(query, token)`.

---

## Open Questions

1. Should `CapMask` be variable-length (e.g. `Vec<u64>`) to support >64 roles?
2. Should proof-gate verification (ADR-227) be composable with CapMask in a
   single call?
3. Is the strict-isolation CapGraph worth implementing as a separate variant for
   high-security deployments?
4. Can capability bits be embedded in PQ codebooks for compressed + gated search?
5. What is the right ef_multiplier trade-off for production use (latency vs
   recall)?
