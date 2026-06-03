---
adr: 196
title: "Temporal-Decay ANN — Recency-Aware Nearest-Neighbour Search for Agent Memory"
status: accepted
date: 2026-06-03
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-189, ADR-183]
tags: [ann, temporal, agent-memory, recency, vector-search, coherence, nightly-research]
---

# ADR-196 — Temporal-Decay ANN: Recency-Aware Nearest-Neighbour Search for Agent Memory

## Status

**Accepted.** Implemented on branch `research/nightly/2026-06-03-temporal-decay-hnsw`
as `crates/ruvector-td-hnsw`.  All 9 unit tests pass; all acceptance tests pass;
build is green with `cargo build --release -p ruvector-td-hnsw`.

---

## Context

RuVector has a growing role as the memory substrate for autonomous agents running
via ruFlo and Claude Flow.  Agent memory has a property absent from most
document databases: **temporal relevance decays**.  A conversation snippet from
30 minutes ago is almost always more useful to a reasoning agent than the same
snippet from 30 days ago, even if both are semantically equidistant from the
current query.

Current RuVector indices (`ruvector-core` HNSW, `ruvector-rairs` IVF,
`ruvector-diskann`) rank candidates purely by geometric distance.  There is no
mechanism to down-weight stale memories without an explicit delete-and-reinsert
cycle.  When an agent has 50,000 memories spanning 6 months, the top-10
nearest neighbours by L2 are almost uniformly drawn from the bulk of the corpus
regardless of age — the 10% that are fresh are chronically under-represented.

The benchmark below (10,000 vectors, 128 dimensions, 1,000 queries, 10% fresh
distribution) demonstrates this precisely: **baseline fresh_recall = 0.095**
(roughly the prior on freshness) vs **temporal-decay fresh_recall = 1.000**
(temporal decay always retrieves fresh results when freshness + proximity
overlap).

---

## Decision

We introduce **`crates/ruvector-td-hnsw`** implementing three variants of
temporal-decay nearest-neighbour search via a shared flat index and a common
`TdIndex` interface.

### Core formula

```
d_eff(q, v) = d_raw(q, v) × temporal_weight(age(v))

temporal_weight(age_secs) = 1.0 + decay_strength × (1.0 − exp(−age_secs / half_life_secs))
```

- At `age = 0`: weight = 1.0 (no penalty)
- At `age = half_life`: weight ≈ 1.0 + 0.632 × decay_strength
- At `age → ∞`: weight → 1.0 + decay_strength (maximum penalty)

### Three variants

| Variant | Distance | Gate | Description |
|---|---|---|---|
| `Baseline` | raw L2² | none | Standard flat search; correctness reference |
| `TemporalDecay` | L2² × weight(age) | none | Recency-biased ranking |
| `CoherenceGated` | L2² × weight(age) | age > T AND d > cutoff → skip | Decay + pruning of stale+distant entries |

### API shape

```rust
pub struct DecayConfig { decay_strength: f32, half_life_secs: f64, ... }
pub struct TdIndex { ... }

impl TdIndex {
    pub fn new(variant: IndexVariant, config: DecayConfig) -> Self;
    pub fn insert(&mut self, id: u64, vector: Vec<f32>, timestamp_secs: u64);
    pub fn search(&self, query: &[f32], k: usize, now_secs: u64) -> Vec<SearchResult>;
    pub fn fresh_recall(&self, query, k, now_secs, recent_threshold_secs) -> f32;
}
```

---

## Consequences

**Positive**
- Fresh-recall improves from ~10% (proportional to corpus freshness fraction) to
  ≥99% on a 10/90 fresh/stale corpus (measured, not estimated).
- CoherenceGated variant is 20% faster than Baseline on the same corpus because
  it prunes stale+distant candidates early: p50 1,432µs vs 1,796µs.
- Zero changes to stored vector format — timestamp is a separate `u64` field, no
  re-embedding required.
- `DecayConfig` is purely additive; setting `decay_strength = 0.0` gives exact
  baseline behaviour.

**Negative / Trade-offs**
- With very strong decay, semantically important historical context can be
  demoted even when it is the closest embedding.  Decay strength and half-life
  must be tuned per use-case.
- Flat (brute-force) search — O(n) per query.  Scalability above ~100k entries
  requires integrating this decay formula into a graph-based index (HNSW) or
  IVF cluster selection, which remains future work.
- CoherenceGate can under-fill k if many stale+distant candidates exist; callers
  must handle `results.len() < k`.

---

## Alternatives Considered

### A. Separate temporal index (time-bucketed B-tree + vector join)
Maintain a B-tree on timestamps, query recent bucket first, then merge with
vector search.  Requires synchronisation between two indexes, complicates
inserts, and produces non-smooth recency transitions.  Rejected for complexity.

### B. TTL-based eviction
Delete stale entries above an age threshold.  Simple but loses historical
context entirely.  An agent that needs to reason about a months-old fact cannot
retrieve it at all.  Rejected for information loss.

### C. Soft-decay at embedding level (time-conditioned embeddings)
Re-embed each document with a temporal token appended.  Requires re-encoding the
full corpus whenever the reference time changes.  Impractical for a live
agent memory system.  Rejected for operational cost.

### D. Score fusion (BM25 freshness score × vector score)
Weight by a separately computed BM25-style freshness signal.  Requires a second
index and inverted list.  Correct direction, but adds infra cost that the
`DecayConfig` replaces with a single function call.  Future integration possible.

---

## Implementation Plan

1. **Now (done)** — `crates/ruvector-td-hnsw` with flat search, three variants,
   9 unit tests, benchmark binary, acceptance test.
2. **Next** — Integrate `DecayConfig` into `ruvector-core` HNSW: apply decay
   weight during neighbor-selection in the greedy layer traversal.
3. **Next** — Integrate into `ruvector-rairs` IVF: apply decay weight during
   candidate re-ranking after list probe.
4. **Later** — Expose via MCP tool surface: `memory_search(query, k, decay_config)`.
5. **Later** — ruFlo workflow: auto-tune `half_life_secs` per agent session based
   on observed retrieval utility.

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-td-hnsw --bin td-benchmark`
on the research branch.  Dataset: 10,000 vectors, 128 dims, 1,000 queries, k=10,
decay_strength=3.0, half_life=3600s, fresh_threshold=3600s.  Distribution: 70%
old (>24h), 20% medium (1-24h), 10% fresh (<1h).

```
OS: linux / ARCH: x86_64

Variant                   N    Dims  Queries  Mean µs      QPS   p50 µs   p95 µs  FreshRec
Baseline (no decay)   10000     128     1000   1807.3      553   1796.0   1911.0     0.095
TemporalDecay         10000     128     1000   1850.9      540   1844.0   1938.0     1.000
CoherenceGated        10000     128     1000   1448.1      691   1432.0   1657.0     1.000

Memory: 528 bytes/entry × 10,000 = 5,156 KB ≈ 5 MB

ACCEPTANCE: PASS
[PASS] TD fresh_recall (1.000) > Baseline (0.095)
[PASS] CG fresh_recall (1.000) within 15% of TD (1.000)
```

---

## Failure Modes

| Failure | Mitigation |
|---|---|
| Over-decay: recent embeddings semantically wrong | Reduce `decay_strength`; validate on held-out query set |
| Under-decay: fresh results not preferred | Increase `decay_strength` or decrease `half_life_secs` |
| CoherenceGate returns fewer than k results | Caller checks `results.len()`; relax `coherence_cutoff` |
| Clock skew between insert and query | Use monotonic counter rather than wall clock |
| Flat search too slow at >100k entries | Future: integrate into HNSW graph traversal |

---

## Security Considerations

- Timestamps are caller-supplied; a malicious caller could inject future
  timestamps to artificially prioritise vectors.
- In multi-tenant deployments, each tenant's `now_secs` must be server-side
  computed to prevent timestamp manipulation.
- No sensitive data is stored in `DecayConfig`; it is pure configuration.

---

## Migration Path

- Existing `TdIndex` users with `DecayConfig::no_decay()` get identical results
  to a plain flat search.
- Future HNSW integration: `SearchParams` gains an `Option<DecayConfig>` field
  behind the `temporal-decay` feature flag.
- No index rebuild required when enabling decay on an existing corpus; timestamps
  must be stored alongside vectors at insert time.

---

## Open Questions

1. What is the right default `half_life_secs` for a conversational agent vs a
   long-horizon research agent?
2. Should `decay_strength` be auto-tuned from retrieval utility signals in ruFlo?
3. Can the coherence gate be expressed as a SIMD-friendly predicate to recover
   speed at scale?
4. How does temporal decay interact with RVF's snapshot format — should the
   reference timestamp be stored in the manifest?
