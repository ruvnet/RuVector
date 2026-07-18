# ADR-272: Locality-Sensitive Fingerprinting for Semantic Memory Deduplication

**Status:** Proposed
**Date:** 2026-07-18
**Crate:** `ruvector-lsf-dedup`
**Branch:** `research/nightly/2026-07-18-lsf-memory-dedup`

---

## Context

RuVector's agent memory infrastructure (`ruvector-agent-memory`) implements
_capacity-based_ compaction: given a full store, it evicts entries based on
recency, frequency, or coherence score. That design answers "which memory to
evict?" but not "should this memory be stored at all?"

Agent workflows produce near-duplicate vectors at high rates:
- A coding agent observes the same function signature on consecutive tool calls.
- A conversational agent re-fetches the same user preference across turns.
- A ruFlo workflow loop re-processes the same document fragment.

Without pre-insert dedup, the store grows with semantically redundant entries
that (a) waste capacity, (b) dilute top-k recall, and (c) add noise to coherence
scoring in `ruvector-coherence-hnsw`.

---

## Decision

Add `ruvector-lsf-dedup`: a zero-dependency Rust crate that provides pre-insert
near-duplicate suppression via Locality-Sensitive Fingerprinting.

Three strategies are implemented behind a common `SemanticFingerprinter` trait:

| Strategy | Mechanism | Fingerprint size | Use case |
|----------|-----------|-----------------|---------|
| `SimHasher` | 64-bit hyperplane sign-projection | 8 bytes | Default; fastest |
| `MinHasher` | Jaccard of quantised-bin features | k×4 bytes | Scale-robustness |
| `HybridDedupStore` | SimHash pre-filter + exact cosine | 8 bytes + original vector | Highest precision |

All strategies log every insert decision to a `Vec<DedupDecision>` proof trail.

**API shape intended for production:**

```rust
pub trait SemanticFingerprinter: Send + Sync {
    type Fingerprint: Clone + Send + Sync;
    fn fingerprint(&self, vec: &[f32]) -> Self::Fingerprint;
    fn estimate_similarity(&self, a: &Self::Fingerprint, b: &Self::Fingerprint) -> f32;
}
```

The `DedupStore<H>` and `HybridDedupStore<H>` generic stores wrap any hasher.
The proof trail type `DedupDecision` is stable.

---

## Consequences

**Benefits:**
- Prevents unbounded memory store growth from near-duplicate observations.
- Improves recall quality (no dilution from near-identical entries).
- Improves coherence scoring (less noise in the HNSW graph).
- Zero external dependencies; works on any Rust target including WASM.
- SimHash is scale-free: 8 bytes per entry regardless of dimension.
- Proof trail enables post-hoc audit of dedup decisions.

**Costs:**
- Pre-insert fingerprint computation: ~2 μs (SimHash) to ~80 μs (MinHash) per insert on x86-64.
- Linear scan over stored fingerprints: O(n) per insert (acceptable for n < 10 000; needs bucketing for larger stores).
- False negatives possible when two conceptually similar vectors have cosine < threshold (e.g., different embedding models).
- Threshold must be calibrated per embedding model and domain.

---

## Alternatives Considered

**1. Exact dedup by vector hash**
Rejected. Only catches exact byte-for-byte duplicates; does not handle rephrased or re-observed near-duplicates.

**2. Post-storage dedup (batch sweep)**
Considered. More thorough than pre-insert but requires a background sweep and does not prevent initial storage cost.
Could complement this ADR in future.

**3. IVFPQ near-duplicate index**
Considered. More scalable for millions of entries but requires IVFPQ setup (quantisation, training pass), far heavier than LSF fingerprinting.
Appropriate upgrade path once stores exceed 100 K entries.

**4. Learned similarity head (model inference)**
Rejected for pre-insert use. Model inference adds 10–100 ms per insert; inappropriate for a synchronous insert guard.

---

## Implementation Plan

1. **Now (this PR):** `ruvector-lsf-dedup` crate ships as independent library.
2. **Next:** Wire into `ruvector-agent-memory::DedupStore` as optional feature flag.
3. **Next:** Add bucketed fingerprint index for O(log n) scan.
4. **Next:** Build `ruvector-lsf-dedup-wasm` for edge / Cognitum Seed targets.
5. **Later:** Integrate with `ruvector-proof-gate` for signed dedup decisions.
6. **Later:** Distributed fingerprint gossip for ruvector-raft multi-replica stores.

---

## Benchmark Evidence

Run: `cargo run --release -p ruvector-lsf-dedup`

Dataset: 1,900 vectors, 128 dims, ~36% duplicates (cosine ≥ 0.93 ground truth).

| Strategy | Recall | Precision | FPR   | Time (ms) |
|----------|--------|-----------|-------|-----------|
| SimHash  | 0.969  | 1.000     | 0.000 | 15.12     |
| MinHash  | 0.983  | 1.000     | 0.000 | 153.74    |
| Hybrid   | 1.000  | 1.000     | 0.000 | 17.58     |

All acceptance thresholds exceeded. Hybrid achieves perfect recall and precision
on this dataset (zero false negatives, zero false positives).

---

## Failure Modes

| Failure | Condition | Impact |
|---------|-----------|--------|
| Threshold too tight | Near-duplicates have cosine < threshold | High false-negative rate; duplicates stored |
| Threshold too loose | Unique vectors have cosine > threshold | False positives; legitimate memories rejected |
| Linear scan too slow | Store > 10 000 entries | Insert latency grows > 100 ms |
| MinHash unstable | Embedding magnitudes drift significantly | Feature bins shift; similarity underestimated |
| SimHash collision | Two unrelated vectors share 56+ identical hash bits | Very rare (< 10^-7 per pair at 64 bits) |

---

## Security Considerations

- If the dedup outcome (duplicate/unique) is exposed to callers, adversaries can
  probe for membership inference by crafting near-duplicate queries.
  **Mitigation:** Do not return the `similarity` field in public API; return
  only boolean "stored" or "not stored."
- The proof trail contains insert sequences and similarity scores.
  In multi-tenant deployments, isolate proof trails per tenant namespace.
- Combining LSF dedup with `ruvector-proof-gate` signed writes provides
  tamper-evident evidence that no authorised memory was silently suppressed.

---

## Migration Path

`ruvector-agent-memory` currently has no dedup. Integration requires:

1. Add `ruvector-lsf-dedup` as optional dependency in `ruvector-agent-memory/Cargo.toml`.
2. Wrap `MemoryStore::insert` with an LSF check when feature `dedup` is enabled.
3. Expose `DedupDecision` trail via `MemoryStore::dedup_log()`.

No breaking API changes required. Dedup is additive.

---

## Open Questions

1. **Threshold calibration strategy:** Should the threshold be per-store, per-embedding-model,
   or self-calibrated from observed similarity distributions?
2. **Merge policy:** When a duplicate is detected, should the metadata be merged
   into the existing entry rather than silently dropped?
3. **Distributed dedup:** How should fingerprints be synchronised across
   ruvector-raft replicas to prevent the same duplicate entering two different nodes?
4. **WASM target:** Should `SimHasher` be exposed via `#[wasm_bindgen]` in a
   companion crate, or embedded in `ruvector-wasm` as a feature flag?

---

## Why This Belongs in RuVector

RuVector is explicitly positioned as a "Rust-native cognition substrate for agents."
Agent cognition requires not just storage and retrieval, but **memory hygiene**.
Pre-insert deduplication is the first layer of memory hygiene: it prevents the
store from becoming a noisy, diluted record of redundant observations.

This ADR does not describe a pure experiment. The Hybrid strategy achieves
recall=1.000, precision=1.000, FPR=0.000 on the benchmark dataset with zero
external dependencies and 17 ms total time for 1,900 inserts. It is
production-viable today for stores up to ~10,000 entries.

What should remain behind a feature flag: the proof trail logging (adds
per-decision allocation). The core fingerprinting is always on.

What would make us reject this direction: if production workloads show
false-positive rates > 1% (legitimate memories rejected), the approach
must be replaced with a learned similarity model or an IVFPQ index with
higher precision at the cost of more complex setup.
