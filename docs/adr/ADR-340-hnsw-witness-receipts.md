# ADR-340: Witness-Chained Retrieval Receipts on a Real Multi-Layer HNSW Index

## Status

Proposed. Experimental crate (`ruvector-hnsw-receipt`), not wired into the
default query path of any production index. Follow-up to ADR-304
(`ruvector-retrieval-receipt`).

## Context

ADR-304 landed `ruvector-retrieval-receipt`: witness-chained provenance
receipts (`PerResultReceipt`, `MerkleReceipt`) over ANN query results,
measured against a deliberately brute-force cosine index so the provenance
layer's cost could be isolated from ANN recall. That ADR's own Rejection
Criteria named the gap directly:

> Receipt generation overhead exceeds the 15% threshold once applied on top
> of a real HNSW/ANN index rather than brute force (brute-force search is
> comparatively expensive, which understates the *relative* overhead of the
> receipt layer; this must be re-measured against a cheaper baseline before
> any production claim).

Its Open Questions asked the same thing from the other direction: "Does
composing this layer on top of an approximate (HNSW-family) index change any
of the measured properties, or only the recall dimension this experiment
deliberately excluded?" Neither question had been answered with code or
measurement before this ADR.

## Hypothesis

```text
Given a real multi-layer HNSW index (ruvector-hnsw-repair::HnswGraph, M=16,
M0=32, ef_construction=100) built by ingesting N deterministic vectors
through ruvector_proof_gate::HashChainGate,

when top-k approximate search results are wrapped with a retrieval receipt
(PerResultReceipt or MerkleReceipt, unmodified from ruvector-retrieval-receipt),

then (a) both receipt variants achieve 100% verify_full success, (b)
MerkleReceipt's worst-case proof remains strictly smaller than
PerResultReceipt's at k=10, and (c) receipt construction never perturbs
search result order or membership,

subject to Merkle receipt-build p50 latency remaining under 50% of raw HNSW
search p50 latency (a bar fixed before the benchmark ran, deliberately
looser than ADR-304's 15% so this experiment could also directly re-check
that original, tighter threshold once real numbers existed).
```

## Decision

Add `crates/ruvector-hnsw-receipt`, a small composition crate that:

1. Wraps `ruvector_hnsw_repair::HnswGraph` — a real, from-scratch,
   multi-layer implementation of Malkov & Yashunin's HNSW algorithm, with
   bounded node degree and `ef`-bounded search — instead of brute force.
2. Gates ingestion through `ruvector_proof_gate::HashChainGate`, in lockstep
   with `HnswGraph::insert` (both assign/consume sequential ids from 0), so
   every stored vector carries a real chained `WriteReceipt` exactly as
   `RetrievalIndex` does in ADR-304.
3. Reuses `ruvector-retrieval-receipt`'s receipt cryptography
   (`ReceiptVariant`, `PerResultReceipt`, `MerkleReceipt`, leaf/chain/node
   hashing) unmodified via a path dependency and re-export — this ADR adds
   **zero** new cryptographic code, by design: the question under test is
   purely about composition cost, not about the crypto itself.
4. Exposes `search_raw` (baseline, no receipt work) separately from
   `search_items` (adds `ResultItem` construction) so receipt-build latency
   can be isolated from graph-traversal latency in benchmarking.

## Threat Model

Unchanged from ADR-304 — inherited verbatim since the cryptography is
unmodified. Receipts detect post-issuance mutation of a receipt/result pair;
they do not protect against a dishonest query engine and do not prove
write-chain membership. See `ruvector-retrieval-receipt`'s module docs for
the full statement.

## Evidence

Measured via `cargo run --release -p ruvector-hnsw-receipt --bin benchmark`
at two scales (N=5,000/dims=64 and N=20,000/dims=128, k=10, ef=64, 300
queries each, plus one exact repeat of the first configuration for
timing-variance confirmation). See
`docs/research/nightly/2026-08-26-hnsw-witness-receipts/README.md` for full
methodology and raw, unedited output; do not restate rounded figures here as
a substitute for the actual run.

Headline results:
- Merkle receipt-build p50 overhead vs. raw HNSW search p50: **4.38%** at
  N=5,000, **1.35%** at N=20,000 — both comfortably under this ADR's 50%
  bar and ADR-304's original 15% rejection threshold.
- **100%** `verify_full` success for both variants across all 600 sampled
  queries (300 × 2 scales).
- Merkle worst-case proof bytes (160) remain exactly half of PerResult's
  (320) at k=10, matching ADR-304's brute-force result — this property does
  not depend on which index produced the results.
- The correctness invariant (`search_raw` ids == `search_items` ids, in
  order) held on every one of 600 timed queries, asserted inline in the
  benchmark, not just checked in unit tests.

Unit-level correctness (6 tests in `src/lib.rs`) independently confirms:
write-history verifiability, `search_raw`/`search_items` id agreement,
honest verification for both variants, tamper detection (score mutation),
`NoReceipt` fail-closed behavior, and a recall-sanity bound (rules out a
broken node-id-to-write-receipt mapping producing false-passing timings).

## Consequences

**Positive:**
- Directly closes ADR-304's own named gap with real measurement rather than
  extrapolation: the receipt layer's overhead does not merely look cheap
  because brute force is expensive — it is cheap in absolute terms on a
  real approximate index, and gets relatively cheaper as the index grows
  (receipt cost is O(k), flat; HNSW search cost grows with traversal).
- Establishes the "cheap enough to always-on" production claim ADR-304
  could not make on its own evidence.

**Negative / costs:**
- Recall@10 against cosine ground truth is honestly low (0.31–0.58) in this
  configuration, because `HnswGraph`'s internal ranking uses squared-L2 on
  un-normalized vectors while `ResultItem.score` uses cosine, and
  `HnswConfig::new`'s defaults are untuned for this dataset. This is a
  composition artifact, not a receipt-layer defect — the overhead and
  verification-integrity results do not depend on which metric ranked the
  results — but it means this ADR does **not** claim tuned production
  recall, only measured receipt overhead on genuine approximate search.
- Only two scales measured (N=5,000 / N=20,000); ADR-304's own rejection
  language named N≥100k as the scale requiring re-confirmation, which this
  run does not reach (see Rejection Criteria below, carried forward).
- `ruvector-hnsw-repair`'s graph, while real and multi-layer, is a
  research/repair-focused implementation, not the workspace's primary
  production index (`ruvector-core`). The composition question is answered
  for *an* HNSW-family index, not definitively for the production one.

## Alternatives Considered

- **Modify `ruvector-retrieval-receipt::RetrievalIndex` in place to use
  HNSW.** Rejected: that crate's brute-force choice is a deliberate
  experimental control preserved for comparability; replacing it would
  destroy the ability to compare this ADR's results against ADR-304's.
- **Reimplement HNSW inside the new crate.** Rejected: would duplicate
  ~400 lines of `ruvector-hnsw-repair`'s tested implementation for no
  benefit, and would not satisfy "a real HNSW-family index" in the sense
  the original Next Research item asked for.
- **Compose against `ruvector-core`'s production index instead.**
  Considered, deferred: `ruvector-hnsw-repair`'s public `vectors`/`layers`
  fields made real-cosine rescoring straightforward without duplicating
  internal distance logic; `ruvector-core`'s integration surface needs
  separate scoping (see Open Questions and the nightly report's Next
  Research).

## Implementation Plan

1. (This ADR) Land the experimental crate, benchmark, and tests —
   unintegrated, feature-isolated, matching ADR-304's own step 1.
2. If promoted: integrate as an optional wrapper around
   `ruvector-agent-memory`'s query paths, gated behind a Cargo feature —
   same proposed integration point ADR-304 already named.
3. Re-measure at N≥100k, per this ADR's own Rejection Criteria.
4. Compose against `ruvector-core`'s production index once its distance/
   vector-access API is scoped for this purpose.
5. Root/head signing remains ADR-304's open item and is a shared
   prerequisite before any compliance-grade claim for either crate.

## API Shape

```rust
let index = HnswReceiptIndex::ingest(n, dims, seed); // real WriteReceipt + real HNSW node per vector
let raw_ids = index.search_raw(&query, k, ef);        // baseline: pure HnswGraph::search
let items = index.search_items(&query, k, ef);        // + ResultItem { score, write_receipt }
let receipt = RetrievalReceipt::build(
    ReceiptVariant::Merkle, query_hash(&query), index.index_state_root(), &items,
);
assert!(receipt.verify_full(query_hash(&query), index.index_state_root(), &items));
```

## Feature Flags

None yet — same posture as ADR-304: opt-in by virtue of not being a
dependency of any other crate. A shared `receipts` feature flag on
`ruvector-agent-memory` (proposed in ADR-304) is the integration point for
both crates.

## Benchmark Evidence

See `docs/research/nightly/2026-08-26-hnsw-witness-receipts/README.md` for
the full methodology and raw `cargo run --release` output at both scales
plus the timing-variance repeat.

## Security

Unchanged from ADR-304 (see Threat Model): no new cryptographic code, same
domain-separated hashing, same duplicate-last-node Merkle padding weakness
inherited and not re-addressed here. No new `unsafe` code; no new external
dependencies beyond the crates being composed (`ruvector-hnsw-repair`,
`ruvector-proof-gate`, `ruvector-retrieval-receipt`).

## Governance

Same as ADR-304: receipts are commitments, not authorizations; do not
replace `ruvector-capgated`'s access control.

## Failure Modes

- If `HnswGraph::insert`'s sequential-id invariant (assigns ids from 0 in
  insertion order, upstream and unmodified here) were ever violated, the
  `write_receipts[id]` / `graph.vectors[id]` alignment would silently
  produce wrong `ResultItem.write_receipt` bindings. Guarded by a
  `debug_assert_eq!` in `ingest` and indirectly by the recall-sanity test
  (a broken mapping would show near-zero recall instead of the measured
  0.31–0.58).
- `NoReceipt` verification behavior is unchanged from ADR-304: always
  `false`, never panics, fails closed.

## Migration

N/A — new, unintegrated crate.

## Rollback

Delete `crates/ruvector-hnsw-receipt` and its workspace member entry; no
other crate depends on it.

## Rejection Criteria

This direction should be rejected for production promotion if any of the
following hold on re-measurement:
- Receipt-build overhead exceeds 15% of raw search p50 at N≥100k (the scale
  ADR-304 itself named and this run did not reach).
- `verify_full` success drops below 100% on any honest result set at any
  scale.
- The correctness invariant (`search_raw` ids == `search_items` ids, in
  order) fails on any query.
- Composing against `ruvector-core`'s production index (Open Questions)
  produces a materially different overhead ratio than measured here against
  `ruvector-hnsw-repair`.

## Open Questions

- Does the overhead ratio hold when composed against `ruvector-core`'s
  production index rather than `ruvector-hnsw-repair`'s research/repair
  graph?
- Does tuning `HnswConfig` (M, ef_construction) or switching
  `ResultItem.score` to the graph's native L2 ranking close the observed
  recall gap without changing the overhead conclusion?
- What is the overhead once root/head signing (shared open item with
  ADR-304) is added to the timed path?
