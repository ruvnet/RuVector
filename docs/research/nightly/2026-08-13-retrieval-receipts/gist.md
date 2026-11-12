# Retrieval Receipts: Making ANN Query Results Auditable

## Problem

Vector databases give AI agents memory. Once an agent retrieves a result
set and acts on it, that result set becomes evidence — for a citation, a
compliance decision, a trading action, a diagnosis. But no major vector
database (Milvus, Qdrant, Weaviate, Pinecone, LanceDB, FAISS, pgvector,
Chroma, Vespa) produces cryptographic evidence of *what a query actually
returned*. If a retrieval layer bug — or a compromised component — swaps,
drops, or reorders results between the index and the agent's context
window, nothing detects it. The agent's downstream action is built on
unverifiable evidence.

RuVector already solved half of this problem. `ruvector-proof-gate`
(ADR-227) wraps vector *writes* in a SHA-256 hash chain or Merkle
Mountain Range, so every ingested vector carries a tamper-evident receipt.
That answers "was this vector written honestly?" It does not answer "was
this vector — and only this vector, in this rank, with this score —
actually what a specific query returned?"

## Hypothesis

Given a 5,000-vector index (ingested through `ruvector-proof-gate`'s
`HashChainGate`, so every vector already has a `WriteReceipt`), wrapping a
top-10 query's result set in a cryptographic receipt should:

1. Detect any tampering with the result set (score mutation, ID
   substitution, reordering, or corruption of a result's underlying write
   provenance) with 100% reliability.
2. Do so with generation overhead under 15% of the search it accompanies.
3. Offer at least one variant whose single-result verification proof is
   asymptotically smaller than the naive approach.

## Technical Design

The core primitive is a `ResultItem`: `{vector_id, rank, score,
write_receipt}`. The `write_receipt` field is the design's load-bearing
decision — it's the actual `WriteReceipt` `ruvector-proof-gate` produced
when that vector was ingested, not a re-derived stand-in. Each result
leaf binds *copies* of that receipt's `gate_variant`, `chain_commitment`,
and `payload_hash`, so a receipt/result pair cannot be mutated after
issuance without verification failing.

The threat model must be stated plainly: receipts are unsigned
commitments produced by the query engine itself. They detect
**post-issuance mutation** of a receipt/result pair (in transit or in
storage). They do **not** protect against a dishonest query engine
(nothing binds a score to an actual cosine computation, or the committed
set to the true top-k), and they do **not** prove write-chain membership
— verification never consults the write gate, so mutating the ingestion
history after issuance leaves existing receipts verifying. Anchoring
leaves to `MerkleGate`'s MMR inclusion proofs is the named future-work
item that would make the write→read link a real membership binding.

Two structured receipt variants wrap a query's k-result set:

**`PerResultReceipt`** — a sequential SHA-256 chain over the k result
leaves, structurally identical to `HashChainGate`'s design but scoped to
one query instead of the whole write history. Verifying result `i` in
isolation (without trusting a live gate instance) means replaying the
chain from a fixed genesis using leaves `0..=i`: O(i) work, O(i) bytes.

**`MerkleReceipt`** — a binary Merkle tree over the k leaves, with
domain-separated leaf/internal-node hashing (distinct byte prefixes
`b"...leaf:"` and `b"...node:"`) to prevent type confusion between tree
positions. A single result's inclusion proof is the sibling path to the
root: O(log k) work, O(log k) bytes — independent of *which* result and
independent of k for practical purposes.

Both variants are implemented against a real (not mocked) brute-force
cosine index whose ingestion path is a genuine `HashChainGate`.
Brute force is deliberate, not a shortcut: the experiment isolates the
provenance layer's cost from ANN approximation quality. Composing this on
top of an HNSW-style index would conflate two independently falsifiable
questions — receipt overhead and recall loss — into one number that
couldn't cleanly attribute a regression to either cause.

## Implementation

`crates/ruvector-retrieval-receipt`, ~700 lines of Rust across four files:

- `index.rs` — the write-gate-backed brute-force index and a deterministic
  xorshift64 dataset/query generator (same construction as
  `ruvector-proof-gate`'s existing `synthetic_payloads`, for
  reproducibility without an RNG dependency).
- `receipt.rs` — `PerResultReceipt` and `MerkleReceipt`, including Merkle
  tree construction, inclusion-proof generation, and verification.
- `lib.rs` — a unifying `RetrievalReceipt` enum plus 14 unit tests: honest
  verification for both variants, four independently tested tamper kinds
  (score mutation, reordering, cross-query ID substitution, gate-variant
  substitution), empty-result fail-closed behavior, and a direct
  assertion that Merkle proof bytes are smaller than per-result proof
  bytes at k=10.
- `bin/benchmark.rs` — the measurement harness below.

Total new dependency surface: zero. `sha2` was already a
`ruvector-proof-gate` dependency; this crate adds only a path dependency
on `ruvector-proof-gate` itself, reusing its `HashChainGate`, `WriteGate`,
`WritePayload`, and `WriteReceipt` types directly.

## Actual Benchmark Evidence

`cargo run --release -p ruvector-retrieval-receipt --bin benchmark -- 5000
128 10 200` — 5,000 vectors, 128 dimensions, k=10, 200 queries, 200 tamper
trials per structured variant (50 per kind × 4 kinds):

```text
baseline brute-force search: mean=1114440ns p95=1371306ns over 200 queries

variant               gen_mean_ns     gen_p95_ns  verify_worst_ns    proof_bytes   total_bytes_mean    tamper_detect
NoReceipt                     237            278                0              0                0.0              n/a
PerResultReceipt            18310          30103             8229            320              640.0          200/200
MerkleReceipt               19582          26120             3839            160              352.0          200/200

tamper detection 100% across all kinds: true
merkle worst-case proof bytes (160) < per-result worst-case proof bytes (320): true
generation overhead < 15% of baseline search: merkle=1.8% per_result=1.6% -> true

ACCEPTANCE RESULT: ACCEPT
```

`cargo test --release -p ruvector-retrieval-receipt`: 14/14 passing.
The 200/200 tamper-rejection result is expected from SHA-256 by
construction (a mutated preimage changes its hash) — it is a regression
check on the implementation, not an empirical detection rate.

One caveat on the proof-size comparison: `PerResultReceipt`'s proof is
defined as the genesis-anchored chain replay (`(idx+1)*32` bytes); a
head-anchored verifier would need only the `k−idx` suffix, so the "160 vs
320 bytes" gap at the worst index is a property of that baseline
definition. The durable claim is the asymptotic one — O(log k) Merkle
proofs vs O(k) chain replay regardless of which result is disputed. Under
the same definition the gap at k=100 works out to 256 vs ~3,200 bytes
worst case — stated as arithmetic extrapolation, not a re-run benchmark
result, and flagged in the ADR's rejection criteria as needing direct
re-measurement before being treated as a production claim.

## Limitations

- Brute-force only; composition with a real ANN index is unmeasured.
- Receipts commit to *copies* of `WriteReceipt` fields, not to write-chain
  membership: a mutated ingestion history does not invalidate
  already-issued receipts, and a dishonest query engine is out of scope.
  MerkleGate MMR membership binding is the named future-work item.
- No signature scheme over receipt roots — this crate produces
  commitments, matching `ruvector-proof-gate`'s current scope, not a
  complete non-repudiation system by itself.
- The Merkle tree's odd-width padding (duplicate-last-node) has a known
  malleability class (CVE-2012-2459-style) when an adversary controls the
  leaf set. Here the server always controls the leaf set (its own top-k
  output), bounding but not eliminating the risk in principle.
- Single hardware configuration; no WASM or ARM/edge measurement in this
  run.

## Production Relevance

The two structured variants are not competing for the same use case.
`PerResultReceipt` is simpler and needs no proof-generation step beyond
the chain itself — cheaper to reason about for a small, fixed k where the
whole receipt is always shipped together. `MerkleReceipt` wins when a
verifier needs to challenge *one* disputed result without transporting or
trusting the rest — exactly the shape of a compliance audit ("prove this
specific citation was really retrieved") or an RVF portable-evidence
bundle where proof size matters.

## RuVector Ecosystem Implications

This crate is the read-side mirror of `ruvector-proof-gate` and composes
directly with `ruvector-agent-memory` (as a query wrapper),
`ruvector-capgated` (as its authorization complement, not substitute),
MCP (a narrow `retrieval_verify` read-only tool), RVF (a portable
`{query, results, receipt}` bundle shape), and RVM (gating evidence
acceptance on `index_state_root` freshness). None of those integrations
are implemented in this nightly; each is scoped as concrete future work in
the accompanying ADR-304 and research README, not claimed as delivered.

## Future Direction

The immediate next step is composing this layer onto a real HNSW-family
index and re-measuring the overhead ratio against that (necessarily lower)
baseline latency — the current 1.6-1.8% overhead was measured against a
comparatively expensive brute-force scan, which likely understates the
relative cost on a faster index. Root/head signing is the second concrete
gap: without it, retrieval receipts prove internal consistency but not
who attested to the root, which is required before any of the compliance
or dispute-resolution applications above could be taken seriously in
practice.

## References

- `ruvector-proof-gate` (ADR-227, in-repo) — the write-provenance
  foundation this work extends.
- `ruvector-capgated` (ADR-268, in-repo) — the complementary
  read-authorization layer.
- RFC 6962 (Certificate Transparency) — partial inspiration for the
  domain-separated hashing scheme used here.
- CVE-2012-2459 — the Merkle malleability class disclosed as a known,
  unresolved limitation rather than omitted.
