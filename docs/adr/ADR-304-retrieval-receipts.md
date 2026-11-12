# ADR-304: Retrieval Receipts — Witness-Chained Provenance for ANN Query Results

## Status

Proposed. Experimental crate (`ruvector-retrieval-receipt`), not wired into
the default query path of any production index.

## Context

`ruvector-proof-gate` (ADR-227) gives RuVector a tamper-evident *write*
path: every admitted vector produces a `WriteReceipt` committing to a
SHA-256 hash chain or Merkle Mountain Range. `ruvector-capgated` (ADR-268)
gives RuVector *read authorization*: a query must present a capability
token to see a vector at all.

Neither answers a third, distinct question that matters specifically for
agentic RAG: **given a result set an agent used to produce an answer, can a
third party — an auditor, a compliance reviewer, or another agent —
independently confirm that a specific vector was genuinely part of that
result set, against a specific index state, without re-running the query
and without trusting whoever ran it?** (What this ADR actually delivers
is narrower than that ideal — unsigned commitments that detect
post-issuance mutation, with the engine still trusted at issuance time;
see Threat Model.)

This is the same problem write-receipts solve for ingestion, applied to
retrieval. Its absence matters because:

- Agent memory is now routinely used as evidence for downstream actions
  (a code agent cites a retrieved function; a compliance agent cites a
  retrieved policy clause). Without a receipt, "the agent said it retrieved
  X" is unfalsifiable after the fact.
- The MemoryGraft-style attacks documented in the proof-gate ADR target
  write integrity. A symmetric attack targets *read* integrity: a
  compromised or buggy retrieval layer can silently swap, drop, or reorder
  results between the index and the agent's context window. Write receipts
  alone do not detect this — the underlying vectors may be perfectly
  intact while the result set handed to the agent is not.
- No major vector database (Milvus, Qdrant, Weaviate, Pinecone, LanceDB,
  FAISS, pgvector, Chroma, Vespa) documents a retrieval-receipt mechanism
  as of this research. This is confirmed by inspecting each project's
  public API documentation for a query response, not by a general
  familiarity claim.

## Hypothesis

```text
Given a 5,000-vector index ingested through ruvector-proof-gate's
HashChainGate (so every vector already carries a WriteReceipt),

when a top-10 brute-force cosine query is wrapped with a retrieval
receipt (PerResultReceipt: sequential SHA-256 chain, or MerkleReceipt:
binary Merkle tree over the result set),

then (a) both receipt variants detect 100% of injected result-set
tampering (score mutation, vector-ID substitution, rank reordering,
write-receipt hash corruption) across repeated trials, and (b) MerkleReceipt's
worst-case single-result verification proof is strictly smaller, in bytes,
than PerResultReceipt's equivalent proof at k=10,

subject to receipt-generation latency remaining under 15% of the
brute-force search latency it accompanies (the provenance layer must not
dominate the retrieval cost it is documenting).
```

Explicitly out of scope: approximate-ANN recall. The index used here is an
exact brute-force cosine scan by construction, so recall is always 1.0 and
is not a variable under test — conflating "did the receipt scheme add
overhead" with "did approximation lose recall" would make neither claim
falsifiable. A production integration would sit this layer on top of an
existing ANN index (HNSW, DiskANN-style, etc.) unchanged; that composition
is future work, not claimed here.

## Decision

Add `crates/ruvector-retrieval-receipt`, a small crate that:

1. Wraps a brute-force cosine index whose ingestion path is a real
   `ruvector_proof_gate::HashChainGate`, so every stored vector carries an
   actual `WriteReceipt`.
2. Defines `ResultItem { vector_id, rank, score, write_receipt }` — the
   unit a retrieval receipt commits to. Each result leaf binds *copies* of
   the write receipt's `gate_variant`, `chain_commitment`, and
   `payload_hash`. This makes the write-time evidence part of what the
   receipt commits to, so a receipt/result pair cannot be mutated after
   issuance without detection — but it is a commitment to copies, not a
   membership proof. Verification never consults the write gate, so it
   does **not** prove the bound `WriteReceipt` belongs to any live write
   chain, and mutating the ingestion history *after* issuance leaves
   already-issued receipts verifying. (`HashChainGate::verify_receipt`
   requires the live gate's full chain; `ruvector-proof-gate`'s hash-chain
   variant offers no offline membership proof.) Anchoring each leaf to
   `MerkleGate`'s MMR inclusion proofs is the named future-work item that
   would turn the write→read link into a real membership binding.
3. Implements three variants behind `ReceiptVariant`:
   - `None` — establishes the search-only cost floor.
   - `PerResult` — sequential SHA-256 chain over the k result leaves,
     mirroring `HashChainGate`'s design applied to a single query's result
     set instead of the whole write history.
   - `Merkle` — binary Merkle tree over the k result leaves with
     RFC-6962-style domain-separated leaf/internal-node hashing (distinct
     `b"...leaf:"` / `b"...node:"` prefixes) and O(log k) inclusion proofs.

## Threat Model

Stated plainly so the guarantee is not over-read: a retrieval receipt
detects **post-issuance mutation** of a receipt/result pair — in transit
or in storage. It does **not** protect against a dishonest query engine
(leaves are engine-chosen and unsigned; nothing binds a leaf's score to an
actual cosine computation, or the committed k-set to the true top-k), and
it does **not** prove write-chain membership (see Decision §2). "Offline
verification" therefore means: a holder of the receipt can check, without
talking to the engine, that the results they hold are the ones the engine
committed to — not that the engine computed them honestly.

## Evidence

Measured via `cargo run --release -p ruvector-retrieval-receipt --bin
benchmark` (n=5000, dims=128, k=10, 200 queries, 200 tamper trials per
variant — 50 per tamper kind × 4 kinds). See the nightly research README
for the full output table and raw numbers; do not restate rounded figures
here as a substitute for the actual run.

Unit-level correctness (14 tests in `src/lib.rs`) independently confirms,
per variant:
- Honest result sets always verify (`per_result_receipt_verifies_honest_results`,
  `merkle_receipt_verifies_honest_results`).
- Score mutation, reordering, cross-query vector-ID substitution, and
  gate-variant substitution are each individually detected. (Detection of
  a mutated preimage is expected from SHA-256 by construction; these are
  regression checks on the implementation, not an empirical detection
  rate.)
- Empty result sets fail closed for every variant
  (`empty_result_set_fails_closed_for_all_variants`).
- `MerkleReceipt`'s worst-case proof is smaller than `PerResultReceipt`'s
  at k=10 (`merkle_proof_bytes_are_sublinear_vs_per_result_at_k10`).
- Re-ingesting under a different seed produces a different
  `index_state_root` (`index_state_root_changes_receipts_across_reingestion`).
  Note that test compares roots only — it does not construct a receipt —
  so cross-index replay rejection follows from the leaf's binding to
  `index_root` (exercised by the honest/tamper tests), rather than being
  directly exercised end-to-end by that test.

## Consequences

**Positive:**
- Makes a query's committed result set tamper-evident after issuance: an
  agent's RAG evidence record (ingestion receipt copy + retrieval receipt)
  can be checked for post-hoc mutation using only existing
  `ruvector-proof-gate` hash primitives, no new cryptographic machinery.
  This is integrity of the *record* of retrieval, not proof of honest
  retrieval and not write-chain membership — see Threat Model.
- `MerkleReceipt` gives a compact (O(log k)), portable proof for a single
  disputed result — useful when only one cited memory needs to be
  challenged, not the whole answer.

**Negative / costs:**
- Every additional receipt byte and hash operation is pure overhead versus
  `NoReceipt`; this ADR does not claim retrieval receipts should be the
  default for every query, only that they are cheap enough to be
  selectively enabled (see acceptance evidence for the actual overhead
  percentage).
- The `MerkleReceipt` odd-width padding (duplicate-last-node) is a known
  weakness (CVE-2012-2459-class malleability) when an adversary controls
  the *leaf set*. In this design the leaf set is always the server's own
  top-k output; the querying client supplies neither leaves nor their
  count. This bounds the practical risk but does not eliminate the
  weakness in principle — a future hardening pass should adopt RFC 6962's
  bit-length-prefixed hashing before this scheme is exposed to any path
  where an untrusted party influences leaf count.
- Root/chain-head trust is out of scope here, exactly as it is in
  `ruvector-proof-gate`: this crate produces commitments, not signatures.
  A production deployment needs the index operator (or an RVM-enforced
  proof-gated write path) to sign `index_state_root` and each query's
  receipt root periodically, the same open item `ruvector-proof-gate`
  already carries.

## Alternatives Considered

- **Extend `MerkleGate`'s MMR directly to reads.** Rejected for this
  experiment: an MMR is optimized for an *append-only* stream (writes);
  a single query's result set is a small, fixed-size, one-shot set, for
  which a plain binary Merkle tree is simpler and has the same asymptotic
  proof size.
- **Sign every result individually (no chaining/tree at all).** Rejected:
  this is `PerResultReceipt` without the chaining — it would detect
  substitution of an individual result but not reordering or set-membership
  tampering (an attacker could present a subset of validly-signed results
  as if it were the complete top-k). Chaining/tree structure is what makes
  the *result set*, not just each item, tamper-evident.
- **Capability-gated read (ADR-268) is "provenance enough."** Rejected:
  capability gating controls *who may see* a vector; it says nothing about
  *what was actually returned* to whoever was authorized to see it. The two
  are complementary, not substitutes.

## Implementation Plan

1. (This ADR) Land the experimental crate, benchmark, and tests —
   unintegrated, feature-isolated.
2. If promoted: integrate as an optional wrapper around
   `ruvector-agent-memory` query paths, gated behind a Cargo feature so the
   default build pays zero cost.
3. Bind each result leaf to a `MerkleGate` MMR inclusion proof so a
   receipt proves write-chain *membership* offline instead of merely
   committing to copies of `WriteReceipt` fields — the prerequisite for
   any chain-of-custody-grade claim.
4. Wire `index_state_root`/`chain_head` signing through the existing
   witness-signing story once `ruvector-proof-gate` gains one (currently
   neither crate signs; both are commitment-only).
5. MCP surface: a narrow `retrieval_verify` read-only tool that accepts a
   receipt + one result item and returns a boolean, never exposing raw
   index internals.

## API Shape

```rust
let index = RetrievalIndex::ingest(n, dims, seed); // real WriteReceipt per vector
let results = index.search(&query, k);             // Vec<ResultItem>
let receipt = RetrievalReceipt::build(
    ReceiptVariant::Merkle, query_hash(&query), index.index_state_root(), &results,
);
assert!(receipt.verify_full(query_hash(&query), index.index_state_root(), &results));
```

## Feature Flags

None yet — the crate is opt-in by virtue of not being a dependency of any
other crate. A `receipts` feature flag on `ruvector-agent-memory` is the
proposed integration point if promoted (see Implementation Plan).

## Benchmark Evidence

See `docs/research/nightly/2026-08-13-retrieval-receipts/README.md` for
the full methodology and raw `cargo run --release` output.

## Security

- Domain-separated hashing (`leaf:` vs `node:` vs `chain:` byte prefixes)
  prevents second-preimage confusion between leaf and internal-node
  positions in the Merkle tree.
- The documented duplicate-last-node padding weakness (see Consequences)
  is the primary open security item; it does not affect this experiment's
  threat model (server-controlled leaf set) but must be fixed before any
  broader exposure.
- No new `unsafe` code. No external network calls. WASM-compatible (same
  dependency shape as `ruvector-proof-gate`: only `sha2`).

## Governance

Receipts are commitments, not authorizations — they do not replace
`ruvector-capgated`'s access control and must not be treated as such by
any integrating crate.

## Failure Modes

- If `results.len() != receipt.leaves.len()`, every `verify_full` call
  returns `false` (tested: `per_result_receipt_detects_reorder`,
  `merkle_receipt_detects_reorder`, both exercise a length-preserving
  reorder rather than a truncation, since truncation is the trivially
  caught case).
- `NoReceipt::verify_item` and `verify_full` always return `false`, never
  panic and never silently report "no evidence" as "verified" — a caller
  that mistakenly branches on this return value fails closed, not open.

## Migration

N/A — new, unintegrated crate.

## Rollback

Delete `crates/ruvector-retrieval-receipt` and its workspace member entry;
no other crate depends on it.

## Rejection Criteria

This direction should be rejected for production promotion if any of the
following hold on re-measurement at larger scale (n≥100k, k≥100):
- Any tamper-kind regression test fails. Detection of a mutated preimage
  follows from SHA-256 collision resistance — the 200/200 trial result is
  a correctness regression check, not an empirical detection rate — so a
  failure here would indicate an implementation bug, which is
  disqualifying.
- `MerkleReceipt`'s proof-size advantage disappears or inverts at larger k
  (it should not, asymptotically, but must be re-confirmed rather than
  assumed).
- Receipt generation overhead exceeds the 15% threshold once applied on
  top of a real HNSW/ANN index rather than brute force (brute-force search
  is comparatively expensive, which understates the *relative* overhead of
  the receipt layer; this must be re-measured against a cheaper baseline
  before any production claim).

## Open Questions

- What is the right signing story for `index_state_root` and per-query
  receipt roots? This ADR deliberately leaves signing out of scope,
  matching `ruvector-proof-gate`'s current state.
- Should retrieval receipts be persisted (for later audit) or generated
  on-demand and discarded if not requested? Persisting O(k) bytes per
  query at agent scale is a real storage-growth question not addressed
  here.
- Does composing this layer on top of an approximate (HNSW-family) index
  change any of the measured properties, or only the recall dimension this
  experiment deliberately excluded?
