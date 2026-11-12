# Retrieval Receipts: Witness-Chained Provenance for ANN Query Results

**150-char summary:** Cryptographic receipts making ANN query results tamper-evident after issuance — MerkleReceipt gives O(log k) audit proofs vs a hash chain's O(k).

**Date:** 2026-08-13
**Crate:** `crates/ruvector-retrieval-receipt`
**ADR:** [ADR-304](../../../adr/ADR-304-retrieval-receipts.md)

---

## Abstract

`ruvector-proof-gate` (ADR-227) already gives RuVector tamper-evident vector
*writes*: a SHA-256 hash chain or Merkle Mountain Range commits to every
admitted vector. No public vector database (Milvus, Qdrant, Weaviate,
Pinecone, LanceDB, FAISS, pgvector, Chroma, Vespa) documents an equivalent
mechanism for the *read* path — none produce evidence that lets a later
holder of a result set check, independently of the system that ran the
query, that the set was not mutated after it was returned.

This nightly implements and benchmarks **retrieval receipts**: cryptographic
commitments over a query's top-k result set that bind each result to the
`WriteReceipt` produced when that vector was ingested. Three variants are
measured on real Rust release builds:

| Variant | Generation | Verify 1-of-k (worst case) | Proof size (worst case, k=10) | Tamper detection |
|---|---|---|---|---|
| `NoReceipt` | 237 ns | N/A (unverifiable) | 0 bytes | N/A |
| `PerResultReceipt` | 18,310 ns | 8,229 ns | 320 bytes | 200/200 |
| `MerkleReceipt` | 19,582 ns | 3,839 ns | **160 bytes** | 200/200 |

**Key measured result:** `MerkleReceipt`'s worst-case single-result
verification proof is 160 bytes and verifies in 3,839 ns, vs 320 bytes /
8,229 ns for `PerResultReceipt` at k=10. One baseline caveat applies to
that comparison: `PerResultReceipt`'s proof size is defined here as the
genesis-anchored chain replay (`(idx+1) * 32` bytes — leaves `0..=idx`);
a verifier anchored at the chain head instead would need only the
`k−idx` suffix, which at the measured worst index is smaller than the
Merkle path. The durable claim is therefore the asymptotic one — O(log k)
Merkle proofs vs O(k) chain replay regardless of which result is disputed
— not the specific 2x constant. Both variants rejected all 200/200
injected tamper trials; that is expected by construction (a mutated
preimage changes its SHA-256 hash), so it is a regression check on the
implementation, not an empirical detection rate. Receipt generation adds
**1.6-1.8%** to the 1.1 ms brute-force search it accompanies — far under
the 15% acceptance threshold set before the run.

All numbers are from `cargo run --release -p ruvector-retrieval-receipt
--bin benchmark -- 5000 128 10 200` on the hardware below. Raw output is
reproduced verbatim in [Benchmark Results](#benchmark-results).

**Hardware:** x86-64, 4 logical CPUs, Linux 6.18.5, `rustc` release build.

---

## Hypothesis

```text
Given a 5,000-vector index ingested through ruvector-proof-gate's
HashChainGate (so every vector already carries a WriteReceipt),

when a top-10 brute-force cosine query is wrapped with a retrieval
receipt (PerResultReceipt: sequential SHA-256 chain, or MerkleReceipt:
binary Merkle tree over the result set),

then (a) both receipt variants detect 100% of injected result-set
tampering across repeated trials, and (b) MerkleReceipt's worst-case
single-result verification proof is strictly smaller, in bytes, than
PerResultReceipt's equivalent proof at k=10,

subject to receipt-generation latency remaining under 15% of the
brute-force search latency it accompanies.
```

**Result: ACCEPT.** Every clause held on measurement; see
[Acceptance Result](#acceptance-result).

**What this does NOT claim:** approximate-ANN recall. The index is exact
brute-force cosine by construction, so recall is always 1.0 and is
deliberately not a variable under test here — see
[Why Brute Force](#why-brute-force-not-hnsw).

---

## Why This Matters for RuVector

RuVector positions itself as a Rust-native cognition substrate for agents,
not merely a vector database. Two integrity primitives already exist:

- **ADR-227 (`ruvector-proof-gate`):** you need a cryptographic receipt to
  prove a vector was *written* honestly.
- **ADR-268 (`ruvector-capgated`):** you need a capability token to be
  *authorized to read* a vector at all.

Neither answers: given a result set an agent actually used to produce an
answer, can a third party later confirm — offline, without re-running the
query — that the result set they hold is the one the engine committed to
at query time? Note the precise shape of that guarantee: receipts are
unsigned commitments produced by the engine itself, so they detect
post-issuance mutation of a receipt/result pair; they do not make a
dishonest engine honest (see [Threat Model](#threat-model)). That is
still the missing link for agent evidence trails: an audit of "why did
the agent say X" needs a tamper-evident record of the retrieval event
that surfaced it.

This connects five RuVector ecosystem capabilities in one crate:

1. **Vector search** (`ruvector-retrieval-receipt`'s own brute-force cosine
   index) — the thing being made auditable.
2. **Witness/provenance** (`ruvector-proof-gate`) — the write-side
   foundation this crate extends to reads, reusing its `HashChainGate` and
   `WriteReceipt` types directly rather than re-implementing them.
3. **Agent memory** — the natural consumer: an agent's RAG evidence trail
   is exactly a sequence of retrieval events over agent memory.
4. **MCP** — a narrow `retrieval_verify` tool is a natural, low-authority
   surface for this capability (see [MCP Implications](#mcp-implications)).
5. **RVF** — a signed, portable "retrieval bundle" (query + receipt +
   result set) is a direct RVF artifact shape (see
   [RVF Implications](#rvf-implications)).

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingest["Write path (ruvector-proof-gate, existing)"]
        W[WritePayload] --> G[HashChainGate]
        G --> WR[WriteReceipt]
        WR --> IDX[(RetrievalIndex\nvectors + write_receipts)]
    end

    subgraph Query["Read path (ruvector-retrieval-receipt, new)"]
        Q[Query vector] --> S[Brute-force\ncosine search]
        IDX --> S
        S --> R["Vec&lt;ResultItem&gt;\n(id, rank, score, write_receipt)"]
        R --> B{Receipt variant}
        B -->|None| N0[no receipt]
        B -->|PerResult| PR["Sequential SHA-256 chain\nover k result leaves"]
        B -->|Merkle| MR["Binary Merkle tree\nover k result leaves\n+ O(log k) inclusion proof"]
    end

    PR --> V[Offline verifier:\nreplay O(idx) leaves]
    MR --> V2[Offline verifier:\ncheck O(log k) sibling path]

    style Ingest fill:#1f6feb22,stroke:#1f6feb
    style Query fill:#8957e522,stroke:#8957e5
```

Each result leaf commits to: the query hash, the index state root at query
time, the result's rank/vector_id/score, and *copies* of the underlying
`WriteReceipt`'s `gate_variant`, `chain_commitment`, and `payload_hash`.
Binding those copies makes the write-time evidence part of what the
receipt commits to: an auditor can confirm "this exact ingestion record
was cited," and any post-issuance mutation of either the result fields or
the bound write-receipt copy breaks verification.

### Threat Model

What a retrieval receipt does and does not prove:

- **Does:** detect post-issuance mutation of a receipt/result pair, in
  transit or in storage. That is the whole guarantee.
- **Does not:** protect against a dishonest query engine. Leaves are
  engine-chosen and unsigned; nothing binds a leaf's score to an actual
  cosine computation, or the committed k-set to the true top-k.
- **Does not:** prove write-chain membership. Verification recomputes
  hashes over the caller-supplied copies and never consults the write
  gate, so mutating the ingestion history *after* a receipt is issued
  leaves that receipt verifying. `HashChainGate::verify_receipt` requires
  the live gate's full chain — the hash-chain variant offers no offline
  membership proof. Anchoring leaves to `MerkleGate`'s MMR inclusion
  proofs is the named future-work item that would make the write→read
  link a real membership binding.

---

## Implementation

- `src/index.rs` — `RetrievalIndex`: wraps a real
  `ruvector_proof_gate::HashChainGate` for ingestion (so every stored
  vector carries an actual `WriteReceipt`, not a stand-in), plus a
  brute-force exact cosine `search()`.
- `src/receipt.rs` — `PerResultReceipt` (sequential chain, mirrors
  `HashChainGate`'s design applied per-query) and `MerkleReceipt` (binary
  tree with RFC-6962-style domain-separated leaf/node hashing:
  `b"...leaf:"` vs `b"...node:"` prefixes prevent leaf/internal-node type
  confusion).
- `src/lib.rs` — `RetrievalReceipt` enum unifying all three variants for
  benchmarking, plus 14 unit tests covering honest verification, four
  independent tamper kinds per structured variant, gate-variant binding,
  empty-result fail-closed behavior, and cross-index root divergence.
- `src/bin/benchmark.rs` — the benchmark producing the numbers below,
  including the tamper-detection trial harness.

### Why Brute Force, Not HNSW

The variable under test is the *provenance layer's* cost and correctness,
not retrieval quality. Composing this on top of an approximate index would
conflate "did the receipt scheme add overhead" with "did approximation
lose recall" — two independent questions that would no longer be
separately falsifiable. Brute-force cosine search is exact by
construction, so recall is fixed at 1.0 and receipt overhead can be
measured in isolation. Layering this on a real HNSW/DiskANN-style index is
explicitly future work (see [Rejection Criteria](#rejection-criteria-not-yet-triggered)).

---

## Benchmark Methodology

- **Command:** `cargo run --release -p ruvector-retrieval-receipt --bin
  benchmark -- 5000 128 10 200`
- **Dataset:** 5,000 deterministic 128-dim vectors (xorshift64, fixed seed
  `0xC0FFEE01D00D`), ingested through `HashChainGate` one at a time.
- **Queries:** 200 deterministic 128-dim query vectors (independent seed
  `0xA5A55A5A1111`), each searched for top-10 by cosine similarity.
- **Repetitions:** every query is measured once for generation latency
  (200 samples per variant → mean/p95 reported); tamper trials run 50
  times per tamper kind × 4 kinds = 200 trials per structured variant.
- **Baseline isolation:** the brute-force search itself is timed once,
  shared across all three variants, so receipt overhead is measured as
  `(receipt_generation_time) / (shared_search_time)`, not conflated with
  search variance between runs.
- **Warmup:** none required — first-call JIT/warmup effects don't apply to
  ahead-of-time-compiled release Rust; the ingest phase (5,000 real
  `HashChainGate.admit()` calls) runs before any timed query.
- **Tamper kinds:** score mutation (`+0.5` to one result's score),
  vector-ID substitution (`wrapping_add(999_999)`), rank swap (adjacent
  result reorder), write-receipt hash flip (bit-flip one byte of a
  result's bound `payload_hash`). Each is applied to an otherwise-honest,
  freshly generated receipt+result pair.

## Benchmark Results

Raw output, `cargo run --release -p ruvector-retrieval-receipt --bin
benchmark -- 5000 128 10 200`:

```text
=== ruvector-retrieval-receipt benchmark ===
n=5000 dims=128 k=10 queries=200 tamper_trials_per_kind=50
hardware: 4 logical CPUs (see `nproc`), rustc build profile: release-required for meaningful numbers
ingest: 5000 vectors in 27.828 ms (179.7 writes/ms), index_state_root non-zero: true

baseline brute-force search: mean=1114440ns p95=1371306ns over 200 queries

variant               gen_mean_ns     gen_p95_ns  verify_worst_ns    proof_bytes   total_bytes_mean    tamper_detect
NoReceipt                     237            278                0              0                0.0              n/a
PerResultReceipt            18310          30103             8229            320              640.0          200/200
MerkleReceipt               19582          26120             3839            160              352.0          200/200

=== acceptance ===
tamper detection 100% across all kinds: true
merkle worst-case proof bytes (160) < per-result worst-case proof bytes (320): true
generation overhead < 15% of baseline search: merkle=1.8% per_result=1.6% -> true

ACCEPTANCE RESULT: ACCEPT
```

`cargo test --release -p ruvector-retrieval-receipt`: **14 passed, 0
failed** (deterministic-seed unit tests covering honest verification, four
tamper kinds independently, gate-variant binding, empty-result
fail-closed behavior, cross-index root divergence, and the proof-size
sublinearity claim in isolation from the benchmark binary).

## Acceptance Result

```text
ACCEPT
```

All three clauses of the formalized hypothesis held: (a) all 200/200
tamper trials rejected for both structured variants — expected from
SHA-256 by construction, reported as a regression check rather than an
empirical detection rate; (b) `MerkleReceipt`'s worst-case proof (160
bytes) is smaller than `PerResultReceipt`'s (320 bytes) at k=10 under the
genesis-anchored proof-size definition (see the baseline caveat in the
abstract); (c) generation overhead (1.6-1.8%) is well under the 15%
threshold fixed before this run.

---

## Memory Math

- `PerResultReceipt`: `leaves` + `commitments`, 32 bytes each, per result
  → `2 * k * 32` bytes total (640 bytes at k=10, matches measured
  `total_bytes_mean`).
- `MerkleReceipt`: `leaves` (32 bytes each) + `root` (32 bytes) →
  `(k + 1) * 32` bytes total (352 bytes at k=10, matches measured).
- Worst-case single-item proof: `PerResultReceipt` needs `(idx+1)*32`
  bytes (320 at idx=9) under the genesis-anchored replay definition used
  throughout this experiment — a head-anchored verifier would instead need
  only the suffix from `idx` to the chain head, so this figure is a
  property of the chosen baseline definition, not of hash chains in
  general. `MerkleReceipt` needs `32 + ceil(log2 k)*32` bytes (160 at
  k=10, `ceil(log2 10) = 4` sibling hashes) regardless of anchoring.
- At k=100 the asymptotic gap widens under the same genesis-anchored
  definition: `PerResultReceipt` worst-case proof ≈ 3,200 bytes;
  `MerkleReceipt` ≈ `32 + 7*32` = 256 bytes. This crate does not
  re-measure that scale-up; it is a direct consequence of the O(idx) vs
  O(log k) complexity already confirmed at k=10 and is stated here as
  arithmetic, not fabricated as a second benchmark run.

## Performance Math

Search cost is `O(n * dims)` = 5,000 × 128 ≈ 640K multiply-accumulate
operations per query — this dominates the ~1.1 ms measured baseline.
Receipt generation is `O(k)` SHA-256 calls (k=10) — a few microseconds
regardless of index size. The 1.6-1.8% overhead measured here will shrink
further, in relative terms, against a real ANN index (HNSW-class query
latency is typically far below a 5,000-vector brute-force scan at higher
n), which is exactly why the [Rejection Criteria](#rejection-criteria-not-yet-triggered)
require re-measuring this ratio against a non-brute-force baseline before
any production overhead claim.

---

## Failure Modes

- Result-set length mismatch (truncation/extension) → `verify_full`
  returns `false` for both structured variants (not separately
  benchmarked here since it is the trivially-caught case; reordering,
  the harder case, is the one measured).
- `NoReceipt` always returns `false` from `verify_item`/`verify_full` —
  fails closed, never silently reports "verified."
- Odd-width Merkle levels are padded by duplicating the last node — a
  known malleability weakness (CVE-2012-2459-class) if an adversary
  controls the leaf *set*. Here the leaf set is always the server's own
  top-k output; the client supplies neither leaves nor their count, which
  bounds (but does not eliminate in principle) the practical risk. See
  ADR-304's Security section.

## Rejected Alternatives

See ADR-304 "Alternatives Considered": extending `MerkleGate`'s MMR
directly to reads (rejected — MMR is optimized for append-only streams,
not a fixed one-shot result set), and per-result independent signatures
without chaining/tree structure (rejected — detects individual
substitution but not reordering or subset-presented-as-complete attacks).

## Security

- Domain-separated hashing (distinct byte-string prefixes for leaf vs.
  internal-node vs. chain-step hashing) prevents type confusion between
  tree positions.
- No `unsafe` code. Only dependency beyond the workspace's existing
  `ruvector-proof-gate` is `sha2` (already a proof-gate dependency).
- The duplicate-last-node Merkle padding weakness is documented, not
  hidden — see Failure Modes above and ADR-304.
- This crate produces commitments, not signatures. `index_state_root` and
  per-query receipt roots are not signed here; that is an explicitly open
  item shared with `ruvector-proof-gate`, which also does not sign.

## Governance

Retrieval receipts are commitments, not authorizations. They must not be
treated as a substitute for `ruvector-capgated`'s read-access control —
the two are complementary layers (who may see a vector vs. what was
actually returned to whoever was authorized).

## MCP Implications

A narrow, read-only `retrieval_verify` MCP tool is a natural surface:
inputs = `{receipt, result_item, query_hash, index_state_root}`; output =
`{verified: bool}`; no side effects, no index mutation, no broad query
authority exposed. This was not implemented in this nightly — it is a
concrete next step, not a vague "MCP could expose this."

## WASM Implications

Identical dependency shape to `ruvector-proof-gate` (`sha2` only, no
`unsafe`), which is already WASM-compatible. No WASM build or size
measurement was performed in this nightly; claiming a specific binary-size
delta without measuring it would violate the no-fabricated-evidence rule,
so this is stated as an expectation, not a result.

## RVF Implications

A "retrieval bundle" — `{query, query_hash, result_set, receipt,
index_state_root}` — is a natural RVF portable-artifact shape: it is
self-contained, deterministically replayable (re-run `verify_full` against
the bundle with no external state beyond the bundle itself), and
copy-on-write friendly (a new receipt does not mutate the underlying
index). Not implemented here; flagged as materially relevant per the
mandatory RVF-analysis step.

## RVM Implications

An RVM coherence domain could enforce that only receipts whose
`index_state_root` matches a currently-attested index state are accepted
by a downstream agent — i.e., reject retrieval evidence from a stale or
forked index snapshot. This is a proof-gated-mutation-adjacent use case
(gating *acceptance of evidence*, not a write), plausible but not
implemented or benchmarked here.

## ruFlo Implications

A concrete ruFlo workflow: on each agent RAG turn, ruFlo generates a
`MerkleReceipt` transparently, stores only the 32-byte root persistently
(cheap), and retains full per-query receipts in a short-lived buffer;
if a downstream review flags an agent's answer, ruFlo replays the buffered
receipt against the disputed result to produce an audit artifact on
demand — avoiding the storage cost of persisting full receipts for every
query while preserving disputability for a bounded recent window.

## Practical Applications

| # | User | Problem | Capability used | Integration | Business value | Main risk | Horizon |
|---|---|---|---|---|---|---|---|
| 1 | Compliance-regulated agent deployments | "Prove what evidence the agent actually used" | `MerkleReceipt` + `ruvector-proof-gate` | Wrap `ruvector-agent-memory` queries | Audit-passable RAG | Signing story still open (ADR-304) | Now-2027 |
| 2 | Multi-agent code assistants | Disputed "the agent hallucinated this function" claims | Retrieval receipt as ground truth | MCP `retrieval_verify` tool | Reduced trust-repair cost | Adoption friction (opt-in feature) | Now-2027 |
| 3 | Enterprise RAG platforms | Silent retrieval-layer bugs swapping results | Tamper-evident result sets | Feature-flagged wrapper | Faster incident diagnosis | False sense of security if misapplied to writes | 2027-2029 |
| 4 | Legal/medical retrieval systems | Tamper-evident records of cited evidence | Write+read receipt binding | RVF portable bundle | Regulatory eligibility | Not chain-of-custody today: needs MerkleGate membership proofs + signed roots (neither built), plus Merkle padding hardening | 2027-2030 |
| 5 | Federated agent memory (edge + cloud sync) | Confirming synced results match origin index state | `index_state_root` binding | RVM coherence domain | Detects sync corruption | Needs signed roots, not yet built | 2028-2032 |
| 6 | Scientific literature search agents | Reproducible citation trails | Deterministic receipt replay | RVF replay bundle | Reproducibility compliance | Requires persisted receipts (storage cost) | 2027-2030 |
| 7 | Security incident-response retrieval | "What did the SOC agent actually pull from the threat-intel index" | Full write→read chain | ruFlo audit workflow | Faster post-incident review | Needs retention policy | Now-2028 |
| 8 | Autonomous negotiation/trading agents | Disputes over what market data an agent acted on | Per-decision retrieval receipt | Agent decision logging | Liability/dispute resolution | Latency-sensitive paths may skip receipts | 2028-2032 |

## Long Horizon Applications

| # | Thesis | Required advances | RuVector role | Why this experiment matters | Primary uncertainty | Falsification path |
|---|---|---|---|---|---|---|
| 1 | Agent operating systems require non-repudiable memory I/O, not just access control | Signed roots, kernel-level receipt enforcement | Substrate providing both write and read receipts natively | Establishes the read-side primitive; write-side already exists | Whether receipt overhead survives at OS-call frequency | Overhead grows superlinearly at scale |
| 2 | Swarm memory needs cross-agent evidence exchange with cryptographic trust | Distributed signing, revocation | RVF bundles as the exchange unit | This experiment defines the bundle's core fields | Trust model across mutually distrusting agents | Bundles forgeable without signing |
| 3 | Proof-gated autonomous infrastructure (RVM) needs verifiable "what did the controller see" logs | RVM coherence-domain integration | `index_state_root` as the domain-consistency check | First concrete field an RVM domain could gate on | Whether staleness detection composes with liveness | Stale-state false positives at scale |
| 4 | Robotics memory needs tamper-evident sensor-fusion retrieval for safety certification | Real-time receipt generation under hard latency budgets | Same commit scheme, tighter latency budget | Establishes baseline overhead is small (1.6-1.8%) pre-hard-real-time | Whether Merkle build fits a control loop budget | Latency budget violated at required frequency |
| 5 | Synthetic nervous systems: reflexive retrieval needs provenance without conscious-layer overhead | Ultra-low-overhead receipt variant | A fourth "streaming" variant not yet designed | Shows current variants' floor cost | Whether a cheaper-than-Merkle variant exists | No variant beats measured NoReceipt floor meaningfully |
| 6 | Self-healing graph memory needs to distinguish "corrupted index" from "corrupted transit" | Combine with `ruvector-mincut`/coherence scoring | Receipt failure as a graph-repair trigger signal | Provides the failure signal this would consume | Whether receipt failures correlate with real corruption vs. noise | High false-positive rate on real corruption events |
| 7 | Scientific autonomous systems need falsifiable retrieval logs for peer review | Standardized bundle format, external verifiers | RVF as the interchange format | Defines bundle fields concretely | Whether external (non-RuVector) tools can verify | Format too RuVector-specific to be portable |
| 8 | World models need to audit which memories shaped a prediction | Extend receipts to multi-hop retrieval chains | Composable receipts across retrieval stages | Single-hop version proven first | Whether multi-hop composition preserves O(log k) proofs | Composition cost grows non-sublinearly |

## Falsification Criteria

See ADR-304 "Rejection Criteria" — reproduced here for completeness:

## Rejection Criteria (Not Yet Triggered)

- Any tamper-kind regression test fails at larger scale (n≥100k, k≥100).
  Detection follows from SHA-256 collision resistance — 200/200 is a
  regression check, not an empirical rate — so a failure would indicate
  an implementation bug, which is disqualifying.
- `MerkleReceipt`'s proof-size advantage disappears or inverts at larger
  k — should not happen asymptotically but is unverified beyond k=10.
- Receipt overhead exceeds 15% once measured against a real HNSW/ANN
  baseline instead of brute force — brute force's higher absolute latency
  understates the *relative* cost of the receipt layer; this must be
  re-measured, not assumed favorable.

## Limitations

- Only exact brute-force retrieval was measured; approximate-index
  composition is unverified.
- Receipts commit to *copies* of `WriteReceipt` fields, not to write-chain
  membership: a mutated ingestion history does not invalidate
  already-issued receipts, and a dishonest query engine is out of scope
  entirely — see [Threat Model](#threat-model). MerkleGate MMR membership
  binding is the named future-work item.
- No signing of roots/heads — receipts are commitments only, matching
  `ruvector-proof-gate`'s current scope, not a complete non-repudiation
  system on their own.
- Single hardware configuration (4 logical CPUs, one Linux kernel); no
  cross-platform or ARM/edge measurement performed.
- Merkle padding weakness (documented above) needs RFC 6962-style
  hardening before exposure to any untrusted-leaf-count scenario.

## Next Research

1. Compose `ruvector-retrieval-receipt` on top of a real HNSW-family index
   and re-measure the overhead ratio (flagged explicitly in Rejection
   Criteria as required before any production overhead claim).
2. Design and benchmark root/head signing (Ed25519 over `index_state_root`
   + receipt root) to close the non-repudiation gap this crate leaves
   open.
3. Multi-hop receipt composition for chained retrieval (retrieve →
   re-rank → retrieve again), needed for the "world models" long-horizon
   application above.

## References

- `ruvector-proof-gate` source and ADR-227 (in-repo, existing).
- `ruvector-capgated` and ADR-268 (in-repo, existing).
- Certificate Transparency (RFC 6962) — domain-separated Merkle hashing
  scheme this crate partially adopts (leaf/node prefix separation) and
  partially does not (bit-length prefixing for the padding weakness,
  noted as future hardening).
- CVE-2012-2459 — the duplicate-last-node Merkle malleability class this
  crate's known limitation belongs to.
- Public API documentation review of Milvus, Qdrant, Weaviate, Pinecone,
  LanceDB, FAISS, pgvector, Chroma, and Vespa query-response formats,
  none of which document a retrieval-receipt/provenance mechanism as of
  this research (documented_external_capability: none found;
  directly_measured_capability: N/A, no comparable systems installed
  locally to benchmark against).
