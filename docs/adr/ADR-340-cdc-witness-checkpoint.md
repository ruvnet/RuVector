# ADR-340: Content-Defined Chunking for Incremental, Witness-Chained Index Checkpoints

## Status

Proposed. Experimental crate (`ruvector-cdc-checkpoint`), not wired into
`ruvector-snapshot`'s production checkpoint path or any other crate.

## Context

`ruvector-snapshot` durably persists a vector/graph collection by
re-serializing and re-writing it. `ruvector-proof-gate` (ADR-227) and
`ruvector-retrieval-receipt` (ADR-304) make the write and read paths of an
index tamper-evident. None of the three address a cost that becomes
material once a collection is checkpointed *repeatedly* rather than once:
a full re-snapshot's write cost scales with the size of the whole
collection, not with the size of whatever actually changed since the last
checkpoint. For an agent-memory collection that is expected to grow
indefinitely and be checkpointed on a schedule, that is exactly backwards.

Two adjacent, well-known techniques address parts of this:

- **Fixed-size block chunking + dedup** (the naive incremental-backup
  approach) reduces repeated bytes, but a single edit shifts every
  downstream block boundary's *content* even though the boundary
  *positions* never move, invalidating nearly every block after the edit.
- **Content-defined chunking (CDC)** — FastCDC (Xia et al., USENIX ATC
  2016) and its predecessors — anchors chunk boundaries to local byte
  content via a rolling hash, so an edit only invalidates the chunk(s)
  touching it; everything else resynchronizes.

No public vector database (Milvus, Qdrant, Weaviate, Pinecone, LanceDB,
FAISS, pgvector, Chroma, Vespa) documents CDC-based incremental
snapshotting for its index/backup path as of this research (verified by
reviewing each project's public backup/snapshot documentation, not by a
general familiarity claim).

Separately, this run's own instructions describe orchestrating the
research via `npx metaharness` / `npx ruvector harness {doctor,darwin,
flywheel}`. Both were checked before assuming they exist:
`npx metaharness --help` resolves to a harness-*scaffolding* generator
(creates new agent-harness projects from templates; `score`/`analyze`/
`genome` subcommands assess a target repo's scaffolding readiness), not a
Darwin/Flywheel research-orchestration engine over this repository.
`npx ruvector harness doctor --json` and `... status --json` both fail
with `npm error could not determine executable to run` — no such CLI is
installed. This ADR's Decision therefore includes an in-crate bounded
parameter sweep standing in for a Darwin evolution phase, recorded
explicitly rather than silently substituted.

## Hypothesis

```text
Given an HNSW-style vector+graph index checkpointed periodically for
durability, with churn between checkpoints representative of a busy
agent-memory workload (~0.2% inserts, ~0.1% deletes, ~0.3% updates per
round, of a 20,000-row index, over 30 rounds),

when checkpoints are produced via content-defined chunking (FastCDC-style
gear-hash rolling chunker) plus a content-addressed store, instead of a
full re-snapshot or fixed-size block chunking,

then steady-state incremental bytes written per checkpoint round should
be substantially smaller under CDC than under either baseline (fixed
thresholds: <=50% of fixed-block, <=20% of full-snapshot),

subject to every checkpoint remaining bit-identically reconstructible via
witness-chain verification, for every round and every variant, and
chunking throughput remaining above a 20 MB/s floor.
```

## Decision

Add `crates/ruvector-cdc-checkpoint`, a small crate that:

1. Implements a deterministic FastCDC-style gear-hash chunker
   (`chunker::cdc_boundaries`) and a fixed-size baseline
   (`chunker::fixed_boundaries`), sharing one `(start, end)` range
   interface so both plug into the same downstream store/witness code.
2. Implements a SHA-256-keyed content-addressed `ChunkStore`
   (`store.rs`) that reports, per insert, exactly how many bytes were
   newly persisted versus deduplicated against existing content — the
   metric the acceptance test compares across variants.
3. Implements a sequential, domain-separated SHA-256 witness chain over
   checkpoint manifests (`witness.rs`), directly mirroring
   `ruvector-retrieval-receipt::receipt`'s chaining pattern (same
   `prev || leaf` chain-step construction, same domain-separation-by-byte-prefix
   discipline) applied to `{chunk_hashes, content_hash}` per checkpoint
   round instead of `{query, result}` per query. `verify()` returns one
   of three explicit failure variants (`MissingChunk`,
   `ContentHashMismatch`, `ChainRootMismatch`) rather than a boolean, so a
   caller (and this ADR's tests) can distinguish *which* invariant broke.
4. Implements a synthetic, deterministic (splitmix64-seeded) vector+graph
   index (`workload.rs`) with a realistic churn model (tombstone-on-delete,
   insert-appends, in-place update), used only to generate the byte blob
   each variant chunks — not a reimplementation of `ruvector-core`'s
   actual binary layout (see Alternatives Considered).
5. Unifies all three chunking strategies (`Variant::{FullSnapshot,
   FixedBlock, Cdc}`) behind one `Checkpointer` type so the benchmark's
   only independent variable is chunking strategy, not incidental
   implementation differences between variants.
6. Includes a bounded, single-generation, four-candidate parameter sweep
   over the CDC chunker's target average chunk size
   (`darwin_lite_sweep` in `src/bin/benchmark.rs`), standing in for the
   Darwin evolution phase this run's instructions call for, given no
   Darwin CLI/engine is installed (see Context).

## Evidence

Measured via `cargo run --release -p ruvector-cdc-checkpoint --bin
benchmark` (n=20,000 vectors, dim=128, degree=16, 30 checkpoint rounds,
seed `0xC0FFEE0012345678`). See the nightly research README for the full
raw output; do not restate rounded figures here as a substitute for the
actual run.

Headline steady-state result: CDC (avg_size=2048) writes 609,120
bytes/round versus fixed-block's 5,146,173 (ratio 0.1184, threshold
<=0.50) and full-snapshot's 11,734,378 (ratio 0.0519, threshold <=0.20).
Both threshold clauses are met with wide margin. Chunking throughput
(460.7 MB/s) is ~23x the 20 MB/s acceptance floor.

Unit-level correctness (15 tests in `src/lib.rs` and its submodules)
independently confirms: chunk-boundary contiguity with no gaps/overlap
(`cdc_boundaries_cover_all_bytes_exactly_once`); boundary determinism
(`cdc_boundaries_are_deterministic`); the defining resynchronization
property — boundaries before an inserted region are byte-for-byte
unchanged (`cdc_resynchronizes_after_a_mid_stream_insertion`), contrasted
directly with fixed-block's lack of it
(`fixed_boundaries_shift_every_block_content_after_an_edit`); dedup
accounting (`dedup_hit_reports_zero_new_bytes`); exact reconstruction
(`store_and_reconstruct_round_trips_exactly`); all three witness
verification failure modes, distinguished
(`manifest_lying_about_content_hash_is_rejected`,
`manifest_with_forged_chain_root_is_rejected`, `missing_chunk_is_detected`);
and a direct end-to-end regression check that all three variants
reconstruct exactly across five rounds of real churn
(`all_three_variants_reconstruct_every_round_exactly`).

The benchmark was run twice end-to-end; every byte-count and chunk-count
metric was bit-for-bit identical (fully deterministic workload and
chunking — no sampling variance), with throughput varying ~2-3% between
runs from ordinary scheduler jitter.

`cargo clippy -p ruvector-cdc-checkpoint --all-targets`: clean, after
fixing one `clippy::map_entry` finding in `store.rs`.

## Consequences

**Positive:**

- Demonstrates, with real measurement rather than architectural argument,
  that content-defined chunking's resynchronization property produces a
  large (~8.4x measured here) reduction in incremental checkpoint bytes
  versus a real (non-strawman) fixed-block-chunking baseline, on top of
  an even larger (~19.3x) reduction versus naive full re-snapshotting.
- The witness-chain manifest is small (32 bytes × chunk count) relative
  to the checkpoint payload, giving a cheap, tamper-evident index into a
  much larger content-addressed store — useful independent of whether CDC
  specifically is adopted, since the same manifest shape works over
  fixed-block chunks too (as measured here).
- Reuses `ruvector-retrieval-receipt`'s exact chaining discipline
  (domain-separated byte-prefix hashing, explicit multi-variant failure
  modes) rather than inventing a new provenance scheme, keeping the
  ecosystem's tamper-evidence patterns consistent across write, read, and
  now checkpoint paths.

**Negative / costs:**

- CDC's chunking throughput (460.7 MB/s) is roughly 2.5x slower than
  fixed-block's (1,140.4 MB/s) for the same input, because CDC must
  inspect every input byte through the gear-hash rolling function
  regardless of how much output ends up deduplicated — irrelevant at this
  benchmark's scale (~24 ms/round) but a real cost that would need
  re-measuring at collection sizes where chunking time is not negligible
  against checkpoint interval.
- The measured ~12%/~5% ratios are specific to this benchmark's churn
  profile (small, scattered edits). A churn pattern that rewrites the
  entire blob every round (e.g. full re-embedding after a model swap)
  would erase CDC's advantage entirely, since there is nothing left to
  deduplicate against — disclosed in the nightly README's Limitations,
  not hidden behind the headline ratio.
- No signing of chain roots: witness manifests here are commitments, not
  signatures, matching the same open item `ruvector-proof-gate` and
  `ruvector-retrieval-receipt` already carry.
- The synthetic index format is not `ruvector-core`'s real binary layout
  (see Alternatives Considered) — the measured ratios are not yet a claim
  about the production checkpoint path.

## Alternatives Considered

- **Fixed-size block chunking as the production candidate itself, not
  just the baseline.** Rejected: it captures none of CDC's
  resynchronization property; a single edit near the start of a blob
  invalidates the content of nearly every downstream block even though
  block *positions* never move, as directly demonstrated by
  `fixed_boundaries_shift_every_block_content_after_an_edit` and confirmed
  by the measured 5,146,173-vs-609,120 bytes/round gap.
- **Reimplementing `ruvector-core`'s actual HNSW binary layout for this
  experiment.** Rejected for scope: correctly reproducing an internal,
  versioned binary format inside one crate in one night risks silently
  testing against an incorrect reproduction of that format, which would
  be indistinguishable from testing a strawman. The synthetic format used
  here shares the real shape (flat vector table + adjacency, tombstone
  deletes) and the churn ratios are realistic, but this is explicitly
  named as the next integration step, not claimed as done.
- **Rabin fingerprinting instead of a gear hash for the rolling chunker.**
  Not implemented: FastCDC's own contribution over Rabin fingerprinting is
  a cheaper per-byte rolling-hash update (one shift + one table lookup +
  one add) for comparable boundary quality; a second from-scratch Rabin
  implementation was judged lower value than the fixed-block contrast
  already establishing "why content-defined chunking at all."
- **Waiting for the `ruvector harness darwin` CLI rather than an in-crate
  sweep.** Rejected: the CLI does not exist in this environment (verified,
  see Context), and the run's own instructions require verifying
  capabilities before assuming them rather than blocking on unconfirmed
  tooling.

## Implementation Plan

1. (This ADR) Land the experimental crate, benchmark, Darwin-lite sweep,
   and tests — unintegrated, feature-isolated.
2. If promoted: integrate as an optional checkpoint backend for
   `ruvector-snapshot`, reproducing its actual serialization format
   instead of this crate's synthetic one, gated behind a Cargo feature so
   the default build and default checkpoint path are unaffected.
3. Re-measure the same steady-state ratios against the real format and at
   larger collection sizes (10x, 100x) — required before any production
   overhead/savings claim (see Rejection Criteria).
4. Design signed chain roots (mirroring the same open item named in
   ADR-304) so checkpoint provenance survives beyond commitment-only
   tamper-evidence.
5. A concrete `rvf-manifest`/`rvf-wire` prototype translating
   `CheckpointManifest` into an actual RVF portable-artifact shape, moving
   the RVF-implications analysis in the nightly README from "structurally
   compatible" to "measured."
6. MCP surface: a narrow, read-only `checkpoint_verify` tool
   (`{manifest, prev_root} -> {verified: bool, error: Option<str>}`),
   never exposing raw chunk contents beyond what verification requires.

## API Shape

```rust
let mut state = IndexState::build(n, dim, degree, seed);
let mut checkpointer = Checkpointer::new(Variant::Cdc(CdcParams::new(512, 2048, 8192)));

for round in 0..rounds {
    if round > 0 { state.churn(n_insert, n_delete, n_update); }
    let blob = state.serialize();
    let (stats, manifest) = checkpointer.checkpoint(round, &blob);
    let root_before = checkpointer.root_before(&manifest);
    let bytes = witness::verify(&root_before, &manifest, checkpointer.store())
        .expect("checkpoint must verify");
    assert_eq!(bytes, blob);
}
```

## Feature Flags

None yet — the crate is opt-in by virtue of not being a dependency of any
other crate. A `cdc-checkpoint` feature flag on `ruvector-snapshot` is the
proposed integration point if promoted (see Implementation Plan).

## Benchmark Evidence

See `docs/research/nightly/2026-08-27-cdc-witness-checkpoint/README.md`
for the full methodology and raw `cargo run --release` output.

## Security

- No `unsafe` code in the crate (no blocks present; the crate does not
  yet carry `#![forbid(unsafe_code)]` as a compile-time guarantee — a
  one-line follow-up).
- Only dependency is `sha2`, already used by `ruvector-proof-gate` and
  `ruvector-retrieval-receipt`.
- Domain-separated hashing (`b"ruvector:cdc:leaf:"`,
  `b"ruvector:cdc:chain:"`, `b"ruvector:cdc:content:"` prefixes) prevents
  hash confusion between a chunk leaf, a chain step, and a content-hash
  computation.
- **Threat model, stated precisely:** the witness chain detects
  post-issuance corruption or forgery of a checkpoint manifest or its
  referenced chunks. It does **not** prove the checkpoint producer itself
  ran honestly, and it does **not** bind a checkpoint to an
  independently-attested index state — both would require this crate's
  write path to be proof-gated (e.g. via `ruvector-proof-gate`), which is
  not integrated here.
- Chunk-count amplification from an adversarially constructed blob is
  bounded (`blob_len / min_size` chunks in the worst case) but was not
  attempted or measured; open question, not a made claim.

## Governance

Witness manifests here are commitments, not authorizations — the same
posture as `ruvector-retrieval-receipt` (ADR-304). A verified checkpoint
proves "this is the checkpoint that was committed," not "this checkpoint
was produced by an authorized process." Capability/authorization gating
(the `ruvector-capgated` pattern) is a complementary, unimplemented layer.

## Failure Modes

- `witness::verify` returns `MissingChunk` if any referenced chunk is
  absent from the store (tested: `missing_chunk_is_detected`).
- A manifest whose declared `content_hash` does not match the store's
  actual reconstruction is rejected with `ContentHashMismatch` before the
  chain root is even checked (tested:
  `manifest_lying_about_content_hash_is_rejected`).
- A manifest with an honest content hash but a forged `chain_root` is
  rejected with `ChainRootMismatch` (tested:
  `manifest_with_forged_chain_root_is_rejected`) — deliberately
  distinguished from the content-hash failure so a consumer knows which
  invariant broke.
- A full-blob-rewrite churn pattern degrades CDC to approximately
  full-snapshot cost, since nothing remains to deduplicate against — a
  boundary condition, not a defect.

## Migration

N/A — new, unintegrated crate.

## Rollback

Delete `crates/ruvector-cdc-checkpoint` and its workspace member entry in
the root `Cargo.toml`; no other crate depends on it.

## Rejection Criteria

This direction should be rejected for production promotion if any of the
following hold on re-measurement at larger scale or against the real
`ruvector-snapshot` format:

- CDC's advantage over fixed-block chunking shrinks materially below the
  measured ~12% ratio once measured against `ruvector-core`'s real binary
  layout instead of this crate's synthetic one.
- Chunking throughput falls to a level that matters relative to real
  checkpoint frequency at larger collection sizes (unmeasured above
  ~11 MB/round here).
- A representative agent-memory churn pattern, measured from real
  workload traces rather than this experiment's assumed ratios, turns out
  to resemble "rewrite everything" more than "small scattered edits" —
  this would erase CDC's advantage entirely (disclosed in Consequences).
- The witness-chain manifest's own cumulative storage growth becomes a
  material fraction of bytes saved at long checkpoint histories
  (unmeasured beyond 30 rounds here).

## Open Questions

- What does the real `ruvector-snapshot` binary format's edit-locality
  look like under realistic agent-memory churn, and does it chunk as
  favorably as this experiment's synthetic flat layout?
- What is the right signing story for `chain_root` — this ADR
  deliberately leaves signing out of scope, matching
  `ruvector-proof-gate`'s and `ruvector-retrieval-receipt`'s current state.
- Should the witness-chain manifest itself be pruned/compacted over very
  long checkpoint histories, and can that be done without losing
  tamper-evidence for retained checkpoints?
- Does an adversarially constructed blob's worst-case chunk-count
  amplification matter in practice, and should `ChunkStore` bound it
  explicitly (e.g. a maximum chunks-per-checkpoint guard)?
