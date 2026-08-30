# Content-Defined Chunking Cuts Vector-Index Checkpoint Bytes by ~8-19x

## Problem

Vector/graph indexes used as agent memory are increasingly checkpointed
periodically for durability and portability, not written once. A naive
checkpoint re-serializes and re-writes the whole collection every round,
so its cost scales with the *size of the collection*, not the size of
whatever actually changed since the last checkpoint — backwards for a
collection that grows indefinitely and gets checkpointed on a schedule.

Fixed-size block chunking (the common naive incremental-backup fix)
partially helps by deduplicating unchanged blocks, but it has a
well-known weakness: a single insertion or deletion shifts every
downstream byte, so even though block *boundaries* stay at the same
positions, their *content* changes — invalidating nearly every block
after the edit.

## Hypothesis

Given an HNSW-style vector+graph index checkpointed under realistic
agent-memory churn (small, scattered insert/delete/update batches between
checkpoints), replacing full re-snapshotting or fixed-block chunking with
content-defined chunking (CDC) — a FastCDC-style rolling-hash chunker that
anchors boundaries to local byte content instead of fixed offsets — should
substantially reduce steady-state incremental checkpoint bytes, while
every checkpoint remains bit-identically reconstructible and
tamper-evident via a witness hash chain.

## Technical Design

Three pieces, each small and independently testable:

1. **Chunker** — a deterministic FastCDC-style gear-hash rolling chunker
   (a 256-entry gear table generated at compile time via `splitmix64`, so
   boundaries are reproducible across builds) alongside a fixed-size
   baseline chunker, sharing one `(start, end)` range interface.
2. **Content-addressed store** — chunks keyed by SHA-256; inserting an
   already-present chunk costs zero new bytes. This turns "how much did
   this checkpoint actually change" into a directly measurable quantity:
   bytes written for genuinely new chunks.
3. **Witness chain** — a sequential, domain-separated SHA-256 hash chain
   over each checkpoint's ordered chunk-hash list plus its full-content
   hash, chained to the previous checkpoint's root. Verification
   reconstructs the checkpoint from the store and recomputes the root,
   returning one of three explicit failure modes (`MissingChunk`,
   `ContentHashMismatch`, `ChainRootMismatch`) rather than a bare boolean
   — so a consumer knows exactly which invariant broke.

The defining property under test is **resynchronization**: fixed-size
chunking's block boundaries are pure position arithmetic, so an edit near
the start of a blob changes the *content* of every downstream block
without moving any boundary — every block after the edit must be
re-stored. Content-defined boundaries are anchored to local byte content,
so they resynchronize within a bounded distance of the edit; blocks
entirely before the edited region are provably unaffected. A unit test
demonstrates this directly by inserting bytes mid-stream and asserting
that boundaries before the insertion point are byte-for-byte identical
before and after.

## Implementation

A synthetic, deterministic (seeded, not `rand`-crate-dependent) vector+graph
index generates the byte blob each variant chunks: a flat vector table
(with tombstone-on-delete slots) plus per-node adjacency lists, updated
via a `churn()` method modeling realistic agent-memory update patterns —
new memories arriving, superseded ones retired, others re-embedded in
place. All three chunking strategies (`FullSnapshot`, `FixedBlock`, `Cdc`)
share one `Checkpointer` type, so the benchmark's only independent
variable is chunking strategy, not incidental implementation differences.

```rust
let mut checkpointer = Checkpointer::new(Variant::Cdc(CdcParams::new(512, 2048, 8192)));
let (stats, manifest) = checkpointer.checkpoint(round, &blob);
let bytes = witness::verify(&checkpointer.root_before(&manifest), &manifest, checkpointer.store())
    .expect("checkpoint must verify");
assert_eq!(bytes, blob); // bit-identical reconstruction, every round
```

## Benchmark Evidence

20,000 vectors, dim=128, degree=16 adjacency, 30 checkpoint rounds, churn
of 0.2% inserts / 0.1% deletes / 0.3% updates per round — deterministic
seed, release build, `rustc 1.94.1`, x86-64 Linux.

| Variant | avg new bytes/round (steady-state) | final resident bytes (30 rounds) | chunking throughput |
|---|---|---|---|
| Full re-snapshot | 11,734,378 | 351,836,960 | 1,084.7 MB/s |
| Fixed-block (4096B) + dedup | 5,146,173 | 160,779,040 | 1,140.4 MB/s |
| **CDC (avg_size=2048) + dedup + witness chain** | **609,120** | **29,204,492** | 460.7 MB/s |

CDC writes **11.84%** of fixed-block's bytes and **5.19%** of full
re-snapshot's bytes per round at steady state — against fixed acceptance
thresholds of ≤50% and ≤20% respectively, set before the run. Every
checkpoint, every round, every variant reconstructed bit-identically
through the actual witness-verification path (90 checks total) — the
benchmark binary asserts this in-loop and would panic rather than report
a false pass.

A bounded, single-generation, four-candidate parameter sweep over the
CDC chunker's target chunk size found `avg_size=1024` further improves
steady-state bytes (341,856/round) at roughly double the chunk count
(9,115 vs 4,575) — a real, measured bytes-vs-bookkeeping tradeoff, not
free further improvement.

Full raw output, methodology, and 15 passing unit tests are in the
[nightly research README](README.md) and
[ADR-340](../../../adr/ADR-340-cdc-witness-checkpoint.md).

## Limitations

- Uses a synthetic index format matching `ruvector-snapshot`'s general
  shape (vector table + adjacency, tombstone deletes), not its actual
  production binary layout — integrating against the real format and
  re-measuring is the named next step, not claimed as already done.
- Measured at one collection size and one churn profile. A churn pattern
  that rewrites the entire collection every round (e.g. full re-embedding
  after a model swap) would erase CDC's advantage entirely, since nothing
  would remain to deduplicate against.
- Witness manifests are unsigned commitments, not signatures — they
  detect post-issuance tampering or corruption, not a dishonest
  checkpoint producer.
- No RVF/RVM wire-format integration, cross-platform measurement, or WASM
  build was attempted.

## Production Relevance

The measured ratios (~8x smaller than fixed-block, ~19x smaller than full
re-snapshot at steady state) are large enough to matter for any
periodically-checkpointed, growing agent-memory collection — storage
cost, network sync cost to edge replicas, and flash write-cycle budget on
embedded deployments all scale with checkpoint bytes. The witness chain
adds tamper-evidence to that reduced payload at negligible additional
cost (one SHA-256 call per chunk boundary).

## RuVector Ecosystem Implications

This connects five ecosystem capabilities: vector/graph index durability
(`ruvector-snapshot`'s role), witness/provenance (reusing
`ruvector-retrieval-receipt`'s chaining pattern, not reinventing it),
agent memory (the churn model's source), RVF (a chunked, witness-chained
checkpoint is a structural fit for a portable, incrementally-syncable
artifact — not yet integrated against the real `rvf-manifest`/`rvf-wire`
format), and ruFlo (a concrete scheduled checkpoint-then-verify workflow).

## Future Direction

1. Integrate against `ruvector-snapshot`'s real serialization format and
   re-measure.
2. Sweep collection size and churn intensity to find where the measured
   ratios hold, improve, or degrade, and where chunking throughput
   becomes a binding constraint.
3. Design signed chain roots, closing the same non-repudiation gap
   `ruvector-proof-gate` and `ruvector-retrieval-receipt` already carry.
4. A concrete RVF manifest/wire prototype, moving the RVF analysis from
   structural compatibility to a measured integration.

## References

- Xia, W. et al., "FastCDC: a Fast and Efficient Content-Defined Chunking
  Approach for Data Deduplication," USENIX ATC 2016.
- `ruvector-retrieval-receipt` / ADR-304 — the witness-chaining pattern
  this work reuses for checkpoints instead of query receipts.
- `ruvector-proof-gate` / ADR-227 — the write-path tamper-evidence this
  work is deliberately distinguished from (checkpoint provenance, not
  write provenance).
- Public snapshot/backup documentation review of Milvus, Qdrant, Weaviate,
  Pinecone, LanceDB, FAISS, pgvector, Chroma, and Vespa: none document a
  content-defined-chunking-based incremental snapshot mechanism as of
  this research.
