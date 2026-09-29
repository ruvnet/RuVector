//! # ruvector-edge-snapshot
//!
//! Durability core for the ruvector edge services (ADR-351 §3 rows
//! rv-snapshot / rv-embed / rv-ingest, §6.3 R2 layout, §6.5 audit, §15 M3).
//! Pure Rust, no I/O, no clock, no RNG: it compiles for
//! `wasm32-unknown-unknown` and the Worker supplies bytes, time
//! ([`Clock`]), signatures ([`Signer`]) and embeddings ([`EmbeddingPort`]).
//!
//! * **Snapshots** — [`SnapshotWriter`] streams a shard's rows (id order) into
//!   rvf-wire `VEC_SEG`s packed into ≤ 8 MiB, ≤ [`MAX_ROWS_PER_CHUNK`]-row
//!   chunk objects with sha256 each, and a [`SealedManifest`] (tenant,
//!   collection, shard, epoch, dim, metric, rows, schema version, audit
//!   head, chunk list) whose root is sha256 over its canonical encoding,
//!   optionally Ed25519 signed. Chunk keys ([`SnapshotWriter::chunk_key`])
//!   are final before the manifest exists, so chunks are PUT as produced.
//! * **Witness chain** — [`WitnessChain`] is the per-tenant append-only hash
//!   chain over manifest roots; concurrent snapshots append in any order.
//!   [`verify_chain`] checks it against the ledger's trusted head.
//! * **Restore** — [`RestoreSession`] refuses a foreign tenant, another
//!   collection or shard, an unknown schema, a root mismatch, a bad or missing
//!   signature (keys are tenant-scoped), a root absent from the caller's
//!   chain, quota overruns and any chunk that is reordered, truncated,
//!   resized or altered — each with its own [`SnapshotError`] variant.
//! * **RVF export / import** — [`RvfExporter`] writes the rvf-runtime 0.2.0
//!   layout (legacy CRC content hash, manifest last) plus an id/metadata
//!   sidecar and, for `:export`, a signed export witness ([`verify_export`]);
//!   [`inspect_tail`] + [`RvfImporter`] pull bounded upsert batches out of
//!   any such file with dimension, metric, size and quota checks up front.
//! * **Bulk import jobs** — [`ImportJob`] (queued → running → done/failed),
//!   bound to its upload and pinned limits, resumable by [`ImportCursor`],
//!   idempotent via [`ImportJob::op_id`].
//! * **Embeddings** — [`EmbeddingBatcher`] plans bge-small-en-v1.5 calls
//!   under text/token limits and validates responses.
//!
//! ## rvf CLI cross-check
//!
//! `tests/rvf_cli.rs` writes a witnessed export under `CARGO_TARGET_TMPDIR`
//! and, when `RVF_CLI` points at an `rvf` binary (built from
//! `crates/rvf/rvf-cli`), runs `rvf inspect --json` and `rvf query --json`
//! against it: see that test for the exact commands and the asserted output
//! (dimension, `total_vectors`, segment types `Profile`/`Vec`, and the
//! nearest neighbour of a stored vector at distance 0).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod bytes;
pub mod embed;
pub mod error;
pub mod job;
pub mod manifest;
pub mod restore;
mod rowcodec;
pub mod rvf_export;
pub mod rvf_format;
pub mod rvf_import;
pub mod rvf_summary;
pub mod rvf_witness;
pub mod types;
pub mod witness;
pub mod writer;

pub use embed::{
    embed_all, estimate_tokens, EmbedBatch, EmbedLimits, EmbedOptions, EmbeddingBatcher,
    EmbeddingPort,
};
pub use embed::{BGE_SMALL_DIM, BGE_SMALL_MODEL};
pub use error::{EmbedError, ImportError, JobError, SnapshotError};
pub use job::{batch_op_id, FailCode, ImportJob, JobBinding, JobState, ObjectFacts};
pub use manifest::{key_prefix, signing_message, ChunkRef, Manifest, SealedManifest};
pub use manifest::{SignatureBlock, MAX_ROWS_PER_CHUNK};
pub use restore::{
    ChainProof, RestoreQuota, RestoreSession, RestoreSummary, RestoreTarget, SignaturePolicy,
};
pub use rvf_export::{ExportSummary, RvfExporter, DEFAULT_ROWS_PER_SEGMENT};
pub use rvf_import::{ImportBatch, ImportCursor, ImportSpec, ImportTotals, RvfImporter};
pub use rvf_summary::{inspect_tail, ImportLimits, RvfSummary};
pub use rvf_witness::{export_signing_message, verify_export, ExportIdentity, ExportWitness};
pub use types::{Clock, FixedClock, Metric, Row, ShardRef};
pub use witness::{genesis, verify_chain, verify_chain_from, verify_signature, Ed25519Verifier};
pub use witness::{SignatureVerifier, Signer, WitnessChain, WitnessEntry};
pub use writer::{manifest_key, snapshot_key, Chunk, SnapshotLimits, SnapshotSpec};
pub use writer::{SnapshotWriter, DEFAULT_SEGMENT_PAYLOAD, MAX_CHUNK_BYTES};
