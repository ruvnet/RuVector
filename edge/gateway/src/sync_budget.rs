//! Size budget of the synchronous M3 collection passes (snapshot create,
//! export, restore).
//!
//! Each of them reads, re-encodes, checksums and writes the whole
//! collection inside **one** HTTP invocation. A killed pass is not
//! resumable (a snapshot burns its epoch, an export leaves an unfinished
//! multipart upload, a restore leaves its journal to the next restore), so
//! an oversized pass is refused up front instead: `413` when the collection
//! holds more stored floats (rows × dim) than its pass allows, checked
//! before the first side effect and again while the rows are read (a
//! collection that grows mid-read is aborted, not killed).
//!
//! **Export** ([`EXPORT_MAX_FLOATS`], Workers Paid) streams: one 512-row
//! page per DO call, one 8 MiB R2 part plus one segment pair in the
//! gateway, so its bound is CPU. Native `--release` (serde_json, `opt-level
//! = "z"`): JSON decode ≈ 25.6 ns/float, encode ≈ 25.6 ns/float, soft
//! SHA-256 ≈ 15 ns/float; with the ≈ 1.5× wasm factor the pass is ≈ 100–120
//! ns/float, so 2^24 floats is ≈ 1.7–2 s (≈ 6 % of the 30 s `cpu_ms`) and
//! ≈ 86 page calls + 8 R2 parts per 16M floats of the 10,000-subrequest
//! Paid default.
//! A second cap, [`EXPORT_MAX_BYTES`], counts stored bytes (values, ids,
//! metadata text) so a low-dim, metadata-heavy collection cannot use the
//! float cap to push gigabytes of metadata through one pass.
//!
//! **Snapshot and restore** ([`SYNC_MAX_FLOATS`]) stay at the Free value:
//! restore's `VectorShard` commit (`m3_shard::commit`) collects the whole
//! staged set (f32 values plus parsed metadata) into one `Vec` in one DO
//! turn, so its bound is the isolate's ≈ 5 MB of spare memory, not CPU;
//! and snapshot shares the cap so that every snapshot stays restorable.
//! Raising it needs a paging commit first.

use crate::m3_wire::M3Backend;
use crate::service::{count_of, shard_meta, unexpected};
use crate::wire::{CollectionWire, ShardCall, ShardOut};
use ruvector_edge_store::{CallerContext, ErrorCode, OpError};

/// Stored floats (rows × dim) one snapshot or restore may process: 1 MiB
/// of f32 (682 rows at 384 dims). Memory-bound (restore commit), see the
/// module doc.
pub const SYNC_MAX_FLOATS: u64 = 1 << 18;

/// Stored floats one export may process: 64 MiB of f32 (43,690 rows at
/// 384 dims, about one full shard's `M2_SHARD_FLOAT_CAP`). CPU-bound; 2^26
/// would still be ≈ 27 % of 30 s, 2^24 keeps headroom. (Free: 2^18.)
pub const EXPORT_MAX_FLOATS: u64 = 1 << 24;

/// Stored bytes one export may process alongside the float cap: `f32`
/// values (4 per float) plus every row's id and metadata text. The float
/// cap alone let a low-dim collection carry ≈ 4 KiB of metadata per row
/// through the JSON decode / re-encode (2^24 rows at one dim). 128 MiB is
/// twice the float cap's own 64 MiB, so ids and light metadata never bind
/// before [`EXPORT_MAX_FLOATS`] (a full 384-d export with 256-byte ids is
/// ≈ 78 MB), while metadata-heavy rows stop near 32k at 4 KiB each; string
/// bytes cost less than float bytes, so the pass stays within ≈ 3–4 s of
/// wasm. The pre-check sees only row counts (`Stats` carries no metadata
/// bytes), so the byte cap is enforced while the rows are read (the
/// multipart upload is aborted, `413`).
pub const EXPORT_MAX_BYTES: u64 = 128 << 20;

/// Rows of a `dim`-dimensional collection within the snapshot / restore
/// budget.
pub fn max_rows(dim: u32) -> u64 {
    SYNC_MAX_FLOATS / u64::from(dim.max(1))
}

/// Rows of a `dim`-dimensional collection within the export budget.
pub fn export_max_rows(dim: u32) -> u64 {
    EXPORT_MAX_FLOATS / u64::from(dim.max(1))
}

/// The refusal.
pub fn too_large() -> OpError {
    OpError::new(
        ErrorCode::PayloadTooLarge,
        "collection too large for a synchronous snapshot, export or restore",
    )
}

/// `Err(413)` when `rows` of collection `e` exceed the budget.
pub fn check(e: &CollectionWire, rows: u64) -> Result<(), OpError> {
    if rows > max_rows(e.cfg.dim) {
        return Err(too_large());
    }
    Ok(())
}

/// Live row count of `e` (one `Stats` per shard), checked against the
/// snapshot / restore budget before any side effect.
pub async fn check_live<B: M3Backend>(
    b: &B,
    ctx: &CallerContext,
    e: &CollectionWire,
) -> Result<u64, OpError> {
    live_within(b, ctx, e, max_rows(e.cfg.dim)).await
}

/// [`check_live`] against the export budget.
pub async fn check_live_export<B: M3Backend>(
    b: &B,
    ctx: &CallerContext,
    e: &CollectionWire,
) -> Result<u64, OpError> {
    live_within(b, ctx, e, export_max_rows(e.cfg.dim)).await
}

async fn live_within<B: M3Backend>(
    b: &B,
    ctx: &CallerContext,
    e: &CollectionWire,
    max: u64,
) -> Result<u64, OpError> {
    let mut rows = 0u64;
    for i in count_of(e)?.indices() {
        let dm = shard_meta(ctx, e, i)?;
        match crate::backend::shard(b, &dm, ShardCall::Stats).await {
            Ok(ShardOut::Stats { count, .. }) => rows = rows.saturating_add(count),
            Ok(_) => return Err(unexpected()),
            Err(err) => return Err(err.into_op()),
        }
    }
    if rows > max {
        return Err(too_large());
    }
    Ok(rows)
}

/// Rows (and, for export, stored bytes) read so far by one pass, refused
/// once past the budget.
#[derive(Debug)]
pub struct Meter {
    rows: u64,
    max: u64,
    bytes: u64,
    max_bytes: u64,
}

impl Meter {
    /// A snapshot / restore meter for collection `e`.
    pub fn new(e: &CollectionWire) -> Self {
        Self::within(max_rows(e.cfg.dim))
    }

    /// An export meter for collection `e`: rows and stored bytes.
    pub fn export(e: &CollectionWire) -> Self {
        Self::within(export_max_rows(e.cfg.dim)).with_bytes(EXPORT_MAX_BYTES)
    }

    /// A meter refusing more than `max` rows.
    pub fn within(max: u64) -> Self {
        Meter {
            rows: 0,
            max,
            bytes: 0,
            max_bytes: u64::MAX,
        }
    }

    /// This meter, also refusing more than `max_bytes` stored bytes.
    pub fn with_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Count `n` more rows.
    pub fn add(&mut self, n: usize) -> Result<(), OpError> {
        self.rows = self.rows.saturating_add(n as u64);
        if self.rows > self.max {
            return Err(too_large());
        }
        Ok(())
    }

    /// Count `n` more stored bytes (values, ids, metadata text).
    pub fn add_bytes(&mut self, n: u64) -> Result<(), OpError> {
        self.bytes = self.bytes.saturating_add(n);
        if self.bytes > self.max_bytes {
            return Err(too_large());
        }
        Ok(())
    }
}
