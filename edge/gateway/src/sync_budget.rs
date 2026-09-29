//! Size budget of the synchronous M3 collection passes (snapshot create,
//! export, restore).
//!
//! Each of them reads, re-encodes, checksums and writes the whole
//! collection inside **one** HTTP invocation. On Workers Free (10 ms CPU
//! per request) that is only safe for a small collection: one 512-row page
//! at 384 dims is ~786 KB of f32, and at the ~100 MB/s wasm hashing rate
//! the rest of the gateway assumes that is already most of the budget. A
//! killed pass is not resumable (a snapshot burns its epoch, an export
//! leaves an unfinished multipart upload, a restore leaves its journal to
//! the next restore), so the pass is refused up front instead: `413` when
//! the collection holds more than [`SYNC_MAX_FLOATS`] stored floats
//! (rows × dim), checked before the first side effect and again while the
//! rows are read (a collection that grows mid-read is aborted, not killed).
//!
//! Raising this constant is a **Workers Paid precondition** (30 s default
//! CPU) until snapshot / export / restore become resumable queued jobs
//! like the bulk import. All figures are arithmetic, not wasm measurements.

use crate::m3_wire::M3Backend;
use crate::service::{count_of, shard_meta, unexpected};
use crate::wire::{CollectionWire, ShardCall, ShardOut};
use ruvector_edge_store::{CallerContext, ErrorCode, OpError};

/// Stored floats (rows × dim) one synchronous pass may process: 1 MiB of
/// f32 (682 rows at 384 dims).
pub const SYNC_MAX_FLOATS: u64 = 1 << 18;

/// Rows of a `dim`-dimensional collection within the budget.
pub fn max_rows(dim: u32) -> u64 {
    SYNC_MAX_FLOATS / u64::from(dim.max(1))
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
/// budget before any side effect.
pub async fn check_live<B: M3Backend>(
    b: &B,
    ctx: &CallerContext,
    e: &CollectionWire,
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
    check(e, rows)?;
    Ok(rows)
}

/// Rows read so far by one pass, refused once past the budget.
#[derive(Debug)]
pub struct Meter {
    rows: u64,
    max: u64,
}

impl Meter {
    /// A meter for collection `e`.
    pub fn new(e: &CollectionWire) -> Self {
        Meter {
            rows: 0,
            max: max_rows(e.cfg.dim),
        }
    }

    /// Count `n` more rows.
    pub fn add(&mut self, n: usize) -> Result<(), OpError> {
        self.rows = self.rows.saturating_add(n as u64);
        if self.rows > self.max {
            return Err(too_large());
        }
        Ok(())
    }
}
