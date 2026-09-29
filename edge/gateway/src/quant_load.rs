//! `QuantShard` cold load, replay, chunked rebuild and snapshot flush
//! (ADR-351 §3 rv-quant; persist v2 = `rbqx0002`, G4c).
//!
//! Cold load decodes the stored snapshot frames (packed codes, norms, keys;
//! only the seeded rotation is regenerated), applies the tombstones written
//! after it, then re-encodes the rows written after it (`wseq >
//! snapshot_seq`). Without a usable snapshot (never flushed, or corrupt:
//! self-healing) every row is re-encoded from its f32 original.
//!
//! Re-encoding is the expensive part (`rows × (rotation apply + dim)` work
//! units; ≈ 11k units per 384-dim row with the Hadamard rotation), so it
//! runs in **turns** of at most [`TURN_UNITS`]: a request that finds the
//! shard mid-rebuild advances one turn and answers `503 shard_unavailable`
//! (retryable) until it is done, and the DO alarm keeps advancing it. A
//! snapshot load that alone exceeds the quant budget is `413` (the shard is
//! too large for this isolate), never a `500`.

use crate::quant_store::{self as qs, QMeta};
use ruvector_edge_quant::budget::{self, Budget};
use ruvector_edge_quant::persist::{load_frames, save_frames, MAX_FRAME_BYTES};
use ruvector_edge_quant::{QuantConfig, QuantError, QuantShard, RandomRotationKind};
use ruvector_edge_store::{ErrorCode, OpError, SqlStore};

/// Work units one request or alarm turn may spend re-encoding rows
/// (≈ 4 ms native at the crate's measured ≈ 1 ns/unit; wasm ≈ 1.1–1.5×).
pub const TURN_UNITS: u64 = 4_000_000;
/// Snapshot frame size (one DO SQLite row each; rows are capped at 2 MB).
pub const FRAME_BYTES: usize = MAX_FRAME_BYTES;
/// Rows written since the snapshot that make a flush urgent.
pub const FLUSH_ROWS: u64 = 1024;
/// Delay of a non-urgent flush after a write.
pub const FLUSH_AFTER_MS: u64 = 30_000;

/// The rotation every edge quant shard uses (O(D log D) apply, near-free
/// rebuild on load; see `QuantConfig::new`).
pub const ROTATION: RandomRotationKind = RandomRotationKind::HadamardSigned;

/// A loaded, queryable shard.
pub struct Resident {
    /// The codes.
    pub q: QuantShard,
    /// Rows (re-)encoded since the stored snapshot.
    pub dirty_rows: u64,
}

/// A shard being (re-)encoded from its rows.
pub struct Rebuild {
    q: QuantShard,
    /// `Some(snapshot_seq)`: replay rows written after the snapshot;
    /// `None`: re-encode every row.
    since: Option<u64>,
    cursor: u64,
    /// Rows encoded so far.
    pub done_rows: u64,
}

impl Rebuild {
    /// Bytes the partly built shard already holds (registered while it
    /// rebuilds, not only once ready).
    pub fn resident_bytes(&self) -> u64 {
        self.q.resident_bytes()
    }

    /// Work units of re-encoding `rows` rows of this shard.
    pub fn encode_units(&self, rows: u64) -> u64 {
        budget::encode_units(rows, self.q.dim(), ROTATION)
    }

    /// Finished: the shard, dirty when rows were (re-)encoded.
    pub fn into_resident(self) -> Resident {
        Resident {
            q: self.q,
            dirty_rows: self.done_rows,
        }
    }
}

/// Measured cost of the last cold open (tests and telemetry).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// Snapshot rows decoded.
    pub snapshot_rows: u64,
    /// Snapshot bytes read.
    pub snapshot_bytes: u64,
    /// Work units charged for the snapshot decode (`budget::load_units`).
    pub load_units: u64,
    /// `true` when a stored snapshot failed verification (rebuild instead).
    pub corrupt: bool,
}

/// The shard configuration a `qmeta` describes.
pub fn config(meta: &QMeta, budget: Budget) -> Result<QuantConfig, OpError> {
    let metric = meta.metric.ok_or(OpError::new(
        ErrorCode::ShardUnavailable,
        "quant shard config",
    ))?;
    Ok(QuantConfig {
        dim: meta.dim as usize,
        metric,
        rotation: ROTATION,
        seed: meta.seed,
        budget,
    })
}

/// Map a quant error: budgets stay `413`, the rest keep their codes.
pub fn qerr(e: QuantError) -> OpError {
    OpError::from(e)
}

/// Cold open: decode the snapshot (if any) and apply later tombstones;
/// the returned [`Rebuild`] still has to replay / re-encode rows.
pub fn open(
    store: &dyn SqlStore,
    meta: &QMeta,
    budget: Budget,
) -> Result<(Rebuild, LoadReport), OpError> {
    let cfg = config(meta, budget)?;
    let mut report = LoadReport::default();
    if meta.frames > 0 {
        let frames = qs::frames(store)?;
        report.snapshot_bytes = frames.iter().map(|f| f.len() as u64).sum();
        let loaded = if frames.len() as u64 == meta.frames {
            load_frames(frames.iter().map(Vec::as_slice), budget)
        } else {
            Err(QuantError::Io("frame count".into()))
        };
        match loaded {
            Ok(mut q) if q.dim() == cfg.dim && q.config().metric == cfg.metric => {
                report.snapshot_rows = q.len() as u64;
                report.load_units = budget::load_units(
                    report.snapshot_bytes,
                    report.snapshot_rows,
                    cfg.dim,
                    ROTATION,
                );
                q.delete(&qs::tombs_since(store, meta.snap_seq)?);
                let rb = Rebuild {
                    q,
                    since: Some(meta.snap_seq),
                    cursor: 0,
                    done_rows: 0,
                };
                return Ok((rb, report));
            }
            Err(e @ QuantError::BudgetExceeded { .. }) => return Err(qerr(e)),
            // Corrupt, foreign or torn: re-derive from the rows.
            _ => report.corrupt = true,
        }
    }
    let q = QuantShard::new(cfg).map_err(qerr)?;
    let rb = Rebuild {
        q,
        since: None,
        cursor: 0,
        done_rows: 0,
    };
    Ok((rb, report))
}

/// Rows one turn of `units` re-encodes at this shard's dimension.
pub fn rows_per_turn(dim: usize, units: u64) -> usize {
    let per_row = budget::encode_units(1, dim, ROTATION).max(1);
    usize::try_from((units / per_row).max(1)).unwrap_or(usize::MAX)
}

/// Advance a rebuild by at most `units` of encoding; `true` when done.
pub fn step(rb: &mut Rebuild, store: &dyn SqlStore, units: u64) -> Result<bool, OpError> {
    let limit = rows_per_turn(rb.q.dim(), units);
    let rows = qs::page(store, rb.since, rb.cursor, limit)?;
    if let Some((last, _)) = rows.last() {
        rb.cursor = *last;
    }
    let refs: Vec<(u64, &[f32])> = rows.iter().map(|(k, v)| (*k, v.as_slice())).collect();
    if !refs.is_empty() {
        rb.q.upsert(&refs).map_err(qerr)?;
    }
    rb.done_rows += refs.len() as u64;
    Ok(rows.len() < limit)
}

/// Write a fresh snapshot of `r` covering `meta.write_seq`.
pub fn flush(r: &mut Resident, store: &dyn SqlStore, meta: &mut QMeta) -> Result<(), OpError> {
    let frames = save_frames(&r.q, FRAME_BYTES).map_err(qerr)?;
    qs::put_frames(store, &frames, meta.write_seq)?;
    meta.frames = frames.len() as u64;
    meta.snap_seq = meta.write_seq;
    r.dirty_rows = 0;
    Ok(())
}
