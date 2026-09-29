//! Memory and CPU budget accounting (ADR-351 §10 layer 3: "step/row counters
//! return `413 budget_exceeded` before `cpu_ms`").
//!
//! CPU is counted in abstract **work units**: one unit is roughly one 64-bit
//! popcount word or one f32 multiply-add. Every operation computes its cost
//! from sizes alone and checks it against the budget *before* doing the work
//! (or, for a cold load, before allocating anything), so a refusal is cheap
//! and deterministic across native and wasm32. The tests calibrate units to
//! wall time natively (see the crate docs for the measured ns/unit).
//!
//! Resident memory is the *capacity* of the code/norm/key arrays plus a
//! key-index entry per row plus the fixed rotation and cos-LUT, against the
//! ruvector-edge-store per-shard cap ([`SHARD_RESIDENT_CAP_BYTES`], 14 MB).
//! [`resident_bytes`] is the exact-capacity projection (a cold load builds
//! exact-capacity arrays); growth on upsert is charged at its peak, old
//! `packed` buffer included (see `QuantShard::upsert`), so the cap bounds
//! the heap and not just the modelled length.

use crate::error::{BudgetResource, QuantError, Result};
use ruvector_edge_store::shard::SHARD_RESIDENT_CAP_BYTES;
use ruvector_rabitq::RandomRotationKind;

/// Upper bound on the per-row cost of the `BTreeMap<u64, u32>` key index.
/// std's B-tree keeps every non-root node at ≥ 5 of 11 entries (also after
/// removals); on 64-bit a leaf is 144 B and an internal node 240 B, so the
/// tree holds at most `144·n/5 + 96·n/25 + 240 ≈ 32.7·n + 240` bytes.
/// Sequential keys measure ≈ 30 B/row (`tests/memory.rs`).
pub const KEY_INDEX_BYTES_PER_ROW: u64 = 34;

/// Row-independent part of that bound (one root/internal node).
pub const KEY_INDEX_FIXED_BYTES: u64 = 256;

/// Per-row cold-load work beyond the raw bytes (key-index insert, norm and
/// padding validation); calibrated so a Hadamard load (no rotation cost)
/// measures ≈ 1 ns/unit natively.
pub const LOAD_UNITS_PER_ROW: u64 = 100;

/// Smallest capacity step (rows) when the shard's arrays grow.
pub const MIN_GROW_ROWS: u64 = 64;

/// ADR-351 §7: `top_k` ≤ 100.
pub const MAX_TOP_K: u32 = 100;

/// Hard per-shard limits. All checks are `requested > limit → 413`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Resident bytes of one shard (default: the edge-store 14 MB cap).
    pub max_resident_bytes: u64,
    /// Rows per shard.
    pub max_vectors: u64,
    /// Work units of one query.
    pub max_query_units: u64,
    /// Work units of one cold load.
    pub max_load_units: u64,
    /// Rerank candidates (f32 rows fetched from the store) per query.
    pub max_rerank_candidates: u32,
    /// `top_k` per query.
    pub max_top_k: u32,
}

impl Default for Budget {
    /// Sized for the M4 design point (50k × 384 per shard) with headroom:
    /// a 50k × 384 query costs ≈ 0.88M units with 1,000 rerank candidates,
    /// a Haar cold load ≈ 62M units (dominated by the rotation rebuild).
    /// Measured natively (x86_64, release): ≈ 0.5–1.2 ns per unit; wasm32
    /// under Node (release, `tests/wasm.rs`): ≈ 1.1–1.5 ns per unit, so
    /// the load budget is ≲ 0.6 s of wasm CPU.
    fn default() -> Self {
        Budget {
            max_resident_bytes: SHARD_RESIDENT_CAP_BYTES,
            max_vectors: 150_000,
            max_query_units: 8_000_000,
            max_load_units: 400_000_000,
            max_rerank_candidates: 1_000,
            max_top_k: MAX_TOP_K,
        }
    }
}

/// Refuse with `413` if `requested > limit`.
#[inline]
pub fn check(resource: BudgetResource, limit: u64, requested: u64) -> Result<()> {
    if requested > limit {
        Err(QuantError::BudgetExceeded {
            resource,
            limit,
            requested,
        })
    } else {
        Ok(())
    }
}

/// 64-bit words per code at `dim`.
#[inline]
pub fn n_words(dim: usize) -> usize {
    dim.div_ceil(64)
}

/// Bytes one row of *capacity* holds in the SoA arrays (code words, f32
/// norm, u64 key).
#[inline]
pub fn vec_row_bytes(dim: usize) -> u64 {
    (n_words(dim) as u64) * 8 + 4 + 8
}

/// Resident bytes of one row at exact capacity (arrays + key index).
#[inline]
pub fn row_bytes(dim: usize) -> u64 {
    vec_row_bytes(dim) + KEY_INDEX_BYTES_PER_ROW
}

/// Bytes held by a rotation of this kind (mirrors `RandomRotation::bytes`).
pub fn rotation_bytes(dim: usize, kind: RandomRotationKind) -> u64 {
    let d = dim as u64;
    match kind {
        RandomRotationKind::HaarDense => d * d * 4,
        RandomRotationKind::HadamardSigned => 3 * (dim.next_power_of_two() as u64) * 4,
    }
}

/// Fixed (row-independent) resident bytes: rotation + cos-LUT + key-index root.
pub fn fixed_bytes(dim: usize, kind: RandomRotationKind) -> u64 {
    rotation_bytes(dim, kind) + (dim as u64 + 1) * 4 + KEY_INDEX_FIXED_BYTES
}

/// Projected resident bytes of a shard with `n` rows.
pub fn resident_bytes(n: u64, dim: usize, kind: RandomRotationKind) -> u64 {
    fixed_bytes(dim, kind).saturating_add(n.saturating_mul(row_bytes(dim)))
}

/// Work units to apply the rotation to one vector.
pub fn rotation_apply_units(dim: usize, kind: RandomRotationKind) -> u64 {
    let d = dim as u64;
    match kind {
        RandomRotationKind::HaarDense => d * d,
        RandomRotationKind::HadamardSigned => {
            let p = dim.next_power_of_two() as u64;
            3 * p + 2 * p * u64::from(p.trailing_zeros())
        }
    }
}

/// Work units to (re)generate the rotation from its seed.
pub fn rotation_build_units(dim: usize, kind: RandomRotationKind) -> u64 {
    let d = dim as u64;
    match kind {
        // Gaussian fill D² + Gram–Schmidt ≈ D³ multiply-adds.
        RandomRotationKind::HaarDense => d * d + d * d * d,
        RandomRotationKind::HadamardSigned => 3 * (dim.next_power_of_two() as u64),
    }
}

/// Work units of one query over `n` rows reranking `candidates` rows.
pub fn query_units(n: u64, dim: usize, kind: RandomRotationKind, candidates: u64) -> u64 {
    let scan = n.saturating_mul(n_words(dim) as u64 + 1);
    let rerank = candidates.saturating_mul(dim as u64);
    scan.saturating_add(rerank)
        .saturating_add(rotation_apply_units(dim, kind))
}

/// Work units of a cold load of `payload_bytes` holding `n` rows.
pub fn load_units(payload_bytes: u64, n: u64, dim: usize, kind: RandomRotationKind) -> u64 {
    (payload_bytes / 8)
        .saturating_add(n.saturating_mul(LOAD_UNITS_PER_ROW))
        .saturating_add(rotation_build_units(dim, kind))
        .saturating_add(rotation_apply_units(dim, kind))
}

/// Work units to encode `rows` vectors on upsert.
pub fn encode_units(rows: u64, dim: usize, kind: RandomRotationKind) -> u64 {
    rows.saturating_mul(rotation_apply_units(dim, kind) + dim as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn design_point_fits_default_budget() {
        let b = Budget::default();
        let k = RandomRotationKind::HaarDense;
        let res = resident_bytes(50_000, 384, k);
        assert!(res <= b.max_resident_bytes, "{res}");
        assert!(query_units(50_000, 384, k, 1_000) <= b.max_query_units);
        let payload = 50_000 * (48 + 4 + 8);
        assert!(load_units(payload, 50_000, 384, k) <= b.max_load_units);
    }

    #[test]
    fn refusal_is_413() {
        let e = check(BudgetResource::QueryUnits, 10, 11).unwrap_err();
        assert_eq!(e.status(), 413);
        assert!(check(BudgetResource::QueryUnits, 10, 10).is_ok());
    }
}
