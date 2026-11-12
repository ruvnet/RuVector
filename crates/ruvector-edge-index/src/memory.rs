//! Memory and load-time accounting so the store can enforce per-shard and
//! per-isolate caps before inserting (ADR §6.1 isolate budget, M2a/M2b
//! acceptance).
//!
//! Estimates are per resident **slot** and exclude the f32 vectors, which
//! stay in SQLite. Slots are `max_iid + 1` (`QuantFlatIndex::slots`,
//! `HnswIndex::node_count`), *not* the live count: tombstones and iid gaps
//! cost memory until a compaction. The live figure for a built index is
//! `memory_bytes()` (allocated capacity, which is what a cap pays for);
//! `growth_bytes(iid)` previews the next allocation.

use crate::hnsw::HnswParams;

/// ADR §6.1 resident-set cap per isolate (56 MB, decimal as in the ADR).
pub const ISOLATE_BUDGET_BYTES: usize = 56_000_000;

/// ADR §15 M2b lazy-load bound (p95), which caps the M2b shard size.
pub const LOAD_BUDGET_MS: f64 = 1000.0;

/// Measured full decode + validation cost (`from_chunk_iter`, 10k × 384,
/// m0 = 32, CRC-32C per chunk + one sha256), native x86-64, release
/// profile: 5.0 ms for 4.94 MiB. See the `bench` test.
pub const DECODE_MS_PER_MIB_NATIVE: f64 = 1.0;

/// The same decode under wasm32-wasip1 in node 22 (V8, no SHA-NI), release
/// profile: 11.2 ms for 4.94 MiB (≈ 2.3 ms/MiB; + 200-op replay p95 99 ms
/// at 10k). The figure to size Worker shards with.
pub const DECODE_MS_PER_MIB_WASM: f64 = 2.3;

/// M2a bytes per slot: one code byte per dimension plus the presence bit.
pub fn flat_bytes_per_vector(dim: usize) -> f64 {
    dim as f64 + 1.0 / 8.0
}

/// M2b expected bytes per slot: code, level byte, upper offset, layer-0
/// links (`4·m0`), expected upper links (`4·m·E[level]`, where
/// `E[level] = 1/(m-1)` for the `1/ln m` level multiplier) and the
/// tombstone bit. The persisted payload is the same minus the offset.
pub fn hnsw_bytes_per_vector(dim: usize, p: &HnswParams) -> f64 {
    let m = f64::from(p.m.max(2));
    dim as f64 + 1.0 + 4.0 + 4.0 * f64::from(p.m0) + 4.0 * m / (m - 1.0) + 1.0 / 8.0
}

/// Expected resident bytes for `slots` HNSW slots (`node_count()`, not
/// `len()`), plus quantizer params.
pub fn hnsw_estimate_bytes(slots: usize, dim: usize, p: &HnswParams) -> usize {
    (slots as f64 * hnsw_bytes_per_vector(dim, p)) as usize + 8 * dim
}

/// Expected resident bytes for `slots` flat slots (`slots()`, not
/// `len()`), plus quantizer params.
pub fn flat_estimate_bytes(slots: usize, dim: usize) -> usize {
    (slots as f64 * flat_bytes_per_vector(dim)) as usize + 8 * dim
}

/// Largest vector count whose resident cost fits `budget_bytes`.
pub fn max_vectors_for_budget(budget_bytes: usize, bytes_per_vector: f64) -> usize {
    if bytes_per_vector <= 0.0 {
        return 0;
    }
    (budget_bytes as f64 / bytes_per_vector) as usize
}

/// Largest M2b node count whose chunk decode fits `budget_ms` at a
/// measured `ms_per_mib` (e.g. [`DECODE_MS_PER_MIB_WASM`]); the store's
/// M2b shard cap is the minimum of this and the memory-derived cap, with
/// headroom left for the SQLite read and the ≤ 200-op replay.
pub fn max_nodes_for_load_budget(
    budget_ms: f64,
    ms_per_mib: f64,
    dim: usize,
    p: &HnswParams,
) -> usize {
    if ms_per_mib <= 0.0 || budget_ms <= 0.0 {
        return 0;
    }
    let payload_per_node = hnsw_bytes_per_vector(dim, p) - 4.0;
    let bytes = budget_ms / ms_per_mib * (1u64 << 20) as f64;
    (bytes / payload_per_node) as usize
}

/// How many shards of `n` vectors at `bytes_per_vector` fit one isolate.
pub fn shards_per_isolate(n: usize, bytes_per_vector: f64) -> usize {
    let per = n as f64 * bytes_per_vector;
    if per <= 0.0 {
        return usize::MAX;
    }
    (ISOLATE_BUDGET_BYTES as f64 / per) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adr_shapes() {
        let p = HnswParams::default();
        // 384-d, m0 = 32: ≈ 521 B/node → 100k ≈ 52 MB (ADR "≈ 51 MB").
        let b = hnsw_bytes_per_vector(384, &p);
        assert!((515.0..530.0).contains(&b), "{b}");
        // M1 per-shard cap 3M floats = 7812 × 384: M2a codes fit many times.
        assert!(shards_per_isolate(7812, flat_bytes_per_vector(384)) >= 3);
        assert!(max_vectors_for_budget(ISOLATE_BUDGET_BYTES / 2, b) >= 50_000);
        assert_eq!(max_vectors_for_budget(10, 0.0), 0);
        // 1 s at 10 ms/MiB ≈ 100 MiB of payload ≈ 200k nodes at 384-d.
        let n = max_nodes_for_load_budget(LOAD_BUDGET_MS, 10.0, 384, &p);
        assert!((190_000..210_000).contains(&n), "{n}");
        assert_eq!(max_nodes_for_load_budget(1000.0, 0.0, 384, &p), 0);
    }
}
