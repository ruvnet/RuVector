//! Tiered agent memory with coherence-driven hot/warm/cold tier promotion.
//!
//! Three variants are provided:
//! - `FlatMemory`: linear scan baseline, no tiering.
//! - `LruTieredMemory`: access-frequency tiering using LRU eviction.
//! - `CoherenceTieredMemory`: coherence-score tiering using running query centroid.

pub mod coherence_tiered;
pub mod flat;
pub mod lru_tiered;

/// Placement of a vector within the tiered store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Hot,
    Warm,
    Cold,
}

/// A single search result with its id, distance, and the tier it came from.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
    pub tier: Tier,
}

/// Per-tier occupancy and memory estimates.
#[derive(Debug, Clone)]
pub struct TierStats {
    pub hot_count: usize,
    pub warm_count: usize,
    pub cold_count: usize,
    pub hot_bytes: usize,
    pub warm_bytes: usize,
    pub cold_bytes: usize,
}

impl TierStats {
    pub fn total_bytes(&self) -> usize {
        self.hot_bytes + self.warm_bytes + self.cold_bytes
    }
    pub fn total_vectors(&self) -> usize {
        self.hot_count + self.warm_count + self.cold_count
    }
}

/// Core trait every tiered backend implements.
pub trait TieredMemoryStore {
    /// Insert a vector with the given id.
    fn insert(&mut self, id: u64, vector: Vec<f32>);
    /// Search for the k nearest neighbors of `query`.
    fn search(&mut self, query: &[f32], k: usize) -> Vec<SearchResult>;
    /// Return occupancy and memory stats.
    fn tier_stats(&self) -> TierStats;
    /// Name of this variant for reporting.
    fn name(&self) -> &str;
}

// ── shared math ─────────────────────────────────────────────────────────────

/// Squared L2 distance between two equal-length slices.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Cosine similarity (returns value in [-1, 1]).
#[inline]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

// ── 8-bit scalar quantization used by warm tier ──────────────────────────────

#[derive(Debug, Clone)]
pub struct QuantizedVec {
    pub data: Vec<u8>,
    pub min: f32,
    pub scale: f32,
}

impl QuantizedVec {
    pub fn encode(v: &[f32]) -> Self {
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (max - min).max(1e-9);
        let scale = range / 255.0;
        let data = v
            .iter()
            .map(|&x| ((x - min) / scale).round() as u8)
            .collect();
        QuantizedVec { data, min, scale }
    }

    pub fn decode(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&b| b as f32 * self.scale + self.min)
            .collect()
    }
}

/// Bytes consumed by a full-precision vector.
pub fn fp32_bytes(dims: usize) -> usize {
    dims * 4
}
/// Bytes consumed by an 8-bit quantized vector (data + 2 x f32 header).
pub fn q8_bytes(dims: usize) -> usize {
    dims + 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_zero_for_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!(l2_sq(&v, &v) < 1e-9);
    }

    #[test]
    fn cosine_sim_one_for_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn quantized_roundtrip_within_tolerance() {
        let v: Vec<f32> = (0..128).map(|i| i as f32 * 0.1).collect();
        let q = QuantizedVec::encode(&v);
        let decoded = q.decode();
        let max_err = v
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // scale = range/255 ≈ 12.7/255 ≈ 0.05; max quantization error ≤ 0.5 * scale
        assert!(max_err < 0.06, "max_err={max_err}");
    }
}
