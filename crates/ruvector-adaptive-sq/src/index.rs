//! Three scalar-quantization index variants for the adaptive SQ benchmark.
//!
//! All three implement the [`SqIndex`] trait and perform linear-scan search
//! over their stored (quantized) vectors.  The quality difference between
//! variants arises solely from quantization precision, not from the search
//! algorithm.
//!
//! | Variant      | Precision | Memory (N×D)     | Notes                      |
//! |--------------|-----------|------------------|----------------------------|
//! | `Uniform8`   | 8-bit     | N × D bytes      | Baseline                   |
//! | `Uniform16`  | 16-bit    | 2 × N × D bytes  | Upper-bound quality        |
//! | `AdaptiveSq` | mixed     | (hp×2 + lp×1) bytes | Coherence-routed       |

use crate::coherence::{density_scores, precision_threshold};
use crate::quantizer::{
    compute_global_bounds, encode_u16, encode_u8, l2_sq_u16, l2_sq_u8,
};

// ─── Trait ───────────────────────────────────────────────────────────────────

/// Common interface for all SQ index variants.
pub trait SqIndex {
    /// Name used in benchmark output.
    fn name(&self) -> &str;

    /// Search for the `k` approximate nearest neighbours of `query`.
    ///
    /// Returns a list of `(id, estimated_l2_sq)` in ascending distance order.
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)>;

    /// Estimated byte footprint of quantized code storage.
    fn memory_bytes(&self) -> usize;

    /// Fraction of vectors stored at high precision (0.0 for uniform variants).
    fn hp_ratio(&self) -> f32 {
        0.0
    }
}

// ─── Uniform 8-bit ───────────────────────────────────────────────────────────

/// All vectors stored at 8-bit scalar quantization.
pub struct Uniform8Index {
    dim: usize,
    n: usize,
    codes: Vec<u8>, // flat [n × dim]
    mins: Vec<f32>,
    ranges: Vec<f32>,
}

impl Uniform8Index {
    pub fn build(vectors: &[Vec<f32>], dim: usize) -> Self {
        let (mins, ranges) = compute_global_bounds(vectors, dim);
        let n = vectors.len();
        let mut codes = Vec::with_capacity(n * dim);
        for v in vectors {
            for d in 0..dim {
                codes.push(encode_u8(v[d], mins[d], ranges[d]));
            }
        }
        Uniform8Index {
            dim,
            n,
            codes,
            mins,
            ranges,
        }
    }
}

impl SqIndex for Uniform8Index {
    fn name(&self) -> &str {
        "Uniform8"
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut dists: Vec<(usize, f32)> = self
            .codes
            .chunks_exact(self.dim)
            .enumerate()
            .map(|(i, chunk)| {
                let d = l2_sq_u8(chunk, query, &self.mins, &self.ranges);
                (i, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }

    fn memory_bytes(&self) -> usize {
        self.n * self.dim // one byte per element
    }
}

// ─── Uniform 16-bit ──────────────────────────────────────────────────────────

/// All vectors stored at 16-bit scalar quantization.
pub struct Uniform16Index {
    dim: usize,
    n: usize,
    codes: Vec<u16>, // flat [n × dim]
    mins: Vec<f32>,
    ranges: Vec<f32>,
}

impl Uniform16Index {
    pub fn build(vectors: &[Vec<f32>], dim: usize) -> Self {
        let (mins, ranges) = compute_global_bounds(vectors, dim);
        let n = vectors.len();
        let mut codes = Vec::with_capacity(n * dim);
        for v in vectors {
            for d in 0..dim {
                codes.push(encode_u16(v[d], mins[d], ranges[d]));
            }
        }
        Uniform16Index {
            dim,
            n,
            codes,
            mins,
            ranges,
        }
    }
}

impl SqIndex for Uniform16Index {
    fn name(&self) -> &str {
        "Uniform16"
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut dists: Vec<(usize, f32)> = self
            .codes
            .chunks_exact(self.dim)
            .enumerate()
            .map(|(i, chunk)| {
                let d = l2_sq_u16(chunk, query, &self.mins, &self.ranges);
                (i, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }

    fn memory_bytes(&self) -> usize {
        self.n * self.dim * 2 // two bytes per element
    }
}

// ─── Adaptive Mixed-Precision ─────────────────────────────────────────────────

/// Per-vector precision tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tier {
    Low,  // 8-bit
    High, // 16-bit
}

/// Mixed-precision index: dense-region vectors at 16-bit, sparse at 8-bit.
///
/// The routing decision is fixed at `build()` time using local neighbourhood
/// density.  Search reconstructs each vector on-the-fly at its assigned
/// precision.
pub struct AdaptiveSqIndex {
    dim: usize,
    // Routing table: tiers[original_id] = (Tier, local_offset)
    tiers: Vec<(Tier, usize)>,
    // HP storage (16-bit)
    hp_codes: Vec<u16>, // flat [n_hp × dim]
    hp_mins: Vec<f32>,
    hp_ranges: Vec<f32>,
    // LP storage (8-bit)
    lp_codes: Vec<u8>, // flat [n_lp × dim]
    lp_mins: Vec<f32>,
    lp_ranges: Vec<f32>,
    n_hp: usize,
    n_lp: usize,
    total: usize,
}

impl AdaptiveSqIndex {
    /// Build the adaptive index.
    ///
    /// `knn_k`: neighbours used for density scoring (typically 10–20).
    /// `threshold_factor`: multiplied by mean density to set HP/LP boundary.
    pub fn build(vectors: &[Vec<f32>], dim: usize, knn_k: usize, threshold_factor: f32) -> Self {
        let n = vectors.len();

        // 1. Compute density scores for all vectors.
        let scores = density_scores(vectors, knn_k);
        let threshold = precision_threshold(&scores, threshold_factor);

        // 2. Compute global bounds (shared for all vectors, both tiers).
        let (hp_mins, hp_ranges) = compute_global_bounds(vectors, dim);
        let (lp_mins, lp_ranges) = (hp_mins.clone(), hp_ranges.clone());

        // 3. Route each vector and encode.
        let mut tiers = Vec::with_capacity(n);
        let mut hp_codes = Vec::new();
        let mut lp_codes = Vec::new();
        let mut n_hp = 0usize;
        let mut n_lp = 0usize;

        for (i, v) in vectors.iter().enumerate() {
            if scores[i] <= threshold {
                // Dense region → high precision (16-bit).
                let offset = n_hp;
                for d in 0..dim {
                    hp_codes.push(encode_u16(v[d], hp_mins[d], hp_ranges[d]));
                }
                tiers.push((Tier::High, offset));
                n_hp += 1;
            } else {
                // Sparse region → low precision (8-bit).
                let offset = n_lp;
                for d in 0..dim {
                    lp_codes.push(encode_u8(v[d], lp_mins[d], lp_ranges[d]));
                }
                tiers.push((Tier::Low, offset));
                n_lp += 1;
            }
        }

        AdaptiveSqIndex {
            dim,
            tiers,
            hp_codes,
            hp_mins,
            hp_ranges,
            lp_codes,
            lp_mins,
            lp_ranges,
            n_hp,
            n_lp,
            total: n,
        }
    }

    pub fn hp_count(&self) -> usize {
        self.n_hp
    }

    pub fn lp_count(&self) -> usize {
        self.n_lp
    }
}

impl SqIndex for AdaptiveSqIndex {
    fn name(&self) -> &str {
        "AdaptiveSQ"
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut dists: Vec<(usize, f32)> = (0..self.total)
            .map(|orig_id| {
                let (tier, offset) = self.tiers[orig_id];
                let d = match tier {
                    Tier::High => {
                        let start = offset * self.dim;
                        let chunk = &self.hp_codes[start..start + self.dim];
                        l2_sq_u16(chunk, query, &self.hp_mins, &self.hp_ranges)
                    }
                    Tier::Low => {
                        let start = offset * self.dim;
                        let chunk = &self.lp_codes[start..start + self.dim];
                        l2_sq_u8(chunk, query, &self.lp_mins, &self.lp_ranges)
                    }
                };
                (orig_id, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }

    fn memory_bytes(&self) -> usize {
        self.n_hp * self.dim * 2 + self.n_lp * self.dim
    }

    fn hp_ratio(&self) -> f32 {
        self.n_hp as f32 / self.total as f32
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{generate, ground_truth, recall_at_k};

    fn tiny_dataset() -> Vec<Vec<f32>> {
        // 20 vectors, dim=4, two clearly separated clusters
        let mut vecs = Vec::new();
        for i in 0..10 {
            let x = i as f32 * 0.01;
            vecs.push(vec![x, x, x, x]);
        }
        for i in 0..10 {
            let x = 10.0 + i as f32 * 0.01;
            vecs.push(vec![x, x, x, x]);
        }
        vecs
    }

    #[test]
    fn uniform8_finds_self() {
        let vecs = tiny_dataset();
        let idx = Uniform8Index::build(&vecs, 4);
        let q = vecs[0].clone();
        let results = idx.search(&q, 3);
        assert_eq!(results[0].0, 0, "Nearest to vecs[0] must be vecs[0]");
    }

    #[test]
    fn uniform16_finds_self() {
        let vecs = tiny_dataset();
        let idx = Uniform16Index::build(&vecs, 4);
        let q = vecs[5].clone();
        let results = idx.search(&q, 3);
        assert_eq!(results[0].0, 5, "Nearest to vecs[5] must be vecs[5]");
    }

    #[test]
    fn adaptive_finds_self() {
        let vecs = tiny_dataset();
        let idx = AdaptiveSqIndex::build(&vecs, 4, 3, 0.6);
        let q = vecs[0].clone();
        let results = idx.search(&q, 3);
        assert_eq!(results[0].0, 0, "Nearest to vecs[0] must be vecs[0]");
    }

    #[test]
    fn adaptive_memory_between_u8_and_u16() {
        let vecs = tiny_dataset();
        let u8_mem = Uniform8Index::build(&vecs, 4).memory_bytes();
        let u16_mem = Uniform16Index::build(&vecs, 4).memory_bytes();
        let adp_mem = AdaptiveSqIndex::build(&vecs, 4, 3, 0.6).memory_bytes();
        assert!(
            adp_mem >= u8_mem && adp_mem <= u16_mem,
            "Adaptive memory {adp_mem} should be in [{u8_mem}, {u16_mem}]"
        );
    }

    #[test]
    fn adaptive_routes_dense_cluster_to_hp() {
        // Use a dataset with genuine density variation: a very tight knot of 5 vectors
        // surrounded by 15 spread-out vectors.  The knot should score well below the
        // mean density, triggering HP routing.
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        // Tight knot at origin (pairwise distance < 0.01)
        for i in 0..5 {
            let x = i as f32 * 0.001;
            vecs.push(vec![x, x, x, x]);
        }
        // Spread-out vectors far from each other
        for i in 0..15 {
            let x = (i + 1) as f32 * 2.0;
            vecs.push(vec![x, x, x, x]);
        }
        // factor=1.0 routes vectors below the mean to HP; the tight knot qualifies
        let idx = AdaptiveSqIndex::build(&vecs, 4, 3, 1.0);
        assert!(idx.hp_count() > 0, "At least one HP vector expected; got 0");
    }

    #[test]
    fn recall_acceptance_on_clustered_data() {
        // Small N for unit test speed.
        let ds = generate(300, 16, 3, 4, 0.3, 0.02, 0.3, 999);
        let queries: Vec<Vec<f32>> = ds.vectors[..20].to_vec();

        let u16_idx = Uniform16Index::build(&ds.vectors, ds.dim);
        let adp_idx = AdaptiveSqIndex::build(&ds.vectors, ds.dim, 8, 0.6);

        let k = 5;
        let mut u16_recall = 0.0f32;
        let mut adp_recall = 0.0f32;

        for q in &queries {
            let truth = ground_truth(q, &ds.vectors);
            let u16_res = u16_idx.search(q, k);
            let adp_res = adp_idx.search(q, k);
            u16_recall += recall_at_k(&u16_res, &truth, k);
            adp_recall += recall_at_k(&adp_res, &truth, k);
        }
        u16_recall /= queries.len() as f32;
        adp_recall /= queries.len() as f32;

        let threshold = 0.93 * u16_recall;
        assert!(
            adp_recall >= threshold,
            "AdaptiveSQ recall {adp_recall:.4} must be ≥ 0.93 × U16 recall {u16_recall:.4} = {threshold:.4}"
        );
    }

    #[test]
    fn memory_acceptance_adaptive_vs_u16() {
        let ds = generate(300, 16, 3, 4, 0.3, 0.02, 0.3, 999);
        let u16_mem = Uniform16Index::build(&ds.vectors, ds.dim).memory_bytes();
        let adp_mem = AdaptiveSqIndex::build(&ds.vectors, ds.dim, 8, 0.6).memory_bytes();
        let limit = (u16_mem as f32 * 0.75) as usize;
        assert!(
            adp_mem <= limit,
            "AdaptiveSQ memory {adp_mem} B must be ≤ 75 % of U16 memory {u16_mem} B = {limit} B"
        );
    }
}
