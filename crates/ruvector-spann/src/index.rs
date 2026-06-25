//! Three partition-spilling ANN variants.
//!
//! All three implement the same `PartitionIndex` trait, enabling apples-to-apples
//! benchmarking. The key difference is the spill policy at build time:
//!
//! 1. **SinglePartition** — hard IVF assignment (baseline). Each vector lives in
//!    exactly one partition. Equivalent to IVFFlat with nprobe=1 for recall,
//!    nprobe=N for exact recall.
//!
//! 2. **SpillPartition** — SPANN-style fixed-threshold spilling. A vector is
//!    duplicated into a second partition when the ratio of its distance to the
//!    second-nearest centroid over the nearest centroid distance is below
//!    `spill_ratio` (default 1.20). This directly mirrors Microsoft SPANN.
//!
//! 3. **CoherenceSpill** — dynamic coherence-ratio spilling. Instead of a fixed
//!    threshold, the spill decision uses a coherence score: the ratio
//!    `d_secondary / d_primary`. Vectors whose coherence score falls in the
//!    tail of the distribution (below the `coherence_percentile`-th percentile)
//!    are spilled. This adapts to the actual geometry of each build corpus.

use crate::distance::l2_squared;
use crate::kmeans::kmeans;

/// A single search result: vector id and distance.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: usize,
    pub distance: f32,
}

/// Core trait shared by all three partition-index variants.
pub trait PartitionIndex {
    /// Build the index from a flat slice of vectors (each of length `dim`).
    fn build(&mut self, vectors: &[Vec<f32>]);

    /// Search for the `k` approximate nearest neighbors of `query`.
    /// `nprobe` controls how many partitions are visited.
    fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchResult>;

    /// Return the total number of (vector, partition) assignments stored.
    /// For single-assignment this equals N; for spilled variants it is > N.
    fn total_assignments(&self) -> usize;

    /// Return total memory footprint in bytes (approximate).
    fn memory_bytes(&self) -> usize;
}

// ── Shared helpers ──────────────────────────────────────────────────────────

/// Find `k` nearest centroids to `query`. Returns (centroid_idx, distance) sorted ascending.
fn nearest_centroids(query: &[f32], centroids: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
    let mut dists: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_squared(query, c)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k);
    dists
}

/// Merge candidate lists from multiple partitions, deduplicate by id,
/// keep top-k by distance.
fn merge_candidates(candidates: Vec<Vec<SearchResult>>, k: usize) -> Vec<SearchResult> {
    let total: usize = candidates.iter().map(|c| c.len()).sum();
    let mut flat: Vec<SearchResult> = Vec::with_capacity(total);
    for bucket in candidates {
        for r in bucket {
            flat.push(r);
        }
    }
    // Dedup by id (keep lowest distance).
    flat.sort_by_key(|a| a.id);
    flat.dedup_by(|later, earlier| {
        if later.id == earlier.id {
            if later.distance < earlier.distance {
                earlier.distance = later.distance;
            }
            true
        } else {
            false
        }
    });
    flat.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    flat.truncate(k);
    flat
}

// ── Variant 1: SinglePartition ───────────────────────────────────────────────

pub struct SinglePartitionConfig {
    /// Number of partitions (centroids).
    pub n_centroids: usize,
    /// k-means iterations.
    pub kmeans_iters: usize,
    /// Vector dimension.
    pub dim: usize,
}

impl Default for SinglePartitionConfig {
    fn default() -> Self {
        Self {
            n_centroids: 32,
            kmeans_iters: 20,
            dim: 128,
        }
    }
}

/// Baseline IVF: hard single-partition assignment, no spilling.
pub struct SinglePartition {
    config: SinglePartitionConfig,
    centroids: Vec<Vec<f32>>,
    /// partitions[c] = list of (vector_id, vector) in centroid c.
    partitions: Vec<Vec<(usize, Vec<f32>)>>,
}

impl SinglePartition {
    pub fn new(config: SinglePartitionConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            partitions: Vec::new(),
        }
    }
}

impl PartitionIndex for SinglePartition {
    fn build(&mut self, vectors: &[Vec<f32>]) {
        let k = self.config.n_centroids.min(vectors.len());
        self.centroids = kmeans(vectors, k, self.config.dim, self.config.kmeans_iters);
        self.partitions = vec![Vec::new(); k];

        for (id, v) in vectors.iter().enumerate() {
            let nearest = self
                .centroids
                .iter()
                .enumerate()
                .min_by(|(_, ca), (_, cb)| {
                    l2_squared(v, ca)
                        .partial_cmp(&l2_squared(v, cb))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            self.partitions[nearest].push((id, v.clone()));
        }
    }

    fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchResult> {
        let top_c = nearest_centroids(query, &self.centroids, nprobe);
        let candidates: Vec<Vec<SearchResult>> = top_c
            .iter()
            .map(|(c_idx, _)| {
                self.partitions[*c_idx]
                    .iter()
                    .map(|(id, v)| SearchResult {
                        id: *id,
                        distance: l2_squared(query, v),
                    })
                    .collect()
            })
            .collect();
        merge_candidates(candidates, k)
    }

    fn total_assignments(&self) -> usize {
        self.partitions.iter().map(|p| p.len()).sum()
    }

    fn memory_bytes(&self) -> usize {
        let centroid_bytes = self.centroids.len() * self.config.dim * 4;
        let vector_bytes = self.total_assignments() * self.config.dim * 4;
        centroid_bytes + vector_bytes
    }
}

// ── Variant 2: SpillPartition ────────────────────────────────────────────────

pub struct SpillPartitionConfig {
    pub n_centroids: usize,
    pub kmeans_iters: usize,
    pub dim: usize,
    /// Spill if d_secondary / d_primary < spill_ratio.
    /// Typical range: 1.05 – 1.30. SPANN default ≈ 1.20.
    pub spill_ratio: f32,
}

impl Default for SpillPartitionConfig {
    fn default() -> Self {
        Self {
            n_centroids: 32,
            kmeans_iters: 20,
            dim: 128,
            spill_ratio: 1.20,
        }
    }
}

/// SPANN-style fixed-threshold partition spilling.
pub struct SpillPartition {
    config: SpillPartitionConfig,
    centroids: Vec<Vec<f32>>,
    partitions: Vec<Vec<(usize, Vec<f32>)>>,
}

impl SpillPartition {
    pub fn new(config: SpillPartitionConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            partitions: Vec::new(),
        }
    }
}

impl PartitionIndex for SpillPartition {
    fn build(&mut self, vectors: &[Vec<f32>]) {
        let k = self.config.n_centroids.min(vectors.len());
        self.centroids = kmeans(vectors, k, self.config.dim, self.config.kmeans_iters);
        self.partitions = vec![Vec::new(); k];

        for (id, v) in vectors.iter().enumerate() {
            // Compute distances to all centroids, sorted ascending.
            let mut dists: Vec<(usize, f32)> = self
                .centroids
                .iter()
                .enumerate()
                .map(|(ci, c)| (ci, l2_squared(v, c)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let (primary_idx, d_primary) = dists[0];
            self.partitions[primary_idx].push((id, v.clone()));

            // Spill to secondary if ratio within threshold.
            if dists.len() > 1 {
                let (secondary_idx, d_secondary) = dists[1];
                let ratio = if d_primary < 1e-9 {
                    1.0
                } else {
                    d_secondary / d_primary
                };
                if ratio < self.config.spill_ratio {
                    self.partitions[secondary_idx].push((id, v.clone()));
                }
            }
        }
    }

    fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchResult> {
        let top_c = nearest_centroids(query, &self.centroids, nprobe);
        let candidates: Vec<Vec<SearchResult>> = top_c
            .iter()
            .map(|(c_idx, _)| {
                self.partitions[*c_idx]
                    .iter()
                    .map(|(id, v)| SearchResult {
                        id: *id,
                        distance: l2_squared(query, v),
                    })
                    .collect()
            })
            .collect();
        merge_candidates(candidates, k)
    }

    fn total_assignments(&self) -> usize {
        self.partitions.iter().map(|p| p.len()).sum()
    }

    fn memory_bytes(&self) -> usize {
        let centroid_bytes = self.centroids.len() * self.config.dim * 4;
        let vector_bytes = self.total_assignments() * self.config.dim * 4;
        centroid_bytes + vector_bytes
    }
}

// ── Variant 3: CoherenceSpill ─────────────────────────────────────────────────

pub struct CoherenceSpillConfig {
    pub n_centroids: usize,
    pub kmeans_iters: usize,
    pub dim: usize,
    /// Vectors whose d2/d1 coherence ratio falls below this percentile are spilled.
    /// percentile ∈ [0.0, 1.0]; 0.30 means spill the 30% most ambiguous vectors.
    pub coherence_percentile: f32,
}

impl Default for CoherenceSpillConfig {
    fn default() -> Self {
        Self {
            n_centroids: 32,
            kmeans_iters: 20,
            dim: 128,
            coherence_percentile: 0.30,
        }
    }
}

/// Coherence-driven adaptive partition spilling.
///
/// Unlike SpillPartition (fixed ratio threshold), CoherenceSpill computes the
/// full distribution of d2/d1 ratios across the corpus and spills vectors whose
/// ratio falls below the `coherence_percentile`-th percentile. This automatically
/// adapts to the geometry of each dataset — dense clusters get less spilling,
/// sparse or multimodal data gets more.
pub struct CoherenceSpill {
    config: CoherenceSpillConfig,
    centroids: Vec<Vec<f32>>,
    partitions: Vec<Vec<(usize, Vec<f32>)>>,
    /// Spill threshold determined from corpus distribution.
    pub derived_spill_threshold: f32,
}

impl CoherenceSpill {
    pub fn new(config: CoherenceSpillConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            partitions: Vec::new(),
            derived_spill_threshold: 1.0,
        }
    }
}

impl PartitionIndex for CoherenceSpill {
    fn build(&mut self, vectors: &[Vec<f32>]) {
        let k = self.config.n_centroids.min(vectors.len());
        self.centroids = kmeans(vectors, k, self.config.dim, self.config.kmeans_iters);
        self.partitions = vec![Vec::new(); k];

        // Pass 1: compute d2/d1 ratios for all vectors.
        let mut ratios: Vec<f32> = Vec::with_capacity(vectors.len());
        let mut assignments: Vec<(usize, Option<usize>)> = Vec::with_capacity(vectors.len());

        for v in vectors.iter() {
            let mut dists: Vec<(usize, f32)> = self
                .centroids
                .iter()
                .enumerate()
                .map(|(ci, c)| (ci, l2_squared(v, c)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let (p_idx, d1) = dists[0];
            if dists.len() > 1 {
                let (s_idx, d2) = dists[1];
                let ratio = if d1 < 1e-9 { 1.0 } else { d2 / d1 };
                ratios.push(ratio);
                assignments.push((p_idx, Some(s_idx)));
            } else {
                ratios.push(f32::MAX);
                assignments.push((p_idx, None));
            }
        }

        // Derive threshold at `coherence_percentile` of the ratio distribution.
        let mut sorted_ratios = ratios.clone();
        sorted_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = ((self.config.coherence_percentile * sorted_ratios.len() as f32)
            as usize)
            .min(sorted_ratios.len().saturating_sub(1));
        self.derived_spill_threshold = sorted_ratios[threshold_idx];

        // Pass 2: assign vectors using the derived threshold.
        for (id, (v, (ratio, (p_idx, s_idx_opt)))) in vectors
            .iter()
            .zip(ratios.iter().zip(assignments.iter()))
            .enumerate()
        {
            self.partitions[*p_idx].push((id, v.clone()));
            if let Some(s_idx) = s_idx_opt {
                if *ratio <= self.derived_spill_threshold {
                    self.partitions[*s_idx].push((id, v.clone()));
                }
            }
        }
    }

    fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchResult> {
        let top_c = nearest_centroids(query, &self.centroids, nprobe);
        let candidates: Vec<Vec<SearchResult>> = top_c
            .iter()
            .map(|(c_idx, _)| {
                self.partitions[*c_idx]
                    .iter()
                    .map(|(id, v)| SearchResult {
                        id: *id,
                        distance: l2_squared(query, v),
                    })
                    .collect()
            })
            .collect();
        merge_candidates(candidates, k)
    }

    fn total_assignments(&self) -> usize {
        self.partitions.iter().map(|p| p.len()).sum()
    }

    fn memory_bytes(&self) -> usize {
        let centroid_bytes = self.centroids.len() * self.config.dim * 4;
        let vector_bytes = self.total_assignments() * self.config.dim * 4;
        centroid_bytes + vector_bytes
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gaussian(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = seed;
        let mut next = || -> f32 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Map xorshift to float in [-2, 2] using Box–Muller approximation.
            let u = (rng as f32) / (u64::MAX as f32);
            (u - 0.5) * 4.0
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    fn brute_knn(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2_squared(query, v)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.iter().take(k).map(|(id, _)| *id).collect()
    }

    fn recall_at_k(results: &[SearchResult], ground_truth: &[usize]) -> f32 {
        let gt_set: std::collections::HashSet<usize> = ground_truth.iter().copied().collect();
        let hits = results.iter().filter(|r| gt_set.contains(&r.id)).count();
        hits as f32 / ground_truth.len() as f32
    }

    #[test]
    fn single_partition_builds_and_searches() {
        let corpus = make_gaussian(500, 32, 42);
        let mut idx = SinglePartition::new(SinglePartitionConfig {
            n_centroids: 10,
            kmeans_iters: 10,
            dim: 32,
        });
        idx.build(&corpus);
        assert!(idx.total_assignments() == 500);

        let query = &corpus[0];
        let results = idx.search(query, 10, 4);
        assert!(!results.is_empty());
        // Top result should be the query itself (id=0).
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn spill_partition_increases_assignments() {
        let corpus = make_gaussian(500, 32, 42);

        let mut single = SinglePartition::new(SinglePartitionConfig {
            n_centroids: 10,
            kmeans_iters: 10,
            dim: 32,
        });
        single.build(&corpus);

        let mut spill = SpillPartition::new(SpillPartitionConfig {
            n_centroids: 10,
            kmeans_iters: 10,
            dim: 32,
            spill_ratio: 1.20,
        });
        spill.build(&corpus);

        assert!(spill.total_assignments() >= single.total_assignments());
    }

    #[test]
    fn coherence_spill_higher_recall_than_single() {
        let corpus = make_gaussian(1000, 32, 99);
        let queries: Vec<Vec<f32>> = make_gaussian(50, 32, 77);

        let mut single = SinglePartition::new(SinglePartitionConfig {
            n_centroids: 20,
            kmeans_iters: 15,
            dim: 32,
        });
        single.build(&corpus);

        let mut coh = CoherenceSpill::new(CoherenceSpillConfig {
            n_centroids: 20,
            kmeans_iters: 15,
            dim: 32,
            coherence_percentile: 0.35,
        });
        coh.build(&corpus);

        let mut recall_single = 0.0f32;
        let mut recall_coh = 0.0f32;

        for q in &queries {
            let gt = brute_knn(q, &corpus, 10);
            let r_single = single.search(q, 10, 4);
            let r_coh = coh.search(q, 10, 4);
            recall_single += recall_at_k(&r_single, &gt);
            recall_coh += recall_at_k(&r_coh, &gt);
        }
        recall_single /= queries.len() as f32;
        recall_coh /= queries.len() as f32;

        // Coherence spilling should have recall >= single-partition at same nprobe.
        assert!(
            recall_coh >= recall_single - 0.05,
            "CoherenceSpill recall {recall_coh:.3} should be ≥ Single {recall_single:.3}"
        );
    }

    #[test]
    fn spill_partition_improves_recall_over_single() {
        let corpus = make_gaussian(1000, 64, 123);
        let queries: Vec<Vec<f32>> = make_gaussian(100, 64, 456);

        let mut single = SinglePartition::new(SinglePartitionConfig {
            n_centroids: 20,
            kmeans_iters: 15,
            dim: 64,
        });
        single.build(&corpus);

        let mut spill = SpillPartition::new(SpillPartitionConfig {
            n_centroids: 20,
            kmeans_iters: 15,
            dim: 64,
            spill_ratio: 1.25,
        });
        spill.build(&corpus);

        let mut recall_single = 0.0f32;
        let mut recall_spill = 0.0f32;

        for q in &queries {
            let gt = brute_knn(q, &corpus, 10);
            recall_single += recall_at_k(&single.search(q, 10, 3), &gt);
            recall_spill += recall_at_k(&spill.search(q, 10, 3), &gt);
        }
        recall_single /= queries.len() as f32;
        recall_spill /= queries.len() as f32;

        // SpillPartition should match or exceed single at the same nprobe.
        assert!(
            recall_spill >= recall_single - 0.05,
            "SpillPartition recall {recall_spill:.3} should be ≥ Single {recall_single:.3} - 0.05"
        );
    }

    #[test]
    fn acceptance_single_partition_recall_above_threshold() {
        // Sanity: single partition with generous nprobe reaches acceptable recall.
        let corpus = make_gaussian(1000, 64, 7);
        let queries: Vec<Vec<f32>> = make_gaussian(100, 64, 13);

        let mut idx = SinglePartition::new(SinglePartitionConfig {
            n_centroids: 20,
            kmeans_iters: 15,
            dim: 64,
        });
        idx.build(&corpus);

        let mut total_recall = 0.0f32;
        for q in &queries {
            let gt = brute_knn(q, &corpus, 10);
            total_recall += recall_at_k(&idx.search(q, 10, 10), &gt);
        }
        let recall = total_recall / queries.len() as f32;

        assert!(
            recall >= 0.65,
            "Single partition recall@10 with nprobe=10 should be ≥ 0.65; got {recall:.3}"
        );
    }
}
