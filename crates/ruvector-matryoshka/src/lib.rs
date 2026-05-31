//! Matryoshka HNSW: dimension-adaptive multi-resolution vector search.
//!
//! Implements three search strategies for datasets that exhibit Matryoshka
//! representation structure (early dimensions carry higher discriminative
//! signal than later dimensions, as produced by MRL-trained models):
//!
//! - [`FullScan`]: brute-force at full dimensions (baseline)
//! - [`CoarseScan`]: brute-force using only the first `coarse_dim` dimensions
//! - [`CascadeSearch`]: coarse filter at `coarse_dim`, then rerank at full
//!   dimensions — the core Matryoshka search strategy
//!
//! Reference: Kusupati et al., "Matryoshka Representation Learning",
//! NeurIPS 2022, arXiv:2205.13147.

use std::collections::HashSet;
use std::fmt;
use std::time::Instant;

// ── Configuration ────────────────────────────────────────────────────────────

/// Parameters governing a Matryoshka search index.
#[derive(Debug, Clone)]
pub struct MatryoshkaConfig {
    /// Full embedding dimension (e.g. 128).
    pub full_dim: usize,
    /// Coarse embedding dimension for first-pass candidate selection (e.g. 32).
    pub coarse_dim: usize,
    /// Number of candidates fetched from coarse search before full reranking.
    pub cascade_candidates: usize,
}

impl MatryoshkaConfig {
    pub fn new(full_dim: usize, coarse_dim: usize, cascade_candidates: usize) -> Self {
        assert!(coarse_dim <= full_dim, "coarse_dim must be ≤ full_dim");
        assert!(
            cascade_candidates > 0,
            "cascade_candidates must be positive"
        );
        Self {
            full_dim,
            coarse_dim,
            cascade_candidates,
        }
    }

    /// Memory required per vector at coarse vs full precision (bytes).
    pub fn memory_ratio(&self) -> f64 {
        self.coarse_dim as f64 / self.full_dim as f64
    }
}

// ── Vector ───────────────────────────────────────────────────────────────────

/// A stored vector with a logical identifier.
#[derive(Debug, Clone)]
pub struct Vector {
    pub id: usize,
    pub data: Vec<f32>,
}

impl Vector {
    pub fn new(id: usize, data: Vec<f32>) -> Self {
        Self { id, data }
    }

    /// Squared L2 distance using only the first `dim` dimensions.
    #[inline]
    pub fn l2_sq_truncated(&self, query: &[f32], dim: usize) -> f32 {
        let d = dim.min(self.data.len()).min(query.len());
        self.data[..d]
            .iter()
            .zip(&query[..d])
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum()
    }

    /// Squared L2 distance at full precision.
    #[inline]
    pub fn l2_sq(&self, query: &[f32]) -> f32 {
        self.l2_sq_truncated(query, self.data.len())
    }
}

// ── Results ──────────────────────────────────────────────────────────────────

/// A single nearest-neighbour hit.
#[derive(Debug, Clone)]
pub struct Hit {
    pub id: usize,
    pub distance: f32,
}

// ── Trait ────────────────────────────────────────────────────────────────────

/// Common interface for all Matryoshka search variants.
pub trait MatryoshkaIndex {
    fn name(&self) -> &str;
    fn build(&mut self, vectors: &[Vector]);
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;
    /// Heap bytes occupied by stored vectors.
    fn memory_bytes(&self) -> usize;
}

// ── Variant 1: FullScan ──────────────────────────────────────────────────────

/// Brute-force search using all `full_dim` dimensions. Ground-truth baseline.
pub struct FullScan {
    vectors: Vec<Vector>,
}

impl FullScan {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
        }
    }
}

impl Default for FullScan {
    fn default() -> Self {
        Self::new()
    }
}

impl MatryoshkaIndex for FullScan {
    fn name(&self) -> &str {
        "FullScan (D=full)"
    }

    fn build(&mut self, vectors: &[Vector]) {
        self.vectors = vectors.to_vec();
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut heap: Vec<(f32, usize)> = self
            .vectors
            .iter()
            .map(|v| (v.l2_sq(query), v.id))
            .collect();
        heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        heap.into_iter()
            .take(k)
            .map(|(d, id)| Hit { id, distance: d })
            .collect()
    }

    fn memory_bytes(&self) -> usize {
        self.vectors.iter().map(|v| v.data.len() * 4).sum()
    }
}

// ── Variant 2: CoarseScan ───────────────────────────────────────────────────

/// Brute-force search using only the first `coarse_dim` dimensions.
/// Fast but loses recall on higher-dimensional distinctions.
pub struct CoarseScan {
    vectors: Vec<Vector>,
    coarse_dim: usize,
}

impl CoarseScan {
    pub fn new(coarse_dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            coarse_dim,
        }
    }
}

impl MatryoshkaIndex for CoarseScan {
    fn name(&self) -> &str {
        "CoarseScan (D=coarse)"
    }

    fn build(&mut self, vectors: &[Vector]) {
        self.vectors = vectors.to_vec();
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut heap: Vec<(f32, usize)> = self
            .vectors
            .iter()
            .map(|v| (v.l2_sq_truncated(query, self.coarse_dim), v.id))
            .collect();
        heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        heap.into_iter()
            .take(k)
            .map(|(d, id)| Hit { id, distance: d })
            .collect()
    }

    fn memory_bytes(&self) -> usize {
        // Stores full vectors; active compute is coarse_dim only
        self.vectors.iter().map(|v| v.data.len() * 4).sum()
    }
}

// ── Variant 3: CascadeSearch ─────────────────────────────────────────────────

/// Two-pass Matryoshka cascade: coarse candidate selection followed by
/// full-precision reranking.
///
/// Stage 1 — linear scan over all N vectors using only `coarse_dim` dimensions,
///           retaining the top `cascade_candidates` by coarse distance.
///
/// Stage 2 — recompute exact L2 at full precision for the retained candidates,
///           return top-k.
///
/// When data has Matryoshka structure (early dims are most discriminative),
/// Stage 1 eliminates the vast majority of false neighbours cheaply, and
/// Stage 2 recovers high recall without scanning the full corpus at full cost.
pub struct CascadeSearch {
    vectors: Vec<Vector>,
    config: MatryoshkaConfig,
}

impl CascadeSearch {
    pub fn new(config: MatryoshkaConfig) -> Self {
        Self {
            vectors: Vec::new(),
            config,
        }
    }
}

impl MatryoshkaIndex for CascadeSearch {
    fn name(&self) -> &str {
        "CascadeSearch (coarse→full)"
    }

    fn build(&mut self, vectors: &[Vector]) {
        self.vectors = vectors.to_vec();
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let n_candidates = self.config.cascade_candidates.max(k);

        // Stage 1: coarse scan — O(N * coarse_dim) distance ops
        let mut coarse: Vec<(f32, usize)> = self
            .vectors
            .iter()
            .map(|v| (v.l2_sq_truncated(query, self.config.coarse_dim), v.id))
            .collect();
        coarse.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Stage 2: full rerank — O(candidates * full_dim) distance ops
        let mut refined: Vec<(f32, usize)> = coarse
            .into_iter()
            .take(n_candidates)
            .map(|(_, id)| (self.vectors[id].l2_sq(query), id))
            .collect();
        refined.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        refined
            .into_iter()
            .take(k)
            .map(|(d, id)| Hit { id, distance: d })
            .collect()
    }

    fn memory_bytes(&self) -> usize {
        self.vectors.iter().map(|v| v.data.len() * 4).sum()
    }
}

// ── Dataset generator ────────────────────────────────────────────────────────

/// Generate cluster centres for a Matryoshka dataset.
///
/// Centres are spread uniformly in `[-3, 3]^dim`.  The same `seed` must be
/// passed to both `generate_matryoshka_dataset` and `generate_queries` so that
/// queries and database vectors share the same cluster geometry — a requirement
/// for the Matryoshka cascade to be well-defined.
fn make_cluster_centers(n_clusters: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-3.0_f32..3.0)).collect())
        .collect()
}

/// Place `n` points around the provided cluster centres.
///
/// Noise scale increases with dimension index to simulate MRL training:
///
/// - dims `0 .. dim/4`:   σ = 0.12  (high signal — most discriminative)
/// - dims `dim/4 .. dim/2`: σ = 0.50  (medium signal)
/// - dims `dim/2 .. dim`: σ = 0.80  (lower signal, still cluster-structured — not pure noise)
fn place_points(centers: &[Vec<f32>], n: usize, dim: usize, noise_seed: u64) -> Vec<Vector> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(noise_seed);
    (0..n)
        .map(|i| {
            let c = &centers[i % centers.len()];
            let data: Vec<f32> = (0..dim)
                .map(|d| {
                    let sigma: f32 = if d < dim / 4 {
                        0.12
                    } else if d < dim / 2 {
                        0.50
                    } else {
                        0.80
                    };
                    c[d] + rng.gen_range(-sigma..sigma)
                })
                .collect();
            Vector::new(i, data)
        })
        .collect()
}

/// Generate a synthetic database with Matryoshka-like structure.
///
/// `seed` controls cluster geometry; both dataset and queries must share it.
pub fn generate_matryoshka_dataset(
    n: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> Vec<Vector> {
    let centers = make_cluster_centers(n_clusters, dim, seed);
    // Use seed+1 for per-point noise so centres and points don't share the rng stream.
    place_points(&centers, n, dim, seed.wrapping_add(1))
}

/// Generate query vectors over the same cluster centres as the database.
///
/// **`seed` must match the one passed to `generate_matryoshka_dataset`.**
pub fn generate_queries(
    n_queries: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let centers = make_cluster_centers(n_clusters, dim, seed);
    // Use seed+0xBEEF so query noise is independent from database point noise.
    place_points(&centers, n_queries, dim, seed.wrapping_add(0xBEEF))
        .into_iter()
        .map(|v| v.data)
        .collect()
}

// ── Evaluation helpers ───────────────────────────────────────────────────────

/// Recall@k: fraction of the true top-k neighbours found in `retrieved`.
pub fn recall_at_k(ground_truth: &[Hit], retrieved: &[Hit]) -> f64 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let gt_ids: HashSet<usize> = ground_truth.iter().map(|h| h.id).collect();
    let k = ground_truth.len().min(retrieved.len());
    let found = retrieved.iter().filter(|h| gt_ids.contains(&h.id)).count();
    found as f64 / k as f64
}

// ── Benchmark harness ────────────────────────────────────────────────────────

/// Per-query timing and recall collected during a benchmark run.
#[derive(Debug)]
pub struct BenchStats {
    pub mean_latency_us: f64,
    pub p50_latency_us: f64,
    pub p95_latency_us: f64,
    pub throughput_qps: f64,
    pub mean_recall: f64,
    pub memory_kb: usize,
}

impl fmt::Display for BenchStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mean={:.1}µs  p50={:.1}µs  p95={:.1}µs  qps={:.0}  recall={:.4}  mem={}KB",
            self.mean_latency_us,
            self.p50_latency_us,
            self.p95_latency_us,
            self.throughput_qps,
            self.mean_recall,
            self.memory_kb
        )
    }
}

/// Run `queries` against `index`, compare to `ground_truth`, return stats.
pub fn run_benchmark(
    index: &dyn MatryoshkaIndex,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<Hit>],
    k: usize,
) -> BenchStats {
    let mut latencies_us: Vec<f64> = Vec::with_capacity(queries.len());
    let mut recalls: Vec<f64> = Vec::with_capacity(queries.len());

    for (query, gt) in queries.iter().zip(ground_truth.iter()) {
        let t0 = Instant::now();
        let hits = index.search(query, k);
        latencies_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
        recalls.push(recall_at_k(gt, &hits));
    }

    latencies_us.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = latencies_us.len();
    let mean_lat = latencies_us.iter().sum::<f64>() / n as f64;
    let p50 = latencies_us[n / 2];
    let p95 = latencies_us[(n as f64 * 0.95) as usize];
    let total_s: f64 = latencies_us.iter().sum::<f64>() / 1_000_000.0;

    BenchStats {
        mean_latency_us: mean_lat,
        p50_latency_us: p50,
        p95_latency_us: p95,
        throughput_qps: n as f64 / total_s,
        mean_recall: recalls.iter().sum::<f64>() / n as f64,
        memory_kb: index.memory_bytes() / 1024,
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 2_000;
    const DIM: usize = 128;
    const COARSE_DIM: usize = 32;
    const K: usize = 10;
    const N_CLUSTERS: usize = 20;
    const N_QUERIES: usize = 100;
    const CASCADE_CANDS: usize = 150;

    fn build_dataset() -> Vec<Vector> {
        generate_matryoshka_dataset(N, DIM, N_CLUSTERS, 42)
    }

    fn build_queries() -> Vec<Vec<f32>> {
        generate_queries(N_QUERIES, DIM, N_CLUSTERS, 42)
    }

    #[test]
    fn full_scan_returns_k_results() {
        let data = build_dataset();
        let mut idx = FullScan::new();
        idx.build(&data);
        let q = build_queries();
        let hits = idx.search(&q[0], K);
        assert_eq!(hits.len(), K);
    }

    #[test]
    fn coarse_scan_faster_than_full() {
        let data = build_dataset();
        let q = build_queries();

        let mut full = FullScan::new();
        full.build(&data);
        let mut coarse = CoarseScan::new(COARSE_DIM);
        coarse.build(&data);

        let gt = run_benchmark(&full, &q, &vec![vec![]; q.len()], K);
        let cs = run_benchmark(&coarse, &q, &vec![vec![]; q.len()], K);

        // Coarse search must be noticeably faster (≥1.5×)
        assert!(
            cs.throughput_qps >= gt.throughput_qps * 1.5,
            "Expected coarse QPS {:.0} ≥ 1.5× full QPS {:.0}",
            cs.throughput_qps,
            gt.throughput_qps
        );
    }

    #[test]
    fn cascade_recall_above_threshold() {
        let data = build_dataset();
        let q = build_queries();

        let mut full = FullScan::new();
        full.build(&data);

        // Build ground truth
        let gt: Vec<Vec<Hit>> = q.iter().map(|query| full.search(query, K)).collect();

        let cfg = MatryoshkaConfig::new(DIM, COARSE_DIM, CASCADE_CANDS);
        let mut cascade = CascadeSearch::new(cfg);
        cascade.build(&data);

        let stats = run_benchmark(&cascade, &q, &gt, K);

        // Acceptance: ≥90% recall@10 with Matryoshka-structured data
        assert!(
            stats.mean_recall >= 0.90,
            "CascadeSearch recall {:.4} < 0.90 acceptance threshold",
            stats.mean_recall
        );
    }

    #[test]
    fn cascade_faster_than_full() {
        let data = build_dataset();
        let q = build_queries();

        let mut full = FullScan::new();
        full.build(&data);

        let cfg = MatryoshkaConfig::new(DIM, COARSE_DIM, CASCADE_CANDS);
        let mut cascade = CascadeSearch::new(cfg.clone());
        cascade.build(&data);

        let gt_stats = run_benchmark(&full, &q, &vec![vec![]; q.len()], K);
        let ca_stats = run_benchmark(&cascade, &q, &vec![vec![]; q.len()], K);

        // Cascade must be faster than full scan (QPS improvement)
        assert!(
            ca_stats.throughput_qps > gt_stats.throughput_qps,
            "Expected cascade QPS {:.0} > full QPS {:.0}",
            ca_stats.throughput_qps,
            gt_stats.throughput_qps
        );
    }

    #[test]
    fn recall_at_k_perfect_match() {
        let hits: Vec<Hit> = (0..K)
            .map(|i| Hit {
                id: i,
                distance: i as f32,
            })
            .collect();
        assert_eq!(recall_at_k(&hits, &hits), 1.0);
    }

    #[test]
    fn recall_at_k_no_match() {
        let gt: Vec<Hit> = (0..K)
            .map(|i| Hit {
                id: i,
                distance: 0.0,
            })
            .collect();
        let retrieved: Vec<Hit> = (K..2 * K)
            .map(|i| Hit {
                id: i,
                distance: 0.0,
            })
            .collect();
        assert_eq!(recall_at_k(&gt, &retrieved), 0.0);
    }

    #[test]
    fn matryoshka_config_memory_ratio() {
        let cfg = MatryoshkaConfig::new(128, 32, 200);
        let ratio = cfg.memory_ratio();
        assert!((ratio - 0.25).abs() < 1e-6, "ratio should be 0.25");
    }

    #[test]
    fn dataset_correct_size_and_dim() {
        let data = generate_matryoshka_dataset(500, 64, 10, 99);
        assert_eq!(data.len(), 500);
        assert!(data.iter().all(|v| v.data.len() == 64));
    }
}
