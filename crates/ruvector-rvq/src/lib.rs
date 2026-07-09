//! Residual Vector Quantization (RVQ) for compact agent memory.
//!
//! Three variants at equal byte budget:
//!   - [`ScalarQuantizer`]: per-dimension min-max quantization (high bytes, low error)
//!   - [`ProductQuantizer`]: independent sub-space quantization (low bytes, moderate error)
//!   - [`ResidualQuantizer`]: iterative residual refinement (low bytes, lowest error)

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Common interface for vector quantizers.
pub trait VectorQuantizer: Send + Sync {
    /// Train the quantizer on a set of vectors.
    fn train(&mut self, vectors: &[Vec<f32>]);
    /// Encode a single vector into a byte sequence.
    fn encode(&self, v: &[f32]) -> Vec<u8>;
    /// Decode a byte sequence back into an approximate vector.
    fn decode(&self, codes: &[u8]) -> Vec<f32>;
    /// Number of bytes per encoded vector.
    fn bytes_per_vector(&self) -> usize;
    /// Total codebook memory in bytes (f32 weights only).
    fn codebook_bytes(&self) -> usize;
    /// Human-readable name for reporting.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Utility: L2 squared distance, nearest centroid
// ---------------------------------------------------------------------------

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

fn nearest(v: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2_sq(v, c)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .expect("centroids must be non-empty")
}

// ---------------------------------------------------------------------------
// K-means (Lloyd's algorithm, deterministic seed)
// ---------------------------------------------------------------------------

pub fn kmeans(data: &[Vec<f32>], k: usize, iters: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    assert!(!data.is_empty());
    assert!(
        data.len() >= k,
        "need at least k data points to initialise centroids"
    );
    let d = data[0].len();

    // Initialise: random sample k distinct indices
    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.shuffle(rng);
    let mut centroids: Vec<Vec<f32>> = indices[..k].iter().map(|&i| data[i].clone()).collect();

    let mut assignments = vec![0usize; data.len()];

    for _ in 0..iters {
        // Assignment
        let mut changed = false;
        for (vi, v) in data.iter().enumerate() {
            let ni = nearest(v, &centroids);
            if assignments[vi] != ni {
                assignments[vi] = ni;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update: recompute centroid as mean of assigned vectors
        let mut sums = vec![vec![0f32; d]; k];
        let mut counts = vec![0usize; k];
        for (vi, v) in data.iter().enumerate() {
            let c = assignments[vi];
            counts[c] += 1;
            for (j, &x) in v.iter().enumerate() {
                sums[c][j] += x;
            }
        }
        for (ci, centroid) in centroids.iter_mut().enumerate() {
            if counts[ci] > 0 {
                let n = counts[ci] as f32;
                for (j, s) in sums[ci].iter().enumerate() {
                    centroid[j] = s / n;
                }
            }
            // Empty cluster: keep previous centroid (avoids NaN)
        }
    }

    centroids
}

// ---------------------------------------------------------------------------
// Variant 1: ScalarQuantizer (8-bit per dimension)
// ---------------------------------------------------------------------------

/// 8-bit scalar quantizer — full-fidelity baseline, uses D bytes per vector.
pub struct ScalarQuantizer {
    d: usize,
    min_vals: Vec<f32>,
    max_vals: Vec<f32>,
}

impl ScalarQuantizer {
    pub fn new(d: usize) -> Self {
        Self {
            d,
            min_vals: Vec::new(),
            max_vals: Vec::new(),
        }
    }
}

impl VectorQuantizer for ScalarQuantizer {
    fn train(&mut self, vectors: &[Vec<f32>]) {
        self.min_vals = vec![f32::MAX; self.d];
        self.max_vals = vec![f32::MIN; self.d];
        for v in vectors {
            for (i, &x) in v.iter().enumerate() {
                if x < self.min_vals[i] {
                    self.min_vals[i] = x;
                }
                if x > self.max_vals[i] {
                    self.max_vals[i] = x;
                }
            }
        }
    }

    fn encode(&self, v: &[f32]) -> Vec<u8> {
        v.iter()
            .enumerate()
            .map(|(i, &x)| {
                let range = (self.max_vals[i] - self.min_vals[i]).max(1e-9);
                let t = (x - self.min_vals[i]) / range;
                (t.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect()
    }

    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        codes
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let t = c as f32 / 255.0;
                self.min_vals[i] + t * (self.max_vals[i] - self.min_vals[i])
            })
            .collect()
    }

    fn bytes_per_vector(&self) -> usize {
        self.d
    }
    fn codebook_bytes(&self) -> usize {
        self.d * 2 * 4
    }
    fn name(&self) -> &'static str {
        "ScalarQ-8bit"
    }
}

// ---------------------------------------------------------------------------
// Variant 2: ProductQuantizer (single-stage sub-space decomposition)
// ---------------------------------------------------------------------------

/// Product quantizer: splits d-dim vector into M sub-spaces, each independently
/// quantized to K centroids. Byte budget = M (log2(K)=8 bits per code).
pub struct ProductQuantizer {
    m: usize,
    k: usize,
    d_sub: usize,
    codebooks: Vec<Vec<Vec<f32>>>, // [m][k][d_sub]
}

impl ProductQuantizer {
    pub fn new(d: usize, m: usize, k: usize) -> Self {
        assert_eq!(d % m, 0, "d must be divisible by m");
        Self {
            m,
            k,
            d_sub: d / m,
            codebooks: Vec::new(),
        }
    }
}

impl VectorQuantizer for ProductQuantizer {
    fn train(&mut self, vectors: &[Vec<f32>]) {
        let mut rng = StdRng::seed_from_u64(42);
        self.codebooks = (0..self.m)
            .map(|s| {
                let lo = s * self.d_sub;
                let hi = lo + self.d_sub;
                let sub: Vec<Vec<f32>> = vectors.iter().map(|v| v[lo..hi].to_vec()).collect();
                kmeans(&sub, self.k, 15, &mut rng)
            })
            .collect();
    }

    fn encode(&self, v: &[f32]) -> Vec<u8> {
        (0..self.m)
            .map(|s| {
                let lo = s * self.d_sub;
                let hi = lo + self.d_sub;
                nearest(&v[lo..hi], &self.codebooks[s]) as u8
            })
            .collect()
    }

    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.m * self.d_sub);
        for (s, &code) in codes.iter().enumerate() {
            out.extend_from_slice(&self.codebooks[s][code as usize]);
        }
        out
    }

    fn bytes_per_vector(&self) -> usize {
        self.m
    }
    fn codebook_bytes(&self) -> usize {
        self.m * self.k * self.d_sub * 4
    }
    fn name(&self) -> &'static str {
        "ProductQ"
    }
}

// ---------------------------------------------------------------------------
// Variant 3: ResidualQuantizer (multi-stage iterative refinement)
// ---------------------------------------------------------------------------

/// Residual vector quantizer: each stage quantizes the full-dimensional residual
/// from the previous stage. At same byte budget as PQ, achieves lower MSQE
/// because each stage refines the remaining approximation error.
pub struct ResidualQuantizer {
    stages: usize,
    k: usize,
    d: usize,
    codebooks: Vec<Vec<Vec<f32>>>, // [stages][k][d]
}

impl ResidualQuantizer {
    pub fn new(d: usize, stages: usize, k: usize) -> Self {
        Self {
            stages,
            k,
            d,
            codebooks: Vec::new(),
        }
    }
}

impl VectorQuantizer for ResidualQuantizer {
    fn train(&mut self, vectors: &[Vec<f32>]) {
        let mut rng = StdRng::seed_from_u64(42);
        let mut residuals: Vec<Vec<f32>> = vectors.to_vec();
        self.codebooks = Vec::with_capacity(self.stages);

        for _ in 0..self.stages {
            let cb = kmeans(&residuals, self.k, 15, &mut rng);
            // Compute residuals for next stage
            residuals = residuals
                .iter()
                .map(|v| {
                    let idx = nearest(v, &cb);
                    v.iter().zip(cb[idx].iter()).map(|(&a, &b)| a - b).collect()
                })
                .collect();
            self.codebooks.push(cb);
        }
    }

    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.stages);
        let mut residual: Vec<f32> = v.to_vec();
        for cb in &self.codebooks {
            let idx = nearest(&residual, cb);
            codes.push(idx as u8);
            residual = residual
                .iter()
                .zip(cb[idx].iter())
                .map(|(&a, &b)| a - b)
                .collect();
        }
        codes
    }

    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut result = vec![0f32; self.d];
        for (stage, &code) in codes.iter().enumerate() {
            for (r, &c) in result
                .iter_mut()
                .zip(self.codebooks[stage][code as usize].iter())
            {
                *r += c;
            }
        }
        result
    }

    fn bytes_per_vector(&self) -> usize {
        self.stages
    }
    fn codebook_bytes(&self) -> usize {
        self.stages * self.k * self.d * 4
    }
    fn name(&self) -> &'static str {
        "ResidualQ-4"
    }
}

// ---------------------------------------------------------------------------
// Evaluation metrics
// ---------------------------------------------------------------------------

/// Mean squared quantization error averaged over vectors and dimensions.
pub fn msqe(q: &dyn VectorQuantizer, vectors: &[Vec<f32>]) -> f64 {
    let total: f64 = vectors
        .iter()
        .map(|v| {
            let codes = q.encode(v);
            let dec = q.decode(&codes);
            v.iter()
                .zip(dec.iter())
                .map(|(&a, &b)| (a - b).powi(2) as f64)
                .sum::<f64>()
        })
        .sum();
    total / (vectors.len() * vectors[0].len()) as f64
}

/// Recall@k: fraction of true k-NN found by approximate search in decoded space.
///
/// Ground truth is computed in original (uncompressed) space.
/// Approximate search decodes all DB vectors then does brute-force L2.
pub fn recall_at_k(
    q: &dyn VectorQuantizer,
    db: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> f64 {
    let decoded_db: Vec<Vec<f32>> = db
        .iter()
        .map(|v| {
            let codes = q.encode(v);
            q.decode(&codes)
        })
        .collect();

    let mut total_recall = 0.0f64;

    for query in queries {
        let mut exact: Vec<(usize, f32)> = db
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2_sq(query, v)))
            .collect();
        exact.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_set: std::collections::HashSet<usize> =
            exact[..k].iter().map(|&(i, _)| i).collect();

        let mut approx: Vec<(usize, f32)> = decoded_db
            .iter()
            .enumerate()
            .map(|(i, v)| (i, l2_sq(query, v)))
            .collect();
        approx.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let approx_set: std::collections::HashSet<usize> =
            approx[..k].iter().map(|&(i, _)| i).collect();

        total_recall += true_set.intersection(&approx_set).count() as f64 / k as f64;
    }

    total_recall / queries.len() as f64
}

// ---------------------------------------------------------------------------
// Deterministic dataset generation
// ---------------------------------------------------------------------------

/// Generate N isotropic Gaussian vectors of dimension D. IID per-dim; favours PQ.
pub fn generate_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
    (0..n)
        .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
        .collect()
}

/// Generate N clustered vectors with `n_clusters` centers in D-dim space.
///
/// Cluster *centers* always use a fixed internal seed so that quantizers trained on
/// the training split generalise to the test split (both see the same cluster structure).
/// Per-point noise uses the caller's `seed`, so train and test produce different points
/// around the same centers.
///
/// This models real embedding data where semantic clusters span the full vector space.
/// Cross-dimension correlations that PQ's independent sub-space codes cannot resolve
/// are instead captured by RVQ's iterative residual refinement.
///
/// # Parameters
/// - `sigma_cluster` — standard deviation of cluster *center* positions (inter-cluster scale)
/// - `sigma_noise`   — intra-cluster noise per dimension (should be ≪ sigma_cluster)
pub fn generate_clustered_vectors(
    n: usize,
    d: usize,
    n_clusters: usize,
    sigma_cluster: f32,
    sigma_noise: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    // Fixed seed for cluster centers — shared between any call with the same (d, n_clusters, sigma).
    let mut center_rng = StdRng::seed_from_u64(0xDEAD_BEEF_CAFE_1234);
    let center_dist = Normal::new(0.0_f32, sigma_cluster).unwrap();
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            (0..d)
                .map(|_| center_dist.sample(&mut center_rng))
                .collect()
        })
        .collect();

    // Per-point noise varies by caller seed (train vs test independence).
    let mut rng = StdRng::seed_from_u64(seed);
    let noise_dist = Normal::new(0.0_f32, sigma_noise).unwrap();

    (0..n)
        .map(|i| {
            let c = i % n_clusters; // round-robin for balanced cluster coverage
            centers[c]
                .iter()
                .map(|&x| x + noise_dist.sample(&mut rng))
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Clustered data parameters — models real embedding distributions.
    // n_clusters > K_CENTROIDS means centroids must cover multiple clusters,
    // which is where RVQ's iterative refinement outperforms PQ's product code.
    const D: usize = 32;
    const N_TRAIN: usize = 2_000;
    const N_TEST: usize = 400;
    const N_CLUSTERS: usize = 20; // more clusters than K → forces centroid sharing
    const SIGMA_CLUSTER: f32 = 3.0; // large inter-cluster spread
    const SIGMA_NOISE: f32 = 0.15; // tight intra-cluster noise
    const K: usize = 8; // K < N_CLUSTERS → each centroid covers ~2-3 clusters
    const M: usize = 4; // PQ sub-spaces
    const STAGES: usize = 4; // RVQ stages — equal byte budget to PQ

    fn make_clustered() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let train =
            generate_clustered_vectors(N_TRAIN, D, N_CLUSTERS, SIGMA_CLUSTER, SIGMA_NOISE, 1);
        let test =
            generate_clustered_vectors(N_TEST, D, N_CLUSTERS, SIGMA_CLUSTER, SIGMA_NOISE, 99);
        (train, test)
    }

    #[test]
    fn scalar_q_roundtrip_reduces_error() {
        let (train, test) = make_clustered();
        let mut sq = ScalarQuantizer::new(D);
        sq.train(&train);
        let err = msqe(&sq, &test);
        // SQ-8bit over clustered data: quantisation noise only (< intra-cluster variance)
        assert!(err < 0.2, "SQ-8bit MSQE too high: {err:.6}");
    }

    #[test]
    fn product_q_bytes_correct() {
        let (train, _) = make_clustered();
        let mut pq = ProductQuantizer::new(D, M, K);
        pq.train(&train);
        let v = &train[0];
        let codes = pq.encode(v);
        assert_eq!(codes.len(), M);
        let decoded = pq.decode(&codes);
        assert_eq!(decoded.len(), D);
    }

    #[test]
    fn residual_q_bytes_correct() {
        let (train, _) = make_clustered();
        let mut rq = ResidualQuantizer::new(D, STAGES, K);
        rq.train(&train);
        let v = &train[0];
        let codes = rq.encode(v);
        assert_eq!(codes.len(), STAGES);
        let decoded = rq.decode(&codes);
        assert_eq!(decoded.len(), D);
    }

    #[test]
    fn residual_q_lower_msqe_than_product_q() {
        // Core research claim: on clustered data (K < n_clusters), RVQ's iterative
        // residual refinement resolves inter-cluster confusion that PQ's independent
        // sub-space codes leave unresolved.
        let (train, test) = make_clustered();

        let mut pq = ProductQuantizer::new(D, M, K);
        pq.train(&train);
        let pq_err = msqe(&pq, &test);

        let mut rq = ResidualQuantizer::new(D, STAGES, K);
        rq.train(&train);
        let rq_err = msqe(&rq, &test);

        assert_eq!(
            pq.bytes_per_vector(),
            rq.bytes_per_vector(),
            "byte budget must be equal for fair comparison"
        );

        assert!(
            rq_err < pq_err,
            "RVQ MSQE ({rq_err:.6}) must be < PQ MSQE ({pq_err:.6}) on clustered data at same bytes"
        );
    }

    #[test]
    fn residual_q_achieves_reasonable_recall() {
        // With K=8 centroids and N_CLUSTERS=20, recall is constrained by the
        // K/n_clusters ratio (≈0.4): any compressed format with K < n_clusters
        // cannot perfectly separate all clusters.  Target ≥ 0.40 is realistic here;
        // recall scales up with K in production use (see benchmark, K=32).
        let (train, test) = make_clustered();
        let mut rq = ResidualQuantizer::new(D, STAGES, K);
        rq.train(&train);
        let queries =
            generate_clustered_vectors(20, D, N_CLUSTERS, SIGMA_CLUSTER, SIGMA_NOISE, 999);
        let r = recall_at_k(&rq, &test, &queries, 10);
        assert!(r >= 0.40, "RVQ recall@10 too low: {r:.3}");
    }

    #[test]
    fn all_quantizers_have_equal_byte_budget() {
        let (train, _) = make_clustered();
        let mut pq = ProductQuantizer::new(D, M, K);
        pq.train(&train);
        let mut rq = ResidualQuantizer::new(D, STAGES, K);
        rq.train(&train);
        assert_eq!(pq.bytes_per_vector(), M);
        assert_eq!(rq.bytes_per_vector(), STAGES);
        assert_eq!(M, STAGES); // both 4 bytes per vector
    }
}
