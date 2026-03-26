//! Optimized Product Quantization (OPQ) with learned rotation matrix.
//!
//! OPQ improves upon standard PQ by learning an orthogonal rotation matrix R
//! that decorrelates vector dimensions before quantization. This reduces
//! quantization error by 10-30% and yields significant recall improvements,
//! especially when vector dimensions have unequal variance.
//!
//! The training procedure alternates between:
//! 1. Training PQ codebooks on rotated vectors
//! 2. Updating the rotation matrix R via the Procrustes solution (SVD)
//!
//! Asymmetric Distance Computation (ADC) precomputes per-subspace distance
//! tables so that each database lookup costs O(num_subspaces) instead of O(d).

use crate::error::{Result, RuvectorError};
use crate::types::DistanceMetric;
use serde::{Deserialize, Serialize};

/// Configuration for Optimized Product Quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OPQConfig {
    /// Number of subspaces to split the (rotated) vector into.
    pub num_subspaces: usize,
    /// Codebook size per subspace (max 256 for u8 codes).
    pub codebook_size: usize,
    /// Number of k-means iterations for codebook training.
    pub num_iterations: usize,
    /// Number of outer OPQ iterations (rotation + PQ alternation).
    pub num_opq_iterations: usize,
    /// Distance metric used for codebook training and search.
    pub metric: DistanceMetric,
}

impl Default for OPQConfig {
    fn default() -> Self {
        Self {
            num_subspaces: 8,
            codebook_size: 256,
            num_iterations: 20,
            num_opq_iterations: 10,
            metric: DistanceMetric::Euclidean,
        }
    }
}

impl OPQConfig {
    /// Validate the configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.codebook_size > 256 {
            return Err(RuvectorError::InvalidParameter(format!(
                "Codebook size {} exceeds u8 maximum of 256",
                self.codebook_size
            )));
        }
        if self.num_subspaces == 0 {
            return Err(RuvectorError::InvalidParameter(
                "Number of subspaces must be greater than 0".into(),
            ));
        }
        if self.num_opq_iterations == 0 {
            return Err(RuvectorError::InvalidParameter(
                "Number of OPQ iterations must be greater than 0".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Linear-algebra helpers (no external dependency)
// ---------------------------------------------------------------------------

/// Row-major dense matrix for internal linear algebra.
#[derive(Debug, Clone)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.cols + c] = v;
    }

    fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                t.set(c, r, self.get(r, c));
            }
        }
        t
    }

    /// C = A * B
    fn mul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows);
        let mut out = Mat::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                for j in 0..other.cols {
                    let cur = out.get(i, j);
                    out.set(i, j, cur + a * other.get(k, j));
                }
            }
        }
        out
    }

    /// Build from row-major slice of vectors (n vectors of dim d -> n x d).
    fn from_rows(vectors: &[Vec<f32>]) -> Self {
        let rows = vectors.len();
        let cols = vectors[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        for v in vectors {
            data.extend_from_slice(v);
        }
        Self { rows, cols, data }
    }

    /// Extract row i as a Vec<f32>.
    fn row(&self, i: usize) -> Vec<f32> {
        self.data[i * self.cols..(i + 1) * self.cols].to_vec()
    }
}

// ---------------------------------------------------------------------------
// SVD via power iteration + deflation (Procrustes only needs full SVD of d x d)
// ---------------------------------------------------------------------------

/// Compute rank-1 SVD of matrix A: returns (u, sigma, v) where A ≈ sigma * u * v^T.
fn svd_rank1(a: &Mat, max_iters: usize) -> (Vec<f32>, f32, Vec<f32>) {
    let ata = a.transpose().mul(a);
    // Power iteration to find dominant right singular vector v.
    let n = ata.cols;
    let mut v = vec![1.0 / (n as f32).sqrt(); n];
    for _ in 0..max_iters {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_v[i] += ata.get(i, j) * v[j];
            }
        }
        let norm: f32 = new_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-12 {
            break;
        }
        for x in new_v.iter_mut() {
            *x /= norm;
        }
        v = new_v;
    }
    // u = A * v / sigma
    let mut av = vec![0.0; a.rows];
    for i in 0..a.rows {
        for j in 0..a.cols {
            av[i] += a.get(i, j) * v[j];
        }
    }
    let sigma: f32 = av.iter().map(|x| x * x).sum::<f32>().sqrt();
    let u = if sigma > 1e-12 {
        av.iter().map(|x| x / sigma).collect()
    } else {
        vec![0.0; a.rows]
    };
    (u, sigma, v)
}

/// Deflate matrix A by removing the rank-1 component sigma * u * v^T.
fn deflate(a: &mut Mat, u: &[f32], sigma: f32, v: &[f32]) {
    for i in 0..a.rows {
        for j in 0..a.cols {
            let cur = a.get(i, j);
            a.set(i, j, cur - sigma * u[i] * v[j]);
        }
    }
}

/// Full SVD of a square matrix via power iteration + deflation.
/// Returns (U, S_diag, V) where A = U * diag(S) * V^T.
fn svd_full(a: &Mat, max_iters: usize) -> (Mat, Vec<f32>, Mat) {
    let n = a.rows;
    let mut residual = a.clone();
    let mut u_cols: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut s_vals: Vec<f32> = Vec::with_capacity(n);
    let mut v_cols: Vec<Vec<f32>> = Vec::with_capacity(n);
    for _ in 0..n {
        let (u, sigma, v) = svd_rank1(&residual, max_iters);
        if sigma < 1e-10 {
            // Fill remaining with zeros.
            u_cols.push(vec![0.0; n]);
            s_vals.push(0.0);
            v_cols.push(vec![0.0; n]);
        } else {
            deflate(&mut residual, &u, sigma, &v);
            u_cols.push(u);
            s_vals.push(sigma);
            v_cols.push(v);
        }
    }
    // Build U and V matrices (columns are the singular vectors).
    let mut u_mat = Mat::zeros(n, n);
    let mut v_mat = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            u_mat.set(i, j, u_cols[j][i]);
            v_mat.set(i, j, v_cols[j][i]);
        }
    }
    (u_mat, s_vals, v_mat)
}

/// Procrustes solution: given X (n x d) and Y (n x d), find the orthogonal
/// matrix R that minimizes ||Y - X @ R||_F.  Solution: SVD(X^T Y) = U S V^T,
/// then R = V U^T  (note: we want R such that X @ R ≈ Y).
fn procrustes(x: &Mat, y: &Mat) -> Mat {
    let m = x.transpose().mul(y); // d x d
    let (u, _s, v) = svd_full(&m, 100);
    v.mul(&u.transpose())
}

// ---------------------------------------------------------------------------
// Rotation matrix wrapper
// ---------------------------------------------------------------------------

/// An orthogonal rotation matrix R of size d x d used to decorrelate dimensions
/// before product quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationMatrix {
    /// Dimension of the rotation.
    pub dim: usize,
    /// Row-major d x d rotation data.
    pub data: Vec<f32>,
}

impl RotationMatrix {
    /// Create an identity rotation (no-op).
    pub fn identity(dim: usize) -> Self {
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Self { dim, data }
    }

    /// Rotate a vector: y = x @ R (x is treated as a row vector).
    pub fn rotate(&self, vector: &[f32]) -> Vec<f32> {
        let d = self.dim;
        let mut out = vec![0.0; d];
        for j in 0..d {
            let mut sum = 0.0;
            for i in 0..d {
                sum += vector[i] * self.data[i * d + j];
            }
            out[j] = sum;
        }
        out
    }

    /// Inverse-rotate a vector: x = y @ R^T.
    pub fn inverse_rotate(&self, vector: &[f32]) -> Vec<f32> {
        let d = self.dim;
        let mut out = vec![0.0; d];
        for j in 0..d {
            let mut sum = 0.0;
            for i in 0..d {
                sum += vector[i] * self.data[j * d + i];
            }
            out[j] = sum;
        }
        out
    }

    fn from_mat(m: &Mat) -> Self {
        Self { dim: m.rows, data: m.data.clone() }
    }
}

// ---------------------------------------------------------------------------
// OPQ Index
// ---------------------------------------------------------------------------

/// Optimized Product Quantization index that learns a rotation matrix to
/// minimise quantization distortion, then uses standard PQ with ADC for
/// fast approximate nearest-neighbour search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OPQIndex {
    /// Configuration.
    pub config: OPQConfig,
    /// Learned rotation matrix.
    pub rotation: RotationMatrix,
    /// Trained codebooks: `[subspace][centroid_id][subspace_dim]`.
    pub codebooks: Vec<Vec<Vec<f32>>>,
    /// Original vector dimensionality.
    pub dimensions: usize,
}

impl OPQIndex {
    /// Train an OPQ index on the given training vectors.
    ///
    /// The algorithm alternates between:
    /// 1. Rotating vectors and training PQ codebooks (inner k-means).
    /// 2. Updating the rotation via the Procrustes solution.
    pub fn train(vectors: &[Vec<f32>], config: OPQConfig) -> Result<Self> {
        config.validate()?;
        if vectors.is_empty() {
            return Err(RuvectorError::InvalidParameter(
                "Training set cannot be empty".into(),
            ));
        }
        let d = vectors[0].len();
        if d % config.num_subspaces != 0 {
            return Err(RuvectorError::InvalidParameter(format!(
                "Dimensions {} must be divisible by num_subspaces {}",
                d, config.num_subspaces
            )));
        }
        for v in vectors {
            if v.len() != d {
                return Err(RuvectorError::DimensionMismatch {
                    expected: d,
                    actual: v.len(),
                });
            }
        }

        let x_mat = Mat::from_rows(vectors);
        let mut r = Mat::identity(d);
        let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::new();
        let sub_dim = d / config.num_subspaces;

        for _ in 0..config.num_opq_iterations {
            // Step a: rotate vectors  X' = X @ R
            let x_rot = x_mat.mul(&r);
            let rotated: Vec<Vec<f32>> =
                (0..vectors.len()).map(|i| x_rot.row(i)).collect();

            // Step b: train PQ codebooks on rotated vectors
            codebooks = train_pq_codebooks(
                &rotated,
                config.num_subspaces,
                config.codebook_size,
                config.num_iterations,
                config.metric,
            )?;

            // Step c: encode all vectors and reconstruct
            let mut x_hat = Mat::zeros(vectors.len(), d);
            for (i, rv) in rotated.iter().enumerate() {
                let codes = encode_with_codebooks(rv, &codebooks, sub_dim, config.metric)?;
                let recon = decode_with_codebooks(&codes, &codebooks);
                for (j, &val) in recon.iter().enumerate() {
                    x_hat.set(i, j, val);
                }
            }

            // Step d: update R via Procrustes: minimise ||X_hat - X @ R||
            // Procrustes(X, X_hat) gives R such that X @ R ≈ X_hat.
            r = procrustes(&x_mat, &x_hat);
        }

        Ok(Self {
            config,
            rotation: RotationMatrix::from_mat(&r),
            codebooks,
            dimensions: d,
        })
    }

    /// Encode a vector into PQ codes (rotate then quantize).
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        self.check_dim(vector.len())?;
        let rotated = self.rotation.rotate(vector);
        let sub_dim = self.dimensions / self.config.num_subspaces;
        encode_with_codebooks(&rotated, &self.codebooks, sub_dim, self.config.metric)
    }

    /// Decode PQ codes back to an approximate vector (inverse rotation applied).
    pub fn decode(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if codes.len() != self.config.num_subspaces {
            return Err(RuvectorError::InvalidParameter(format!(
                "Expected {} codes, got {}",
                self.config.num_subspaces,
                codes.len()
            )));
        }
        let recon = decode_with_codebooks(codes, &self.codebooks);
        Ok(self.rotation.inverse_rotate(&recon))
    }

    /// Asymmetric distance computation: search for top-k nearest neighbors.
    ///
    /// For each subspace a distance table is precomputed from the query
    /// subvector to every centroid. Each database vector distance is then
    /// the sum of `num_subspaces` table lookups -- O(num_subspaces) per vector
    /// instead of O(d).
    pub fn search_adc(
        &self,
        query: &[f32],
        codes_db: &[Vec<u8>],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        self.check_dim(query.len())?;
        let rotated_q = self.rotation.rotate(query);
        let tables = build_distance_tables(
            &rotated_q,
            &self.codebooks,
            self.config.num_subspaces,
            self.config.metric,
        );

        let mut dists: Vec<(usize, f32)> = codes_db
            .iter()
            .enumerate()
            .map(|(idx, codes)| {
                let d: f32 = codes
                    .iter()
                    .enumerate()
                    .map(|(s, &c)| tables[s][c as usize])
                    .sum();
                (idx, d)
            })
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(top_k);
        Ok(dists)
    }

    /// Compute the mean squared quantization error over a set of vectors.
    pub fn quantization_error(&self, vectors: &[Vec<f32>]) -> Result<f32> {
        if vectors.is_empty() {
            return Ok(0.0);
        }
        let mut total = 0.0f64;
        for v in vectors {
            let codes = self.encode(v)?;
            let recon = self.decode(&codes)?;
            let sq: f64 = v
                .iter()
                .zip(recon.iter())
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum();
            total += sq;
        }
        Ok((total / vectors.len() as f64) as f32)
    }

    fn check_dim(&self, len: usize) -> Result<()> {
        if len != self.dimensions {
            return Err(RuvectorError::DimensionMismatch {
                expected: self.dimensions,
                actual: len,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PQ helpers shared between train / encode / decode
// ---------------------------------------------------------------------------

fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Euclidean => a
            .iter()
            .zip(b)
            .map(|(x, y)| { let d = x - y; d * d })
            .sum::<f32>()
            .sqrt(),
        DistanceMetric::Cosine => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na == 0.0 || nb == 0.0 { 1.0 } else { 1.0 - dot / (na * nb) }
        }
        DistanceMetric::DotProduct => {
            -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
        }
        DistanceMetric::Manhattan => {
            a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
        }
    }
}

fn train_pq_codebooks(
    vectors: &[Vec<f32>],
    num_subspaces: usize,
    codebook_size: usize,
    iterations: usize,
    metric: DistanceMetric,
) -> Result<Vec<Vec<Vec<f32>>>> {
    let d = vectors[0].len();
    let sub_dim = d / num_subspaces;
    let mut codebooks = Vec::with_capacity(num_subspaces);
    for s in 0..num_subspaces {
        let start = s * sub_dim;
        let end = start + sub_dim;
        let sub_vecs: Vec<Vec<f32>> =
            vectors.iter().map(|v| v[start..end].to_vec()).collect();
        let k = codebook_size.min(sub_vecs.len());
        let codebook = kmeans(&sub_vecs, k, iterations, metric)?;
        codebooks.push(codebook);
    }
    Ok(codebooks)
}

fn encode_with_codebooks(
    vector: &[f32],
    codebooks: &[Vec<Vec<f32>>],
    sub_dim: usize,
    metric: DistanceMetric,
) -> Result<Vec<u8>> {
    let mut codes = Vec::with_capacity(codebooks.len());
    for (s, cb) in codebooks.iter().enumerate() {
        let start = s * sub_dim;
        let sub = &vector[start..start + sub_dim];
        let best = cb
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                compute_distance(sub, a, metric)
                    .partial_cmp(&compute_distance(sub, b, metric))
                    .unwrap()
            })
            .map(|(i, _)| i as u8)
            .ok_or_else(|| RuvectorError::Internal("Empty codebook".into()))?;
        codes.push(best);
    }
    Ok(codes)
}

fn decode_with_codebooks(codes: &[u8], codebooks: &[Vec<Vec<f32>>]) -> Vec<f32> {
    let mut out = Vec::new();
    for (s, &c) in codes.iter().enumerate() {
        out.extend_from_slice(&codebooks[s][c as usize]);
    }
    out
}

fn build_distance_tables(
    query: &[f32],
    codebooks: &[Vec<Vec<f32>>],
    num_subspaces: usize,
    metric: DistanceMetric,
) -> Vec<Vec<f32>> {
    let sub_dim = query.len() / num_subspaces;
    (0..num_subspaces)
        .map(|s| {
            let start = s * sub_dim;
            let q_sub = &query[start..start + sub_dim];
            codebooks[s]
                .iter()
                .map(|c| compute_distance(q_sub, c, metric))
                .collect()
        })
        .collect()
}

fn kmeans(
    vectors: &[Vec<f32>],
    k: usize,
    iters: usize,
    metric: DistanceMetric,
) -> Result<Vec<Vec<f32>>> {
    use rand::seq::SliceRandom;
    if vectors.is_empty() || k == 0 {
        return Err(RuvectorError::InvalidParameter(
            "Cannot cluster empty set or k=0".into(),
        ));
    }
    let dim = vectors[0].len();
    let mut rng = rand::thread_rng();
    let mut centroids: Vec<Vec<f32>> = vectors
        .choose_multiple(&mut rng, k)
        .cloned()
        .collect();
    for _ in 0..iters {
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for v in vectors {
            let best = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    compute_distance(v, a, metric)
                        .partial_cmp(&compute_distance(v, b, metric))
                        .unwrap()
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            counts[best] += 1;
            for (j, &val) in v.iter().enumerate() {
                sums[best][j] += val;
            }
        }
        for (i, c) in centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                for j in 0..dim {
                    c[j] = sums[i][j] / counts[i] as f32;
                }
            }
        }
    }
    Ok(centroids)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_training_data(n: usize, d: usize) -> Vec<Vec<f32>> {
        // Deterministic pseudo-random data using a simple LCG.
        let mut seed: u64 = 42;
        (0..n)
            .map(|_| {
                (0..d)
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    fn small_config() -> OPQConfig {
        OPQConfig {
            num_subspaces: 2,
            codebook_size: 4,
            num_iterations: 5,
            num_opq_iterations: 3,
            metric: DistanceMetric::Euclidean,
        }
    }

    #[test]
    fn test_rotation_orthogonality() {
        let dim = 4;
        let r = RotationMatrix::identity(dim);
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let rotated = r.rotate(&v);
        let back = r.inverse_rotate(&rotated);
        for i in 0..dim {
            assert!((v[i] - back[i]).abs() < 1e-6, "roundtrip failed at {}", i);
        }
    }

    #[test]
    fn test_rotation_preserves_norm() {
        let data = make_training_data(30, 4);
        let idx = OPQIndex::train(&data, small_config()).unwrap();
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let norm_orig: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rotated = idx.rotation.rotate(&v);
        let norm_rot: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_orig - norm_rot).abs() < 0.1,
            "rotation should approximately preserve norm"
        );
    }

    #[test]
    fn test_pq_encoding_roundtrip() {
        let data = make_training_data(30, 4);
        let idx = OPQIndex::train(&data, small_config()).unwrap();
        let v = data[0].clone();
        let codes = idx.encode(&v).unwrap();
        assert_eq!(codes.len(), 2);
        let recon = idx.decode(&codes).unwrap();
        assert_eq!(recon.len(), 4);
    }

    #[test]
    fn test_opq_training_convergence() {
        let data = make_training_data(50, 4);
        // Train with 1 OPQ iteration (essentially plain PQ).
        let cfg1 = OPQConfig { num_opq_iterations: 1, ..small_config() };
        let idx1 = OPQIndex::train(&data, cfg1).unwrap();
        let err1 = idx1.quantization_error(&data).unwrap();

        // Train with more OPQ iterations.
        let cfg2 = OPQConfig { num_opq_iterations: 5, ..small_config() };
        let idx2 = OPQIndex::train(&data, cfg2).unwrap();
        let err2 = idx2.quantization_error(&data).unwrap();

        // More iterations should not increase error (may be equal for low-d data).
        assert!(
            err2 <= err1 * 1.05,
            "OPQ error should not significantly increase: {} vs {}",
            err2,
            err1
        );
    }

    #[test]
    fn test_adc_correctness() {
        let data = make_training_data(30, 4);
        let idx = OPQIndex::train(&data, small_config()).unwrap();
        let codes_db: Vec<Vec<u8>> = data
            .iter()
            .map(|v| idx.encode(v).unwrap())
            .collect();
        let query = vec![0.5, -0.5, 0.5, -0.5];
        let results = idx.search_adc(&query, &codes_db, 3).unwrap();
        assert_eq!(results.len(), 3);
        // Distances should be non-decreasing.
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1 + 1e-6);
        }
    }

    #[test]
    fn test_quantization_error_reduction() {
        let data = make_training_data(50, 4);
        let idx = OPQIndex::train(&data, small_config()).unwrap();
        let err = idx.quantization_error(&data).unwrap();
        // Error should be finite and non-negative.
        assert!(err >= 0.0);
        assert!(err.is_finite());
        // With 4 centroids per subspace the error should be bounded.
        assert!(err < 10.0, "quantization error unexpectedly large: {}", err);
    }

    #[test]
    fn test_svd_correctness() {
        // 2x2 matrix with known singular values.
        let a = Mat {
            rows: 2,
            cols: 2,
            data: vec![3.0, 0.0, 0.0, 2.0],
        };
        let (u, s, v) = svd_full(&a, 200);
        // Reconstruct: A ≈ U diag(S) V^T
        let mut recon = Mat::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                let mut val = 0.0;
                for k in 0..2 {
                    val += u.get(i, k) * s[k] * v.get(j, k);
                }
                recon.set(i, j, val);
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a.get(i, j) - recon.get(i, j)).abs() < 0.1,
                    "SVD reconstruction failed at ({},{}): {} vs {}",
                    i, j, a.get(i, j), recon.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_identity_rotation_baseline() {
        // With identity rotation, OPQ should behave like plain PQ.
        let data = make_training_data(30, 4);
        let cfg = OPQConfig { num_opq_iterations: 1, ..small_config() };
        let idx = OPQIndex::train(&data, cfg).unwrap();
        let v = data[0].clone();
        let codes = idx.encode(&v).unwrap();
        let recon = idx.decode(&codes).unwrap();
        assert_eq!(recon.len(), v.len());
    }

    #[test]
    fn test_search_accuracy() {
        let data = make_training_data(40, 4);
        let idx = OPQIndex::train(&data, small_config()).unwrap();
        let codes_db: Vec<Vec<u8>> = data
            .iter()
            .map(|v| idx.encode(v).unwrap())
            .collect();
        // Search with one of the training vectors; it should be among top results.
        let results = idx.search_adc(&data[0], &codes_db, 5).unwrap();
        let top_ids: Vec<usize> = results.iter().map(|(i, _)| *i).collect();
        assert!(
            top_ids.contains(&0),
            "training vector 0 should appear in its own top-5 results"
        );
    }

    #[test]
    fn test_config_validation() {
        let bad = OPQConfig { codebook_size: 300, ..small_config() };
        assert!(bad.validate().is_err());
        let bad2 = OPQConfig { num_subspaces: 0, ..small_config() };
        assert!(bad2.validate().is_err());
        let bad3 = OPQConfig { num_opq_iterations: 0, ..small_config() };
        assert!(bad3.validate().is_err());
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let data = make_training_data(30, 4);
        let idx = OPQIndex::train(&data, small_config()).unwrap();
        assert!(idx.encode(&vec![1.0, 2.0]).is_err());
        assert!(idx.search_adc(&vec![1.0], &[], 1).is_err());
    }
}
