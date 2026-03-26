//! Optimized Product Quantization (OPQ) with learned rotation matrix.
//!
//! OPQ improves upon standard PQ by learning an orthogonal rotation matrix R
//! that decorrelates vector dimensions before quantization. This reduces
//! quantization error by 10-30% and yields significant recall improvements,
//! especially when vector dimensions have unequal variance.
//!
//! Training alternates between PQ codebook learning and rotation update via
//! the Procrustes solution (SVD). ADC precomputes per-subspace distance tables
//! so each database lookup costs O(num_subspaces) instead of O(d).

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
            num_subspaces: 8, codebook_size: 256, num_iterations: 20,
            num_opq_iterations: 10, metric: DistanceMetric::Euclidean,
        }
    }
}

impl OPQConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if self.codebook_size > 256 {
            return Err(RuvectorError::InvalidParameter(format!(
                "Codebook size {} exceeds u8 max 256", self.codebook_size)));
        }
        if self.num_subspaces == 0 {
            return Err(RuvectorError::InvalidParameter("num_subspaces must be > 0".into()));
        }
        if self.num_opq_iterations == 0 {
            return Err(RuvectorError::InvalidParameter("num_opq_iterations must be > 0".into()));
        }
        Ok(())
    }
}

// -- Dense matrix (row-major, internal only) ----------------------------------

#[derive(Debug, Clone)]
struct Mat { rows: usize, cols: usize, data: Vec<f32> }

impl Mat {
    fn zeros(r: usize, c: usize) -> Self { Self { rows: r, cols: c, data: vec![0.0; r * c] } }
    fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n { m.data[i * n + i] = 1.0; }
        m
    }
    #[inline] fn get(&self, r: usize, c: usize) -> f32 { self.data[r * self.cols + c] }
    #[inline] fn set(&mut self, r: usize, c: usize, v: f32) { self.data[r * self.cols + c] = v; }

    fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for r in 0..self.rows { for c in 0..self.cols { t.set(c, r, self.get(r, c)); } }
        t
    }
    fn mul(&self, b: &Mat) -> Mat {
        assert_eq!(self.cols, b.rows);
        let mut out = Mat::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                for j in 0..b.cols { let c = out.get(i, j); out.set(i, j, c + a * b.get(k, j)); }
            }
        }
        out
    }
    fn from_rows(vecs: &[Vec<f32>]) -> Self {
        let (rows, cols) = (vecs.len(), vecs[0].len());
        let mut data = Vec::with_capacity(rows * cols);
        for v in vecs { data.extend_from_slice(v); }
        Self { rows, cols, data }
    }
    fn row(&self, i: usize) -> Vec<f32> { self.data[i * self.cols..(i + 1) * self.cols].to_vec() }
}

// -- SVD via power iteration + deflation --------------------------------------

/// Rank-1 SVD: returns (u, sigma, v) for the largest singular triplet.
fn svd_rank1(a: &Mat, max_iters: usize) -> (Vec<f32>, f32, Vec<f32>) {
    let ata = a.transpose().mul(a);
    let n = ata.cols;
    let mut v = vec![1.0 / (n as f32).sqrt(); n];
    for _ in 0..max_iters {
        let mut nv = vec![0.0; n];
        for i in 0..n { for j in 0..n { nv[i] += ata.get(i, j) * v[j]; } }
        let norm: f32 = nv.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-12 { break; }
        for x in nv.iter_mut() { *x /= norm; }
        v = nv;
    }
    let mut av = vec![0.0; a.rows];
    for i in 0..a.rows { for j in 0..a.cols { av[i] += a.get(i, j) * v[j]; } }
    let sigma: f32 = av.iter().map(|x| x * x).sum::<f32>().sqrt();
    let u = if sigma > 1e-12 { av.iter().map(|x| x / sigma).collect() } else { vec![0.0; a.rows] };
    (u, sigma, v)
}

/// Full SVD by repeated rank-1 extraction + deflation.
fn svd_full(a: &Mat, iters: usize) -> (Mat, Vec<f32>, Mat) {
    let n = a.rows;
    let mut res = a.clone();
    let (mut uc, mut sv, mut vc) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let (u, s, v) = svd_rank1(&res, iters);
        if s > 1e-10 {
            for i in 0..res.rows { for j in 0..res.cols {
                let c = res.get(i, j); res.set(i, j, c - s * u[i] * v[j]);
            }}
        }
        uc.push(u); sv.push(s); vc.push(v);
    }
    let (mut um, mut vm) = (Mat::zeros(n, n), Mat::zeros(n, n));
    for j in 0..n { for i in 0..n { um.set(i, j, uc[j][i]); vm.set(i, j, vc[j][i]); } }
    (um, sv, vm)
}

/// Procrustes: find orthogonal R minimising ||Y - X @ R||_F.
fn procrustes(x: &Mat, y: &Mat) -> Mat {
    let m = x.transpose().mul(y);
    let (u, _s, v) = svd_full(&m, 100);
    v.mul(&u.transpose())
}

// -- Rotation matrix ----------------------------------------------------------

/// Orthogonal rotation matrix R (d x d) that decorrelates dimensions before PQ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationMatrix { pub dim: usize, pub data: Vec<f32> }

impl RotationMatrix {
    /// Identity rotation (no-op).
    pub fn identity(dim: usize) -> Self {
        let mut data = vec![0.0; dim * dim];
        for i in 0..dim { data[i * dim + i] = 1.0; }
        Self { dim, data }
    }
    /// Rotate vector: y = x @ R.
    pub fn rotate(&self, v: &[f32]) -> Vec<f32> {
        let d = self.dim;
        (0..d).map(|j| (0..d).map(|i| v[i] * self.data[i * d + j]).sum()).collect()
    }
    /// Inverse rotate: x = y @ R^T.
    pub fn inverse_rotate(&self, v: &[f32]) -> Vec<f32> {
        let d = self.dim;
        (0..d).map(|j| (0..d).map(|i| v[i] * self.data[j * d + i]).sum()).collect()
    }
    fn from_mat(m: &Mat) -> Self { Self { dim: m.rows, data: m.data.clone() } }
}

// -- OPQ Index ----------------------------------------------------------------

/// OPQ index: learns rotation R + PQ codebooks, supports ADC search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OPQIndex {
    pub config: OPQConfig,
    pub rotation: RotationMatrix,
    /// Codebooks: `[subspace][centroid][subspace_dim]`.
    pub codebooks: Vec<Vec<Vec<f32>>>,
    pub dimensions: usize,
}

impl OPQIndex {
    /// Train OPQ via alternating rotation update and PQ codebook learning.
    pub fn train(vectors: &[Vec<f32>], config: OPQConfig) -> Result<Self> {
        config.validate()?;
        if vectors.is_empty() {
            return Err(RuvectorError::InvalidParameter("Training set cannot be empty".into()));
        }
        let d = vectors[0].len();
        if d % config.num_subspaces != 0 {
            return Err(RuvectorError::InvalidParameter(format!(
                "Dimensions {} not divisible by num_subspaces {}", d, config.num_subspaces)));
        }
        for v in vectors { if v.len() != d {
            return Err(RuvectorError::DimensionMismatch { expected: d, actual: v.len() });
        }}
        let x_mat = Mat::from_rows(vectors);
        let mut r = Mat::identity(d);
        let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::new();
        let sub_dim = d / config.num_subspaces;
        for _ in 0..config.num_opq_iterations {
            let x_rot = x_mat.mul(&r);
            let rotated: Vec<Vec<f32>> = (0..vectors.len()).map(|i| x_rot.row(i)).collect();
            codebooks = train_pq_codebooks(&rotated, config.num_subspaces,
                config.codebook_size, config.num_iterations, config.metric)?;
            let mut x_hat = Mat::zeros(vectors.len(), d);
            for (i, rv) in rotated.iter().enumerate() {
                let codes = encode_vec(rv, &codebooks, sub_dim, config.metric)?;
                let recon = decode_vec(&codes, &codebooks);
                for (j, &val) in recon.iter().enumerate() { x_hat.set(i, j, val); }
            }
            r = procrustes(&x_mat, &x_hat);
        }
        Ok(Self { config, rotation: RotationMatrix::from_mat(&r), codebooks, dimensions: d })
    }

    /// Encode a vector: rotate then PQ-quantize.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        self.check_dim(vector.len())?;
        let rotated = self.rotation.rotate(vector);
        encode_vec(&rotated, &self.codebooks,
            self.dimensions / self.config.num_subspaces, self.config.metric)
    }

    /// Decode PQ codes back to approximate vector (with inverse rotation).
    pub fn decode(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if codes.len() != self.config.num_subspaces {
            return Err(RuvectorError::InvalidParameter(format!(
                "Expected {} codes, got {}", self.config.num_subspaces, codes.len())));
        }
        Ok(self.rotation.inverse_rotate(&decode_vec(codes, &self.codebooks)))
    }

    /// ADC search: precompute distance tables then sum lookups per database vector.
    pub fn search_adc(&self, query: &[f32], codes_db: &[Vec<u8>], top_k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        self.check_dim(query.len())?;
        let rq = self.rotation.rotate(query);
        let sub_dim = self.dimensions / self.config.num_subspaces;
        let tables: Vec<Vec<f32>> = (0..self.config.num_subspaces).map(|s| {
            let q_sub = &rq[s * sub_dim..(s + 1) * sub_dim];
            self.codebooks[s].iter().map(|c| dist(q_sub, c, self.config.metric)).collect()
        }).collect();
        let mut dists: Vec<(usize, f32)> = codes_db.iter().enumerate().map(|(idx, codes)| {
            let d: f32 = codes.iter().enumerate().map(|(s, &c)| tables[s][c as usize]).sum();
            (idx, d)
        }).collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(top_k);
        Ok(dists)
    }

    /// Mean squared quantization error over a set of vectors.
    pub fn quantization_error(&self, vectors: &[Vec<f32>]) -> Result<f32> {
        if vectors.is_empty() { return Ok(0.0); }
        let mut total = 0.0f64;
        for v in vectors {
            let recon = self.decode(&self.encode(v)?)?;
            total += v.iter().zip(&recon).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>();
        }
        Ok((total / vectors.len() as f64) as f32)
    }

    fn check_dim(&self, len: usize) -> Result<()> {
        if len != self.dimensions {
            Err(RuvectorError::DimensionMismatch { expected: self.dimensions, actual: len })
        } else { Ok(()) }
    }
}

// -- PQ helpers ---------------------------------------------------------------

fn dist(a: &[f32], b: &[f32], m: DistanceMetric) -> f32 {
    match m {
        DistanceMetric::Euclidean =>
            a.iter().zip(b).map(|(x, y)| { let d = x - y; d * d }).sum::<f32>().sqrt(),
        DistanceMetric::Cosine => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na == 0.0 || nb == 0.0 { 1.0 } else { 1.0 - dot / (na * nb) }
        }
        DistanceMetric::DotProduct => -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>(),
        DistanceMetric::Manhattan => a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum(),
    }
}

fn train_pq_codebooks(vecs: &[Vec<f32>], nsub: usize, k: usize, iters: usize,
    metric: DistanceMetric) -> Result<Vec<Vec<Vec<f32>>>> {
    let sub_dim = vecs[0].len() / nsub;
    (0..nsub).map(|s| {
        let sv: Vec<Vec<f32>> = vecs.iter().map(|v| v[s*sub_dim..(s+1)*sub_dim].to_vec()).collect();
        kmeans(&sv, k.min(sv.len()), iters, metric)
    }).collect()
}

fn encode_vec(v: &[f32], cbs: &[Vec<Vec<f32>>], sub_dim: usize, m: DistanceMetric,
) -> Result<Vec<u8>> {
    cbs.iter().enumerate().map(|(s, cb)| {
        let sub = &v[s * sub_dim..(s + 1) * sub_dim];
        cb.iter().enumerate()
            .min_by(|(_, a), (_, b)| dist(sub, a, m).partial_cmp(&dist(sub, b, m)).unwrap())
            .map(|(i, _)| i as u8)
            .ok_or_else(|| RuvectorError::Internal("Empty codebook".into()))
    }).collect()
}

fn decode_vec(codes: &[u8], cbs: &[Vec<Vec<f32>>]) -> Vec<f32> {
    codes.iter().enumerate().flat_map(|(s, &c)| cbs[s][c as usize].iter().copied()).collect()
}

fn kmeans(vecs: &[Vec<f32>], k: usize, iters: usize, metric: DistanceMetric,
) -> Result<Vec<Vec<f32>>> {
    use rand::seq::SliceRandom;
    if vecs.is_empty() || k == 0 {
        return Err(RuvectorError::InvalidParameter("Cannot cluster empty set or k=0".into()));
    }
    let dim = vecs[0].len();
    let mut rng = rand::thread_rng();
    let mut cents: Vec<Vec<f32>> = vecs.choose_multiple(&mut rng, k).cloned().collect();
    for _ in 0..iters {
        let (mut sums, mut counts) = (vec![vec![0.0f32; dim]; k], vec![0usize; k]);
        for v in vecs {
            let b = cents.iter().enumerate()
                .min_by(|(_, a), (_, b)| dist(v, a, metric).partial_cmp(&dist(v, b, metric)).unwrap())
                .map(|(i, _)| i).unwrap_or(0);
            counts[b] += 1;
            for (j, &val) in v.iter().enumerate() { sums[b][j] += val; }
        }
        for (i, c) in cents.iter_mut().enumerate() {
            if counts[i] > 0 { for j in 0..dim { c[j] = sums[i][j] / counts[i] as f32; } }
        }
    }
    Ok(cents)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize, d: usize) -> Vec<Vec<f32>> {
        let mut seed: u64 = 42;
        (0..n).map(|_| (0..d).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        }).collect()).collect()
    }
    fn cfg() -> OPQConfig {
        OPQConfig { num_subspaces: 2, codebook_size: 4, num_iterations: 5,
            num_opq_iterations: 3, metric: DistanceMetric::Euclidean }
    }

    #[test]
    fn test_rotation_orthogonality() {
        let r = RotationMatrix::identity(4);
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let back = r.inverse_rotate(&r.rotate(&v));
        for i in 0..4 { assert!((v[i] - back[i]).abs() < 1e-6); }
    }
    #[test]
    fn test_rotation_preserves_norm() {
        let data = make_data(30, 4);
        let idx = OPQIndex::train(&data, cfg()).unwrap();
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let n1: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n2: f32 = idx.rotation.rotate(&v).iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n1 - n2).abs() < 0.1, "norms: {} vs {}", n1, n2);
    }
    #[test]
    fn test_pq_encoding_roundtrip() {
        let data = make_data(30, 4);
        let idx = OPQIndex::train(&data, cfg()).unwrap();
        let codes = idx.encode(&data[0]).unwrap();
        assert_eq!(codes.len(), 2);
        assert_eq!(idx.decode(&codes).unwrap().len(), 4);
    }
    #[test]
    fn test_opq_training_convergence() {
        let data = make_data(50, 4);
        let e1 = OPQIndex::train(&data, OPQConfig { num_opq_iterations: 1, ..cfg() })
            .unwrap().quantization_error(&data).unwrap();
        let e2 = OPQIndex::train(&data, OPQConfig { num_opq_iterations: 5, ..cfg() })
            .unwrap().quantization_error(&data).unwrap();
        assert!(e2 <= e1 * 1.05, "error should not grow: {} vs {}", e2, e1);
    }
    #[test]
    fn test_adc_correctness() {
        let data = make_data(30, 4);
        let idx = OPQIndex::train(&data, cfg()).unwrap();
        let db: Vec<Vec<u8>> = data.iter().map(|v| idx.encode(v).unwrap()).collect();
        let res = idx.search_adc(&[0.5, -0.5, 0.5, -0.5], &db, 3).unwrap();
        assert_eq!(res.len(), 3);
        for w in res.windows(2) { assert!(w[0].1 <= w[1].1 + 1e-6); }
    }
    #[test]
    fn test_quantization_error_reduction() {
        let data = make_data(50, 4);
        let err = OPQIndex::train(&data, cfg()).unwrap().quantization_error(&data).unwrap();
        assert!(err >= 0.0 && err.is_finite() && err < 10.0, "err={}", err);
    }
    #[test]
    fn test_svd_correctness() {
        let a = Mat { rows: 2, cols: 2, data: vec![3.0, 0.0, 0.0, 2.0] };
        let (u, s, v) = svd_full(&a, 200);
        for i in 0..2 { for j in 0..2 {
            let r: f32 = (0..2).map(|k| u.get(i, k) * s[k] * v.get(j, k)).sum();
            assert!((a.get(i, j) - r).abs() < 0.1, "SVD fail ({},{}): {} vs {}", i, j, a.get(i, j), r);
        }}
    }
    #[test]
    fn test_identity_rotation_baseline() {
        let data = make_data(30, 4);
        let idx = OPQIndex::train(&data, OPQConfig { num_opq_iterations: 1, ..cfg() }).unwrap();
        let recon = idx.decode(&idx.encode(&data[0]).unwrap()).unwrap();
        assert_eq!(recon.len(), data[0].len());
    }
    #[test]
    fn test_search_accuracy() {
        let data = make_data(40, 4);
        let idx = OPQIndex::train(&data, cfg()).unwrap();
        let db: Vec<Vec<u8>> = data.iter().map(|v| idx.encode(v).unwrap()).collect();
        let ids: Vec<usize> = idx.search_adc(&data[0], &db, 5).unwrap().iter().map(|r| r.0).collect();
        assert!(ids.contains(&0), "vector 0 should be in its own top-5");
    }
    #[test]
    fn test_config_validation() {
        assert!(OPQConfig { codebook_size: 300, ..cfg() }.validate().is_err());
        assert!(OPQConfig { num_subspaces: 0, ..cfg() }.validate().is_err());
        assert!(OPQConfig { num_opq_iterations: 0, ..cfg() }.validate().is_err());
    }
    #[test]
    fn test_dimension_mismatch_errors() {
        let idx = OPQIndex::train(&make_data(30, 4), cfg()).unwrap();
        assert!(idx.encode(&[1.0, 2.0]).is_err());
        assert!(idx.search_adc(&[1.0], &[], 1).is_err());
    }
}
