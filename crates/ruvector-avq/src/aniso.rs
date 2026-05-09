//! Anisotropic Vector Quantization (AVQ).
//!
//! The score `<q, x>` for a query `q` and a database point `x` is
//! distorted by the *parallel* component of the residual
//! `r = x - x_tilde` (where `x_tilde` is the quantization of `x`),
//! not by the orthogonal component. AVQ trains the codebooks under
//! the anisotropic loss
//!
//! ```text
//!     L_eta(x, x_tilde) = eta * ||r_par||^2 + ||r_perp||^2,
//! ```
//!
//! with `r_par = (r . d_hat) d_hat` and `d_hat = x / ||x||`. Setting `eta = 1`
//! recovers uniform-MSE PQ. Larger `eta` (typical: 4 to 16)
//! progressively prioritizes preserving inner-product score over
//! reconstruction.
//!
//! Per-subspace decomposition: with `x = (x[1],...,x[m])` and the
//! same split for residual `r[s]`, project onto the subspace
//! component of the *full-vector* unit direction
//! `d_hat[s] = x[s] / ||x||`. The parallel/orthogonal split
//! decomposes additively across subspaces.
//!
//! This implementation does block-coordinate descent: for each
//! subspace we (a) re-assign each training point to the codeword
//! that minimizes the anisotropic loss for that point's *current*
//! direction, then (b) update each codeword by closed-form
//! weighted least squares on the points assigned to it.
//!
//! Reference: Guo et al. ICML 2020.

use crate::error::AvqError;
use crate::pq::ProductQuantizer;
use crate::traits::{Encoder, Scorer};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[derive(Debug, Clone)]
pub struct AnisotropicPq {
    inner: ProductQuantizer,
    pub eta: f32,
}

impl AnisotropicPq {
    /// Train `m` subquantizers of `k` codewords each under the
    /// anisotropic loss with weight `eta`. `eta = 1.0` reduces to
    /// uniform PQ. The ScaNN paper recommends ~4–16 for typical
    /// dense embedding workloads.
    pub fn fit(
        train: &[f32],
        dim: usize,
        m: usize,
        k: usize,
        eta: f32,
        seed: u64,
    ) -> Result<Self, AvqError> {
        if eta < 1.0 {
            return Err(AvqError::BadEta(eta));
        }
        // Warm-start from uniform PQ (eta=1) so we begin near a good
        // local optimum. This matches ScaNN's training recipe.
        let mut pq = ProductQuantizer::fit(train, dim, m, k, seed)?;
        let n = train.len() / dim;
        let ds = pq.ds;

        // Pre-compute squared norms for direction normalization.
        let mut norms2 = vec![0.0f32; n];
        for i in 0..n {
            let row = &train[i * dim..(i + 1) * dim];
            let mut s = 0.0f32;
            for &v in row {
                s += v * v;
            }
            norms2[i] = s.max(1e-30);
        }
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0xA17C));

        // Run several rounds of block-coordinate descent across
        // subspaces. Each round: assign-by-aniso-loss, then close-form
        // weighted update. We track the aniso loss summed across
        // subspaces (using the SAME assignments produced by the aniso
        // assignment step — so the loss must be monotone non-increasing
        // by Lloyd's monotonicity argument).
        for _round in 0..8 {
            for s in 0..m {
                aniso_subspace_step(&mut pq, train, n, dim, ds, s, &norms2, eta, &mut rng);
            }
        }
        Ok(AnisotropicPq { inner: pq, eta })
    }

    pub fn inner(&self) -> &ProductQuantizer {
        &self.inner
    }

    /// Diagnostic: average per-point anisotropic loss on training data.
    /// Useful for validating that training reduces the AVQ objective.
    pub fn aniso_loss(&self, train: &[f32]) -> f64 {
        let dim = self.inner.dim;
        let m = self.inner.m;
        let ds = self.inner.ds;
        let n = train.len() / dim;
        let mut codes = vec![0u8; n * m];
        self.encode(train, &mut codes);
        let mut total = 0.0f64;
        for i in 0..n {
            let row = &train[i * dim..(i + 1) * dim];
            let mut norm2 = 0.0f32;
            for &v in row {
                norm2 += v * v;
            }
            let inv_norm = 1.0 / norm2.max(1e-30).sqrt();
            for s in 0..m {
                let xs = &row[s * ds..(s + 1) * ds];
                let cent = &self.inner.centroids[s]
                    [codes[i * m + s] as usize * ds..(codes[i * m + s] as usize + 1) * ds];
                let mut r2 = 0.0f32;
                let mut r_dot_d = 0.0f32;
                for j in 0..ds {
                    let r = xs[j] - cent[j];
                    r2 += r * r;
                    r_dot_d += r * (xs[j] * inv_norm);
                }
                let par2 = r_dot_d * r_dot_d;
                let perp2 = (r2 - par2).max(0.0);
                total += (self.eta as f64) * par2 as f64 + perp2 as f64;
            }
        }
        total / n as f64
    }

    /// Build the asymmetric inner-product LUT (delegates to the
    /// shared PQ layer — at scoring time we re-use the same machinery).
    pub fn build_lut_ip(&self, query: &[f32]) -> Vec<f32> {
        self.inner.build_lut_ip(query)
    }
}

/// One sweep of anisotropic re-assignment + closed-form update for
/// subspace `s`. The closed-form is derived from setting the gradient
/// of `Sum_i [w_par * ||r_par||^2 + w_perp * ||r_perp||^2]` to zero,
/// yielding `c_s = (Sum A_i x_i)(Sum A_i)^-1` with the per-point
/// `A_i = w_perp I + (w_par - w_perp) d_hat d_hat^T`. We solve the
/// resulting `ds x ds` symmetric positive-definite system per centroid
/// with a tiny in-place Cholesky.
fn aniso_subspace_step(
    pq: &mut ProductQuantizer,
    train: &[f32],
    n: usize,
    dim: usize,
    ds: usize,
    s: usize,
    norms2: &[f32],
    eta: f32,
    rng: &mut StdRng,
) {
    let k = pq.k;
    // gather subspace and per-point unit-direction slice
    let mut assign = vec![0u8; n];

    // Anisotropic assignment: pick c minimizing eta times the parallel
    // residual squared plus the orthogonal residual squared, where the
    // residual is `x_s - centroid_c` and `d_hat` is the unit-direction
    // full-vector slice projected into this subspace.
    for i in 0..n {
        let off = i * dim + s * ds;
        let xs = &train[off..off + ds];
        let inv_norm = 1.0 / norms2[i].sqrt();
        // direction slice (unit), in this subspace
        let mut dh = [0.0f32; 64];
        let dh = &mut dh[..ds];
        for j in 0..ds {
            dh[j] = xs[j] * inv_norm;
        }
        let mut best = 0u8;
        let mut best_l = f32::INFINITY;
        for c in 0..k {
            let cent = &pq.centroids[s][c * ds..(c + 1) * ds];
            // residual r and its dot with d_hat (subspace component)
            let mut r_dot_d = 0.0f32;
            let mut r2 = 0.0f32;
            for j in 0..ds {
                let r = xs[j] - cent[j];
                r_dot_d += r * dh[j];
                r2 += r * r;
            }
            let par2 = r_dot_d * r_dot_d;
            let perp2 = (r2 - par2).max(0.0);
            let l = eta * par2 + perp2;
            if l < best_l {
                best_l = l;
                best = c as u8;
            }
        }
        assign[i] = best;
    }

    // Closed-form weighted update per centroid. For each c we
    // accumulate the per-point normal matrix and rhs, then solve via
    // a small dense Cholesky. With `w_par = eta` and `w_perp = 1`
    // (the standard AVQ choice; see Guo et al. eq. 9 with t=1
    // yielding Eq. 10) the system is symmetric positive-definite.
    let w_par = eta;
    let w_perp = 1.0f32;

    // For each centroid, accumulate small ds×ds normal matrices.
    // ds is small (<= 64 typical), so this is cheap.
    let stride = ds * ds;
    let mut nmat = vec![0.0f64; k * stride];
    let mut bvec = vec![0.0f64; k * ds];
    let mut counts = vec![0u32; k];

    for i in 0..n {
        let c = assign[i] as usize;
        counts[c] += 1;
        let off = i * dim + s * ds;
        let xs = &train[off..off + ds];
        let inv_norm = 1.0 / norms2[i].sqrt();
        let mut dh = [0.0f64; 64];
        let dh = &mut dh[..ds];
        for j in 0..ds {
            dh[j] = (xs[j] * inv_norm) as f64;
        }
        // N += w_perp * I + (w_par - w_perp) * dh dh^T
        let m = &mut nmat[c * stride..(c + 1) * stride];
        for j in 0..ds {
            m[j * ds + j] += w_perp as f64;
            for l in 0..ds {
                m[j * ds + l] += (w_par - w_perp) as f64 * dh[j] * dh[l];
            }
        }
        // b += (w_perp I + (w_par - w_perp) dh dh^T) x_s
        let bs = &mut bvec[c * ds..(c + 1) * ds];
        // compute t = dh^T x_s
        let mut t = 0.0f64;
        for j in 0..ds {
            t += dh[j] * xs[j] as f64;
        }
        for j in 0..ds {
            bs[j] += w_perp as f64 * xs[j] as f64 + (w_par - w_perp) as f64 * dh[j] * t;
        }
    }

    // Solve N c = b for each centroid; reseed empties.
    for c in 0..k {
        if counts[c] == 0 {
            // reseed from a random training point's subspace slice
            let r = rng.gen_range(0..n);
            let off = r * dim + s * ds;
            let dst = &mut pq.centroids[s][c * ds..(c + 1) * ds];
            dst.copy_from_slice(&train[off..off + ds]);
            continue;
        }
        let mat = &mut nmat[c * stride..(c + 1) * stride];
        let b = &mut bvec[c * ds..(c + 1) * ds];
        // small Tikhonov regularizer for numerical stability
        for j in 0..ds {
            mat[j * ds + j] += 1e-6;
        }
        if cholesky_solve_inplace(mat, b, ds) {
            let dst = &mut pq.centroids[s][c * ds..(c + 1) * ds];
            for j in 0..ds {
                dst[j] = b[j] as f32;
            }
        }
        // if Cholesky fails (shouldn't), keep previous centroid
    }
}

/// In-place Cholesky decomp + solve for symmetric positive-definite
/// `n×n` matrix `a` (row-major) and rhs `b` of length `n`. Returns
/// false on numerical failure. Solution is written back to `b`.
fn cholesky_solve_inplace(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    // Decompose A = L L^T, lower-triangular in-place.
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return false;
                }
                a[i * n + j] = sum.sqrt();
            } else {
                a[i * n + j] = sum / a[j * n + j];
            }
        }
    }
    // Forward solve L y = b (y stored in b)
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i * n + k] * b[k];
        }
        b[i] = sum / a[i * n + i];
    }
    // Back solve L^T x = y
    for i in (0..n).rev() {
        let mut sum = b[i];
        for k in (i + 1)..n {
            sum -= a[k * n + i] * b[k];
        }
        b[i] = sum / a[i * n + i];
    }
    true
}

impl Encoder for AnisotropicPq {
    fn code_size(&self) -> usize {
        self.inner.code_size()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn encode(&self, xs: &[f32], codes: &mut [u8]) {
        // Anisotropic encode: pick the codeword that minimizes the
        // SAME loss the codebook was trained on. Encoding by raw L2
        // here would partly undo the anisotropic shaping. Per Guo et
        // al. (ICML 2020), assignment must use the matching loss.
        let dim = self.inner.dim;
        let m = self.inner.m;
        let k = self.inner.k;
        let ds = self.inner.ds;
        let eta = self.eta;
        let n = xs.len() / dim;
        for i in 0..n {
            let row = &xs[i * dim..(i + 1) * dim];
            let mut norm2 = 0.0f32;
            for &v in row {
                norm2 += v * v;
            }
            let inv_norm = 1.0 / norm2.max(1e-30).sqrt();
            for s in 0..m {
                let xs_s = &row[s * ds..(s + 1) * ds];
                let mut dh = [0.0f32; 64];
                let dh = &mut dh[..ds];
                for j in 0..ds {
                    dh[j] = xs_s[j] * inv_norm;
                }
                let cs = &self.inner.centroids[s];
                let mut best = 0u8;
                let mut best_l = f32::INFINITY;
                for c in 0..k {
                    let cent = &cs[c * ds..(c + 1) * ds];
                    let mut r_dot_d = 0.0f32;
                    let mut r2 = 0.0f32;
                    for j in 0..ds {
                        let r = xs_s[j] - cent[j];
                        r_dot_d += r * dh[j];
                        r2 += r * r;
                    }
                    let par2 = r_dot_d * r_dot_d;
                    let perp2 = (r2 - par2).max(0.0);
                    let l = eta * par2 + perp2;
                    if l < best_l {
                        best_l = l;
                        best = c as u8;
                    }
                }
                codes[i * m + s] = best;
            }
        }
    }
}

impl Scorer for AnisotropicPq {
    fn score_ip(&self, query: &[f32], codes: &[u8], out: &mut [f32]) {
        self.inner.score_ip(query, codes, out)
    }
}
