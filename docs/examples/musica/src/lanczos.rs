//! SIMD-optimized sparse Lanczos eigensolver for graph Laplacians.
//!
//! Computes the smallest k eigenvectors of L = D - W using Lanczos iteration
//! with selective reorthogonalization. Designed for audio separation graphs
//! where k is typically 2-6 and matrices are sparse (32-2000 nodes).
//!
//! The code is structured for auto-vectorization: inner loops process
//! contiguous f64 slices without branches.

/// Compressed Sparse Row representation of a symmetric matrix.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Row pointer: row_ptr[i]..row_ptr[i+1] are the entries in row i.
    pub row_ptr: Vec<usize>,
    /// Column indices.
    pub col_idx: Vec<usize>,
    /// Non-zero values.
    pub values: Vec<f64>,
    /// Matrix dimension.
    pub n: usize,
}

/// Result of eigendecomposition.
#[derive(Debug, Clone)]
pub struct EigenResult {
    /// Eigenvalues (sorted ascending).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (one per eigenvalue).
    pub eigenvectors: Vec<Vec<f64>>,
    /// Number of Lanczos iterations used.
    pub iterations: usize,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Lanczos solver configuration.
#[derive(Debug, Clone)]
pub struct LanczosConfig {
    /// Number of eigenpairs to compute.
    pub k: usize,
    /// Maximum Lanczos iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Whether to reorthogonalize Lanczos vectors.
    pub reorthogonalize: bool,
}

impl Default for LanczosConfig {
    fn default() -> Self {
        Self {
            k: 4,
            max_iter: 100,
            tol: 1e-8,
            reorthogonalize: true,
        }
    }
}

impl SparseMatrix {
    /// Create empty n x n matrix.
    pub fn new(n: usize) -> Self {
        Self {
            row_ptr: vec![0; n + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
            n,
        }
    }

    /// Build graph Laplacian L = D - W from weighted edges.
    pub fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        // Build adjacency lists
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut degree = vec![0.0f64; n];

        for &(u, v, w) in edges {
            if u < n && v < n && u != v {
                adj[u].push((v, w));
                adj[v].push((u, w));
                degree[u] += w;
                degree[v] += w;
            }
        }

        // Sort adjacency for CSR
        for row in &mut adj {
            row.sort_by_key(|&(col, _)| col);
        }

        // Build CSR for L = D - W
        let mut row_ptr = vec![0usize; n + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();

        for i in 0..n {
            // Diagonal: degree[i]
            // Off-diagonal: -w for each neighbor

            // Insert entries in column order, including diagonal
            let mut entries: Vec<(usize, f64)> = Vec::new();

            // Add off-diagonal entries
            for &(j, w) in &adj[i] {
                entries.push((j, -w));
            }

            // Add diagonal
            entries.push((i, degree[i]));
            entries.sort_by_key(|&(col, _)| col);

            // Merge duplicates
            let mut merged: Vec<(usize, f64)> = Vec::new();
            for (col, val) in entries {
                if let Some(last) = merged.last_mut() {
                    if last.0 == col {
                        last.1 += val;
                        continue;
                    }
                }
                merged.push((col, val));
            }

            for (col, val) in &merged {
                col_idx.push(*col);
                values.push(*val);
            }
            row_ptr[i + 1] = col_idx.len();
        }

        Self {
            row_ptr,
            col_idx,
            values,
            n,
        }
    }

    /// Matrix-vector product y = A * x (auto-vectorization friendly).
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        assert!(x.len() >= self.n && y.len() >= self.n);
        for i in 0..self.n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut sum = 0.0f64;
            // Inner loop is contiguous access — compiler will auto-vectorize
            for idx in start..end {
                sum += self.values[idx] * x[self.col_idx[idx]];
            }
            y[i] = sum;
        }
    }

    /// Matrix dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Matrix dimension (alias for compatibility).
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }
}

// ── SIMD-friendly vector operations ─────────────────────────────────────

/// Dot product (auto-vectorizes on contiguous slices).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f64;
    // Process in chunks of 4 for auto-vectorization
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        sum += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3];
    }
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        sum += a[i] * b[i];
    }
    sum
}

/// L2 norm.
#[inline]
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// axpy: y = y + alpha * x
#[inline]
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    let n = x.len().min(y.len());
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        y[base] += alpha * x[base];
        y[base + 1] += alpha * x[base + 1];
        y[base + 2] += alpha * x[base + 2];
        y[base + 3] += alpha * x[base + 3];
    }
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        y[i] += alpha * x[i];
    }
}

/// Scale vector: x = alpha * x
#[inline]
fn scale(alpha: f64, x: &mut [f64]) {
    let n = x.len();
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        x[base] *= alpha;
        x[base + 1] *= alpha;
        x[base + 2] *= alpha;
        x[base + 3] *= alpha;
    }
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        x[i] *= alpha;
    }
}

// ── Lanczos algorithm ───────────────────────────────────────────────────

/// Compute the k smallest eigenpairs of a sparse symmetric matrix
/// using the Lanczos algorithm with selective reorthogonalization.
pub fn lanczos_eigenpairs(laplacian: &SparseMatrix, config: &LanczosConfig) -> EigenResult {
    let n = laplacian.dim();
    if n == 0 {
        return EigenResult {
            eigenvalues: vec![],
            eigenvectors: vec![],
            iterations: 0,
            converged: true,
        };
    }

    let k = config.k.min(n);
    let m = config.max_iter.min(n).max(k + 5);

    // Lanczos vectors
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(m + 1);

    // Tridiagonal matrix entries
    let mut alpha_diag = Vec::with_capacity(m);
    let mut beta_off: Vec<f64> = Vec::with_capacity(m);

    // Initial vector (normalized)
    let mut q0 = vec![0.0; n];
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    for (i, v) in q0.iter_mut().enumerate() {
        // Use slightly non-uniform init to avoid trivial eigenvector
        *v = inv_sqrt_n + (i as f64 * 0.01 / n as f64);
    }
    let n0 = norm(&q0);
    scale(1.0 / n0, &mut q0);
    q.push(q0);

    let mut w = vec![0.0; n];

    for j in 0..m {
        // w = A * q[j]
        laplacian.matvec(&q[j], &mut w);

        // alpha_j = q[j]' * w
        let alpha_j = dot(&q[j], &w);
        alpha_diag.push(alpha_j);

        // w = w - alpha_j * q[j]
        axpy(-alpha_j, &q[j], &mut w);

        // w = w - beta_{j-1} * q[j-1]
        if j > 0 {
            axpy(-beta_off[j - 1], &q[j - 1], &mut w);
        }

        // Reorthogonalize against all previous vectors
        if config.reorthogonalize {
            for qi in &q {
                let proj = dot(&w, qi);
                axpy(-proj, qi, &mut w);
            }
        }

        let beta_j = norm(&w);
        beta_off.push(beta_j);

        if beta_j < config.tol {
            break; // Invariant subspace found
        }

        // Normalize and store
        let mut q_next = w.clone();
        scale(1.0 / beta_j, &mut q_next);
        q.push(q_next);
    }

    let iters = alpha_diag.len();

    // Solve tridiagonal eigenproblem
    let (eigenvalues, eigvec_tri) = tridiagonal_qr(&alpha_diag, &beta_off, config.tol);

    // Map back to original space: v = Q * z
    let mut result_eigenvalues = Vec::new();
    let mut result_eigenvectors = Vec::new();

    for i in 0..k.min(eigenvalues.len()) {
        result_eigenvalues.push(eigenvalues[i]);

        let mut v = vec![0.0; n];
        for j in 0..iters.min(q.len()) {
            if j < eigvec_tri[i].len() {
                axpy(eigvec_tri[i][j], &q[j], &mut v);
            }
        }

        // Normalize
        let nv = norm(&v);
        if nv > 1e-12 {
            scale(1.0 / nv, &mut v);
        }

        result_eigenvectors.push(v);
    }

    EigenResult {
        eigenvalues: result_eigenvalues,
        eigenvectors: result_eigenvectors,
        iterations: iters,
        converged: iters < m,
    }
}

/// Implicit QR algorithm for symmetric tridiagonal matrix.
/// Returns (eigenvalues sorted ascending, eigenvectors as columns of Q).
fn tridiagonal_qr(alpha: &[f64], beta: &[f64], tol: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = alpha.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![alpha[0]], vec![vec![1.0]]);
    }

    // Copy tridiagonal entries
    let mut d = alpha.to_vec();
    let mut e: Vec<f64> = (0..n - 1).map(|i| beta[i.min(beta.len() - 1)]).collect();

    // Accumulate eigenvectors
    let mut z: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut v = vec![0.0; n];
        v[i] = 1.0;
        v
    }).collect();

    // QR iteration (Wilkinson shift)
    for _ in 0..n * 30 {
        // Find unreduced submatrix
        let mut bottom = n - 1;
        while bottom > 0 && e[bottom - 1].abs() < tol * (d[bottom - 1].abs() + d[bottom].abs()).max(tol) {
            bottom -= 1;
        }
        if bottom == 0 {
            break;
        }

        let mut top = bottom - 1;
        while top > 0 && e[top - 1].abs() >= tol * (d[top - 1].abs() + d[top].abs()).max(tol) {
            top -= 1;
        }

        // Wilkinson shift
        let delta = (d[bottom - 1] - d[bottom]) / 2.0;
        let shift = d[bottom]
            - e[bottom - 1] * e[bottom - 1]
                / (delta + delta.signum() * (delta * delta + e[bottom - 1] * e[bottom - 1]).sqrt());

        // Givens rotations
        let mut x = d[top] - shift;
        let mut z_val = e[top];

        for k in top..bottom {
            let (c, s) = givens(x, z_val);

            if k > top {
                e[k - 1] = (x * x + z_val * z_val).sqrt();
            }

            let d1 = d[k];
            let d2 = d[k + 1];
            let ek = e[k];

            d[k] = c * c * d1 + 2.0 * c * s * ek + s * s * d2;
            d[k + 1] = s * s * d1 - 2.0 * c * s * ek + c * c * d2;
            e[k] = c * s * (d2 - d1) + (c * c - s * s) * ek;

            // Update eigenvectors
            for i in 0..n {
                let zi_k = z[i][k];
                let zi_k1 = z[i][k + 1];
                z[i][k] = c * zi_k + s * zi_k1;
                z[i][k + 1] = -s * zi_k + c * zi_k1;
            }

            if k < bottom - 1 {
                x = e[k];
                z_val = s * e[k + 1];
                e[k + 1] *= c;
            }
        }
    }

    // Sort by eigenvalue
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap());

    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| d[i]).collect();
    let sorted_eigenvectors: Vec<Vec<f64>> = indices
        .iter()
        .map(|&idx| {
            (0..n).map(|i| z[i][idx]).collect()
        })
        .collect();

    (sorted_eigenvalues, sorted_eigenvectors)
}

/// Compute Givens rotation coefficients.
#[inline]
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-15 {
        (1.0, 0.0)
    } else if b.abs() > a.abs() {
        let t = -a / b;
        let s = 1.0 / (1.0 + t * t).sqrt();
        (s * t, s)
    } else {
        let t = -b / a;
        let c = 1.0 / (1.0 + t * t).sqrt();
        (c, c * t)
    }
}

/// Simple power iteration for the Fiedler vector only.
/// Faster than full Lanczos when only one eigenvector is needed.
pub fn power_iteration_fiedler(laplacian: &SparseMatrix, max_iter: usize) -> Vec<f64> {
    let n = laplacian.dim();
    if n <= 1 {
        return vec![0.0; n];
    }

    // Find approximate largest eigenvalue for shift
    let mut v = vec![0.0; n];
    let mut w = vec![0.0; n];

    // Init with non-constant vector
    for (i, val) in v.iter_mut().enumerate() {
        *val = (i as f64 / n as f64) - 0.5;
    }

    // Remove constant component
    let mean: f64 = v.iter().sum::<f64>() / n as f64;
    for x in &mut v {
        *x -= mean;
    }
    let nv = norm(&v);
    if nv > 1e-12 {
        scale(1.0 / nv, &mut v);
    }

    // Estimate max eigenvalue
    laplacian.matvec(&v, &mut w);
    let lambda_max_est = dot(&v, &w).abs() * 2.0 + 1.0;

    // Inverse iteration on (lambda_max*I - L) to find largest eigenvector of shifted system
    // This gives the Fiedler vector (smallest non-trivial eigenvector of L)
    for _ in 0..max_iter {
        // w = (lambda_max * I - L) * v
        laplacian.matvec(&v, &mut w);
        for i in 0..n {
            w[i] = lambda_max_est * v[i] - w[i];
        }

        // Remove constant component (project out trivial eigenvector)
        let mean: f64 = w.iter().sum::<f64>() / n as f64;
        for x in &mut w {
            *x -= mean;
        }

        // Normalize
        let nw = norm(&w);
        if nw < 1e-12 {
            break;
        }
        scale(1.0 / nw, &mut w);

        v.copy_from_slice(&w);
    }

    v
}

/// Align current eigenvectors with previous frame's eigenvectors
/// using sign consistency (simplified Procrustes).
///
/// For each eigenvector pair, computes the inner product with the
/// corresponding previous vector and flips the sign if negative.
/// This prevents sign-flip discontinuities across STFT frames.
pub fn align_eigenvectors(current: &mut [Vec<f64>], previous: &[Vec<f64>]) {
    let k = current.len().min(previous.len());

    for i in 0..k {
        let n = current[i].len().min(previous[i].len());
        if n == 0 {
            continue;
        }

        // Compute overlap with previous frame's eigenvector
        let d = dot(&current[i][..n], &previous[i][..n]);
        if d < 0.0 {
            // Flip sign to maintain consistency across frames
            for x in &mut current[i] {
                *x = -*x;
            }
        }
    }
}

/// Batch mode: compute eigenpairs for multiple windows (graph Laplacians),
/// with cross-frame eigenvector alignment applied automatically.
///
/// Each `SparseMatrix` in `laplacians` represents one STFT window's graph.
/// Returns one `EigenResult` per window, with eigenvectors aligned to
/// the previous window via Procrustes sign consistency.
pub fn batch_lanczos(
    laplacians: &[SparseMatrix],
    config: &LanczosConfig,
) -> Vec<EigenResult> {
    if laplacians.is_empty() {
        return Vec::new();
    }

    let mut results = Vec::with_capacity(laplacians.len());

    // Process first window
    let first = lanczos_eigenpairs(&laplacians[0], config);
    results.push(first);

    // Process subsequent windows with alignment
    for i in 1..laplacians.len() {
        let mut result = lanczos_eigenpairs(&laplacians[i], config);

        // Align to previous frame
        let prev_vecs = &results[i - 1].eigenvectors;
        align_eigenvectors(&mut result.eigenvectors, prev_vecs);

        results.push(result);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplacian_construction() {
        // Triangle graph: 0-1, 1-2, 0-2, all weight 1
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)];
        let lap = SparseMatrix::from_edges(3, &edges);

        // L should have diagonal [2, 2, 2] and off-diagonal [-1]
        let mut y = vec![0.0; 3];
        lap.matvec(&[1.0, 0.0, 0.0], &mut y);
        assert!((y[0] - 2.0).abs() < 1e-10);
        assert!((y[1] - (-1.0)).abs() < 1e-10);
        assert!((y[2] - (-1.0)).abs() < 1e-10);

        // Constant vector should give zero (L * 1 = 0)
        lap.matvec(&[1.0, 1.0, 1.0], &mut y);
        for &val in &y {
            assert!(val.abs() < 1e-10, "L*1 should be 0, got {val}");
        }
    }

    #[test]
    fn test_fiedler_path_graph() {
        // Path: 0-1-2-3-4
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)];
        let lap = SparseMatrix::from_edges(5, &edges);
        let fiedler = power_iteration_fiedler(&lap, 100);

        // Should be monotonic (or approximately so) for path graph
        let diffs: Vec<f64> = fiedler.windows(2).map(|w| w[1] - w[0]).collect();
        let all_positive = diffs.iter().all(|&d| d > -0.05);
        let all_negative = diffs.iter().all(|&d| d < 0.05);
        assert!(
            all_positive || all_negative,
            "Fiedler vector should be roughly monotonic for path: {:?}",
            fiedler
        );
    }

    #[test]
    fn test_fiedler_two_clusters() {
        // Two clusters connected by a weak bridge
        // Cluster A: 0,1,2 (fully connected, weight 5)
        // Cluster B: 3,4,5 (fully connected, weight 5)
        // Bridge: 2-3 (weight 0.1)
        let mut edges = vec![];
        for i in 0..3 {
            for j in i + 1..3 {
                edges.push((i, j, 5.0));
            }
        }
        for i in 3..6 {
            for j in i + 1..6 {
                edges.push((i, j, 5.0));
            }
        }
        edges.push((2, 3, 0.1));

        let lap = SparseMatrix::from_edges(6, &edges);
        let fiedler = power_iteration_fiedler(&lap, 100);

        // Fiedler vector should clearly split the two clusters
        let cluster_a_sign = fiedler[0].signum();
        let cluster_b_sign = fiedler[3].signum();
        assert_ne!(
            cluster_a_sign as i32, cluster_b_sign as i32,
            "Two clusters should have opposite signs: {:?}",
            fiedler
        );

        // All nodes in each cluster should have same sign
        for i in 0..3 {
            assert_eq!(
                fiedler[i].signum() as i32,
                cluster_a_sign as i32,
                "Node {i} should be in cluster A"
            );
        }
        for i in 3..6 {
            assert_eq!(
                fiedler[i].signum() as i32,
                cluster_b_sign as i32,
                "Node {i} should be in cluster B"
            );
        }
    }

    #[test]
    fn test_eigenvalue_ordering() {
        let edges = vec![
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0),
            (3, 4, 1.0), (0, 4, 1.0), (1, 3, 0.5),
        ];
        let lap = SparseMatrix::from_edges(5, &edges);
        let config = LanczosConfig { k: 3, max_iter: 50, tol: 1e-8, reorthogonalize: true };
        let result = lanczos_eigenpairs(&lap, &config);

        // Eigenvalues should be non-negative
        for &ev in &result.eigenvalues {
            assert!(ev >= -1e-6, "Eigenvalue {ev} should be non-negative");
        }

        // Should be sorted ascending
        for w in result.eigenvalues.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "Eigenvalues not sorted: {} > {}", w[0], w[1]);
        }

        // Smallest eigenvalue should be non-negative
        // (May not be exactly zero due to Lanczos approximation)
        assert!(
            result.eigenvalues[0] >= -0.1,
            "Smallest eigenvalue should be non-negative, got {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn test_lanczos_vs_power_iteration() {
        // Both should agree on Fiedler vector direction
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)];
        let lap = SparseMatrix::from_edges(5, &edges);

        let power_fiedler = power_iteration_fiedler(&lap, 100);
        let config = LanczosConfig { k: 2, max_iter: 50, tol: 1e-8, reorthogonalize: true };
        let lanczos_result = lanczos_eigenpairs(&lap, &config);

        if lanczos_result.eigenvectors.len() >= 2 {
            let lanczos_fiedler = &lanczos_result.eigenvectors[1];

            // Directions should agree (modulo sign)
            let d = dot(&power_fiedler, lanczos_fiedler);
            assert!(
                d.abs() > 0.5,
                "Power and Lanczos Fiedler vectors should be aligned: dot={d:.3}"
            );
        }
    }

    #[test]
    fn test_dot_product_simd() {
        let a: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..100).map(|i| (100 - i) as f64 * 0.1).collect();

        let result = dot(&a, &b);
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!(
            (result - expected).abs() < 1e-10,
            "SIMD dot product mismatch: {result} vs {expected}"
        );
    }
}
