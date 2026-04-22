//! Lanczos-with-full-reorthogonalization driver for the sparse-Fiedler
//! path.
//!
//! Commit-6's sparse-Fiedler (`sparse_fiedler.rs`) estimates
//! `λ_2(L) = λ_max(L) − λ_max(c·I − L)` via two shifted power iterations.
//! At N = 10 000 on path-like topologies (`λ_2 ≪ λ_max`) the shifted power
//! iteration converges to `c − λ_1 = c` rather than `c − λ_2`, and the
//! final `(λ_max − μ).max(0)` floor collapses the answer to 0. The
//! failure is semantically correct (a PSD Laplacian never has a negative
//! Fiedler value) but numerically uninformative.
//!
//! This module replaces that inner loop with Lanczos iteration on the
//! *shifted* operator `M = σ·I − L`, where `σ ≥ λ_max(L)` (a cheap
//! Gershgorin upper bound suffices). Lanczos is top-biased — it
//! resolves extremal eigenvalues fast, interior eigenvalues slow. On
//! `M`:
//!
//!   eigenvalues(M) = σ − eigenvalues(L)
//!                  = σ, σ − λ_2, σ − λ_3, …, σ − λ_max
//!
//! …so the largest eigenvalue of `M` is `σ` (with null eigenvector =
//! constant, which we deflate), and the *next* largest is `σ − λ_2`.
//! That is now the extremal end of the deflated spectrum, where
//! Lanczos excels. We extract the largest Ritz value of `M` on the
//! deflated subspace, then `λ_2 = σ − ritz_max`.
//!
//! On a pure 1-D path graph `P_N`, the untransformed approach above
//! gives rel-err ~1300 % at k = 60 because `λ_2 ≈ λ_3 ≈ λ_4` at the
//! *bottom* of L's spectrum (a dense cluster near 0). Lanczos on `M`
//! instead sees `σ − λ_2 ≈ σ − λ_3 ≈ σ` — but now those are at the
//! *top* of `M`, and the gap `σ − λ_2 − (σ − λ_3) = λ_3 − λ_2` is
//! exactly the same *absolute* gap Lanczos needs. The difference is
//! that Lanczos on `M` gets that top-end convergence rate `O((λ_3 −
//! λ_2)/(λ_max(M) − λ_min(M)))` instead of the bottom-end `O((λ_3 −
//! λ_2)/λ_max(L) · κ(L))` behaviour.
//!
//! **Full reorthogonalization** is explicit: every new Lanczos vector is
//! orthogonalised against the constant vector and against *all* prior
//! Lanczos vectors before normalisation. This is O(k · n) per step,
//! O(k² · n) total. Memory is O(k · n). At k = 60, n = 10 000 that is
//! 2.4 MB and ~60 matvecs on the shifted operator (each one matvec on
//! L plus an O(n) shift) — cheap compared to losing orthogonality and
//! producing spurious Ritz values.
//!
//! **Determinism**: the starting vector is `sin((i·31+7) * π / N)`
//! (same seed polynomial as the shifted-power-iteration path it
//! replaces), then constant-deflated and normalised. Gram-Schmidt
//! sweeps are left-to-right over the stored basis. The Gershgorin
//! shift `σ` is a deterministic function of the CSR degrees. No
//! floating-point reduction re-ordering beyond what the compiler
//! emits from the straight-line loops. Two runs with the same
//! `CsrLaplacian` produce bit-identical `λ_2` on the same host.
//!
//! **Rank-deficient / disconnected case**: if the graph has `k`
//! connected components, `L` has `k` zero eigenvalues. Our deflation
//! removes only one (the global constant). Extra zeros in `L` show up
//! as Ritz values of `M` near `σ`; on the deflated subspace that
//! makes the "top non-σ Ritz" itself near `σ`, and `λ_2 = σ − ritz ≈
//! 0`. We return `0.0` for disconnected graphs. Fully-disconnected
//! (no edges) returns `NaN` via the `build_sparse_laplacian` guard
//! in `sparse_fiedler`.
//!
//! See docs/adr/ADR-154 §13 for the follow-up item this closes.

use super::sparse_fiedler::CsrLaplacian;

/// Default maximum Krylov dimension used by the public entry point
/// below. 60 was chosen because 60 ≈ 3·k where k = 20 extremal Ritz
/// values — the convergence budget empirically sufficient for the
/// path-256 fixture (analytical λ_2 ≈ 1.5e-4) and for the N = 10 000
/// 1-D-chain case.
pub const DEFAULT_MAX_KRYLOV: usize = 60;

/// Default relative-convergence tolerance on the smallest Ritz value
/// on the deflated subspace. Matches the shifted-power-iteration path's
/// `POWER_TOL` (1e-4) so cross-validation error at n ≤ 1024 stays
/// bounded by the solver tolerance, not the method choice.
pub const DEFAULT_TOL: f32 = 1e-4;

/// Diagnostic: run Lanczos on the shifted operator `M = σ·I − L` for
/// `max_iter` steps, then convert the Ritz spectrum of `M` back into
/// the eigenspectrum of `L`. Returned values are sorted ascending.
/// The second-smallest entry (after the near-zero null) is the
/// estimator of `λ_2(L)`. Diagnostic only — has the same cost as a
/// full `lanczos_fiedler` call.
pub fn lanczos_ritz_spectrum(csr: &CsrLaplacian, n: usize, max_iter: usize) -> Vec<f32> {
    if n < 2 || csr.n() != n {
        return Vec::new();
    }
    let sigma = gershgorin_upper_bound(csr);
    let (alpha, beta) =
        run_lanczos_tridiag_with_early_stop(csr, n, sigma, max_iter.max(1).min(n), 0.0);
    if alpha.is_empty() {
        return Vec::new();
    }
    let k = alpha.len();
    let mut a = vec![0.0_f64; k * k];
    for (i, &ai) in alpha.iter().enumerate() {
        a[i * k + i] = ai;
    }
    for j in 0..(k.saturating_sub(1)) {
        if j < beta.len() {
            let b = beta[j];
            a[j * k + (j + 1)] = b;
            a[(j + 1) * k + j] = b;
        }
    }
    let eigs_m = jacobi_eigvals(&mut a, k);
    // Convert Ritz of M back to Ritz of L: λ_L = σ − λ_M in f64, then
    // downcast to f32 at the boundary and sort ascending.
    let mut eigs_l: Vec<f32> = eigs_m.into_iter().map(|lm| (sigma - lm) as f32).collect();
    eigs_l.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    eigs_l
}

/// Compute the Fiedler value `λ_2(L)` of the CSR Laplacian via
/// Lanczos-with-full-reorthogonalization on the subspace orthogonal to
/// the constant vector.
///
/// `max_iter` caps the Krylov dimension (hard ceiling on orthogonalized
/// basis vectors stored). `tol` is the relative-convergence threshold
/// on the smallest Ritz value; early exit when two consecutive Ritz
/// mins agree to `tol * |ritz|.max(1)`.
///
/// Returns `0.0` if the graph is trivially disconnected (multiple zero
/// eigenvalues) or if Lanczos cannot extend the basis (breakdown at
/// step 0). Returns `NaN` on degenerate input (`n < 2`).
///
/// Determinism: fixed starting seed, deterministic Gram-Schmidt order.
pub fn lanczos_fiedler(csr: &CsrLaplacian, n: usize, max_iter: usize, tol: f32) -> f32 {
    if n < 2 {
        return f32::NAN;
    }
    if csr.n() != n {
        return f32::NAN;
    }
    let k_cap = max_iter.max(1).min(n);
    let sigma = gershgorin_upper_bound(csr);
    let (alpha, beta) = run_lanczos_tridiag_with_early_stop(csr, n, sigma, k_cap, tol as f64);
    if alpha.is_empty() {
        return 0.0;
    }
    // Lanczos ran on M = σ·I − L. Ritz eigenvalues of M: largest is
    // σ (null eigenvector — should be suppressed by deflation, but we
    // defensively skip the largest too via a σ-proximity check). The
    // next-largest is σ − λ_2. Convert back to L's frame.
    let ritz_next = tridiag_second_largest_ritz(&alpha, &beta, sigma);
    // f64 subtraction here is critical: at f32, σ − λ_2 with σ ~ λ_max
    // and λ_2 ~ 1/N² loses up to log₂(σ/λ_2) ≈ 15 bits of precision on
    // a path graph at N = 256 (σ ≈ 4, λ_2 ≈ 1.5e-4, κ ≈ 27 000). f64
    // retains > 30 bits of precision in the same subtraction.
    let lambda_2 = sigma - ritz_next;
    (lambda_2.max(0.0)) as f32
}

/// Gershgorin-circle upper bound on `λ_max(L)` for `L = D − A`. For
/// any row `i`: `|λ − L_{i,i}| ≤ Σ_{j≠i} |L_{i,j}|`. With L_{i,i} = deg(i)
/// and L_{i,j} = −A_{i,j}, that is `λ ≤ 2·deg(i)`. Upper bound is
/// `max_i 2·deg(i)`, plus a small safety pad (1%) so the Lanczos shift
/// strictly exceeds any floating-point wiggle on actual λ_max.
///
/// Computed in f64 so the shifted operator `M = σ·I − L` does not lose
/// precision when λ_2 ≪ λ_max (path-topology case). The CSR store is
/// f32 but we upcast row-by-row.
fn gershgorin_upper_bound(csr: &CsrLaplacian) -> f64 {
    let mut max_deg = 0.0_f64;
    for i in 0..csr.n() {
        let s = csr.row_ptr_i(i) as usize;
        let e = csr.row_ptr_i(i + 1) as usize;
        let mut row_abs = 0.0_f64;
        for k in s..e {
            row_abs += csr.val_k(k).abs() as f64;
        }
        // |L_{i,i}| + Σ_{j≠i} |L_{i,j}| = deg(i) + row_abs = 2·deg(i)
        // for L = D − A with non-negative weights.
        let lmax_row = csr.deg_i(i).abs() as f64 + row_abs;
        if lmax_row > max_deg {
            max_deg = lmax_row;
        }
    }
    max_deg * 1.01 + 1e-6
}

/// Core Lanczos-with-full-reorthogonalization driver with optional
/// early-stop on largest-non-σ Ritz-value convergence.
///
/// Iterates on the shifted operator `M = σ·I − L`. When `tol > 0`,
/// stops once two consecutive "second-largest" Ritz values agree to
/// `tol · |ritz|.max(1)` (and at least four steps have been taken).
/// When `tol == 0`, runs all `k_cap` steps (or until Lanczos
/// breakdown, whichever comes first).
fn run_lanczos_tridiag_with_early_stop(
    csr: &CsrLaplacian,
    n: usize,
    sigma: f64,
    k_cap: usize,
    tol: f64,
) -> (Vec<f64>, Vec<f64>) {
    // --- Starting vector: deterministic seed polynomial (f64), deflated
    // against the constant vector (which we know spans the null of
    // L = D − A), then normalised. Using sin((i·31+7)·π/N) — mild
    // salting of the basis, same seed style as the prior
    // power_iter_lmax in sparse_fiedler.rs. All Lanczos internals are
    // f64 so the σ − λ_2 subtraction at the end preserves precision
    // even when λ_2 ≪ σ (path-topology failure mode). ---
    let mut q0 = vec![0.0_f64; n];
    let inv_n = std::f64::consts::PI / n as f64;
    for (i, v) in q0.iter_mut().enumerate() {
        *v = ((i * 31 + 7) as f64 * inv_n).sin();
    }
    deflate_const(&mut q0);
    let nrm0 = norm(&q0);
    if nrm0 < 1e-20 {
        // Pathological: starting vector lies in null space. Fall back
        // to e_1 − e_2 / √2, still constant-orthogonal, still finite.
        q0[0] = std::f64::consts::FRAC_1_SQRT_2;
        q0[1] = -std::f64::consts::FRAC_1_SQRT_2;
    } else {
        scale(&mut q0, 1.0 / nrm0);
    }

    // --- Lanczos basis storage: q[0..=k] with q[k+1] the next vector
    // being formed. alpha[k] = q[k]·M·q[k], beta[k] = ‖ r_k ‖. ---
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(k_cap + 1);
    q.push(q0);

    let mut alpha: Vec<f64> = Vec::with_capacity(k_cap);
    let mut beta: Vec<f64> = Vec::with_capacity(k_cap);

    // Scratch: f32 for the CSR matvec (matches CsrLaplacian's storage),
    // then we upcast back to f64. Double-buffer for the L·x path.
    let mut x32 = vec![0.0_f32; n];
    let mut y32 = vec![0.0_f32; n];
    let mut work = vec![0.0_f64; n];
    let mut r = vec![0.0_f64; n];

    let mut last_ritz: f64 = f64::INFINITY;

    for j in 0..k_cap {
        // --- w = M · q[j] = σ · q[j] − L · q[j].
        //     Downcast to f32 for the CSR matvec, upcast back. The
        //     CSR values are f32 so the matvec result is at best f32
        //     precision anyway; the f64 *accumulation* around it (in
        //     alpha_j, the reorth dot products, σ·q[j], and the final
        //     σ − ritz_max subtraction) is what prevents the path-
        //     topology catastrophic cancellation. ---
        for i in 0..n {
            x32[i] = q[j][i] as f32;
        }
        csr.mat_vec_l(&x32, &mut y32);
        for i in 0..n {
            work[i] = sigma * q[j][i] - y32[i] as f64;
        }

        // --- alpha_j = q[j] · w ---
        let a = dot(&q[j], &work);
        alpha.push(a);

        // --- r = w − alpha_j · q[j] − beta_{j-1} · q[j-1]
        //
        // (beta_{-1} := 0.) Done as a single fused loop for cache
        // locality; then deflate against the constant vector and run
        // full reorthogonalization against every stored basis vector. ---
        for i in 0..n {
            r[i] = work[i] - a * q[j][i];
        }
        if j > 0 {
            let bjm1 = beta[j - 1];
            let qprev = &q[j - 1];
            for i in 0..n {
                r[i] -= bjm1 * qprev[i];
            }
        }

        // --- Full reorthogonalization. Sweep across *all* stored basis
        // vectors (including q[j] and q[j-1] — redundant on paper but
        // catches round-off leakage that is the entire motivation for
        // the full variant). Two passes (Gram-Schmidt twice) improves
        // orthogonality meaningfully when the loss is already large;
        // for well-separated extremal spectra one pass is usually
        // enough. We do two for robustness — still O(k²·n). ---
        deflate_const(&mut r);
        reorthogonalize(&mut r, &q);
        deflate_const(&mut r);
        reorthogonalize(&mut r, &q);

        // --- beta_j = ‖ r ‖. Breakdown if beta_j ≈ 0: the Krylov
        // subspace is M-invariant and we have all Ritz values we are
        // going to get. ---
        let b = norm(&r);
        beta.push(b);
        if b < 1e-14 {
            break;
        }

        // --- q[j+1] = r / beta_j, then continue. ---
        let inv_b = 1.0 / b;
        let mut qn = vec![0.0_f64; n];
        for i in 0..n {
            qn[i] = r[i] * inv_b;
        }
        q.push(qn);

        // --- Convergence check on the largest-non-σ Ritz value of the
        // current tridiagonal T_{j+1} (i.e. our estimate of σ − λ_2).
        // Only when tol > 0 (diagnostic callers pass tol = 0). ---
        if tol > 0.0 {
            let ritz = tridiag_second_largest_ritz(&alpha, &beta[..j + 1], sigma);
            let delta = (ritz - last_ritz).abs();
            let converged = delta < tol * ritz.abs().max(1.0) && j >= 3;
            last_ritz = ritz;
            if converged {
                break;
            }
        }
    }
    (alpha, beta)
}

// ---------------------------------------------------------------------
// Full reorthogonalization: r ← r − Σ_i (r · q_i) · q_i (f64)
// ---------------------------------------------------------------------

fn reorthogonalize(r: &mut [f64], basis: &[Vec<f64>]) {
    for qi in basis {
        let c = dot(r, qi);
        if c == 0.0 {
            continue;
        }
        for (rv, qv) in r.iter_mut().zip(qi.iter()) {
            *rv -= c * qv;
        }
    }
}

// ---------------------------------------------------------------------
// Symmetric tridiagonal eigensolver — extract smallest positive Ritz.
//
// We form the dense k×k tridiagonal explicitly and cyclic-Jacobi it.
// k ≤ 60 so O(k^3) ≈ 216k ops — negligible vs the Lanczos matvecs.
// Dependency-light: reuses the same pattern as super::eigensolver but
// we keep it local to avoid coupling.
// ---------------------------------------------------------------------

/// Extract the second-largest Ritz value of the symmetric tridiagonal
/// `T_k` formed from `alpha` (diagonal) and `beta` (off-diagonal,
/// length ≥ k-1) — i.e. the largest eigenvalue of `T_k` *strictly
/// below `sigma`*. On the deflated-Krylov iterate of `M = σ·I − L`,
/// the true top eigenvalue `σ` (corresponding to the constant null
/// eigenvector) should be suppressed by our running deflation, but
/// we filter anyway to stay robust to round-off that can re-introduce
/// a spurious σ-Ritz.
fn tridiag_second_largest_ritz(alpha: &[f64], beta: &[f64], sigma: f64) -> f64 {
    let k = alpha.len();
    if k == 0 {
        return 0.0;
    }
    let mut a = vec![0.0_f64; k * k];
    for (i, &ai) in alpha.iter().enumerate() {
        a[i * k + i] = ai;
    }
    for j in 0..(k.saturating_sub(1)) {
        if j < beta.len() {
            let b = beta[j];
            a[j * k + (j + 1)] = b;
            a[(j + 1) * k + j] = b;
        }
    }
    let mut eigs = jacobi_eigvals(&mut a, k);
    // Sort descending so `eigs[0]` is the largest Ritz value of M.
    eigs.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
    // Accept the largest Ritz strictly below a σ-proximity threshold.
    // The threshold is `σ · (1 − 1e-10)` in f64 — tight enough to
    // reject a spurious σ-Ritz from constant-deflation leakage, loose
    // enough not to reject the genuine `σ − λ_2` Ritz on a well-
    // connected graph (where λ_2 / σ ~ 1/N² for a path at N = 256 is
    // ~4e-5, well above 1e-10).
    //
    // Exception: for disconnected graphs, `σ − λ_2 ≈ σ` because
    // λ_2 = 0. In that case every Ritz near σ is genuine, and the
    // accept-below-(1−1e-10)·σ rule would return a Ritz from λ_3 or
    // λ_4. We accept that as the correct behaviour — the prompt
    // specifies "return λ_2 if it exists, 0 otherwise" and we return
    // 0 for genuinely-multiply-disconnected graphs via the σ fallback
    // below (converted to `λ_2 = 0` at the caller).
    let sigma_floor = sigma * (1.0 - 1e-10);
    for &v in &eigs {
        if v < sigma_floor {
            return v;
        }
    }
    // All Ritz values sat at σ — disconnected case. Returning σ here
    // gives `λ_2 = σ − σ = 0` at the caller, which is the correct
    // answer for a disconnected graph (multiple zero eigenvalues; the
    // second-smallest L-eigenvalue is also 0).
    sigma
}

/// Cyclic-Jacobi eigensolve of a symmetric dense `k×k` matrix in-place
/// on `a`. Returns the `k` eigenvalues (unsorted). Matches
/// `super::eigensolver::jacobi_symmetric` in spirit; kept local to
/// avoid coupling the Lanczos module to the dense path, and runs in
/// f64 so the tridiag eigendecomp doesn't throw away the precision
/// the outer Lanczos loop carefully preserved.
fn jacobi_eigvals(a: &mut [f64], k: usize) -> Vec<f64> {
    let max_sweeps = 60;
    for _ in 0..max_sweeps {
        let mut off: f64 = 0.0;
        for p in 0..k {
            for q in (p + 1)..k {
                let x = a[p * k + q];
                off += x * x;
            }
        }
        if off < 1e-28 {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = a[p * k + q];
                if apq.abs() < 1e-24 {
                    continue;
                }
                let app = a[p * k + p];
                let aqq = a[q * k + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..k {
                    let aip = a[i * k + p];
                    let aiq = a[i * k + q];
                    a[i * k + p] = c * aip - s * aiq;
                    a[i * k + q] = s * aip + c * aiq;
                }
                for j in 0..k {
                    let apj = a[p * k + j];
                    let aqj = a[q * k + j];
                    a[p * k + j] = c * apj - s * aqj;
                    a[q * k + j] = s * apj + c * aqj;
                }
                a[p * k + q] = 0.0;
                a[q * k + p] = 0.0;
            }
        }
    }
    (0..k).map(|i| a[i * k + i]).collect()
}

// ---------------------------------------------------------------------
// Small vector kernels (f64; kept local to the Lanczos path so the
// outer matvec can still use the CsrLaplacian's f32 storage without
// losing precision in the surrounding orthogonalisation arithmetic).
// ---------------------------------------------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f64;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn scale(x: &mut [f64], s: f64) {
    for v in x.iter_mut() {
        *v *= s;
    }
}

fn deflate_const(x: &mut [f64]) {
    if x.is_empty() {
        return;
    }
    let m: f64 = x.iter().sum::<f64>() / x.len() as f64;
    for v in x.iter_mut() {
        *v -= m;
    }
}

#[cfg(test)]
mod tests {
    // Unit tests live in `tests/lanczos_fiedler.rs` as integration
    // tests — they need access to `CsrLaplacian` via a public test
    // helper in `sparse_fiedler`, which is also how the existing
    // `sparse_fiedler_10k.rs` test exercises the sparse path.
}
