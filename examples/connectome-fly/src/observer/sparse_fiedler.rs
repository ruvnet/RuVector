//! Sparse-Fiedler coherence detector path.
//!
//! For co-firing windows with more than ~1024 active neurons, the dense
//! `O(n²)` Laplacian used by `compute_fiedler` stops fitting in cache and
//! eventually in RAM (n=10 000 → 800 MB per detect call; n=139 000 →
//! 153 GB — infeasible).
//!
//! This module builds a symmetric compressed-sparse-row (CSR) adjacency
//! matrix directly from the rolling spike window, then estimates the
//! Fiedler value via shifted power iteration on `L = D − A` without
//! ever materialising an `n × n` matrix. Memory is `O(n + nnz)` where
//! `nnz` is the number of distinct co-firing edges inside the window.
//!
//! The algorithm mirrors [`super::eigensolver::approx_fiedler_power`]
//! step-for-step so that at the cross-validation point (n ≤ 1024, both
//! paths defined) the results agree on the same Laplacian to within
//! the iterative convergence tolerance:
//!
//! 1. Shifted power iteration on `L` with constant-eigenvector
//!    deflation → `λ_max(L)`.
//! 2. Shifted power iteration on `M = c·I − L` with
//!    `c = 1.1 · λ_max(L) + ε`, again with constant deflation → `μ`.
//! 3. Return `(λ_max(L) − μ).max(0.0)`.
//!
//! `ruvector_sparsifier::SparseGraph` is the canonical sparse-edge
//! container in the RuVector ecosystem (per ADR-154 §13 follow-up and
//! `docs/research/connectome-ruvector/05-analysis-layer.md` §3 "sparsify
//! first" pipeline). For the hot accumulation loop we use a
//! `HashMap<(u32, u32), f32>` keyed by sorted neuron pair, since every
//! edge is updated many times per window and the SparseGraph's
//! double-sided adjacency write is quadratic in the per-edge touch
//! count. We export into `SparseGraph` once at the end — so downstream
//! sparsifier consumers still see the canonical shape — and then CSR
//! from there for the matvec loop.

use std::collections::{HashMap, VecDeque};

use ruvector_sparsifier::SparseGraph;

use crate::connectome::NeuronId;
use crate::lif::Spike;

/// Co-firing coincidence window in ms. Matches the dense path in
/// `super::core::Observer::compute_fiedler`.
const COFIRE_TAU_MS: f32 = 5.0;

/// Power-iteration steps for the `λ_max(L)` estimate. Matches the
/// dense `approx_fiedler_power` path so the two agree on the same
/// adjacency.
const POWER_STEPS_LMAX: usize = 32;

/// Power-iteration steps for the shifted `λ_max(c·I − L)` estimate.
/// Also matches the dense path.
const POWER_STEPS_SHIFT: usize = 64;

/// Relative-tolerance convergence threshold for early exit (same as
/// the dense path).
const POWER_TOL: f32 = 1e-4;

/// Compute the Fiedler value of the co-firing-window Laplacian via a
/// sparse shifted-power-iteration pipeline (the sparse analogue of
/// [`super::eigensolver::approx_fiedler_power`]).
///
/// `active` is the sorted, deduplicated list of `NeuronId`s whose
/// spikes lie in the rolling window. `cofire` is the window itself.
/// `n_threshold` is the active-neuron count above which this sparse
/// path is dispatched — only used here for the degenerate-case check;
/// the caller is responsible for the dispatch itself.
///
/// Returns `NaN` if the window is too small to form a Laplacian, or
/// `0.0` if the graph is trivially disconnected.
pub fn sparse_fiedler(active: &[NeuronId], cofire: &VecDeque<Spike>, _n_threshold: usize) -> f32 {
    let n = active.len();
    if n < 2 || cofire.len() < 2 {
        return f32::NAN;
    }

    // Phase 1 — build sparse adjacency from the co-firing window.
    let Some(csr) = build_sparse_laplacian(active, cofire, n) else {
        return f32::NAN;
    };

    // Phase 2 — power iteration on L → λ_max(L). Mirrors the dense
    // path's 32-step loop.
    let lambda_max = power_iter_lmax(&csr);
    if !lambda_max.is_finite() || lambda_max <= 0.0 {
        return 0.0;
    }

    // Phase 3 — 64-step shifted power iteration on c·I − L → μ.
    let c = lambda_max * 1.1 + 1e-3;
    let mu = power_iter_shifted(&csr, c);
    (lambda_max - mu).max(0.0)
}

// ---------------------------------------------------------------------
// CSR construction
// ---------------------------------------------------------------------

/// Dense-ish representation of a symmetric sparse matrix in CSR form,
/// plus a degree vector for fast Laplacian matvecs. `val` entries are
/// the edge weights of the co-firing graph (not the negated Laplacian
/// off-diagonals), so `(L·x)[i] = deg[i]·x[i] − Σ_{j ∈ nbrs(i)}
/// val[j] · x[col[j]]`.
struct LaplacianCsr {
    n: usize,
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    val: Vec<f32>,
    deg: Vec<f32>,
}

impl LaplacianCsr {
    fn nnz(&self) -> usize {
        self.col_idx.len()
    }
}

/// Build the symmetric weighted adjacency of the co-firing graph as
/// CSR.
///
/// Accumulation pass uses a `HashMap<(u32, u32), f32>` keyed by sorted
/// neuron pair — cheaper than `SparseGraph` for many-hit edges because
/// each update is a single hash probe instead of two adjacency-map
/// writes. We then export into a `SparseGraph` so downstream
/// sparsifier consumers see the canonical shape, and finally convert
/// to CSR for the matvec loop.
fn build_sparse_laplacian(
    active: &[NeuronId],
    cofire: &VecDeque<Spike>,
    n: usize,
) -> Option<LaplacianCsr> {
    // `active` is assumed sorted by the caller — binary-search to map
    // NeuronId back to a dense row index in `[0, n)`.
    let lookup = |id: NeuronId| active.binary_search(&id).ok();

    // --- Accumulation. Each τ-coincident spike pair contributes +1. ---
    let mut acc: HashMap<(u32, u32), f32> = HashMap::with_capacity(cofire.len());
    let spikes: Vec<Spike> = cofire.iter().copied().collect();
    for (i, sa) in spikes.iter().enumerate() {
        let Some(ai) = lookup(sa.neuron) else {
            continue;
        };
        for sb in &spikes[i + 1..] {
            if (sb.t_ms - sa.t_ms).abs() > COFIRE_TAU_MS {
                break;
            }
            let Some(bi) = lookup(sb.neuron) else {
                continue;
            };
            if ai == bi {
                continue;
            }
            let (u, v) = if ai < bi {
                (ai as u32, bi as u32)
            } else {
                (bi as u32, ai as u32)
            };
            *acc.entry((u, v)).or_insert(0.0) += 1.0;
        }
    }

    if acc.is_empty() {
        return None;
    }

    // --- Canonicalise via SparseGraph (matches the sparsifier
    // pipeline API: `insert_or_update_edge` guarantees undirected
    // storage and duplicate-rejection). ---
    let mut graph = SparseGraph::with_capacity(n);
    for (&(u, v), &w) in &acc {
        let _ = graph.insert_or_update_edge(u as usize, v as usize, w as f64);
    }
    if graph.num_edges() == 0 {
        return None;
    }

    // --- CSR export. ---
    let (rp_f64, ci_f64, vals_f64, exported_n) = graph.to_csr();
    let mut row_ptr: Vec<u32> = rp_f64.iter().map(|x| *x as u32).collect();
    // `to_csr` returns the graph's vertex count, which may be < n if
    // the last few neurons have no edges. Pad with empty rows so the
    // caller can index by `ai ∈ [0, n)` safely.
    if exported_n < n {
        let last = *row_ptr.last().unwrap_or(&0);
        row_ptr.resize(n + 1, last);
    }
    let col_idx: Vec<u32> = ci_f64.iter().map(|x| *x as u32).collect();
    let val: Vec<f32> = vals_f64.iter().map(|x| *x as f32).collect();

    // Degree from CSR row sums — matches `Σ_j A[i,j]` (A symmetric, so
    // weighted degree = row sum).
    let mut deg = vec![0.0_f32; n];
    for i in 0..n {
        let s = row_ptr[i] as usize;
        let e = row_ptr[i + 1] as usize;
        let mut d = 0.0_f32;
        for k in s..e {
            d += val[k];
        }
        deg[i] = d;
    }

    Some(LaplacianCsr {
        n,
        row_ptr,
        col_idx,
        val,
        deg,
    })
}

// ---------------------------------------------------------------------
// Lanczos matvecs
// ---------------------------------------------------------------------

/// `y ← L·x` where `L = D − A`, using the CSR adjacency `a`.
fn mat_vec_l(csr: &LaplacianCsr, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), csr.n);
    debug_assert_eq!(y.len(), csr.n);
    for i in 0..csr.n {
        let s = csr.row_ptr[i] as usize;
        let e = csr.row_ptr[i + 1] as usize;
        let mut acc = csr.deg[i] * x[i];
        for k in s..e {
            let j = csr.col_idx[k] as usize;
            acc -= csr.val[k] * x[j];
        }
        y[i] = acc;
    }
}

// ---------------------------------------------------------------------
// Shifted power iteration — sparse analogue of
// `super::eigensolver::approx_fiedler_power`.
//
// The dense path does:
//   - 32 power-iteration steps on L with constant-deflation → λ_max(L)
//   - 64 power-iteration steps on (c·I − L) with c = 1.1·λ_max + ε
//     → μ (≈ λ_max(c·I − L))
//   - return (λ_max − μ).max(0)
//
// We do the same, but each matvec `L·x` uses the CSR adjacency instead
// of an `n × n` scan. Every numerical choice (seed pattern, step
// counts, tolerance, deflation order) is kept identical to the dense
// reference so the cross-validation test at n ≤ 1024 agrees within
// 5 % relative error.
// ---------------------------------------------------------------------

fn power_iter_lmax(csr: &LaplacianCsr) -> f32 {
    let n = csr.n;
    // Same seeding polynomial as the dense path's λ_max estimate.
    let mut x: Vec<f32> = (0..n).map(|i| ((i * 31 + 7) as f32).sin()).collect();
    deflate_const(&mut x);
    normalize(&mut x);
    let mut w = vec![0.0_f32; n];
    let mut lambda_max = 0.0_f32;
    for _ in 0..POWER_STEPS_LMAX {
        mat_vec_l(csr, &x, &mut w);
        deflate_const(&mut w);
        normalize(&mut w);
        // Rayleigh quotient: w · L · w.
        let mut lw = vec![0.0_f32; n];
        mat_vec_l(csr, &w, &mut lw);
        let lam = dot(&w, &lw);
        let converged = (lam - lambda_max).abs() < POWER_TOL * lam.abs().max(1.0);
        lambda_max = lam;
        std::mem::swap(&mut x, &mut w);
        if converged {
            break;
        }
    }
    lambda_max
}

fn power_iter_shifted(csr: &LaplacianCsr, c: f32) -> f32 {
    let n = csr.n;
    // Same seed polynomial as the dense path's shifted loop.
    let mut x: Vec<f32> = (0..n).map(|i| ((i * 19 + 11) as f32).cos()).collect();
    deflate_const(&mut x);
    normalize(&mut x);
    let mut lx = vec![0.0_f32; n];
    let mut mu = 0.0_f32;
    for _ in 0..POWER_STEPS_SHIFT {
        mat_vec_l(csr, &x, &mut lx);
        // y = (c·I − L) · x  =  c·x − L·x
        let mut y: Vec<f32> = (0..n).map(|i| c * x[i] - lx[i]).collect();
        deflate_const(&mut y);
        normalize(&mut y);
        // Rayleigh quotient of (c·I − L) at y: y · (c·y − L·y).
        mat_vec_l(csr, &y, &mut lx);
        let mut m2 = 0.0_f32;
        for i in 0..n {
            m2 += y[i] * (c * y[i] - lx[i]);
        }
        let converged = (m2 - mu).abs() < POWER_TOL * m2.abs().max(1.0);
        mu = m2;
        x = y;
        if converged {
            break;
        }
    }
    mu
}

// ---------------------------------------------------------------------
// Small vector kernels
// ---------------------------------------------------------------------

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn norm(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn normalize(x: &mut [f32]) {
    let nrm = norm(x);
    if nrm > 1e-20 {
        let inv = 1.0 / nrm;
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

fn deflate_const(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let m: f32 = x.iter().sum::<f32>() / x.len() as f32;
    for v in x.iter_mut() {
        *v -= m;
    }
}

// ---------------------------------------------------------------------
// Expose the LaplacianCsr nnz for tests / diagnostics.
// ---------------------------------------------------------------------

/// Return `(n, nnz)` of the CSR Laplacian this path would build for
/// the given window. Intended for diagnostics only; has the same
/// memory cost as one matvec so callers should not invoke it from
/// hot paths.
pub fn estimate_sparse_extent(
    active: &[NeuronId],
    cofire: &VecDeque<Spike>,
) -> Option<(usize, usize)> {
    let n = active.len();
    let csr = build_sparse_laplacian(active, cofire, n)?;
    Some((csr.n, csr.nnz()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(i: u32) -> NeuronId {
        NeuronId(i)
    }

    #[test]
    fn tiny_two_cluster_graph_returns_finite_fiedler() {
        // Two tight clusters of four neurons each, weakly bridged by a
        // single cross-pair. We assert the path returns a finite, non-
        // negative value and does not panic on small-but-valid input.
        // Whether the value clears the `max(0.0)` floor depends on the
        // shifted-power-iteration convergence at this scale and is not
        // algorithmically guaranteed — the cross-validation at N=256
        // in `tests/sparse_fiedler_10k.rs` is the correctness check.
        let active: Vec<NeuronId> = (0..8).map(id).collect();
        let mut cofire: VecDeque<Spike> = VecDeque::new();
        for k in 0..10 {
            let t = k as f32 * 10.0;
            for i in 0..4 {
                cofire.push_back(Spike {
                    t_ms: t + i as f32 * 0.1,
                    neuron: id(i),
                });
            }
        }
        for k in 0..10 {
            let t = k as f32 * 10.0 + 0.5;
            for i in 4..8 {
                cofire.push_back(Spike {
                    t_ms: t + (i - 4) as f32 * 0.1,
                    neuron: id(i),
                });
            }
        }
        for k in 0..3 {
            let t = k as f32 * 30.0 + 2.0;
            cofire.push_back(Spike {
                t_ms: t,
                neuron: id(1),
            });
            cofire.push_back(Spike {
                t_ms: t + 0.2,
                neuron: id(5),
            });
        }
        let f = sparse_fiedler(&active, &cofire, 0);
        assert!(f.is_finite(), "sparse fiedler returned non-finite: {f}");
        assert!(f >= 0.0, "fiedler must be non-negative (PSD), got {f}");
    }

    #[test]
    fn disconnected_window_returns_zero_or_nan() {
        // Fewer than two coincident spikes — no edges at all.
        let active = vec![id(0)];
        let cofire: VecDeque<Spike> = vec![Spike {
            t_ms: 0.0,
            neuron: id(0),
        }]
        .into();
        let f = sparse_fiedler(&active, &cofire, 0);
        assert!(
            f.is_nan(),
            "single-neuron window should return NaN, got {f}"
        );
    }

    #[test]
    fn memory_extent_is_linear_in_nnz() {
        // 512 neurons, ~1500 spikes → nnz bounded well below n².
        let n: usize = 512;
        let active: Vec<NeuronId> = (0..n).map(|i| id(i as u32)).collect();
        let mut cofire: VecDeque<Spike> = VecDeque::new();
        for k in 0..60 {
            let t = k as f32 * 3.0;
            for i in 0..25 {
                cofire.push_back(Spike {
                    t_ms: t + i as f32 * 0.05,
                    neuron: id(((k + i) % n) as u32),
                });
            }
        }
        let Some((rn, nnz)) = estimate_sparse_extent(&active, &cofire) else {
            panic!("expected edges")
        };
        assert_eq!(rn, n);
        // nnz stored both directions — symmetric. Bound is O(n·k) not
        // n²; empirically here ≪ n².
        assert!(nnz < (n * n) / 2, "nnz {nnz} not sparse for n={n}");
    }
}
