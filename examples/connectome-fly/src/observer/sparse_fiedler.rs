//! Sparse-Fiedler coherence detector path.
//!
//! For co-firing windows with more than ~1024 active neurons, the dense
//! `O(n²)` Laplacian used by `compute_fiedler` stops fitting in cache and
//! eventually in RAM (n=10 000 → 800 MB per detect call; n=139 000 →
//! 153 GB — infeasible).
//!
//! This module builds a symmetric compressed-sparse-row (CSR) adjacency
//! matrix directly from the rolling spike window, then estimates the
//! Fiedler value via Lanczos-with-full-reorthogonalization on
//! `L = D − A` (see [`super::lanczos::lanczos_fiedler`]). Memory is
//! `O(n + nnz + k·n)` where `nnz` is the number of distinct co-firing
//! edges inside the window and `k` is the Krylov dimension (≤ 60
//! default).
//!
//! Prior implementation (shifted power iteration) fell back to the
//! PSD-floor convention `(λ_max − μ).max(0)` → 0 on path-like
//! topologies where `λ_2 ≪ λ_max`. Commit 6 (`b805d7158`) named this
//! follow-up; ADR-154 §13 tracks it. The Lanczos driver replaces the
//! inner eigensolve and keeps the outer CSR-accumulation scaffolding
//! unchanged.
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

use super::lanczos::{lanczos_fiedler, DEFAULT_MAX_KRYLOV, DEFAULT_TOL};
use crate::connectome::NeuronId;
use crate::lif::Spike;

/// Co-firing coincidence window in ms. Matches the dense path in
/// `super::core::Observer::compute_fiedler`.
const COFIRE_TAU_MS: f32 = 5.0;

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

    // Phase 2 — Lanczos-with-full-reorthogonalization on L →
    // smallest-positive Ritz value ≈ λ_2(L). Replaces the prior
    // shifted-power-iteration pair which collapsed to 0 on path-like
    // topologies where `λ_2 ≪ λ_max` (see module docs and ADR-154
    // §13). The eigensolve step swaps; the CSR-accumulation
    // scaffolding above is unchanged.
    lanczos_fiedler(&csr, n, DEFAULT_MAX_KRYLOV, DEFAULT_TOL)
}

// ---------------------------------------------------------------------
// CSR construction
// ---------------------------------------------------------------------

/// Dense-ish representation of a symmetric sparse matrix in CSR form,
/// plus a degree vector for fast Laplacian matvecs. `val` entries are
/// the edge weights of the co-firing graph (not the negated Laplacian
/// off-diagonals), so `(L·x)[i] = deg[i]·x[i] − Σ_{j ∈ nbrs(i)}
/// val[j] · x[col[j]]`.
///
/// Public so the Lanczos driver in [`super::lanczos`] and tests can
/// drive it directly without going through the co-firing-window
/// accumulation path.
pub struct CsrLaplacian {
    n: usize,
    row_ptr: Vec<u32>,
    col_idx: Vec<u32>,
    val: Vec<f32>,
    deg: Vec<f32>,
}

impl CsrLaplacian {
    /// Number of rows / columns.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of stored non-zero adjacency entries (edges are stored
    /// symmetrically, so `nnz ≈ 2·|E|`).
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }

    /// `y ← L · x` where `L = D − A`. CSR store is the adjacency `A`,
    /// `deg` is the row sum of `A`. Same definition the prior power-
    /// iteration code used; promoted to a method so the Lanczos driver
    /// (and tests) can share it.
    pub fn mat_vec_l(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.n);
        debug_assert_eq!(y.len(), self.n);
        for i in 0..self.n {
            let s = self.row_ptr[i] as usize;
            let e = self.row_ptr[i + 1] as usize;
            let mut acc = self.deg[i] * x[i];
            for k in s..e {
                let j = self.col_idx[k] as usize;
                acc -= self.val[k] * x[j];
            }
            y[i] = acc;
        }
    }

    /// Row-pointer accessor for the Lanczos driver's Gershgorin
    /// upper-bound computation. `i ∈ [0, n]`.
    #[inline]
    pub(super) fn row_ptr_i(&self, i: usize) -> u32 {
        self.row_ptr[i]
    }

    /// CSR value accessor (indexed by nonzero slot). `k ∈ [0, nnz)`.
    #[inline]
    pub(super) fn val_k(&self, k: usize) -> f32 {
        self.val[k]
    }

    /// Degree at row `i`. `i ∈ [0, n)`.
    #[inline]
    pub(super) fn deg_i(&self, i: usize) -> f32 {
        self.deg[i]
    }

    /// Build a `CsrLaplacian` directly from a list of undirected
    /// weighted edges. Intended for tests / fixtures — production
    /// callers go through [`sparse_fiedler`] which accumulates from a
    /// co-firing spike window.
    ///
    /// Edges with `u == v` are skipped. Duplicates are summed. Each
    /// edge contributes symmetrically (row u ↔ row v).
    pub fn from_edges(n: usize, edges: &[(u32, u32, f32)]) -> Option<Self> {
        if n < 2 {
            return None;
        }
        let mut acc: HashMap<(u32, u32), f32> = HashMap::with_capacity(edges.len());
        for &(u, v, w) in edges {
            if u == v || (u as usize) >= n || (v as usize) >= n {
                continue;
            }
            let key = if u < v { (u, v) } else { (v, u) };
            *acc.entry(key).or_insert(0.0) += w;
        }
        if acc.is_empty() {
            return None;
        }
        let mut graph = SparseGraph::with_capacity(n);
        for (&(u, v), &w) in &acc {
            let _ = graph.insert_or_update_edge(u as usize, v as usize, w as f64);
        }
        Some(csr_from_graph(graph, n))
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
) -> Option<CsrLaplacian> {
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

    Some(csr_from_graph(graph, n))
}

/// Convert a `SparseGraph` to our `CsrLaplacian` representation,
/// padding empty trailing rows if the graph's internal vertex count is
/// less than `n`.
fn csr_from_graph(graph: SparseGraph, n: usize) -> CsrLaplacian {
    let (rp_f64, ci_f64, vals_f64, exported_n) = graph.to_csr();
    let mut row_ptr: Vec<u32> = rp_f64.iter().map(|x| *x as u32).collect();
    if exported_n < n {
        let last = *row_ptr.last().unwrap_or(&0);
        row_ptr.resize(n + 1, last);
    }
    let col_idx: Vec<u32> = ci_f64.iter().map(|x| *x as u32).collect();
    let val: Vec<f32> = vals_f64.iter().map(|x| *x as f32).collect();

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

    CsrLaplacian {
        n,
        row_ptr,
        col_idx,
        val,
        deg,
    }
}

// ---------------------------------------------------------------------
// Expose the CSR Laplacian extent + builder for tests / diagnostics.
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
    Some((csr.n(), csr.nnz()))
}

/// Build the same `CsrLaplacian` the sparse-Fiedler path would build
/// for the given active-neuron set and co-firing window. Exposed for
/// tests that want to drive the Lanczos eigensolver directly without
/// re-implementing the accumulation path.
pub fn build_csr_from_window(
    active: &[NeuronId],
    cofire: &VecDeque<Spike>,
) -> Option<CsrLaplacian> {
    let n = active.len();
    build_sparse_laplacian(active, cofire, n)
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
