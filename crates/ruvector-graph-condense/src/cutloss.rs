//! Differentiable relaxed-min-cut loss core (all maths in `f64`).
//!
//! Pure functions shared by [`crate::diffcut`]: the compact graph view, the
//! row-softmax, the loss (`L_cut + λ·L_ortho`) and its analytic gradients. Kept
//! separate from the optimiser/orchestration so each file stays small and the
//! gradient-checked maths is isolated.

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use ruvector_mincut::{DynamicGraph, VertexId};
use std::collections::HashMap;

pub(crate) const EPS: f64 = 1e-12;

/// The three components of the loss at a point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinCutLoss {
    /// Relaxed normalized-cut term in `[-1, 0]` (lower is better).
    pub cut: f64,
    /// Orthogonality / balance term in `[0, 2]` (lower is better).
    pub ortho: f64,
    /// `cut + λ·ortho`.
    pub total: f64,
}

/// Contiguous, index-mapped view of a graph for the loss maths.
///
/// Carries both an edge list (for minibatch scatter) and a CSR adjacency (for
/// conflict-free, row-parallel `A·S`).
pub(crate) struct CompactGraph {
    pub(crate) n: usize,
    pub(crate) degree: Vec<f64>,
    pub(crate) edges: Vec<(usize, usize, f64)>,
    pub(crate) vertices: Vec<VertexId>,
    /// CSR row offsets, length `n + 1`.
    nbr_off: Vec<usize>,
    /// CSR neighbours `(col, weight)`, length `2 * num_edges`.
    nbr: Vec<(usize, f64)>,
}

impl CompactGraph {
    pub(crate) fn from_graph(graph: &DynamicGraph) -> Self {
        let mut vertices = graph.vertices();
        vertices.sort_unstable(); // deterministic row order
        let mut index: HashMap<VertexId, usize> = HashMap::with_capacity(vertices.len());
        for (i, &v) in vertices.iter().enumerate() {
            index.insert(v, i);
        }
        let n = vertices.len();
        let mut degree = vec![0f64; n];
        let mut edges = Vec::with_capacity(graph.num_edges());
        let mut deg_count = vec![0usize; n];
        for e in graph.edges() {
            let i = index[&e.source];
            let j = index[&e.target];
            edges.push((i, j, e.weight));
            degree[i] += e.weight;
            degree[j] += e.weight;
            deg_count[i] += 1;
            deg_count[j] += 1;
        }
        // Build CSR (both directions) from the edge list.
        let mut nbr_off = vec![0usize; n + 1];
        for i in 0..n {
            nbr_off[i + 1] = nbr_off[i] + deg_count[i];
        }
        let mut cursor = nbr_off[..n].to_vec();
        let mut nbr = vec![(0usize, 0f64); edges.len() * 2];
        for &(i, j, w) in &edges {
            nbr[cursor[i]] = (j, w);
            cursor[i] += 1;
            nbr[cursor[j]] = (i, w);
            cursor[j] += 1;
        }
        Self {
            n,
            degree,
            edges,
            vertices,
            nbr_off,
            nbr,
        }
    }

    /// Vertex-id → row-index map (rows are sorted-ascending vertices).
    pub(crate) fn index_map(&self) -> HashMap<VertexId, usize> {
        self.vertices
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i))
            .collect()
    }
}

pub(crate) fn softmax_rows(logits: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut s = vec![0f64; n * k];
    for i in 0..n {
        let row = &logits[i * k..(i + 1) * k];
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0f64;
        for c in 0..k {
            let e = (row[c] - max).exp();
            s[i * k + c] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for c in 0..k {
            s[i * k + c] *= inv;
        }
    }
    s
}

/// `A · S` (`N×K`) via CSR — each output row depends only on its node's
/// neighbours, so it is conflict-free and row-parallel. Deterministic
/// regardless of thread count (fixed row + neighbour order).
pub(crate) fn as_matrix(
    g: &CompactGraph,
    s: &[f64],
    n: usize,
    k: usize,
    parallel: bool,
) -> Vec<f64> {
    let mut as_mat = vec![0f64; n * k];
    let row_fn = |i: usize, row: &mut [f64]| {
        for idx in g.nbr_off[i]..g.nbr_off[i + 1] {
            let (j, w) = g.nbr[idx];
            let sj = &s[j * k..(j + 1) * k];
            for c in 0..k {
                row[c] += w * sj[c];
            }
        }
    };
    #[cfg(feature = "parallel")]
    if parallel {
        as_mat
            .par_chunks_mut(k)
            .enumerate()
            .for_each(|(i, row)| row_fn(i, row));
        return as_mat;
    }
    let _ = parallel;
    as_mat
        .chunks_mut(k)
        .enumerate()
        .for_each(|(i, row)| row_fn(i, row));
    as_mat
}

/// Stochastic `A · S` estimate from a sampled subset of edges, scaled by
/// `|E| / |sample|`. O(|sample|·K) per call — the lever for million-edge graphs.
pub(crate) fn as_matrix_minibatch(
    g: &CompactGraph,
    s: &[f64],
    n: usize,
    k: usize,
    sample: &[usize],
) -> Vec<f64> {
    let mut as_mat = vec![0f64; n * k];
    if sample.is_empty() {
        return as_mat;
    }
    let scale = g.edges.len() as f64 / sample.len() as f64;
    for &e in sample {
        let (i, j, w) = g.edges[e];
        let ws = w * scale;
        for c in 0..k {
            as_mat[i * k + c] += ws * s[j * k + c];
            as_mat[j * k + c] += ws * s[i * k + c];
        }
    }
    as_mat
}

/// Forward-only loss (full-batch, sequential).
pub(crate) fn forward(g: &CompactGraph, s: &[f64], k: usize, lambda: f64) -> MinCutLoss {
    let as_mat = as_matrix(g, s, g.n, k, false);
    let (cut, _, ortho, _) = cut_and_ortho(g, s, &as_mat, k, false, false);
    MinCutLoss {
        cut,
        ortho,
        total: cut + lambda * ortho,
    }
}

/// Loss and gradient w.r.t. `S` (full-batch, sequential) — convenience used by
/// the gradient-check test.
#[cfg(test)]
pub(crate) fn loss_and_grad_wrt_soft(
    g: &CompactGraph,
    s: &[f64],
    k: usize,
    lambda: f64,
) -> (MinCutLoss, Vec<f64>) {
    let as_mat = as_matrix(g, s, g.n, k, false);
    loss_and_grad_with_as(g, s, &as_mat, k, lambda, false)
}

/// Loss and gradient given a precomputed `A·S`. `parallel` parallelises the
/// heavy `O(N·K²)` loops (SᵀS build, ortho gradient) deterministically.
pub(crate) fn loss_and_grad_with_as(
    g: &CompactGraph,
    s: &[f64],
    as_mat: &[f64],
    k: usize,
    lambda: f64,
    parallel: bool,
) -> (MinCutLoss, Vec<f64>) {
    let (cut, grad_cut, ortho, grad_ortho) = cut_and_ortho(g, s, as_mat, k, true, parallel);
    let n = g.n;
    let mut grad = grad_cut;
    for idx in 0..n * k {
        grad[idx] += lambda * grad_ortho[idx];
    }
    (
        MinCutLoss {
            cut,
            ortho,
            total: cut + lambda * ortho,
        },
        grad,
    )
}

/// Rows per Rayon task — coarse enough to amortise dispatch overhead.
fn rows_per_task(n: usize) -> usize {
    #[cfg(feature = "parallel")]
    let threads = rayon::current_num_threads();
    #[cfg(not(feature = "parallel"))]
    let threads = 1usize;
    (n / (threads * 4)).max(1)
}

/// `P = SᵀS` (`K×K`). Both paths use the *same* chunked partial-sum ordering
/// (parallel only changes who computes each chunk), so parallel is bit-identical
/// to sequential — no float-reordering surprises.
fn gram(s: &[f64], n: usize, k: usize, parallel: bool) -> Vec<f64> {
    let chunk = rows_per_task(n) * k;
    let acc_block = |block: &[f64]| -> Vec<f64> {
        let mut local = vec![0f64; k * k];
        for row in block.chunks(k) {
            for a in 0..k {
                let sa = row[a];
                if sa != 0.0 {
                    for b in 0..k {
                        local[a * k + b] += sa * row[b];
                    }
                }
            }
        }
        local
    };
    #[cfg(feature = "parallel")]
    let partials: Vec<Vec<f64>> = if parallel {
        s.par_chunks(chunk).map(acc_block).collect()
    } else {
        s.chunks(chunk).map(acc_block).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let partials: Vec<Vec<f64>> = {
        let _ = parallel;
        s.chunks(chunk).map(acc_block).collect()
    };
    let mut p = vec![0f64; k * k];
    for part in partials {
        for i in 0..k * k {
            p[i] += part[i];
        }
    }
    p
}

/// Shared core given a precomputed `A·S`: (cut, grad_cut, ortho, grad_ortho).
/// The gradient vectors are empty when `want_grad` is false.
fn cut_and_ortho(
    g: &CompactGraph,
    s: &[f64],
    as_mat: &[f64],
    k: usize,
    want_grad: bool,
    parallel: bool,
) -> (f64, Vec<f64>, f64, Vec<f64>) {
    let n = g.n;

    // numer = Tr(SᵀAS), denom = Tr(SᵀDS)  (O(N·K), kept sequential).
    let mut numer = 0f64;
    for idx in 0..n * k {
        numer += s[idx] * as_mat[idx];
    }
    let mut denom = 0f64;
    for i in 0..n {
        let mut s2 = 0f64;
        for c in 0..k {
            let v = s[i * k + c];
            s2 += v * v;
        }
        denom += g.degree[i] * s2;
    }
    let cut = if denom > EPS { -numer / denom } else { 0.0 };

    let mut grad_cut = Vec::new();
    if want_grad {
        grad_cut = vec![0f64; n * k];
        if denom > EPS {
            // ∂L_cut/∂S = -2/denom · (AS + L_cut·DS); rows are independent.
            let coef = -2.0 / denom;
            let row = |i: usize, gc: &mut [f64]| {
                let di = g.degree[i];
                for c in 0..k {
                    gc[c] = coef * (as_mat[i * k + c] + cut * di * s[i * k + c]);
                }
            };
            #[cfg(feature = "parallel")]
            if parallel {
                grad_cut
                    .par_chunks_mut(k)
                    .enumerate()
                    .for_each(|(i, gc)| row(i, gc));
            } else {
                grad_cut
                    .chunks_mut(k)
                    .enumerate()
                    .for_each(|(i, gc)| row(i, gc));
            }
            #[cfg(not(feature = "parallel"))]
            grad_cut
                .chunks_mut(k)
                .enumerate()
                .for_each(|(i, gc)| row(i, gc));
        }
    }

    let p = gram(s, n, k, parallel);
    let np = p.iter().map(|x| x * x).sum::<f64>().sqrt();
    let inv_sqrt_k = 1.0 / (k as f64).sqrt();

    let mut ortho = 0f64;
    let mut q = vec![0f64; k * k];
    if np > EPS {
        let mut sq = 0f64;
        for a in 0..k {
            for b in 0..k {
                let target = if a == b { inv_sqrt_k } else { 0.0 };
                let qv = p[a * k + b] / np - target;
                q[a * k + b] = qv;
                sq += qv * qv;
            }
        }
        ortho = sq.sqrt();
    }

    let mut grad_ortho = Vec::new();
    if want_grad {
        grad_ortho = vec![0f64; n * k];
        if np > EPS && ortho > EPS {
            // Gf = Q/ortho ; G_P = Gf/np − (⟨Gf,P⟩/np³)·P ; ∂L/∂S = 2·S·G_P
            let mut dot = 0f64;
            for idx in 0..k * k {
                dot += (q[idx] / ortho) * p[idx];
            }
            let np3 = np * np * np;
            let mut gp = vec![0f64; k * k];
            for idx in 0..k * k {
                gp[idx] = (q[idx] / ortho) / np - (dot / np3) * p[idx];
            }
            // ∂L/∂S row i = 2 · S[i] · G_P; rows independent.
            let row = |s_row: &[f64], go: &mut [f64]| {
                for kk in 0..k {
                    let mut acc = 0f64;
                    for b in 0..k {
                        acc += s_row[b] * gp[b * k + kk];
                    }
                    go[kk] = 2.0 * acc;
                }
            };
            #[cfg(feature = "parallel")]
            if parallel {
                grad_ortho
                    .par_chunks_mut(k)
                    .zip(s.par_chunks(k))
                    .for_each(|(go, s_row)| row(s_row, go));
            } else {
                grad_ortho
                    .chunks_mut(k)
                    .zip(s.chunks(k))
                    .for_each(|(go, s_row)| row(s_row, go));
            }
            #[cfg(not(feature = "parallel"))]
            grad_ortho
                .chunks_mut(k)
                .zip(s.chunks(k))
                .for_each(|(go, s_row)| row(s_row, go));
        }
    }

    (cut, grad_cut, ortho, grad_ortho)
}

/// Backprop a gradient w.r.t. `S` through the row-softmax to the logits `Θ`.
pub(crate) fn softmax_backprop(s: &[f64], grad_s: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut grad = vec![0f64; n * k];
    for i in 0..n {
        let mut dot = 0f64;
        for c in 0..k {
            dot += grad_s[i * k + c] * s[i * k + c];
        }
        for c in 0..k {
            grad[i * k + c] = s[i * k + c] * (grad_s[i * k + c] - dot);
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    fn barbell() -> DynamicGraph {
        let g = DynamicGraph::new();
        for &(u, v, w) in &[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 3, 1.0),
            (2, 3, 0.05),
        ] {
            g.insert_edge(u, v, w).unwrap();
        }
        g
    }

    #[test]
    fn gradient_matches_finite_differences() {
        // Decisive correctness test: analytic ∂L/∂Θ vs finite differences across
        // several K (proves the K-general gradient formulas, not just K=2).
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let g = CompactGraph::from_graph(&barbell());
        let n = g.n;
        let lambda = 1.0;
        let h = 1e-6;
        for k in [2usize, 3, 4] {
            let mut rng = StdRng::seed_from_u64(99 + k as u64);
            let mut theta = vec![0f64; n * k];
            for t in &mut theta {
                *t = rng.gen_range(-0.5..0.5);
            }
            let s = softmax_rows(&theta, n, k);
            let (_, grad_s) = loss_and_grad_wrt_soft(&g, &s, k, lambda);
            let analytic = softmax_backprop(&s, &grad_s, n, k);
            let mut max_abs_err = 0f64;
            for idx in 0..n * k {
                let mut tp = theta.clone();
                tp[idx] += h;
                let lp = forward(&g, &softmax_rows(&tp, n, k), k, lambda).total;
                let mut tm = theta.clone();
                tm[idx] -= h;
                let lm = forward(&g, &softmax_rows(&tm, n, k), k, lambda).total;
                let num = (lp - lm) / (2.0 * h);
                max_abs_err = max_abs_err.max((num - analytic[idx]).abs());
            }
            assert!(max_abs_err < 1e-5, "k={k}: grad mismatch {max_abs_err}");
        }
    }

    #[test]
    fn uniform_assignment_fools_cut_but_not_ortho() {
        let g = CompactGraph::from_graph(&barbell());
        let soft = vec![0.5f64; g.n * 2];
        let l = forward(&g, &soft, 2, 1.0);
        // numer==denom -> cut "fooled" to -1; ortho catches the collapse.
        assert!((l.cut + 1.0).abs() < 1e-9, "cut {}", l.cut);
        assert!(l.ortho > 0.5, "ortho {}", l.ortho);
    }
}
