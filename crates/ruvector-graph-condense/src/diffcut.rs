//! Differentiable (relaxed) min-cut loss and a gradient-descent condenser.
//!
//! The piece the 2024–2026 graph-condensation surveys flag as genuinely
//! unpublished: a **differentiable min-cut / normalized-cut objective used as the
//! condensation mechanism** (spectral terms like SGDD's LED and GDEM's
//! eigenbasis exist; an explicit relaxed-min-cut loss does not).
//!
//! ## The objective (after Bianchi et al., MinCutPool 2020)
//!
//! For a soft assignment `S ∈ R^{N×K}` (row-softmax of logits), weighted
//! adjacency `A`, degree matrix `D = diag(A·1)`:
//! `L_cut = -Tr(SᵀAS)/Tr(SᵀDS) ∈ [-1,0]` (relaxed normalized cut),
//! `L_ortho = ‖SᵀS/‖SᵀS‖_F − I_K/√K‖_F ∈ [0,2]` (anti-collapse), and
//! `L = L_cut + λ·L_ortho`. Logits are optimised by gradient descent with
//! **analytic gradients** (all maths in `f64`, no autodiff dependency), verified
//! against finite differences in the test module. Hardening the trained
//! assignment (argmax) yields the regions consumed by [`crate::condense`] via
//! [`crate::CondenseMethod::DiffMinCut`].

use crate::error::{CondenseError, Result};
use ruvector_mincut::{DynamicGraph, VertexId};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

const EPS: f64 = 1e-12;

/// Configuration for the differentiable min-cut condenser.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffCutConfig {
    /// Number of clusters `K` (upper bound on condensed super-nodes).
    pub num_clusters: usize,
    /// Weight `λ` on the orthogonality (anti-collapse) term.
    pub ortho_weight: f64,
    /// Gradient-descent step size.
    pub learning_rate: f64,
    /// Number of gradient-descent iterations.
    pub iterations: usize,
    /// RNG seed for logit initialisation (determinism).
    pub seed: u64,
}

impl Default for DiffCutConfig {
    fn default() -> Self {
        Self {
            num_clusters: 8,
            ortho_weight: 1.0,
            learning_rate: 0.3,
            iterations: 300,
            seed: 0x0D1F_FC07,
        }
    }
}

impl DiffCutConfig {
    fn validate(&self) -> Result<()> {
        if self.num_clusters == 0 {
            return Err(CondenseError::InvalidConfig(
                "num_clusters must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

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

/// Result of training: the learned assignment plus provenance.
#[derive(Debug, Clone)]
pub struct DiffCutResult {
    /// Row-softmax assignment matrix, row-major `N×K`.
    soft: Vec<f64>,
    /// Graph vertices in row order (sorted ascending for determinism).
    vertices: Vec<VertexId>,
    k: usize,
    /// Loss at the final iteration.
    loss: MinCutLoss,
}

impl DiffCutResult {
    /// Number of clusters `K`.
    pub fn num_clusters(&self) -> usize {
        self.k
    }

    /// Final loss.
    pub fn loss(&self) -> MinCutLoss {
        self.loss
    }

    /// Borrow the soft assignment matrix (row-major `N×K`).
    pub fn soft_assignment(&self) -> &[f64] {
        &self.soft
    }

    /// Hard regions: group vertices by argmax cluster. Empty clusters are
    /// dropped; every vertex is assigned exactly once.
    pub fn hard_regions(&self) -> Vec<Vec<VertexId>> {
        let n = self.vertices.len();
        let mut buckets: HashMap<usize, Vec<VertexId>> = HashMap::new();
        for i in 0..n {
            let row = &self.soft[i * self.k..(i + 1) * self.k];
            let mut best = 0usize;
            let mut best_v = row[0];
            for (c, &v) in row.iter().enumerate().skip(1) {
                if v > best_v {
                    best_v = v;
                    best = c;
                }
            }
            buckets.entry(best).or_default().push(self.vertices[i]);
        }
        buckets.into_values().collect()
    }
}

/// Trainable differentiable min-cut condenser.
#[derive(Debug, Clone)]
pub struct DiffCutCondenser {
    config: DiffCutConfig,
}

impl DiffCutCondenser {
    /// Create a condenser with the given configuration.
    pub fn new(config: DiffCutConfig) -> Self {
        Self { config }
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &DiffCutConfig {
        &self.config
    }

    /// Train the soft assignment by gradient descent on the min-cut loss.
    ///
    /// # Errors
    /// [`CondenseError::EmptyGraph`] for a graph with no vertices, or
    /// [`CondenseError::InvalidConfig`] for `num_clusters == 0`.
    pub fn train(&self, graph: &DynamicGraph) -> Result<DiffCutResult> {
        self.config.validate()?;
        let g = CompactGraph::from_graph(graph);
        if g.n == 0 {
            return Err(CondenseError::EmptyGraph);
        }
        let n = g.n;
        let k = self.config.num_clusters;

        // Initialise logits with small random noise to break row symmetry.
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut theta = vec![0f64; n * k];
        for t in &mut theta {
            *t = rng.gen_range(-0.1..0.1);
        }

        for _ in 0..self.config.iterations {
            let soft = softmax_rows(&theta, n, k);
            let (_, grad_s) = loss_and_grad_wrt_soft(&g, &soft, k, self.config.ortho_weight);
            let grad_theta = softmax_backprop(&soft, &grad_s, n, k);
            for idx in 0..n * k {
                theta[idx] -= self.config.learning_rate * grad_theta[idx];
            }
        }
        // Final assignment and loss from the converged logits.
        let soft = softmax_rows(&theta, n, k);
        let loss = forward(&g, &soft, k, self.config.ortho_weight);

        Ok(DiffCutResult {
            soft,
            vertices: g.vertices,
            k,
            loss,
        })
    }
}

/// Evaluate the min-cut loss for an arbitrary soft assignment (row-major `N×K`,
/// rows in ascending-vertex order). Useful as a quality metric for any
/// assignment, learned or hand-built.
///
/// # Errors
/// [`CondenseError::DimensionMismatch`] if `soft.len() != N*k`.
pub fn min_cut_loss(
    graph: &DynamicGraph,
    soft: &[f64],
    k: usize,
    ortho_weight: f64,
) -> Result<MinCutLoss> {
    let g = CompactGraph::from_graph(graph);
    if soft.len() != g.n * k {
        return Err(CondenseError::DimensionMismatch {
            expected: g.n * k,
            got: soft.len(),
        });
    }
    Ok(forward(&g, soft, k, ortho_weight))
}

// ---------------------------------------------------------------------------
// Internal compact graph + maths (all f64).
// ---------------------------------------------------------------------------

struct CompactGraph {
    n: usize,
    degree: Vec<f64>,
    edges: Vec<(usize, usize, f64)>,
    vertices: Vec<VertexId>,
}

impl CompactGraph {
    fn from_graph(graph: &DynamicGraph) -> Self {
        let mut vertices = graph.vertices();
        vertices.sort_unstable(); // deterministic row order
        let mut index: HashMap<VertexId, usize> = HashMap::with_capacity(vertices.len());
        for (i, &v) in vertices.iter().enumerate() {
            index.insert(v, i);
        }
        let n = vertices.len();
        let mut degree = vec![0f64; n];
        let mut edges = Vec::with_capacity(graph.num_edges());
        for e in graph.edges() {
            let i = index[&e.source];
            let j = index[&e.target];
            let w = e.weight;
            edges.push((i, j, w));
            degree[i] += w;
            degree[j] += w;
        }
        Self {
            n,
            degree,
            edges,
            vertices,
        }
    }
}

fn softmax_rows(logits: &[f64], n: usize, k: usize) -> Vec<f64> {
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

/// Forward-only loss.
fn forward(g: &CompactGraph, s: &[f64], k: usize, lambda: f64) -> MinCutLoss {
    let (cut, _, ortho, _) = cut_and_ortho(g, s, k, /*want_grad=*/ false);
    MinCutLoss {
        cut,
        ortho,
        total: cut + lambda * ortho,
    }
}

/// Loss and gradient w.r.t. the soft assignment `S`.
fn loss_and_grad_wrt_soft(
    g: &CompactGraph,
    s: &[f64],
    k: usize,
    lambda: f64,
) -> (MinCutLoss, Vec<f64>) {
    let (cut, grad_cut, ortho, grad_ortho) = cut_and_ortho(g, s, k, true);
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

/// Shared core: returns (cut, grad_cut_wrt_S, ortho, grad_ortho_wrt_S).
/// When `want_grad` is false the gradient vectors are empty.
fn cut_and_ortho(
    g: &CompactGraph,
    s: &[f64],
    k: usize,
    want_grad: bool,
) -> (f64, Vec<f64>, f64, Vec<f64>) {
    let n = g.n;

    // AS = A · S  (A symmetric, accumulate both directions).
    let mut as_mat = vec![0f64; n * k];
    for &(i, j, w) in &g.edges {
        for c in 0..k {
            as_mat[i * k + c] += w * s[j * k + c];
            as_mat[j * k + c] += w * s[i * k + c];
        }
    }

    // numer = Tr(SᵀAS), denom = Tr(SᵀDS).
    let mut numer = 0f64;
    for idx in 0..n * k {
        numer += s[idx] * as_mat[idx];
    }
    let mut denom = 0f64;
    for i in 0..n {
        let di = g.degree[i];
        let mut s2 = 0f64;
        for c in 0..k {
            let v = s[i * k + c];
            s2 += v * v;
        }
        denom += di * s2;
    }
    let cut = if denom > EPS { -numer / denom } else { 0.0 };

    let mut grad_cut = Vec::new();
    if want_grad {
        grad_cut = vec![0f64; n * k];
        if denom > EPS {
            // ∂L_cut/∂S = -2/denom · (AS + L_cut·DS)
            let coef = -2.0 / denom;
            for i in 0..n {
                let di = g.degree[i];
                for c in 0..k {
                    let ds = di * s[i * k + c];
                    grad_cut[i * k + c] = coef * (as_mat[i * k + c] + cut * ds);
                }
            }
        }
    }

    // P = SᵀS  (K×K).
    let mut p = vec![0f64; k * k];
    for i in 0..n {
        for a in 0..k {
            let sa = s[i * k + a];
            if sa != 0.0 {
                for b in 0..k {
                    p[a * k + b] += sa * s[i * k + b];
                }
            }
        }
    }
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
            for i in 0..n {
                for kk in 0..k {
                    let mut acc = 0f64;
                    for b in 0..k {
                        acc += s[i * k + b] * gp[b * k + kk];
                    }
                    grad_ortho[i * k + kk] = 2.0 * acc;
                }
            }
        }
    }

    (cut, grad_cut, ortho, grad_ortho)
}

/// Backprop a gradient w.r.t. `S` through the row-softmax to the logits `Θ`.
fn softmax_backprop(s: &[f64], grad_s: &[f64], n: usize, k: usize) -> Vec<f64> {
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
        // Two triangles joined by a weak bridge — the cleanest cut.
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
        // The decisive test: analytic ∂L/∂Θ vs central finite differences.
        let g = CompactGraph::from_graph(&barbell());
        let n = g.n;
        let k = 2;
        let lambda = 1.0;

        let mut rng = StdRng::seed_from_u64(99);
        let mut theta = vec![0f64; n * k];
        for t in &mut theta {
            *t = rng.gen_range(-0.5..0.5);
        }

        // Analytic gradient w.r.t. theta.
        let s = softmax_rows(&theta, n, k);
        let (_, grad_s) = loss_and_grad_wrt_soft(&g, &s, k, lambda);
        let analytic = softmax_backprop(&s, &grad_s, n, k);

        // Central finite differences of `total` w.r.t. each theta entry.
        let h = 1e-6;
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
        assert!(
            max_abs_err < 1e-5,
            "analytic vs numeric grad mismatch: {max_abs_err}"
        );
    }

    #[test]
    fn min_cut_loss_evaluates_uniform_assignment() {
        // Uniform soft assignment: every node split 50/50 over 2 clusters.
        let g = barbell();
        let n = CompactGraph::from_graph(&g).n;
        let soft = vec![0.5f64; n * 2];
        let l = forward(&CompactGraph::from_graph(&g), &soft, 2, 1.0);
        // A uniform assignment makes numer == denom, so the cut term hits its
        // best value (-1) — it is "fooled". The orthogonality term is exactly
        // what catches this collapse (SᵀS is far from identity), so it is large.
        assert!((l.cut + 1.0).abs() < 1e-9, "cut {}", l.cut);
        assert!(l.ortho > 0.5, "ortho {}", l.ortho);
        assert!((l.total - (l.cut + l.ortho)).abs() < 1e-12);
    }
}
