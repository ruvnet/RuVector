//! Trainable differentiable min-cut condenser — the relaxed normalized-cut
//! objective (MinCutPool-style; loss + analytic gradients live in
//! [`crate::cutloss`]) optimised into a cluster assignment.
//!
//! The 2024–2026 surveys flag a differentiable min-cut term in the condensation
//! loss as unpublished. This module makes that objective practical **on large-K
//! problems** with three standard-but-essential ingredients:
//!
//! - **Adam** (default) instead of plain GD — adaptive, robust on the
//!   ill-conditioned, non-convex cut objective.
//! - **Warm-start init** (default) — seed the logits from the cheap
//!   [`crate::CondenseMethod::WeakBoundary`] structural prior and *refine* with
//!   the differentiable objective, rather than descending from random noise.
//!   This is the same coreset/K-Center idea GCond/SFGC use, and it is what makes
//!   K ≫ 2 converge.
//! - **Restarts** — keep the lowest-loss run.
//!
//! Hardening the trained assignment (argmax) yields the regions consumed by
//! [`crate::condense`] via [`crate::CondenseMethod::DiffMinCut`].

use crate::cutloss::{
    as_matrix, as_matrix_minibatch, forward, loss_and_grad_with_as, softmax_backprop, softmax_rows,
    CompactGraph,
};
use crate::error::{CondenseError, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_mincut::{DynamicGraph, VertexId};
use std::collections::HashMap;

pub use crate::cutloss::MinCutLoss;

/// First-order optimiser used to minimise the loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Optimizer {
    /// (Heavy-ball) stochastic gradient descent. `momentum = 0` is plain GD.
    Sgd {
        /// Momentum coefficient in `[0, 1)`.
        momentum: f64,
    },
    /// Adam — adaptive moments; far more robust for large `K`.
    Adam {
        /// First-moment decay (typ. 0.9).
        beta1: f64,
        /// Second-moment decay (typ. 0.999).
        beta2: f64,
        /// Numerical-stability epsilon (typ. 1e-8).
        epsilon: f64,
    },
}

impl Default for Optimizer {
    fn default() -> Self {
        Optimizer::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// How the cluster logits are initialised before optimisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InitStrategy {
    /// Unit-scale random logits.
    Random,
    /// **Default.** Seed from the [`crate::CondenseMethod::WeakBoundary`]
    /// structural prior, then refine — the key to large-K convergence.
    #[default]
    WarmStart,
}

/// Configuration for the differentiable min-cut condenser. `Default` is a
/// large-K-ready setup: Adam + warm-start.
#[derive(Debug, Clone, PartialEq)]
pub struct DiffCutConfig {
    /// Number of clusters `K` (upper bound on condensed super-nodes).
    pub num_clusters: usize,
    /// Weight `λ` on the orthogonality (anti-collapse) term.
    pub ortho_weight: f64,
    /// Optimiser step size (Adam likes ~0.05; SGD ~0.3).
    pub learning_rate: f64,
    /// Number of optimisation iterations per restart.
    pub iterations: usize,
    /// Optimiser.
    pub optimizer: Optimizer,
    /// Logit initialisation strategy.
    pub init: InitStrategy,
    /// Number of independent restarts; the lowest-loss run wins (min 1).
    pub restarts: usize,
    /// Early-stop when the loss improves by less than this between iterations
    /// (`0.0` disables). Warm-start starts near the optimum, so this typically
    /// cuts most of `iterations`.
    pub tolerance: f64,
    /// Use Rayon to parallelise the per-iteration `A·S` and parameter update.
    /// Deterministic (row-parallel); pays off on large graphs, adds overhead on
    /// tiny ones, so it defaults to `false`.
    pub parallel: bool,
    /// If `Some(b)`, estimate the gradient from `b` randomly sampled edges per
    /// iteration (stochastic) instead of the full edge set — the lever for
    /// million-edge graphs. `None` = full batch (exact).
    pub minibatch_edges: Option<usize>,
    /// RNG seed (determinism).
    pub seed: u64,
}

impl Default for DiffCutConfig {
    fn default() -> Self {
        Self {
            num_clusters: 8,
            ortho_weight: 1.0,
            learning_rate: 0.05,
            iterations: 300,
            optimizer: Optimizer::default(),
            init: InitStrategy::default(),
            restarts: 1,
            tolerance: 1e-6,
            parallel: false,
            minibatch_edges: None,
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

/// Result of training: the learned assignment plus provenance.
#[derive(Debug, Clone)]
pub struct DiffCutResult {
    soft: Vec<f64>,
    vertices: Vec<VertexId>,
    k: usize,
    loss: MinCutLoss,
    iterations_run: usize,
}

impl DiffCutResult {
    /// Number of clusters `K`.
    pub fn num_clusters(&self) -> usize {
        self.k
    }

    /// Final (best-restart) loss.
    pub fn loss(&self) -> MinCutLoss {
        self.loss
    }

    /// Iterations actually run in the best restart (≤ `iterations`; lower when
    /// early-stopping triggered).
    pub fn iterations_run(&self) -> usize {
        self.iterations_run
    }

    /// Borrow the soft assignment matrix (row-major `N×K`).
    pub fn soft_assignment(&self) -> &[f64] {
        &self.soft
    }

    /// Hard regions: group vertices by argmax cluster. Empty clusters are
    /// dropped; every vertex is assigned exactly once.
    pub fn hard_regions(&self) -> Vec<Vec<VertexId>> {
        let mut buckets: HashMap<usize, Vec<VertexId>> = HashMap::new();
        for i in 0..self.vertices.len() {
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

    /// Train the soft assignment by minimising the min-cut loss.
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
        let (n, k) = (g.n, self.config.num_clusters);
        let restarts = self.config.restarts.max(1);

        let mut best: Option<(Vec<f64>, MinCutLoss, usize)> = None;
        for r in 0..restarts {
            let seed = self
                .config
                .seed
                .wrapping_add((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut rng = StdRng::seed_from_u64(seed);
            let mut theta = match self.config.init {
                InitStrategy::Random => random_logits(n, k, &mut rng),
                InitStrategy::WarmStart => warm_start_logits(&g, graph, k, &mut rng),
            };
            let iters = self.optimize(&g, &mut theta, n, k, &mut rng);
            let soft = softmax_rows(&theta, n, k);
            let loss = forward(&g, &soft, k, self.config.ortho_weight);
            if best.as_ref().map_or(true, |(_, b, _)| loss.total < b.total) {
                best = Some((soft, loss, iters));
            }
        }

        let (soft, loss, iterations_run) = best.expect("restarts >= 1");
        Ok(DiffCutResult {
            soft,
            vertices: g.vertices,
            k,
            loss,
            iterations_run,
        })
    }

    /// Run the configured optimiser in place on `theta`; returns the number of
    /// iterations actually performed (early-stops on loss convergence). `rng`
    /// drives edge-minibatch sampling when enabled.
    fn optimize(
        &self,
        g: &CompactGraph,
        theta: &mut [f64],
        n: usize,
        k: usize,
        rng: &mut StdRng,
    ) -> usize {
        let lr = self.config.learning_rate;
        let lambda = self.config.ortho_weight;
        let tol = self.config.tolerance;
        let parallel = self.config.parallel;
        let nnz = g.edges.len();
        let minibatch = self.config.minibatch_edges.filter(|_| nnz > 0);
        let mut prev = f64::INFINITY;
        let mut vel = vec![0f64; n * k];
        let mut m = vec![0f64; n * k];
        let mut v = vec![0f64; n * k];
        let mut iters_run = 0;

        for t in 1..=self.config.iterations {
            let soft = softmax_rows(theta, n, k);
            // A·S: full (parallel optional) or a stochastic edge minibatch.
            let as_mat = match minibatch {
                Some(b) => {
                    let b = b.min(nnz);
                    let sample: Vec<usize> = (0..b).map(|_| rng.gen_range(0..nnz)).collect();
                    as_matrix_minibatch(g, &soft, n, k, &sample)
                }
                None => as_matrix(g, &soft, n, k, parallel),
            };
            // loss_and_grad gives the loss at the *current* theta for free.
            let (loss, grad_s) = loss_and_grad_with_as(g, &soft, &as_mat, k, lambda, parallel);
            let gt = softmax_backprop(&soft, &grad_s, n, k);

            match self.config.optimizer {
                Optimizer::Sgd { momentum } => {
                    for idx in 0..n * k {
                        vel[idx] = momentum * vel[idx] - lr * gt[idx];
                        theta[idx] += vel[idx];
                    }
                }
                Optimizer::Adam {
                    beta1,
                    beta2,
                    epsilon,
                } => {
                    let bc1 = 1.0 - beta1.powi(t as i32);
                    let bc2 = 1.0 - beta2.powi(t as i32);
                    for idx in 0..n * k {
                        m[idx] = beta1 * m[idx] + (1.0 - beta1) * gt[idx];
                        v[idx] = beta2 * v[idx] + (1.0 - beta2) * gt[idx] * gt[idx];
                        let mhat = m[idx] / bc1;
                        let vhat = v[idx] / bc2;
                        theta[idx] -= lr * mhat / (vhat.sqrt() + epsilon);
                    }
                }
            }

            iters_run = t;
            if tol > 0.0 && (prev - loss.total).abs() < tol {
                break;
            }
            prev = loss.total;
        }
        iters_run
    }
}

/// Unit-scale random logits.
fn random_logits(n: usize, k: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut theta = vec![0f64; n * k];
    for t in &mut theta {
        *t = rng.gen_range(-1.0..1.0);
    }
    theta
}

/// Warm-start logits from the WeakBoundary structural prior: each detected
/// region is mapped to a cluster (largest regions get their own; overflow is
/// distributed round-robin) and biased into the logits, plus small noise.
fn warm_start_logits(
    g: &CompactGraph,
    graph: &DynamicGraph,
    k: usize,
    rng: &mut StdRng,
) -> Vec<f64> {
    const BIAS: f64 = 4.0; // softmax(4 vs 0) ~ 0.98 mass on the seeded cluster
    let index = g.index_map();

    let mut regions = crate::regions::weak_boundary_regions(graph, 0.5);
    // If the structural prior found no usable split (e.g. an unweighted graph,
    // where WeakBoundary collapses to one component), warm-start would seed every
    // node into one cluster and the optimiser would stay collapsed. Fall back to
    // random init and let the min-cut objective do the partitioning.
    if regions.len() < 2 {
        return random_logits(g.n, k, rng);
    }
    // Deterministic order (weak_boundary_regions yields HashMap order): largest
    // first, ties broken by smallest member id.
    regions.sort_by(|a, b| {
        b.len()
            .cmp(&a.len())
            .then_with(|| a.iter().min().cmp(&b.iter().min()))
    });

    let mut cluster_of = vec![0usize; g.n];
    for (ri, region) in regions.iter().enumerate() {
        let cluster = if ri < k { ri } else { ri % k };
        for v in region {
            if let Some(&row) = index.get(v) {
                cluster_of[row] = cluster;
            }
        }
    }

    let mut theta = vec![0f64; g.n * k];
    for row in 0..g.n {
        for c in 0..k {
            theta[row * k + c] = rng.gen_range(-0.1..0.1);
        }
        theta[row * k + cluster_of[row]] += BIAS;
    }
    theta
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
    fn warm_start_seeds_a_good_partition() {
        // Warm start alone (0 iterations) should already encode the 2 triangles.
        let g = barbell();
        let res = DiffCutCondenser::new(DiffCutConfig {
            num_clusters: 2,
            iterations: 0,
            ..Default::default()
        })
        .train(&g)
        .unwrap();
        let mut regions = res.hard_regions();
        for r in &mut regions {
            r.sort_unstable();
        }
        regions.sort_by_key(|r| r[0]);
        assert_eq!(regions, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn adam_refines_to_low_cut() {
        let g = barbell();
        let res = DiffCutCondenser::new(DiffCutConfig {
            num_clusters: 2,
            ..Default::default()
        })
        .train(&g)
        .unwrap();
        assert!(res.loss().cut < -0.9, "cut {}", res.loss().cut);
    }
}
