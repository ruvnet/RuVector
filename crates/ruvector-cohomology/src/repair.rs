//! Group-sparse semantic repair (issue #669, Phase 3).
//!
//! Computes a correction field `q` with a group-sparse penalty so that
//! `b − q` admits a globally coherent explanation, at minimum weighted
//! repair cost. Protected edges are immutable unless the policy explicitly
//! authorizes modification.

use crate::affine::{AffineSheafSystem, EdgeField, EdgeWeights};
use crate::solvers::admm::{solve_group_lasso, AdmmConfig, GroupLassoProblem};
use crate::solvers::SolverConfig;
use crate::syndrome::{compute_syndrome, SyndromeConfig, SyndromeResult};
use crate::witness::{quantize, RepairSummary};
use crate::{CohomologyError, EdgeKey, GaugePolicy};
use serde::{Deserialize, Serialize};

/// Per-edge repair cost components (issue #669: business impact, security
/// severity, latency/disruption, reversibility).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct EdgeCost {
    /// Business impact of altering this constraint.
    pub business: f64,
    /// Security severity.
    pub security: f64,
    /// Latency / disruption cost.
    pub latency: f64,
    /// Cost of reversing the change later.
    pub reversal: f64,
}

/// Mixing weights for the combined cost
/// `c_e = α·business + β·security + γ·latency + η·reversal`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostWeights {
    /// α — business weight.
    pub alpha: f64,
    /// β — security weight.
    pub beta: f64,
    /// γ — latency weight.
    pub gamma: f64,
    /// η — reversal weight.
    pub eta: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self { alpha: 1.0, beta: 1.0, gamma: 1.0, eta: 1.0 }
    }
}

impl EdgeCost {
    /// Combined scalar cost under the given mixing weights, floored at zero.
    pub fn combined(&self, w: &CostWeights) -> f64 {
        (w.alpha * self.business + w.beta * self.security + w.gamma * self.latency
            + w.eta * self.reversal)
            .max(0.0)
    }
}

/// Repair policy (issue #669 core interface).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairPolicy {
    /// Group sparsity strength λ.
    pub lambda: f64,
    /// ADMM penalty ρ.
    pub rho: f64,
    /// Cost mixing weights.
    pub cost_weights: CostWeights,
    /// Fallback cost for edges without an explicit [`EdgeCost`].
    pub default_cost: f64,
    /// Outer ADMM iteration cap.
    pub max_iter: usize,
    /// ADMM residual tolerance.
    pub tol: f64,
    /// Explicit authorization to modify protected edges. Without it,
    /// protected edges are hard-constrained to zero correction.
    pub authorize_protected: bool,
    /// Debias the correction after support selection: refit an unpenalized
    /// least-squares correction restricted to the ADMM support, so the
    /// group-lasso shrinkage bias does not leave residual contradiction.
    pub debias: bool,
}

impl Default for RepairPolicy {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            rho: 1.0,
            cost_weights: CostWeights::default(),
            default_cost: 1.0,
            max_iter: 400,
            tol: 1e-9,
            authorize_protected: false,
            debias: true,
        }
    }
}

/// A proposed (not yet applied) repair.
#[derive(Debug, Clone)]
pub struct RepairPlan {
    /// Nonzero correction blocks in canonical edge order.
    pub corrections: Vec<(EdgeKey, Vec<f64>)>,
    /// Repair objective `½‖W^{1/2}(b − q − δx)‖² + λΣc_e‖q_e‖`.
    pub objective: f64,
    /// Syndrome energy before repair.
    pub pre_energy: f64,
    /// Syndrome energy after applying the correction.
    pub post_energy: f64,
    /// ADMM iterations consumed.
    pub iterations: usize,
    /// Whether ADMM converged to tolerance.
    pub converged: bool,
    /// Canonical repair summary (embedded in witnesses).
    pub summary: RepairSummary,
}

/// Propose a group-sparse repair for the system's current contradiction.
///
/// `costs` and `protected` are per-edge in canonical order; pass `None` for
/// uniform defaults.
pub fn propose_repair(
    system: &AffineSheafSystem,
    syndrome: &SyndromeResult,
    costs: Option<&[EdgeCost]>,
    protected: Option<&[bool]>,
    policy: &RepairPolicy,
    syn_cfg: &SyndromeConfig,
) -> Result<RepairPlan, CohomologyError> {
    let op = &system.operator;
    let edge_count = op.edge_count();
    if let Some(c) = costs {
        if c.len() != edge_count {
            return Err(CohomologyError::DimensionMismatch {
                expected: edge_count,
                actual: c.len(),
                context: "repair costs".into(),
            });
        }
    }
    if let Some(p) = protected {
        if p.len() != edge_count {
            return Err(CohomologyError::DimensionMismatch {
                expected: edge_count,
                actual: p.len(),
                context: "protected mask".into(),
            });
        }
    }

    // An all-zero cost would zero the group penalty and degenerate the
    // decoder into plain least squares (every edge "repaired" for free), so
    // unset costs fall back to the policy default.
    let combined_costs: Vec<f64> = (0..edge_count)
        .map(|i| {
            let c = costs
                .map(|c| c[i].combined(&policy.cost_weights))
                .unwrap_or(policy.default_cost);
            if c > 0.0 { c } else { policy.default_cost }
        })
        .collect();
    let effective_protected: Vec<bool> = (0..edge_count)
        .map(|i| !policy.authorize_protected && protected.map(|p| p[i]).unwrap_or(false))
        .collect();

    let edge_blocks: Vec<(usize, usize)> =
        op.edge_blocks().iter().map(|e| (e.offset, e.dim)).collect();
    let apply_delta = |x: &[f64], y: &mut [f64]| op.apply(x, y);
    let apply_delta_t = |y: &[f64], x: &mut [f64]| op.apply_transpose(y, x);
    let problem = GroupLassoProblem {
        apply_delta: &apply_delta,
        apply_delta_t: &apply_delta_t,
        n0: op.dim_c0(),
        n1: op.dim_c1(),
        edge_blocks: &edge_blocks,
        weights: &system.weights.values,
        costs: &combined_costs,
        protected: &effective_protected,
        b: &system.observations.data,
    };
    let admm_cfg = AdmmConfig {
        lambda: policy.lambda,
        rho: policy.rho,
        max_iter: policy.max_iter,
        tol: policy.tol,
        inner: SolverConfig { tol: syn_cfg.solver.tol, max_iter: syn_cfg.solver.max_iter },
    };
    let result = solve_group_lasso(&problem, &admm_cfg);
    let mut q = result.q;
    let mut objective = result.objective;

    if policy.debias {
        // Refit an unpenalized correction restricted to the selected
        // support: alternate q_e ← b_e − (δx)_e on support edges with the
        // least-squares x for b − q, until the correction stabilizes. This
        // removes the group-lasso shrinkage bias so the chosen support
        // cancels its contradiction completely.
        let support: Vec<bool> = op
            .edge_blocks()
            .iter()
            .enumerate()
            .map(|(i, e)| {
                !effective_protected[i]
                    && q[e.offset..e.offset + e.dim].iter().any(|v| *v != 0.0)
            })
            .collect();
        if support.iter().any(|s| *s) {
            let apply_l = |v: &[f64], out: &mut [f64]| {
                let mut mid = vec![0.0; op.dim_c1()];
                op.apply(v, &mut mid);
                for (i, e) in op.edge_blocks().iter().enumerate() {
                    let w = system.weights.values[i];
                    for m in &mut mid[e.offset..e.offset + e.dim] {
                        *m *= w;
                    }
                }
                op.apply_transpose(&mid, out);
            };
            let mut x = result.x.clone();
            let mut dx = vec![0.0; op.dim_c1()];
            for _ in 0..25 {
                // x-step: least squares on b − q.
                let mut weighted = vec![0.0; op.dim_c1()];
                for (wv, (bi, qi)) in weighted
                    .iter_mut()
                    .zip(system.observations.data.iter().zip(q.iter()))
                {
                    *wv = bi - qi;
                }
                for (i, e) in op.edge_blocks().iter().enumerate() {
                    let w = system.weights.values[i];
                    for wv in &mut weighted[e.offset..e.offset + e.dim] {
                        *wv *= w;
                    }
                }
                let mut rhs = vec![0.0; op.dim_c0()];
                op.apply_transpose(&weighted, &mut rhs);
                let (x_new, _) = crate::solvers::cg::solve_psd(
                    apply_l,
                    &rhs,
                    Some(&x),
                    &admm_cfg.inner,
                );
                x = x_new;
                op.apply(&x, &mut dx);
                // q-step: exact residual absorption on the support.
                let mut delta: f64 = 0.0;
                for (i, e) in op.edge_blocks().iter().enumerate() {
                    if !support[i] {
                        continue;
                    }
                    for j in e.offset..e.offset + e.dim {
                        let target = system.observations.data[j] - dx[j];
                        delta = delta.max((q[j] - target).abs());
                        q[j] = target;
                    }
                }
                if delta < policy.tol {
                    break;
                }
            }
            objective = crate::solvers::admm::objective(&problem, policy.lambda, &x, &q);
        }
    }

    // Extract nonzero corrections in canonical order.
    let mut corrections = Vec::new();
    for e in op.edge_blocks() {
        let block = &q[e.offset..e.offset + e.dim];
        if block.iter().any(|v| *v != 0.0) {
            corrections.push((e.key, block.to_vec()));
        }
    }

    // Post-repair syndrome energy on the corrected observations.
    let mut corrected = system.observations.data.clone();
    for (c, qi) in corrected.iter_mut().zip(q.iter()) {
        *c -= qi;
    }
    let corrected_system = AffineSheafSystem {
        topology_epoch: system.topology_epoch,
        observation_epoch: system.observation_epoch,
        operator: op.clone(),
        observations: EdgeField { data: corrected },
        weights: EdgeWeights { values: system.weights.values.clone() },
        gauge: GaugePolicy::MinimumNorm,
    };
    let post = compute_syndrome(&corrected_system, syn_cfg)?;

    let summary = RepairSummary {
        support: corrections.iter().map(|(k, _)| *k).collect(),
        objective_q: quantize(objective, syn_cfg.quantization_scale),
        pre_energy_q: quantize(syndrome.energy, syn_cfg.quantization_scale),
        post_energy_q: quantize(post.energy, syn_cfg.quantization_scale),
    };

    Ok(RepairPlan {
        corrections,
        objective,
        pre_energy: syndrome.energy,
        post_energy: post.energy,
        iterations: result.iterations,
        converged: result.converged,
        summary,
    })
}

/// Apply a repair plan to the system's observations (`b ← b − q`).
///
/// Fails on any protected edge unless the policy authorizes modification —
/// diagnosis is deliberately separate from authorization to act.
pub fn apply_repair(
    system: &mut AffineSheafSystem,
    plan: &RepairPlan,
    protected: Option<&[bool]>,
    policy: &RepairPolicy,
) -> Result<(), CohomologyError> {
    if let Some(p) = protected {
        if !policy.authorize_protected {
            for (key, _) in &plan.corrections {
                if let Some(pos) = system.operator.edge_position(key) {
                    if p[pos] {
                        return Err(CohomologyError::ProtectedEdge(key.u, key.v));
                    }
                }
            }
        }
    }
    for (key, q) in &plan.corrections {
        let (off, dim) = system
            .operator
            .edge_slice(key)
            .ok_or(CohomologyError::UnknownEdge(key.u, key.v))?;
        if q.len() != dim {
            return Err(CohomologyError::DimensionMismatch {
                expected: dim,
                actual: q.len(),
                context: format!("repair block on edge ({}, {})", key.u, key.v),
            });
        }
        for (b, qi) in system.observations.data[off..off + dim].iter_mut().zip(q.iter()) {
            *b -= qi;
        }
    }
    system.observation_epoch += 1;
    Ok(())
}
