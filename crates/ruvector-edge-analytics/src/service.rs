//! Min-cut queries under limits and a budget (ADR-351 §3 rv-mincut, §10).
//!
//! `plan` validates parameters, applies the graph limits and compares the
//! cost estimate with the profile's budget; `query` then runs the solver.
//! An over-budget inline query returns `BudgetExceeded` (413) with
//! `job_eligible` set when the job profile would accept it.

use crate::cost::{estimate_exact, precheck, CostEstimate};
use crate::error::{AnalyticsError, Result};
use crate::graph::{EdgeRecord, GraphLimits, TenantGraph};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Work and memory budget for one query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Maximum estimated work units (~ns native).
    pub max_work: u64,
    /// Maximum estimated peak heap bytes.
    pub max_memory_bytes: u64,
}

/// Limits plus budget: what one execution context may spend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Profile {
    /// Graph size limits.
    pub limits: GraphLimits,
    /// Cost budget.
    pub budget: Budget,
}

impl Profile {
    /// Inline `POST /v1/mincut`: ~0.5 s native CPU and half of a 128 MB
    /// isolate that also holds resident vector shards. The memory budget,
    /// not the ADR's 200k-edge ceiling, is what binds on sparse graphs
    /// (~130k edges at the measured ~500 B/edge estimate).
    pub const INLINE: Profile = Profile {
        limits: GraphLimits::INLINE,
        budget: Budget {
            max_work: 500_000_000,
            max_memory_bytes: 64 << 20,
        },
    };
    /// Async `POST /v1/mincut/jobs`: one invocation with the isolate to
    /// itself (100 MB of 128 MB), ~20 s native CPU **[U]** (DO alarm CPU).
    pub const JOB: Profile = Profile {
        limits: GraphLimits::JOB,
        budget: Budget {
            max_work: 20_000_000_000,
            max_memory_bytes: 100 << 20,
        },
    };
}

/// Query mode.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum QueryMode {
    /// Exact global minimum cut with a witness partition.
    Exact,
    /// A `(1 + epsilon)`-approximate cut. Answered with the **exact** solver
    /// (the report says `mode: exact`), which meets any such guarantee;
    /// admission, 413 and job routing are those of [`QueryMode::Exact`].
    ///
    /// `ruvector-mincut`'s `ApproxMinCut` is deliberately not served: its
    /// sparsifier estimate has no sound bound and measured up to 10x *below*
    /// the true minimum (no cut of that weight exists; see
    /// `tests/equivalence.rs`), so returning it as `value` would break the
    /// contract the `epsilon` parameter implies.
    Approximate {
        /// `0 < epsilon <= 1`.
        epsilon: f64,
    },
}

/// A min-cut answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CutReport {
    /// Mode that produced it (always `exact`: see [`QueryMode::Approximate`]).
    pub mode: QueryMode,
    /// Graph revision it was computed on.
    pub revision: u64,
    /// Cut value; `None` when no cut exists (fewer than two vertices).
    pub value: Option<f64>,
    /// Exact only: `(S, T)`, each sorted ascending.
    pub partition: Option<(Vec<u64>, Vec<u64>)>,
    /// Exact only: crossing edges, canonical and sorted.
    pub cut_edges: Option<Vec<EdgeRecord>>,
    /// The estimate the query was admitted with.
    pub estimate: CostEstimate,
}

impl CutReport {
    /// Canonical digest of the answer (value, partition, cut edges), for
    /// cross-target and cross-run equality checks.
    pub fn digest_hex(&self) -> String {
        let mut h = Sha256::new();
        h.update(match self.mode {
            QueryMode::Exact => [0u8],
            QueryMode::Approximate { .. } => [1u8],
        });
        match self.value {
            None => h.update([0xffu8]),
            Some(v) => {
                h.update([0u8]);
                h.update(v.to_bits().to_le_bytes());
            }
        }
        if let Some((s, t)) = &self.partition {
            for side in [s, t] {
                h.update((side.len() as u64).to_le_bytes());
                for x in side {
                    h.update(x.to_le_bytes());
                }
            }
        }
        if let Some(edges) = &self.cut_edges {
            h.update((edges.len() as u64).to_le_bytes());
            for e in edges {
                h.update(e.u.to_le_bytes());
                h.update(e.v.to_le_bytes());
                h.update(e.w.to_bits().to_le_bytes());
            }
        }
        hex::encode(h.finalize())
    }
}

fn check_budget(est: &CostEstimate, budget: &Budget, job_eligible: bool) -> Result<()> {
    if est.memory_bytes > budget.max_memory_bytes {
        return Err(AnalyticsError::BudgetExceeded {
            resource: "memory_bytes",
            estimated: est.memory_bytes,
            budget: budget.max_memory_bytes,
            job_eligible,
        });
    }
    if est.work > budget.max_work {
        return Err(AnalyticsError::BudgetExceeded {
            resource: "work",
            estimated: est.work,
            budget: budget.max_work,
            job_eligible,
        });
    }
    Ok(())
}

fn fits(g: &TenantGraph, est: &CostEstimate, p: &Profile) -> bool {
    p.limits.check(g.vertex_count(), g.edge_count()).is_ok()
        && check_budget(est, &p.budget, false).is_ok()
}

/// Validate, estimate and admit a query against `profile`. On a budget
/// failure `job_eligible` reports whether [`Profile::JOB`] would admit it.
pub fn plan(g: &TenantGraph, mode: &QueryMode, profile: &Profile) -> Result<CostEstimate> {
    profile.limits.check(g.vertex_count(), g.edge_count())?;
    if let QueryMode::Approximate { epsilon } = *mode {
        if !(epsilon.is_finite() && epsilon > 0.0 && epsilon <= 1.0) {
            return Err(AnalyticsError::Invalid("epsilon must be in (0, 1]"));
        }
    }
    let est = estimate_exact(&precheck(g), g.edge_count());
    let job_eligible = profile != &Profile::JOB && fits(g, &est, &Profile::JOB);
    check_budget(&est, &profile.budget, job_eligible)?;
    Ok(est)
}

/// Plan, then run. Never panics on validated input; every refusal is typed.
pub fn query(g: &TenantGraph, mode: &QueryMode, profile: &Profile) -> Result<CutReport> {
    let estimate = plan(g, mode, profile)?;
    run_exact(g, estimate)
}

fn run_exact(g: &TenantGraph, estimate: CostEstimate) -> Result<CutReport> {
    let mc = g.rebuild_exact()?;
    let v = mc.min_cut_value();
    let value = v.is_finite().then_some(v);
    let (s, t) = mc.partition();
    let mut cut: Vec<EdgeRecord> = mc
        .cut_edges()
        .into_iter()
        .map(|e| {
            let (u, v) = e.canonical_endpoints();
            EdgeRecord { u, v, w: e.weight }
        })
        .collect();
    cut.sort_unstable_by_key(|e| (e.u, e.v));
    Ok(CutReport {
        mode: QueryMode::Exact,
        revision: g.revision(),
        value,
        partition: Some((s, t)),
        cut_edges: Some(cut),
        estimate,
    })
}
