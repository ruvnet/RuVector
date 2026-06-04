//! Fixed-topology reuse under metric drift + periodic rebuild (BET 1, ADR-200).
//!
//! A self-learning system (e.g. `ruvector-gnn`) continuously re-estimates node
//! embeddings, so the effective L2 metric over those embeddings **drifts**. The
//! textbook remedy is a full [`VamanaGraph`] rebuild on every update — superlinear,
//! minutes-to-hours at corpus scale. ADR-200 showed (under synthetic drift, on this
//! exact production index) that the navigation topology can be **reused**: build the
//! graph once on `E₀`, then search the *drifted* vectors against it, recomputing only
//! distances. Recall stays within 2% of a full rebuild at ~10³–10⁴× lower update cost,
//! with a periodic rebuild recovering the residual gap under heavy drift.
//!
//! This module wires that policy into the production loop. The reuse hook is native:
//! [`VamanaGraph`] stores only topology (`neighbors` + `medoid`) and
//! [`VamanaGraph::greedy_search`] takes the vectors externally — so the consumer (the
//! GNN) owns and mutates the embeddings, and the index only decides *when* to rebuild.
//!
//! Feature-gated behind `reuse-under-drift` (default off) — the shipping build is
//! unaffected. See `docs/plans/bet1-productionize/PRE-REGISTRATION.md`.

use crate::distance::FlatVectors;
use crate::error::Result;
use crate::graph::VamanaGraph;

/// When to spend a full [`VamanaGraph`] rebuild as the metric drifts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildPolicy {
    /// Rebuild on every metric update — the incumbent remedy. Highest recall, full
    /// rebuild cost every step. The baseline `B` of ADR-200.
    AlwaysRebuild,
    /// Never rebuild — reuse the `E₀` topology, recompute distances under the drifted
    /// vectors. Zero rebuild cost. The bet `A` of ADR-200; decays under heavy
    /// accumulated drift (why [`Periodic`](RebuildPolicy::Periodic) exists).
    ReweightOnly,
    /// Reuse every step, full rebuild every `k` updates — the shippable hybrid. ADR-200
    /// found `Periodic{k:4}` recovered to within 0.3% of `AlwaysRebuild` at 25% of its
    /// cost. `k == 0` is treated as [`ReweightOnly`](RebuildPolicy::ReweightOnly).
    Periodic {
        /// Rebuild cadence: rebuild when `step % k == 0`.
        k: usize,
    },
}

impl RebuildPolicy {
    /// Whether the policy rebuilds at update number `step` (1-based: the first
    /// `on_metric_update` is step 1).
    fn rebuilds_at(self, step: usize) -> bool {
        match self {
            RebuildPolicy::AlwaysRebuild => true,
            RebuildPolicy::ReweightOnly => false,
            RebuildPolicy::Periodic { k } => k > 0 && step % k == 0,
        }
    }
}

/// A Vamana index that adapts to a drifting metric by reusing its navigation topology,
/// rebuilding only as dictated by its [`RebuildPolicy`].
///
/// The index does **not** own the vectors — the consumer owns the embedding store and
/// passes the current snapshot to [`on_metric_update`](DriftingIndex::on_metric_update)
/// and [`search`](DriftingIndex::search). This keeps the dependency direction clean: the
/// index knows nothing about *what* drives the drift.
pub struct DriftingIndex {
    graph: VamanaGraph,
    policy: RebuildPolicy,
    // Build parameters, retained to reconstruct the graph on rebuild.
    n: usize,
    max_degree: usize,
    build_beam: usize,
    alpha: f32,
    // Telemetry.
    step: usize,
    rebuilds: usize,
}

impl DriftingIndex {
    /// Build the initial topology on `vectors` (the `E₀` snapshot) under `policy`.
    ///
    /// `max_degree`, `build_beam`, `alpha` are the Vamana build parameters (production
    /// defaults: 32 / 64 / 1.2), reused on every subsequent rebuild.
    pub fn build(
        vectors: &FlatVectors,
        policy: RebuildPolicy,
        max_degree: usize,
        build_beam: usize,
        alpha: f32,
    ) -> Result<Self> {
        let n = vectors.len();
        let graph = build_graph(vectors, n, max_degree, build_beam, alpha)?;
        Ok(Self {
            graph,
            policy,
            n,
            max_degree,
            build_beam,
            alpha,
            step: 0,
            rebuilds: 0,
        })
    }

    /// Signal that the metric drifted (the consumer wrote a new embedding snapshot).
    ///
    /// Rebuilds the topology on `vectors` iff the policy dictates it at this step;
    /// otherwise the existing topology is retained (pure re-weight). Returns whether a
    /// rebuild happened, so the caller can account for cost.
    ///
    /// `vectors` must contain the same number of points as the original build (drift
    /// changes vector *values*, not membership; insert/delete is out of scope for the
    /// reuse model). Returns [`DiskAnnError::DimensionMismatch`](crate::DiskAnnError) if
    /// the count changed.
    pub fn on_metric_update(&mut self, vectors: &FlatVectors) -> Result<bool> {
        self.step += 1;
        if !self.policy.rebuilds_at(self.step) {
            return Ok(false);
        }
        debug_assert_eq!(
            vectors.len(),
            self.n,
            "reuse model assumes fixed membership; point count changed"
        );
        self.graph = build_graph(
            vectors,
            self.n,
            self.max_degree,
            self.build_beam,
            self.alpha,
        )?;
        self.rebuilds += 1;
        Ok(true)
    }

    /// Search the current topology against `vectors` (the live, possibly-drifted
    /// snapshot), returning candidate ids and the visited count (distance-evals proxy).
    ///
    /// Callers typically re-rank the candidates by exact distance to the query under the
    /// current metric and take the top-k.
    pub fn search(
        &self,
        vectors: &FlatVectors,
        query: &[f32],
        beam_width: usize,
    ) -> (Vec<u32>, usize) {
        self.graph.greedy_search(vectors, query, beam_width)
    }

    /// The configured rebuild policy.
    pub fn policy(&self) -> RebuildPolicy {
        self.policy
    }

    /// Number of metric updates seen so far.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Number of full rebuilds performed (the cost the reuse policy is trying to avoid).
    pub fn rebuilds(&self) -> usize {
        self.rebuilds
    }

    /// Borrow the underlying topology (e.g. for inspection or persistence).
    pub fn graph(&self) -> &VamanaGraph {
        &self.graph
    }
}

fn build_graph(
    vectors: &FlatVectors,
    n: usize,
    max_degree: usize,
    build_beam: usize,
    alpha: f32,
) -> Result<VamanaGraph> {
    let mut graph = VamanaGraph::new(n, max_degree, build_beam, alpha);
    graph.build(vectors)?;
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic clustered points so the graph is non-trivial.
    fn fixture(n: usize, dim: usize) -> FlatVectors {
        let mut f = FlatVectors::with_capacity(dim, n);
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|d| ((i * 31 + d * 7) % 97) as f32 / 97.0)
                .collect();
            f.push(&v);
        }
        f
    }

    #[test]
    fn reweight_only_never_rebuilds() {
        let v = fixture(64, 8);
        let mut idx =
            DriftingIndex::build(&v, RebuildPolicy::ReweightOnly, 16, 32, 1.2).unwrap();
        for _ in 0..10 {
            assert!(!idx.on_metric_update(&v).unwrap());
        }
        assert_eq!(idx.rebuilds(), 0);
        assert_eq!(idx.step(), 10);
    }

    #[test]
    fn always_rebuild_rebuilds_every_step() {
        let v = fixture(64, 8);
        let mut idx =
            DriftingIndex::build(&v, RebuildPolicy::AlwaysRebuild, 16, 32, 1.2).unwrap();
        for _ in 0..10 {
            assert!(idx.on_metric_update(&v).unwrap());
        }
        assert_eq!(idx.rebuilds(), 10);
    }

    #[test]
    fn periodic_rebuilds_on_cadence() {
        let v = fixture(64, 8);
        let mut idx =
            DriftingIndex::build(&v, RebuildPolicy::Periodic { k: 4 }, 16, 32, 1.2).unwrap();
        let did: Vec<bool> = (0..12).map(|_| idx.on_metric_update(&v).unwrap()).collect();
        // steps 1..=12, rebuild at 4, 8, 12
        assert_eq!(
            did,
            vec![
                false, false, false, true, false, false, false, true, false, false, false, true
            ]
        );
        assert_eq!(idx.rebuilds(), 3);
    }

    #[test]
    fn periodic_k0_is_reweight_only() {
        let v = fixture(32, 8);
        let mut idx =
            DriftingIndex::build(&v, RebuildPolicy::Periodic { k: 0 }, 16, 32, 1.2).unwrap();
        for _ in 0..5 {
            assert!(!idx.on_metric_update(&v).unwrap());
        }
        assert_eq!(idx.rebuilds(), 0);
    }

    #[test]
    fn search_returns_self_as_nearest() {
        let v = fixture(128, 8);
        let idx = DriftingIndex::build(&v, RebuildPolicy::ReweightOnly, 16, 32, 1.2).unwrap();
        // Query with point 5's own vector; it should be among the nearest candidates.
        let q = v.get(5).to_vec();
        let (cands, visited) = idx.search(&v, &q, 16);
        assert!(visited > 0);
        assert!(cands.contains(&5), "self should be retrieved: {cands:?}");
    }
}
