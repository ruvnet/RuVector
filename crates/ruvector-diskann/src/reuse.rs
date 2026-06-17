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

use crate::distance::{FlatVectors, VisitedSet};
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

    /// Force a topology rebuild on `vectors`, bypassing the policy. The primitive an
    /// externally-driven trigger (e.g. a sampled-recall monitor) is built on: the caller
    /// owns the rebuild *signal*, the index owns the rebuild *mechanism*. Counts toward
    /// `rebuilds()` but does not advance the update `step`.
    pub fn force_rebuild(&mut self, vectors: &FlatVectors) -> Result<()> {
        debug_assert_eq!(vectors.len(), self.n, "force_rebuild: point count changed");
        self.graph = build_graph(
            vectors,
            self.n,
            self.max_degree,
            self.build_beam,
            self.alpha,
        )?;
        self.rebuilds += 1;
        Ok(())
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

/// Exact top-`k` neighbours of point `q` under L2 on `vectors` (brute force, excludes `q`).
fn brute_force_topk(vectors: &FlatVectors, q: usize, k: usize) -> Vec<u32> {
    let qv = vectors.get(q);
    let mut scored: Vec<(f32, u32)> = (0..vectors.len())
        .filter(|&i| i != q)
        .map(|i| (crate::distance::l2_squared(vectors.get(i), qv), i as u32))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

/// A drift-adaptive index whose rebuilds are driven by a **sampled-recall probe** instead of
/// a fixed cadence: on each metric update it estimates live recall@k on a small held-out
/// probe set and rebuilds only when that estimate falls below `floor`.
///
/// Under *bursty* drift this beats fixed [`Periodic`](RebuildPolicy::Periodic) — it spends
/// rebuilds where the drift actually is, skipping calm stretches (ADR-202 addendum:
/// validated WIN, ~42% fewer rebuilds than periodic at matched recall, and beats the
/// Frobenius-norm monitor ADR-200 found wanting). The knob `floor` *is* the recall SLA
/// (e.g. 0.95 = "keep recall ≥ 95%"), unlike `k`/`τ` which are indirect proxies.
///
/// **Cost:** the probe costs `probe_queries.len() × n` distance-evals per update — ~1–2
/// orders of magnitude below a rebuild — the price of measuring recall directly. Wraps a
/// [`DriftingIndex`] in `ReweightOnly` mode and drives [`force_rebuild`](DriftingIndex::force_rebuild).
pub struct RecallTrigger {
    index: DriftingIndex,
    probe_queries: Vec<u32>,
    k: usize,
    floor: f32,
    search_beam: usize,
}

impl RecallTrigger {
    /// Build the trigger on `vectors` (the `E₀` snapshot). `probe_queries` is a small, fixed
    /// held-out set of point indices used to estimate recall; `floor` is the recall target.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        vectors: &FlatVectors,
        probe_queries: Vec<u32>,
        k: usize,
        floor: f32,
        search_beam: usize,
        max_degree: usize,
        build_beam: usize,
        alpha: f32,
    ) -> Result<Self> {
        let index = DriftingIndex::build(
            vectors,
            RebuildPolicy::ReweightOnly,
            max_degree,
            build_beam,
            alpha,
        )?;
        Ok(Self {
            index,
            probe_queries,
            k,
            floor,
            search_beam,
        })
    }

    /// Probe-estimated recall@k of the current topology against exact neighbours under
    /// `vectors` (mean over the probe set). 1.0 if the probe set is empty.
    pub fn probe_recall(&self, vectors: &FlatVectors) -> f32 {
        if self.probe_queries.is_empty() {
            return 1.0;
        }
        let mut sum = 0.0f32;
        for &q in &self.probe_queries {
            let qi = q as usize;
            let truth = brute_force_topk(vectors, qi, self.k);
            let qv = vectors.get(qi);
            let (cands, _) = self.index.search(vectors, qv, self.search_beam);
            let mut scored: Vec<(f32, u32)> = cands
                .iter()
                .map(|&c| (crate::distance::l2_squared(vectors.get(c as usize), qv), c))
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            let hits = scored
                .into_iter()
                .filter(|&(_, c)| c as usize != qi)
                .take(self.k)
                .filter(|(_, c)| truth.contains(c))
                .count();
            sum += hits as f32 / self.k.max(1) as f32;
        }
        sum / self.probe_queries.len() as f32
    }

    /// React to a metric update: rebuild on `vectors` iff the probe recall is below `floor`.
    /// Returns whether a rebuild happened.
    pub fn on_metric_update(&mut self, vectors: &FlatVectors) -> Result<bool> {
        if self.probe_recall(vectors) < self.floor {
            self.index.force_rebuild(vectors)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Search the current topology against `vectors`.
    pub fn search(
        &self,
        vectors: &FlatVectors,
        query: &[f32],
        beam_width: usize,
    ) -> (Vec<u32>, usize) {
        self.index.search(vectors, query, beam_width)
    }

    /// Number of rebuilds the trigger has fired.
    pub fn rebuilds(&self) -> usize {
        self.index.rebuilds()
    }
}

/// A drift-adaptive index that repairs only the **displaced subset** of the graph instead of
/// rebuilding the whole topology — the BET-1 "missing middle" between
/// [`RebuildPolicy::ReweightOnly`] (repair nothing) and [`RebuildPolicy::AlwaysRebuild`]
/// (repair everything).
///
/// Under metric drift membership is fixed: a point never leaves the set, its coordinates only
/// move. So the faithful incremental operation is **not** FreshDiskANN delete+reinsert (whose
/// delete-consolidation is inapplicable when nothing is removed). It is, for each displaced
/// node `u`: recompute `u`'s out-edges (`greedy_search → robust_prune` at the new position),
/// set `neighbors[u]`, and add back-edges into its new out-neighbours — exactly the per-node
/// step [`VamanaGraph::build`] runs, applied to one node. Residual stale *in*-edges from
/// non-displaced neighbours that `u` moved away from are left to **decay** — the same tolerance
/// ADR-200/202 proved Vamana has; a neighbour that is itself reindexed re-prunes and drops the
/// stale edge.
///
/// `reindex_frac` selects the top fraction of nodes by **displacement since their last reindex**
/// to repair each update — the cost/recall knob, analogous to [`RebuildPolicy::Periodic`]'s `k`.
/// `0.0` repairs nothing (≡ `ReweightOnly`); `1.0` repairs every moved node each step (a costly
/// upper bound approaching a rebuild).
///
/// Feature-gated behind `reuse-under-drift`. See
/// `docs/plans/bet1-productionize/PRE-REGISTRATION-incremental.md`.
pub struct IncrementalIndex {
    graph: VamanaGraph,
    /// Each node's position as of its last reindex (E₀ for never-reindexed nodes); flat,
    /// dim-major — the displacement baseline. Length `n * dim`.
    reference: Vec<f32>,
    dim: usize,
    n: usize,
    max_degree: usize,
    build_beam: usize,
    alpha: f32,
    reindex_frac: f32,
    // Telemetry.
    updates: usize,
    reindexed_total: usize,
}

impl IncrementalIndex {
    /// Build the initial topology on `vectors` (the `E₀` snapshot). `reindex_frac` is the
    /// fraction of (most-displaced) nodes to repair per update; `max_degree`/`build_beam`/`alpha`
    /// are the Vamana build parameters (production defaults 32 / 64 / 1.2).
    pub fn build(
        vectors: &FlatVectors,
        reindex_frac: f32,
        max_degree: usize,
        build_beam: usize,
        alpha: f32,
    ) -> Result<Self> {
        let n = vectors.len();
        let dim = vectors.dim;
        let graph = build_graph(vectors, n, max_degree, build_beam, alpha)?;
        Ok(Self {
            graph,
            reference: vectors.data.clone(),
            dim,
            n,
            max_degree,
            build_beam,
            alpha,
            reindex_frac: reindex_frac.clamp(0.0, 1.0),
            updates: 0,
            reindexed_total: 0,
        })
    }

    /// L2² displacement of node `u` since its last reindex.
    fn displacement(&self, vectors: &FlatVectors, u: usize) -> f32 {
        let s = u * self.dim;
        crate::distance::l2_squared(vectors.get(u), &self.reference[s..s + self.dim])
    }

    /// React to a metric update: reindex the top `reindex_frac` of nodes by displacement since
    /// their last reindex (skipping nodes that did not move). Returns how many nodes were
    /// reindexed, for cost accounting.
    ///
    /// `vectors` must keep the same point count as the original build (drift changes vector
    /// *values*, not membership).
    pub fn on_metric_update(&mut self, vectors: &FlatVectors) -> Result<usize> {
        debug_assert_eq!(
            vectors.len(),
            self.n,
            "incremental model assumes fixed membership; point count changed"
        );
        self.updates += 1;
        let budget = ((self.n as f32) * self.reindex_frac).round() as usize;
        if budget == 0 {
            return Ok(0);
        }
        // Rank nodes by displacement (largest first); repair the top `budget` that actually moved.
        let mut disp: Vec<(f32, u32)> = (0..self.n)
            .map(|u| (self.displacement(vectors, u), u as u32))
            .filter(|&(d, _)| d > 0.0)
            .collect();
        disp.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let take = budget.min(disp.len());

        let mut visited = VisitedSet::new(self.n);
        for &(_, u) in disp.iter().take(take) {
            self.reindex_node(vectors, u, &mut visited);
            // This node is now consistent with the live snapshot — reset its displacement baseline.
            let s = u as usize * self.dim;
            self.reference[s..s + self.dim].copy_from_slice(vectors.get(u as usize));
        }
        self.reindexed_total += take;
        Ok(take)
    }

    /// Recompute `u`'s out-edges at its current position and refresh its back-edges. Two-phase
    /// (all reads, then all writes) so the `&self` borrow in `robust_prune` never overlaps the
    /// `&mut` writes into `neighbors`.
    fn reindex_node(&mut self, vectors: &FlatVectors, u: u32, visited: &mut VisitedSet) {
        let uq = vectors.get(u as usize).to_vec();
        // Candidate generation from the live (drifted) graph, then α-robust prune to out-edges.
        let (cands, _) = self
            .graph
            .greedy_search_fast(vectors, &uq, self.build_beam, visited);
        let pruned = self.graph.robust_prune(vectors, u, &cands, self.alpha);

        // Phase 1 — compute back-edge writes without mutating (read-only borrows of the graph).
        let mut writes: Vec<(usize, Option<Vec<u32>>)> = Vec::with_capacity(pruned.len());
        for &c in &pruned {
            let cu = c as usize;
            if cu == u as usize || self.graph.neighbors[cu].contains(&u) {
                continue;
            }
            if self.graph.neighbors[cu].len() < self.max_degree {
                writes.push((cu, None)); // simple append of u
            } else {
                let mut combined = self.graph.neighbors[cu].clone();
                combined.push(u);
                let repruned = self.graph.robust_prune(vectors, c, &combined, self.alpha);
                writes.push((cu, Some(repruned)));
            }
        }

        // Phase 2 — apply writes.
        self.graph.neighbors[u as usize] = pruned;
        for (cu, rep) in writes {
            match rep {
                Some(r) => self.graph.neighbors[cu] = r,
                None => self.graph.neighbors[cu].push(u),
            }
        }
    }

    /// Search the current topology against `vectors` (the live snapshot).
    pub fn search(
        &self,
        vectors: &FlatVectors,
        query: &[f32],
        beam_width: usize,
    ) -> (Vec<u32>, usize) {
        self.graph.greedy_search(vectors, query, beam_width)
    }

    /// Total nodes reindexed across all updates (the cumulative cost proxy).
    pub fn reindexed_total(&self) -> usize {
        self.reindexed_total
    }

    /// Number of metric updates seen.
    pub fn updates(&self) -> usize {
        self.updates
    }

    /// Borrow the underlying topology (e.g. for degree-bound inspection).
    pub fn graph(&self) -> &VamanaGraph {
        &self.graph
    }
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
        let mut idx = DriftingIndex::build(&v, RebuildPolicy::ReweightOnly, 16, 32, 1.2).unwrap();
        for _ in 0..10 {
            assert!(!idx.on_metric_update(&v).unwrap());
        }
        assert_eq!(idx.rebuilds(), 0);
        assert_eq!(idx.step(), 10);
    }

    #[test]
    fn always_rebuild_rebuilds_every_step() {
        let v = fixture(64, 8);
        let mut idx = DriftingIndex::build(&v, RebuildPolicy::AlwaysRebuild, 16, 32, 1.2).unwrap();
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
            vec![false, false, false, true, false, false, false, true, false, false, false, true]
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
    fn force_rebuild_counts_but_does_not_advance_step() {
        let v = fixture(64, 8);
        let mut idx = DriftingIndex::build(&v, RebuildPolicy::ReweightOnly, 16, 32, 1.2).unwrap();
        idx.on_metric_update(&v).unwrap(); // step 1, no rebuild
        idx.force_rebuild(&v).unwrap(); // external trigger fires
        idx.force_rebuild(&v).unwrap();
        assert_eq!(
            idx.step(),
            1,
            "force_rebuild must not advance the update step"
        );
        assert_eq!(
            idx.rebuilds(),
            2,
            "force_rebuild must count toward rebuilds"
        );
    }

    /// A geometrically distinct fixture so swapping it in collapses the E0 graph's recall.
    fn fixture_b(n: usize, dim: usize) -> FlatVectors {
        let mut f = FlatVectors::with_capacity(dim, n);
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|d| (((n - i) * 53 + d * 17) % 89) as f32 / 89.0)
                .collect();
            f.push(&v);
        }
        f
    }

    #[test]
    fn recall_trigger_holds_under_no_drift() {
        let v = fixture(128, 8);
        let probes: Vec<u32> = (0..16).collect();
        let mut t = RecallTrigger::build(&v, probes, 5, 0.9, 32, 16, 32, 1.2).unwrap();
        // same vectors → the index searches what it was built on → recall ~1.0 → no rebuild
        assert!(t.probe_recall(&v) >= 0.9);
        assert!(!t.on_metric_update(&v).unwrap());
        assert_eq!(t.rebuilds(), 0);
    }

    #[test]
    fn recall_trigger_fires_then_recovers_under_drift() {
        let v = fixture(128, 8);
        let probes: Vec<u32> = (0..16).collect();
        let mut t = RecallTrigger::build(&v, probes, 5, 0.9, 32, 16, 32, 1.2).unwrap();
        // swap in a geometrically different vector set: recall collapses → trigger fires
        let vb = fixture_b(128, 8);
        assert!(
            t.probe_recall(&vb) < 0.9,
            "drift should drop probe recall below floor"
        );
        assert!(
            t.on_metric_update(&vb).unwrap(),
            "trigger must fire on the drift"
        );
        assert_eq!(t.rebuilds(), 1);
        // after rebuilding on vb, recall is restored → a second update does not re-fire
        assert!(!t.on_metric_update(&vb).unwrap());
        assert_eq!(t.rebuilds(), 1);
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

    // ---- IncrementalIndex (BET-1 missing-middle) ----

    /// Mean recall@k of a candidate-producing search against brute-force truth on `vectors`.
    fn measure_recall<F>(vectors: &FlatVectors, k: usize, nq: usize, search: F) -> f64
    where
        F: Fn(&[f32]) -> Vec<u32>,
    {
        let mut acc = 0.0;
        for q in 0..nq {
            let truth = brute_force_topk(vectors, q, k);
            let qv = vectors.get(q).to_vec();
            let cands = search(&qv);
            let mut scored: Vec<(f32, u32)> = cands
                .iter()
                .map(|&c| (crate::distance::l2_squared(vectors.get(c as usize), &qv), c))
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0));
            let got: Vec<u32> = scored
                .into_iter()
                .filter(|&(_, c)| c as usize != q)
                .take(k)
                .map(|(_, c)| c)
                .collect();
            acc += got.iter().filter(|g| truth.contains(g)).count() as f64 / k as f64;
        }
        acc / nq as f64
    }

    #[test]
    fn incremental_frac_zero_is_reweight_only() {
        let v = fixture(64, 8);
        let mut idx = IncrementalIndex::build(&v, 0.0, 16, 32, 1.2).unwrap();
        let vb = fixture_b(64, 8);
        assert_eq!(idx.on_metric_update(&vb).unwrap(), 0);
        assert_eq!(idx.reindexed_total(), 0);
        assert_eq!(idx.updates(), 1);
    }

    #[test]
    fn incremental_keeps_degree_bounded() {
        let v = fixture(160, 8);
        let vb = fixture_b(160, 8);
        let mut idx = IncrementalIndex::build(&v, 1.0, 16, 32, 1.2).unwrap();
        idx.on_metric_update(&vb).unwrap();
        for nbrs in &idx.graph().neighbors {
            assert!(nbrs.len() <= 16, "degree bound violated: {}", nbrs.len());
        }
    }

    #[test]
    fn incremental_full_reindex_recovers_navigability() {
        // Build on A, drift to B (every point moves). Pure reuse should lose recall on B;
        // a full incremental reindex (f=1.0) should recover it, approaching a fresh rebuild.
        let va = fixture(200, 16);
        let vb = fixture_b(200, 16);
        let (k, beam, nq) = (10usize, 48usize, 20usize);

        let reuse = DriftingIndex::build(&va, RebuildPolicy::ReweightOnly, 24, 48, 1.2).unwrap();
        let mut inc = IncrementalIndex::build(&va, 1.0, 24, 48, 1.2).unwrap();
        let touched = inc.on_metric_update(&vb).unwrap();
        assert!(touched > 0, "drift should displace nodes to reindex");
        let fresh = DriftingIndex::build(&vb, RebuildPolicy::AlwaysRebuild, 24, 48, 1.2).unwrap();

        let r_reuse = measure_recall(&vb, k, nq, |q| reuse.search(&vb, q, beam).0);
        let r_inc = measure_recall(&vb, k, nq, |q| inc.search(&vb, q, beam).0);
        let r_fresh = measure_recall(&vb, k, nq, |q| fresh.search(&vb, q, beam).0);

        // Incremental must produce a navigable graph on B and not be worse than pure reuse.
        assert!(
            r_inc >= 0.7,
            "reindexed graph not navigable: r_inc={r_inc:.3}"
        );
        assert!(
            r_inc >= r_reuse - 0.05,
            "incremental ({r_inc:.3}) should be no worse than reuse ({r_reuse:.3})"
        );
        // ...and land within a generous margin of a fresh rebuild (sanity, not the research claim).
        assert!(
            r_inc >= r_fresh - 0.2,
            "incremental ({r_inc:.3}) far below fresh rebuild ({r_fresh:.3})"
        );
    }
}
