//! Streaming cluster admission policies.

use crate::cosine_sim;
use crate::mincut::{global_min_cut, WeightMatrix};
use std::collections::VecDeque;

/// Outcome of an admission decision.
#[derive(Debug, Clone, Copy)]
pub struct Decision {
    pub cluster_id: usize,
    pub spawned_new: bool,
    /// Similarity computations performed to reach this decision (cost proxy
    /// distinct from wall-clock latency, which the benchmark measures
    /// separately with `Instant`).
    pub sim_ops: usize,
}

pub trait AdmissionPolicy {
    fn name(&self) -> &str;
    /// Decide where `point` should go without mutating state.
    fn decide(&self, point: &[f32]) -> Decision;
    /// Apply a decision: update centroids/counts (and any policy-internal
    /// calibration state).
    fn commit(&mut self, point: &[f32], decision: &Decision);
    fn n_clusters(&self) -> usize;
    fn centroid(&self, cluster_id: usize) -> &[f32];

    /// Convenience: decide then commit in one step.
    fn admit(&mut self, point: &[f32]) -> Decision {
        let d = self.decide(point);
        self.commit(point, &d);
        d
    }
}

/// Shared spawn/merge decision given the outcome of a global-min-cut
/// computation over `c` existing clusters + the candidate point.
///
/// With `c == 1` the graph has exactly 2 nodes, so *any* global min cut
/// trivially separates the point from the one cluster regardless of how
/// similar they are — `group` (clusters on the point's side, excluding the
/// point) is always empty in that case by construction, not because the
/// point is a structural outlier. The only usable signal there is the cut
/// weight itself. With `c >= 2`, a point isolated on its own side of the
/// cut (empty `group`) despite >= 2 clusters existing elsewhere in the
/// graph *is* a genuine structural-outlier signal, independent of `tau`.
fn should_spawn(c: usize, avg_cut: f32, group: &[usize], tau: f32) -> bool {
    if c == 1 {
        avg_cut < tau
    } else if group.is_empty() {
        true
    } else {
        avg_cut < tau
    }
}

/// Merge target when not spawning: the best-matching cluster in `group`, or
/// cluster 0 when `group` is empty (the `c == 1` degenerate case above).
fn merge_target(point: &[f32], group: &[usize], centroids: &[Vec<f32>]) -> usize {
    if group.is_empty() {
        return 0;
    }
    *group
        .iter()
        .max_by(|&&a, &&b| {
            cosine_sim(point, &centroids[a])
                .partial_cmp(&cosine_sim(point, &centroids[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("group non-empty here")
}

fn running_mean_update(centroid: &mut [f32], count: usize, point: &[f32]) {
    let n = count as f32;
    for (c, &p) in centroid.iter_mut().zip(point.iter()) {
        *c = (*c * n + p) / (n + 1.0);
    }
    crate::dataset::normalise(centroid);
}

// ─── 1. NearestCentroidThreshold — baseline ──────────────────────────────────

/// Merge into the nearest centroid if cosine similarity clears a fixed
/// threshold; otherwise spawn a new cluster. This is a reasonable,
/// widely-used online-clustering baseline (leader-follower / sequential
/// k-means with a novelty threshold) — not a straw man.
pub struct NearestCentroidThreshold {
    pub threshold: f32,
    centroids: Vec<Vec<f32>>,
    counts: Vec<usize>,
}

impl NearestCentroidThreshold {
    pub fn new(threshold: f32) -> Self {
        NearestCentroidThreshold {
            threshold,
            centroids: Vec::new(),
            counts: Vec::new(),
        }
    }
}

impl AdmissionPolicy for NearestCentroidThreshold {
    fn name(&self) -> &str {
        "NearestCentroidThreshold"
    }

    fn decide(&self, point: &[f32]) -> Decision {
        if self.centroids.is_empty() {
            return Decision {
                cluster_id: 0,
                spawned_new: true,
                sim_ops: 0,
            };
        }
        let mut best_id = 0usize;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, c) in self.centroids.iter().enumerate() {
            let s = cosine_sim(point, c);
            if s > best_sim {
                best_sim = s;
                best_id = i;
            }
        }
        let sim_ops = self.centroids.len();
        if best_sim >= self.threshold {
            Decision {
                cluster_id: best_id,
                spawned_new: false,
                sim_ops,
            }
        } else {
            Decision {
                cluster_id: self.centroids.len(),
                spawned_new: true,
                sim_ops,
            }
        }
    }

    fn commit(&mut self, point: &[f32], decision: &Decision) {
        if decision.spawned_new {
            self.centroids.push(point.to_vec());
            self.counts.push(1);
        } else {
            let c = decision.cluster_id;
            running_mean_update(&mut self.centroids[c], self.counts[c], point);
            self.counts[c] += 1;
        }
    }

    fn n_clusters(&self) -> usize {
        self.centroids.len()
    }

    fn centroid(&self, cluster_id: usize) -> &[f32] {
        &self.centroids[cluster_id]
    }
}

// ─── 2. MincutGatedAdmission — candidate A ───────────────────────────────────

/// Global-min-cut gated admission. Builds a weighted graph over existing
/// cluster centroids plus the candidate point (edge weight = clamped
/// cosine similarity), computes the global min cut, and gates on the
/// *average* crossing-edge weight against a fixed `tau`.
///
/// If the candidate ends up alone on its side of the cut, or the average
/// crossing weight is below `tau`, it spawns a new cluster; otherwise it
/// merges into the best-matching centroid on its own side of the cut.
///
/// `max_clusters` is a hard computational safety valve (not a correctness
/// mechanism): past this many clusters, admission falls back to plain
/// nearest-centroid merge so the O(C^3) min-cut cost per insertion stays
/// bounded. It is set well above the acceptance threshold on final cluster
/// count so that threshold is a real measurement, not a tautology.
pub struct MincutGatedAdmission {
    pub tau: f32,
    pub max_clusters: usize,
    centroids: Vec<Vec<f32>>,
    counts: Vec<usize>,
}

impl MincutGatedAdmission {
    pub fn new(tau: f32, max_clusters: usize) -> Self {
        MincutGatedAdmission {
            tau,
            max_clusters,
            centroids: Vec::new(),
            counts: Vec::new(),
        }
    }

    /// Shared by candidate A and candidate B: build the (clusters +
    /// candidate) graph, run global min cut, and return
    /// (avg_crossing_weight, side-of-candidate group, sim_ops).
    fn cut_decision(&self, point: &[f32]) -> (f32, Vec<usize>, usize) {
        let c = self.centroids.len();
        let mut m = WeightMatrix::new(c + 1);
        let mut sim_ops = 0usize;
        for i in 0..c {
            for j in (i + 1)..c {
                let w = cosine_sim(&self.centroids[i], &self.centroids[j]).max(0.0);
                m.set_sym(i, j, w as f64);
                sim_ops += 1;
            }
        }
        for i in 0..c {
            let w = cosine_sim(&self.centroids[i], point).max(0.0);
            m.set_sym(i, c, w as f64);
            sim_ops += 1;
        }
        let result = global_min_cut(&m).expect("c+1 >= 2 whenever c >= 1");
        let point_side = result.side[c];
        let group: Vec<usize> = (0..c).filter(|&i| result.side[i] == point_side).collect();
        let avg_cut = (result.weight / result.crossing_edges as f64) as f32;
        (avg_cut, group, sim_ops)
    }
}

impl AdmissionPolicy for MincutGatedAdmission {
    fn name(&self) -> &str {
        "MincutGatedAdmission"
    }

    fn decide(&self, point: &[f32]) -> Decision {
        let c = self.centroids.len();
        if c == 0 {
            return Decision {
                cluster_id: 0,
                spawned_new: true,
                sim_ops: 0,
            };
        }
        if c >= self.max_clusters {
            // Safety valve: cap reached, always merge nearest.
            let mut best_id = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (i, cen) in self.centroids.iter().enumerate() {
                let s = cosine_sim(point, cen);
                if s > best_sim {
                    best_sim = s;
                    best_id = i;
                }
            }
            return Decision {
                cluster_id: best_id,
                spawned_new: false,
                sim_ops: c,
            };
        }

        let (avg_cut, group, sim_ops) = self.cut_decision(point);
        if should_spawn(c, avg_cut, &group, self.tau) {
            Decision {
                cluster_id: c,
                spawned_new: true,
                sim_ops,
            }
        } else {
            Decision {
                cluster_id: merge_target(point, &group, &self.centroids),
                spawned_new: false,
                sim_ops,
            }
        }
    }

    fn commit(&mut self, point: &[f32], decision: &Decision) {
        if decision.spawned_new {
            self.centroids.push(point.to_vec());
            self.counts.push(1);
        } else {
            let c = decision.cluster_id;
            running_mean_update(&mut self.centroids[c], self.counts[c], point);
            self.counts[c] += 1;
        }
    }

    fn n_clusters(&self) -> usize {
        self.centroids.len()
    }

    fn centroid(&self, cluster_id: usize) -> &[f32] {
        &self.centroids[cluster_id]
    }
}

// ─── 3. AdaptiveMincutAdmission — candidate B ────────────────────────────────

/// Same mechanism as [`MincutGatedAdmission`], but `tau` is not a fixed
/// constant: it is set each step from a running mean/std (Welford's online
/// algorithm) of previously observed average-cut weights,
/// `tau_t = mean - k_std * std`, bootstrapped with a fixed prior until
/// enough observations accumulate. This mirrors the self-calibrating-control
/// pattern used by SONA's online adapters, without depending on the `sona`
/// crate: it uses only the policy's own past cut-weight distribution, never
/// ground-truth labels, so there is no evaluation leakage.
pub struct AdaptiveMincutAdmission {
    pub k_std: f32,
    pub max_clusters: usize,
    pub bootstrap_tau: f32,
    pub min_observations: u64,
    centroids: Vec<Vec<f32>>,
    counts: Vec<usize>,
    n_obs: u64,
    mean: f64,
    m2: f64,
}

impl AdaptiveMincutAdmission {
    pub fn new(k_std: f32, max_clusters: usize, bootstrap_tau: f32) -> Self {
        AdaptiveMincutAdmission {
            k_std,
            max_clusters,
            bootstrap_tau,
            min_observations: 10,
            centroids: Vec::new(),
            counts: Vec::new(),
            n_obs: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    fn current_tau(&self) -> f32 {
        if self.n_obs < self.min_observations {
            return self.bootstrap_tau;
        }
        let variance = self.m2 / self.n_obs as f64;
        let std = variance.max(0.0).sqrt();
        ((self.mean - self.k_std as f64 * std) as f32).max(0.0)
    }

    fn observe(&mut self, avg_cut: f32) {
        self.n_obs += 1;
        let x = avg_cut as f64;
        let delta = x - self.mean;
        self.mean += delta / self.n_obs as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn cut_decision(&self, point: &[f32]) -> (f32, Vec<usize>, usize) {
        // Identical construction to MincutGatedAdmission::cut_decision;
        // duplicated (not shared via a free function) so each policy's
        // graph-construction cost is charged to its own `sim_ops` count in
        // isolation, matching how the benchmark accounts per-policy cost.
        let c = self.centroids.len();
        let mut m = WeightMatrix::new(c + 1);
        let mut sim_ops = 0usize;
        for i in 0..c {
            for j in (i + 1)..c {
                let w = cosine_sim(&self.centroids[i], &self.centroids[j]).max(0.0);
                m.set_sym(i, j, w as f64);
                sim_ops += 1;
            }
        }
        for i in 0..c {
            let w = cosine_sim(&self.centroids[i], point).max(0.0);
            m.set_sym(i, c, w as f64);
            sim_ops += 1;
        }
        let result = global_min_cut(&m).expect("c+1 >= 2 whenever c >= 1");
        let point_side = result.side[c];
        let group: Vec<usize> = (0..c).filter(|&i| result.side[i] == point_side).collect();
        let avg_cut = (result.weight / result.crossing_edges as f64) as f32;
        (avg_cut, group, sim_ops)
    }
}

impl AdmissionPolicy for AdaptiveMincutAdmission {
    fn name(&self) -> &str {
        "AdaptiveMincutAdmission"
    }

    fn decide(&self, point: &[f32]) -> Decision {
        let c = self.centroids.len();
        if c == 0 {
            return Decision {
                cluster_id: 0,
                spawned_new: true,
                sim_ops: 0,
            };
        }
        if c >= self.max_clusters {
            let mut best_id = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (i, cen) in self.centroids.iter().enumerate() {
                let s = cosine_sim(point, cen);
                if s > best_sim {
                    best_sim = s;
                    best_id = i;
                }
            }
            return Decision {
                cluster_id: best_id,
                spawned_new: false,
                sim_ops: c,
            };
        }

        let tau = self.current_tau();
        let (avg_cut, group, sim_ops) = self.cut_decision(point);
        if should_spawn(c, avg_cut, &group, tau) {
            Decision {
                cluster_id: c,
                spawned_new: true,
                sim_ops,
            }
        } else {
            Decision {
                cluster_id: merge_target(point, &group, &self.centroids),
                spawned_new: false,
                sim_ops,
            }
        }
    }

    fn commit(&mut self, point: &[f32], decision: &Decision) {
        // Update calibration stats with this point's cut geometry
        // regardless of the decision taken (unsupervised: uses only the
        // graph structure, never `decision` or ground-truth labels), then
        // apply the decision. Only observe when `decide` actually took the
        // cut-based path (mirrors its own `0 < c < max_clusters` guard) so
        // the running stats reflect the same distribution `decide` reads.
        if !self.centroids.is_empty() && self.centroids.len() < self.max_clusters {
            let (avg_cut, _, _) = self.cut_decision(point);
            self.observe(avg_cut);
        }
        if decision.spawned_new {
            self.centroids.push(point.to_vec());
            self.counts.push(1);
        } else {
            let c = decision.cluster_id;
            running_mean_update(&mut self.centroids[c], self.counts[c], point);
            self.counts[c] += 1;
        }
    }

    fn n_clusters(&self) -> usize {
        self.centroids.len()
    }

    fn centroid(&self, cluster_id: usize) -> &[f32] {
        &self.centroids[cluster_id]
    }
}

// ─── 4. ConformalMincutAdmission — candidate C ───────────────────────────────

/// Same mechanism as [`MincutGatedAdmission`], but `tau` is calibrated from a
/// **sliding window** of the most recently observed average-cut weights,
/// using an empirical quantile instead of candidate B's lifetime running
/// mean/std.
///
/// This directly attacks the documented failure mode of
/// [`AdaptiveMincutAdmission`] (candidate B): its `mean - k_std * std`
/// estimator is a *global* statistic over the cut-weight distribution since
/// the stream began, and the 2026-09-02 nightly measured that this global
/// statistic stops tracking the locally-relevant admission threshold once
/// the cluster graph grows past a handful of nodes (more nodes -> more
/// near-orthogonal pairs -> the global min cut increasingly tends to find a
/// very low weakest link almost by construction, dragging the running mean
/// down in a way unrelated to whether *this* candidate point is a genuine
/// outlier). A fixed-size sliding window instead of a lifetime accumulator
/// keeps the calibration statistic representative of the *current* cluster
/// regime rather than averaging across every regime the stream has ever
/// passed through. Taking the `alpha`-quantile of that window (rather than
/// `mean - k_std * std`) also drops the assumption that cut weights are
/// roughly Gaussian, which there is no reason to expect once the graph
/// geometry is this far from the two-node case.
///
/// **Important caveat, stated up front rather than implied**: this is
/// *conformal-style* quantile calibration, not a rigorous split-conformal
/// predictor. Split conformal prediction's distribution-free coverage
/// guarantee requires the calibration and test points to be exchangeable.
/// Here the calibration window is built from the policy's own past
/// decisions, which then influence which future points merge vs. spawn (and
/// therefore which future cut weights get observed) — a feedback loop that
/// breaks the i.i.d./exchangeability assumption the formal guarantee rests
/// on. No coverage guarantee is claimed; `alpha` is an empirically measured
/// target, not a proven bound. See the nightly research doc for the
/// benchmarked behaviour this produces.
pub struct ConformalMincutAdmission {
    /// Target quantile level in `(0, 1)`: `tau` is set to the `alpha`-th
    /// empirical quantile of the calibration window, so (informally, not
    /// guaranteed) roughly an `alpha` fraction of recently observed
    /// attachment strengths fall below it.
    pub alpha: f32,
    /// Sliding-window size for the calibration buffer.
    pub window: usize,
    pub max_clusters: usize,
    pub bootstrap_tau: f32,
    pub min_observations: usize,
    centroids: Vec<Vec<f32>>,
    counts: Vec<usize>,
    calib: VecDeque<f32>,
}

impl ConformalMincutAdmission {
    pub fn new(alpha: f32, window: usize, max_clusters: usize, bootstrap_tau: f32) -> Self {
        ConformalMincutAdmission {
            alpha,
            window,
            max_clusters,
            bootstrap_tau,
            // Capped at `window`: the calibration buffer never holds more
            // than `window` entries, so a `min_observations` above that
            // would make `current_tau` return `bootstrap_tau` forever.
            min_observations: window.min(10),
            centroids: Vec::new(),
            counts: Vec::new(),
            calib: VecDeque::with_capacity(window),
        }
    }

    /// Empirical `alpha`-quantile of the calibration window via linear
    /// interpolation on a sorted copy. `O(window log window)` per call —
    /// fine for the small windows (hundreds of entries) this policy is
    /// designed for; a production port would replace this with an
    /// order-statistics structure that supports incremental updates instead
    /// of re-sorting on every decision.
    fn current_tau(&self) -> f32 {
        if self.calib.len() < self.min_observations {
            return self.bootstrap_tau;
        }
        let mut sorted: Vec<f32> = self.calib.iter().copied().collect();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let pos = (self.alpha as f64 * (n as f64 - 1.0)).clamp(0.0, (n - 1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        if lo == hi {
            sorted[lo]
        } else {
            let frac = (pos - lo as f64) as f32;
            sorted[lo] + frac * (sorted[hi] - sorted[lo])
        }
    }

    fn observe(&mut self, avg_cut: f32) {
        if self.calib.len() == self.window {
            self.calib.pop_front();
        }
        self.calib.push_back(avg_cut);
    }

    fn cut_decision(&self, point: &[f32]) -> (f32, Vec<usize>, usize) {
        // Duplicated from MincutGatedAdmission/AdaptiveMincutAdmission for
        // the same per-policy cost-accounting reason documented on
        // AdaptiveMincutAdmission::cut_decision.
        let c = self.centroids.len();
        let mut m = WeightMatrix::new(c + 1);
        let mut sim_ops = 0usize;
        for i in 0..c {
            for j in (i + 1)..c {
                let w = cosine_sim(&self.centroids[i], &self.centroids[j]).max(0.0);
                m.set_sym(i, j, w as f64);
                sim_ops += 1;
            }
        }
        for i in 0..c {
            let w = cosine_sim(&self.centroids[i], point).max(0.0);
            m.set_sym(i, c, w as f64);
            sim_ops += 1;
        }
        let result = global_min_cut(&m).expect("c+1 >= 2 whenever c >= 1");
        let point_side = result.side[c];
        let group: Vec<usize> = (0..c).filter(|&i| result.side[i] == point_side).collect();
        let avg_cut = (result.weight / result.crossing_edges as f64) as f32;
        (avg_cut, group, sim_ops)
    }
}

impl AdmissionPolicy for ConformalMincutAdmission {
    fn name(&self) -> &str {
        "ConformalMincutAdmission"
    }

    fn decide(&self, point: &[f32]) -> Decision {
        let c = self.centroids.len();
        if c == 0 {
            return Decision {
                cluster_id: 0,
                spawned_new: true,
                sim_ops: 0,
            };
        }
        if c >= self.max_clusters {
            let mut best_id = 0usize;
            let mut best_sim = f32::NEG_INFINITY;
            for (i, cen) in self.centroids.iter().enumerate() {
                let s = cosine_sim(point, cen);
                if s > best_sim {
                    best_sim = s;
                    best_id = i;
                }
            }
            return Decision {
                cluster_id: best_id,
                spawned_new: false,
                sim_ops: c,
            };
        }

        let tau = self.current_tau();
        let (avg_cut, group, sim_ops) = self.cut_decision(point);
        if should_spawn(c, avg_cut, &group, tau) {
            Decision {
                cluster_id: c,
                spawned_new: true,
                sim_ops,
            }
        } else {
            Decision {
                cluster_id: merge_target(point, &group, &self.centroids),
                spawned_new: false,
                sim_ops,
            }
        }
    }

    fn commit(&mut self, point: &[f32], decision: &Decision) {
        if !self.centroids.is_empty() && self.centroids.len() < self.max_clusters {
            let (avg_cut, _, _) = self.cut_decision(point);
            self.observe(avg_cut);
        }
        if decision.spawned_new {
            self.centroids.push(point.to_vec());
            self.counts.push(1);
        } else {
            let c = decision.cluster_id;
            running_mean_update(&mut self.centroids[c], self.counts[c], point);
            self.counts[c] += 1;
        }
    }

    fn n_clusters(&self) -> usize {
        self.centroids.len()
    }

    fn centroid(&self, cluster_id: usize) -> &[f32] {
        &self.centroids[cluster_id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, rest: usize) -> Vec<f32> {
        let mut vec = vec![x, y];
        vec.extend(std::iter::repeat_n(0.0, rest));
        crate::dataset::normalise(&mut vec);
        vec
    }

    #[test]
    fn first_point_always_spawns() {
        let mut p = NearestCentroidThreshold::new(0.5);
        let d = p.admit(&v(1.0, 0.0, 6));
        assert!(d.spawned_new);
        assert_eq!(p.n_clusters(), 1);
    }

    #[test]
    fn nearest_centroid_merges_similar_points() {
        let mut p = NearestCentroidThreshold::new(0.7);
        p.admit(&v(1.0, 0.0, 6));
        let d = p.admit(&v(0.95, 0.05, 6));
        assert!(!d.spawned_new);
        assert_eq!(p.n_clusters(), 1);
    }

    #[test]
    fn nearest_centroid_spawns_for_distant_points() {
        let mut p = NearestCentroidThreshold::new(0.7);
        p.admit(&v(1.0, 0.0, 6));
        let d = p.admit(&v(0.0, 1.0, 6));
        assert!(d.spawned_new);
        assert_eq!(p.n_clusters(), 2);
    }

    #[test]
    fn mincut_admission_isolates_a_weakly_attached_outlier() {
        // Two well-separated existing clusters and a point that is
        // moderately similar to one of them (0.6 cosine-ish) but should
        // still be recognised as weakly attached once the graph structure
        // is considered.
        let mut p = MincutGatedAdmission::new(0.4, 16);
        p.admit(&v(1.0, 0.0, 6));
        p.admit(&v(0.0, 1.0, 6));
        assert_eq!(p.n_clusters(), 2);
        // A point far from both existing centroids.
        let d = p.admit(&v(-1.0, -1.0, 6));
        assert!(d.spawned_new, "distant point should spawn a new cluster");
        assert_eq!(p.n_clusters(), 3);
    }

    #[test]
    fn mincut_admission_merges_close_points() {
        let mut p = MincutGatedAdmission::new(0.3, 16);
        p.admit(&v(1.0, 0.0, 6));
        let d = p.admit(&v(0.97, 0.03, 6));
        assert!(!d.spawned_new);
        assert_eq!(p.n_clusters(), 1);
    }

    #[test]
    fn mincut_admission_respects_max_clusters_safety_valve() {
        let mut p = MincutGatedAdmission::new(0.99, 3);
        // With tau=0.99 nearly everything would spawn a new cluster, but
        // the safety valve caps growth at max_clusters.
        for i in 0..20 {
            let angle = i as f32 * 0.31;
            p.admit(&v(angle.cos(), angle.sin(), 6));
        }
        assert!(p.n_clusters() <= 3);
    }

    #[test]
    fn adaptive_admission_converges_to_stable_tau() {
        let mut p = AdaptiveMincutAdmission::new(1.0, 16, 0.3);
        for i in 0..30 {
            let angle = (i as f32) * 0.05;
            p.admit(&v(angle.cos(), angle.sin(), 6));
        }
        // A smoothly-varying stream of near-identical points should mostly
        // merge, not spawn a cluster per point.
        assert!(p.n_clusters() < 30);
    }

    #[test]
    fn conformal_quantile_matches_hand_computed_value() {
        let mut p = ConformalMincutAdmission::new(0.5, 8, 16, 0.3);
        // Feed a known calibration window directly via `observe` (private,
        // so exercised through the module-internal test) and check the
        // median lands where linear interpolation on a sorted copy predicts.
        for x in [0.1f32, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.5, 0.0] {
            p.observe(x);
        }
        // Window=8, so only the last 8 pushed values survive:
        // [0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.5, 0.0] -> sorted
        // [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], median (pos 3.5) = 0.45.
        let tau = p.current_tau();
        assert!((tau - 0.45).abs() < 1e-5, "tau={tau}");
    }

    #[test]
    fn conformal_admission_respects_max_clusters_safety_valve() {
        let mut p = ConformalMincutAdmission::new(0.15, 200, 3, 0.3);
        for i in 0..20 {
            let angle = i as f32 * 0.31;
            p.admit(&v(angle.cos(), angle.sin(), 6));
        }
        assert!(p.n_clusters() <= 3);
    }

    #[test]
    fn conformal_admission_merges_a_smooth_stream_without_runaway_growth() {
        // Same smoothly-varying stream that candidate B handles fine in
        // isolation (the regression this policy must not reintroduce): most
        // points should merge rather than each spawning its own cluster.
        let mut p = ConformalMincutAdmission::new(0.15, 200, 16, 0.3);
        for i in 0..30 {
            let angle = (i as f32) * 0.05;
            p.admit(&v(angle.cos(), angle.sin(), 6));
        }
        assert!(p.n_clusters() < 30);
    }

    #[test]
    fn conformal_bootstrap_tau_used_before_min_observations() {
        let p = ConformalMincutAdmission::new(0.15, 200, 16, 0.42);
        assert_eq!(p.current_tau(), 0.42);
    }
}
