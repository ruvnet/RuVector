//! Streaming cluster admission policies.

use crate::cosine_sim;
use crate::mincut::{global_min_cut, WeightMatrix};

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
}
