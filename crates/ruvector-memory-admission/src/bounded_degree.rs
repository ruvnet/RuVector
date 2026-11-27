//! Bounded-degree admission — tonight's candidate, answering ADR-344's own
//! Next Research item 1.
//!
//! `policy::MincutGatedAdmission` (ADR-344, accepted) builds its min-cut
//! graph over *all* `C` existing centroids: O(C^2) to build, O(C^3) for
//! Stoer-Wagner. That nightly's Next Research explicitly asked whether
//! restricting the graph to each point's `K` nearest centroids (not all
//! `C`) preserves the quality gain at lower cost. [`BoundedDegreeAdmission`]
//! answers that: identical min-cut-gated mechanism, but the graph is built
//! over only the `min(K, C)` centroids most similar (cosine) to the
//! incoming point.
//!
//! When `k_neighbors >= C` this selects every centroid and is
//! mathematically identical to [`crate::policy::MincutGatedAdmission`] —
//! checked directly by `bounded_degree_matches_full_graph_when_k_covers_all_centroids`
//! below, which is this module's load-bearing correctness regression test.

use crate::cosine_sim;
use crate::mincut::{global_min_cut, WeightMatrix};
use crate::policy::{merge_target, running_mean_update, should_spawn, AdmissionPolicy, Decision};

/// Degree-bounded variant of the ADR-344 min-cut-gated admission mechanism.
///
/// Per-insertion cost: selecting the `K` nearest centroids is O(C * K)
/// (bounded insertion into a size-K running top-list — no external
/// dependency, no full sort), then the min-cut graph build is O(K^2) and
/// Stoer-Wagner is O(K^3). Total: O(C*K + K^3), linear in `C` for fixed
/// `K` — not literally flat in `C`, but asymptotically far cheaper than the
/// full graph's O(C^3) once `C` grows past `K`. This is the precise claim
/// this nightly's benchmark measures; see
/// `docs/research/nightly/2026-11-18-bounded-degree-mincut-admission/README.md`.
pub struct BoundedDegreeAdmission {
    pub tau: f32,
    pub max_clusters: usize,
    pub k_neighbors: usize,
    centroids: Vec<Vec<f32>>,
    counts: Vec<usize>,
}

impl BoundedDegreeAdmission {
    pub fn new(tau: f32, max_clusters: usize, k_neighbors: usize) -> Self {
        assert!(k_neighbors >= 1, "k_neighbors must be at least 1");
        BoundedDegreeAdmission {
            tau,
            max_clusters,
            k_neighbors,
            centroids: Vec::new(),
            counts: Vec::new(),
        }
    }

    /// Benchmark-only constructor: seed the policy directly with a fixed set
    /// of existing centroids (count 1 each), bypassing `admit`. See
    /// `policy::MincutGatedAdmission::from_centroids` — same purpose, same
    /// caveat (not for use in `decide`'s normal tau-gated path).
    pub fn from_centroids(centroids: Vec<Vec<f32>>, k_neighbors: usize) -> Self {
        let counts = vec![1usize; centroids.len()];
        BoundedDegreeAdmission {
            tau: f32::INFINITY,
            max_clusters: usize::MAX,
            k_neighbors,
            centroids,
            counts,
        }
    }

    /// Selects up to `k_neighbors` centroids most similar to `point`
    /// (descending cosine similarity). O(C * K): each of the C candidates
    /// does an O(K) scan-and-shift insertion into a size-capped list.
    /// Returns (selected original centroid ids, cost of the C similarity
    /// evaluations that drove the selection).
    fn select_neighbors(&self, point: &[f32]) -> (Vec<usize>, usize) {
        let k = self.k_neighbors.min(self.centroids.len());
        let mut best: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
        for (i, c) in self.centroids.iter().enumerate() {
            let s = cosine_sim(point, c);
            if best.len() < k || s > best[best.len() - 1].1 {
                let pos = best
                    .iter()
                    .position(|&(_, bs)| s > bs)
                    .unwrap_or(best.len());
                best.insert(pos, (i, s));
                if best.len() > k {
                    best.truncate(k);
                }
            }
        }
        let select_cost = self.centroids.len();
        (best.into_iter().map(|(i, _)| i).collect(), select_cost)
    }

    /// Build the bounded-degree graph (selected centroids + point), run
    /// global min cut, and return (avg_crossing_weight, point-side group as
    /// *original* centroid ids, total sim_ops, number of centroids the cut
    /// actually saw — i.e. `min(k_neighbors, C)`).
    fn cut_decision(&self, point: &[f32]) -> (f32, Vec<usize>, usize, usize) {
        let (selected, select_cost) = self.select_neighbors(point);
        let k = selected.len();
        let mut m = WeightMatrix::new(k + 1);
        let mut pair_cost = 0usize;
        for a in 0..k {
            for b in (a + 1)..k {
                let w =
                    cosine_sim(&self.centroids[selected[a]], &self.centroids[selected[b]]).max(0.0);
                m.set_sym(a, b, w as f64);
                pair_cost += 1;
            }
        }
        for (a, &orig) in selected.iter().enumerate() {
            let w = cosine_sim(&self.centroids[orig], point).max(0.0);
            m.set_sym(a, k, w as f64);
        }
        let result = global_min_cut(&m).expect("k+1 >= 2 whenever k >= 1");
        let point_side = result.side[k];
        let group: Vec<usize> = (0..k)
            .filter(|&a| result.side[a] == point_side)
            .map(|a| selected[a])
            .collect();
        let avg_cut = (result.weight / result.crossing_edges as f64) as f32;
        (avg_cut, group, select_cost + pair_cost, k)
    }
}

impl AdmissionPolicy for BoundedDegreeAdmission {
    fn name(&self) -> &str {
        "BoundedDegreeAdmission"
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
            // Same computational safety valve as MincutGatedAdmission.
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

        let (avg_cut, group, sim_ops, effective_c) = self.cut_decision(point);
        // `effective_c` (the local graph's node count, min(K, C)), not the
        // global `c`, is what determines whether this is the degenerate
        // 2-node cut case — see `policy::should_spawn`'s own doc comment.
        if should_spawn(effective_c, avg_cut, &group, self.tau) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::MincutGatedAdmission;

    fn v(x: f32, y: f32, rest: usize) -> Vec<f32> {
        let mut vec = vec![x, y];
        vec.extend(std::iter::repeat_n(0.0, rest));
        crate::dataset::normalise(&mut vec);
        vec
    }

    #[test]
    fn first_point_always_spawns() {
        let mut p = BoundedDegreeAdmission::new(0.4, 16, 8);
        let d = p.admit(&v(1.0, 0.0, 6));
        assert!(d.spawned_new);
        assert_eq!(p.n_clusters(), 1);
    }

    #[test]
    fn isolates_a_distant_outlier() {
        let mut p = BoundedDegreeAdmission::new(0.4, 16, 8);
        p.admit(&v(1.0, 0.0, 6));
        p.admit(&v(0.0, 1.0, 6));
        let d = p.admit(&v(-1.0, -1.0, 6));
        assert!(d.spawned_new, "distant point should spawn a new cluster");
        assert_eq!(p.n_clusters(), 3);
    }

    #[test]
    fn merges_close_points() {
        let mut p = BoundedDegreeAdmission::new(0.3, 16, 8);
        p.admit(&v(1.0, 0.0, 6));
        let d = p.admit(&v(0.97, 0.03, 6));
        assert!(!d.spawned_new);
        assert_eq!(p.n_clusters(), 1);
    }

    #[test]
    fn respects_max_clusters_safety_valve() {
        let mut p = BoundedDegreeAdmission::new(0.99, 3, 8);
        for i in 0..20 {
            let angle = i as f32 * 0.31;
            p.admit(&v(angle.cos(), angle.sin(), 6));
        }
        assert!(p.n_clusters() <= 3);
    }

    /// Load-bearing correctness regression: with `k_neighbors >= C` always
    /// (set well above every cluster count this test reaches), the bounded
    /// graph *is* the full graph, so every decision must be bit-identical
    /// to `MincutGatedAdmission` fed the same deterministic point stream.
    #[test]
    fn bounded_degree_matches_full_graph_when_k_covers_all_centroids() {
        let points: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                let angle = i as f32 * 0.41;
                v(angle.cos(), angle.sin(), 6)
            })
            .collect();

        let mut full = MincutGatedAdmission::new(0.35, 64);
        let mut bounded = BoundedDegreeAdmission::new(0.35, 64, 64); // K=64 > any C reached here

        for p in &points {
            let d_full = full.admit(p);
            let d_bounded = bounded.admit(p);
            assert_eq!(
                d_full.spawned_new, d_bounded.spawned_new,
                "spawn decision diverged with K covering all centroids"
            );
            assert_eq!(
                d_full.cluster_id, d_bounded.cluster_id,
                "cluster assignment diverged with K covering all centroids"
            );
        }
        assert_eq!(full.n_clusters(), bounded.n_clusters());
    }

    /// With `K` strictly smaller than the eventual cluster count, bounded
    /// and full-graph admission are allowed to diverge (that is the whole
    /// point of the approximation) — but bounded admission must still be a
    /// *valid* admission policy: every point lands somewhere, cluster count
    /// stays bounded, and centroids stay unit-norm.
    #[test]
    fn bounded_degree_stays_well_formed_when_k_is_smaller_than_c() {
        let mut p = BoundedDegreeAdmission::new(0.35, 32, 4);
        let mut n_admitted = 0usize;
        for i in 0..100 {
            let angle = i as f32 * 0.7;
            p.admit(&v(angle.cos(), angle.sin(), 6));
            n_admitted += 1;
        }
        assert_eq!(n_admitted, 100);
        assert!(p.n_clusters() >= 1 && p.n_clusters() <= 32);
        for c in 0..p.n_clusters() {
            let norm: f32 = p.centroid(c).iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "centroid {c} should stay unit-norm"
            );
        }
    }

    #[test]
    fn decide_without_commit_does_not_mutate_state() {
        let mut p = BoundedDegreeAdmission::new(0.35, 32, 8);
        for i in 0..30 {
            let angle = i as f32 * 0.31;
            p.admit(&v(angle.cos(), angle.sin(), 6));
        }
        let before = p.n_clusters();
        for i in 0..10 {
            let angle = i as f32 * 1.7;
            let _ = p.decide(&v(angle.cos(), angle.sin(), 6));
        }
        assert_eq!(
            p.n_clusters(),
            before,
            "decide() must not mutate policy state"
        );
    }
}
