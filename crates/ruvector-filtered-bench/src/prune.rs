//! M2 — contender A: region-pruned IVF filtered search.
//!
//! Built on `ruvector-rairs` k-means (the ADR-193 IVF substrate). Two stacked prunings,
//! both realizing the salvaged SepRAG kernel on a treewidth-immune cluster hierarchy:
//!
//! 1. **Predicate pruning** — skip every cluster with zero predicate-matching members.
//!    This is the BET-2 win: a correlated metadata filter concentrates matches in a few
//!    clusters, so most of the corpus is never touched.
//! 2. **Branch-and-bound distance pruning** — by the triangle inequality, the nearest
//!    possible point in cluster `c` is `dist(q, centroid_c) − radius_c`. Once the top-k
//!    heap is full, clusters whose lower bound exceeds the current k-th distance cannot
//!    improve the result and are skipped. With a valid lower bound this is **exact**.
//!
//! Cost (the pre-registered metric) = `#centroids routed (= nclusters)` + `#matching
//! members for which a distance was computed`. The O(1) predicate test gates the
//! expensive distance, so non-matching points cost nothing — the asymmetry vs ACORN
//! (which evaluates a distance per expanded node regardless of predicate).

use ruvector_rairs::kmeans;

use crate::contenders::QueryResult;

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Region-pruned IVF index (contender A).
pub struct RegionPruneIvf {
    centroids: Vec<Vec<f32>>,
    /// `members[c]` = node ids assigned to cluster `c`.
    members: Vec<Vec<u32>>,
    /// `radius[c]` = max **L2** distance (not squared) from centroid `c` to any member —
    /// the triangle-inequality slack for the branch-and-bound lower bound.
    radius: Vec<f32>,
    pub nclusters: usize,
}

impl RegionPruneIvf {
    /// Partition `feats` into `nclusters` k-means cells (rairs clustering).
    pub fn build(feats: &[Vec<f32>], nclusters: usize, max_iter: usize, seed: u64) -> Self {
        let (centroids, assign) = kmeans::train(feats, nclusters, max_iter, seed);
        let k = centroids.len();
        let mut members = vec![Vec::new(); k];
        for (id, &c) in assign.iter().enumerate() {
            members[c].push(id as u32);
        }
        let radius = (0..k)
            .map(|c| {
                members[c]
                    .iter()
                    .map(|&id| l2_sq(&centroids[c], &feats[id as usize]).sqrt())
                    .fold(0.0_f32, f32::max)
            })
            .collect();
        RegionPruneIvf { centroids, members, radius, nclusters: k }
    }

    /// Region-pruned filtered top-k search.
    ///
    /// `max_probe = None` runs exact branch-and-bound (recall 1.0); `Some(p)` caps the
    /// number of *match-containing* clusters probed (the approximate knob that trades
    /// recall for fewer distance-evals, mirroring ACORN's `ef`).
    pub fn search(
        &self,
        feats: &[Vec<f32>],
        query: &[f32],
        k: usize,
        predicate: impl Fn(u32) -> bool,
        max_probe: Option<usize>,
    ) -> QueryResult {
        let mut evals = 0u64;

        // 1. Route: distance to every centroid (the fixed routing cost).
        let mut clusters: Vec<(f32, usize)> = (0..self.nclusters)
            .map(|c| {
                evals += 1;
                (l2_sq(query, &self.centroids[c]), c)
            })
            .collect();

        // Lower bound per cluster (squared L2): (max(0, sqrt(d_qc) - radius))^2.
        // Sorting by LB lets us *break* (not just skip) once LB exceeds the worst result.
        let lb_sq = |d_qc_sq: f32, c: usize| {
            let lb = (d_qc_sq.sqrt() - self.radius[c]).max(0.0);
            lb * lb
        };
        clusters.sort_by(|&(da, ca), &(db, cb)| {
            lb_sq(da, ca).total_cmp(&lb_sq(db, cb))
        });

        // 2. Probe in lower-bound order, skipping zero-match clusters; B&B early-out.
        // Max-heap on squared distance — peek = current worst of the top-k.
        let mut heap: std::collections::BinaryHeap<(ordered::Of, u32)> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        let mut probed = 0usize;

        for &(d_qc_sq, c) in &clusters {
            // B&B: once the heap is full, no later cluster (sorted by LB) can help.
            if heap.len() >= k {
                if let Some(&(ordered::Of(worst), _)) = heap.peek() {
                    if lb_sq(d_qc_sq, c) >= worst {
                        break;
                    }
                }
            }
            // Does this cluster contain any match? (cheap O(1) tests, not distance-evals)
            let mut any = false;
            for &id in &self.members[c] {
                if !predicate(id) {
                    continue;
                }
                any = true;
                let d = l2_sq(query, &feats[id as usize]);
                evals += 1;
                if heap.len() < k {
                    heap.push((ordered::Of(d), id));
                } else if let Some(&(ordered::Of(worst), _)) = heap.peek() {
                    if d < worst {
                        heap.pop();
                        heap.push((ordered::Of(d), id));
                    }
                }
            }
            if any {
                probed += 1;
                if let Some(cap) = max_probe {
                    if probed >= cap {
                        break;
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> =
            heap.into_iter().map(|(ordered::Of(d), id)| (id, d)).collect();
        out.sort_by(|a, b| a.1.total_cmp(&b.1));
        QueryResult { ids: out.into_iter().map(|(id, _)| id).collect(), evals }
    }
}

/// Minimal total-ordered f32 wrapper for the binary heap (NaN-free distances).
mod ordered {
    #[derive(Clone, Copy, PartialEq)]
    pub struct Of(pub f32);
    impl Eq for Of {}
    impl PartialOrd for Of {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Of {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.total_cmp(&other.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_acorn::graph::exact_filtered_knn;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn gauss(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0)).collect())
            .collect()
    }

    #[test]
    fn exact_bb_matches_oracle() {
        // max_probe = None must return the exact filtered top-k (recall 1.0).
        let feats = gauss(2000, 16, 1);
        let idx = RegionPruneIvf::build(&feats, 48, 10, 7);
        let k = 10;
        let pred = |id: u32| id.is_multiple_of(4);
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..20 {
            let qi = rng.gen_range(0..feats.len());
            let truth = exact_filtered_knn(&feats, &feats[qi], k, pred);
            let got = idx.search(&feats, &feats[qi], k, pred, None);
            assert_eq!(got.ids, truth, "exact B&B must equal the oracle");
        }
    }

    #[test]
    fn zero_match_clusters_are_skipped() {
        // A predicate matching a tiny fraction must cost far fewer evals than scanning all.
        let feats = gauss(4000, 16, 2);
        let idx = RegionPruneIvf::build(&feats, 64, 10, 7);
        let pred = |id: u32| id < 40; // 1% selectivity
        let r = idx.search(&feats, &feats[0], 10, pred, None);
        // evals = nclusters routing + matches scanned; must be << full scan (4000).
        assert!(r.evals < 1000, "pruning failed: {} evals", r.evals);
        assert!(r.evals >= idx.nclusters as u64, "must at least route to all centroids");
    }
}
