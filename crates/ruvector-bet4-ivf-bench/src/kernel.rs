//! `BnBIvf` — the BET 4 contender: an IVF index probed in **lower-bound order with
//! branch-and-bound early termination**, over the same `ruvector-rairs` k-means substrate as
//! the plain-`IvfFlat` incumbent.
//!
//! For a query `q` and cluster `c` with centroid `μ_c` and radius `r_c = max_{v∈c} ‖v−μ_c‖`,
//! the triangle inequality gives a lower bound on the distance to *any* member of `c`:
//! `LB(q,c) = max(0, ‖q−μ_c‖ − r_c)`. Probing clusters in ascending `LB` while tracking the
//! running k-th-best distance `τ`, we may stop the instant `LB(c) ≥ τ`: every not-yet-probed
//! cluster has an even larger `LB`, so none can contain a top-k point. That single break makes
//! full-budget B&B **exact** (recall → 1.0) yet lets it skip clusters a fixed `nprobe` would
//! scan. A `max_probe` cap turns it into an approximate knob (the analogue of `nprobe`) for the
//! matched-recall comparison.

use crate::oracle::l2;
use ruvector_rairs::{kmeans, SearchResult};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// IVF index supporting lower-bound-ordered branch-and-bound probing.
pub struct BnBIvf {
    centroids: Vec<Vec<f32>>,
    /// Per cluster: `(id, vector)` of its members.
    lists: Vec<Vec<(usize, Vec<f32>)>>,
    /// Per cluster: max member distance to its centroid (the B&B radius).
    radii: Vec<f32>,
}

/// Top-k accumulator element. `BinaryHeap` is a max-heap, so the **worst** (largest distance)
/// candidate sits on top and is the one evicted when a closer point arrives.
struct Cand {
    dist: f32,
    id: usize,
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.total_cmp(&o.dist)
    }
}

impl BnBIvf {
    /// Build over `corpus` using `ruvector-rairs` k-means (`nclusters`, `max_iter`, `seed`).
    /// Using the same `(corpus, nclusters, max_iter, seed)` as `IvfFlat::train` yields identical
    /// centroids — the shared-index guarantee the pre-registration requires.
    pub fn build(corpus: &[Vec<f32>], nclusters: usize, max_iter: usize, seed: u64) -> Self {
        assert!(!corpus.is_empty(), "empty corpus");
        let k = nclusters.min(corpus.len()).max(1);
        let (centroids, assignments) = kmeans::train(corpus, k, max_iter, seed);
        let kc = centroids.len();
        let mut lists: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); kc];
        for (i, v) in corpus.iter().enumerate() {
            lists[assignments[i]].push((i, v.clone()));
        }
        let radii: Vec<f32> = (0..kc)
            .map(|c| {
                lists[c]
                    .iter()
                    .map(|(_, v)| l2(v, &centroids[c]))
                    .fold(0.0f32, f32::max)
            })
            .collect();
        Self {
            centroids,
            lists,
            radii,
        }
    }

    /// Number of inverted lists (clusters).
    pub fn num_lists(&self) -> usize {
        self.centroids.len()
    }

    /// Search for the top-`k` neighbours of `q`.
    ///
    /// `max_probe = None` runs full-budget B&B (**exact**); `Some(m)` probes at most `m`
    /// clusters in lower-bound order (approximate, the `nprobe` analogue). Returns the top-k
    /// (ascending distance), the number of **member** distance-evals charged, and the number of
    /// clusters actually probed. The `nclusters` centroid evals (routing) are *not* folded into
    /// the member count — the harness charges them separately and equally to both contenders.
    pub fn search(
        &self,
        q: &[f32],
        k: usize,
        max_probe: Option<usize>,
    ) -> (Vec<SearchResult>, usize, usize) {
        let nclusters = self.centroids.len();
        // Routing: lower bound per cluster, then ascending-LB order.
        let mut order: Vec<(f32, usize)> = (0..nclusters)
            .map(|c| {
                let lb = (l2(q, &self.centroids[c]) - self.radii[c]).max(0.0);
                (lb, c)
            })
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));

        let cap = max_probe.unwrap_or(nclusters).min(nclusters);
        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
        let mut member_evals = 0usize;
        let mut probed = 0usize;

        for (lb, c) in order {
            if probed >= cap {
                break;
            }
            // Branch-and-bound: once the heap is full and the best possible distance in this
            // (and every later) cluster is no better than the current k-th best, stop.
            if heap.len() == k {
                let kth = heap.peek().unwrap().dist;
                if lb >= kth {
                    break;
                }
            }
            for (id, v) in &self.lists[c] {
                let d = l2(q, v);
                member_evals += 1;
                if heap.len() < k {
                    heap.push(Cand { dist: d, id: *id });
                } else if d < heap.peek().unwrap().dist {
                    heap.pop();
                    heap.push(Cand { dist: d, id: *id });
                }
            }
            probed += 1;
        }

        let mut res: Vec<SearchResult> = heap
            .into_iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: c.dist,
            })
            .collect();
        res.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        (res, member_evals, probed)
    }
}
