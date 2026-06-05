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

/// The shared IVF substrate (centroids + inverted lists) built **once** from a seeded k-means, then
/// reused to construct every contender for a given `nclusters` — so the expensive clustering is paid
/// once per cell, not once per contender, and all contenders provably share an identical index.
pub struct IvfParts {
    pub centroids: Vec<Vec<f32>>,
    /// Per cluster: `(id, vector)` of its members.
    pub lists: Vec<Vec<(usize, Vec<f32>)>>,
}

/// Build the shared IVF substrate (`ruvector-rairs` k-means, identical to `IvfFlat::train`).
pub fn build_ivf(corpus: &[Vec<f32>], nclusters: usize, max_iter: usize, seed: u64) -> IvfParts {
    assert!(!corpus.is_empty(), "empty corpus");
    let k = nclusters.min(corpus.len()).max(1);
    let (centroids, assignments) = kmeans::train(corpus, k, max_iter, seed);
    let kc = centroids.len();
    let mut lists: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); kc];
    for (i, v) in corpus.iter().enumerate() {
        lists[assignments[i]].push((i, v.clone()));
    }
    IvfParts { centroids, lists }
}

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

/// Offer candidate `(id, d)` to a bounded top-`k` max-heap: insert while under capacity, else
/// replace the current worst iff `d` is closer. Shared by both probe strategies so they accumulate
/// results identically — only their cluster-visit order/stopping differs.
#[inline]
fn consider(heap: &mut BinaryHeap<Cand>, k: usize, id: usize, d: f32) {
    if heap.len() < k {
        heap.push(Cand { dist: d, id });
    } else if d < heap.peek().unwrap().dist {
        heap.pop();
        heap.push(Cand { dist: d, id });
    }
}

/// Drain a top-`k` heap into an ascending-distance result vector.
fn finalize(heap: BinaryHeap<Cand>) -> Vec<SearchResult> {
    let mut res: Vec<SearchResult> = heap
        .into_iter()
        .map(|c| SearchResult {
            id: c.id,
            distance: c.dist,
        })
        .collect();
    res.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    res
}

impl BnBIvf {
    /// Build over `corpus` using `ruvector-rairs` k-means (`nclusters`, `max_iter`, `seed`).
    /// Using the same `(corpus, nclusters, max_iter, seed)` as `IvfFlat::train` yields identical
    /// centroids — the shared-index guarantee the pre-registration requires.
    pub fn build(corpus: &[Vec<f32>], nclusters: usize, max_iter: usize, seed: u64) -> Self {
        Self::from_parts(&build_ivf(corpus, nclusters, max_iter, seed))
    }

    /// Construct from a pre-built shared [`IvfParts`] (skips re-clustering). Computes the B&B radii.
    pub fn from_parts(parts: &IvfParts) -> Self {
        let centroids = parts.centroids.clone();
        let lists = parts.lists.clone();
        let kc = centroids.len();
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
                member_evals += 1;
                consider(&mut heap, k, *id, l2(q, v));
            }
            probed += 1;
        }

        (finalize(heap), member_evals, probed)
    }

    /// The **steelman B&B**: visit clusters in centroid-distance order (the effective `nprobe`
    /// ordering, so τ tightens fast), but **skip** scanning any cluster the lower bound proves
    /// cannot hold a top-k point (`LB(q,c) ≥ τ`). Unlike [`search`](Self::search)'s global early
    /// `break`, skipping is correctness-safe in *any* visit order (a skipped cluster genuinely
    /// cannot contain a closer point); a global break would be unsound here because a later,
    /// large-radius cluster can have a *smaller* LB than the current one.
    ///
    /// `max_probe` caps the number of clusters **considered** (the apples-to-apples budget against
    /// `nprobe`); LB-skips save member scans within that budget. This is the strongest version of
    /// the bet — if it cannot beat `nprobe`, the bound itself doesn't pay. Returns
    /// `(top-k, member_evals, clusters_considered)`.
    pub fn search_bnb_skip(
        &self,
        q: &[f32],
        k: usize,
        max_probe: Option<usize>,
    ) -> (Vec<SearchResult>, usize, usize) {
        let nclusters = self.centroids.len();
        let mut order: Vec<(f32, usize)> = (0..nclusters)
            .map(|c| (l2(q, &self.centroids[c]), c))
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let cap = max_probe.unwrap_or(nclusters).min(nclusters);

        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
        let mut member_evals = 0usize;
        let mut considered = 0usize;
        for (dc, c) in order {
            if considered >= cap {
                break;
            }
            considered += 1;
            if heap.len() == k {
                let kth = heap.peek().unwrap().dist;
                if (dc - self.radii[c]).max(0.0) >= kth {
                    continue; // LB-skip: provably cannot improve the top-k
                }
            }
            for (id, v) in &self.lists[c] {
                member_evals += 1;
                consider(&mut heap, k, *id, l2(q, v));
            }
        }
        (finalize(heap), member_evals, considered)
    }

    /// The **BET-5 steelman incumbent**: plain `nprobe` list selection, but each member's exact L2 is
    /// computed dim-by-dim and **early-abandoned** the instant the running squared partial exceeds the
    /// current k-th-best (`τ²`). This is *exact* (an abandoned member provably exceeds `τ`, so it
    /// cannot enter the top-k) and is the natural PQ-free within-list pruning the PQ contender must
    /// beat. Returns `(top-k, dims_touched, members)`; the harness charges `dims_touched / D`
    /// full-L2-equivalents (full credit for skipped dims), and reports the dim-prune fraction as the
    /// control on whether exact within-list pruning works at all on concentrated 128-d.
    pub fn search_nprobe_abandon(
        &self,
        q: &[f32],
        k: usize,
        nprobe: usize,
    ) -> (Vec<SearchResult>, usize, usize) {
        let nclusters = self.centroids.len();
        let mut cd: Vec<(f32, usize)> = (0..nclusters)
            .map(|c| (l2(q, &self.centroids[c]), c))
            .collect();
        cd.sort_by(|a, b| a.0.total_cmp(&b.0));
        let np = nprobe.clamp(1, nclusters);

        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
        let mut dims_touched = 0usize;
        let mut members = 0usize;
        for &(_, c) in cd.iter().take(np) {
            for (id, v) in &self.lists[c] {
                members += 1;
                // τ² threshold: finite only when the top-k heap is full.
                let tau_sq = if heap.len() == k {
                    let t = heap.peek().unwrap().dist;
                    t * t
                } else {
                    f32::INFINITY
                };
                let mut acc = 0f32;
                let mut abandoned = false;
                for (x, y) in q.iter().zip(v) {
                    let d = x - y;
                    acc += d * d;
                    dims_touched += 1;
                    if acc > tau_sq {
                        abandoned = true;
                        break;
                    }
                }
                if !abandoned {
                    consider(&mut heap, k, *id, acc.sqrt());
                }
            }
        }
        (finalize(heap), dims_touched, members)
    }

    /// The **plain-IVF incumbent** strategy on this same shared index: visit the `nprobe` nearest
    /// centroids (by centroid distance) and scan **all** their members — no lower-bound ordering,
    /// no early termination. This is exactly `ruvector-rairs::IvfFlat::search`'s algorithm
    /// (validated equal by `instrumented_nprobe_matches_rairs`), instrumented to count member
    /// distance-evals and sharing B&B's centroids/lists so the comparison isolates the probe loop.
    pub fn search_nprobe(
        &self,
        q: &[f32],
        k: usize,
        nprobe: usize,
    ) -> (Vec<SearchResult>, usize, usize) {
        let nclusters = self.centroids.len();
        let mut cd: Vec<(f32, usize)> = (0..nclusters)
            .map(|c| (l2(q, &self.centroids[c]), c))
            .collect();
        cd.sort_by(|a, b| a.0.total_cmp(&b.0));
        let np = nprobe.clamp(1, nclusters);

        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
        let mut member_evals = 0usize;
        for &(_, c) in cd.iter().take(np) {
            for (id, v) in &self.lists[c] {
                member_evals += 1;
                consider(&mut heap, k, *id, l2(q, v));
            }
        }
        (finalize(heap), member_evals, np)
    }
}
