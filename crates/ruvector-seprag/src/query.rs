//! Query layer (ADR-196, Phase 3).
//!
//! Three k-NN paths, in increasing sophistication, all over the same topology:
//!
//! 1. [`upward`] — exact distances from a vertex to all its elimination-tree
//!    ancestors (the CCH "search space").
//! 2. [`knn_exhaustive`] — pairwise up-search meet for every POI. Exact by the
//!    CH up-down theorem; used to validate customization against the Dijkstra
//!    oracle.
//! 3. [`KnnIndex::knn`] — bucket-based branch-and-bound with admissible
//!    early-stop. Must match (2) exactly while touching far fewer buckets. The
//!    elimination-tree ancestors *are* the separator hierarchy, so stopping once
//!    `d(s -> x) >= delta_k` prunes whole separator regions.

use crate::contraction::{Topology, NONE};
use crate::graph::{cmp_dist_id, NodeId};
use std::collections::HashMap;

/// Exact distances from rank `s` to every upward-reachable vertex (its ancestors
/// in the elimination tree), keyed by rank.
#[must_use]
pub fn upward(topo: &Topology, metric: &crate::customize::Metric, s: u32) -> HashMap<u32, f64> {
    // Collect the upward closure, then relax in ascending rank (a DAG order).
    let mut reach: Vec<u32> = Vec::new();
    let mut seen = vec![false; topo.n];
    let mut stack = vec![s];
    seen[s as usize] = true;
    while let Some(u) = stack.pop() {
        reach.push(u);
        for &x in &topo.up[u as usize] {
            if !seen[x as usize] {
                seen[x as usize] = true;
                stack.push(x);
            }
        }
    }
    reach.sort_unstable();

    let mut dist: HashMap<u32, f64> = HashMap::new();
    dist.insert(s, 0.0);
    for &u in &reach {
        let du = match dist.get(&u) {
            Some(&d) => d,
            None => continue,
        };
        for (i, &x) in topo.up[u as usize].iter().enumerate() {
            let w = metric.w[u as usize][i];
            if !w.is_finite() {
                continue;
            }
            let nd = du + w;
            let e = dist.entry(x).or_insert(f64::INFINITY);
            if nd < *e {
                *e = nd;
            }
        }
    }
    dist
}

/// Exhaustive CCH k-NN: combine the query's up-search with each POI's up-search.
/// `d(s,p) = min over common ancestors m of d(s,m) + d(p,m)`. Exact.
#[must_use]
pub fn knn_exhaustive(
    topo: &Topology,
    metric: &crate::customize::Metric,
    src: NodeId,
    pois: &[NodeId],
    k: usize,
) -> Vec<(NodeId, f64)> {
    let ds = upward(topo, metric, topo.rank[src as usize]);
    let mut out: Vec<(NodeId, f64)> = Vec::new();
    for &p in pois {
        let dp = upward(topo, metric, topo.rank[p as usize]);
        // Iterate the smaller map for the intersection.
        let (small, big) = if ds.len() <= dp.len() { (&ds, &dp) } else { (&dp, &ds) };
        let mut best = f64::INFINITY;
        for (m, &dm) in small {
            if let Some(&dother) = big.get(m) {
                best = best.min(dm + dother);
            }
        }
        if best.is_finite() {
            out.push((p, best));
        }
    }
    out.sort_by(|a, b| cmp_dist_id(*a, *b));
    out.truncate(k);
    out
}

/// Pre-built bucket index over a fixed POI set for fast repeated queries.
pub struct KnnIndex<'a> {
    topo: &'a Topology,
    metric: &'a crate::customize::Metric,
    /// `bucket[rank]` = POIs `p` whose ancestor set includes `rank`, with the
    /// exact distance `d(p, rank)`. Sorted ascending by distance for early-out.
    bucket: Vec<Vec<(NodeId, f64)>>,
}

/// Diagnostics for one query — the M0 search-space-reduction evidence.
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryStats {
    /// Distinct ancestor vertices of the query that were examined.
    pub ancestors_visited: usize,
    /// Bucket entries (POI, dist) actually inspected.
    pub bucket_entries_scanned: usize,
    /// Ancestor vertices skipped by the admissible early-stop (region pruning).
    pub ancestors_pruned: usize,
}

impl<'a> KnnIndex<'a> {
    #[must_use]
    pub fn build(topo: &'a Topology, metric: &'a crate::customize::Metric, pois: &[NodeId]) -> Self {
        let mut bucket: Vec<Vec<(NodeId, f64)>> = vec![Vec::new(); topo.n];
        for &p in pois {
            for (anc, dp) in upward(topo, metric, topo.rank[p as usize]) {
                bucket[anc as usize].push((p, dp));
            }
        }
        for row in &mut bucket {
            row.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        KnnIndex { topo, metric, bucket }
    }

    /// k-NN with branch-and-bound. `prune = false` disables the early-stop (the
    /// "no-prune oracle mode" of M0): results must be identical, proving the
    /// pruning never drops a true top-k.
    pub fn knn(&self, src: NodeId, k: usize, prune: bool, stats: &mut QueryStats) -> Vec<(NodeId, f64)> {
        let ds = upward(self.topo, self.metric, self.topo.rank[src as usize]);
        // Ancestors ordered by ascending d(s -> x): the key to admissible pruning.
        let mut ancs: Vec<(u32, f64)> = ds.iter().map(|(&x, &d)| (x, d)).collect();
        ancs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut best: HashMap<NodeId, f64> = HashMap::new();
        for (x, dsx) in ancs {
            let delta_k = kth_smallest(best.values().copied(), k);
            if prune && dsx >= delta_k {
                // d(s,p) >= d(s,x) for the minimising x; nothing further can enter top-k.
                stats.ancestors_pruned += 1;
                continue; // (could break; continue keeps the count honest)
            }
            stats.ancestors_visited += 1;
            let row = &self.bucket[x as usize];
            for &(p, dp) in row {
                // Per-bucket early-out: rows are sorted by dp ascending.
                if prune && dsx + dp >= delta_k && best.len() >= k {
                    break;
                }
                stats.bucket_entries_scanned += 1;
                let cand = dsx + dp;
                let e = best.entry(p).or_insert(f64::INFINITY);
                if cand < *e {
                    *e = cand;
                }
            }
        }

        let mut out: Vec<(NodeId, f64)> = best.into_iter().filter(|(_, d)| d.is_finite()).collect();
        out.sort_by(|a, b| cmp_dist_id(*a, *b));
        out.truncate(k);
        out
    }
}

/// k-th smallest value of an iterator, or `+inf` if fewer than `k` present.
fn kth_smallest(vals: impl Iterator<Item = f64>, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let mut v: Vec<f64> = vals.collect();
    if v.len() < k {
        return f64::INFINITY;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[k - 1]
}

/// Convenience: elimination-tree depth (root-path length) of a rank — the query
/// search-space size bound. Useful for separator-quality diagnostics (ADR-199).
#[must_use]
pub fn elim_depth(topo: &Topology, mut r: u32) -> usize {
    let mut d = 0;
    while r != NONE {
        let p = topo.elim_parent[r as usize];
        if p == NONE {
            break;
        }
        r = p;
        d += 1;
    }
    d
}
