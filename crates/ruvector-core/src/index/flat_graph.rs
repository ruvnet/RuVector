//! Flat-array HNSW graph (the hnswlib memory layout).
//!
//! Vectors live in one contiguous `Vec<f32>`, layer-0 neighbours in one contiguous `Vec<u32>` (`[count, n0, n1, ..]` per
//! node), upper layers in a small per-node `Vec<u32>`. A traversal step is therefore one array read per neighbour
//! instead of the `Arc<PointWithOrder>` -> `Arc<Point>` -> `Vec<f32>` pointer chase of the `hnsw_rs` representation, the
//! visited set is an epoch-tagged array (no hashing, no clearing), and the next neighbours' vectors are prefetched while
//! the current one is scored.
//!
//! Algorithm: Malkov & Yashunin, with the neighbour-selection heuristic (their Algorithm 4, no pruned-connection
//! back-fill) and bidirectional links re-pruned on overflow, as in hnswlib. Single writer (`&mut self` inserts), any
//! number of concurrent readers (`&self` search); the caller provides the `RwLock`.
//!
//! Deletion is not supported here: callers tombstone ids at their own layer, exactly as they did with `hnsw_rs`.

use crate::simd_intrinsics;
use crate::types::DistanceMetric;
use parking_lot::Mutex;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

mod filtered;
#[allow(unused_imports)] // the HNSW wrapper reads the fields directly
pub(crate) use filtered::FilteredHits;

/// A node with its distance to the query being served. `Ord` is by distance (ties by id), so a `BinaryHeap` pops the
/// FARTHEST first and `Reverse<Cand>` pops the nearest first.
#[derive(Clone, Copy, Debug)]
struct Cand {
    dist: f32,
    id: u32,
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
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
        self.dist.total_cmp(&o.dist).then(self.id.cmp(&o.id))
    }
}

/// Epoch-tagged visited list: `visit` is one load and one store, and a new query costs O(1), not O(n).
struct Visited {
    tags: Vec<u16>,
    epoch: u16,
}
impl Visited {
    fn new() -> Self {
        Self {
            tags: Vec::new(),
            epoch: 0,
        }
    }
    fn reset(&mut self, n: usize) {
        if self.tags.len() < n {
            self.tags.resize(n, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.tags.fill(0);
            self.epoch = 1;
        }
    }
    /// True the first time `id` is seen in this epoch.
    #[inline(always)]
    fn visit(&mut self, id: u32) -> bool {
        let t = &mut self.tags[id as usize];
        if *t == self.epoch {
            false
        } else {
            *t = self.epoch;
            true
        }
    }
}

#[inline(always)]
fn prefetch(p: *const f32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

pub struct FlatGraph {
    dim: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
    metric: DistanceMetric,
    level_mult: f64,
    rng: StdRng,
    /// `n * dim` floats; unit-normalised copies for the cosine metric.
    data: Vec<f32>,
    /// External id of each node (what the caller passed to `insert`).
    ext: Vec<usize>,
    levels: Vec<u8>,
    /// `n * (m0 + 1)` slots: `[count, neighbours..]` per node.
    l0: Vec<u32>,
    /// Per node, `levels[node] * (m + 1)` slots for layers 1..=level; empty for level-0 nodes.
    upper: Vec<Vec<u32>>,
    entry: Option<u32>,
    max_level: usize,
    visited: Mutex<Vec<Visited>>,
}

impl FlatGraph {
    pub fn new(dim: usize, metric: DistanceMetric, m: usize, ef_construction: usize) -> Self {
        let m = m.max(2);
        Self {
            dim,
            m,
            m0: m * 2,
            ef_construction: ef_construction.max(m),
            metric,
            level_mult: 1.0 / (m as f64).ln(),
            rng: StdRng::seed_from_u64(100), // hnswlib's default seed: reproducible builds
            data: Vec::new(),
            ext: Vec::new(),
            levels: Vec::new(),
            l0: Vec::new(),
            upper: Vec::new(),
            entry: None,
            max_level: 0,
            visited: Mutex::new(Vec::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    #[inline(always)]
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::Euclidean => simd_intrinsics::euclidean_distance_simd(a, b),
            DistanceMetric::Cosine => (1.0 - simd_intrinsics::dot_product_simd(a, b)).max(0.0),
            DistanceMetric::DotProduct => -simd_intrinsics::dot_product_simd(a, b),
            DistanceMetric::Manhattan => simd_intrinsics::manhattan_distance_simd(a, b),
        }
    }

    /// The vector as stored (unit length for cosine).
    fn prepare(&self, v: &[f32]) -> Vec<f32> {
        if self.metric == DistanceMetric::Cosine {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                return v.iter().map(|x| x / norm).collect();
            }
        }
        v.to_vec()
    }

    #[inline(always)]
    fn vec(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.data[s..s + self.dim]
    }

    #[inline(always)]
    fn list(&self, id: u32, layer: usize) -> &[u32] {
        if layer == 0 {
            let b = id as usize * (self.m0 + 1);
            let n = self.l0[b] as usize;
            &self.l0[b + 1..b + 1 + n]
        } else {
            let u = &self.upper[id as usize];
            let b = (layer - 1) * (self.m + 1);
            let n = u[b] as usize;
            &u[b + 1..b + 1 + n]
        }
    }

    fn set_list(&mut self, id: u32, layer: usize, items: &[u32]) {
        if layer == 0 {
            let b = id as usize * (self.m0 + 1);
            self.l0[b] = items.len() as u32;
            self.l0[b + 1..b + 1 + items.len()].copy_from_slice(items);
        } else {
            let b = (layer - 1) * (self.m + 1);
            let u = &mut self.upper[id as usize];
            u[b] = items.len() as u32;
            u[b + 1..b + 1 + items.len()].copy_from_slice(items);
        }
    }

    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.gen::<f64>();
        let u = (1.0 - r).max(f64::MIN_POSITIVE);
        ((-u.ln() * self.level_mult).floor() as usize).min(31)
    }

    fn take_visited(&self) -> Visited {
        let mut v = self.visited.lock().pop().unwrap_or_else(Visited::new);
        v.reset(self.len());
        v
    }

    fn give_visited(&self, v: Visited) {
        self.visited.lock().push(v);
    }

    /// Initial capacities of the result and frontier heaps for a beam of width `ef`, capped at the node count so that a
    /// huge `ef`/`k` cannot request a huge allocation (or overflow `ef + 1`).
    fn heap_caps(&self, ef: usize) -> (usize, usize) {
        let n = self.len().saturating_add(1);
        (ef.saturating_add(1).min(n), ef.saturating_mul(2).min(n))
    }

    /// Greedy descent on one upper layer (ef = 1).
    fn greedy(&self, q: &[f32], mut cur: u32, mut cur_d: f32, layer: usize) -> (u32, f32) {
        loop {
            let mut changed = false;
            let node = cur;
            for &nb in self.list(node, layer) {
                let d = self.dist(q, self.vec(nb));
                if d < cur_d {
                    cur_d = d;
                    cur = nb;
                    changed = true;
                }
            }
            if !changed {
                return (cur, cur_d);
            }
        }
    }

    /// Best-first search of one layer from `ep`; returns up to `ef` nearest nodes as a max-heap (farthest on top).
    fn search_layer(
        &self,
        q: &[f32],
        ep: u32,
        ep_d: f32,
        layer: usize,
        ef: usize,
    ) -> BinaryHeap<Cand> {
        let mut vis = self.take_visited();
        // `ef` can come straight from a caller's `k`; neither heap can ever hold more than the graph has nodes.
        let (top_cap, cand_cap) = self.heap_caps(ef);
        let mut top: BinaryHeap<Cand> = BinaryHeap::with_capacity(top_cap);
        let mut cand: BinaryHeap<Reverse<Cand>> = BinaryHeap::with_capacity(cand_cap);
        vis.visit(ep);
        top.push(Cand { dist: ep_d, id: ep });
        cand.push(Reverse(Cand { dist: ep_d, id: ep }));
        let mut lower = ep_d;
        while let Some(Reverse(c)) = cand.pop() {
            if c.dist > lower && top.len() >= ef {
                break;
            }
            let nbrs = self.list(c.id, layer);
            // Pull the neighbours' vectors toward the cache while the first ones are scored.
            for &nb in nbrs {
                prefetch(self.data.as_ptr().wrapping_add(nb as usize * self.dim));
            }
            for &nb in nbrs {
                if !vis.visit(nb) {
                    continue;
                }
                let d = self.dist(q, self.vec(nb));
                if top.len() < ef || d < lower {
                    cand.push(Reverse(Cand { dist: d, id: nb }));
                    top.push(Cand { dist: d, id: nb });
                    if top.len() > ef {
                        top.pop();
                    }
                    if let Some(t) = top.peek() {
                        lower = t.dist;
                    }
                }
            }
        }
        self.give_visited(vis);
        top
    }

    /// Neighbour selection heuristic: closest-first, keep a candidate only if no already-kept neighbour is closer to it
    /// than the query is. With fewer than `target` candidates all are kept.
    fn select(&self, cands: BinaryHeap<Cand>, target: usize) -> Vec<Cand> {
        let mut sorted = cands.into_vec();
        sorted.sort_unstable();
        if sorted.len() < target {
            return sorted;
        }
        let mut kept: Vec<Cand> = Vec::with_capacity(target);
        for c in sorted {
            if kept.len() >= target {
                break;
            }
            let cv = self.vec(c.id);
            if kept.iter().all(|r| self.dist(self.vec(r.id), cv) >= c.dist) {
                kept.push(c);
            }
        }
        kept
    }

    /// Link `new_id` to its selected neighbours on `layer` and re-prune any neighbour that overflows. Returns the node
    /// closest to the new point, the entry for the next layer down.
    fn connect(&mut self, new_id: u32, top: BinaryHeap<Cand>, layer: usize) -> u32 {
        let nearest = top.iter().min().map(|c| c.id).unwrap_or(new_id);
        let selected = self.select(top, self.m);
        let max_here = if layer == 0 { self.m0 } else { self.m };
        let ids: Vec<u32> = selected.iter().map(|c| c.id).collect();
        self.set_list(new_id, layer, &ids);
        for s in &selected {
            let mut cur: Vec<u32> = self.list(s.id, layer).to_vec();
            if cur.contains(&new_id) {
                continue;
            }
            if cur.len() < max_here {
                cur.push(new_id);
                self.set_list(s.id, layer, &cur);
            } else {
                let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(cur.len() + 1);
                heap.push(Cand {
                    dist: s.dist,
                    id: new_id,
                });
                let sv = self.vec(s.id);
                for &o in &cur {
                    heap.push(Cand {
                        dist: self.dist(sv, self.vec(o)),
                        id: o,
                    });
                }
                let kept: Vec<u32> = self.select(heap, max_here).iter().map(|c| c.id).collect();
                self.set_list(s.id, layer, &kept);
            }
        }
        nearest
    }

    /// Reserve the next node id for a prepared vector of the given level (no links yet, so searches cannot reach it).
    fn alloc(&mut self, q: &[f32], ext_id: usize, level: usize) -> u32 {
        let id = self.levels.len() as u32;
        self.data.extend_from_slice(q);
        self.ext.push(ext_id);
        self.levels.push(level as u8);
        self.l0.extend(std::iter::repeat(0u32).take(self.m0 + 1));
        self.upper.push(if level > 0 {
            vec![0u32; level * (self.m + 1)]
        } else {
            Vec::new()
        });
        id
    }

    /// Candidate lists (`ef_construction` nearest, farthest on top) for each layer `0..=min(level, max_level)` of the
    /// CURRENT graph, indexed by layer. Read-only, so many points can be planned at once.
    fn plan(&self, q: &[f32], level: usize) -> Vec<BinaryHeap<Cand>> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let mut cur = entry;
        let mut cur_d = self.dist(q, self.vec(cur));
        for l in ((level + 1)..=self.max_level).rev() {
            (cur, cur_d) = self.greedy(q, cur, cur_d, l);
        }
        let top_layer = level.min(self.max_level);
        let mut out: Vec<BinaryHeap<Cand>> = Vec::with_capacity(top_layer + 1);
        for l in (0..=top_layer).rev() {
            let top = self.search_layer(q, cur, cur_d, l, self.ef_construction);
            if let Some(n) = top.iter().min() {
                cur = n.id;
                cur_d = n.dist;
            }
            out.push(top);
        }
        out.reverse();
        out
    }

    /// Connect an allocated node using its per-layer candidate lists, then promote it to entry point if it is the highest.
    fn link(&mut self, id: u32, level: usize, plan: Vec<BinaryHeap<Cand>>) {
        for (l, top) in plan.into_iter().enumerate().rev() {
            self.connect(id, top, l);
        }
        if level > self.max_level {
            self.entry = Some(id);
            self.max_level = level;
        }
    }

    /// Insert one vector; returns the node id (its position in insertion order).
    pub fn insert(&mut self, v: &[f32], ext_id: usize) -> u32 {
        debug_assert_eq!(v.len(), self.dim);
        let q = self.prepare(v);
        let level = self.random_level();
        let id = self.alloc(&q, ext_id, level);
        if self.entry.is_none() {
            self.entry = Some(id);
            self.max_level = level;
            return id;
        }
        let plan = self.plan(&q, level);
        self.link(id, level, plan);
        id
    }

    /// Insert many vectors. The graph is built in chunks: the candidate searches of a chunk run in parallel on the graph
    /// as it stood when the chunk began (plus the chunk's own earlier points, added as candidates), then the links are
    /// applied one point at a time in input order. The result depends only on the input and the seed, never on the number
    /// of threads. Small graphs are built point by point, where batching would hurt connectivity.
    pub fn insert_batch(&mut self, items: &[(&[f32], usize)]) {
        const SEQUENTIAL_PREFIX: usize = 2048;
        const CHUNK: usize = 256;
        let mut next = 0;
        while next < items.len() && (self.len() < SEQUENTIAL_PREFIX || items.len() - next < CHUNK) {
            self.insert(items[next].0, items[next].1);
            next += 1;
        }
        for chunk in items[next..].chunks(CHUNK) {
            let prepared: Vec<Vec<f32>> = chunk.iter().map(|(v, _)| self.prepare(v)).collect();
            let levels: Vec<usize> = chunk.iter().map(|_| self.random_level()).collect();
            #[cfg(feature = "parallel")]
            let plans: Vec<Vec<BinaryHeap<Cand>>> = {
                use rayon::prelude::*;
                (0..chunk.len())
                    .into_par_iter()
                    .map(|j| self.plan(&prepared[j], levels[j]))
                    .collect()
            };
            #[cfg(not(feature = "parallel"))]
            let plans: Vec<Vec<BinaryHeap<Cand>>> = (0..chunk.len())
                .map(|j| self.plan(&prepared[j], levels[j]))
                .collect();
            let base = self.len() as u32;
            for j in 0..chunk.len() {
                self.alloc(&prepared[j], chunk[j].1, levels[j]);
            }
            for (j, mut plan) in plans.into_iter().enumerate() {
                for (l, top) in plan.iter_mut().enumerate() {
                    for k in 0..j {
                        if levels[k] >= l {
                            top.push(Cand {
                                dist: self.dist(&prepared[j], &prepared[k]),
                                id: base + k as u32,
                            });
                        }
                    }
                    while top.len() > self.ef_construction {
                        top.pop();
                    }
                }
                self.link(base + j as u32, levels[j], plan);
            }
        }
    }

    /// The `k` nearest nodes with `ef` candidates considered: `(external id, distance)`, ascending by distance.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let q = self.prepare(query);
        let mut cur = entry;
        let mut cur_d = self.dist(&q, self.vec(cur));
        for l in (1..=self.max_level).rev() {
            (cur, cur_d) = self.greedy(&q, cur, cur_d, l);
        }
        // A beam wider than the graph explores nothing more, so clamp (k may be usize::MAX from a binding).
        let mut top = self.search_layer(&q, cur, cur_d, 0, ef.max(k).min(self.len()));
        while top.len() > k {
            top.pop();
        }
        let mut out: Vec<Cand> = top.into_vec();
        out.sort_unstable();
        out.into_iter()
            .map(|c| (self.ext[c.id as usize], c.dist))
            .collect()
    }

    /// Edges that point at a node whose top level is below the layer of the edge, as `"u->v@layer"`.
    #[doc(hidden)]
    pub fn edges_above_endpoint_level(&self) -> Vec<String> {
        let mut out = Vec::new();
        for u in 0..self.len() as u32 {
            for l in 1..=self.levels[u as usize] as usize {
                for &v in self.list(u, l) {
                    if (self.levels[v as usize] as usize) < l {
                        out.push(format!(
                            "{}->{}@{}",
                            self.ext[u as usize], self.ext[v as usize], l
                        ));
                    }
                }
            }
        }
        out
    }

    /// External ids of nodes that cannot be reached from the entry point over layer-0 edges.
    #[doc(hidden)]
    pub fn points_unreachable_at_layer0(&self) -> Vec<usize> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let mut seen = vec![false; self.len()];
        let mut stack = vec![entry];
        seen[entry as usize] = true;
        while let Some(u) = stack.pop() {
            for &v in self.list(u, 0) {
                if !seen[v as usize] {
                    seen[v as usize] = true;
                    stack.push(v);
                }
            }
        }
        (0..self.len())
            .filter(|&i| !seen[i])
            .map(|i| self.ext[i])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / (u32::MAX >> 1) as f32
    }

    fn data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        (0..n)
            .map(|_| (0..dim).map(|_| lcg(&mut s)).collect())
            .collect()
    }

    fn brute(
        base: &[Vec<f32>],
        q: &[f32],
        k: usize,
        f: impl Fn(&[f32], &[f32]) -> f32,
    ) -> Vec<usize> {
        let mut d: Vec<(f32, usize)> = base.iter().enumerate().map(|(i, v)| (f(q, v), i)).collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0));
        d.into_iter().take(k).map(|x| x.1).collect()
    }

    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn recall_against_brute_force_and_graph_invariants() {
        let (n, dim, k) = (4000, 24, 10);
        let base = data(n, dim, 1);
        let queries = data(100, dim, 2);
        let mut g = FlatGraph::new(dim, DistanceMetric::Euclidean, 16, 200);
        for (i, v) in base.iter().enumerate() {
            assert_eq!(g.insert(v, i) as usize, i);
        }
        assert_eq!(g.len(), n);
        assert!(
            g.edges_above_endpoint_level().is_empty(),
            "no edge may point at a node below its layer"
        );
        assert!(
            g.points_unreachable_at_layer0().is_empty(),
            "every node is reachable at layer 0"
        );
        let mut hits = 0;
        for q in &queries {
            let truth = brute(&base, q, k, l2);
            let got: Vec<usize> = g.search(q, k, 80).into_iter().map(|x| x.0).collect();
            hits += got.iter().filter(|x| truth.contains(x)).count();
        }
        let recall = hits as f64 / (queries.len() * k) as f64;
        assert!(recall > 0.97, "recall {recall}");
    }

    #[test]
    fn batch_build_keeps_recall_connectivity_and_is_deterministic() {
        let (n, dim, k) = (6000, 24, 10);
        let base = data(n, dim, 11);
        let queries = data(100, dim, 12);
        let items: Vec<(&[f32], usize)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| (v.as_slice(), i))
            .collect();
        let mut a = FlatGraph::new(dim, DistanceMetric::Euclidean, 16, 200);
        a.insert_batch(&items);
        let mut b = FlatGraph::new(dim, DistanceMetric::Euclidean, 16, 200);
        b.insert_batch(&items);
        assert_eq!(a.len(), n);
        assert!(a.edges_above_endpoint_level().is_empty());
        assert!(a.points_unreachable_at_layer0().is_empty());
        let mut hits = 0;
        for q in &queries {
            let truth = brute(&base, q, k, l2);
            let ra = a.search(q, k, 80);
            assert_eq!(
                ra,
                b.search(q, k, 80),
                "same input and seed give the same graph"
            );
            hits += ra.iter().filter(|x| truth.contains(&x.0)).count();
        }
        let recall = hits as f64 / (queries.len() * k) as f64;
        assert!(recall > 0.97, "batch recall {recall}");
    }

    #[test]
    fn results_are_sorted_distances_are_true_and_ext_ids_are_returned() {
        let base = data(300, 8, 3);
        let mut g = FlatGraph::new(8, DistanceMetric::Euclidean, 8, 64);
        for (i, v) in base.iter().enumerate() {
            g.insert(v, 1000 + i);
        }
        let r = g.search(&base[17], 5, 50);
        assert_eq!(r[0].0, 1017);
        assert!(r[0].1 < 1e-5);
        assert!(r.windows(2).all(|w| w[0].1 <= w[1].1));
        assert!((r[1].1 - l2(&base[17], &base[r[1].0 - 1000])).abs() < 1e-4);
    }

    #[test]
    fn cosine_distance_ranks_by_angle_and_is_scale_free() {
        let mut g = FlatGraph::new(2, DistanceMetric::Cosine, 8, 64);
        g.insert(&[1.0, 0.0], 0);
        g.insert(&[10.0, 0.5], 1);
        g.insert(&[0.0, 1.0], 2);
        g.insert(&[-1.0, 0.0], 3);
        let r = g.search(&[100.0, 0.0], 4, 10);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert!(r[0].1 < 1e-6 && (r[3].1 - 2.0).abs() < 1e-5);
    }

    #[test]
    fn dot_product_ranks_by_largest_inner_product() {
        let mut g = FlatGraph::new(2, DistanceMetric::DotProduct, 8, 64);
        g.insert(&[1.0, 0.0], 0);
        g.insert(&[3.0, 0.0], 1);
        g.insert(&[2.0, 0.0], 2);
        let r = g.search(&[1.0, 0.0], 3, 10);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 2, 0]);
    }

    #[test]
    fn empty_single_and_small_graphs_do_not_panic() {
        let mut g = FlatGraph::new(4, DistanceMetric::Euclidean, 16, 200);
        assert!(g.search(&[0.0; 4], 3, 10).is_empty());
        g.insert(&[1.0; 4], 7);
        assert_eq!(g.search(&[0.0; 4], 3, 10).len(), 1);
        g.insert(&[2.0; 4], 8);
        let r = g.search(&[0.0; 4], 5, 10);
        assert_eq!(r.iter().map(|x| x.0).collect::<Vec<_>>(), vec![7, 8]);
    }

    #[test]
    fn many_searches_reuse_the_visited_list_across_epoch_wraparound() {
        let base = data(200, 8, 4);
        let mut g = FlatGraph::new(8, DistanceMetric::Euclidean, 8, 32);
        for (i, v) in base.iter().enumerate() {
            g.insert(v, i);
        }
        for _ in 0..70_000 {
            assert_eq!(g.search(&base[5], 1, 16)[0].0, 5);
        }
    }
}
