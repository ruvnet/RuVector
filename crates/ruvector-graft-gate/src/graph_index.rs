//! A minimal NSW-style single-layer proximity graph: greedy best-first
//! search plus reciprocal-edge linking on insert, pruned to `m` nearest
//! neighbors per node by cosine similarity. This is deliberately the
//! *base layer* of an HNSW-family index, not the full multi-level
//! structure — the variable under test in this crate is insertion-time
//! gating, not multi-level graph search quality, so the minimal single
//! layer keeps the measured surface small (see the nightly README's
//! "Why Single-Layer NSW, Not Full HNSW" section, mirroring the scoping
//! precedent set by `ruvector-retrieval-receipt`'s "Why Brute Force").

use crate::vector::{cosine, Vector};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

#[derive(Clone, Copy, PartialEq)]
struct ScoredId {
    sim: f32,
    id: u32,
}

impl Eq for ScoredId {}
impl PartialOrd for ScoredId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sim.total_cmp(&other.sim)
    }
}

/// A single fixed entry point can leave a plain (non-hierarchical) NSW
/// graph unreachable from most of the vector space: a greedy walk from
/// one point only climbs similarity locally, so if a query lands in a
/// topic region unrelated to the entry point's own region, nothing in the
/// search ever crosses over (measured directly — see ADR-340's
/// "Implementation Notes on Graph Connectivity"). Every one of the first
/// `EARLY_ENTRY_COUNT` inserted nodes is kept as an entry point (which,
/// given this crate's round-robin-by-cluster data generation, covers
/// every cluster before real insertion volume begins), plus one more
/// every `ENTRY_POINT_INTERVAL` insertions after that for resilience as
/// the graph grows and early nodes' own links get pruned by later,
/// closer neighbors (`link_reciprocal`).
const EARLY_ENTRY_COUNT: u32 = 64;
const ENTRY_POINT_INTERVAL: u32 = 137;

#[derive(Clone)]
pub struct GraphIndex {
    pub dim: usize,
    pub m: usize,
    pub vectors: Vec<Vector>,
    pub neighbors: Vec<Vec<u32>>,
    pub entry_points: Vec<u32>,
}

impl GraphIndex {
    pub fn new(dim: usize, m: usize) -> Self {
        Self {
            dim,
            m,
            vectors: Vec::new(),
            neighbors: Vec::new(),
            entry_points: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Convenience accessor mirroring the crate's earlier single-entry-point
    /// API; kept for callers (and tests) that only care whether the index
    /// has been bootstrapped at all.
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_points.first().copied()
    }

    /// Greedy best-first search over the proximity graph, seeded from all
    /// registered entry points. Returns up to `ef` `(id, cosine_similarity)`
    /// pairs sorted by descending similarity. This is the same search
    /// every insertion pays for (baseline included) to locate link
    /// candidates — gate variants reuse this same call's output rather
    /// than searching twice, so gating overhead measured in the benchmark
    /// is purely the marginal decision cost, not search cost.
    pub fn search(&self, query: &[f32], ef: usize) -> Vec<(u32, f32)> {
        if self.entry_points.is_empty() {
            return Vec::new();
        }
        let mut visited: HashSet<u32> = HashSet::new();
        let mut candidates: BinaryHeap<ScoredId> = BinaryHeap::new();
        let mut results: BinaryHeap<Reverse<ScoredId>> = BinaryHeap::new();

        for &entry in &self.entry_points {
            if visited.insert(entry) {
                let sim0 = cosine(query, &self.vectors[entry as usize]);
                candidates.push(ScoredId {
                    sim: sim0,
                    id: entry,
                });
                results.push(Reverse(ScoredId {
                    sim: sim0,
                    id: entry,
                }));
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        while let Some(ScoredId { sim: csim, id: cid }) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if csim < worst.0.sim {
                        break;
                    }
                }
            }
            for &nb in &self.neighbors[cid as usize] {
                if visited.insert(nb) {
                    let nsim = cosine(query, &self.vectors[nb as usize]);
                    let should_add = results.len() < ef
                        || results.peek().map(|w| nsim > w.0.sim).unwrap_or(true);
                    if should_add {
                        candidates.push(ScoredId { sim: nsim, id: nb });
                        results.push(Reverse(ScoredId { sim: nsim, id: nb }));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> = results
            .into_iter()
            .map(|Reverse(s)| (s.id, s.sim))
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }

    /// Insert `v`, linking to up to `self.m` of `candidate_neighbors`
    /// (assumed sorted descending by similarity — typically `search`'s
    /// own output), and add reciprocal edges pruned back to `self.m` per
    /// node. Returns the new node's id.
    pub fn insert_with_neighbors(&mut self, v: Vector, candidate_neighbors: &[(u32, f32)]) -> u32 {
        let id = self.vectors.len() as u32;
        self.vectors.push(v);
        self.neighbors.push(Vec::new());
        if self.entry_points.is_empty() {
            self.entry_points.push(id);
            return id;
        }
        if id < EARLY_ENTRY_COUNT || id % ENTRY_POINT_INTERVAL == 0 {
            self.entry_points.push(id);
        }
        let take = candidate_neighbors.len().min(self.m);
        let chosen: Vec<u32> = candidate_neighbors[..take]
            .iter()
            .map(|&(nid, _)| nid)
            .collect();
        self.neighbors[id as usize] = chosen.clone();
        for nb in chosen {
            self.link_reciprocal(nb, id);
        }
        id
    }

    fn link_reciprocal(&mut self, node: u32, new_id: u32) {
        let node_vec = self.vectors[node as usize].clone();
        {
            let list = &mut self.neighbors[node as usize];
            if !list.contains(&new_id) {
                list.push(new_id);
            }
        }
        if self.neighbors[node as usize].len() > self.m {
            let mut scored: Vec<(u32, f32)> = self.neighbors[node as usize]
                .iter()
                .map(|&nid| (nid, cosine(&node_vec, &self.vectors[nid as usize])))
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            scored.truncate(self.m);
            self.neighbors[node as usize] = scored.into_iter().map(|(nid, _)| nid).collect();
        }
    }
}

/// Exact brute-force top-k by cosine similarity over an arbitrary vector
/// slice, indexed by position. Used only to compute recall ground truth —
/// never part of the path being measured for gate overhead.
pub fn brute_force_top_k(vectors: &[Vector], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, cosine(query, v)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{gen_ball_point, gen_centroids};
    use crate::rng::Xorshift64;

    fn build_small_index(n: usize, dim: usize, m: usize) -> GraphIndex {
        let mut rng = Xorshift64::new(123);
        let centroids = gen_centroids(&mut rng, 4, dim);
        let mut idx = GraphIndex::new(dim, m);
        for i in 0..n {
            let c = &centroids[i % centroids.len()];
            let v = gen_ball_point(&mut rng, c, 0.1);
            let sr = idx.search(&v, 32);
            idx.insert_with_neighbors(v, &sr);
        }
        idx
    }

    #[test]
    fn search_finds_self_as_best_match() {
        let idx = build_small_index(200, 16, 8);
        for id in [0u32, 50, 150] {
            let q = idx.vectors[id as usize].clone();
            let r = idx.search(&q, 10);
            assert_eq!(r[0].0, id, "expected exact self-match to rank first");
            assert!((r[0].1 - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn neighbor_lists_never_exceed_m() {
        let idx = build_small_index(300, 16, 6);
        for list in &idx.neighbors {
            assert!(list.len() <= 6, "neighbor list exceeded m: {}", list.len());
        }
    }

    #[test]
    fn brute_force_matches_search_on_tiny_well_connected_graph() {
        let idx = build_small_index(150, 16, 32);
        let q = idx.vectors[10].clone();
        let bf = brute_force_top_k(&idx.vectors, &q, 5);
        let gs: Vec<usize> = idx
            .search(&q, 64)
            .into_iter()
            .take(5)
            .map(|(id, _)| id as usize)
            .collect();
        let overlap = bf.iter().filter(|id| gs.contains(id)).count();
        assert!(
            overlap >= 4,
            "graph search should closely match brute force at high ef, overlap={overlap}"
        );
    }

    #[test]
    fn empty_index_search_returns_empty() {
        let idx = GraphIndex::new(8, 8);
        assert!(idx.search(&[0.0; 8], 10).is_empty());
    }
}
