use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// f32 wrapper that implements total `Ord` (NaN == NaN, NaN > everything).
#[derive(Clone, Copy, PartialEq)]
struct Dist(f32);

impl Eq for Dist {}
impl PartialOrd for Dist {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Dist {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(Ordering::Equal)
    }
}

/// Squared L2 distance between two equal-length slices.
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Minimal 2-layer HNSW index.
///
/// Layer assignment follows HNSW paper: P(level >= l) = exp(-l · ln(m)).
/// Layer 0 holds all N nodes with up to `2·m` connections each.
/// Layer 1 holds ~N/m nodes with up to `m` connections each.
pub struct HnswIndex {
    pub dim: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,

    pub vectors: Vec<Vec<f32>>,
    // connections[node_id][layer] = neighbor ids
    pub connections: Vec<Vec<Vec<u32>>>,

    pub entry_point: Option<u32>,
    pub max_layer: usize,
    rng: u64,
}

impl HnswIndex {
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        HnswIndex {
            dim,
            m,
            m0: m * 2,
            ef_construction,
            vectors: Vec::new(),
            connections: Vec::new(),
            entry_point: None,
            max_layer: 0,
            rng: 0x9e3779b97f4a7c15,
        }
    }

    /// XorShift64 pseudorandom; seeded from field `rng`.
    fn rand_u64(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    fn assign_level(&mut self) -> usize {
        let ml = 1.0 / (self.m as f64).ln();
        let r = (self.rand_u64() as f64) / (u64::MAX as f64);
        let lvl = (-r.ln() * ml).floor() as usize;
        lvl.min(4) // cap at 4 layers
    }

    /// Insert a vector and return its assigned id.
    pub fn insert(&mut self, vector: Vec<f32>) -> u32 {
        assert_eq!(vector.len(), self.dim);
        let id = self.vectors.len() as u32;
        let level = self.assign_level();

        self.vectors.push(vector);
        self.connections.push(vec![Vec::new(); level + 1]);

        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.max_layer = level;
            return id;
        }

        let mut ep = self.entry_point.unwrap();

        // Greedy descent from max_layer down to level+1 (no connections, just advance ep).
        let top = self.max_layer;
        for lc in ((level + 1)..=top).rev() {
            let cands = self.search_layer(ep, &self.vectors[id as usize].clone(), 1, lc);
            if let Some(&(best, _)) = cands.first() {
                ep = best;
            }
        }

        // From level down to 0: add bidirectional connections.
        for lc in (0..=level.min(self.max_layer)).rev() {
            let ef = self.ef_construction;
            let q_clone = self.vectors[id as usize].clone();
            let cands = self.search_layer(ep, &q_clone, ef, lc);

            let m_lc = if lc == 0 { self.m0 } else { self.m };
            let neighbours: Vec<u32> = cands.iter().take(m_lc).map(|&(nid, _)| nid).collect();

            self.connections[id as usize][lc] = neighbours.clone();

            // Bidirectional: add `id` to each neighbour's list and prune if needed.
            for nid in neighbours {
                let nlayer = &mut self.connections[nid as usize];
                if lc < nlayer.len() {
                    nlayer[lc].push(id);
                    if nlayer[lc].len() > m_lc * 2 {
                        // Prune: keep closest m_lc neighbours.
                        let mut scored: Vec<(u32, f32)> = nlayer[lc]
                            .iter()
                            .map(|&c| {
                                (
                                    c,
                                    sq_l2(&self.vectors[nid as usize], &self.vectors[c as usize]),
                                )
                            })
                            .collect();
                        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        nlayer[lc] = scored.into_iter().take(m_lc).map(|(c, _)| c).collect();
                    }
                }
            }

            // Advance entry point for next layer.
            if let Some(&(best, _)) = cands.first() {
                ep = best;
            }
        }

        if level > self.max_layer {
            self.max_layer = level;
            self.entry_point = Some(id);
        }

        id
    }

    /// Greedy beam search on one layer; returns up to `ef` candidates sorted by distance ASC.
    fn search_layer(&self, ep: u32, q: &[f32], ef: usize, layer: usize) -> Vec<(u32, f32)> {
        let mut visited = HashSet::with_capacity(ef * 4);
        let ep_dist = sq_l2(&self.vectors[ep as usize], q);

        // Candidates: min-heap (closest at top).
        let mut cands: BinaryHeap<std::cmp::Reverse<(Dist, u32)>> = BinaryHeap::new();
        // Results: max-heap (farthest at top for O(1) worst-dist lookup).
        let mut results: BinaryHeap<(Dist, u32)> = BinaryHeap::new();

        cands.push(std::cmp::Reverse((Dist(ep_dist), ep)));
        results.push((Dist(ep_dist), ep));
        visited.insert(ep);

        while let Some(std::cmp::Reverse((Dist(c_dist), c_id))) = cands.pop() {
            let worst_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
            if c_dist > worst_dist && results.len() >= ef {
                break;
            }

            let layer_conn = &self.connections[c_id as usize];
            if layer >= layer_conn.len() {
                continue;
            }
            for &n in &layer_conn[layer] {
                if visited.insert(n) {
                    let nd = sq_l2(&self.vectors[n as usize], q);
                    let worst = results.peek().map(|(d, _)| d.0).unwrap_or(f32::MAX);
                    if nd < worst || results.len() < ef {
                        cands.push(std::cmp::Reverse((Dist(nd), n)));
                        results.push((Dist(nd), n));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> = results.into_iter().map(|(d, id)| (id, d.0)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out
    }

    /// k-ANN query; returns up to `k` (id, sq_dist) pairs sorted closest first.
    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        assert_eq!(q.len(), self.dim);
        let ep = match self.entry_point {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut ep_cur = ep;
        for lc in (1..=self.max_layer).rev() {
            let res = self.search_layer(ep_cur, q, 1, lc);
            if let Some(&(best, _)) = res.first() {
                ep_cur = best;
            }
        }

        let ef_actual = ef.max(k);
        let mut res = self.search_layer(ep_cur, q, ef_actual, 0);
        res.truncate(k);
        res
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Approximate byte footprint (vectors + graph).
    pub fn memory_bytes(&self) -> usize {
        let vecs = self.vectors.iter().map(|v| v.len() * 4).sum::<usize>();
        let graph = self
            .connections
            .iter()
            .flat_map(|lvls| lvls.iter())
            .map(|nbrs| nbrs.len() * 4)
            .sum::<usize>();
        vecs + graph
    }
}

/// Exact brute-force search (ground truth).
pub fn brute_force_knn(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut scored: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, sq_l2(v, query)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq_l2_identical() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(sq_l2(&v, &v), 0.0);
    }

    #[test]
    fn sq_l2_unit() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!((sq_l2(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn hnsw_insert_and_search() {
        let dim = 8;
        let mut idx = HnswIndex::new(dim, 8, 40);
        for i in 0..200u32 {
            let v: Vec<f32> = (0..dim).map(|d| i as f32 * 0.01 + d as f32).collect();
            idx.insert(v);
        }
        let q: Vec<f32> = (0..dim).map(|d| d as f32).collect();
        let res = idx.search(&q, 5, 50);
        assert_eq!(res.len(), 5);
        // Closest should be id=0 (all zeros offset)
        assert_eq!(res[0].0, 0);
    }

    #[test]
    fn brute_force_returns_k() {
        let vecs: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 4]).collect();
        let q = vec![0.0_f32; 4];
        let res = brute_force_knn(&vecs, &q, 10);
        assert_eq!(res.len(), 10);
        assert_eq!(res[0].0, 0);
    }
}
