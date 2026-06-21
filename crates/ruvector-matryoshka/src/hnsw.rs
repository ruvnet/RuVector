//! Minimal HNSW graph parameterized by a working dimension.
//!
//! Designed to run at a truncated prefix of the full-dimension vectors,
//! enabling the coarse stages of a Matryoshka funnel search.
//!
//! The critical invariant in search_layer:
//! - `open`    = min-heap (closest pops first)  ← used for traversal
//! - `results` = max-heap (furthest pops first) ← used for pruning

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Clone, Debug)]
pub struct HnswConfig {
    /// Dimension used for all distance computations in this graph.
    pub dim: usize,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ml: f64,
}

impl HnswConfig {
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        let ml = 1.0 / (m as f64).ln();
        Self {
            dim,
            m,
            m0: m * 2,
            ef_construction,
            ml,
        }
    }
}

/// L2-squared distance using only the first `dim` elements.
#[inline(always)]
pub fn l2_sq_prefix(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let n = dim.min(a.len()).min(b.len());
    let mut s = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

// ─── Min-heap candidate (closest pops first, used for traversal) ─────────────

#[derive(Clone)]
struct MinC {
    dist: f32,
    id: u32,
}

impl PartialEq for MinC {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id
    }
}
impl Eq for MinC {}
impl PartialOrd for MinC {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinC {
    fn cmp(&self, o: &Self) -> Ordering {
        // Reverse dist order so BinaryHeap (max-heap) behaves as min-heap.
        o.dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then(self.id.cmp(&o.id))
    }
}

// ─── Max-heap candidate (furthest pops first, used for result pruning) ────────

#[derive(Clone)]
struct MaxC {
    dist: f32,
    id: u32,
}

impl PartialEq for MaxC {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id
    }
}
impl Eq for MaxC {}
impl PartialOrd for MaxC {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MaxC {
    fn cmp(&self, o: &Self) -> Ordering {
        // Natural dist order so BinaryHeap pops the furthest element first.
        self.dist
            .partial_cmp(&o.dist)
            .unwrap_or(Ordering::Equal)
            .then(o.id.cmp(&self.id))
    }
}

// ─── HnswGraph ────────────────────────────────────────────────────────────────

pub struct HnswGraph {
    pub config: HnswConfig,
    /// Prefix-projected vectors stored at `config.dim` dimensions.
    pub vecs: Vec<Vec<f32>>,
    pub node_level: Vec<usize>,
    /// `layers[level][node_id]` = neighbour node ids.
    pub layers: Vec<Vec<Vec<u32>>>,
    pub entry: Option<u32>,
    rng: u64,
}

impl HnswGraph {
    pub fn new(config: HnswConfig) -> Self {
        Self {
            config,
            vecs: Vec::new(),
            node_level: Vec::new(),
            layers: vec![Vec::new()],
            entry: None,
            rng: 0xABCD_EF01_2345_6789,
        }
    }

    pub fn insert(&mut self, v: Vec<f32>) -> u32 {
        let id = self.vecs.len() as u32;
        self.vecs.push(v);

        let level = self.random_level();
        self.node_level.push(level);

        while self.layers.len() <= level {
            self.layers.push(Vec::new());
        }
        for l in 0..=level {
            while self.layers[l].len() <= id as usize {
                self.layers[l].push(Vec::new());
            }
        }

        if self.entry.is_none() {
            self.entry = Some(id);
            return id;
        }

        let top = self.node_level[self.entry.unwrap() as usize];
        let mut ep = self.entry.unwrap();

        // Greedy descend through layers above insertion level.
        for lc in ((level + 1)..=top).rev() {
            ep = self.greedy_one_id(ep, id, lc);
        }

        // Insert at each layer 0..=min(level, top).
        for lc in (0..=level.min(top)).rev() {
            let ef_c = self.config.ef_construction;
            let cands = self.search_layer_id(ep, id, ef_c, lc);
            let m_max = if lc == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let selected: Vec<u32> = cands.iter().take(m_max).map(|c| c.1).collect();

            self.layers[lc][id as usize] = selected.clone();

            for &nb in &selected {
                let max_nb = if lc == 0 {
                    self.config.m0
                } else {
                    self.config.m
                };
                let nb_neighbours = self.layers[lc][nb as usize].clone();
                if nb_neighbours.len() < max_nb {
                    self.layers[lc][nb as usize].push(id);
                } else {
                    let dim = self.config.dim;
                    let nb_vec = self.vecs[nb as usize].clone();
                    let mut all: Vec<(f32, u32)> = nb_neighbours
                        .iter()
                        .map(|&x| (l2_sq_prefix(&nb_vec, &self.vecs[x as usize], dim), x))
                        .collect();
                    all.push((l2_sq_prefix(&nb_vec, &self.vecs[id as usize], dim), id));
                    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    all.truncate(max_nb);
                    self.layers[lc][nb as usize] = all.iter().map(|(_, id)| *id).collect();
                }
            }

            ep = cands.first().map(|c| c.1).unwrap_or(ep);
        }

        if level > top {
            self.entry = Some(id);
        }

        id
    }

    /// k-NN search; returns node ids sorted closest-first.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<u32> {
        let ep = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let top = self.node_level[ep as usize];
        let mut ep = ep;

        for lc in (1..=top).rev() {
            ep = self.greedy_one_query(ep, query, lc);
        }

        let cands = self.search_layer_query(ep, query, ef.max(k), 0);
        cands.into_iter().take(k).map(|(_, id)| id).collect()
    }

    // ────── internal helpers ──────────────────────────────────────────────────

    fn dist_id(&self, a: u32, b: u32) -> f32 {
        l2_sq_prefix(
            &self.vecs[a as usize],
            &self.vecs[b as usize],
            self.config.dim,
        )
    }
    fn dist_qv(&self, query: &[f32], b: u32) -> f32 {
        l2_sq_prefix(query, &self.vecs[b as usize], self.config.dim)
    }

    fn greedy_one_id(&self, ep: u32, q: u32, lc: usize) -> u32 {
        let mut best = ep;
        let mut best_d = self.dist_id(q, ep);
        loop {
            let mut improved = false;
            for &nb in &self.layers[lc][best as usize] {
                let d = self.dist_id(q, nb);
                if d < best_d {
                    best_d = d;
                    best = nb;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        best
    }

    fn greedy_one_query(&self, ep: u32, query: &[f32], lc: usize) -> u32 {
        let mut best = ep;
        let mut best_d = self.dist_qv(query, ep);
        loop {
            let mut improved = false;
            for &nb in &self.layers[lc][best as usize] {
                let d = self.dist_qv(query, nb);
                if d < best_d {
                    best_d = d;
                    best = nb;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        best
    }

    /// Standard HNSW beam search at a single layer.
    ///
    /// Returns up to `ef` candidates sorted closest-first.
    fn search_layer_impl(
        &self,
        ep: u32,
        ep_dist: f32,
        ef: usize,
        lc: usize,
        dist_fn: impl Fn(u32) -> f32,
    ) -> Vec<(f32, u32)> {
        let mut visited: HashSet<u32> = HashSet::new();
        visited.insert(ep);

        // min-heap: pop the closest candidate to expand next
        let mut open: BinaryHeap<MinC> = BinaryHeap::new();
        // max-heap: pop the furthest when over capacity, peek at worst result
        let mut results: BinaryHeap<MaxC> = BinaryHeap::new();

        open.push(MinC {
            dist: ep_dist,
            id: ep,
        });
        results.push(MaxC {
            dist: ep_dist,
            id: ep,
        });

        while let Some(curr) = open.pop() {
            // If the closest unvisited candidate is farther than our worst
            // result, all remaining candidates must also be farther → stop.
            let worst = results.peek().map(|c| c.dist).unwrap_or(f32::MAX);
            if curr.dist > worst {
                break;
            }

            let neighbours = self.layers[lc]
                .get(curr.id as usize)
                .cloned()
                .unwrap_or_default();
            for nb in neighbours {
                if !visited.insert(nb) {
                    continue;
                }
                let d = dist_fn(nb);
                let worst = results.peek().map(|c| c.dist).unwrap_or(f32::MAX);
                if d < worst || results.len() < ef {
                    open.push(MinC { dist: d, id: nb });
                    results.push(MaxC { dist: d, id: nb });
                    if results.len() > ef {
                        results.pop(); // evict the furthest — correct pruning
                    }
                }
            }
        }

        let mut out: Vec<(f32, u32)> = results.into_iter().map(|c| (c.dist, c.id)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out
    }

    fn search_layer_id(&self, ep: u32, q: u32, ef: usize, lc: usize) -> Vec<(f32, u32)> {
        let ep_d = self.dist_id(q, ep);
        self.search_layer_impl(ep, ep_d, ef, lc, |nb| self.dist_id(q, nb))
    }

    fn search_layer_query(&self, ep: u32, query: &[f32], ef: usize, lc: usize) -> Vec<(f32, u32)> {
        let ep_d = self.dist_qv(query, ep);
        self.search_layer_impl(ep, ep_d, ef, lc, |nb| self.dist_qv(query, nb))
    }

    fn random_level(&mut self) -> usize {
        self.rng = self
            .rng
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let r = (self.rng >> 33) as f64 / (u32::MAX as f64);
        (-r.max(1e-15).ln() * self.config.ml) as usize
    }
}
