/// Minimal self-contained HNSW implementation for namespace partition benchmarks.
///
/// Supports multi-level graph construction (M, M0, ef_construction) and greedy
/// descent search with configurable ef.  No external dependencies — pure Rust.
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

// ── Distance ─────────────────────────────────────────────────────────────────

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ── Priority queue helpers ────────────────────────────────────────────────────

/// Min-heap entry (smaller dist = higher priority).
#[derive(Clone, PartialEq)]
struct MinEntry {
    dist: f32,
    idx: usize,
}
impl Eq for MinEntry {}
impl PartialOrd for MinEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinEntry {
    fn cmp(&self, o: &Self) -> Ordering {
        // Reverse so BinaryHeap becomes a min-heap.
        o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap entry (larger dist = higher priority).
#[derive(Clone, PartialEq)]
struct MaxEntry {
    dist: f32,
    idx: usize,
}
impl Eq for MaxEntry {}
impl PartialOrd for MaxEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MaxEntry {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.partial_cmp(&o.dist).unwrap_or(Ordering::Equal)
    }
}

// ── HNSW node ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct HnswNode {
    pub id: u64,
    pub vector: Vec<f32>,
    /// neighbors[lc] = neighbor node indices at layer lc
    pub layers: Vec<Vec<usize>>,
}

// ── HNSW graph ───────────────────────────────────────────────────────────────

pub struct MiniHnsw {
    pub nodes: Vec<HnswNode>,
    entry: Option<usize>,
    m: usize,
    m0: usize,
    ef_construction: usize,
    /// 1 / ln(M) — controls level distribution
    ml: f64,
    /// LCG state for deterministic level generation
    lcg: u64,
}

impl MiniHnsw {
    pub fn new(m: usize, ef_construction: usize, seed: u64) -> Self {
        let ml = 1.0 / (m as f64).ln();
        Self {
            nodes: Vec::new(),
            entry: None,
            m,
            m0: m * 2,
            ef_construction,
            ml,
            lcg: seed,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Deterministic level via LCG — no dependency on `rand` crate.
    fn next_level(&mut self) -> usize {
        self.lcg = self
            .lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (self.lcg >> 33) as f64 / (u32::MAX as f64 + 1.0);
        let lvl = (-u.ln() * self.ml).floor() as usize;
        lvl.min(15)
    }

    /// Greedy search within a single layer returning up to `ef` nearest nodes.
    fn search_layer(&self, query: &[f32], ep: usize, ef: usize, layer: usize) -> Vec<(f32, usize)> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut candidates: BinaryHeap<MinEntry> = BinaryHeap::new();
        let mut found: BinaryHeap<MaxEntry> = BinaryHeap::new();

        let d0 = l2_sq(query, &self.nodes[ep].vector);
        candidates.push(MinEntry { dist: d0, idx: ep });
        found.push(MaxEntry { dist: d0, idx: ep });
        visited.insert(ep);

        while let Some(MinEntry { dist: cd, idx: ci }) = candidates.pop() {
            let worst = found.peek().map(|e| e.dist).unwrap_or(f32::MAX);
            if cd > worst && found.len() >= ef {
                break;
            }
            if layer < self.nodes[ci].layers.len() {
                for &ni in &self.nodes[ci].layers[layer] {
                    if visited.insert(ni) {
                        let nd = l2_sq(query, &self.nodes[ni].vector);
                        let cur_worst = found.peek().map(|e| e.dist).unwrap_or(f32::MAX);
                        if nd < cur_worst || found.len() < ef {
                            candidates.push(MinEntry { dist: nd, idx: ni });
                            found.push(MaxEntry { dist: nd, idx: ni });
                            if found.len() > ef {
                                found.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<(f32, usize)> = found.into_iter().map(|e| (e.dist, e.idx)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        out
    }

    /// Heuristic neighbor pruning: keep closest M that also improve diversity.
    fn prune(&self, node_idx: usize, candidates: &[(f32, usize)], m: usize) -> Vec<usize> {
        let mut result: Vec<usize> = Vec::with_capacity(m);
        for &(d_cand, cand) in candidates {
            if result.len() >= m {
                break;
            }
            let dominated = result
                .iter()
                .any(|&r| l2_sq(&self.nodes[cand].vector, &self.nodes[r].vector) < d_cand);
            if !dominated {
                result.push(cand);
            }
        }
        // Fall back to closest-first if heuristic leaves gaps
        if result.len() < m {
            for &(_, cand) in candidates {
                if result.len() >= m {
                    break;
                }
                if !result.contains(&cand) {
                    result.push(cand);
                }
            }
        }
        result
    }

    pub fn insert(&mut self, id: u64, vector: Vec<f32>) {
        let new_idx = self.nodes.len();
        let level = self.next_level();

        let node = HnswNode {
            id,
            vector: vector.clone(),
            layers: vec![Vec::new(); level + 1],
        };
        self.nodes.push(node);

        let Some(entry) = self.entry else {
            self.entry = Some(new_idx);
            return;
        };

        let max_level = self.nodes[entry].layers.len().saturating_sub(1);
        let mut ep = entry;

        // Greedy descent to target level
        for lc in (level + 1..=max_level).rev() {
            let found = self.search_layer(&vector, ep, 1, lc);
            if let Some(&(_, best)) = found.first() {
                ep = best;
            }
        }

        // Insert at each level from min(level, max_level) down to 0
        for lc in (0..=level.min(max_level)).rev() {
            let m_lc = if lc == 0 { self.m0 } else { self.m };
            let found = self.search_layer(&vector, ep, self.ef_construction, lc);

            if let Some(&(_, best)) = found.first() {
                ep = best;
            }

            let neighbors = self.prune(new_idx, &found, m_lc);
            self.nodes[new_idx].layers[lc] = neighbors.clone();

            // Add reverse connections
            for &ni in &neighbors {
                if self.nodes[ni].layers.len() > lc {
                    self.nodes[ni].layers[lc].push(new_idx);
                    if self.nodes[ni].layers[lc].len() > m_lc {
                        let rev_cands: Vec<(f32, usize)> = self.nodes[ni].layers[lc]
                            .iter()
                            .map(|&j| (l2_sq(&self.nodes[ni].vector, &self.nodes[j].vector), j))
                            .collect();
                        let pruned = self.prune(ni, &rev_cands, m_lc);
                        self.nodes[ni].layers[lc] = pruned;
                    }
                }
            }
        }

        // If new node reaches higher level, make it the entry point
        if level > max_level {
            self.entry = Some(new_idx);
        }
    }

    /// Return the k nearest vectors to `query`, sorted by ascending L2².
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(f32, u64)> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let max_level = self.nodes[entry].layers.len().saturating_sub(1);
        let mut ep = entry;

        for lc in (1..=max_level).rev() {
            let found = self.search_layer(query, ep, 1, lc);
            if let Some(&(_, best)) = found.first() {
                ep = best;
            }
        }

        let mut found = self.search_layer(query, ep, ef.max(k), 0);
        found.truncate(k);
        found.iter().map(|&(d, i)| (d, self.nodes[i].id)).collect()
    }

    /// Rough memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.nodes.iter().fold(0usize, |acc, n| {
            acc + n.vector.len() * 4 + n.layers.iter().map(|l| l.len() * 8).sum::<usize>() + 32
            // struct overhead approx
        })
    }
}
