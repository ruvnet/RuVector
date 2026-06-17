//! Deterministic synthetic graph generators for the M0 correctness gate.
//!
//! All randomness uses a seeded SplitMix64 so runs are reproducible (an M0 exit
//! criterion). Weights are drawn from a wide continuous range to make shortest
//! paths effectively tie-free, so top-k membership is unambiguous in tests.

use crate::graph::{Graph, NodeId};

/// Minimal deterministic PRNG (SplitMix64). Zero external deps.
pub struct Rng(u64);

impl Rng {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform f64 in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Edge weight in `[0.5, 1.5)` — positive, wide, tie-free in practice.
    fn weight(&mut self) -> f64 {
        0.5 + self.next_f64()
    }
}

/// Stochastic Block Model: `blocks` communities of `per_block` vertices each,
/// dense intra-block (`p_in`), sparse inter-block (`p_out`). Clean separators by
/// construction — the SepRAG happy path.
#[must_use]
pub fn sbm(blocks: usize, per_block: usize, p_in: f64, p_out: f64, seed: u64) -> Graph {
    let n = blocks * per_block;
    let mut g = Graph::new(n);
    let mut rng = Rng::new(seed);
    let block_of = |v: usize| v / per_block;
    for u in 0..n {
        for v in (u + 1)..n {
            let p = if block_of(u) == block_of(v) { p_in } else { p_out };
            if rng.next_f64() < p {
                g.add_edge(u as NodeId, v as NodeId, rng.weight());
            }
        }
    }
    g
}

/// `w x h` 4-neighbour grid with random edge weights. Known ~min(w,h) separators.
#[must_use]
pub fn grid(w: usize, h: usize, seed: u64) -> Graph {
    let mut g = Graph::new(w * h);
    let mut rng = Rng::new(seed);
    let id = |x: usize, y: usize| (y * w + x) as NodeId;
    for y in 0..h {
        for x in 0..w {
            if x + 1 < w {
                g.add_edge(id(x, y), id(x + 1, y), rng.weight());
            }
            if y + 1 < h {
                g.add_edge(id(x, y), id(x, y + 1), rng.weight());
            }
        }
    }
    g
}

/// A path graph (degenerate: separators are single vertices, deep elim tree).
#[must_use]
pub fn path(n: usize, seed: u64) -> Graph {
    let mut g = Graph::new(n);
    let mut rng = Rng::new(seed);
    for i in 0..n.saturating_sub(1) {
        g.add_edge(i as NodeId, (i + 1) as NodeId, rng.weight());
    }
    g
}

/// A clique (degenerate worst case: full fill-in, no layer separator).
#[must_use]
pub fn clique(n: usize, seed: u64) -> Graph {
    let mut g = Graph::new(n);
    let mut rng = Rng::new(seed);
    for u in 0..n {
        for v in (u + 1)..n {
            g.add_edge(u as NodeId, v as NodeId, rng.weight());
        }
    }
    g
}

/// Deterministically sample `count` distinct POIs from `0..n`.
#[must_use]
pub fn sample_pois(n: usize, count: usize, seed: u64) -> Vec<NodeId> {
    let mut rng = Rng::new(seed ^ 0x504F_4953_504F_4953); // "POISPOIS"
    let mut chosen = vec![false; n];
    let mut out = Vec::new();
    let target = count.min(n);
    while out.len() < target {
        let v = (rng.next_u64() as usize) % n;
        if !chosen[v] {
            chosen[v] = true;
            out.push(v as NodeId);
        }
    }
    out
}
