//! NSW (Navigable Small World) graph built on scalar-quantized vectors.
//!
//! Graph traversal uses fast integer-distance comparisons on SQ codes.
//! After finding `ef` candidates, full-precision L2 re-ranking is applied.
//!
//! Entry-point strategy: a sparse "seed layer" (every `seed_step`-th
//! inserted node) provides diverse starting points for both construction
//! beam searches and query time traversal.  This replaces the fixed
//! node-0 entry used in naive NSW and dramatically improves recall.
//!
//! Two exported structs share the same internal engine:
//! - [`GraphSq8`]: 8-bit quantization (4× memory compression).
//! - [`GraphSq4`]: 4-bit quantization (8× memory compression).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use crate::{l2_sq, NnResult, NnSearch, ScalarQuantizer};

// Ord wrapper for non-NaN f32 distances (all L2 values are ≥ 0).
#[derive(Clone, Copy, PartialEq)]
struct F32(f32);
impl Eq for F32 {}
impl PartialOrd for F32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&o.0)
    }
}
impl Ord for F32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.partial_cmp(o).unwrap_or(std::cmp::Ordering::Equal)
    }
}

struct Node {
    code: Vec<u8>,
    original: Vec<f32>,
    neighbors: Vec<usize>,
}

/// Internal NSW graph engine with seed-layer entry diversification.
struct SqGraph {
    nodes: Vec<Node>,
    quantizer: ScalarQuantizer,
    m: usize,
    ef_build: usize,
    bits: u8,
    /// Sparse seed nodes for diverse entry points (O(√n) in practice).
    seeds: Vec<usize>,
    /// Add a seed every `seed_step` insertions (e.g. every 100th node).
    seed_step: usize,
}

impl SqGraph {
    fn new(quantizer: ScalarQuantizer, m: usize, ef_build: usize, seed_step: usize) -> Self {
        let bits = quantizer.bits;
        Self {
            nodes: Vec::new(),
            quantizer,
            m,
            ef_build,
            bits,
            seeds: Vec::new(),
            seed_step,
        }
    }

    #[inline]
    fn dist(&self, a: &[u8], b: &[u8]) -> f32 {
        if self.bits == 8 {
            self.quantizer.l2_sq8(a, b)
        } else {
            self.quantizer.l2_sq4(a, b)
        }
    }

    fn encode(&self, v: &[f32]) -> Vec<u8> {
        if self.bits == 8 {
            self.quantizer.encode8(v)
        } else {
            self.quantizer.encode4(v)
        }
    }

    /// Find `n` seeds nearest to `q_code`, sorted by ascending quantized distance.
    fn top_seeds(&self, q_code: &[u8], n: usize) -> Vec<usize> {
        let mut scored: Vec<(F32, usize)> = self
            .seeds
            .iter()
            .map(|&i| (F32(self.dist(q_code, &self.nodes[i].code)), i))
            .collect();
        scored.sort();
        scored.into_iter().take(n).map(|(_, i)| i).collect()
    }

    /// Greedy beam search in the quantized graph.
    /// Returns up to `ef` candidates sorted by ascending quantized distance.
    fn beam_search(&self, q_code: &[u8], entry: usize, ef: usize) -> Vec<(f32, usize)> {
        if self.nodes.is_empty() {
            return vec![];
        }

        let mut visited: HashSet<usize> = HashSet::new();
        // min-heap: pop nearest first (explore candidates)
        let mut frontier: BinaryHeap<Reverse<(F32, usize)>> = BinaryHeap::new();
        // max-heap: pop farthest first (bound result set to ef)
        let mut result: BinaryHeap<(F32, usize)> = BinaryHeap::new();

        let d0 = self.dist(q_code, &self.nodes[entry].code);
        frontier.push(Reverse((F32(d0), entry)));
        result.push((F32(d0), entry));
        visited.insert(entry);

        while let Some(Reverse((F32(d_curr), curr))) = frontier.pop() {
            let worst = result.peek().map(|(F32(d), _)| *d).unwrap_or(f32::INFINITY);
            if d_curr > worst && result.len() >= ef {
                break;
            }
            for &nb in &self.nodes[curr].neighbors {
                if visited.insert(nb) {
                    let d_nb = self.dist(q_code, &self.nodes[nb].code);
                    let worst2 = result.peek().map(|(F32(d), _)| *d).unwrap_or(f32::INFINITY);
                    if d_nb < worst2 || result.len() < ef {
                        frontier.push(Reverse((F32(d_nb), nb)));
                        result.push((F32(d_nb), nb));
                        if result.len() > ef {
                            result.pop();
                        }
                    }
                }
            }
        }

        let mut res: Vec<(f32, usize)> = result.into_iter().map(|(F32(d), i)| (d, i)).collect();
        res.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        res
    }

    fn insert(&mut self, vector: Vec<f32>) {
        let code = self.encode(&vector);
        let idx = self.nodes.len();

        if idx == 0 {
            self.seeds.push(0);
            self.nodes.push(Node {
                code,
                original: vector,
                neighbors: Vec::new(),
            });
            return;
        }

        // Diverse entry: start construction beam from nearest seed.
        let entry = self.top_seeds(&code, 1).into_iter().next().unwrap_or(0);
        let cands = self.beam_search(&code, entry, self.ef_build);

        let nb_indices: Vec<usize> = cands.iter().take(self.m).map(|(_, i)| *i).collect();

        // Bidirectional edges; cap at 2*M to bound degree.
        for &nb in &nb_indices {
            if self.nodes[nb].neighbors.len() < self.m * 2 {
                self.nodes[nb].neighbors.push(idx);
            }
        }

        self.nodes.push(Node {
            code,
            original: vector,
            neighbors: nb_indices,
        });

        // Register as seed periodically.
        if idx % self.seed_step == 0 {
            self.seeds.push(idx);
        }
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<NnResult> {
        if self.nodes.is_empty() {
            return vec![];
        }
        let q_code = self.encode(query);

        // Multi-start: run independent beam searches from the 3 nearest seeds,
        // each with full ef width.  The merged union covers more of the graph
        // than any single wide beam from one entry point.
        let n_starts: usize = 3.min(self.seeds.len());
        let per_ef = ef; // full width per start for maximum coverage
        let top = self.top_seeds(&q_code, n_starts);

        let mut seen: HashSet<usize> = HashSet::new();
        let mut merged: Vec<(f32, usize)> = Vec::new();
        for start in top {
            for (d, i) in self.beam_search(&q_code, start, per_ef) {
                if seen.insert(i) {
                    merged.push((d, i));
                }
            }
        }
        // Sort merged by quantized distance, cap to ef for re-ranking budget.
        merged.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(ef);

        // Full-precision re-rank over the merged candidate set.
        let mut exact: Vec<NnResult> = merged
            .iter()
            .map(|(_, i)| NnResult {
                id: *i,
                distance: l2_sq(query, &self.nodes[*i].original),
            })
            .collect();
        exact.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        exact.truncate(k);
        exact
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn memory_bytes(&self) -> usize {
        let enc_bytes = if self.bits == 8 {
            self.quantizer.encoded_bytes8()
        } else {
            self.quantizer.encoded_bytes4()
        };
        let orig_bytes = self.quantizer.dims * 4;
        // per node: code + original + neighbors (usize = 8 bytes each)
        self.nodes.len() * (enc_bytes + orig_bytes + self.m * 2 * 8) + self.seeds.len() * 8
    }
}

// ─── Public wrappers ─────────────────────────────────────────────────────────

/// NSW graph with 8-bit scalar quantization.  4× memory compression vs f32.
///
/// Build with `new(quantizer, m=16, ef_build=200, seed_step=100, ef_search=200)`.
pub struct GraphSq8 {
    inner: SqGraph,
    ef_search: usize,
}

impl GraphSq8 {
    /// `seed_step`: add a seed entry-point every N insertions (lower = more seeds,
    /// higher recall, slightly more query overhead).  Typical: `max(1, n/100)`.
    pub fn new(
        quantizer: ScalarQuantizer,
        m: usize,
        ef_build: usize,
        seed_step: usize,
        ef_search: usize,
    ) -> Self {
        Self {
            inner: SqGraph::new(quantizer, m, ef_build, seed_step),
            ef_search,
        }
    }
}

impl NnSearch for GraphSq8 {
    fn insert(&mut self, vector: Vec<f32>) {
        self.inner.insert(vector);
    }
    fn search(&self, query: &[f32], k: usize) -> Vec<NnResult> {
        self.inner.search(query, k, self.ef_search)
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }
}

/// NSW graph with 4-bit scalar quantization.  8× memory compression vs f32.
pub struct GraphSq4 {
    inner: SqGraph,
    ef_search: usize,
}

impl GraphSq4 {
    pub fn new(
        quantizer: ScalarQuantizer,
        m: usize,
        ef_build: usize,
        seed_step: usize,
        ef_search: usize,
    ) -> Self {
        Self {
            inner: SqGraph::new(quantizer, m, ef_build, seed_step),
            ef_search,
        }
    }
}

impl NnSearch for GraphSq4 {
    fn insert(&mut self, vector: Vec<f32>) {
        self.inner.insert(vector);
    }
    fn search(&self, query: &[f32], k: usize) -> Vec<NnResult> {
        self.inner.search(query, k, self.ef_search)
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }
}
