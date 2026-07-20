//! Adaptive beam-width ANN search with query difficulty estimation.
//!
//! Standard HNSW uses a fixed `ef` (beam width) for all queries, regardless
//! of whether the query is "easy" (clearly separated nearest neighbours) or
//! "hard" (equidistant neighbours, boundary effects).  This crate demonstrates
//! three strategies that sit on top of a shared proximity graph:
//!
//! - [`search::FixedEfSearch`]   — baseline: always ef=32
//! - [`search::TwoStageSearch`]  — try ef=16, escalate to ef=64 only if hard
//! - [`search::AdaptiveEfSearch`]— predict ef ∈ [16,64] from difficulty score
//!
//! All three implement [`search::AnnSearch`] against [`ProximityGraph`], the
//! same underlying index (HNSW layer-0 analogue: proximity graph with M edges
//! per node, built by greedy beam-search insertion).

pub mod difficulty;
pub mod search;

use std::cmp::Reverse;
use std::collections::BinaryHeap;

// ── Public types ──────────────────────────────────────────────────────────────

/// A single search result.
#[derive(Debug, Clone, PartialEq)]
pub struct AnnResult {
    pub id: usize,
    pub distance: f32,
}

/// Per-query diagnostics returned by every search variant.
#[derive(Debug, Clone)]
pub struct SearchStats {
    pub variant: &'static str,
    /// Effective ef used for the final result pass.
    pub ef_used: usize,
    /// Difficulty score in [0,1] (0 = easy, 1 = hard).
    pub difficulty: f32,
    /// Total l2_sq calls made.
    pub distance_ops: usize,
    /// Whether the search escalated from fast to full pass.
    pub escalated: bool,
}

// ── Core distance primitive ───────────────────────────────────────────────────

/// Squared L2 distance.  Positive-definite, so f32 bits compare correctly.
#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ── Proximity graph index ─────────────────────────────────────────────────────

/// Proximity graph (HNSW layer-0 analogue).
///
/// Each node has at most `m_max = 2 * m` out-edges to approximate nearest
/// neighbours inserted at build time.  Beam search over this graph is the
/// common primitive used by all search variants.
pub struct ProximityGraph {
    pub vectors: Vec<Vec<f32>>,
    pub neighbors: Vec<Vec<usize>>,
    pub m: usize,
    pub m_max: usize,
}

impl ProximityGraph {
    pub fn new(m: usize) -> Self {
        ProximityGraph {
            vectors: Vec::new(),
            neighbors: Vec::new(),
            m,
            m_max: m * 2,
        }
    }

    /// Insert `vec` into the graph, linking it to M nearest existing vectors.
    pub fn insert(&mut self, vec: Vec<f32>, ef_construction: usize) {
        let new_id = self.vectors.len();
        self.vectors.push(vec);
        self.neighbors.push(Vec::new());

        if new_id == 0 {
            return;
        }

        let q = self.vectors[new_id].clone();
        let (candidates, _) = self.beam_search(&q, self.m, ef_construction);
        let my_neighbors: Vec<usize> = candidates.iter().take(self.m).map(|r| r.id).collect();

        for &nb in &my_neighbors {
            if self.neighbors[nb].len() < self.m_max {
                self.neighbors[nb].push(new_id);
            }
        }
        self.neighbors[new_id] = my_neighbors;
    }

    /// Beam search from entry node 0 with beam width `ef`.
    ///
    /// Returns top-`k` results sorted by distance, plus the number of
    /// `l2_sq` calls made.
    pub fn beam_search(&self, query: &[f32], k: usize, ef: usize) -> (Vec<AnnResult>, usize) {
        let n = self.vectors.len();
        if n == 0 {
            return (vec![], 0);
        }

        let mut ops = 0usize;
        let mut visited = vec![false; n];
        visited[0] = true;

        let d0 = l2_sq(query, &self.vectors[0]);
        ops += 1;

        // c_heap: min-heap of candidates to explore (closest first via Reverse)
        // w_heap: max-heap of working result set (farthest pops first)
        // Using f32 bits for keys — valid since L2 distances are non-negative.
        let mut c_heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
        let mut w_heap: BinaryHeap<(u32, usize)> = BinaryHeap::new();

        c_heap.push(Reverse((d0.to_bits(), 0)));
        w_heap.push((d0.to_bits(), 0));

        while let Some(Reverse((cd_bits, c_id))) = c_heap.pop() {
            let cd = f32::from_bits(cd_bits);
            let fd = w_heap
                .peek()
                .map(|&(b, _)| f32::from_bits(b))
                .unwrap_or(f32::MAX);
            if cd > fd {
                break;
            }

            for &nb in &self.neighbors[c_id] {
                if visited[nb] {
                    continue;
                }
                visited[nb] = true;

                let nd = l2_sq(query, &self.vectors[nb]);
                ops += 1;

                let fd2 = w_heap
                    .peek()
                    .map(|&(b, _)| f32::from_bits(b))
                    .unwrap_or(f32::MAX);
                if nd < fd2 || w_heap.len() < ef {
                    c_heap.push(Reverse((nd.to_bits(), nb)));
                    w_heap.push((nd.to_bits(), nb));
                    if w_heap.len() > ef {
                        w_heap.pop();
                    }
                }
            }
        }

        let mut results: Vec<AnnResult> = w_heap
            .into_iter()
            .map(|(d_bits, id)| AnnResult {
                id,
                distance: f32::from_bits(d_bits),
            })
            .collect();
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        (results, ops)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Approximate memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        let vec_mem: usize = self.vectors.iter().map(|v| v.len() * 4).sum();
        let nb_mem: usize = self.neighbors.iter().map(|nb| nb.len() * 8).sum();
        vec_mem + nb_mem
    }
}

// ── Brute-force ground truth (for tests + benchmark) ─────────────────────────

/// Return indices of k nearest vectors to query by exact L2.
pub fn brute_knn(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2_sq(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.iter().take(k).map(|(id, _)| *id).collect()
}

/// Recall@k: fraction of ground truth IDs present in results.
pub fn recall_at_k(results: &[AnnResult], ground_truth: &[usize]) -> f32 {
    use std::collections::HashSet;
    let gt_set: HashSet<usize> = ground_truth.iter().copied().collect();
    results.iter().filter(|r| gt_set.contains(&r.id)).count() as f32 / ground_truth.len() as f32
}

// ── Deterministic data generation ─────────────────────────────────────────────

/// Xorshift64 PRNG — reproducible, no external deps.
pub struct Xorshift64(pub u64);

impl Xorshift64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    /// Box-Muller normal variate.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

/// Generate `n` random Gaussian vectors of dimension `dim`.
pub fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Xorshift64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.next_normal()).collect())
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{AdaptiveEfSearch, AnnSearch, FixedEfSearch, TwoStageSearch};

    fn small_graph(n: usize, dim: usize) -> ProximityGraph {
        let corpus = generate_vectors(n, dim, 42);
        let mut g = ProximityGraph::new(8);
        for v in corpus {
            g.insert(v, 32);
        }
        g
    }

    #[test]
    fn results_sorted_ascending() {
        let g = small_graph(200, 32);
        let query = generate_vectors(1, 32, 99)[0].clone();
        let (results, _) = g.beam_search(&query, 10, 32);
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance + 1e-6);
        }
    }

    #[test]
    fn difficulty_easy_vs_hard() {
        use crate::difficulty::distance_ratio_score;
        let easy = vec![
            AnnResult {
                id: 0,
                distance: 0.1,
            },
            AnnResult {
                id: 1,
                distance: 10.0,
            },
        ];
        let hard = vec![
            AnnResult {
                id: 0,
                distance: 1.0,
            },
            AnnResult {
                id: 1,
                distance: 1.05,
            },
        ];
        assert!(
            distance_ratio_score(&easy) < 0.05,
            "easy query should have low score"
        );
        assert!(
            distance_ratio_score(&hard) > 0.90,
            "hard query should have high score"
        );
    }

    #[test]
    fn all_variants_return_results() {
        let g = small_graph(300, 32);
        let query = generate_vectors(1, 32, 7)[0].clone();
        let k = 10;

        let (r1, _) = FixedEfSearch { ef: 32 }.search(&g, &query, k);
        let (r2, _) = TwoStageSearch {
            ef_fast: 16,
            ef_full: 64,
            threshold: 0.7,
        }
        .search(&g, &query, k);
        let (r3, _) = AdaptiveEfSearch {
            ef_min: 16,
            ef_max: 64,
            threshold: 0.4,
        }
        .search(&g, &query, k);

        assert!(!r1.is_empty());
        assert!(!r2.is_empty());
        assert!(!r3.is_empty());
    }

    #[test]
    fn recall_above_floor() {
        // FixedEf(64) on N=500, D=32 should achieve >= 0.75 recall@10.
        let corpus = generate_vectors(500, 32, 1);
        let mut g = ProximityGraph::new(8);
        for v in &corpus {
            g.insert(v.clone(), 32);
        }
        let queries = generate_vectors(20, 32, 2);
        let searcher = FixedEfSearch { ef: 64 };
        let mut total = 0.0f32;

        for q in &queries {
            let (results, _) = searcher.search(&g, q, 10);
            let gt = brute_knn(q, &corpus, 10);
            total += recall_at_k(&results, &gt);
        }

        let mean = total / queries.len() as f32;
        assert!(mean >= 0.75, "FixedEf(64) recall {:.3} < 0.75 floor", mean);
    }

    #[test]
    fn adaptive_ef_recall_acceptable() {
        // AdaptiveEf should achieve recall within 10% of FixedEf(32) on same corpus.
        let corpus = generate_vectors(500, 32, 3);
        let mut g = ProximityGraph::new(8);
        for v in &corpus {
            g.insert(v.clone(), 32);
        }
        let queries = generate_vectors(20, 32, 4);
        let fixed = FixedEfSearch { ef: 32 };
        let adaptive = AdaptiveEfSearch {
            ef_min: 16,
            ef_max: 64,
            threshold: 0.4,
        };

        let (mut r_fixed, mut r_adap) = (0.0f32, 0.0f32);
        for q in &queries {
            let gt = brute_knn(q, &corpus, 10);
            let (res_f, _) = fixed.search(&g, q, 10);
            let (res_a, _) = adaptive.search(&g, q, 10);
            r_fixed += recall_at_k(&res_f, &gt);
            r_adap += recall_at_k(&res_a, &gt);
        }

        let mean_f = r_fixed / queries.len() as f32;
        let mean_a = r_adap / queries.len() as f32;
        // AdaptiveEf recall must be >= 90% of FixedEf recall
        assert!(
            mean_a >= mean_f * 0.90,
            "AdaptiveEf recall {:.3} < 90% of FixedEf recall {:.3}",
            mean_a,
            mean_f
        );
    }
}
