//! Three beam-search variants differentiated by entropy-based beam control.
//!
//! All variants operate on a [`FlatGraph`] (single-layer k-NN proximity graph)
//! which represents HNSW layer-0. The entropy innovation applies identically
//! to multi-layer HNSW; the flat graph keeps the PoC self-contained.

use crate::graph::FlatGraph;
use crate::heap_entropy;
use std::collections::{BinaryHeap, HashSet};

// ─── core types ──────────────────────────────────────────────────────────────

/// A single ANN result: (vector id, L2-squared distance to query).
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: usize,
    pub dist: f32,
}

// Max-heap by distance (farthest on top → easy to drop when over capacity).
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── trait ───────────────────────────────────────────────────────────────────

/// Common search interface shared by all three variants.
pub trait Searcher: Send + Sync {
    /// Return up to `k` approximate nearest neighbours.
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;

    /// Human-readable name for reporting.
    fn name(&self) -> &str;

    /// Estimated index memory in bytes (vectors + adjacency lists).
    fn memory_bytes(&self) -> usize;
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Brute-force entry point: scan all nodes, return the exact closest.
///
/// A flat single-layer k-NN graph has no upper-layer routing to bootstrap
/// from. Production HNSW handles this with upper graph layers; for this PoC
/// (which focuses on beam-width control, not entry routing), brute-force
/// entry eliminates the entry-quality variable and lets the entropy signal
/// be studied cleanly.
///
/// Cost: O(n × dim) — acceptable for PoC datasets up to ~50 k vectors.
fn find_entry(graph: &FlatGraph, query: &[f32]) -> usize {
    if graph.is_empty() {
        return 0;
    }
    (0..graph.len())
        .min_by(|&i, &j| {
            let di = l2sq(query, &graph.vectors[i]);
            let dj = l2sq(query, &graph.vectors[j]);
            di.partial_cmp(&dj).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0)
}

/// Shared beam expansion: iterate frontier BinaryHeap (min-by-dist, using Reverse).
/// Returns the top-k results sorted ascending by distance.
fn beam_expand(graph: &FlatGraph, query: &[f32], entry: usize, ef: usize) -> Vec<Hit> {
    // candidates: min-heap by dist (next to expand = closest)
    let mut candidates: BinaryHeap<std::cmp::Reverse<Hit>> = BinaryHeap::new();
    // results: max-heap, capacity ef (farthest drops when full)
    let mut results: BinaryHeap<Hit> = BinaryHeap::new();
    let mut visited: HashSet<usize> = HashSet::new();

    let entry_dist = l2sq(query, &graph.vectors[entry]);
    candidates.push(std::cmp::Reverse(Hit {
        id: entry,
        dist: entry_dist,
    }));
    results.push(Hit {
        id: entry,
        dist: entry_dist,
    });
    visited.insert(entry);

    while let Some(std::cmp::Reverse(current)) = candidates.pop() {
        // Pruning: if this candidate is farther than worst in results heap,
        // and results is full, stop.
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if current.dist > worst.dist {
                    break;
                }
            }
        }

        for &(nd, neighbour) in &graph.adjacency[current.id] {
            if visited.contains(&neighbour) {
                continue;
            }
            visited.insert(neighbour);

            let dist = l2sq(query, &graph.vectors[neighbour]);

            // Always add to candidates for future exploration
            candidates.push(std::cmp::Reverse(Hit {
                id: neighbour,
                dist,
            }));

            // Update results heap
            if results.len() < ef {
                results.push(Hit {
                    id: neighbour,
                    dist,
                });
            } else if let Some(worst) = results.peek() {
                if dist < worst.dist {
                    results.pop();
                    results.push(Hit {
                        id: neighbour,
                        dist,
                    });
                }
            }
            let _ = nd; // graph stores dist² alongside neighbour id
        }
    }

    // into_sorted_vec() on a max-heap returns ascending (closest first)
    results.into_sorted_vec()
}

// ─── Variant 1: FixedEfSearch (baseline) ─────────────────────────────────────

/// Baseline HNSW-style greedy beam search with a fixed `ef_search` budget.
///
/// This is the standard approach: search until the results heap has `ef` entries
/// and no closer candidate remains in the frontier.
pub struct FixedEfSearch<'a> {
    pub graph: &'a FlatGraph,
    pub ef_search: usize,
}

impl<'a> Searcher for FixedEfSearch<'a> {
    fn name(&self) -> &str {
        "FixedEf"
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let ef = self.ef_search.max(k);
        let entry = find_entry(self.graph, query);
        let mut results = beam_expand(self.graph, query, entry, ef);
        results.truncate(k);
        results
    }

    fn memory_bytes(&self) -> usize {
        let vecs: usize = self.graph.vectors.iter().map(|v| v.len() * 4).sum();
        let adj: usize = self
            .graph
            .adjacency
            .iter()
            .map(|a| a.len() * 8)
            .sum::<usize>();
        vecs + adj
    }
}

// ─── Variant 2: EntropyThresholdBeam ─────────────────────────────────────────

/// Entropy-reactive beam search: stop traversal when heap entropy drops below H_min.
///
/// After each neighbour expansion, compute the Shannon entropy of the result-heap
/// distance distribution via softmin temperature. If H < h_stop, the search
/// frontier has converged (neighbours are tightly clustered) and early termination
/// is safe. If H remains high, continue expanding.
///
/// This variant uses the **same ef budget as FixedEf** and may stop earlier for
/// easy (low-entropy) queries.
///
/// **Measured PoC outcome:** with brute-force entry on the benchmark data the
/// heap distribution is near-uniform for every query (H ≈ ln(heap_size)), so
/// the `h_stop` gate never fires — this variant adds per-step entropy overhead
/// without any early exit or recall change.
pub struct EntropyThresholdBeam<'a> {
    pub graph: &'a FlatGraph,
    pub ef_search: usize,
    /// Entropy below which we stop expanding (in nats).
    /// Typical range: 0.3 – 1.5 depending on dataset.
    pub h_stop: f32,
    /// Softmin temperature for entropy computation.
    pub temperature: f32,
}

impl<'a> Searcher for EntropyThresholdBeam<'a> {
    fn name(&self) -> &str {
        "EntropyThreshold"
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let ef = self.ef_search.max(k);
        let entry = find_entry(self.graph, query);

        let mut candidates: BinaryHeap<std::cmp::Reverse<Hit>> = BinaryHeap::new();
        let mut results: BinaryHeap<Hit> = BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();

        let entry_dist = l2sq(query, &self.graph.vectors[entry]);
        candidates.push(std::cmp::Reverse(Hit {
            id: entry,
            dist: entry_dist,
        }));
        results.push(Hit {
            id: entry,
            dist: entry_dist,
        });
        visited.insert(entry);

        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.dist > worst.dist {
                        break;
                    }
                }
            }

            // ── Entropy gate ──────────────────────────────────────────────
            // Only check entropy when we have at least k results; checking too
            // early would short-circuit before gathering enough candidates.
            if results.len() >= k.max(4) {
                let dists: Vec<f32> = results.iter().map(|h| h.dist).collect();
                let h = heap_entropy(&dists, self.temperature);
                if h < self.h_stop {
                    break; // already converged — stop expanding
                }
            }

            for &(_, neighbour) in &self.graph.adjacency[current.id] {
                if visited.contains(&neighbour) {
                    continue;
                }
                visited.insert(neighbour);

                let dist = l2sq(query, &self.graph.vectors[neighbour]);
                candidates.push(std::cmp::Reverse(Hit {
                    id: neighbour,
                    dist,
                }));

                if results.len() < ef {
                    results.push(Hit {
                        id: neighbour,
                        dist,
                    });
                } else if let Some(worst) = results.peek() {
                    if dist < worst.dist {
                        results.pop();
                        results.push(Hit {
                            id: neighbour,
                            dist,
                        });
                    }
                }
            }
        }

        let mut out: Vec<Hit> = results.into_sorted_vec(); // ascending: closest first
        out.truncate(k);
        out
    }

    fn memory_bytes(&self) -> usize {
        let vecs: usize = self.graph.vectors.iter().map(|v| v.len() * 4).sum();
        let adj: usize = self
            .graph
            .adjacency
            .iter()
            .map(|a| a.len() * 8)
            .sum::<usize>();
        vecs + adj
    }
}

// ─── Variant 3: EntropyScaledEf ──────────────────────────────────────────────

/// Entropy-predictive beam search: sample entropy at a fixed probe depth, then
/// scale `ef` for the remainder of the search.
///
/// The intuition mirrors EDEN (arXiv:2605.09745): after `probe_depth` expansion
/// steps, the heap's entropy would predict whether the query is "easy" or
/// "hard". We set (H_max = ln(|probe results|), matching the code below):
///
///   ef_actual = base_ef * clamp(1 + alpha * H / ln(|results|), ef_min_factor, ef_max_factor)
///
/// The hypothesis:
/// - Easy queries (H near 0): ef_actual ≈ base_ef * 1.0 (no change needed)
/// - Hard queries (H near H_max): ef_actual = base_ef * ef_max_factor (expand more)
///
/// The ef scaling happens once per query after the probe phase; no per-step entropy.
///
/// **Measured PoC outcome (negative result):** on the benchmark data H ≈ H_max
/// for every query, so `ef_actual` saturates near `base_ef * ef_max_factor`
/// (122–124 for every query at base_ef=50, alpha=1.5, ef_max_factor=2.5).
/// There is no per-query adaptivity: `FixedEfSearch` with the matched budget
/// (ef=124) reproduces this variant's recall to four decimal places.
pub struct EntropyScaledEf<'a> {
    pub graph: &'a FlatGraph,
    pub base_ef: usize,
    /// Number of expansion steps before sampling entropy.
    pub probe_depth: usize,
    /// Multiplier sensitivity: ef = base_ef * (1 + alpha * H / H_max).
    pub alpha: f32,
    /// Minimum ef as a fraction of base_ef (floor, e.g. 0.5).
    pub ef_min_factor: f32,
    /// Maximum ef as a multiple of base_ef (ceiling, e.g. 3.0).
    pub ef_max_factor: f32,
    pub temperature: f32,
}

impl<'a> Searcher for EntropyScaledEf<'a> {
    fn name(&self) -> &str {
        "EntropyScaledEf"
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let entry = find_entry(self.graph, query);

        let mut candidates: BinaryHeap<std::cmp::Reverse<Hit>> = BinaryHeap::new();
        let mut results: BinaryHeap<Hit> = BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();

        let entry_dist = l2sq(query, &self.graph.vectors[entry]);
        candidates.push(std::cmp::Reverse(Hit {
            id: entry,
            dist: entry_dist,
        }));
        results.push(Hit {
            id: entry,
            dist: entry_dist,
        });
        visited.insert(entry);

        // Phase 1: probe phase — expand probe_depth steps with base_ef cap
        let probe_ef = self.base_ef;
        let mut steps = 0usize;
        while steps < self.probe_depth {
            if let Some(std::cmp::Reverse(current)) = candidates.pop() {
                if results.len() >= probe_ef {
                    if let Some(worst) = results.peek() {
                        if current.dist > worst.dist {
                            break;
                        }
                    }
                }
                for &(_, neighbour) in &self.graph.adjacency[current.id] {
                    if visited.contains(&neighbour) {
                        continue;
                    }
                    visited.insert(neighbour);
                    let dist = l2sq(query, &self.graph.vectors[neighbour]);
                    candidates.push(std::cmp::Reverse(Hit {
                        id: neighbour,
                        dist,
                    }));
                    if results.len() < probe_ef {
                        results.push(Hit {
                            id: neighbour,
                            dist,
                        });
                    } else if let Some(worst) = results.peek() {
                        if dist < worst.dist {
                            results.pop();
                            results.push(Hit {
                                id: neighbour,
                                dist,
                            });
                        }
                    }
                }
                steps += 1;
            } else {
                break;
            }
        }

        // Phase 2: compute entropy from probe results, scale ef
        let dists: Vec<f32> = results.iter().map(|h| h.dist).collect();
        let h = heap_entropy(&dists, self.temperature);
        // H_max for the probe's result set (not k): uniform over dists.len() items
        let h_max = (dists.len().max(2) as f32).ln();
        let scale = 1.0 + self.alpha * (h / h_max.max(1e-6));
        let scale = scale.max(self.ef_min_factor).min(self.ef_max_factor);
        let ef_actual = ((self.base_ef as f32 * scale) as usize).max(k);

        // Phase 3: continue search with the entropy-scaled ef
        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            if results.len() >= ef_actual {
                if let Some(worst) = results.peek() {
                    if current.dist > worst.dist {
                        break;
                    }
                }
            }
            for &(_, neighbour) in &self.graph.adjacency[current.id] {
                if visited.contains(&neighbour) {
                    continue;
                }
                visited.insert(neighbour);
                let dist = l2sq(query, &self.graph.vectors[neighbour]);
                candidates.push(std::cmp::Reverse(Hit {
                    id: neighbour,
                    dist,
                }));
                if results.len() < ef_actual {
                    results.push(Hit {
                        id: neighbour,
                        dist,
                    });
                } else if let Some(worst) = results.peek() {
                    if dist < worst.dist {
                        results.pop();
                        results.push(Hit {
                            id: neighbour,
                            dist,
                        });
                    }
                }
            }
        }

        let mut out: Vec<Hit> = results.into_sorted_vec(); // ascending: closest first
        out.truncate(k);
        out
    }

    fn memory_bytes(&self) -> usize {
        let vecs: usize = self.graph.vectors.iter().map(|v| v.len() * 4).sum();
        let adj: usize = self
            .graph
            .adjacency
            .iter()
            .map(|a| a.len() * 8)
            .sum::<usize>();
        vecs + adj
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dataset::{clustered_vectors, ground_truth, random_unit_vectors},
        graph::{FlatGraph, GraphConfig},
        recall_at_k,
    };

    fn build_test_graph(n: usize, dim: usize) -> FlatGraph {
        let vecs = random_unit_vectors(n, dim, 123);
        FlatGraph::build(vecs, GraphConfig { k_neighbours: 10 })
    }

    #[test]
    fn fixed_ef_returns_k_results() {
        let g = build_test_graph(200, 16);
        let query = random_unit_vectors(1, 16, 999);
        let searcher = FixedEfSearch {
            graph: &g,
            ef_search: 40,
        };
        let hits = searcher.search(&query[0], 10);
        assert_eq!(hits.len(), 10, "should return exactly k hits");
    }

    #[test]
    fn hits_are_sorted_ascending() {
        let g = build_test_graph(200, 16);
        let query = random_unit_vectors(1, 16, 77);
        let searcher = FixedEfSearch {
            graph: &g,
            ef_search: 50,
        };
        let hits = searcher.search(&query[0], 8);
        for w in hits.windows(2) {
            assert!(
                w[0].dist <= w[1].dist,
                "hits not sorted: {} > {}",
                w[0].dist,
                w[1].dist
            );
        }
    }

    /// Acceptance test: all three variants must achieve recall >= 0.80 on a
    /// clustered dataset with ef=40 and k=10.
    ///
    /// Queries are drawn from the same clustered distribution as the corpus.
    /// This models the agent-memory use case: agents query for content similar
    /// to what they've stored — queries and corpus are co-distributed.
    #[test]
    fn recall_acceptance_all_variants() {
        const N: usize = 500;
        const DIM: usize = 32;
        const K: usize = 10;
        const EF: usize = 40;
        const N_QUERIES: usize = 50;
        const RECALL_THRESHOLD: f32 = 0.80;

        let corpus = clustered_vectors(N, DIM, 10, 0.2, 42);
        // Use corpus members as queries: guarantees queries are in the corpus
        // distribution. For each query, ground truth includes the query itself;
        // a recall of 1.0 means the search found the exact neighbours.
        let queries: Vec<Vec<f32>> = corpus[..N_QUERIES].to_vec();

        let graph = FlatGraph::build(corpus.clone(), GraphConfig { k_neighbours: 16 });

        let fixed = FixedEfSearch {
            graph: &graph,
            ef_search: EF,
        };
        let entropy_thresh = EntropyThresholdBeam {
            graph: &graph,
            ef_search: EF,
            h_stop: 0.5,
            temperature: 0.3,
        };
        let entropy_scaled = EntropyScaledEf {
            graph: &graph,
            base_ef: EF,
            probe_depth: 5,
            alpha: 1.5,
            ef_min_factor: 0.8,
            ef_max_factor: 2.5,
            temperature: 0.3,
        };

        let searchers: Vec<(&dyn Searcher, &str)> = vec![
            (&fixed, "FixedEf"),
            (&entropy_thresh, "EntropyThreshold"),
            (&entropy_scaled, "EntropyScaledEf"),
        ];

        for (searcher, name) in searchers {
            let total_recall: f32 = queries
                .iter()
                .map(|q| {
                    let gt = ground_truth(q, &corpus, K);
                    let hits = searcher.search(q, K);
                    recall_at_k(&gt, &hits, K)
                })
                .sum::<f32>()
                / N_QUERIES as f32;

            assert!(
                total_recall >= RECALL_THRESHOLD,
                "{name}: recall@{K}={total_recall:.3} < {RECALL_THRESHOLD}"
            );
            println!("{name}: recall@{K}={total_recall:.3} ✓");
        }
    }

    #[test]
    fn entropy_threshold_returns_k_hits_on_easy_query() {
        // NOTE: despite the original hypothesis, the h_stop gate was measured to
        // never fire on this kind of data (heap entropy stays near ln(heap_size)
        // for every query). This test only verifies the variant still returns k
        // well-formed hits; it does NOT demonstrate early exit.
        let corpus = clustered_vectors(300, 16, 3, 0.05, 55); // very tight clusters
        let queries = clustered_vectors(5, 16, 3, 0.01, 56); // queries very near centroids
        let graph = FlatGraph::build(corpus.clone(), GraphConfig { k_neighbours: 12 });

        let searcher = EntropyThresholdBeam {
            graph: &graph,
            ef_search: 60,
            h_stop: 0.5,
            temperature: 0.3,
        };

        // Should still return k results even with early exit
        for q in &queries {
            let hits = searcher.search(q, 5);
            assert_eq!(hits.len(), 5, "should always return k hits");
        }
    }

    #[test]
    fn entropy_scaled_ef_recall_on_hard_queries() {
        // NOTE: this test does NOT observe ef expansion (ef_actual is internal).
        // Measured behaviour is that ef_actual saturates near
        // base_ef * ef_max_factor for every query — see ADR-303. This test only
        // verifies recall stays reasonable on hard queries.
        let corpus = clustered_vectors(400, 16, 8, 0.3, 99); // loose clusters
        let gt_corpus = corpus.clone();
        let graph = FlatGraph::build(corpus, GraphConfig { k_neighbours: 14 });

        // Query designed to be hard (random, not cluster-aligned)
        let hard_queries = random_unit_vectors(10, 16, 321);
        let searcher = EntropyScaledEf {
            graph: &graph,
            base_ef: 30,
            probe_depth: 5,
            alpha: 1.5,
            ef_min_factor: 0.8,
            ef_max_factor: 2.5,
            temperature: 0.3,
        };

        let total_recall: f32 = hard_queries
            .iter()
            .map(|q| {
                let gt = ground_truth(q, &gt_corpus, 5);
                let hits = searcher.search(q, 5);
                recall_at_k(&gt, &hits, 5)
            })
            .sum::<f32>()
            / 10.0;

        // Hard queries are hard; accept recall >= 0.60 here
        assert!(
            total_recall >= 0.60,
            "hard query recall {total_recall:.3} < 0.60"
        );
    }
}
