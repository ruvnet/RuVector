//! Beam-search variants differentiated by their traversal stopping rule.
//!
//! All variants operate on a [`FlatGraph`] (single-layer k-NN proximity
//! graph, representing HNSW layer-0) from a single fixed entry point.
//!
//! - [`FixedEf`]: the standard approach — expand until the result heap holds
//!   `ef_search` entries and no closer frontier candidate remains.
//! - [`AdaptiveGamma`]: the *distance-adaptive* stopping rule from
//!   "Distance Adaptive Beam Search for Provably Accurate Graph-Based
//!   Nearest Neighbor Search" (arXiv:2505.15636). No `ef` parameter: stop
//!   expanding as soon as the closest unexpanded frontier candidate `x`
//!   satisfies `d(q,x) >= (1+gamma) * d(q,x_k)`, where `x_k` is the current
//!   k-th best discovered distance. `gamma in (0, 2]`; the paper proves that
//!   every undiscovered node is then at least `(gamma/2) * max_j d(q,j)`
//!   away, an approximation-factor-`2/gamma` guarantee on navigable graphs
//!   (this flat k-NN graph is not proven navigable, so the guarantee is not
//!   claimed to transfer exactly — see the research README's attack pass).
//!   [`AdaptiveGamma`] with `max_expansions = Some(_)` is the same rule with
//!   a hard expansion cap: a production safety bound against the case where
//!   the graph is not navigable enough for the ratio test to fire.

use crate::graph::FlatGraph;
use std::collections::{BinaryHeap, HashSet};

// ─── core types ──────────────────────────────────────────────────────────────

/// A single ANN result: (vector id, squared-L2 distance to query).
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: usize,
    pub dist: f32,
}

// Max-heap by distance (farthest on top -> easy to evict when over capacity).
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

/// Result of one query, including the work performed to produce it.
pub struct SearchOutcome {
    pub hits: Vec<Hit>,
    /// Count of `l2sq` calls made during this query's traversal (excludes
    /// the one-time, build-time entry-point computation).
    pub dist_computations: u64,
    /// Count of nodes popped from the frontier and expanded.
    pub expansions: u64,
}

// ─── trait ───────────────────────────────────────────────────────────────────

pub trait Searcher: Send + Sync {
    fn search(&self, query: &[f32], k: usize) -> SearchOutcome;
    fn name(&self) -> String;
    fn memory_bytes(&self) -> usize;
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ─── Variant: FixedEf (baseline) ─────────────────────────────────────────────

/// Standard HNSW-style greedy beam search with a fixed `ef_search` budget:
/// expand until the results heap holds `ef_search` entries and the closest
/// remaining frontier candidate is farther than the current worst result.
pub struct FixedEf<'a> {
    pub graph: &'a FlatGraph,
    pub ef_search: usize,
}

impl Searcher for FixedEf<'_> {
    fn name(&self) -> String {
        format!("FixedEf({})", self.ef_search)
    }

    fn search(&self, query: &[f32], k: usize) -> SearchOutcome {
        let ef = self.ef_search.max(k);
        let (entry, entry_dist, mut dist_computations) = self.graph.route_entry(query);

        let mut candidates: BinaryHeap<std::cmp::Reverse<Hit>> = BinaryHeap::new();
        let mut results: BinaryHeap<Hit> = BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut expansions: u64 = 0;

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
            expansions += 1;

            for &(_, neighbour) in &self.graph.adjacency[current.id] {
                if visited.contains(&neighbour) {
                    continue;
                }
                visited.insert(neighbour);

                let dist = l2sq(query, &self.graph.vectors[neighbour]);
                dist_computations += 1;
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

        let mut hits = results.into_sorted_vec();
        hits.truncate(k);
        SearchOutcome {
            hits,
            dist_computations,
            expansions,
        }
    }

    fn memory_bytes(&self) -> usize {
        self.graph.memory_bytes()
    }
}

// ─── Variant: AdaptiveGamma (distance-adaptive stopping rule) ───────────────

/// Distance-adaptive beam search (arXiv:2505.15636): stop expanding when the
/// closest unexpanded frontier candidate is farther than
/// `(1 + gamma) * d(q, x_k)`, where `x_k` is the current k-th best discovered
/// distance. Unlike [`FixedEf`], the result heap capacity is `k` itself —
/// there is no separate `ef` budget to tune.
///
/// `max_expansions`, when set, additionally caps the number of frontier
/// nodes expanded regardless of the gamma criterion — a production safety
/// bound tested independently of the gamma rule's own behaviour.
pub struct AdaptiveGamma<'a> {
    pub graph: &'a FlatGraph,
    pub gamma: f32,
    pub max_expansions: Option<u64>,
}

impl Searcher for AdaptiveGamma<'_> {
    fn name(&self) -> String {
        match self.max_expansions {
            Some(cap) => format!("Adaptive(g={:.1},cap={cap})", self.gamma),
            None => format!("Adaptive(g={:.1})", self.gamma),
        }
    }

    fn search(&self, query: &[f32], k: usize) -> SearchOutcome {
        let (entry, entry_dist, mut dist_computations) = self.graph.route_entry(query);

        let mut candidates: BinaryHeap<std::cmp::Reverse<Hit>> = BinaryHeap::new();
        let mut results: BinaryHeap<Hit> = BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut expansions: u64 = 0;

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
            if results.len() >= k {
                let kth_best = results.peek().unwrap().dist;
                if current.dist >= (1.0 + self.gamma) * kth_best {
                    break;
                }
            }
            if let Some(cap) = self.max_expansions {
                if expansions >= cap {
                    break;
                }
            }
            expansions += 1;

            for &(_, neighbour) in &self.graph.adjacency[current.id] {
                if visited.contains(&neighbour) {
                    continue;
                }
                visited.insert(neighbour);

                let dist = l2sq(query, &self.graph.vectors[neighbour]);
                dist_computations += 1;
                candidates.push(std::cmp::Reverse(Hit {
                    id: neighbour,
                    dist,
                }));

                if results.len() < k {
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

        let mut hits = results.into_sorted_vec();
        hits.truncate(k);
        SearchOutcome {
            hits,
            dist_computations,
            expansions,
        }
    }

    fn memory_bytes(&self) -> usize {
        self.graph.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{clustered_vectors, ground_truth};
    use crate::graph::GraphConfig;

    fn build_test_graph() -> FlatGraph {
        let vecs = clustered_vectors(300, 8, 5, 0.15, 7);
        FlatGraph::build(
            vecs,
            GraphConfig {
                k_neighbours: 12,
                num_entry_seeds: 24,
            },
        )
    }

    #[test]
    fn fixed_ef_returns_k_hits_on_connected_graph() {
        let graph = build_test_graph();
        let searcher = FixedEf {
            graph: &graph,
            ef_search: 30,
        };
        let out = searcher.search(&graph.vectors[0], 10);
        assert_eq!(out.hits.len(), 10);
        assert!(out.dist_computations > 0);
    }

    #[test]
    fn adaptive_gamma_returns_hits_and_counts_work() {
        let graph = build_test_graph();
        let searcher = AdaptiveGamma {
            graph: &graph,
            gamma: 0.5,
            max_expansions: None,
        };
        let out = searcher.search(&graph.vectors[0], 10);
        assert!(!out.hits.is_empty());
        assert!(out.dist_computations > 0);
        assert!(out.expansions > 0);
    }

    #[test]
    fn larger_gamma_never_expands_less_than_smaller_gamma_on_same_query() {
        // A larger gamma relaxes the stopping condition (harder to satisfy
        // d(q,x) >= (1+gamma)*d_k), so it must expand at least as much.
        let graph = build_test_graph();
        let query = &graph.vectors[3];
        let tight = AdaptiveGamma {
            graph: &graph,
            gamma: 0.1,
            max_expansions: None,
        }
        .search(query, 10);
        let loose = AdaptiveGamma {
            graph: &graph,
            gamma: 1.5,
            max_expansions: None,
        }
        .search(query, 10);
        assert!(loose.expansions >= tight.expansions);
    }

    #[test]
    fn capped_variant_never_exceeds_cap() {
        let graph = build_test_graph();
        let searcher = AdaptiveGamma {
            graph: &graph,
            gamma: 2.0,
            max_expansions: Some(5),
        };
        for i in 0..graph.len() {
            let out = searcher.search(&graph.vectors[i], 10);
            assert!(out.expansions <= 5, "expansions={}", out.expansions);
        }
    }

    #[test]
    fn loose_gamma_achieves_high_recall_on_majority_of_self_queries() {
        // gamma=2.0 is the paper's exact-recovery setting on *navigable*
        // graphs. This crate's flat exact-k-NN graph (see graph.rs docs) is
        // not proven navigable: a node can fail to appear in any other
        // node's k-NN adjacency list, making it unreachable from some entry
        // points regardless of gamma. That is a real, measured limitation
        // (see the research README's attack pass), not a bug in the
        // stopping rule — so this test checks the majority statistic, not
        // every single query.
        let graph = build_test_graph();
        let searcher = AdaptiveGamma {
            graph: &graph,
            gamma: 2.0,
            max_expansions: None,
        };
        let sample: Vec<usize> = (0..graph.len()).step_by(7).collect();
        let mut self_hit_first = 0usize;
        for &i in &sample {
            let query = &graph.vectors[i];
            let gt = ground_truth(query, &graph.vectors, 10);
            let out = searcher.search(query, 10);
            if out.hits.first().map(|h| h.id) == Some(gt[0]) {
                self_hit_first += 1;
            }
        }
        let frac = self_hit_first as f64 / sample.len() as f64;
        assert!(
            frac >= 0.8,
            "expected >= 80% of self-queries to find themselves first at gamma=2.0, got {frac:.2}"
        );
    }
}
