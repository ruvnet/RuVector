//! Frontier-cut conductance: a topological (not distance-based) signal for
//! per-query adaptive beam width.
//!
//! Both variants below are intentionally structured as close analogues of
//! the `EntropyThresholdBeam` / `EntropyScaledEf` variants in
//! `ruvector-entropy-ann`, so the only thing that differs between the two
//! crates' benchmarks is the *signal*, not the control structure.

use ruvector_entropy_ann::dataset::l2sq;
use ruvector_entropy_ann::graph::FlatGraph;
use ruvector_entropy_ann::search::{Hit, Searcher};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

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

fn mem_bytes(graph: &FlatGraph) -> usize {
    let vecs: usize = graph.vectors.iter().map(|v| v.len() * 4).sum();
    let adj: usize = graph.adjacency.iter().map(|a| a.len() * 8).sum::<usize>();
    vecs + adj
}

/// Result of a probe-phase conductance measurement.
#[derive(Debug, Clone, Copy)]
pub struct CutProbeStats {
    /// Edges, examined during the probe, that pointed to a node not yet visited.
    pub cut_edges: u32,
    /// Edges, examined during the probe, that pointed back into the visited set.
    pub internal_edges: u32,
    /// `cut_edges / (cut_edges + internal_edges)`. `0.5` when no edges were seen.
    pub phi: f32,
}

/// Run a `probe_depth`-step beam expansion from the brute-force entry point
/// and report the frontier-cut conductance `phi` observed during the probe,
/// without continuing the search. Used standalone for signal-distribution
/// analysis (see the benchmark binary) and internally mirrored (not called
/// directly) by [`FrontierCutScaledEf`], which needs the live heap state to
/// continue the search afterwards.
pub fn probe_phi(
    graph: &FlatGraph,
    query: &[f32],
    probe_depth: usize,
    base_ef: usize,
) -> CutProbeStats {
    let entry = find_entry(graph, query);
    let mut candidates: BinaryHeap<Reverse<Hit>> = BinaryHeap::new();
    let mut results: BinaryHeap<Hit> = BinaryHeap::new();
    let mut visited: HashSet<usize> = HashSet::new();

    let entry_dist = l2sq(query, &graph.vectors[entry]);
    candidates.push(Reverse(Hit {
        id: entry,
        dist: entry_dist,
    }));
    results.push(Hit {
        id: entry,
        dist: entry_dist,
    });
    visited.insert(entry);

    let mut cut_edges = 0u32;
    let mut internal_edges = 0u32;
    let mut steps = 0usize;

    while steps < probe_depth {
        let Some(Reverse(current)) = candidates.pop() else {
            break;
        };
        if results.len() >= base_ef {
            if let Some(worst) = results.peek() {
                if current.dist > worst.dist {
                    break;
                }
            }
        }
        for &(_, neighbour) in &graph.adjacency[current.id] {
            if visited.contains(&neighbour) {
                internal_edges += 1;
                continue;
            }
            cut_edges += 1;
            visited.insert(neighbour);
            let dist = l2sq(query, &graph.vectors[neighbour]);
            candidates.push(Reverse(Hit {
                id: neighbour,
                dist,
            }));
            if results.len() < base_ef {
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
    }

    let total = cut_edges + internal_edges;
    let phi = if total == 0 {
        0.5
    } else {
        cut_edges as f32 / total as f32
    };
    CutProbeStats {
        cut_edges,
        internal_edges,
        phi,
    }
}

// ─── Variant A: CutThresholdBeam (reactive) ──────────────────────────────────

/// Reactive beam search: stop expanding once a *sliding-window* frontier-cut
/// conductance falls below `phi_stop`.
///
/// A cumulative-since-start conductance would trend downward for every query
/// as the visited set grows, regardless of query difficulty (more of the
/// graph becomes "already visited" over time no matter where the query
/// sits). To avoid that confound, `phi` is recomputed over a fixed-size
/// window of `window_steps` expansion steps and the gate is checked against
/// the most recently completed window, not the running total.
pub struct CutThresholdBeam<'a> {
    pub graph: &'a FlatGraph,
    pub ef_search: usize,
    /// Conductance below which the frontier is considered "sealed" and the
    /// search stops early.
    pub phi_stop: f32,
    /// Expansion steps per conductance window.
    pub window_steps: usize,
    /// Minimum steps before the gate is allowed to fire.
    pub min_steps: usize,
}

impl<'a> Searcher for CutThresholdBeam<'a> {
    fn name(&self) -> &str {
        "CutThreshold"
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let ef = self.ef_search.max(k);
        let entry = find_entry(self.graph, query);

        let mut candidates: BinaryHeap<Reverse<Hit>> = BinaryHeap::new();
        let mut results: BinaryHeap<Hit> = BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();

        let entry_dist = l2sq(query, &self.graph.vectors[entry]);
        candidates.push(Reverse(Hit {
            id: entry,
            dist: entry_dist,
        }));
        results.push(Hit {
            id: entry,
            dist: entry_dist,
        });
        visited.insert(entry);

        let mut window_cut = 0u32;
        let mut window_internal = 0u32;
        // Start optimistic (fully open): never gate before the first window completes.
        let mut last_phi = 1.0_f32;
        let mut steps = 0usize;

        while let Some(Reverse(current)) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.dist > worst.dist {
                        break;
                    }
                }
            }

            if steps >= self.min_steps && results.len() >= k.max(4) && last_phi < self.phi_stop {
                break;
            }

            for &(_, neighbour) in &self.graph.adjacency[current.id] {
                if visited.contains(&neighbour) {
                    window_internal += 1;
                    continue;
                }
                window_cut += 1;
                visited.insert(neighbour);

                let dist = l2sq(query, &self.graph.vectors[neighbour]);
                candidates.push(Reverse(Hit {
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

            steps += 1;
            if steps.is_multiple_of(self.window_steps) {
                let total = window_cut + window_internal;
                if total > 0 {
                    last_phi = window_cut as f32 / total as f32;
                }
                window_cut = 0;
                window_internal = 0;
            }
        }

        let mut out: Vec<Hit> = results.into_sorted_vec();
        out.truncate(k);
        out
    }

    fn memory_bytes(&self) -> usize {
        mem_bytes(self.graph)
    }
}

// ─── Variant B: FrontierCutScaledEf (predictive) ─────────────────────────────

/// Predictive beam search: probe frontier-cut conductance for `probe_depth`
/// steps, then scale `ef` for the remainder of the search.
///
/// Structurally identical to `EntropyScaledEf` (same probe/scale/continue
/// phases, same clamp form) so the only variable between the two crates'
/// predictive variants is the signal: conductance `phi` instead of heap
/// entropy `H`.
///
/// ```text
/// ef_actual = base_ef * clamp(1 + alpha * phi, ef_min_factor, ef_max_factor)
/// ```
pub struct FrontierCutScaledEf<'a> {
    pub graph: &'a FlatGraph,
    pub base_ef: usize,
    pub probe_depth: usize,
    pub alpha: f32,
    pub ef_min_factor: f32,
    pub ef_max_factor: f32,
}

impl<'a> Searcher for FrontierCutScaledEf<'a> {
    fn name(&self) -> &str {
        "FrontierCutScaledEf"
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let entry = find_entry(self.graph, query);

        let mut candidates: BinaryHeap<Reverse<Hit>> = BinaryHeap::new();
        let mut results: BinaryHeap<Hit> = BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();

        let entry_dist = l2sq(query, &self.graph.vectors[entry]);
        candidates.push(Reverse(Hit {
            id: entry,
            dist: entry_dist,
        }));
        results.push(Hit {
            id: entry,
            dist: entry_dist,
        });
        visited.insert(entry);

        // Phase 1: probe — expand probe_depth steps, tally cut vs internal edges.
        let probe_ef = self.base_ef;
        let mut cut_edges = 0u32;
        let mut internal_edges = 0u32;
        let mut steps = 0usize;
        while steps < self.probe_depth {
            let Some(Reverse(current)) = candidates.pop() else {
                break;
            };
            if results.len() >= probe_ef {
                if let Some(worst) = results.peek() {
                    if current.dist > worst.dist {
                        break;
                    }
                }
            }
            for &(_, neighbour) in &self.graph.adjacency[current.id] {
                if visited.contains(&neighbour) {
                    internal_edges += 1;
                    continue;
                }
                cut_edges += 1;
                visited.insert(neighbour);
                let dist = l2sq(query, &self.graph.vectors[neighbour]);
                candidates.push(Reverse(Hit {
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
        }

        // Phase 2: compute phi from the probe, scale ef.
        let total = cut_edges + internal_edges;
        let phi = if total == 0 {
            0.5
        } else {
            cut_edges as f32 / total as f32
        };
        let scale = 1.0 + self.alpha * phi;
        let scale = scale.max(self.ef_min_factor).min(self.ef_max_factor);
        let ef_actual = ((self.base_ef as f32 * scale) as usize).max(k);

        // Phase 3: continue with the conductance-scaled ef.
        while let Some(Reverse(current)) = candidates.pop() {
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
                candidates.push(Reverse(Hit {
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

        let mut out: Vec<Hit> = results.into_sorted_vec();
        out.truncate(k);
        out
    }

    fn memory_bytes(&self) -> usize {
        mem_bytes(self.graph)
    }
}

impl<'a> FrontierCutScaledEf<'a> {
    /// The `ef_actual` this variant would choose for `query`, without running
    /// the rest of the search. Exposed for benchmark-time adaptivity analysis
    /// (does `ef_actual` actually vary across queries, or does it saturate
    /// like `EntropyScaledEf` did?).
    pub fn predicted_ef(&self, query: &[f32]) -> usize {
        let stats = probe_phi(self.graph, query, self.probe_depth, self.base_ef);
        let scale = 1.0 + self.alpha * stats.phi;
        let scale = scale.max(self.ef_min_factor).min(self.ef_max_factor);
        (self.base_ef as f32 * scale) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_entropy_ann::dataset::{clustered_vectors, ground_truth, random_unit_vectors};
    use ruvector_entropy_ann::graph::GraphConfig;
    use ruvector_entropy_ann::recall_at_k;

    fn build_test_graph(n: usize, dim: usize) -> FlatGraph {
        let vecs = random_unit_vectors(n, dim, 123);
        FlatGraph::build(vecs, GraphConfig { k_neighbours: 10 })
    }

    #[test]
    fn cut_threshold_returns_k_results() {
        let g = build_test_graph(300, 16);
        let query = random_unit_vectors(1, 16, 999);
        let searcher = CutThresholdBeam {
            graph: &g,
            ef_search: 40,
            phi_stop: 0.1,
            window_steps: 3,
            min_steps: 3,
        };
        let hits = searcher.search(&query[0], 10);
        assert_eq!(hits.len(), 10);
    }

    #[test]
    fn frontier_cut_scaled_ef_returns_k_results() {
        let g = build_test_graph(300, 16);
        let query = random_unit_vectors(1, 16, 7);
        let searcher = FrontierCutScaledEf {
            graph: &g,
            base_ef: 40,
            probe_depth: 5,
            alpha: 1.5,
            ef_min_factor: 0.8,
            ef_max_factor: 2.5,
        };
        let hits = searcher.search(&query[0], 10);
        assert_eq!(hits.len(), 10);
    }

    #[test]
    fn hits_are_sorted_ascending() {
        let g = build_test_graph(300, 16);
        let query = random_unit_vectors(1, 16, 77);
        let searcher = FrontierCutScaledEf {
            graph: &g,
            base_ef: 40,
            probe_depth: 5,
            alpha: 1.5,
            ef_min_factor: 0.8,
            ef_max_factor: 2.5,
        };
        let hits = searcher.search(&query[0], 8);
        for w in hits.windows(2) {
            assert!(w[0].dist <= w[1].dist);
        }
    }

    #[test]
    fn phi_is_between_zero_and_one() {
        let g = build_test_graph(300, 16);
        let query = random_unit_vectors(1, 16, 42);
        let stats = probe_phi(&g, &query[0], 5, 40);
        assert!((0.0..=1.0).contains(&stats.phi), "phi={}", stats.phi);
    }

    /// Acceptance test mirroring ruvector-entropy-ann's recall_acceptance_all_variants:
    /// both new variants must hit the same recall bar on the same clustered dataset.
    #[test]
    fn recall_acceptance_both_variants() {
        const N: usize = 500;
        const DIM: usize = 32;
        const K: usize = 10;
        const EF: usize = 40;
        const N_QUERIES: usize = 50;
        const RECALL_THRESHOLD: f32 = 0.80;

        let corpus = clustered_vectors(N, DIM, 10, 0.2, 42);
        let queries: Vec<Vec<f32>> = corpus[..N_QUERIES].to_vec();
        let graph = FlatGraph::build(corpus.clone(), GraphConfig { k_neighbours: 16 });

        let cut_thresh = CutThresholdBeam {
            graph: &graph,
            ef_search: EF,
            phi_stop: 0.15,
            window_steps: 3,
            min_steps: 3,
        };
        let cut_scaled = FrontierCutScaledEf {
            graph: &graph,
            base_ef: EF,
            probe_depth: 5,
            alpha: 1.5,
            ef_min_factor: 0.8,
            ef_max_factor: 2.5,
        };

        for (searcher, name) in [
            (&cut_thresh as &dyn Searcher, "CutThreshold"),
            (&cut_scaled as &dyn Searcher, "FrontierCutScaledEf"),
        ] {
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
        }
    }
}
