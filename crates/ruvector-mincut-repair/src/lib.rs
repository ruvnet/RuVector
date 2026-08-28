//! Local-min-cut-guided HNSW deletion repair.
//!
//! [`ruvector-hnsw-repair`](https://docs.rs/ruvector-hnsw-repair) ships three
//! deletion strategies with a fixed cost/recall trade-off: `TombstoneOnly`
//! (cheap, recall degrades), `BatchRepair` (amortised), `EagerRepair`
//! (expensive O(deg · live_count) full-graph scan per delete, best recall).
//! All three apply the same policy to every deleted node regardless of
//! whether that node's removal actually threatens local connectivity.
//!
//! This crate adds a fourth strategy, [`LocalCutGuidedRepair`], that asks a
//! cheaper question first: *does removing this node leave its neighbourhood
//! only weakly connected to the rest of the graph?* It answers that with
//! [`ruvector_mincut::LocalKCut`] — a deterministic local minimum-cut finder
//! (arXiv:2510.08297, "Deterministic and Exact Fully-dynamic Minimum Cut of
//! Superpolylogarithmic Size") already implemented in `ruvector-mincut` for
//! exactly this self-healing-network use case. Only nodes whose deletion
//! exposes a small local cut get [`ruvector_hnsw_repair::repair_one`]'s full
//! eager reconnection; everything else is tombstoned.
//!
//! The min-cut structure mirrors only the HNSW graph's **level-0** adjacency
//! (the layer present at every node and most responsible for base recall).
//! Higher HNSW levels are not modelled; see the crate's nightly research
//! report for why that scope is deliberate rather than an oversight.

use ruvector_hnsw_repair::strategy::repair_one;
use ruvector_hnsw_repair::{DeleteResult, DeletionStrategy, HnswGraph};
use ruvector_mincut::{DynamicGraph, LocalKCut, VertexId};
use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

/// Aggregate stats over a run of deletions, for benchmarking.
#[derive(Debug, Clone, Default)]
pub struct GuidedRepairStats {
    /// Number of deletes where a local cut was found and eager repair ran.
    pub fragile_count: usize,
    /// Number of deletes handled by cheap tombstone-only.
    pub safe_count: usize,
    /// Total nanoseconds spent on `find_cut` + shadow-graph bookkeeping.
    pub bookkeeping_ns: u128,
}

/// Deletion strategy that runs [`ruvector_hnsw_repair::repair_one`] only on
/// nodes whose removal exposes a local min-cut of size `<= k` in a shadow
/// graph mirroring HNSW level-0 edges; every other node is tombstoned.
pub struct LocalCutGuidedRepair {
    shadow: Arc<DynamicGraph>,
    finder: LocalKCut,
    stats: RefCell<GuidedRepairStats>,
}

impl LocalCutGuidedRepair {
    /// Build the shadow connectivity graph from `graph`'s current level-0
    /// edges and construct a [`LocalKCut`] finder bounded to cuts of size
    /// `<= k`. Call once before the deletion stream begins; the shadow graph
    /// is then kept in sync incrementally as `delete` is called.
    pub fn new(graph: &HnswGraph, k: usize) -> Self {
        let shadow = Arc::new(DynamicGraph::with_capacity(
            graph.vectors.len(),
            graph.vectors.len() * graph.config.m0,
        ));
        for id in 0..graph.vectors.len() {
            shadow.add_vertex(id as VertexId);
        }
        if let Some(level0) = graph.layers.first() {
            for (node, neighbours) in level0.iter().enumerate() {
                let u = node as VertexId;
                for &nb in neighbours {
                    let v = nb as VertexId;
                    if u < v {
                        let _ = shadow.insert_edge(u, v, 1.0);
                    }
                }
            }
        }
        let finder = LocalKCut::new(shadow.clone(), k.max(1));
        Self {
            shadow,
            finder,
            stats: RefCell::new(GuidedRepairStats::default()),
        }
    }

    /// Stats accumulated since construction.
    pub fn stats(&self) -> GuidedRepairStats {
        self.stats.borrow().clone()
    }
}

impl LocalCutGuidedRepair {
    /// Same as [`DeletionStrategy::delete`] but also returns whether `id`
    /// was judged fragile (and therefore eagerly repaired) rather than
    /// tombstoned. Useful for tests and diagnostics that need to know which
    /// specific nodes got which treatment.
    pub fn delete_with_diagnostics(
        &self,
        graph: &mut HnswGraph,
        id: usize,
    ) -> (DeleteResult, bool) {
        if id >= graph.deleted.len() {
            return (DeleteResult::default(), false);
        }

        let v = id as VertexId;
        let t0 = Instant::now();
        let fragile = self.shadow.has_vertex(v) && self.finder.find_cut(v).is_some();
        let bookkeeping_ns = t0.elapsed().as_nanos();

        graph.deleted[id] = true;

        let repaired_edges = if fragile { repair_one(graph, id) } else { 0 };

        // Keep the shadow graph in sync for future find_cut queries: drop
        // the deleted vertex and every edge incident to it.
        let _ = self.shadow.remove_vertex(v);

        {
            let mut s = self.stats.borrow_mut();
            if fragile {
                s.fragile_count += 1;
            } else {
                s.safe_count += 1;
            }
            s.bookkeeping_ns += bookkeeping_ns;
        }

        (DeleteResult { repaired_edges }, fragile)
    }
}

impl DeletionStrategy for LocalCutGuidedRepair {
    fn delete(&self, graph: &mut HnswGraph, id: usize) -> DeleteResult {
        self.delete_with_diagnostics(graph, id).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_hnsw_repair::graph::{HnswConfig, HnswGraph};

    fn build_small(n: usize, dim: usize) -> HnswGraph {
        let config = HnswConfig {
            dim,
            m: 4,
            m0: 8,
            ef_construction: 20,
            ml: 1.0 / (4f64.ln()),
        };
        let mut g = HnswGraph::new(config);
        let mut rng = 42u64;
        for _ in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (rng >> 33) as f32 / (u32::MAX as f32)
                })
                .collect();
            g.insert(v);
        }
        g
    }

    #[test]
    fn tombstones_every_deleted_node() {
        let mut g = build_small(200, 8);
        let strat = LocalCutGuidedRepair::new(&g, 2);
        for id in (0..200).step_by(5) {
            strat.delete(&mut g, id);
        }
        for id in (0..200).step_by(5) {
            assert!(g.deleted[id]);
        }
    }

    #[test]
    fn fragile_deletions_leave_no_stale_edges() {
        // Only nodes the strategy actually judged fragile (and therefore
        // eagerly repaired) are guaranteed to leave no stale references —
        // a tombstoned "safe" node is expected to keep them, exactly like
        // `TombstoneOnly`. Track which ids were fragile via the diagnostic
        // API and check the invariant only for those.
        let mut g = build_small(200, 8);
        let strat = LocalCutGuidedRepair::new(&g, 2);
        let mut fragile_ids = Vec::new();
        for id in (0..200).step_by(3) {
            let (_, fragile) = strat.delete_with_diagnostics(&mut g, id);
            if fragile {
                fragile_ids.push(id as u32);
            }
        }
        assert!(
            !fragile_ids.is_empty(),
            "expected at least one fragile deletion in this fixture"
        );
        for node in 0..g.vectors.len() {
            if g.deleted[node] {
                continue;
            }
            for &nb in &g.layers[0][node] {
                assert!(
                    !fragile_ids.contains(&nb),
                    "live node {node} still references fragile-repaired node {nb}"
                );
            }
        }
    }

    #[test]
    fn stats_partition_every_delete() {
        let mut g = build_small(150, 8);
        let strat = LocalCutGuidedRepair::new(&g, 2);
        let n_delete = 30;
        for id in (0..150).step_by(5).take(n_delete) {
            strat.delete(&mut g, id);
        }
        let s = strat.stats();
        assert_eq!(s.fragile_count + s.safe_count, n_delete);
    }

    /// Diagnostic, not a correctness check: measures single-call
    /// `find_cut` latency on a 5,000-node graph. `#[ignore]`d because it is
    /// slow and its purpose is manual investigation, not CI.
    ///
    /// This fixture uses `build_small`'s sparse HNSW config (m=4, m0=8),
    /// which understates the real cost: at the *benchmark*'s construction
    /// density (m=16, m0=32, matching `ruvector-hnsw-repair`'s own
    /// benchmark), a single `find_cut` call on the same 5,000-node scale
    /// measured **163.5 seconds** (see the 2026-08-28-mincut-guided-hnsw-repair
    /// nightly research report for the raw run). The four calls below
    /// (hundreds of ms each) already show the qualitative problem cheaply;
    /// reproducing the 163.5s figure requires the denser config and is not
    /// re-run here to keep this diagnostic itself fast to invoke.
    #[test]
    #[ignore]
    fn probe_find_cut_cost() {
        let g = build_small(5000, 64);
        let t_build = Instant::now();
        let strat = LocalCutGuidedRepair::new(&g, 2);
        eprintln!("construct: {:?}", t_build.elapsed());
        for i in [0usize, 1, 2, 3, 4] {
            let t0 = Instant::now();
            let r = strat.finder.find_cut(i as VertexId);
            eprintln!("find_cut({i}) = {:?} in {:?}", r.is_some(), t0.elapsed());
        }
    }

    #[test]
    fn shadow_graph_shrinks_as_nodes_are_deleted() {
        let mut g = build_small(100, 8);
        let strat = LocalCutGuidedRepair::new(&g, 2);
        let before = strat.shadow.num_vertices();
        for id in 0..10 {
            strat.delete(&mut g, id);
        }
        assert_eq!(strat.shadow.num_vertices(), before - 10);
    }
}
