//! Erasure modes for an HNSW node.
//!
//! Two of the three are prior art from `ruvector-hnsw-repair` (ADR-259); the
//! third, [`ErasureMode::LocalRebuild`], is this experiment's candidate.
//!
//! | Mode | Stale edges removed | Stored vector | Neighbour lists |
//! |---|---|---|---|
//! | `Tombstone` | no | retained verbatim | untouched (dead edges remain) |
//! | `EagerRepair` | yes | retained verbatim | victim's own neighbours spliced in |
//! | `LocalRebuild` | yes | zeroized in place | re-derived from surviving geometry |
//!
//! The distinction that matters for erasure is the last column. `EagerRepair`
//! removes the dangling pointer but *replaces it with an edge chosen from the
//! deleted node's own neighbour list* — which is a function of the deleted
//! vector, i.e. it writes a fresh imprint of the victim into the graph while
//! removing the old one. `LocalRebuild` instead recomputes each affected
//! node's neighbour list by an ordinary ef-search over the surviving graph,
//! so the resulting edges are a function only of vectors that are still
//! present.

use ruvector_hnsw_repair::strategy::{DeletionStrategy, EagerRepair, TombstoneOnly};
use ruvector_hnsw_repair::HnswGraph;

/// Which erasure procedure to apply when a node is deleted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErasureMode {
    /// Baseline: mark deleted, change nothing else (what hnswlib, pgvector,
    /// Qdrant and friends do).
    Tombstone,
    /// Prior art (ADR-259): remove dangling edges, splice in the victim's own
    /// neighbours as replacements.
    EagerRepair,
    /// Candidate: remove dangling edges, re-derive each affected node's
    /// neighbour list from the surviving graph, and zeroize the stored vector.
    LocalRebuild {
        /// `ef` used for the per-referrer neighbour-list recomputation.
        ef_rebuild: usize,
    },
}

impl ErasureMode {
    pub fn label(&self) -> &'static str {
        match self {
            ErasureMode::Tombstone => "Tombstone",
            ErasureMode::EagerRepair => "EagerRepair",
            ErasureMode::LocalRebuild { .. } => "LocalRebuild",
        }
    }
}

/// What an erasure actually did. All fields are counted, not estimated.
#[derive(Clone, Debug, Default)]
pub struct ErasureStats {
    /// Live nodes that held an edge to the victim (summed over levels).
    pub referrers: usize,
    /// Neighbour lists fully recomputed (`LocalRebuild` only).
    pub rebuilt_lists: usize,
    /// Replacement edges added (`EagerRepair` only, as reported by the crate).
    pub repaired_edges: usize,
    /// True if the victim's stored coordinates were overwritten with zeros.
    pub vector_zeroized: bool,
    /// Bytes of the victim's vector still resident in the index after erasure.
    pub retained_vector_bytes: usize,
    pub elapsed_ns: u128,
}

/// Erase node `id` under `mode`, returning measured statistics.
pub fn erase(graph: &mut HnswGraph, id: usize, mode: ErasureMode) -> ErasureStats {
    // The referrer count is audit instrumentation, not part of any erasure
    // procedure, so it is measured *before* the clock starts. Including it
    // would inflate the cheap modes' timings and flatter the candidate.
    let referrers = count_referrers(graph, id);
    let mut stats = ErasureStats {
        referrers,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();

    match mode {
        ErasureMode::Tombstone => {
            TombstoneOnly.delete(graph, id);
        }
        ErasureMode::EagerRepair => {
            stats.repaired_edges = EagerRepair.delete(graph, id).repaired_edges;
        }
        ErasureMode::LocalRebuild { ef_rebuild } => {
            stats.rebuilt_lists = local_rebuild(graph, id, ef_rebuild);
            stats.vector_zeroized = true;
        }
    }

    stats.retained_vector_bytes = if stats.vector_zeroized {
        0
    } else {
        graph.config.dim * std::mem::size_of::<f32>()
    };
    stats.elapsed_ns = t0.elapsed().as_nanos();
    stats
}

/// Count live nodes holding an edge to `id`, summed over all levels.
pub fn count_referrers(graph: &HnswGraph, id: usize) -> usize {
    let target = id as u32;
    let mut n = 0usize;
    for level in 0..graph.layers.len() {
        for node in 0..graph.layers[level].len() {
            if node == id || graph.deleted[node] {
                continue;
            }
            if graph.layers[level][node].contains(&target) {
                n += 1;
            }
        }
    }
    n
}

/// `LocalRebuild`: drop the victim, then re-derive every affected neighbour
/// list from the surviving graph, then zeroize the victim's coordinates.
///
/// Returns the number of neighbour lists recomputed.
fn local_rebuild(graph: &mut HnswGraph, id: usize, ef_rebuild: usize) -> usize {
    if id >= graph.vectors.len() {
        return 0;
    }
    graph.deleted[id] = true;
    let target = id as u32;
    let dim = graph.config.dim;
    let mut rebuilt = 0usize;

    for level in 0..graph.layers.len() {
        let m_max = if level == 0 {
            graph.config.m0
        } else {
            graph.config.m
        };

        // Referrers at this level, captured before mutation.
        let referrers: Vec<usize> = (0..graph.layers[level].len())
            .filter(|&node| {
                node != id && !graph.deleted[node] && graph.layers[level][node].contains(&target)
            })
            .collect();

        for node in referrers {
            // 1. Drop the dangling edge.
            graph.layers[level][node].retain(|&nb| nb != target);
            let pruned = graph.layers[level][node].clone();

            // 2. Re-derive the list by an ordinary ef-search from this node
            //    over the *surviving* graph. `search_layer_ef` already skips
            //    deleted nodes, so the victim cannot re-enter the list, and no
            //    coordinate of the victim participates in the choice.
            let own = graph.vectors[node].clone();
            let candidates = graph.search_layer_ef(node as u32, &own, ef_rebuild.max(m_max), level);
            let mut fresh: Vec<u32> = candidates
                .iter()
                .map(|(_, cid)| *cid)
                .filter(|&cid| cid != node as u32 && !graph.deleted[cid as usize])
                .take(m_max)
                .collect();

            // 3. Never make connectivity worse than the plain pruned list: if
            //    the local search came back thin (possible on sparse upper
            //    layers), keep whichever list is larger.
            if fresh.len() < pruned.len() {
                fresh = pruned;
            }
            graph.layers[level][node] = fresh;
            rebuilt += 1;
        }
    }

    // 4. Zeroize the victim's stored coordinates. The node slot stays (ids are
    //    positional in this implementation) but carries no payload.
    for d in 0..dim {
        graph.vectors[id][d] = 0.0;
    }
    rebuilt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ClusteredSource, DatasetConfig};
    use ruvector_hnsw_repair::{HnswConfig, HnswGraph};

    fn build(n: usize) -> (HnswGraph, ClusteredSource) {
        let cfg = DatasetConfig {
            dim: 16,
            clusters: 4,
            sigma: 0.15,
            seed: 1234,
        };
        let mut src = ClusteredSource::new(cfg.clone());
        let hcfg = HnswConfig {
            dim: cfg.dim,
            m: 8,
            m0: 16,
            ef_construction: 40,
            ml: 1.0 / (8f64.ln()),
        };
        let mut g = HnswGraph::new(hcfg);
        for v in src.sample_many(n) {
            g.insert(v);
        }
        (g, src)
    }

    #[test]
    fn tombstone_retains_vector_and_edges() {
        let (mut g, _) = build(200);
        let before = g.vectors[7].clone();
        let stats = erase(&mut g, 7, ErasureMode::Tombstone);
        assert!(g.deleted[7]);
        assert_eq!(g.vectors[7], before, "tombstone must not touch the payload");
        assert_eq!(stats.retained_vector_bytes, 16 * 4);
        assert_eq!(count_referrers(&g, 7), stats.referrers);
    }

    #[test]
    fn local_rebuild_removes_all_dangling_edges() {
        let (mut g, _) = build(300);
        let stats = erase(&mut g, 11, ErasureMode::LocalRebuild { ef_rebuild: 32 });
        assert!(g.deleted[11]);
        assert_eq!(
            count_referrers(&g, 11),
            0,
            "no live node may still point at the erased id"
        );
        assert!(stats.rebuilt_lists <= stats.referrers);
    }

    #[test]
    fn local_rebuild_zeroizes_the_payload() {
        let (mut g, _) = build(200);
        assert!(g.vectors[5].iter().any(|&x| x != 0.0));
        let stats = erase(&mut g, 5, ErasureMode::LocalRebuild { ef_rebuild: 32 });
        assert!(g.vectors[5].iter().all(|&x| x == 0.0));
        assert!(stats.vector_zeroized);
        assert_eq!(stats.retained_vector_bytes, 0);
    }

    #[test]
    fn eager_repair_retains_the_payload() {
        // The erasure-relevant weakness of the prior-art strategy: the graph is
        // repaired but the deleted coordinates are still sitting in memory.
        let (mut g, _) = build(200);
        let before = g.vectors[9].clone();
        let stats = erase(&mut g, 9, ErasureMode::EagerRepair);
        assert_eq!(g.vectors[9], before);
        assert_eq!(stats.retained_vector_bytes, 16 * 4);
    }

    #[test]
    fn search_still_works_after_local_rebuild() {
        let (mut g, mut src) = build(400);
        for id in [3usize, 17, 42, 88] {
            erase(&mut g, id, ErasureMode::LocalRebuild { ef_rebuild: 32 });
        }
        let q = src.sample();
        let res = g.search(&q, 10, 40);
        assert_eq!(res.len(), 10, "index must stay navigable after rebuilds");
        for id in res {
            assert!(!g.deleted[id as usize]);
        }
    }
}
