//! Variant 3 — Coherence-gated compactor.
//!
//! Extends the graph-merge approach with per-cluster adaptive thresholds:
//! high-coherence clusters (tight, uniform) are merged aggressively, while
//! low-coherence clusters (mixed, heterogeneous) are preserved.
//!
//! Coherence score for a candidate merge = mean(edge weights) - std_dev(edge weights)
//! across all edges incident to the two nodes being merged. High score = tight cluster.

use crate::graph::{CoherenceGraph, UnionFind};
use crate::kmeans::avg_intra_sim;
use crate::{
    centroid, recall_clustered, CompactionResult, Compactor, MemoryEntry, MemoryStore,
    WitnessRecord,
};

/// Coherence-gated memory compactor.
pub struct CoherenceGatedCompactor {
    pub graph_k: usize,
    /// Minimum coherence score required to approve a merge (0.0–1.0).
    pub coherence_floor: f32,
    /// Max cluster size after merge.
    pub max_cluster: usize,
}

impl Default for CoherenceGatedCompactor {
    fn default() -> Self {
        Self {
            graph_k: 15,
            coherence_floor: 0.50,
            max_cluster: 20,
        }
    }
}

impl Compactor for CoherenceGatedCompactor {
    fn name(&self) -> &'static str {
        "coherence-gated"
    }

    fn compact(
        &self,
        store: &mut MemoryStore,
        target_ratio: f64,
        queries: &[Vec<f32>],
        k: usize,
    ) -> CompactionResult {
        let n = store.len();
        let target_clusters = ((n as f64) * target_ratio).round().max(1.0) as usize;
        let before: Vec<MemoryEntry> = store.entries.clone();

        let graph = CoherenceGraph::build(&store.entries, self.graph_k);
        let clusters = self.coherence_merge(&graph, n, target_clusters);

        let mut new_entries: Vec<MemoryEntry> = Vec::with_capacity(clusters.len());
        let mut witness: Vec<WitnessRecord> = Vec::new();
        let mut new_id = store.next_id;

        for cluster in &clusters {
            let embs: Vec<&[f32]> = cluster
                .iter()
                .map(|&i| before[i].embedding.as_slice())
                .collect();
            let c = centroid(&embs);
            let intra_sim = avg_intra_sim(&before, cluster);
            let merged_ids: Vec<u64> = cluster.iter().map(|&i| before[i].id).collect();
            witness.push(WitnessRecord {
                centroid_id: new_id,
                merged_ids,
                intra_sim,
            });
            new_entries.push(MemoryEntry {
                id: new_id,
                embedding: c,
                age: cluster.iter().map(|&i| before[i].age).max().unwrap_or(0),
                metadata: format!("coherence-gated({})", cluster.len()),
            });
            new_id += 1;
        }
        store.entries = new_entries;
        store.next_id = new_id;

        let recall = recall_clustered(queries, &before, &store.entries, &witness, k);
        let compacted = store.len();
        CompactionResult {
            variant: self.name().to_string(),
            original_count: n,
            compacted_count: compacted,
            compaction_ratio: 1.0 - compacted as f64 / n as f64,
            recall_at_k: recall,
            duration_ms: 0,
            witness_records: witness,
        }
    }
}

impl CoherenceGatedCompactor {
    fn coherence_merge(
        &self,
        graph: &CoherenceGraph,
        n: usize,
        target_clusters: usize,
    ) -> Vec<Vec<usize>> {
        // Pre-compute per-node neighbourhood coherence scores (read-only).
        let node_coherence = node_coherence_scores(graph, n);

        let mut uf = UnionFind::new(n);
        let mut sizes: Vec<usize> = vec![1; n];
        let mut current_clusters = n;

        // Sort edges by weight descending (greedy best-first merging).
        let mut sorted_edges: Vec<(f32, usize, usize)> =
            graph.edges.iter().map(|e| (e.weight, e.a, e.b)).collect();
        sorted_edges.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        for (weight, a, b) in &sorted_edges {
            if current_clusters <= target_clusters {
                break;
            }
            let ra = uf.find(*a);
            let rb = uf.find(*b);
            if ra == rb {
                continue;
            }
            let new_size = sizes[ra] + sizes[rb];
            if new_size > self.max_cluster {
                continue;
            }
            // Coherence gate: average node coherence of the two endpoints.
            let coh = (node_coherence[*a] + node_coherence[*b]) / 2.0;
            // Also require the bridging edge to be above a derived threshold.
            let threshold = self.coherence_floor * 0.8; // slightly relaxed
            if coh < self.coherence_floor || *weight < threshold {
                continue;
            }
            uf.union(*a, *b);
            let new_root = uf.find(*a);
            sizes[new_root] = new_size;
            current_clusters -= 1;
        }

        uf.components(n)
    }
}

/// For each node, compute coherence = mean(neighbour_weights) - std_dev(neighbour_weights).
/// Purely read-only over the graph adjacency list — no UF needed.
fn node_coherence_scores(graph: &CoherenceGraph, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let weights: Vec<f32> = graph.adj[i].iter().map(|(_, w)| *w).collect();
            if weights.is_empty() {
                return 0.0_f32;
            }
            let mean = weights.iter().sum::<f32>() / weights.len() as f32;
            let var =
                weights.iter().map(|&w| (w - mean).powi(2)).sum::<f32>() / weights.len() as f32;
            (mean - var.sqrt()).max(0.0)
        })
        .collect()
}
