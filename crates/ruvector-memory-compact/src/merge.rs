//! Variant 2 — Graph-merge compactor.
//!
//! Builds a k-NN cosine similarity graph over the memory store, then
//! merges all connected components formed by edges above a similarity
//! threshold. The threshold is automatically chosen to achieve the
//! requested `target_ratio` (fraction of vectors to keep).

use crate::graph::{CoherenceGraph, UnionFind};
use crate::kmeans::avg_intra_sim;
use crate::{
    centroid, recall_clustered, CompactionResult, Compactor, MemoryEntry, MemoryStore,
    WitnessRecord,
};

/// Threshold-based graph-merge compactor.
///
/// Edges with cosine similarity ≥ `merge_threshold` are contracted;
/// each resulting connected component is replaced by its centroid.
pub struct GraphMergeCompactor {
    /// How many neighbours to compute per node when building the graph.
    pub graph_k: usize,
    /// Cosine-similarity threshold above which two nodes are merged.
    /// If `None`, the threshold is chosen automatically to hit `target_ratio`.
    pub merge_threshold: Option<f32>,
}

impl Default for GraphMergeCompactor {
    fn default() -> Self {
        Self {
            graph_k: 15,
            merge_threshold: None,
        }
    }
}

impl Compactor for GraphMergeCompactor {
    fn name(&self) -> &'static str {
        "graph-merge"
    }

    fn compact(
        &self,
        store: &mut MemoryStore,
        target_ratio: f64,
        queries: &[Vec<f32>],
        k: usize,
    ) -> CompactionResult {
        let n = store.len();
        let before: Vec<MemoryEntry> = store.entries.clone();

        let graph = CoherenceGraph::build(&store.entries, self.graph_k);

        // Collect all edge weights, sorted descending, to binary-search the
        // threshold that gives approximately target_ratio clusters.
        let threshold = self
            .merge_threshold
            .unwrap_or_else(|| pick_threshold(&graph, n, target_ratio));

        let clusters = merge_by_threshold(&graph, n, threshold);

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
                metadata: format!("graph-merge({})", cluster.len()),
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

/// Find connected components after removing all edges below `threshold`.
pub fn merge_by_threshold(graph: &CoherenceGraph, n: usize, threshold: f32) -> Vec<Vec<usize>> {
    let mut uf = UnionFind::new(n);
    for edge in &graph.edges {
        if edge.weight >= threshold {
            uf.union(edge.a, edge.b);
        }
    }
    uf.components(n)
}

/// Binary search for a threshold that produces approximately `target_ratio * n` clusters.
fn pick_threshold(graph: &CoherenceGraph, n: usize, target_ratio: f64) -> f32 {
    let target_clusters = ((n as f64) * target_ratio).round().max(1.0) as usize;

    let mut weights: Vec<f32> = graph.edges.iter().map(|e| e.weight).collect();
    weights.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    weights.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    if weights.is_empty() {
        return 1.1; // no edges → no merging
    }

    // Binary search over edge weight thresholds.
    let mut lo = 0usize;
    let mut hi = weights.len();
    let mut best_thresh = weights[weights.len() / 2];

    while lo < hi {
        let mid = (lo + hi) / 2;
        let thresh = weights[mid];
        let clusters = merge_by_threshold(graph, n, thresh).len();
        if clusters <= target_clusters {
            best_thresh = thresh;
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    best_thresh
}
