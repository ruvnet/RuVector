//! Build a memory similarity graph: an undirected k-NN edge list over
//! memory embeddings, weighted by cosine similarity.

use crate::corpus::MemoryRecord;
use crate::search::cosine;
use ruvector_mincut::{VertexId, Weight};
use std::collections::HashSet;

/// Brute-force k-NN graph construction. O(n^2) in the number of records —
/// fine at this crate's benchmark scale (low thousands) and it is the
/// same ground-truth-quality neighbourhood the recall oracle uses, so the
/// partitioner sees the same notion of "nearby" the evaluator does.
/// Only positive-similarity neighbours become edges; each undirected pair
/// is emitted once.
pub fn build_knn_edges(records: &[MemoryRecord], k: usize) -> Vec<(VertexId, VertexId, Weight)> {
    let refs: Vec<(u64, &[f32])> = records
        .iter()
        .map(|r| (r.id, r.embedding.as_slice()))
        .collect();

    let mut seen: HashSet<(u64, u64)> = HashSet::new();
    let mut edges = Vec::new();

    for r in records {
        let mut scored: Vec<(u64, f32)> = refs
            .iter()
            .filter(|(id, _)| *id != r.id)
            .map(|(id, emb)| (*id, cosine(&r.embedding, emb)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));

        for (neighbor_id, sim) in scored.into_iter().take(k) {
            if sim <= 0.0 {
                continue;
            }
            let key = if r.id < neighbor_id {
                (r.id, neighbor_id)
            } else {
                (neighbor_id, r.id)
            };
            if seen.insert(key) {
                edges.push((key.0, key.1, sim as Weight));
            }
        }
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{generate, CorpusConfig};

    #[test]
    fn every_vertex_gets_at_least_one_edge() {
        let cfg = CorpusConfig {
            n: 300,
            ..CorpusConfig::default()
        };
        let corpus = generate(&cfg);
        let edges = build_knn_edges(&corpus.records, 10);
        let mut touched: HashSet<u64> = HashSet::new();
        for (u, v, _) in &edges {
            touched.insert(*u);
            touched.insert(*v);
        }
        assert_eq!(
            touched.len(),
            corpus.records.len(),
            "records with no positive-similarity neighbour: {}",
            corpus.records.len() - touched.len()
        );
    }

    #[test]
    fn edges_are_deduplicated_and_undirected() {
        let cfg = CorpusConfig {
            n: 150,
            ..CorpusConfig::default()
        };
        let corpus = generate(&cfg);
        let edges = build_knn_edges(&corpus.records, 8);
        let mut seen = HashSet::new();
        for (u, v, _) in &edges {
            assert!(u < v, "expected canonical (min,max) ordering");
            assert!(seen.insert((*u, *v)), "duplicate edge {u}-{v}");
        }
    }
}
