//! Three retrieval variants for temporal coherence agent memory.
//!
//! All implement `VectorSearch` which returns a ranked `Vec<SearchResult>`.

use crate::{cosine_sim, CoherenceGraph, DecayConfig, MemoryId, MemoryStore};

/// A scored retrieval result.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub id: MemoryId,
    pub score: f32,
}

/// Unified search interface for all three variants.
pub trait VectorSearch {
    fn search(&self, query: &[f32], k: usize, store: &MemoryStore) -> Vec<SearchResult>;
}

// ── Variant 1: Pure cosine similarity ────────────────────────────────────────

/// Baseline: rank by cosine similarity only.
pub struct FlatSearch;

impl VectorSearch for FlatSearch {
    fn search(&self, query: &[f32], k: usize, store: &MemoryStore) -> Vec<SearchResult> {
        let mut scored: Vec<SearchResult> = store
            .records()
            .map(|r| SearchResult {
                id: r.id,
                score: cosine_sim(query, &r.vec),
            })
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored.truncate(k);
        scored
    }
}

// ── Variant 2: Cosine × temporal decay ───────────────────────────────────────

/// Temporal: rank by cosine × exponential time-decay.
pub struct TemporalSearch {
    pub decay: DecayConfig,
}

impl VectorSearch for TemporalSearch {
    fn search(&self, query: &[f32], k: usize, store: &MemoryStore) -> Vec<SearchResult> {
        let mut scored: Vec<SearchResult> = store
            .records()
            .map(|r| {
                let sim = cosine_sim(query, &r.vec);
                let d = self.decay.factor(r.metadata.timestamp);
                SearchResult {
                    id: r.id,
                    score: sim * d,
                }
            })
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored.truncate(k);
        scored
    }
}

// ── Variant 3: Cosine × decay × coherence gate ───────────────────────────────

/// Coherence-temporal: rank by cosine × decay × graph-coherence gate.
///
/// The coherence gate is the normalised in-degree of the memory node in the
/// coherence graph, adding a soft "community vote" to the ranking. Memories
/// that are highly similar to many other recent memories rank higher.
pub struct CoherenceSearch {
    pub decay: DecayConfig,
    pub graph: CoherenceGraph,
    /// Weight for coherence contribution: score = sim * ((1-w)*decay + w*gate)
    pub coherence_weight: f32,
}

impl CoherenceSearch {
    pub fn new(decay: DecayConfig, graph: CoherenceGraph, coherence_weight: f32) -> Self {
        Self {
            decay,
            graph,
            coherence_weight: coherence_weight.clamp(0.0, 1.0),
        }
    }
}

impl VectorSearch for CoherenceSearch {
    fn search(&self, query: &[f32], k: usize, store: &MemoryStore) -> Vec<SearchResult> {
        let w = self.coherence_weight;
        let mut scored: Vec<SearchResult> = store
            .records()
            .map(|r| {
                let sim = cosine_sim(query, &r.vec);
                let decay_f = self.decay.factor(r.metadata.timestamp);
                let gate_f = self.graph.gate(r.id);
                // Blend decay and coherence gate with weight w.
                let temporal_coherence = (1.0 - w) * decay_f + w * gate_f;
                SearchResult {
                    id: r.id,
                    score: sim * temporal_coherence,
                }
            })
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DecayConfig, MemoryMetadata, MemoryStore};

    fn simple_store() -> MemoryStore {
        let mut s = MemoryStore::new(4);
        // memory 0: very similar to query, but old (ts=0)
        s.insert(
            vec![1.0, 0.0, 0.0, 0.0],
            MemoryMetadata {
                timestamp: 0,
                source: "a".into(),
                tags: vec![],
            },
        );
        // memory 1: slightly less similar, but recent (ts=900)
        s.insert(
            vec![0.9, 0.1, 0.1, 0.0],
            MemoryMetadata {
                timestamp: 900,
                source: "b".into(),
                tags: vec![],
            },
        );
        // memory 2: very different (ts=1000, recent but irrelevant)
        s.insert(
            vec![0.0, 0.0, 0.0, 1.0],
            MemoryMetadata {
                timestamp: 1000,
                source: "c".into(),
                tags: vec![],
            },
        );
        s
    }

    #[test]
    fn flat_search_returns_k() {
        let store = simple_store();
        let results = FlatSearch.search(&[1.0, 0.0, 0.0, 0.0], 2, &store);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // most similar
    }

    #[test]
    fn temporal_search_promotes_recent() {
        let store = simple_store();
        // With a short half-life, memory 0 (very old) should be penalised.
        let decay = DecayConfig::exponential(1000, 100); // very fast decay
        let ts = TemporalSearch { decay };
        let results = ts.search(&[1.0, 0.0, 0.0, 0.0], 3, &store);
        assert_eq!(results.len(), 3);
        // Memory 1 (ts=900) should beat memory 0 (ts=0) despite slightly lower cosine.
        let pos1 = results.iter().position(|r| r.id == 1).unwrap();
        let pos0 = results.iter().position(|r| r.id == 0).unwrap();
        assert!(
            pos1 < pos0,
            "recent memory 1 should rank above old memory 0 with fast decay"
        );
    }

    #[test]
    fn coherence_search_returns_k() {
        let store = simple_store();
        let decay = DecayConfig::exponential(1000, 500);
        let graph = CoherenceGraph::build(&store, 0.5);
        let cs = CoherenceSearch::new(decay, graph, 0.3);
        let results = cs.search(&[1.0, 0.0, 0.0, 0.0], 2, &store);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn scores_are_non_negative() {
        let store = simple_store();
        let decay = DecayConfig::exponential(1000, 300);
        let graph = CoherenceGraph::build(&store, 0.7);
        let cs = CoherenceSearch::new(decay, graph, 0.4);
        let results = cs.search(&[1.0, 0.0, 0.0, 0.0], 3, &store);
        for r in &results {
            assert!(r.score >= -0.01, "score={}", r.score);
        }
    }

    #[test]
    fn flat_search_ordered_by_score() {
        let store = simple_store();
        let results = FlatSearch.search(&[1.0, 0.0, 0.0, 0.0], 3, &store);
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score, "results not sorted");
        }
    }
}
