//! The three adaptive search variants built on top of [`ProximityGraph`].
//!
//! All implement [`AnnSearch`], which provides a uniform interface for
//! comparison in the benchmark binary.

use crate::{
    difficulty::{distance_ratio_score, predict_ef},
    AnnResult, ProximityGraph, SearchStats,
};

// ── Trait ──────────────────────────────────────────────────────────────────────

/// Uniform interface for all search variants.
pub trait AnnSearch {
    fn name(&self) -> &'static str;

    /// Search for k approximate nearest neighbours of `query`.
    ///
    /// Returns results sorted by ascending distance, plus diagnostic stats.
    fn search(
        &self,
        graph: &ProximityGraph,
        query: &[f32],
        k: usize,
    ) -> (Vec<AnnResult>, SearchStats);
}

// ── Variant 1: FixedEfSearch (baseline) ──────────────────────────────────────

/// Fixed beam width — the conventional HNSW approach.
///
/// All queries use the same `ef`, regardless of whether the query is easy
/// or hard.  This is the **baseline**.  It achieves predictable latency at
/// the cost of wasted computation on easy queries.
pub struct FixedEfSearch {
    pub ef: usize,
}

impl AnnSearch for FixedEfSearch {
    fn name(&self) -> &'static str {
        "FixedEf"
    }

    fn search(
        &self,
        graph: &ProximityGraph,
        query: &[f32],
        k: usize,
    ) -> (Vec<AnnResult>, SearchStats) {
        let (results, ops) = graph.beam_search(query, k, self.ef);
        let difficulty = distance_ratio_score(&results);
        let stats = SearchStats {
            variant: self.name(),
            ef_used: self.ef,
            difficulty,
            distance_ops: ops,
            escalated: false,
        };
        (results, stats)
    }
}

// ── Variant 2: TwoStageSearch ─────────────────────────────────────────────────

/// Two-stage beam search: fast pass first, full pass only if query is hard.
///
/// **Stage 1**: beam search with `ef_fast` (default 16).  Compute the
/// distance-ratio difficulty score from initial candidates.
///
/// **Stage 2**: if difficulty > `threshold`, re-run with `ef_full` (64).
/// Otherwise return the stage-1 result.
///
/// This amortises cost: easy queries (difficulty ≤ threshold) pay ef_fast
/// ops; hard queries pay ef_fast + ef_full ops.  When most queries are easy,
/// overall throughput improves relative to always using ef_full.
pub struct TwoStageSearch {
    pub ef_fast: usize,
    pub ef_full: usize,
    /// Difficulty score above which the full pass fires. [0, 1]
    pub threshold: f32,
}

impl AnnSearch for TwoStageSearch {
    fn name(&self) -> &'static str {
        "TwoStage"
    }

    fn search(
        &self,
        graph: &ProximityGraph,
        query: &[f32],
        k: usize,
    ) -> (Vec<AnnResult>, SearchStats) {
        let (fast_results, ops1) = graph.beam_search(query, k, self.ef_fast);
        let difficulty = distance_ratio_score(&fast_results);

        if difficulty > self.threshold {
            let (full_results, ops2) = graph.beam_search(query, k, self.ef_full);
            let stats = SearchStats {
                variant: self.name(),
                ef_used: self.ef_full,
                difficulty,
                distance_ops: ops1 + ops2,
                escalated: true,
            };
            (full_results, stats)
        } else {
            let stats = SearchStats {
                variant: self.name(),
                ef_used: self.ef_fast,
                difficulty,
                distance_ops: ops1,
                escalated: false,
            };
            (fast_results, stats)
        }
    }
}

// ── Variant 3: AdaptiveEfSearch ───────────────────────────────────────────────

/// Continuous adaptive beam width: predicted ef from difficulty score.
///
/// **Stage 1**: beam search with `ef_min` (fast probe).
///
/// **Prediction**: ef_predicted = round(ef_min + (ef_max − ef_min) × difficulty).
///
/// If predicted ef ≤ ef_min (query deemed easy), return stage-1 result.
/// Otherwise re-run with the predicted ef.  This gives a finer-grained
/// tradeoff than binary TwoStage: medium-difficulty queries use medium ef,
/// avoiding the full ef_max cost.
///
/// Unlike TwoStage's binary escalation, AdaptiveEf samples the entire
/// ef spectrum: a difficulty=0.5 query uses ef≈40 rather than jumping to 64.
pub struct AdaptiveEfSearch {
    pub ef_min: usize,
    pub ef_max: usize,
    /// Minimum difficulty to trigger a second pass. [0, 1]
    pub threshold: f32,
}

impl AnnSearch for AdaptiveEfSearch {
    fn name(&self) -> &'static str {
        "AdaptiveEf"
    }

    fn search(
        &self,
        graph: &ProximityGraph,
        query: &[f32],
        k: usize,
    ) -> (Vec<AnnResult>, SearchStats) {
        let (fast_results, ops1) = graph.beam_search(query, k, self.ef_min);
        let difficulty = distance_ratio_score(&fast_results);
        let predicted_ef = predict_ef(difficulty, self.ef_min, self.ef_max);

        if difficulty <= self.threshold || predicted_ef <= self.ef_min {
            let stats = SearchStats {
                variant: self.name(),
                ef_used: self.ef_min,
                difficulty,
                distance_ops: ops1,
                escalated: false,
            };
            return (fast_results, stats);
        }

        let (results, ops2) = graph.beam_search(query, k, predicted_ef);
        let stats = SearchStats {
            variant: self.name(),
            ef_used: predicted_ef,
            difficulty,
            distance_ops: ops1 + ops2,
            escalated: true,
        };
        (results, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{brute_knn, generate_vectors, recall_at_k, ProximityGraph};

    fn build_graph(n: usize, dim: usize, seed: u64) -> (ProximityGraph, Vec<Vec<f32>>) {
        let corpus = generate_vectors(n, dim, seed);
        let mut g = ProximityGraph::new(8);
        for v in &corpus {
            g.insert(v.clone(), 32);
        }
        (g, corpus)
    }

    #[test]
    fn twostage_escalates_on_hard_query() {
        // Build a cluster around origin so any query near origin triggers escalation.
        let mut g = ProximityGraph::new(8);
        // Insert 200 vectors, all very close to origin → high difficulty
        let corpus: Vec<Vec<f32>> = (0..200u64)
            .map(|i| {
                vec![
                    (i as f32) * 0.001,
                    ((i * 7) as f32) * 0.001,
                    ((i * 3) as f32) * 0.001,
                    ((i * 11) as f32) * 0.001,
                ]
            })
            .collect();
        for v in &corpus {
            g.insert(v.clone(), 32);
        }
        let query = vec![0.05f32, 0.05, 0.05, 0.05];
        let ts = TwoStageSearch {
            ef_fast: 16,
            ef_full: 64,
            threshold: 0.8,
        };
        let (_, stats) = ts.search(&g, &query, 10);
        // Near-origin query in a tight cluster: difficulty should be non-trivially high.
        // In 4D, 200 vectors spaced 0.001 apart produce a ratio ≥ 0.4.
        assert!(
            stats.difficulty > 0.3,
            "expected moderate difficulty from cluster, got {:.3}",
            stats.difficulty
        );
    }

    #[test]
    fn adaptive_ef_stays_low_on_easy_query() {
        // Well-separated corpus: one isolated vector, rest far away.
        let mut g = ProximityGraph::new(8);
        g.insert(vec![0.0, 0.0, 0.0, 0.0], 4);
        for i in 1u32..200 {
            g.insert(vec![100.0 * i as f32, 0.0, 0.0, 0.0], 8);
        }
        let query = vec![0.01f32, 0.0, 0.0, 0.0];
        let ae = AdaptiveEfSearch {
            ef_min: 8,
            ef_max: 64,
            threshold: 0.3,
        };
        let (_, stats) = ae.search(&g, &query, 1);
        // Query near origin → very easy → should not escalate
        assert!(!stats.escalated, "easy query should not escalate");
    }

    #[test]
    fn all_variants_recall_parity() {
        // On a moderate corpus all three should achieve similar recall.
        let (g, corpus) = build_graph(800, 32, 55);
        let queries = generate_vectors(30, 32, 66);
        let k = 10;

        let variants: Vec<Box<dyn AnnSearch>> = vec![
            Box::new(FixedEfSearch { ef: 32 }),
            Box::new(TwoStageSearch {
                ef_fast: 16,
                ef_full: 64,
                threshold: 0.65,
            }),
            Box::new(AdaptiveEfSearch {
                ef_min: 16,
                ef_max: 64,
                threshold: 0.35,
            }),
        ];

        for variant in &variants {
            let mut total = 0.0f32;
            for q in &queries {
                let (results, _) = variant.search(&g, q, k);
                let gt = brute_knn(q, &corpus, k);
                total += recall_at_k(&results, &gt);
            }
            let mean = total / queries.len() as f32;
            assert!(
                mean >= 0.70,
                "{} recall {:.3} below 0.70 floor",
                variant.name(),
                mean
            );
        }
    }
}
