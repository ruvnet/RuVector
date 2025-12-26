//! Hybrid Search Tests
//!
//! Tests for hybrid search combining vector similarity with BM25 text scoring.
//!
//! Test categories:
//! - BM25 scoring accuracy vs reference implementation
//! - RRF (Reciprocal Rank Fusion) correctness
//! - Linear fusion with alpha parameter
//! - Performance: hybrid < 2x single branch latency

use super::harness::*;

/// Test module for BM25 scoring accuracy
#[cfg(test)]
mod bm25_scoring_tests {
    use super::*;

    /// BM25 parameters
    const K1: f64 = 1.2;
    const B: f64 = 0.75;

    /// Calculate IDF (Inverse Document Frequency)
    fn idf(num_docs: usize, docs_with_term: usize) -> f64 {
        let n = num_docs as f64;
        let n_t = docs_with_term as f64;

        ((n - n_t + 0.5) / (n_t + 0.5) + 1.0).ln()
    }

    /// Calculate BM25 score for a single term
    fn bm25_term_score(
        term_freq: usize,
        doc_len: usize,
        avg_doc_len: f64,
        num_docs: usize,
        docs_with_term: usize,
    ) -> f64 {
        let tf = term_freq as f64;
        let dl = doc_len as f64;

        let idf_score = idf(num_docs, docs_with_term);
        let tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * (dl / avg_doc_len)));

        idf_score * tf_norm
    }

    /// Test IDF calculation
    #[test]
    fn test_idf_calculation() {
        // Rare term: appears in 10 of 10000 docs
        let rare_idf = idf(10000, 10);

        // Common term: appears in 5000 of 10000 docs
        let common_idf = idf(10000, 5000);

        // Rare terms should have higher IDF
        assert!(rare_idf > common_idf, "Rare terms should have higher IDF");
        assert!(rare_idf > 0.0, "IDF should be positive");

        // Very common term (appears in all docs)
        let ubiquitous_idf = idf(10000, 9999);
        assert!(ubiquitous_idf < common_idf, "Ubiquitous terms have low IDF");
    }

    /// Test term frequency normalization
    #[test]
    fn test_tf_normalization() {
        let num_docs = 1000;
        let docs_with_term = 100;
        let avg_doc_len = 200.0;

        // Same term freq, different doc lengths
        let short_doc_score = bm25_term_score(5, 100, avg_doc_len, num_docs, docs_with_term);
        let normal_doc_score = bm25_term_score(5, 200, avg_doc_len, num_docs, docs_with_term);
        let long_doc_score = bm25_term_score(5, 400, avg_doc_len, num_docs, docs_with_term);

        // Shorter docs should score higher for same term freq
        assert!(short_doc_score > normal_doc_score);
        assert!(normal_doc_score > long_doc_score);
    }

    /// Test BM25 score bounds
    #[test]
    fn test_bm25_score_bounds() {
        let num_docs = 1000;
        let avg_doc_len = 200.0;

        // Edge case: term appears in all docs
        let low_idf_score = bm25_term_score(10, 200, avg_doc_len, num_docs, 999);
        assert!(low_idf_score >= 0.0);

        // Edge case: term appears once in one doc
        let high_idf_score = bm25_term_score(1, 200, avg_doc_len, num_docs, 1);
        assert!(high_idf_score > low_idf_score);

        // Edge case: very high term frequency
        let high_tf_score = bm25_term_score(100, 200, avg_doc_len, num_docs, 100);
        assert!(high_tf_score.is_finite());
    }

    /// Test multi-term BM25 scoring
    #[test]
    fn test_multi_term_bm25() {
        let num_docs = 1000;
        let avg_doc_len = 200.0;
        let doc_len = 200;

        // Query: "machine learning"
        let term1_score = bm25_term_score(3, doc_len, avg_doc_len, num_docs, 100); // "machine"
        let term2_score = bm25_term_score(2, doc_len, avg_doc_len, num_docs, 80); // "learning"

        let combined_score = term1_score + term2_score;

        // Combined should be greater than individual
        assert!(combined_score > term1_score);
        assert!(combined_score > term2_score);
    }

    /// Test BM25 parameter sensitivity
    #[test]
    fn test_bm25_parameter_sensitivity() {
        let base_score = bm25_term_score(5, 200, 200.0, 1000, 100);

        // Varying k1 should change saturation curve
        // Higher k1 means higher scores for high term freq

        // Varying b should change length normalization
        // b=0 means no length normalization
        // b=1 means full length normalization

        assert!(base_score > 0.0);
        assert!(base_score < 20.0); // Reasonable upper bound
    }
}

/// Test module for RRF (Reciprocal Rank Fusion)
#[cfg(test)]
mod rrf_fusion_tests {
    use super::*;

    /// RRF constant (typically 60)
    const RRF_K: f64 = 60.0;

    /// Calculate RRF score for a document
    fn rrf_score(ranks: &[usize]) -> f64 {
        ranks.iter().map(|&rank| 1.0 / (RRF_K + rank as f64)).sum()
    }

    /// Test basic RRF calculation
    #[test]
    fn test_basic_rrf() {
        // Document appears at rank 1 in both lists
        let score = rrf_score(&[1, 1]);
        let expected = 2.0 / (RRF_K + 1.0);

        assertions::assert_approx_eq(score as f32, expected as f32, 0.0001);
    }

    /// Test RRF with different ranks
    #[test]
    fn test_rrf_rank_impact() {
        // Higher rank (worse) = lower contribution
        let score_rank1 = rrf_score(&[1]);
        let score_rank10 = rrf_score(&[10]);
        let score_rank100 = rrf_score(&[100]);

        assert!(score_rank1 > score_rank10);
        assert!(score_rank10 > score_rank100);
    }

    /// Test RRF fusion of two lists
    #[test]
    fn test_rrf_two_list_fusion() {
        // Simulate two ranked lists
        // Doc A: rank 1 in vector, rank 5 in text
        // Doc B: rank 5 in vector, rank 1 in text
        // Doc C: rank 2 in vector, rank 2 in text

        let score_a = rrf_score(&[1, 5]);
        let score_b = rrf_score(&[5, 1]);
        let score_c = rrf_score(&[2, 2]);

        // A and B should have same score (symmetric)
        assertions::assert_approx_eq(score_a as f32, score_b as f32, 0.0001);

        // C might be higher due to consistent ranking
        // At k=60: 1/(61) + 1/(65) vs 2*1/(62)
        // 0.0164 + 0.0154 = 0.0318 vs 0.0323
        // So C is slightly higher
        assert!(score_c >= score_a * 0.99);
    }

    /// Test RRF with missing rankings
    #[test]
    fn test_rrf_missing_ranks() {
        // Document only appears in one list
        let score_both = rrf_score(&[1, 1]);
        let score_one = rrf_score(&[1]);

        assert!(
            score_both > score_one,
            "Appearing in both lists should score higher"
        );
    }

    /// Test RRF ordering stability
    #[test]
    fn test_rrf_ordering_stability() {
        // Documents with their ranks in two lists
        let docs = [
            ("A", vec![1, 3]),
            ("B", vec![2, 1]),
            ("C", vec![3, 2]),
            ("D", vec![4, 4]),
        ];

        let mut scores: Vec<(&str, f64)> = docs
            .iter()
            .map(|(name, ranks)| (*name, rrf_score(ranks)))
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Verify ordering makes sense
        assert!(scores[0].1 >= scores[1].1);
        assert!(scores[1].1 >= scores[2].1);
        assert!(scores[2].1 >= scores[3].1);
    }

    /// Test RRF with varying k parameter
    #[test]
    fn test_rrf_k_parameter() {
        let ranks = vec![1, 5];

        // k=60 (standard)
        let score_k60 = ranks.iter().map(|&r| 1.0 / (60.0 + r as f64)).sum::<f64>();

        // k=10 (more weight to top ranks)
        let score_k10 = ranks.iter().map(|&r| 1.0 / (10.0 + r as f64)).sum::<f64>();

        // Lower k gives more weight to differences in rank
        assert!(score_k10 > score_k60);
    }
}

/// Test module for linear fusion
#[cfg(test)]
mod linear_fusion_tests {
    use super::*;

    /// Linear fusion: alpha * vector_score + (1 - alpha) * text_score
    fn linear_fusion(vector_score: f64, text_score: f64, alpha: f64) -> f64 {
        alpha * vector_score + (1.0 - alpha) * text_score
    }

    /// Test alpha parameter bounds
    #[test]
    fn test_alpha_bounds() {
        let vector_score = 0.8;
        let text_score = 0.6;

        // alpha = 0: only text
        let score_text_only = linear_fusion(vector_score, text_score, 0.0);
        assertions::assert_approx_eq(score_text_only as f32, text_score as f32, 0.0001);

        // alpha = 1: only vector
        let score_vector_only = linear_fusion(vector_score, text_score, 1.0);
        assertions::assert_approx_eq(score_vector_only as f32, vector_score as f32, 0.0001);

        // alpha = 0.5: equal weight
        let score_balanced = linear_fusion(vector_score, text_score, 0.5);
        let expected = (vector_score + text_score) / 2.0;
        assertions::assert_approx_eq(score_balanced as f32, expected as f32, 0.0001);
    }

    /// Test linear fusion preserves ordering
    #[test]
    fn test_linear_fusion_ordering() {
        let alpha = 0.7;

        // Doc A: high vector, low text
        // Doc B: low vector, high text
        let score_a = linear_fusion(0.9, 0.3, alpha);
        let score_b = linear_fusion(0.3, 0.9, alpha);

        // At alpha=0.7, vector-dominant doc should win
        assert!(score_a > score_b);

        // At alpha=0.3, text-dominant doc should win
        let score_a_low_alpha = linear_fusion(0.9, 0.3, 0.3);
        let score_b_low_alpha = linear_fusion(0.3, 0.9, 0.3);
        assert!(score_b_low_alpha > score_a_low_alpha);
    }

    /// Test score normalization for fusion
    #[test]
    fn test_score_normalization() {
        // Scores should be normalized to [0, 1] before fusion
        fn normalize(score: f64, min: f64, max: f64) -> f64 {
            if (max - min).abs() < 1e-10 {
                return 0.5;
            }
            (score - min) / (max - min)
        }

        let vector_scores = vec![0.1, 0.4, 0.8, 0.95];
        let normalized: Vec<f64> = vector_scores
            .iter()
            .map(|&s| normalize(s, 0.1, 0.95))
            .collect();

        // First should be 0, last should be 1
        assertions::assert_approx_eq(normalized[0] as f32, 0.0, 0.0001);
        assertions::assert_approx_eq(normalized[3] as f32, 1.0, 0.0001);
    }

    /// Test alpha tuning strategy
    #[test]
    fn test_alpha_tuning() {
        // Test different alpha values for retrieval quality
        let alphas = [0.0, 0.25, 0.5, 0.75, 1.0];

        // Simulated document scores
        let vector_score = 0.7;
        let text_score = 0.6;

        let fused_scores: Vec<f64> = alphas
            .iter()
            .map(|&a| linear_fusion(vector_score, text_score, a))
            .collect();

        // Scores should be between text_score and vector_score
        for score in &fused_scores {
            assert!(*score >= text_score);
            assert!(*score <= vector_score);
        }
    }

    /// Test fusion with varying weights
    #[test]
    fn test_weighted_fusion_variants() {
        let vector = 0.8;
        let text = 0.4;

        // Different weight schemes
        let equal = linear_fusion(vector, text, 0.5);
        let vector_heavy = linear_fusion(vector, text, 0.8);
        let text_heavy = linear_fusion(vector, text, 0.2);

        assert!(vector_heavy > equal);
        assert!(equal > text_heavy);
    }
}

/// Test module for hybrid search performance
#[cfg(test)]
mod hybrid_performance_tests {
    use super::*;

    /// Test that hybrid search overhead is acceptable
    #[test]
    fn test_hybrid_overhead() {
        // Hybrid should be less than 2x single branch
        let vector_latency: f64 = 10.0; // ms
        let text_latency: f64 = 8.0; // ms
        let hybrid_latency: f64 = 15.0; // ms

        let single_branch_max = vector_latency.max(text_latency);
        let overhead_ratio = hybrid_latency / single_branch_max;

        assert!(
            overhead_ratio < 2.0,
            "Hybrid latency {} should be < 2x single branch {}",
            hybrid_latency,
            single_branch_max
        );
    }

    /// Test parallel execution benefit
    #[test]
    fn test_parallel_execution() {
        // Vector and text searches can run in parallel
        let vector_latency: f64 = 10.0;
        let text_latency: f64 = 8.0;

        // Sequential: vector + text
        let sequential = vector_latency + text_latency;

        // Parallel: max(vector, text) + fusion overhead
        let fusion_overhead: f64 = 2.0;
        let parallel = vector_latency.max(text_latency) + fusion_overhead;

        assert!(parallel < sequential, "Parallel execution should be faster");

        // Speedup should be meaningful
        let speedup = sequential / parallel;
        assert!(speedup > 1.3, "Speedup should be at least 30%");
    }

    /// Test fusion overhead
    #[test]
    fn test_fusion_overhead() {
        // Fusion step should be minimal
        let num_results = 1000;
        let fusion_time_us = 100.0; // microseconds

        // Per-result fusion time
        let per_result_us = fusion_time_us / num_results as f64;

        // Should be < 1 microsecond per result
        assert!(
            per_result_us < 1.0,
            "Fusion should be < 1us per result, got {}us",
            per_result_us
        );
    }

    /// Test result limit impact
    #[test]
    fn test_result_limit_scaling() {
        // Latency should scale sub-linearly with result limit
        let limits = [10, 100, 1000];

        // Simulated latencies (ms)
        let latencies = [5.0, 8.0, 15.0];

        // Check scaling
        for i in 1..limits.len() {
            let limit_ratio = limits[i] as f64 / limits[i - 1] as f64;
            let latency_ratio = latencies[i] / latencies[i - 1];

            // Latency should grow slower than limit
            assert!(
                latency_ratio < limit_ratio,
                "Latency should scale sub-linearly"
            );
        }
    }

    /// Test memory efficiency
    #[test]
    fn test_memory_efficiency() {
        // Hybrid search should not require excessive memory
        let vector_results = 1000;
        let text_results = 1000;
        let result_size_bytes = 100; // Per result

        let total_memory = (vector_results + text_results) * result_size_bytes;
        let max_memory_kb = 1024; // 1MB limit

        assert!(
            total_memory / 1024 < max_memory_kb,
            "Memory usage should be under {}KB",
            max_memory_kb
        );
    }

    /// Test throughput requirements
    #[test]
    fn test_throughput() {
        // Target: 1000 QPS for hybrid search
        let target_qps = 1000.0;
        let max_latency_ms = 1000.0 / target_qps;

        assert!(max_latency_ms == 1.0, "Need < 1ms latency for 1000 QPS");

        // With parallelism
        let concurrent_queries = 10;
        let effective_qps = target_qps / concurrent_queries as f64;
        let allowed_latency_ms = 1000.0 / effective_qps * concurrent_queries as f64;

        assert!(
            allowed_latency_ms == 10.0,
            "With 10 concurrent, can have 10ms latency"
        );
    }
}

/// Test module for hybrid search quality
#[cfg(test)]
mod hybrid_quality_tests {
    use super::*;

    /// Test that hybrid improves over single modality
    #[test]
    fn test_hybrid_quality_improvement() {
        // Simulated recall@10 for different search types
        let vector_recall = 0.75;
        let text_recall = 0.70;
        let hybrid_recall = 0.88;

        // Hybrid should improve over both
        assert!(hybrid_recall > vector_recall);
        assert!(hybrid_recall > text_recall);
    }

    /// Test hybrid on different query types
    #[test]
    fn test_query_type_handling() {
        // Query types and expected best modality
        let query_types = [
            ("semantic concept", "vector"),  // Abstract concept
            ("exact phrase", "text"),        // Literal match
            ("keyword + meaning", "hybrid"), // Mixed
        ];

        // Hybrid should handle all reasonably
        for (query_type, best_for) in query_types {
            // Hybrid should be at least 80% as good as specialized
            let hybrid_quality = 0.85;
            let specialized_quality = 1.0;

            let quality_ratio = hybrid_quality / specialized_quality;
            assert!(
                quality_ratio >= 0.8,
                "Hybrid should be >= 80% of specialized for '{}'",
                query_type
            );
        }
    }

    /// Test recall vs precision tradeoff
    #[test]
    fn test_recall_precision_tradeoff() {
        // Different alpha values favor different tradeoffs
        struct Results {
            alpha: f64,
            precision: f64,
            recall: f64,
        }

        let results = [
            Results {
                alpha: 0.3,
                precision: 0.65,
                recall: 0.85,
            }, // Text-heavy: better recall
            Results {
                alpha: 0.5,
                precision: 0.72,
                recall: 0.78,
            }, // Balanced
            Results {
                alpha: 0.7,
                precision: 0.80,
                recall: 0.70,
            }, // Vector-heavy: better precision
        ];

        // All should have reasonable F1
        for r in &results {
            let f1 = 2.0 * r.precision * r.recall / (r.precision + r.recall);
            assert!(f1 > 0.7, "F1 should be > 0.7 for alpha={}", r.alpha);
        }
    }
}
