//! Coherence-gated pattern retrieval for the ReasoningBank.
//!
//! `ReasoningBank::find_similar` (see `reasoning_bank.rs`) ranks patterns by
//! raw cosine similarity to the query alone, the same retrieval primitive
//! described for Google/Yale/UIUC's ReasoningBank agent-memory framework
//! (arXiv:2509.25140): an embedding search over stored reasoning strategies
//! returns the top-k nearest items, which are then injected into the agent's
//! context. That is fast and simple, but a single stored pattern that
//! happens to land close to the query by chance — one trajectory's noisy
//! embedding, or (adversarially) a deliberately crafted near-duplicate —
//! ranks exactly like a pattern that many independent trajectories agree is
//! the right strategy for that region of embedding space. Nothing in plain
//! cosine ranking distinguishes a corroborated signal from a lucky (or
//! planted) coincidence.
//!
//! This module adds a **coherence gate**: a pattern's retrieval score is
//! blended with how many *other* patterns in the bank independently sit
//! close to it in embedding space (`PatternCoherenceGraph`). The idea mirrors
//! CAGE's coherence-aware passage reranking (arXiv:2609.04647, which blends
//! each candidate's similarity to the *other retrieved candidates* with its
//! similarity to the query) and GRADA's graph-based reranker against
//! adversarial documents (which penalizes passages that are query-relevant
//! but structurally unsupported by the rest of the retrieved set) — applied
//! here to ReasoningBank's pattern store instead of a RAG passage set.
//!
//! Two scoring strategies are provided:
//! - [`find_similar_coherence_gated`] (candidate A): blends cosine similarity
//!   with the pattern-graph coherence gate.
//! - [`find_similar_quality_weighted`] (candidate B, ablation): blends cosine
//!   similarity with each pattern's existing `avg_quality` metadata instead
//!   of a graph signal, to isolate whether structural corroboration is doing
//!   something that reused bookkeeping fields do not.
//!
//! See `docs/research/nightly/2026_10_31_coherence_gated_reasoning_bank/` for
//! the falsifiable hypothesis, benchmark methodology and measured results
//! this module was built to test.

use crate::types::LearnedPattern;
use std::collections::HashMap;

/// Configuration for coherence-gated retrieval.
#[derive(Clone, Copy, Debug)]
pub struct CoherenceConfig {
    /// Cosine-similarity threshold above which two patterns are considered
    /// mutually corroborating (an edge in the coherence graph).
    pub edge_threshold: f32,
    /// Blend weight for the final score: `alpha * raw_similarity +
    /// (1 - alpha) * gate`. `alpha = 1.0` degenerates to the cosine-only
    /// baseline; `alpha = 0.0` ranks purely by corroboration.
    pub alpha: f32,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            edge_threshold: 0.75,
            alpha: 0.5,
        }
    }
}

/// A corroboration graph over a snapshot of `LearnedPattern`s.
///
/// An edge connects two patterns whose centroids have cosine similarity at
/// or above `edge_threshold`. A pattern's *gate* value is its degree
/// (count of corroborating neighbours) normalised by `n - 1`, so an
/// isolated pattern gates to `0.0` and a pattern corroborated by every other
/// pattern in the bank gates to `1.0`.
///
/// Building the graph is `O(n^2)` in the number of patterns; this is the
/// amortised cost paid once after patterns change (e.g. after
/// `ReasoningBank::extract_patterns` or `consolidate`), not per query —
/// per-query lookups are `O(1)` via [`PatternCoherenceGraph::gate`].
#[derive(Debug, Clone)]
pub struct PatternCoherenceGraph {
    /// pattern id -> normalised degree in [0, 1]
    gate: HashMap<u64, f32>,
    threshold: f32,
}

impl PatternCoherenceGraph {
    /// Build the coherence graph for a pattern snapshot.
    ///
    /// `patterns` with fewer than 2 entries all gate to `0.0` (no possible
    /// corroboration). NaN/zero-norm centroids never count as an edge
    /// (`LearnedPattern::similarity` returns `0.0` for those, which is below
    /// any sane threshold).
    pub fn build(patterns: &[LearnedPattern], edge_threshold: f32) -> Self {
        let n = patterns.len();
        let mut degree: HashMap<u64, u32> = HashMap::with_capacity(n);
        for p in patterns {
            degree.entry(p.id).or_insert(0);
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = patterns[i].similarity(&patterns[j].centroid);
                if sim.is_finite() && sim >= edge_threshold {
                    *degree.entry(patterns[i].id).or_insert(0) += 1;
                    *degree.entry(patterns[j].id).or_insert(0) += 1;
                }
            }
        }

        let denom = (n.saturating_sub(1)).max(1) as f32;
        let gate = degree
            .into_iter()
            .map(|(id, d)| (id, (d as f32 / denom).clamp(0.0, 1.0)))
            .collect();

        Self {
            gate,
            threshold: edge_threshold,
        }
    }

    /// Normalised corroboration gate for a pattern id, `0.0` if unknown.
    pub fn gate(&self, id: u64) -> f32 {
        self.gate.get(&id).copied().unwrap_or(0.0)
    }

    /// The edge threshold this graph was built with.
    pub fn edge_threshold(&self) -> f32 {
        self.threshold
    }

    /// Number of patterns tracked.
    pub fn len(&self) -> usize {
        self.gate.len()
    }

    /// Whether the graph tracks zero patterns.
    pub fn is_empty(&self) -> bool {
        self.gate.is_empty()
    }
}

/// Candidate A: rank patterns by `alpha * cosine + (1 - alpha) * coherence_gate`.
///
/// `graph` must have been built from the same (or a compatible) pattern
/// snapshot as `patterns`; patterns absent from `graph` gate to `0.0`.
/// Returns up to `k` `(pattern_id, score)` pairs, highest score first, ties
/// broken by ascending pattern id for determinism.
pub fn find_similar_coherence_gated(
    patterns: &[LearnedPattern],
    graph: &PatternCoherenceGraph,
    query: &[f32],
    k: usize,
    alpha: f32,
) -> Vec<(u64, f32)> {
    let alpha = alpha.clamp(0.0, 1.0);
    let mut scored: Vec<(u64, f32)> = patterns
        .iter()
        .map(|p| {
            let sim = p.similarity(query);
            let gate = graph.gate(p.id);
            (p.id, alpha * sim + (1.0 - alpha) * gate)
        })
        .collect();

    sort_desc_by_score(&mut scored);
    scored.truncate(k);
    scored
}

/// Candidate B (ablation): rank patterns by `alpha * cosine + (1 - alpha) *
/// normalised_quality`, using each pattern's existing `avg_quality` field
/// instead of any graph-derived signal.
///
/// `avg_quality` is normalised across the input slice to `[0, 1]` so the
/// blend weight `alpha` is comparable to [`find_similar_coherence_gated`].
/// This exists purely to test whether reused bookkeeping metadata already
/// captures what the coherence graph captures — see the module docs and the
/// research write-up for why this ablation matters.
pub fn find_similar_quality_weighted(
    patterns: &[LearnedPattern],
    query: &[f32],
    k: usize,
    alpha: f32,
) -> Vec<(u64, f32)> {
    let alpha = alpha.clamp(0.0, 1.0);

    let (min_q, max_q) = patterns.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
        (lo.min(p.avg_quality), hi.max(p.avg_quality))
    });
    let range = (max_q - min_q).max(1e-8);

    let mut scored: Vec<(u64, f32)> = patterns
        .iter()
        .map(|p| {
            let sim = p.similarity(query);
            let norm_q = if max_q > min_q {
                (p.avg_quality - min_q) / range
            } else {
                0.0
            };
            (p.id, alpha * sim + (1.0 - alpha) * norm_q)
        })
        .collect();

    sort_desc_by_score(&mut scored);
    scored.truncate(k);
    scored
}

fn sort_desc_by_score(scored: &mut [(u64, f32)]) {
    scored.sort_by(|a, b| match b.1.partial_cmp(&a.1) {
        Some(ord) if ord != std::cmp::Ordering::Equal => ord,
        _ => a.0.cmp(&b.0),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(id: u64, centroid: Vec<f32>, avg_quality: f32) -> LearnedPattern {
        let mut p = LearnedPattern::new(id, centroid);
        p.avg_quality = avg_quality;
        p
    }

    #[test]
    fn isolated_pattern_gates_to_zero() {
        // A single pattern with no siblings at all: n=1, denom clamps to 1.
        let patterns = vec![pattern(0, vec![1.0, 0.0], 0.5)];
        let graph = PatternCoherenceGraph::build(&patterns, 0.75);
        assert_eq!(graph.gate(0), 0.0);
    }

    #[test]
    fn corroborated_cluster_gates_higher_than_isolated_outlier() {
        // Three near-duplicate patterns (mutually above threshold) plus one
        // orthogonal outlier with no neighbours.
        let patterns = vec![
            pattern(0, vec![1.0, 0.0, 0.0], 0.5),
            pattern(1, vec![0.98, 0.02, 0.0], 0.5),
            pattern(2, vec![0.97, 0.03, 0.0], 0.5),
            pattern(3, vec![0.0, 0.0, 1.0], 0.5),
        ];
        let graph = PatternCoherenceGraph::build(&patterns, 0.9);
        assert!(graph.gate(0) > graph.gate(3));
        assert_eq!(graph.gate(3), 0.0);
    }

    #[test]
    fn coherence_gated_demotes_isolated_near_duplicate_of_query() {
        // The core hypothesis-relevant case: pattern 99 is an isolated
        // near-perfect match for the query (highest raw cosine similarity:
        // it sits at 2 degrees from the query on the unit circle) but has
        // no corroborating siblings. Patterns 0/1/2 sit at 24/25/26 degrees
        // from the query — a good, but not perfect, match individually —
        // and are mutually within 2 degrees of each other, well above the
        // 0.95 edge threshold, while each is ~22-24 degrees from pattern 99
        // (sim ~0.91-0.93, below threshold): a corroborated cluster next to
        // an uncorroborated near-duplicate of the query.
        let query = vec![1.0, 0.0, 0.0];
        let patterns = vec![
            pattern(0, vec![0.9135, 0.4067, 0.0], 0.5), // 24 degrees
            pattern(1, vec![0.9063, 0.4226, 0.0], 0.5), // 25 degrees
            pattern(2, vec![0.8988, 0.4384, 0.0], 0.5), // 26 degrees
            pattern(99, vec![0.9994, 0.0349, 0.0], 0.5), // 2 degrees, isolated
        ];
        let graph = PatternCoherenceGraph::build(&patterns, 0.95);

        // Baseline: isolated near-duplicate wins outright.
        let mut baseline: Vec<(u64, f32)> = patterns
            .iter()
            .map(|p| (p.id, p.similarity(&query)))
            .collect();
        sort_desc_by_score(&mut baseline);
        assert_eq!(baseline[0].0, 99);

        // Coherence-gated: the isolated pattern's score is pulled down by
        // its zero gate while the corroborated cluster's members keep a
        // nonzero gate, enough to overtake it at alpha=0.5 in this example.
        let gated = find_similar_coherence_gated(&patterns, &graph, &query, 4, 0.5);
        assert_ne!(gated[0].0, 99, "isolated near-duplicate should be demoted");
    }

    #[test]
    fn quality_weighted_ablation_does_not_use_graph_structure() {
        // Same setup as above, but all patterns share identical avg_quality
        // (label-uninformative), so the quality-weighted candidate has
        // nothing to blend with and should behave like the cosine baseline.
        let query = vec![1.0, 0.0, 0.0];
        let patterns = vec![
            pattern(0, vec![0.9135, 0.4067, 0.0], 0.5),
            pattern(1, vec![0.9063, 0.4226, 0.0], 0.5),
            pattern(2, vec![0.8988, 0.4384, 0.0], 0.5),
            pattern(99, vec![0.9994, 0.0349, 0.0], 0.5),
        ];
        let ablated = find_similar_quality_weighted(&patterns, &query, 4, 0.5);
        assert_eq!(ablated[0].0, 99, "uniform quality carries no signal");
    }

    #[test]
    fn malformed_input_does_not_panic() {
        // Mismatched centroid dimension, NaN, and empty-bank edge cases.
        let patterns = vec![
            pattern(0, vec![1.0, f32::NAN], 0.5),
            pattern(1, vec![0.0, 0.0, 0.0], 0.5), // wrong dim vs query
        ];
        let graph = PatternCoherenceGraph::build(&patterns, 0.75);
        let query = vec![1.0, 0.0];
        let result = find_similar_coherence_gated(&patterns, &graph, &query, 10, 0.5);
        assert!(result.len() <= 2);

        let empty: Vec<LearnedPattern> = Vec::new();
        let empty_graph = PatternCoherenceGraph::build(&empty, 0.75);
        assert!(empty_graph.is_empty());
        let result = find_similar_coherence_gated(&empty, &empty_graph, &query, 10, 0.5);
        assert!(result.is_empty());

        // k = 0 must return nothing, k > n must not panic.
        assert!(find_similar_coherence_gated(&patterns, &graph, &query, 0, 0.5).is_empty());
        assert_eq!(
            find_similar_coherence_gated(&patterns, &graph, &query, 999, 0.5).len(),
            2
        );
    }

    #[test]
    fn alpha_is_clamped() {
        let patterns = vec![pattern(0, vec![1.0, 0.0], 0.5)];
        let graph = PatternCoherenceGraph::build(&patterns, 0.75);
        let query = vec![1.0, 0.0];
        // Out-of-range alpha must not panic and must still produce a score
        // within a sane range (clamped internally to [0, 1]).
        let r = find_similar_coherence_gated(&patterns, &graph, &query, 1, 5.0);
        assert!(r[0].1.is_finite());
        let r = find_similar_coherence_gated(&patterns, &graph, &query, 1, -5.0);
        assert!(r[0].1.is_finite());
    }

    #[test]
    fn scale_does_not_panic_and_gate_stays_bounded() {
        // 2000 patterns in 3 loose clusters: correctness/stress check, not
        // a latency assertion (latency is measured in the bench example).
        let mut patterns = Vec::with_capacity(2000);
        for i in 0..2000u64 {
            let cluster = i % 3;
            let base = cluster as f32;
            let jitter = (i as f32 * 0.0001) % 0.05;
            patterns.push(pattern(i, vec![base + jitter, 1.0 - base, 0.0], 0.5));
        }
        let graph = PatternCoherenceGraph::build(&patterns, 0.75);
        for p in &patterns {
            let g = graph.gate(p.id);
            assert!((0.0..=1.0).contains(&g));
        }
    }

    #[test]
    fn concurrent_reads_are_consistent() {
        // The graph and pattern snapshot are plain immutable data once
        // built; concurrent read-only queries against the same Arc-shared
        // snapshot must all agree with a single-threaded reference result.
        use std::sync::Arc;
        use std::thread;

        let mut patterns = Vec::new();
        for i in 0..64u64 {
            let cluster = i % 4;
            patterns.push(pattern(
                i,
                vec![cluster as f32, (3 - cluster) as f32, 0.1],
                0.4 + (i as f32 * 0.001),
            ));
        }
        let patterns = Arc::new(patterns);
        let graph = Arc::new(PatternCoherenceGraph::build(&patterns, 0.6));
        let query = vec![1.0, 2.0, 0.1];

        let reference = find_similar_coherence_gated(&patterns, &graph, &query, 5, 0.5);

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let patterns = Arc::clone(&patterns);
                let graph = Arc::clone(&graph);
                let query = query.clone();
                let reference = reference.clone();
                thread::spawn(move || {
                    let result = find_similar_coherence_gated(&patterns, &graph, &query, 5, 0.5);
                    assert_eq!(result, reference);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }
}
