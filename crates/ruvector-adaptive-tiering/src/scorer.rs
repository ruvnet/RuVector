//! Concrete scoring strategies for adaptive tier placement.

use crate::{
    temperature::{TieringConfig, VectorMeta},
    Scorer,
};

/// Baseline: tier placement by accumulated access count only.
///
/// Vectors that have been returned in search results most often move to the
/// hot tier.  Semantic properties are ignored.
pub struct AccessOnlyScorer;

impl Scorer for AccessOnlyScorer {
    fn name(&self) -> &'static str {
        "AccessOnly"
    }

    fn score(&self, meta: &VectorMeta, _epoch: u64, _cfg: &TieringConfig) -> f32 {
        meta.access_count as f32
    }

    fn needs_coherence(&self) -> bool {
        false
    }
}

/// Variant 2: tier placement by intra-cluster coherence.
///
/// Vectors that are geometrically tight with their neighbours receive a higher
/// score regardless of access history.  This favours semantically coherent
/// knowledge clusters, which are likely to be relevant to future queries even
/// before they have been accessed.
pub struct CoherenceScorer;

impl Scorer for CoherenceScorer {
    fn name(&self) -> &'static str {
        "Coherence"
    }

    fn score(&self, meta: &VectorMeta, _epoch: u64, _cfg: &TieringConfig) -> f32 {
        meta.coherence_score
    }

    fn needs_coherence(&self) -> bool {
        true
    }
}

/// Variant 3: composite semantic temperature.
///
/// Combines exponentially-decayed recency, intra-cluster coherence, and
/// log-normalised graph centrality into a single scalar.  Balances cold-start
/// quality (coherence + centrality) with adaptation to observed query patterns
/// (recency).
pub struct SemanticTempScorer;

impl Scorer for SemanticTempScorer {
    fn name(&self) -> &'static str {
        "SemanticTemp"
    }

    fn score(&self, meta: &VectorMeta, epoch: u64, cfg: &TieringConfig) -> f32 {
        meta.semantic_temperature(epoch, cfg)
    }

    fn needs_coherence(&self) -> bool {
        true
    }
}
