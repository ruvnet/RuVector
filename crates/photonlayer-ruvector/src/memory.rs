//! In-memory experiment store with cosine-similarity nearest-neighbour recall
//! (ADR-260 §11–§12).
//!
//! [`ExperimentMemory`] is the primary entry point. Call [`remember`] to store
//! a finished experiment, then [`nearest`] to recall the most similar prior
//! experiments by embedding.

use photonlayer_core::prelude::{ExperimentReceipt, MetricReport, OpticalConfig};
use ruvector_coherence::cosine_similarity;
use serde::{Deserialize, Serialize};

use crate::embedding::EMBED_DIM;

/// A single stored experiment with its embedding, label, config, and receipt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentRecord {
    /// Unique experiment identifier (matches `receipt.experiment_id`).
    pub id: String,
    /// Human-readable class or outcome label, e.g. `"pass"` or `"fail"`.
    pub label: String,
    /// The optical configuration used.
    pub config: OpticalConfig,
    /// `mask_id` from the [`PhaseMask`] used.
    pub mask_id: String,
    /// 32-dim L2-normalised embedding (mask histogram + frame spectrum).
    pub embedding: Vec<f32>,
    /// Bound RVF receipt for this experiment.
    pub receipt: ExperimentReceipt,
    /// Benchmark metrics collected during the run.
    pub metrics: MetricReport,
}

/// A nearest-neighbour hit returned by [`ExperimentMemory::nearest`].
#[derive(Clone, Debug)]
pub struct NearestHit {
    /// Experiment identifier.
    pub id: String,
    /// Cosine similarity to the query embedding (higher = more similar).
    pub score: f64,
}

/// A nearest-mask hit: lightweight result when only the mask is of interest.
#[derive(Clone, Debug)]
pub struct MaskSearchHit {
    /// Experiment identifier.
    pub id: String,
    /// Cosine similarity to the query embedding.
    pub score: f64,
    /// `mask_id` of the matched experiment.
    pub mask_id: String,
}

/// In-memory store of [`ExperimentRecord`]s supporting cosine-similarity
/// nearest-neighbour search (ADR-260 §11–§12).
///
/// All searches are linear scans over stored embeddings (exact NN). This is
/// appropriate for experiment-scale datasets (<10 k entries); swap for HNSW
/// when larger.
#[derive(Default, Debug)]
pub struct ExperimentMemory {
    records: Vec<ExperimentRecord>,
}

impl ExperimentMemory {
    /// Create an empty experiment memory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a new experiment record.
    ///
    /// # Panics
    /// Panics if the embedding dimension is not [`EMBED_DIM`].
    pub fn remember(&mut self, record: ExperimentRecord) {
        assert_eq!(
            record.embedding.len(),
            EMBED_DIM,
            "embedding must have {} dims, got {}",
            EMBED_DIM,
            record.embedding.len()
        );
        self.records.push(record);
    }

    /// Number of stored experiments.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns `true` when no experiments have been stored.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Return up to `limit` stored experiments sorted by descending cosine
    /// similarity to `query_embedding`.
    pub fn nearest(&self, query_embedding: &[f32], limit: usize) -> Vec<NearestHit> {
        let mut scored: Vec<(f64, &str)> = self
            .records
            .iter()
            .map(|r| {
                (
                    cosine_similarity(query_embedding, &r.embedding),
                    r.id.as_str(),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(limit)
            .map(|(score, id)| NearestHit {
                id: id.to_owned(),
                score,
            })
            .collect()
    }

    /// Return all records with the given label.
    pub fn by_label(&self, label: &str) -> Vec<&ExperimentRecord> {
        self.records.iter().filter(|r| r.label == label).collect()
    }

    /// Iterate over all stored records.
    pub fn records(&self) -> &[ExperimentRecord] {
        &self.records
    }

    /// Look up a record by its exact id.
    pub fn get(&self, id: &str) -> Option<&ExperimentRecord> {
        self.records.iter().find(|r| r.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photonlayer_core::prelude::{
        build_receipt, InputImage, OpticalConfig, OpticalSimulator, PhaseMask, Provenance,
        ScalarSimulator,
    };

    use crate::embedding::experiment_embedding;

    fn make_record(id: &str, label: &str, seed: u64) -> ExperimentRecord {
        let n = 16;
        let pixels: Vec<f32> = (0..n * n).map(|i| (i % n) as f32 / n as f32).collect();
        let img = InputImage::from_norm_f32(n, n, pixels).unwrap();
        let mask = PhaseMask::random(n, n, seed);
        let cfg = OpticalConfig::demo(n, n);
        let frame = ScalarSimulator.simulate(&img, &mask, &cfg).unwrap();
        let metrics = MetricReport::default();
        let receipt = build_receipt(
            id,
            &img,
            &mask,
            &cfg,
            &frame,
            &metrics,
            &Provenance::default(),
        );
        let embedding = experiment_embedding(&mask, &frame);
        ExperimentRecord {
            id: id.to_owned(),
            label: label.to_owned(),
            config: cfg,
            mask_id: mask.mask_id.clone(),
            embedding,
            receipt,
            metrics,
        }
    }

    #[test]
    fn remember_and_len() {
        let mut mem = ExperimentMemory::new();
        assert!(mem.is_empty());
        mem.remember(make_record("exp-1", "pass", 1));
        assert_eq!(mem.len(), 1);
        mem.remember(make_record("exp-2", "fail", 2));
        assert_eq!(mem.len(), 2);
    }

    #[test]
    fn nearest_finds_planted_match() {
        let mut mem = ExperimentMemory::new();
        // Store three different experiments.
        let target_record = make_record("target", "pass", 42);
        let query_emb = target_record.embedding.clone();
        mem.remember(target_record);
        mem.remember(make_record("other-1", "pass", 7));
        mem.remember(make_record("other-2", "fail", 99));

        let hits = mem.nearest(&query_emb, 3);
        // The exact stored embedding must score 1.0 at rank 0.
        assert_eq!(hits[0].id, "target");
        assert!(
            (hits[0].score - 1.0).abs() < 1e-5,
            "score = {}",
            hits[0].score
        );
    }

    #[test]
    fn by_label_filters_correctly() {
        let mut mem = ExperimentMemory::new();
        mem.remember(make_record("a", "pass", 1));
        mem.remember(make_record("b", "fail", 2));
        mem.remember(make_record("c", "pass", 3));
        let passes = mem.by_label("pass");
        assert_eq!(passes.len(), 2);
        let fails = mem.by_label("fail");
        assert_eq!(fails.len(), 1);
    }
}
