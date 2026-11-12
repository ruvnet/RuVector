//! RuVector indexer (ADR-199 §19 `ruvector-indexer`, Phase 4).
//!
//! Wraps a `ruvector_core::VectorDB` collection holding **track embeddings
//! only** (weather embeddings would go in a second, separate collection — the
//! dimensions differ on purpose). Provides:
//!
//! * similarity search ("find tracks like this one"),
//! * a **novelty score** — distance of a new track from its nearest prior
//!   neighbours, the §15 `novelty_score` component.
//!
//! Euclidean distance is used because every embedding dimension is normalized
//! to a comparable `[0, 1]` scale (see `embedding.rs`); cosine would discard
//! magnitude information that is meaningful here (e.g. altitude level).
//!
//! ## Novelty calibration
//!
//! `novelty = min(1, mean(top-3 prior distances) / NOVELTY_CALIBRATION)`.
//!
//! The constant 1.2 is calibrated on the synthetic corpus so that a repeat
//! corridor flight (mean top-3 distance ≈ 0.3–0.5) scores ≈ 0.25–0.4, while
//! the off-corridor low/slow night track (distance ≳ 1.2 from everything)
//! saturates at 1.0. With fewer than `MIN_NEIGHBOURS` prior tracks the score
//! falls back to a neutral 0.5 (the baseline period, ADR §26).

use crate::track::Track;
use ruvector_core::types::{DbOptions, DistanceMetric, SearchQuery, VectorEntry};
use ruvector_core::VectorDB;
use std::collections::HashMap;

/// See module docs: distance scale at which novelty saturates.
pub const NOVELTY_CALIBRATION: f32 = 1.2;
/// Number of nearest prior neighbours averaged for the novelty score.
pub const NOVELTY_K: usize = 3;
/// Below this many indexed tracks, novelty is the neutral 0.5.
pub const MIN_NEIGHBOURS: usize = 1;

/// Track-embedding index backed by an in-memory RuVector `VectorDB`.
pub struct TrackIndexer {
    db: VectorDB,
    len: usize,
}

impl TrackIndexer {
    /// Create an empty index for `dim`-dimensional track embeddings.
    ///
    /// Uses the flat (exact) index: the demo corpus is small and exactness
    /// keeps the acceptance tests deterministic. Swap in `hnsw_config` for
    /// large fleets.
    pub fn new(dim: usize) -> crate::Result<Self> {
        let options = DbOptions {
            dimensions: dim,
            distance_metric: DistanceMetric::Euclidean,
            // Select in-memory storage at runtime, even when Cargo feature
            // unification compiles ruvector-core's persistent backend too.
            storage_path: "memory://sky-monitor-tracks".to_string(),
            hnsw_config: None,
            quantization: None,
        };
        Ok(Self {
            db: VectorDB::new(options)?,
            len: 0,
        })
    }

    /// Number of indexed tracks.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert one track embedding with its provenance metadata
    /// (`track_id`, `icao24`, `label` = callsign or icao24).
    pub fn insert_track(&mut self, track: &Track, embedding: Vec<f32>) -> crate::Result<()> {
        let label = if track.callsign.is_empty() {
            track.icao24.clone()
        } else {
            track.callsign.clone()
        };
        let mut metadata = HashMap::new();
        metadata.insert("track_id".to_string(), serde_json::json!(track.track_id));
        metadata.insert("icao24".to_string(), serde_json::json!(track.icao24));
        metadata.insert("label".to_string(), serde_json::json!(label));
        metadata.insert(
            "overhead".to_string(),
            serde_json::json!(track.is_overhead_candidate),
        );
        self.db.insert(VectorEntry {
            id: Some(track.track_id.clone()),
            vector: embedding,
            metadata: Some(metadata),
        })?;
        self.len += 1;
        Ok(())
    }

    /// Top-`k` most similar indexed tracks (excluding the query's own
    /// `track_id` if it is already indexed), as `(track_id, distance)`.
    pub fn similar_tracks(
        &self,
        embedding: &[f32],
        exclude_track_id: Option<&str>,
        k: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        let results = self.db.search(SearchQuery {
            vector: embedding.to_vec(),
            k: k + 1, // self may come back first
            filter: None,
            ef_search: None,
        })?;
        Ok(results
            .into_iter()
            .filter(|r| exclude_track_id != Some(r.id.as_str()))
            .take(k)
            .map(|r| (r.id, r.score))
            .collect())
    }

    /// Novelty of an embedding **relative to the tracks indexed so far**
    /// (call before inserting the track itself). Range `[0, 1]`.
    pub fn novelty_score(&self, embedding: &[f32]) -> crate::Result<f32> {
        if self.len < MIN_NEIGHBOURS {
            return Ok(0.5); // baseline period: neutral
        }
        let neighbours = self.similar_tracks(embedding, None, NOVELTY_K)?;
        if neighbours.is_empty() {
            return Ok(0.5);
        }
        let mean = neighbours.iter().map(|(_, d)| *d).sum::<f32>() / neighbours.len() as f32;
        Ok((mean / NOVELTY_CALIBRATION).min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::TRACK_EMBEDDING_DIM;
    use crate::pipeline::Pipeline;

    #[test]
    fn indexes_and_finds_similar_corridor_tracks() {
        let (tracks, embeddings) = Pipeline::default().tracks_and_embeddings().unwrap();
        let mut idx = TrackIndexer::new(TRACK_EMBEDDING_DIM).unwrap();
        for (t, e) in tracks.iter().zip(&embeddings) {
            idx.insert_track(t, e.clone()).unwrap();
        }
        assert_eq!(idx.len(), tracks.len());

        // Query with the first eastbound corridor flight: best match (not
        // itself) must be another eastbound corridor flight.
        let i = tracks.iter().position(|t| t.icao24 == "c01a01").unwrap();
        let hits = idx
            .similar_tracks(&embeddings[i], Some(&tracks[i].track_id), 3)
            .unwrap();
        assert!(!hits.is_empty());
        let top_icao = &tracks
            .iter()
            .find(|t| t.track_id == hits[0].0)
            .unwrap()
            .icao24;
        assert!(
            ["a02b02", "a03c03", "c04d04"].contains(&top_icao.as_str()),
            "expected an eastbound corridor flight, got {top_icao}"
        );
    }

    #[test]
    fn novelty_is_neutral_when_empty() {
        let idx = TrackIndexer::new(TRACK_EMBEDDING_DIM).unwrap();
        let z = vec![0.0f32; TRACK_EMBEDDING_DIM];
        assert!((idx.novelty_score(&z).unwrap() - 0.5).abs() < 1e-6);
    }
}
