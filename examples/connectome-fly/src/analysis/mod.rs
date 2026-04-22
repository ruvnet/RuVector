//! Live analysis orchestrator mounting RuVector primitives on the
//! spike stream.
//!
//! Split into:
//!
//! - `types`     — `AnalysisConfig`, `FunctionalPartition`,
//!                 `MotifHit`, `MotifSignature`, the `MotifIndex`.
//! - `partition` — `ruvector-mincut` orchestration on the
//!                 coactivation-weighted connectome.
//! - `motif`     — spike-window raster construction, SDPA-backed
//!                 embedding via `ruvector-attention`, bounded
//!                 in-memory kNN.
//!
//! The public surface is the `Analysis` struct re-exported from
//! here.

pub mod diskann_motif;
pub mod gpu;
pub mod motif;
pub mod partition;
pub mod structural;
pub mod types;

use ruvector_attention::attention::ScaledDotProductAttention;

pub use diskann_motif::{DiskAnnMotifIndex, EmbeddingF32, VamanaParams};
pub use types::{
    AnalysisConfig, FunctionalPartition, MotifEmbedding, MotifHit, MotifIndex, MotifSignature,
};

use crate::connectome::Connectome;
use crate::lif::Spike;

/// Top-level analysis orchestrator.
pub struct Analysis {
    cfg: AnalysisConfig,
    sdpa: ScaledDotProductAttention,
    w_q: Vec<f32>,
    w_k: Vec<f32>,
    w_v: Vec<f32>,
}

impl Analysis {
    /// Build a new analysis orchestrator.
    pub fn new(cfg: AnalysisConfig) -> Self {
        let d = cfg.embed_dim;
        let w_q = motif::det_projection(15 * cfg.motif_bins, d, cfg.proj_seed ^ 0xA);
        let w_k = motif::det_projection(15 * cfg.motif_bins, d, cfg.proj_seed ^ 0xB);
        let w_v = motif::det_projection(15 * cfg.motif_bins, d, cfg.proj_seed ^ 0xC);
        Self {
            cfg,
            sdpa: ScaledDotProductAttention::new(d),
            w_q,
            w_k,
            w_v,
        }
    }

    /// Build a functional partition of the connectome using mincut
    /// weighted by recent spike coactivation.
    pub fn functional_partition(&self, conn: &Connectome, spikes: &[Spike]) -> FunctionalPartition {
        partition::functional_partition(&self.cfg, conn, spikes)
    }

    /// Build a structural partition of the static connectome (no
    /// coactivation weighting). AC-3a: does `ruvector-mincut` recover
    /// SBM module structure from the connectome alone? See
    /// `analysis::structural` for rationale and ADR-154 §3.4 for the
    /// split with `functional_partition`.
    pub fn structural_partition(&self, conn: &Connectome) -> FunctionalPartition {
        structural::structural_partition(&self.cfg, conn)
    }

    /// Greedy-modularity community labels for the static connectome.
    /// Louvain-style level-1, deterministic, no randomness. Used by
    /// AC-3a to publish a paired baseline against `structural_partition`.
    pub fn greedy_modularity_labels(&self, conn: &Connectome) -> Vec<u32> {
        structural::greedy_modularity_labels(conn)
    }

    /// Build motif embeddings over sliding windows and index them.
    /// Returns the index plus the top-k repeated motifs.
    ///
    /// When `cfg.use_diskann = true` the embeddings are inserted into
    /// a `DiskAnnMotifIndex` in addition to the bounded brute-force
    /// `MotifIndex` so downstream callers can drive AC-2-diskann from
    /// the same embedding corpus. The `(MotifIndex, Vec<MotifHit>)`
    /// return shape stays source-compatible — the DiskANN view is
    /// accessed via [`Self::embed_motif_windows`] +
    /// [`DiskAnnMotifIndex::new`].
    pub fn retrieve_motifs(
        &self,
        conn: &Connectome,
        spikes: &[Spike],
        k: usize,
    ) -> (MotifIndex, Vec<MotifHit>) {
        motif::retrieve_motifs(
            &self.cfg, &self.sdpa, &self.w_q, &self.w_k, &self.w_v, conn, spikes, k,
        )
    }

    /// Encode every non-empty motif window with the same SDPA
    /// embedder that drives [`Self::retrieve_motifs`], and return the
    /// list of embeddings with their class / time metadata.
    ///
    /// This is the entry point for the DiskANN retrieval path: the
    /// caller builds a [`DiskAnnMotifIndex`] over the returned
    /// vectors, then runs `precision_at_k` against them.
    pub fn embed_motif_windows(
        &self,
        conn: &Connectome,
        spikes: &[Spike],
    ) -> Vec<MotifEmbedding> {
        motif::embed_windows(
            &self.cfg, &self.sdpa, &self.w_q, &self.w_k, &self.w_v, conn, spikes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::ConnectomeConfig;
    use crate::lif::{Engine, EngineConfig};
    use crate::observer::Observer;
    use crate::stimulus::Stimulus;

    #[test]
    fn analysis_pipeline_runs() {
        let conn = Connectome::generate(&ConnectomeConfig {
            num_neurons: 256,
            avg_out_degree: 16.0,
            ..ConnectomeConfig::default()
        });
        let mut eng = Engine::new(&conn, EngineConfig::default());
        let stim = Stimulus::pulse_train(conn.sensory_neurons(), 30.0, 60.0, 90.0, 100.0);
        let mut obs = Observer::new(conn.num_neurons());
        eng.run_with(&stim, &mut obs, 150.0);
        let spikes = obs.spikes().to_vec();
        let analysis = Analysis::new(AnalysisConfig::default());
        let _part = analysis.functional_partition(&conn, &spikes);
        let (_, hits) = analysis.retrieve_motifs(&conn, &spikes, 5);
        for w in hits.windows(2) {
            assert!(w[0].nearest_distance <= w[1].nearest_distance);
        }
    }
}
