//! Rate-histogram motif encoder — alternative to the SDPA path in
//! `motif.rs`. Designed as a *controlled A/B baseline* for the AC-2
//! encoder-vs-substrate diagnosis in ADR-154 §17 item 10.
//!
//! Design intent:
//!
//! - The shipped SDPA + deterministic-low-rank-projection encoder is
//!   protocol-blind on the expanded 8-protocol labeled corpus
//!   (precision@5 ≈ random). Three remediations plateau at ≤ 0.60.
//!   The ADR calls for pinning the bottleneck: encoder, substrate, or
//!   labels.
//! - This module implements the *encoder* axis: a trivial
//!   row-major flatten of the normalised raster produced by
//!   `motif::build_raster`. No projection, no attention, no additional
//!   normalisation. Every bin of every class is preserved verbatim.
//! - If this cheap baseline scores *higher* than SDPA on the same
//!   8-protocol labeled corpus, SDPA is actively hurting. If it scores
//!   the same or lower, the substrate — not the encoder — is the
//!   bottleneck.
//!
//! The encoder is deterministic (no RNG, no state) and uses exactly
//! one allocation (the output vector).

use crate::connectome::Connectome;
use crate::lif::Spike;

use super::motif::build_raster;
use super::types::{AnalysisConfig, MotifHit, MotifIndex, MotifWindow};

/// Flatten a raster `[n_rows][n_cols]` into a row-major `Vec<f32>`.
///
/// The output length is `n_rows * n_cols` and `out[r * n_cols + c] ==
/// raster[r][c]`. No normalisation beyond what the caller already
/// applied — we preserve the row-normalised form emitted by
/// `motif::build_raster` verbatim so the A/B comparison isolates "what
/// does SDPA add beyond the raster itself".
///
/// Empty rasters return an empty vector.
pub fn rate_histogram_encode(raster: &[Vec<f32>]) -> Vec<f32> {
    if raster.is_empty() {
        return Vec::new();
    }
    let n_cols = raster[0].len();
    // Guard against ragged rasters (shouldn't occur from build_raster but
    // we validate anyway — the public API surface treats this as input).
    for row in raster.iter() {
        debug_assert_eq!(
            row.len(),
            n_cols,
            "rate_histogram_encode: ragged raster (row len differs from first)"
        );
    }
    let n_rows = raster.len();
    let mut out = Vec::with_capacity(n_rows * n_cols);
    for row in raster {
        // Explicit raw-bin-count copy; `extend_from_slice` compiles to a
        // `memcpy` on contiguous data. No projection, no attention.
        out.extend_from_slice(row);
    }
    out
}

/// Build motif embeddings over sliding windows using the rate-histogram
/// encoder and index them. Mirrors `motif::retrieve_motifs` so the two
/// paths can be swapped at call sites without other changes. Returns
/// the index plus the top-k repeated motifs.
///
/// The sliding-window schedule, the in-memory kNN index, and the
/// dominant-class accounting are *identical* to the SDPA path — the
/// only difference is the per-window embedding function. This is the
/// A/B invariant the diagnostic test relies on.
pub fn rate_histogram_retrieve_motifs(
    cfg: &AnalysisConfig,
    conn: &Connectome,
    spikes: &[Spike],
    k: usize,
) -> (MotifIndex, Vec<MotifHit>) {
    let mut index = MotifIndex::new(cfg.index_capacity);
    if spikes.is_empty() {
        return (index, Vec::new());
    }
    let t_end = spikes.last().map(|s| s.t_ms).unwrap_or(0.0);
    let win = cfg.motif_window_ms;
    let bins = cfg.motif_bins;
    let step = win / 2.0;
    let mut t = 0.0;
    while t + win <= t_end + step {
        let (raster, meta) = build_raster(conn, spikes, t, win, bins);
        if meta.spike_count == 0 {
            t += step;
            continue;
        }
        let vec = rate_histogram_encode(&raster);
        index.insert(
            vec,
            MotifWindow {
                t_center_ms: t + win * 0.5,
                spike_count: meta.spike_count,
                dominant_class_idx: meta.dominant_class_idx,
            },
        );
        t += step;
    }
    let hits = index.top_k(k);
    (index, hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_empty_raster_returns_empty_vec() {
        let raster: Vec<Vec<f32>> = Vec::new();
        let v = rate_histogram_encode(&raster);
        assert!(v.is_empty());
    }

    #[test]
    fn encode_is_row_major_and_preserves_values() {
        // 3 rows × 4 cols — pick distinct values so we can spot
        // row-vs-column-major mistakes.
        let raster: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        let flat = rate_histogram_encode(&raster);
        assert_eq!(flat.len(), 12);
        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(flat[r * 4 + c], raster[r][c]);
            }
        }
    }

    #[test]
    fn encode_is_deterministic_across_runs() {
        let raster: Vec<Vec<f32>> = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];
        let a = rate_histogram_encode(&raster);
        let b = rate_histogram_encode(&raster);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "bit-level determinism required");
        }
    }
}
