//! Experiment embeddings for optical mask + detector frame (ADR-260 §12).
//!
//! A 32-dimensional L2-normalised embedding is built by concatenating:
//! - 16 bins from [`PhaseMask::phase_histogram`]  (mask signature)
//! - 16 bins from [`frame_spectrum_embedding`]     (detector frame signature)
//!
//! The same dimensionality is used for mask-only embeddings (just the
//! histogram, padded to 32 dims with zeros) so every embedding lives in the
//! same inner-product space.

use photonlayer_core::prelude::{frame_spectrum_embedding, OpticalFrame, PhaseMask};

/// Number of histogram bins contributed by the mask half.
pub const MASK_BINS: usize = 16;
/// Number of spectrum bins contributed by the frame half.
pub const FRAME_BINS: usize = 16;
/// Total embedding dimension.
pub const EMBED_DIM: usize = MASK_BINS + FRAME_BINS;

/// Normalise a mutable slice to unit L2 norm (in-place). No-op when all-zero.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Build the full 32-dim experiment embedding from a mask + detector frame.
///
/// The returned vector is L2-normalised.
pub fn experiment_embedding(mask: &PhaseMask, frame: &OpticalFrame) -> Vec<f32> {
    let mut emb = Vec::with_capacity(EMBED_DIM);
    emb.extend_from_slice(&mask.phase_histogram(MASK_BINS));
    emb.extend_from_slice(&frame_spectrum_embedding(frame, FRAME_BINS));
    l2_normalize(&mut emb);
    emb
}

/// Build a mask-only embedding (32 dims: histogram + 16 zeros), L2-normalised.
///
/// Useful when only the mask is known (e.g. for coherence family comparison).
pub fn mask_embedding(mask: &PhaseMask) -> Vec<f32> {
    let mut emb = Vec::with_capacity(EMBED_DIM);
    emb.extend_from_slice(&mask.phase_histogram(MASK_BINS));
    emb.extend(std::iter::repeat(0.0f32).take(FRAME_BINS));
    l2_normalize(&mut emb);
    emb
}

#[cfg(test)]
mod tests {
    use super::*;
    use photonlayer_core::prelude::{
        InputImage, OpticalConfig, OpticalSimulator, PhaseMask, ScalarSimulator,
    };

    fn make_frame(seed: u64) -> (PhaseMask, OpticalFrame) {
        let n = 16;
        let pixels: Vec<f32> = (0..n * n).map(|i| (i % n) as f32 / n as f32).collect();
        let img = InputImage::from_norm_f32(n, n, pixels).unwrap();
        let mask = PhaseMask::random(n, n, seed);
        let cfg = OpticalConfig::demo(n, n);
        let frame = ScalarSimulator.simulate(&img, &mask, &cfg).unwrap();
        (mask, frame)
    }

    #[test]
    fn embedding_has_correct_dimension() {
        let (mask, frame) = make_frame(1);
        let emb = experiment_embedding(&mask, &frame);
        assert_eq!(emb.len(), EMBED_DIM);
    }

    #[test]
    fn embedding_is_unit_norm() {
        let (mask, frame) = make_frame(2);
        let emb = experiment_embedding(&mask, &frame);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
    }

    #[test]
    fn embedding_is_deterministic() {
        let (mask, frame) = make_frame(3);
        let a = experiment_embedding(&mask, &frame);
        let b = experiment_embedding(&mask, &frame);
        assert_eq!(a, b);
    }

    #[test]
    fn mask_embedding_has_correct_dimension() {
        let mask = PhaseMask::random(16, 16, 42);
        let emb = mask_embedding(&mask);
        assert_eq!(emb.len(), EMBED_DIM);
    }

    #[test]
    fn different_masks_differ() {
        let a = mask_embedding(&PhaseMask::random(16, 16, 100));
        let b = mask_embedding(&PhaseMask::random(16, 16, 200));
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        // Different random masks should not be identical
        assert!(
            dot < 0.9999,
            "embeddings unexpectedly identical: dot = {dot}"
        );
    }
}
