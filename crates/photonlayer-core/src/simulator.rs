//! End-to-end optical simulation pipeline (ADR-260 §4, §11).
//!
//! `input image -> scalar field -> learned phase mask -> propagation ->
//!  sensor intensity frame`.

use crate::config::OpticalConfig;
use crate::detector::{capture, OpticalFrame};
use crate::error::Result;
use crate::field::{InputImage, OpticalField};
use crate::mask::PhaseMask;
use crate::propagate::propagate;

/// Abstraction over an optical frontend so backends (native, WASM, GPU) can be
/// swapped while preserving the determinism invariant.
pub trait OpticalSimulator {
    fn simulate(
        &self,
        input: &InputImage,
        mask: &PhaseMask,
        config: &OpticalConfig,
    ) -> Result<OpticalFrame>;
}

/// The reference scalar-diffraction simulator used by the CLI, benches, and
/// the WASM playback path.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarSimulator;

impl ScalarSimulator {
    /// Run the full pipeline, returning every intermediate stage for the
    /// five-view studio UI (ADR-260 product section).
    pub fn trace(
        &self,
        input: &InputImage,
        mask: &PhaseMask,
        config: &OpticalConfig,
    ) -> Result<SimulationTrace> {
        let incoming = OpticalField::from_image(input, config.width, config.height)?;
        let mut masked = incoming.clone();
        mask.apply(&mut masked)?;
        let propagated = propagate(&masked, config)?;
        let frame = capture(&propagated, config);
        Ok(SimulationTrace {
            incoming,
            masked,
            propagated,
            frame,
        })
    }
}

impl OpticalSimulator for ScalarSimulator {
    fn simulate(
        &self,
        input: &InputImage,
        mask: &PhaseMask,
        config: &OpticalConfig,
    ) -> Result<OpticalFrame> {
        Ok(self.trace(input, mask, config)?.frame)
    }
}

/// All intermediate stages of one simulation, for visualization and analysis.
#[derive(Clone, Debug)]
pub struct SimulationTrace {
    /// Field entering the optical system (image as amplitude).
    pub incoming: OpticalField,
    /// Field immediately after the learned phase mask.
    pub masked: OpticalField,
    /// Field at the detector plane after propagation.
    pub propagated: OpticalField,
    /// Recorded intensity frame.
    pub frame: OpticalFrame,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::input_frame_similarity;

    fn checker(n: usize) -> InputImage {
        let px: Vec<f32> = (0..n * n)
            .map(|i| {
                let (x, y) = (i % n, i / n);
                if (x / 4 + y / 4) % 2 == 0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        InputImage::from_norm_f32(n, n, px).unwrap()
    }

    #[test]
    fn simulation_is_deterministic() {
        let img = checker(32);
        let mask = PhaseMask::random(32, 32, 11);
        let cfg = OpticalConfig::demo(32, 32);
        let a = ScalarSimulator.simulate(&img, &mask, &cfg).unwrap();
        let b = ScalarSimulator.simulate(&img, &mask, &cfg).unwrap();
        assert_eq!(a.frame_hash, b.frame_hash);
    }

    #[test]
    fn learned_mask_changes_output() {
        let img = checker(32);
        let cfg = OpticalConfig::demo(32, 32);
        let flat = ScalarSimulator
            .simulate(&img, &PhaseMask::identity(32, 32), &cfg)
            .unwrap();
        let rnd = ScalarSimulator
            .simulate(&img, &PhaseMask::random(32, 32, 3), &cfg)
            .unwrap();
        assert_ne!(flat.frame_hash, rnd.frame_hash);
    }

    #[test]
    fn detector_frame_is_not_human_readable() {
        // ADR-260 acceptance: with a random/learned mask the sensor pattern
        // should not visually resemble the input image.
        let img = checker(32);
        let mut cfg = OpticalConfig::demo(32, 32);
        cfg.propagation = crate::config::PropagationMode::Fraunhofer;
        let frame = ScalarSimulator
            .simulate(&img, &PhaseMask::random(32, 32, 5), &cfg)
            .unwrap();
        let sim = input_frame_similarity(&img, &frame).abs();
        assert!(sim < 0.5, "frame too similar to input: {sim}");
    }
}
