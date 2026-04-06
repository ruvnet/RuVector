//! Weighted stereo mixer / gain stage for pipeline use.
//!
//! In a linear hearing-aid pipeline, the Mixer applies a configurable weight
//! (gain) to both channels. Can also operate with independent per-channel
//! weights for stereo balance adjustment.

use super::block::{AudioBlock, AudioProcessor};

/// Weighted stereo mixer.
///
/// For pipeline use, applies a single weight to both channels.
/// Can also be configured with independent left/right gains.
pub struct Mixer {
    /// Weight applied to both channels (pipeline mode).
    weight: f32,
    /// Optional per-channel gains (overrides weight if set).
    left_gain: f32,
    right_gain: f32,
    /// Whether per-channel mode is active.
    per_channel: bool,
}

impl Mixer {
    /// Create a mixer with a uniform weight applied to both channels.
    pub fn new(weight: f32) -> Self {
        Self {
            weight,
            left_gain: weight,
            right_gain: weight,
            per_channel: false,
        }
    }

    /// Create a mixer with independent left/right gains.
    pub fn with_stereo_gains(left_gain: f32, right_gain: f32) -> Self {
        Self {
            weight: (left_gain + right_gain) * 0.5,
            left_gain,
            right_gain,
            per_channel: true,
        }
    }

    /// Unity-gain mixer (pass-through).
    pub fn unity() -> Self {
        Self::new(1.0)
    }

    /// Update the uniform weight.
    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight;
        if !self.per_channel {
            self.left_gain = weight;
            self.right_gain = weight;
        }
    }

    /// Get the current weight.
    pub fn weight(&self) -> f32 {
        self.weight
    }
}

impl AudioProcessor for Mixer {
    fn prepare(&mut self, _sample_rate: f32, _block_size: usize) {
        // No state to initialize.
    }

    fn process(&mut self, block: &mut AudioBlock) {
        let lg = if self.per_channel { self.left_gain } else { self.weight };
        let rg = if self.per_channel { self.right_gain } else { self.weight };

        for s in block.left.iter_mut() {
            *s *= lg;
        }
        for s in block.right.iter_mut() {
            *s *= rg;
        }
    }

    fn name(&self) -> &str {
        "Mixer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_halves_amplitude() {
        let mut mixer = Mixer::new(0.5);
        mixer.prepare(16000.0, 8);

        let input: Vec<f32> = vec![1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.8];
        let expected: Vec<f32> = input.iter().map(|x| x * 0.5).collect();

        let mut block = AudioBlock::new(8, 16000.0);
        block.left = input.clone();
        block.right = input.clone();

        mixer.process(&mut block);

        for (i, (out, exp)) in block.left.iter().zip(expected.iter()).enumerate() {
            assert!(
                (out - exp).abs() < 1e-7,
                "Left[{}]: expected {}, got {}",
                i,
                exp,
                out
            );
        }
        for (i, (out, exp)) in block.right.iter().zip(expected.iter()).enumerate() {
            assert!(
                (out - exp).abs() < 1e-7,
                "Right[{}]: expected {}, got {}",
                i,
                exp,
                out
            );
        }
    }

    #[test]
    fn unity_is_passthrough() {
        let mut mixer = Mixer::unity();
        mixer.prepare(16000.0, 4);

        let input = vec![0.1, 0.2, 0.3, 0.4];
        let mut block = AudioBlock::new(4, 16000.0);
        block.left = input.clone();
        block.right = input.clone();

        mixer.process(&mut block);

        assert_eq!(block.left, input);
        assert_eq!(block.right, input);
    }

    #[test]
    fn stereo_gains_apply_independently() {
        let mut mixer = Mixer::with_stereo_gains(0.5, 2.0);
        mixer.prepare(16000.0, 3);

        let mut block = AudioBlock::new(3, 16000.0);
        block.left = vec![1.0, 1.0, 1.0];
        block.right = vec![1.0, 1.0, 1.0];

        mixer.process(&mut block);

        assert_eq!(block.left, vec![0.5, 0.5, 0.5]);
        assert_eq!(block.right, vec![2.0, 2.0, 2.0]);
    }
}
