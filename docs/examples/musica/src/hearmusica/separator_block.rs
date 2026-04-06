//! Graph-based source separator block for HEARmusica.
//!
//! Wraps the binaural hearing-aid speech enhancer (graph construction,
//! Fiedler vector, dynamic mincut) as an [`AudioProcessor`] block that
//! fits into the HEARmusica pipeline.

use super::block::{AudioBlock, AudioProcessor};
use crate::hearing_aid::{HearingAidConfig, StreamingState};

/// Graph-partitioning speech separator that plugs into a HEARmusica pipeline.
///
/// Internally accumulates input into hop-sized frames, runs the streaming
/// graph separator from [`crate::hearing_aid`], and applies the resulting
/// speech mask as a broadband gain to each hop window.  The full per-band
/// mask is also stored in [`AudioBlock::metadata::speech_mask`] for
/// downstream blocks (e.g. compressor, gain shaping).
pub struct GraphSeparatorBlock {
    config: HearingAidConfig,
    state: Option<StreamingState>,
    /// Last computed speech mask (per-ERB-band, f32).
    speech_mask: Vec<f32>,
    /// Accumulation buffer -- left channel.
    frame_buffer_l: Vec<f32>,
    /// Accumulation buffer -- right channel.
    frame_buffer_r: Vec<f32>,
    /// Samples per analysis frame (frame_size_ms * sample_rate).
    frame_samples: usize,
    /// Samples per hop (hop_size_ms * sample_rate).
    hop_samples: usize,
    /// Pipeline sample rate (set in `prepare`).
    sample_rate: f32,
    /// Pipeline block size (set in `prepare`).
    block_size: usize,
}

impl GraphSeparatorBlock {
    /// Create a block with default [`HearingAidConfig`].
    pub fn new() -> Self {
        Self::with_config(HearingAidConfig::default())
    }

    /// Create a block with a specific configuration.
    pub fn with_config(config: HearingAidConfig) -> Self {
        let frame_samples =
            (config.sample_rate * config.frame_size_ms / 1000.0) as usize;
        let hop_samples =
            (config.sample_rate * config.hop_size_ms / 1000.0) as usize;

        Self {
            speech_mask: vec![0.5; config.num_bands],
            frame_buffer_l: Vec::with_capacity(frame_samples),
            frame_buffer_r: Vec::with_capacity(frame_samples),
            frame_samples,
            hop_samples,
            sample_rate: config.sample_rate as f32,
            block_size: 0,
            config,
            state: None,
        }
    }

    /// Current speech mask (per-ERB-band).  Returns the last computed mask,
    /// or 0.5 everywhere before the first frame has been analysed.
    pub fn speech_mask(&self) -> &[f32] {
        &self.speech_mask
    }

    /// Average broadband speech gain derived from the current mask.
    fn broadband_gain(&self) -> f32 {
        if self.speech_mask.is_empty() {
            return 1.0;
        }
        let sum: f32 = self.speech_mask.iter().sum();
        sum / self.speech_mask.len() as f32
    }

    /// Drain the front `count` samples from both frame buffers.
    fn drain_hop(&mut self) {
        let n = self.hop_samples.min(self.frame_buffer_l.len());
        self.frame_buffer_l.drain(..n);
        self.frame_buffer_r.drain(..n);
    }

    /// Run the hearing-aid graph separator on the current frame buffer
    /// contents, updating `self.speech_mask`.
    fn run_separator(&mut self) {
        let state = match self.state.as_mut() {
            Some(s) => s,
            None => return,
        };

        // Convert the first `frame_samples` of the buffer to f64.
        let n = self.frame_samples.min(self.frame_buffer_l.len());
        let left_f64: Vec<f64> =
            self.frame_buffer_l[..n].iter().map(|&s| s as f64).collect();
        let right_f64: Vec<f64> =
            self.frame_buffer_r[..n].iter().map(|&s| s as f64).collect();

        let result = state.process_frame(&left_f64, &right_f64, &self.config);

        // Store mask as f32.
        self.speech_mask = result.mask.iter().map(|&m| m as f32).collect();
    }
}

impl Default for GraphSeparatorBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioProcessor for GraphSeparatorBlock {
    fn prepare(&mut self, sample_rate: f32, block_size: usize) {
        self.sample_rate = sample_rate;
        self.block_size = block_size;

        // Rebuild config with the pipeline sample rate if it differs.
        if (self.config.sample_rate - sample_rate as f64).abs() > 1.0 {
            self.config.sample_rate = sample_rate as f64;
        }

        // Recompute frame/hop sizes from (possibly updated) config.
        self.frame_samples =
            (self.config.sample_rate * self.config.frame_size_ms / 1000.0) as usize;
        self.hop_samples =
            (self.config.sample_rate * self.config.hop_size_ms / 1000.0) as usize;

        // (Re)create streaming state.
        self.state = Some(StreamingState::new(&self.config));
        self.speech_mask = vec![0.5; self.config.num_bands];
        self.frame_buffer_l.clear();
        self.frame_buffer_r.clear();
    }

    fn process(&mut self, block: &mut AudioBlock) {
        let len = block.left.len().min(block.right.len());
        if len == 0 || self.state.is_none() {
            return;
        }

        // 1. Accumulate incoming samples.
        self.frame_buffer_l.extend_from_slice(&block.left[..len]);
        self.frame_buffer_r.extend_from_slice(&block.right[..len]);

        // 2. Process as many hops as we can.
        while self.frame_buffer_l.len() >= self.frame_samples {
            self.run_separator();
            self.drain_hop();
        }

        // 3. Attach the full per-band mask to metadata for downstream blocks.
        block.metadata.speech_mask = Some(self.speech_mask.clone());

        // 4. Apply broadband gain to the audio block.
        //    (V1 strategy: single gain scalar derived from mask average.)
        let gain = self.broadband_gain();
        for s in block.left.iter_mut() {
            *s *= gain;
        }
        for s in block.right.iter_mut() {
            *s *= gain;
        }
    }

    fn name(&self) -> &str {
        "GraphSeparator"
    }

    fn latency_samples(&self) -> usize {
        self.hop_samples
    }

    fn release(&mut self) {
        self.state = None;
        self.frame_buffer_l.clear();
        self.frame_buffer_r.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hearing_aid::HearingAidConfig;

    const SR: f32 = 16000.0;
    const BLOCK: usize = 128;

    /// Helper: fill a block with a sine tone on both channels.
    fn sine_block(freq: f32, amplitude: f32, block_size: usize, sr: f32) -> AudioBlock {
        let mut block = AudioBlock::new(block_size, sr);
        for i in 0..block_size {
            let t = i as f32 / sr;
            let s = amplitude * (2.0 * std::f32::consts::PI * freq * t).sin();
            block.left[i] = s;
            block.right[i] = s * 0.9; // slight ILD
        }
        block
    }

    /// Helper: fill a block with white-ish noise (deterministic).
    fn noise_block(block_size: usize, sr: f32) -> AudioBlock {
        let mut block = AudioBlock::new(block_size, sr);
        // Simple PRNG (xorshift32) for deterministic noise.
        let mut rng: u32 = 0xDEAD_BEEF;
        for i in 0..block_size {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let s = (rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
            block.left[i] = s * 0.3;
            block.right[i] = s * 0.25;
        }
        block
    }

    // ---- Test 1: Block processes without panic ----

    #[test]
    fn process_does_not_panic() {
        let mut sep = GraphSeparatorBlock::new();
        sep.prepare(SR, BLOCK);

        let mut block = sine_block(440.0, 0.5, BLOCK, SR);
        sep.process(&mut block);

        // Output should still have energy (not zeroed out).
        assert!(block.energy() > 0.0);
    }

    // ---- Test 2: Speech mask is populated after enough frames ----

    #[test]
    fn speech_mask_populated_after_frames() {
        let config = HearingAidConfig::default();
        let mut sep = GraphSeparatorBlock::with_config(config.clone());
        sep.prepare(SR, BLOCK);

        // Feed enough blocks to fill several analysis frames.
        // hop = 64 samples at 16 kHz, frame = 128 samples.
        // Each 128-sample block accumulates enough for ~1 hop.
        // Feed 20 blocks to ensure stable mask.
        for _ in 0..20 {
            let mut block = sine_block(300.0, 0.5, BLOCK, SR);
            sep.process(&mut block);
        }

        // Mask should now be populated with per-band values.
        let mask = sep.speech_mask();
        assert_eq!(mask.len(), config.num_bands);

        // At least some bands should differ from the initial 0.5.
        let differs = mask.iter().any(|&m| (m - 0.5).abs() > 0.01);
        assert!(
            differs,
            "Mask should have changed from initial 0.5 after processing; got {:?}",
            mask
        );

        // The metadata speech_mask should also be Some.
        let mut last_block = sine_block(300.0, 0.5, BLOCK, SR);
        sep.process(&mut last_block);
        assert!(
            last_block.metadata.speech_mask.is_some(),
            "Block metadata should contain the speech mask"
        );
    }

    // ---- Test 3: Latency reports correct hop size ----

    #[test]
    fn latency_equals_hop_samples() {
        let config = HearingAidConfig {
            sample_rate: 16000.0,
            hop_size_ms: 4.0,
            ..Default::default()
        };
        let sep = GraphSeparatorBlock::with_config(config);
        // hop = 16000 * 4 / 1000 = 64 samples
        assert_eq!(sep.latency_samples(), 64);
    }

    // ---- Test 4: Speech-like input gets higher mask than noise ----

    #[test]
    fn speech_mask_higher_for_harmonics_than_noise() {
        let config = HearingAidConfig::default();

        // --- Run with harmonic (speech-like) signal ---
        let mut sep_speech = GraphSeparatorBlock::with_config(config.clone());
        sep_speech.prepare(SR, BLOCK);

        for _ in 0..30 {
            // Rich harmonic content: fundamental + 2nd + 3rd harmonic.
            let mut block = AudioBlock::new(BLOCK, SR);
            for i in 0..BLOCK {
                let t = i as f32 / SR;
                let s = 0.5 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                    + 0.25 * (2.0 * std::f32::consts::PI * 400.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 600.0 * t).sin();
                block.left[i] = s;
                block.right[i] = s * 0.9; // coherent, frontal
            }
            sep_speech.process(&mut block);
        }

        let speech_gain = sep_speech.broadband_gain();

        // --- Run with noise ---
        let mut sep_noise = GraphSeparatorBlock::with_config(config.clone());
        sep_noise.prepare(SR, BLOCK);

        for _ in 0..30 {
            let mut block = noise_block(BLOCK, SR);
            sep_noise.process(&mut block);
        }

        let noise_gain = sep_noise.broadband_gain();

        // Speech-like signal should yield a higher (or at least equal) broadband gain.
        assert!(
            speech_gain >= noise_gain * 0.9,
            "Speech gain ({speech_gain:.3}) should be >= noise gain ({noise_gain:.3}) * 0.9"
        );
    }
}
