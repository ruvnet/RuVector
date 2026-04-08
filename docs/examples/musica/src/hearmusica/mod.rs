//! HEARmusica -- a Rust port of the Tympan open-source hearing aid.
//!
//! Provides a block-based audio processing pipeline with pre-built presets
//! for common hearing aid configurations.

pub mod block;
pub mod compressor;
pub mod delay;
pub mod feedback;
pub mod filter;
pub mod gain;
pub mod limiter;
pub mod mixer;
pub mod presets;
pub mod separator_block;

pub use block::{AudioBlock, AudioProcessor, BlockMetadata};
pub use compressor::WDRCompressor;
pub use delay::DelayLine;
pub use feedback::FeedbackCanceller;
pub use filter::{BiquadFilter, FilterType};
pub use gain::GainProcessor;
pub use limiter::Limiter;
pub use mixer::Mixer;
pub use presets::*;
pub use separator_block::GraphSeparatorBlock;

/// Linear processing pipeline -- blocks execute in sequence.
pub struct Pipeline {
    blocks: Vec<Box<dyn AudioProcessor>>,
    sample_rate: f32,
    block_size: usize,
    prepared: bool,
}

impl Pipeline {
    /// Create a new pipeline with the given sample rate and block size.
    pub fn new(sample_rate: f32, block_size: usize) -> Self {
        Self {
            blocks: Vec::new(),
            sample_rate,
            block_size,
            prepared: false,
        }
    }

    /// Append a processing block to the pipeline.
    pub fn add(&mut self, block: Box<dyn AudioProcessor>) {
        self.prepared = false;
        self.blocks.push(block);
    }

    /// Prepare all blocks for processing.
    pub fn prepare(&mut self) {
        for block in &mut self.blocks {
            block.prepare(self.sample_rate, self.block_size);
        }
        self.prepared = true;
    }

    /// Process a single audio block through the entire pipeline in order.
    pub fn process_block(&mut self, block: &mut AudioBlock) {
        if !self.prepared {
            self.prepare();
        }
        for processor in &mut self.blocks {
            processor.process(block);
        }
    }

    /// Total latency introduced by the pipeline, in samples.
    pub fn total_latency_samples(&self) -> usize {
        self.blocks.iter().map(|b| b.latency_samples()).sum()
    }

    /// Total latency introduced by the pipeline, in milliseconds.
    pub fn total_latency_ms(&self) -> f32 {
        if self.sample_rate <= 0.0 {
            return 0.0;
        }
        self.total_latency_samples() as f32 / self.sample_rate * 1000.0
    }

    /// Return the name of each block in pipeline order.
    pub fn block_names(&self) -> Vec<&str> {
        self.blocks.iter().map(|b| b.name()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hearing_aid::Audiogram;

    #[test]
    fn pipeline_processes_block_without_panic() {
        let mut pipeline = Pipeline::new(16000.0, 128);
        pipeline.add(Box::new(GainProcessor::new(6.0)));
        pipeline.add(Box::new(Limiter::new(-1.0)));
        pipeline.prepare();

        let mut block = AudioBlock::new(128, 16000.0);
        // Fill with a simple sine tone.
        for i in 0..128 {
            let t = i as f32 / 16000.0;
            let s = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
            block.left[i] = s;
            block.right[i] = s;
        }

        pipeline.process_block(&mut block);
        // Should not panic and energy should be non-zero.
        assert!(block.energy() > 0.0);
    }

    #[test]
    fn pipeline_latency_sums_correctly() {
        let mut pipeline = Pipeline::new(16000.0, 128);
        pipeline.add(Box::new(DelayLine::new(2.0)));  // 2 ms
        pipeline.add(Box::new(DelayLine::new(4.0)));  // 4 ms
        pipeline.add(Box::new(GainProcessor::new(0.0)));
        pipeline.prepare();

        // Latency depends on sample rate and delay_ms; check it is non-zero.
        let total = pipeline.total_latency_samples();
        assert!(total > 0, "Pipeline should have non-zero latency");

        let ms = pipeline.total_latency_ms();
        assert!(ms > 0.0, "Expected positive latency, got {ms}");
    }

    #[test]
    fn block_names_match_added_blocks() {
        let mut pipeline = Pipeline::new(48000.0, 256);
        pipeline.add(Box::new(BiquadFilter::new(FilterType::HighPass, 100.0, 0.707)));
        pipeline.add(Box::new(WDRCompressor::new(-30.0, 2.0)));
        pipeline.add(Box::new(GainProcessor::new(10.0)));
        pipeline.add(Box::new(Limiter::new(-1.0)));

        let names = pipeline.block_names();
        assert_eq!(names, vec!["BiquadFilter", "WDRCompressor", "Gain", "Limiter"]);
    }

    #[test]
    fn standard_preset_creates_valid_pipeline() {
        let audiogram = Audiogram::default();
        let mut pipeline = standard_hearing_aid(&audiogram, 16000.0, 128);

        // Should have 4 blocks: filter, compressor, gain, limiter.
        assert_eq!(pipeline.block_names().len(), 4);

        // Process audio without panic.
        let mut block = AudioBlock::new(128, 16000.0);
        for i in 0..128 {
            let t = i as f32 / 16000.0;
            let s = (2.0 * std::f32::consts::PI * 300.0 * t).sin() * 0.3;
            block.left[i] = s;
            block.right[i] = s * 0.9;
        }
        pipeline.process_block(&mut block);

        // Output should still have signal.
        assert!(block.energy() > 0.0);
    }
}
