//! Core trait and data types for HEARmusica audio processing blocks.

/// Audio processing block -- fundamental unit of HEARmusica.
pub trait AudioProcessor: Send {
    /// Prepare the processor for a given sample rate and block size.
    fn prepare(&mut self, sample_rate: f32, block_size: usize);

    /// Process a block of audio in-place.
    fn process(&mut self, block: &mut AudioBlock);

    /// Release resources (optional).
    fn release(&mut self) {}

    /// Human-readable name of this processor.
    fn name(&self) -> &str;

    /// Latency introduced by this processor, in samples.
    fn latency_samples(&self) -> usize {
        0
    }
}

/// A stereo block of audio samples with metadata.
pub struct AudioBlock {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub sample_rate: f32,
    pub block_size: usize,
    pub metadata: BlockMetadata,
}

impl AudioBlock {
    /// Create a silent block of the given size.
    pub fn new(block_size: usize, sample_rate: f32) -> Self {
        Self {
            left: vec![0.0; block_size],
            right: vec![0.0; block_size],
            sample_rate,
            block_size,
            metadata: BlockMetadata::default(),
        }
    }

    /// Create a block from interleaved stereo data (L, R, L, R, ...).
    pub fn from_interleaved(data: &[f32], sample_rate: f32) -> Self {
        let block_size = data.len() / 2;
        let mut left = Vec::with_capacity(block_size);
        let mut right = Vec::with_capacity(block_size);
        for chunk in data.chunks(2) {
            left.push(chunk[0]);
            right.push(if chunk.len() > 1 { chunk[1] } else { 0.0 });
        }
        Self {
            left,
            right,
            sample_rate,
            block_size,
            metadata: BlockMetadata::default(),
        }
    }

    /// RMS energy across both channels.
    pub fn energy(&self) -> f32 {
        let n = self.left.len().max(1) as f32;
        let sum: f32 = self
            .left
            .iter()
            .chain(self.right.iter())
            .map(|s| s * s)
            .sum();
        (sum / (2.0 * n)).sqrt()
    }
}

/// Metadata carried alongside an audio block.
#[derive(Debug, Clone, Default)]
pub struct BlockMetadata {
    /// Monotonically increasing frame index.
    pub frame_index: u64,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
    /// Per-bin speech probability mask (optional).
    pub speech_mask: Option<Vec<f32>>,
    /// Per-bin noise power estimate (optional).
    pub noise_estimate: Option<Vec<f32>>,
}
