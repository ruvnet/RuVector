//! Sample-accurate circular buffer delay line.
//!
//! Provides a fixed-length delay for latency alignment between processing
//! branches or for use as a building block in feedback cancellation.

use super::block::{AudioBlock, AudioProcessor};

/// Sample-accurate circular buffer delay line.
pub struct DelayLine {
    /// Delay in samples (computed from delay_ms and sample_rate).
    delay_samples: usize,
    /// Requested delay in milliseconds.
    delay_ms: f32,
    /// Left channel circular buffer.
    buffer_l: Vec<f32>,
    /// Right channel circular buffer.
    buffer_r: Vec<f32>,
    /// Current write position in the circular buffers.
    write_pos: usize,
    /// Configured sample rate.
    sample_rate: f32,
}

impl DelayLine {
    /// Create a new delay line with the given delay in milliseconds.
    ///
    /// Actual sample count is computed during `prepare()` based on sample rate.
    pub fn new(delay_ms: f32) -> Self {
        Self {
            delay_samples: 0,
            delay_ms,
            buffer_l: Vec::new(),
            buffer_r: Vec::new(),
            write_pos: 0,
            sample_rate: 16000.0,
        }
    }

    /// Create a delay line specifying delay directly in samples.
    pub fn from_samples(delay_samples: usize) -> Self {
        Self {
            delay_samples,
            delay_ms: 0.0,
            buffer_l: vec![0.0; delay_samples],
            buffer_r: vec![0.0; delay_samples],
            write_pos: 0,
            sample_rate: 16000.0,
        }
    }

    /// Update the delay time in milliseconds. Takes effect at next `prepare()`.
    pub fn set_delay_ms(&mut self, ms: f32) {
        self.delay_ms = ms;
    }

    /// Process a single channel through the circular buffer delay.
    fn process_channel(buffer: &mut Vec<f32>, write_pos: &mut usize, samples: &mut [f32], delay: usize) {
        if delay == 0 {
            return;
        }
        let buf_len = buffer.len();
        for sample in samples.iter_mut() {
            let input = *sample;
            // Read from delay position behind write
            let read_pos = (*write_pos + buf_len - delay) % buf_len;
            *sample = buffer[read_pos];
            // Write current input
            buffer[*write_pos] = input;
            *write_pos = (*write_pos + 1) % buf_len;
        }
    }
}

impl AudioProcessor for DelayLine {
    fn prepare(&mut self, sample_rate: f32, _block_size: usize) {
        self.sample_rate = sample_rate;
        if self.delay_ms > 0.0 {
            self.delay_samples = (self.delay_ms * sample_rate / 1000.0).round() as usize;
        }
        if self.delay_samples > 0 {
            self.buffer_l = vec![0.0; self.delay_samples];
            self.buffer_r = vec![0.0; self.delay_samples];
            self.write_pos = 0;
        }
    }

    fn process(&mut self, block: &mut AudioBlock) {
        if self.delay_samples == 0 {
            return; // Pass-through for zero delay
        }
        let delay = self.delay_samples;
        // Process left channel
        let mut wp = self.write_pos;
        Self::process_channel(&mut self.buffer_l, &mut wp, &mut block.left, delay);
        // Process right channel (use same write_pos progression)
        let mut wp_r = self.write_pos;
        Self::process_channel(&mut self.buffer_r, &mut wp_r, &mut block.right, delay);
        // Advance write position once (both channels are in lockstep)
        self.write_pos = wp;
    }

    fn name(&self) -> &str {
        "DelayLine"
    }

    fn latency_samples(&self) -> usize {
        self.delay_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_shifts_output_by_n_samples() {
        let delay_samples = 10;
        let mut dl = DelayLine::from_samples(delay_samples);
        dl.prepare(16000.0, 64);

        let len = 64;
        let input: Vec<f32> = (0..len).map(|i| (i + 1) as f32).collect();

        let mut block = AudioBlock::new(len, 16000.0);
        block.left = input.clone();
        block.right = input.clone();

        dl.process(&mut block);

        // First `delay_samples` outputs should be zero (silence from buffer init)
        for i in 0..delay_samples {
            assert_eq!(
                block.left[i], 0.0,
                "Sample {} should be zero (delayed silence)",
                i
            );
        }

        // After delay, output should match input shifted by delay_samples
        for i in delay_samples..len {
            let expected = input[i - delay_samples];
            assert_eq!(
                block.left[i], expected,
                "Sample {} should be input[{}] = {}",
                i,
                i - delay_samples,
                expected
            );
        }
    }

    #[test]
    fn zero_delay_is_passthrough() {
        let mut dl = DelayLine::new(0.0);
        dl.prepare(16000.0, 32);

        let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let mut block = AudioBlock::new(32, 16000.0);
        block.left = input.clone();
        block.right = input.clone();

        dl.process(&mut block);

        assert_eq!(block.left, input, "Zero delay should pass through unchanged");
        assert_eq!(block.right, input, "Zero delay should pass through unchanged");
    }

    #[test]
    fn delay_from_ms_computes_correct_samples() {
        let mut dl = DelayLine::new(10.0); // 10 ms
        dl.prepare(16000.0, 64); // At 16kHz, 10ms = 160 samples
        assert_eq!(dl.delay_samples, 160);
        assert_eq!(dl.latency_samples(), 160);
    }
}
