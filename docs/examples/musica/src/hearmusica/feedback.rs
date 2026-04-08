//! Adaptive feedback canceller using Normalized LMS (NLMS) algorithm.
//!
//! Port of Tympan's `AudioEffectFeedbackCancel_F32`. Suppresses acoustic
//! feedback (whistling) by estimating and subtracting the feedback path
//! contribution from the microphone signal in real time.

use super::block::{AudioBlock, AudioProcessor};

/// Adaptive feedback canceller using the Normalized LMS algorithm.
///
/// The canceller maintains an adaptive FIR filter that models the acoustic
/// feedback path. Each sample, it predicts the feedback component and
/// subtracts it from the input, then updates filter coefficients to minimize
/// the residual error.
pub struct FeedbackCanceller {
    /// Number of adaptive filter taps.
    filter_length: usize,
    /// NLMS step size (controls adaptation speed vs. stability).
    mu: f32,
    /// Regularization constant to prevent division by zero.
    regularization: f32,
    /// Adaptive filter weights (FIR coefficients).
    coefficients: Vec<f32>,
    /// Circular buffer storing past output samples for the reference signal.
    delay_buffer: Vec<f32>,
    /// Current write position in the circular buffer.
    buffer_pos: usize,
    /// Feedback path delay in samples (distance mic <-> speaker).
    feedback_delay: usize,
    /// Configured sample rate.
    sample_rate: f32,
    /// Configured block size.
    block_size: usize,
}

impl FeedbackCanceller {
    /// Create a new feedback canceller.
    ///
    /// # Arguments
    /// * `filter_length` - Number of adaptive filter taps (default: 128).
    /// * `mu` - NLMS step size (default: 0.01). Smaller = more stable, larger = faster tracking.
    pub fn new(filter_length: usize, mu: f32) -> Self {
        let buf_len = filter_length + 256; // extra room for feedback delay
        Self {
            filter_length,
            mu,
            regularization: 1e-6,
            coefficients: vec![0.0; filter_length],
            delay_buffer: vec![0.0; buf_len],
            buffer_pos: 0,
            feedback_delay: 0,
            sample_rate: 16000.0,
            block_size: 128,
        }
    }

    /// Set the feedback path delay in samples.
    pub fn set_feedback_delay(&mut self, delay_samples: usize) {
        self.feedback_delay = delay_samples;
    }

    /// Set the feedback path delay in milliseconds.
    pub fn set_feedback_delay_ms(&mut self, delay_ms: f32) {
        self.feedback_delay = (delay_ms * self.sample_rate / 1000.0) as usize;
    }

    /// Set the regularization constant.
    pub fn set_regularization(&mut self, reg: f32) {
        self.regularization = reg;
    }

    /// Process a single channel in-place using NLMS.
    ///
    /// The algorithm for each sample:
    /// 1. Form reference vector x from delay buffer: past output samples offset by feedback delay.
    /// 2. Compute estimated feedback: y_hat = dot(coefficients, x).
    /// 3. Compute error: e = input_sample - y_hat.
    /// 4. Update coefficients: w[i] += mu * e * x[i] / (||x||^2 + regularization).
    /// 5. Store output (error) in the delay buffer for future reference.
    fn process_channel(&mut self, samples: &mut [f32]) {
        let buf_len = self.delay_buffer.len();
        let d = self.feedback_delay;
        let l = self.filter_length;

        for sample in samples.iter_mut() {
            let input = *sample;

            // Step 1: Form reference vector and compute y_hat + x_norm simultaneously
            let mut y_hat: f32 = 0.0;
            let mut x_norm_sq: f32 = 0.0;

            for i in 0..l {
                // x[i] = delay_buffer[buffer_pos - d - i - 1], wrapping
                let idx = (self.buffer_pos + buf_len - d - i - 1) % buf_len;
                let xi = self.delay_buffer[idx];
                y_hat += self.coefficients[i] * xi;
                x_norm_sq += xi * xi;
            }

            // Step 3: Compute error (cleaned signal)
            let error = input - y_hat;

            // Step 4: Update coefficients (NLMS)
            let norm_factor = self.mu / (x_norm_sq + self.regularization);
            for i in 0..l {
                let idx = (self.buffer_pos + buf_len - d - i - 1) % buf_len;
                let xi = self.delay_buffer[idx];
                self.coefficients[i] += norm_factor * error * xi;
            }

            // Step 5: Store output in delay buffer and advance
            self.delay_buffer[self.buffer_pos] = error;
            self.buffer_pos = (self.buffer_pos + 1) % buf_len;

            *sample = error;
        }
    }

    /// Reset adaptive filter state (coefficients and delay buffer).
    pub fn reset(&mut self) {
        self.coefficients.iter_mut().for_each(|c| *c = 0.0);
        self.delay_buffer.iter_mut().for_each(|s| *s = 0.0);
        self.buffer_pos = 0;
    }

    /// Get a snapshot of the current filter coefficients.
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }
}

impl AudioProcessor for FeedbackCanceller {
    fn prepare(&mut self, sample_rate: f32, block_size: usize) {
        self.sample_rate = sample_rate;
        self.block_size = block_size;
        // Resize buffer to accommodate filter length + max feedback delay + margin
        let buf_len = self.filter_length + self.feedback_delay + block_size + 64;
        self.delay_buffer.resize(buf_len, 0.0);
        self.delay_buffer.iter_mut().for_each(|s| *s = 0.0);
        self.buffer_pos = 0;
        self.coefficients.iter_mut().for_each(|c| *c = 0.0);
    }

    fn process(&mut self, block: &mut AudioBlock) {
        // Process left channel with NLMS
        self.process_channel(&mut block.left);

        // Save state, then process right channel independently
        // For stereo hearing aids, each ear has its own feedback path,
        // but we share the same adaptive filter for simplicity here.
        // A full implementation would maintain separate state per channel.
        self.process_channel(&mut block.right);
    }

    fn name(&self) -> &str {
        "FeedbackCanceller"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compute RMS of a slice.
    fn rms(data: &[f32]) -> f32 {
        let sum: f32 = data.iter().map(|x| x * x).sum();
        (sum / data.len() as f32).sqrt()
    }

    #[test]
    fn no_feedback_passthrough() {
        // With no feedback (clean input, nothing in delay buffer),
        // the output should essentially equal the input because
        // y_hat ~= 0 when coefficients are zero and buffer is silent.
        let mut fc = FeedbackCanceller::new(64, 0.01);
        fc.prepare(16000.0, 256);

        // Generate a simple sine wave input
        let freq = 440.0;
        let sr = 16000.0;
        let len = 256;
        let input: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5)
            .collect();

        let mut block = AudioBlock::new(len, sr);
        block.left = input.clone();
        block.right = input.clone();

        fc.process(&mut block);

        // Output should be very close to input (coefficients near zero)
        let diff_rms: f32 = block
            .left
            .iter()
            .zip(input.iter())
            .map(|(o, i)| (o - i).powi(2))
            .sum::<f32>()
            / len as f32;
        let diff_rms = diff_rms.sqrt();

        assert!(
            diff_rms < 0.2,
            "Without feedback, output should match input. Diff RMS = {}",
            diff_rms
        );

        // Coefficients should remain near zero
        let coeff_energy: f32 = fc.coefficients().iter().map(|c| c * c).sum();
        assert!(
            coeff_energy < 0.5,
            "Coefficients should stay near zero without feedback. Energy = {}",
            coeff_energy
        );
    }

    #[test]
    fn cancels_synthetic_feedback() {
        // Simulate feedback: a delayed, scaled copy of the output is added to input.
        // The canceller should learn to remove it over time.
        let filter_len = 32;
        let feedback_delay = 5;
        let feedback_gain = 0.4; // Feedback path gain
        let mu = 0.05;

        let mut fc = FeedbackCanceller::new(filter_len, mu);
        fc.set_feedback_delay(feedback_delay);
        fc.prepare(16000.0, 1);

        let num_samples = 2000;
        let mut output_history: Vec<f32> = vec![0.0; num_samples];
        let mut error_history: Vec<f32> = Vec::with_capacity(num_samples);

        // Source signal: white noise
        let source: Vec<f32> = (0..num_samples)
            .map(|i| {
                // Simple pseudo-random using a hash-like function
                let x = (i as f32 * 0.1234).sin() * 43758.5453;
                (x - x.floor()) * 2.0 - 1.0
            })
            .collect();

        for n in 0..num_samples {
            // Feedback: delayed output * gain
            let feedback = if n >= feedback_delay {
                output_history[n - feedback_delay] * feedback_gain
            } else {
                0.0
            };

            // Mic picks up source + feedback
            let mic_input = source[n] + feedback;

            // Process one sample at a time
            let mut block = AudioBlock::new(1, 16000.0);
            block.left[0] = mic_input;
            block.right[0] = mic_input;
            fc.process(&mut block);

            let output = block.left[0];
            output_history[n] = output;
            error_history.push((output - source[n]).abs());
        }

        // Compare early errors (before adaptation) vs late errors (after adaptation)
        let early_error = rms(&error_history[100..300]);
        let late_error = rms(&error_history[1500..2000]);

        assert!(
            late_error < early_error,
            "Feedback canceller should reduce error over time. Early: {}, Late: {}",
            early_error,
            late_error
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut fc = FeedbackCanceller::new(32, 0.01);
        fc.prepare(16000.0, 64);

        // Process some data to build up state
        let mut block = AudioBlock::new(64, 16000.0);
        block.left = vec![0.5; 64];
        fc.process(&mut block);

        // Reset
        fc.reset();

        let coeff_energy: f32 = fc.coefficients().iter().map(|c| c * c).sum();
        assert_eq!(coeff_energy, 0.0, "After reset, coefficients should be zero");
    }
}
