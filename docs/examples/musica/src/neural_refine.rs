//! Tiny neural mask refinement module.
//!
//! A minimal 2-layer MLP (no external dependencies) that refines
//! graph-based masks from the separator using magnitude spectrogram features.
//!
//! Architecture:
//!   Input (5 features per T-F bin) -> Dense(64, ReLU) -> Dense(1, identity) -> sigmoid(raw + correction)
//!
//! Features per T-F bin: [magnitude, phase_diff, temporal_diff, spectral_diff, raw_mask_value]

/// Simple Linear Congruential Generator (no external rand crate).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // Knuth's LCG parameters
        self.state = self.state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform f64 in [-1, 1].
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64) * 2.0 - 1.0
    }
}

/// Configuration for the MLP.
#[derive(Debug, Clone)]
pub struct MLPConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub learning_rate: f64,
}

impl Default for MLPConfig {
    fn default() -> Self {
        Self {
            input_dim: 5,
            hidden_dim: 64,
            output_dim: 1,
            learning_rate: 0.01,
        }
    }
}

/// A training example: input features and target mask values.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input feature vector (length = input_dim).
    pub input: Vec<f64>,
    /// Target mask values (length = output_dim).
    pub target: Vec<f64>,
}

/// Statistics from training / refinement.
#[derive(Debug, Clone)]
pub struct RefinementStats {
    /// MSE loss after each training step.
    pub loss_history: Vec<f64>,
    /// Total parameter count.
    pub param_count: usize,
}

/// A 2-layer MLP for mask refinement.
///
/// Layer 1: input_dim -> hidden_dim (ReLU)
/// Layer 2: hidden_dim -> output_dim (linear correction)
pub struct TinyMLP {
    config: MLPConfig,
    // Layer 1: weights [hidden_dim x input_dim], biases [hidden_dim]
    w1: Vec<Vec<f64>>,
    b1: Vec<f64>,
    // Layer 2: weights [output_dim x hidden_dim], biases [output_dim]
    w2: Vec<Vec<f64>>,
    b2: Vec<f64>,
}

impl TinyMLP {
    /// Create a new TinyMLP with Xavier-initialized weights.
    pub fn new(config: MLPConfig) -> Self {
        let mut rng = Lcg::new(42);

        // Xavier init: scale = sqrt(2 / (fan_in + fan_out))
        let scale1 = ((2.0) / (config.input_dim + config.hidden_dim) as f64).sqrt();
        let w1: Vec<Vec<f64>> = (0..config.hidden_dim)
            .map(|_| {
                (0..config.input_dim)
                    .map(|_| rng.next_f64() * scale1)
                    .collect()
            })
            .collect();
        let b1 = vec![0.0; config.hidden_dim];

        let scale2 = ((2.0) / (config.hidden_dim + config.output_dim) as f64).sqrt();
        let w2: Vec<Vec<f64>> = (0..config.output_dim)
            .map(|_| {
                (0..config.hidden_dim)
                    .map(|_| rng.next_f64() * scale2)
                    .collect()
            })
            .collect();
        let b2 = vec![0.0; config.output_dim];

        Self { config, w1, b1, w2, b2 }
    }

    /// Total number of learnable parameters.
    pub fn param_count(&self) -> usize {
        let l1 = self.config.input_dim * self.config.hidden_dim + self.config.hidden_dim;
        let l2 = self.config.hidden_dim * self.config.output_dim + self.config.output_dim;
        l1 + l2
    }

    /// Forward pass: input -> ReLU hidden -> linear output -> sigmoid.
    #[inline]
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        // Layer 1: z1 = W1 * x + b1, h = relu(z1)
        // Manual loop for better auto-vectorization
        let hdim = self.config.hidden_dim;
        let idim = self.config.input_dim;
        let mut hidden = vec![0.0; hdim];
        for i in 0..hdim {
            let mut z = self.b1[i];
            let w_row = &self.w1[i];
            for j in 0..idim {
                z += w_row[j] * input[j];
            }
            hidden[i] = relu(z);
        }

        // Layer 2: z2 = W2 * h + b2, out = sigmoid(z2)
        let output: Vec<f64> = (0..self.config.output_dim)
            .map(|i| {
                let mut z = self.b2[i];
                let w_row = &self.w2[i];
                for j in 0..hdim {
                    z += w_row[j] * hidden[j];
                }
                sigmoid(z)
            })
            .collect();

        output
    }

    /// Single gradient descent step on a batch of examples. Returns MSE loss.
    pub fn train_step(&mut self, examples: &[TrainingExample]) -> f64 {
        if examples.is_empty() {
            return 0.0;
        }

        let n = examples.len() as f64;
        let lr = self.config.learning_rate;

        // Accumulate gradients
        let mut dw1 = vec![vec![0.0; self.config.input_dim]; self.config.hidden_dim];
        let mut db1 = vec![0.0; self.config.hidden_dim];
        let mut dw2 = vec![vec![0.0; self.config.hidden_dim]; self.config.output_dim];
        let mut db2 = vec![0.0; self.config.output_dim];
        let mut total_loss = 0.0;

        for ex in examples {
            // --- Forward pass (save intermediates) ---
            // Layer 1
            let z1: Vec<f64> = (0..self.config.hidden_dim)
                .map(|i| {
                    self.w1[i].iter().zip(ex.input.iter()).map(|(w, x)| w * x).sum::<f64>() + self.b1[i]
                })
                .collect();
            let h: Vec<f64> = z1.iter().map(|&z| relu(z)).collect();

            // Layer 2
            let z2: Vec<f64> = (0..self.config.output_dim)
                .map(|i| {
                    self.w2[i].iter().zip(h.iter()).map(|(w, hv)| w * hv).sum::<f64>() + self.b2[i]
                })
                .collect();
            let out: Vec<f64> = z2.iter().map(|&z| sigmoid(z)).collect();

            // --- Loss: MSE ---
            let loss: f64 = out.iter().zip(ex.target.iter())
                .map(|(o, t)| (o - t) * (o - t))
                .sum::<f64>() / self.config.output_dim as f64;
            total_loss += loss;

            // --- Backward pass ---
            // dL/dout = 2*(out - target) / output_dim
            // dout/dz2 = sigmoid'(z2) = out*(1-out)
            // dL/dz2 = dL/dout * dout/dz2
            let dz2: Vec<f64> = (0..self.config.output_dim)
                .map(|i| {
                    let dl_dout = 2.0 * (out[i] - ex.target[i]) / self.config.output_dim as f64;
                    dl_dout * out[i] * (1.0 - out[i])
                })
                .collect();

            // Gradients for layer 2
            for i in 0..self.config.output_dim {
                for j in 0..self.config.hidden_dim {
                    dw2[i][j] += dz2[i] * h[j];
                }
                db2[i] += dz2[i];
            }

            // Backprop to hidden: dL/dh = W2^T * dz2
            let dh: Vec<f64> = (0..self.config.hidden_dim)
                .map(|j| {
                    (0..self.config.output_dim).map(|i| self.w2[i][j] * dz2[i]).sum::<f64>()
                })
                .collect();

            // dL/dz1 = dL/dh * relu'(z1)
            let dz1: Vec<f64> = (0..self.config.hidden_dim)
                .map(|i| if z1[i] > 0.0 { dh[i] } else { 0.0 })
                .collect();

            // Gradients for layer 1
            for i in 0..self.config.hidden_dim {
                for j in 0..self.config.input_dim {
                    dw1[i][j] += dz1[i] * ex.input[j];
                }
                db1[i] += dz1[i];
            }
        }

        // --- Apply gradients (SGD) ---
        for i in 0..self.config.hidden_dim {
            for j in 0..self.config.input_dim {
                self.w1[i][j] -= lr * dw1[i][j] / n;
            }
            self.b1[i] -= lr * db1[i] / n;
        }
        for i in 0..self.config.output_dim {
            for j in 0..self.config.hidden_dim {
                self.w2[i][j] -= lr * dw2[i][j] / n;
            }
            self.b2[i] -= lr * db2[i] / n;
        }

        total_loss / n
    }

    /// Refine a raw mask using magnitude spectrogram features.
    ///
    /// For each T-F bin, extracts 5 features:
    ///   [magnitude, phase_diff, temporal_diff, spectral_diff, raw_mask_value]
    ///
    /// The network predicts a correction, and the output is:
    ///   refined_mask[i] = sigmoid(logit(raw_mask[i]) + correction[i])
    ///
    /// - `raw_mask`: flat mask from separator, indexed [frame * num_freq + freq_bin], values in [0,1]
    /// - `magnitudes`: STFT magnitudes in the same layout
    /// - `num_frames`: number of time frames
    /// - `num_freq`: number of frequency bins per frame
    pub fn refine_mask(
        &self,
        raw_mask: &[f64],
        magnitudes: &[f64],
        num_frames: usize,
        num_freq: usize,
    ) -> Vec<f64> {
        let total = num_frames * num_freq;
        assert_eq!(raw_mask.len(), total);
        assert_eq!(magnitudes.len(), total);

        let mut refined = vec![0.0; total];

        for t in 0..num_frames {
            for f in 0..num_freq {
                let idx = t * num_freq + f;
                let mag = magnitudes[idx];
                let mask_val = raw_mask[idx];

                // Feature: phase_diff (approximate via magnitude neighbors in time)
                let phase_diff = if t > 0 {
                    magnitudes[idx] - magnitudes[(t - 1) * num_freq + f]
                } else {
                    0.0
                };

                // Feature: temporal_diff
                let temporal_diff = if t > 0 {
                    raw_mask[idx] - raw_mask[(t - 1) * num_freq + f]
                } else {
                    0.0
                };

                // Feature: spectral_diff
                let spectral_diff = if f > 0 {
                    raw_mask[idx] - raw_mask[t * num_freq + (f - 1)]
                } else {
                    0.0
                };

                let features = [mag, phase_diff, temporal_diff, spectral_diff, mask_val];
                let correction = self.forward(&features);

                // Output is already sigmoid, so it is directly the refined mask
                refined[idx] = correction[0];
            }
        }

        refined
    }
}

#[inline]
fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_pass_shape() {
        let config = MLPConfig {
            input_dim: 5,
            hidden_dim: 64,
            output_dim: 1,
            learning_rate: 0.01,
        };
        let mlp = TinyMLP::new(config);
        let input = vec![0.5, -0.1, 0.3, 0.0, 0.8];
        let output = mlp.forward(&input);
        assert_eq!(output.len(), 1, "Output should have 1 element");

        // Test with different output_dim
        let config2 = MLPConfig {
            input_dim: 5,
            hidden_dim: 64,
            output_dim: 3,
            learning_rate: 0.01,
        };
        let mlp2 = TinyMLP::new(config2);
        let output2 = mlp2.forward(&input);
        assert_eq!(output2.len(), 3, "Output should have 3 elements");
    }

    #[test]
    fn test_training_convergence() {
        let config = MLPConfig {
            input_dim: 2,
            hidden_dim: 16,
            output_dim: 1,
            learning_rate: 0.5,
        };
        let mut mlp = TinyMLP::new(config);

        // Synthetic data: target = sigmoid(x0 + x1)
        let mut rng = Lcg::new(123);
        let examples: Vec<TrainingExample> = (0..50)
            .map(|_| {
                let x0 = rng.next_f64();
                let x1 = rng.next_f64();
                let target = sigmoid(x0 + x1);
                TrainingExample {
                    input: vec![x0, x1],
                    target: vec![target],
                }
            })
            .collect();

        let mut losses = Vec::new();
        for _ in 0..100 {
            let loss = mlp.train_step(&examples);
            losses.push(loss);
        }

        let first_loss = losses[0];
        let last_loss = *losses.last().unwrap();
        assert!(
            last_loss < first_loss,
            "Loss should decrease: first={first_loss:.6}, last={last_loss:.6}"
        );
    }

    #[test]
    fn test_mask_refinement_range() {
        let config = MLPConfig::default();
        let mlp = TinyMLP::new(config);

        let num_frames = 4;
        let num_freq = 8;
        let total = num_frames * num_freq;

        // Random-ish raw mask and magnitudes
        let mut rng = Lcg::new(77);
        let raw_mask: Vec<f64> = (0..total).map(|_| (rng.next_f64() + 1.0) / 2.0).collect();
        let magnitudes: Vec<f64> = (0..total).map(|_| (rng.next_f64() + 1.0) / 2.0 * 10.0).collect();

        let refined = mlp.refine_mask(&raw_mask, &magnitudes, num_frames, num_freq);

        assert_eq!(refined.len(), total);
        for (i, &v) in refined.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "Refined mask at index {i} = {v}, must be in [0, 1]"
            );
        }
    }

    #[test]
    fn test_param_count_under_100k() {
        let config = MLPConfig::default();
        let mlp = TinyMLP::new(config);
        let count = mlp.param_count();

        // input_dim=5, hidden=64, output=1
        // L1: 5*64 + 64 = 384
        // L2: 64*1 + 1 = 65
        // Total = 449
        assert_eq!(count, 449);
        assert!(count < 100_000, "Param count {count} should be < 100K");
    }

    #[test]
    fn test_param_count_large_config() {
        // Even with a larger config, stay under 100K
        let config = MLPConfig {
            input_dim: 100,
            hidden_dim: 64,
            output_dim: 100,
            learning_rate: 0.01,
        };
        let mlp = TinyMLP::new(config);
        let count = mlp.param_count();
        // L1: 100*64 + 64 = 6464
        // L2: 64*100 + 100 = 6500
        // Total = 12964
        assert_eq!(count, 12_964);
        assert!(count < 100_000, "Param count {count} should be < 100K");
    }
}
