//! Batch Normalization Layer
//!
//! Implements batch normalization with learned scale (gamma) and shift (beta) parameters:
//! y = gamma * (x - mean) / sqrt(var + epsilon) + beta
//!
//! During inference, uses running mean and variance computed during training.

use crate::{simd, CnnError, CnnResult, Tensor};

use super::Layer;

/// Alias for BatchNorm (for API compatibility with PyTorch naming).
pub type BatchNorm2d = BatchNorm;

/// Batch Normalization layer
///
/// Normalizes the input across the channel dimension for NHWC tensors.
#[derive(Debug, Clone)]
pub struct BatchNorm {
    /// Number of channels (features)
    num_features: usize,
    /// Learned scale parameter (gamma)
    gamma: Vec<f32>,
    /// Learned shift parameter (beta)
    beta: Vec<f32>,
    /// Running mean (for inference)
    running_mean: Vec<f32>,
    /// Running variance (for inference)
    running_var: Vec<f32>,
    /// Small constant for numerical stability
    epsilon: f32,
    /// Momentum for running statistics update (used in training mode)
    #[allow(dead_code)]
    momentum: f32,
}

impl BatchNorm {
    /// Create a new BatchNorm layer with default initialization
    ///
    /// - gamma initialized to 1.0
    /// - beta initialized to 0.0
    /// - running_mean initialized to 0.0
    /// - running_var initialized to 1.0
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            epsilon: 1e-5,
            momentum: 0.1,
        }
    }

    /// Create BatchNorm with custom epsilon
    pub fn with_epsilon(num_features: usize, epsilon: f32) -> Self {
        let mut bn = Self::new(num_features);
        bn.epsilon = epsilon;
        bn
    }

    /// Set the learned parameters (gamma, beta)
    pub fn set_params(&mut self, gamma: Vec<f32>, beta: Vec<f32>) -> CnnResult<()> {
        if gamma.len() != self.num_features || beta.len() != self.num_features {
            return Err(CnnError::invalid_shape(
                format!("num_features={}", self.num_features),
                format!("gamma={}, beta={}", gamma.len(), beta.len()),
            ));
        }
        self.gamma = gamma;
        self.beta = beta;
        Ok(())
    }

    /// Set the running statistics (mean, var)
    pub fn set_running_stats(&mut self, mean: Vec<f32>, var: Vec<f32>) -> CnnResult<()> {
        if mean.len() != self.num_features || var.len() != self.num_features {
            return Err(CnnError::invalid_shape(
                format!("num_features={}", self.num_features),
                format!("mean={}, var={}", mean.len(), var.len()),
            ));
        }
        self.running_mean = mean;
        self.running_var = var;
        Ok(())
    }

    /// Get the number of features (channels)
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Get a reference to gamma
    pub fn gamma(&self) -> &[f32] {
        &self.gamma
    }

    /// Get a reference to beta
    pub fn beta(&self) -> &[f32] {
        &self.beta
    }

    /// Get a reference to running mean
    pub fn running_mean(&self) -> &[f32] {
        &self.running_mean
    }

    /// Get a reference to running variance
    pub fn running_var(&self) -> &[f32] {
        &self.running_var
    }
}

impl Layer for BatchNorm {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        // Validate input shape (must be NHWC with matching channels)
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(CnnError::invalid_shape(
                "4D tensor (NHWC)",
                format!("{}D tensor", shape.len()),
            ));
        }

        let channels = shape[3];
        if channels != self.num_features {
            return Err(CnnError::invalid_shape(
                format!("{} channels", self.num_features),
                format!("{} channels", channels),
            ));
        }

        let mut output = Tensor::zeros(shape);

        simd::batch_norm_simd(
            input.data(),
            output.data_mut(),
            &self.gamma,
            &self.beta,
            &self.running_mean,
            &self.running_var,
            self.epsilon,
            self.num_features,
        );

        Ok(output)
    }

    fn name(&self) -> &'static str {
        "BatchNorm"
    }

    fn num_params(&self) -> usize {
        // gamma + beta
        self.num_features * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_norm_creation() {
        let bn = BatchNorm::new(64);
        assert_eq!(bn.num_features(), 64);
        assert_eq!(bn.gamma().len(), 64);
        assert_eq!(bn.beta().len(), 64);
        assert_eq!(bn.num_params(), 128);
    }

    #[test]
    fn test_batch_norm_forward() {
        let bn = BatchNorm::new(4);
        let input = Tensor::ones(&[1, 8, 8, 4]);
        let output = bn.forward(&input).unwrap();

        assert_eq!(output.shape(), input.shape());

        // With default params (gamma=1, beta=0, mean=0, var=1):
        // output = 1 * (1 - 0) / sqrt(1 + eps) + 0 ≈ 1
        for &val in output.data() {
            assert!((val - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_batch_norm_normalization() {
        let mut bn = BatchNorm::new(2);

        // Set mean=[1, 2], var=[1, 4]
        bn.set_running_stats(vec![1.0, 2.0], vec![1.0, 4.0])
            .unwrap();

        // Input: [[1, 2], [3, 4]] at each spatial location
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 1, 2]).unwrap();
        let output = bn.forward(&input).unwrap();

        // For channel 0: (x - 1) / sqrt(1 + eps) ≈ (x - 1)
        // For channel 1: (x - 2) / sqrt(4 + eps) ≈ (x - 2) / 2

        // input[0,0] = 1, channel 0: (1-1)/1 = 0
        assert!(output.data()[0].abs() < 0.01);
        // input[0,1] = 2, channel 1: (2-2)/2 = 0
        assert!(output.data()[1].abs() < 0.01);
        // input[1,0] = 3, channel 0: (3-1)/1 = 2
        assert!((output.data()[2] - 2.0).abs() < 0.01);
        // input[1,1] = 4, channel 1: (4-2)/2 = 1
        assert!((output.data()[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_norm_invalid_shape() {
        let bn = BatchNorm::new(4);
        let input = Tensor::ones(&[1, 8, 8, 8]); // Wrong number of channels

        let result = bn.forward(&input);
        assert!(result.is_err());
    }
}
