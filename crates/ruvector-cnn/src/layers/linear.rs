//! Linear (Fully Connected) layer implementation.
//!
//! Implements a standard linear transformation: y = xW^T + b

use super::{Layer, TensorShape};
use crate::error::{CnnError, CnnResult};
use crate::Tensor;

/// Linear (Fully Connected) layer.
///
/// Performs the operation: output = input @ weight^T + bias
#[derive(Clone, Debug)]
pub struct Linear {
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Weight matrix [out_features, in_features]
    weight: Vec<f32>,
    /// Bias vector [out_features], None for no bias
    bias: Option<Vec<f32>>,
}

impl Linear {
    /// Creates a new Linear layer with zero-initialized weights.
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> CnnResult<Self> {
        if in_features == 0 || out_features == 0 {
            return Err(CnnError::InvalidParameter(
                "Features must be > 0".to_string(),
            ));
        }

        let weight = vec![0.0; out_features * in_features];
        let bias = if use_bias {
            Some(vec![0.0; out_features])
        } else {
            None
        };

        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
        })
    }

    /// Creates a Linear layer with provided weights.
    pub fn with_weights(
        in_features: usize,
        out_features: usize,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> CnnResult<Self> {
        if weight.len() != out_features * in_features {
            return Err(CnnError::dim_mismatch(
                out_features * in_features,
                weight.len(),
            ));
        }

        if let Some(ref b) = bias {
            if b.len() != out_features {
                return Err(CnnError::dim_mismatch(out_features, b.len()));
            }
        }

        Ok(Self {
            in_features,
            out_features,
            weight,
            bias,
        })
    }

    /// Returns the input features.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Returns the output features.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Returns a reference to the weight matrix.
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Returns a reference to the bias vector.
    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    /// Sets the weight matrix.
    pub fn set_weight(&mut self, weight: Vec<f32>) -> CnnResult<()> {
        if weight.len() != self.out_features * self.in_features {
            return Err(CnnError::dim_mismatch(
                self.out_features * self.in_features,
                weight.len(),
            ));
        }
        self.weight = weight;
        Ok(())
    }

    /// Sets the bias vector.
    pub fn set_bias(&mut self, bias: Option<Vec<f32>>) -> CnnResult<()> {
        if let Some(ref b) = bias {
            if b.len() != self.out_features {
                return Err(CnnError::dim_mismatch(self.out_features, b.len()));
            }
        }
        self.bias = bias;
        Ok(())
    }

    /// Forward pass for a single input vector.
    pub fn forward_vec(&self, input: &[f32]) -> CnnResult<Vec<f32>> {
        if input.len() != self.in_features {
            return Err(CnnError::dim_mismatch(self.in_features, input.len()));
        }

        let mut output = vec![0.0; self.out_features];

        // output = input @ weight^T
        for o in 0..self.out_features {
            let mut sum = 0.0f32;
            for i in 0..self.in_features {
                sum += input[i] * self.weight[o * self.in_features + i];
            }
            if let Some(ref bias) = self.bias {
                sum += bias[o];
            }
            output[o] = sum;
        }

        Ok(output)
    }

    /// Forward pass for a batch of input vectors.
    pub fn forward_batch(&self, input: &[f32], batch_size: usize) -> CnnResult<Vec<f32>> {
        if input.len() != batch_size * self.in_features {
            return Err(CnnError::dim_mismatch(
                batch_size * self.in_features,
                input.len(),
            ));
        }

        let mut output = vec![0.0; batch_size * self.out_features];

        for n in 0..batch_size {
            let input_offset = n * self.in_features;
            let output_offset = n * self.out_features;

            for o in 0..self.out_features {
                let mut sum = 0.0f32;
                for i in 0..self.in_features {
                    sum += input[input_offset + i] * self.weight[o * self.in_features + i];
                }
                if let Some(ref bias) = self.bias {
                    sum += bias[o];
                }
                output[output_offset + o] = sum;
            }
        }

        Ok(output)
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let shape = input.shape();
        // For linear, flatten all dimensions except batch
        let batch_size = if shape.is_empty() { 1 } else { shape[0] };
        let features = input.numel() / batch_size;

        if features != self.in_features {
            return Err(CnnError::dim_mismatch(self.in_features, features));
        }

        let output_data = self.forward_batch(input.data(), batch_size)?;
        let out_shape = vec![batch_size, self.out_features];
        Tensor::from_data(output_data, &out_shape)
    }

    fn name(&self) -> &'static str {
        "Linear"
    }

    fn num_params(&self) -> usize {
        let weight_params = self.out_features * self.in_features;
        let bias_params = if self.bias.is_some() {
            self.out_features
        } else {
            0
        };
        weight_params + bias_params
    }
}

impl Linear {
    /// Returns the output TensorShape for a given input TensorShape
    pub fn output_shape(&self, input_shape: &TensorShape) -> TensorShape {
        TensorShape {
            n: input_shape.n,
            c: self.out_features,
            h: 1,
            w: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let linear = Linear::new(512, 1000, true).unwrap();
        assert_eq!(linear.in_features(), 512);
        assert_eq!(linear.out_features(), 1000);
        assert!(linear.bias().is_some());
    }

    #[test]
    fn test_linear_no_bias() {
        let linear = Linear::new(512, 1000, false).unwrap();
        assert!(linear.bias().is_none());
    }

    #[test]
    fn test_linear_forward_identity() {
        let linear = Linear::with_weights(
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0], // Identity matrix
            Some(vec![0.0, 0.0]),
        )
        .unwrap();

        let input = vec![1.0, 2.0];
        let output = linear.forward_vec(&input).unwrap();

        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_forward_with_bias() {
        let linear = Linear::with_weights(
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            Some(vec![5.0, 10.0]),
        )
        .unwrap();

        let input = vec![1.0, 2.0];
        let output = linear.forward_vec(&input).unwrap();

        assert!((output[0] - 6.0).abs() < 1e-6);
        assert!((output[1] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_forward_batch() {
        let linear = Linear::with_weights(
            2,
            2,
            vec![1.0, 0.0, 0.0, 1.0],
            None,
        )
        .unwrap();

        let input = vec![1.0, 2.0, 3.0, 4.0]; // batch of 2
        let output = linear.forward_batch(&input, 2).unwrap();

        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 2.0).abs() < 1e-6);
        assert!((output[2] - 3.0).abs() < 1e-6);
        assert!((output[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_num_params() {
        let linear = Linear::new(512, 1000, true).unwrap();
        assert_eq!(linear.num_params(), 512 * 1000 + 1000);
    }

    #[test]
    fn test_linear_output_shape() {
        let linear = Linear::new(576, 1024, true).unwrap();
        let input_shape = TensorShape::new(2, 576, 1, 1);
        let output_shape = linear.output_shape(&input_shape);

        assert_eq!(output_shape.n, 2);
        assert_eq!(output_shape.c, 1024);
        assert_eq!(output_shape.h, 1);
        assert_eq!(output_shape.w, 1);
    }
}
