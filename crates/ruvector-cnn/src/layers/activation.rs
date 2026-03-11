//! Activation Functions
//!
//! SIMD-optimized activation functions:
//! - ReLU: max(0, x)
//! - ReLU6: min(6, max(0, x))
//! - Swish: x * sigmoid(x)
//! - HardSwish: x * relu6(x + 3) / 6
//! - Sigmoid: 1 / (1 + exp(-x))

use crate::{simd, CnnResult, Tensor};

use super::Layer;

/// Types of activation functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActivationType {
    /// ReLU: max(0, x)
    ReLU,
    /// ReLU6: min(6, max(0, x))
    ReLU6,
    /// Swish: x * sigmoid(x)
    Swish,
    /// HardSwish: x * relu6(x + 3) / 6
    HardSwish,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// No activation (identity)
    Identity,
}

/// Generic activation layer that wraps an activation type.
#[derive(Clone, Debug)]
pub struct Activation {
    activation_type: ActivationType,
}

impl Activation {
    /// Creates a new activation layer.
    pub fn new(activation_type: ActivationType) -> Self {
        Self { activation_type }
    }

    /// Returns the activation type.
    pub fn activation_type(&self) -> ActivationType {
        self.activation_type
    }

    /// Applies the activation in-place.
    pub fn apply_inplace(&self, data: &mut [f32]) {
        match self.activation_type {
            ActivationType::ReLU => {
                for x in data.iter_mut() {
                    *x = x.max(0.0);
                }
            }
            ActivationType::ReLU6 => {
                for x in data.iter_mut() {
                    *x = x.max(0.0).min(6.0);
                }
            }
            ActivationType::Swish => {
                for x in data.iter_mut() {
                    let sigmoid = 1.0 / (1.0 + (-*x).exp());
                    *x *= sigmoid;
                }
            }
            ActivationType::HardSwish => {
                for x in data.iter_mut() {
                    *x *= (*x + 3.0).max(0.0).min(6.0) / 6.0;
                }
            }
            ActivationType::Sigmoid => {
                for x in data.iter_mut() {
                    *x = 1.0 / (1.0 + (-*x).exp());
                }
            }
            ActivationType::Identity => {}
        }
    }
}

impl Layer for Activation {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let mut output = input.clone();
        self.apply_inplace(output.data_mut());
        Ok(output)
    }

    fn name(&self) -> &'static str {
        match self.activation_type {
            ActivationType::ReLU => "ReLU",
            ActivationType::ReLU6 => "ReLU6",
            ActivationType::Swish => "Swish",
            ActivationType::HardSwish => "HardSwish",
            ActivationType::Sigmoid => "Sigmoid",
            ActivationType::Identity => "Identity",
        }
    }
}

/// ReLU activation: max(0, x)
#[derive(Debug, Clone, Default)]
pub struct ReLU;

impl ReLU {
    /// Create a new ReLU activation
    pub fn new() -> Self {
        Self
    }
}

impl Layer for ReLU {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let mut output = Tensor::zeros(input.shape());
        simd::relu_simd(input.data(), output.data_mut());
        Ok(output)
    }

    fn name(&self) -> &'static str {
        "ReLU"
    }
}

/// ReLU6 activation: min(6, max(0, x))
/// Used in MobileNet architectures
#[derive(Debug, Clone, Default)]
pub struct ReLU6;

impl ReLU6 {
    /// Create a new ReLU6 activation
    pub fn new() -> Self {
        Self
    }
}

impl Layer for ReLU6 {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let mut output = Tensor::zeros(input.shape());
        simd::relu6_simd(input.data(), output.data_mut());
        Ok(output)
    }

    fn name(&self) -> &'static str {
        "ReLU6"
    }
}

/// Swish activation: x * sigmoid(x)
/// Also known as SiLU (Sigmoid Linear Unit)
#[derive(Debug, Clone, Default)]
pub struct Swish;

impl Swish {
    /// Create a new Swish activation
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Swish {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let mut output = Tensor::zeros(input.shape());
        simd::scalar::swish_scalar(input.data(), output.data_mut());
        Ok(output)
    }

    fn name(&self) -> &'static str {
        "Swish"
    }
}

/// HardSwish activation: x * relu6(x + 3) / 6
/// Efficient approximation of Swish for mobile inference
#[derive(Debug, Clone, Default)]
pub struct HardSwish;

impl HardSwish {
    /// Create a new HardSwish activation
    pub fn new() -> Self {
        Self
    }
}

impl Layer for HardSwish {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let mut output = Tensor::zeros(input.shape());
        simd::scalar::hard_swish_scalar(input.data(), output.data_mut());
        Ok(output)
    }

    fn name(&self) -> &'static str {
        "HardSwish"
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[derive(Debug, Clone, Default)]
pub struct Sigmoid;

impl Sigmoid {
    /// Create a new Sigmoid activation
    pub fn new() -> Self {
        Self
    }
}

impl Layer for Sigmoid {
    fn forward(&self, input: &Tensor) -> CnnResult<Tensor> {
        let mut output = Tensor::zeros(input.shape());
        simd::scalar::sigmoid_scalar(input.data(), output.data_mut());
        Ok(output)
    }

    fn name(&self) -> &'static str {
        "Sigmoid"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        let relu = ReLU::new();
        let input = Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let output = relu.forward(&input).unwrap();

        assert_eq!(output.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu6() {
        let relu6 = ReLU6::new();
        let input = Tensor::from_data(vec![-2.0, 0.0, 3.0, 6.0, 10.0], &[5]).unwrap();
        let output = relu6.forward(&input).unwrap();

        assert_eq!(output.data(), &[0.0, 0.0, 3.0, 6.0, 6.0]);
    }

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid::new();
        let input = Tensor::from_data(vec![0.0], &[1]).unwrap();
        let output = sigmoid.forward(&input).unwrap();

        assert!((output.data()[0] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_swish() {
        let swish = Swish::new();
        let input = Tensor::from_data(vec![0.0, 1.0, -1.0], &[3]).unwrap();
        let output = swish.forward(&input).unwrap();

        // swish(0) = 0 * 0.5 = 0
        assert!(output.data()[0].abs() < 0.001);
        // swish(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((output.data()[1] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_hard_swish() {
        let hs = HardSwish::new();
        let input = Tensor::from_data(vec![-4.0, -3.0, 0.0, 3.0, 4.0], &[5]).unwrap();
        let output = hs.forward(&input).unwrap();

        // hardswish(-4) = -4 * relu6(-1) / 6 = 0
        assert!(output.data()[0].abs() < 0.001);
        // hardswish(-3) = -3 * relu6(0) / 6 = 0
        assert!(output.data()[1].abs() < 0.001);
        // hardswish(0) = 0 * relu6(3) / 6 = 0
        assert!(output.data()[2].abs() < 0.001);
        // hardswish(3) = 3 * relu6(6) / 6 = 3
        assert!((output.data()[3] - 3.0).abs() < 0.001);
    }
}
