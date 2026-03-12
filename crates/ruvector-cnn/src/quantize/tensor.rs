//! Quantized tensor types with metadata.
//!
//! This module provides type-safe INT8 tensors with quantization metadata
//! for efficient neural network inference.

use crate::error::{CnnError, CnnResult};
use super::params::QuantizationParams;
use serde::{Deserialize, Serialize};

/// Metadata for a quantized tensor.
///
/// Stores the quantization parameters and shape information needed
/// to correctly interpret and dequantize the tensor data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationMetadata {
    /// Quantization scale factor.
    pub scale: f32,

    /// Zero point for asymmetric quantization.
    pub zero_point: i32,

    /// Tensor shape (e.g., [batch, height, width, channels]).
    pub shape: Vec<usize>,
}

impl QuantizationMetadata {
    /// Create new quantization metadata.
    pub fn new(scale: f32, zero_point: i32, shape: Vec<usize>) -> Self {
        Self {
            scale,
            zero_point,
            shape,
        }
    }

    /// Total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Validate metadata consistency.
    pub fn validate(&self) -> CnnResult<()> {
        if self.scale <= 0.0 {
            return Err(CnnError::QuantizationError(format!(
                "scale must be positive, got {}",
                self.scale
            )));
        }

        if self.shape.is_empty() {
            return Err(CnnError::QuantizationError(
                "shape cannot be empty".to_string()
            ));
        }

        if self.shape.iter().any(|&d| d == 0) {
            return Err(CnnError::QuantizationError(
                "shape dimensions must be positive".to_string()
            ));
        }

        Ok(())
    }
}

/// Quantized tensor with INT8 data and metadata.
///
/// Stores quantized values along with the information needed to
/// dequantize them back to FP32.
///
/// ## Invariants (Enforced at Construction)
///
/// - **INV-1**: `data.len() == metadata.numel()`
/// - **INV-2**: `metadata.scale > 0.0`
/// - **INV-3**: All values in `data` are in range `[qmin, qmax]`
///
/// ## Example
///
/// ```rust,ignore
/// use ruvector_cnn::quantize::{QuantizedTensor, QuantizationParams, QuantizationMode};
///
/// let fp32_data = vec![1.0, 2.0, -1.0, 0.5];
/// let shape = vec![4];
/// let params = QuantizationParams::from_minmax(-2.0, 2.0, QuantizationMode::Symmetric)?;
///
/// // Quantize
/// let quantized = QuantizedTensor::<i8>::quantize(&fp32_data, &shape, &params)?;
///
/// // Dequantize
/// let dequantized = quantized.dequantize()?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedTensor<T> {
    /// Quantized data (INT8).
    data: Vec<T>,

    /// Quantization metadata.
    metadata: QuantizationMetadata,
}

impl QuantizedTensor<i8> {
    /// Create a new quantized tensor with validation.
    ///
    /// # Arguments
    ///
    /// * `data` - Quantized INT8 values
    /// * `metadata` - Quantization metadata (scale, zero_point, shape)
    ///
    /// # Errors
    ///
    /// - If `data.len() != metadata.numel()` (INV-1)
    /// - If metadata is invalid (INV-2)
    pub fn new(data: Vec<i8>, metadata: QuantizationMetadata) -> CnnResult<Self> {
        metadata.validate()?;

        if data.len() != metadata.numel() {
            return Err(CnnError::InvalidShape {
                expected: format!("data length {}", metadata.numel()),
                got: format!("{}", data.len()),
            });
        }

        Ok(Self { data, metadata })
    }

    /// Quantize FP32 data to INT8.
    ///
    /// # Arguments
    ///
    /// * `fp32_data` - Input FP32 values
    /// * `shape` - Tensor shape
    /// * `params` - Quantization parameters
    ///
    /// # Returns
    ///
    /// Quantized INT8 tensor.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let fp32 = vec![1.0, 2.0, -1.0];
    /// let shape = vec![3];
    /// let params = QuantizationParams::from_minmax(-2.0, 2.0, QuantizationMode::Symmetric)?;
    /// let quantized = QuantizedTensor::quantize(&fp32, &shape, &params)?;
    /// ```
    pub fn quantize(
        fp32_data: &[f32],
        shape: &[usize],
        params: &QuantizationParams,
    ) -> CnnResult<Self> {
        params.validate()?;

        let expected_numel: usize = shape.iter().product();
        if fp32_data.len() != expected_numel {
            return Err(CnnError::InvalidShape {
                expected: format!("data length {}", expected_numel),
                got: format!("{}", fp32_data.len()),
            });
        }

        // Quantize each value
        let data: Vec<i8> = fp32_data
            .iter()
            .map(|&val| params.quantize_value(val))
            .collect();

        let metadata = QuantizationMetadata::new(
            params.scale,
            params.zero_point,
            shape.to_vec(),
        );

        Ok(Self { data, metadata })
    }

    /// Dequantize INT8 data to FP32.
    ///
    /// # Returns
    ///
    /// FP32 values with the same shape.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dequantized = quantized.dequantize()?;
    /// assert_eq!(dequantized.len(), quantized.data().len());
    /// ```
    pub fn dequantize(&self) -> CnnResult<Vec<f32>> {
        self.metadata.validate()?;

        let params = QuantizationParams {
            scale: self.metadata.scale,
            zero_point: self.metadata.zero_point,
            qmin: -127,
            qmax: 127,
        };

        let fp32_data: Vec<f32> = self.data
            .iter()
            .map(|&val| params.dequantize_value(val))
            .collect();

        Ok(fp32_data)
    }

    /// Get reference to quantized data.
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Get mutable reference to quantized data.
    pub fn data_mut(&mut self) -> &mut [i8] {
        &mut self.data
    }

    /// Get reference to metadata.
    pub fn metadata(&self) -> &QuantizationMetadata {
        &self.metadata
    }

    /// Get tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.metadata.shape
    }

    /// Get scale factor.
    pub fn scale(&self) -> f32 {
        self.metadata.scale
    }

    /// Get zero point.
    pub fn zero_point(&self) -> i32 {
        self.metadata.zero_point
    }

    /// Check bounds invariant: all values in `[qmin, qmax]`.
    ///
    /// This is a sanity check to ensure data hasn't been corrupted.
    /// Should always return `true` for properly constructed tensors.
    pub fn check_bounds(&self, qmin: i8, qmax: i8) -> bool {
        self.data.iter().all(|&val| val >= qmin && val <= qmax)
    }

    /// Validate all invariants.
    ///
    /// # Invariants
    ///
    /// - **INV-1**: `data.len() == metadata.numel()`
    /// - **INV-2**: `metadata.scale > 0.0`
    /// - **INV-3**: All values in `[-127, 127]`
    pub fn validate(&self) -> CnnResult<()> {
        // INV-1: Length check
        if self.data.len() != self.metadata.numel() {
            return Err(CnnError::QuantizationError(format!(
                "INV-1 violation: data length {} != metadata.numel() {}",
                self.data.len(),
                self.metadata.numel()
            )));
        }

        // INV-2: Metadata validation (includes scale > 0)
        self.metadata.validate()?;

        // INV-3: Bounds check
        if !self.check_bounds(-127, 127) {
            return Err(CnnError::QuantizationError(
                "INV-3 violation: some values outside [-127, 127]".to_string()
            ));
        }

        Ok(())
    }

    /// Reshape the tensor to a new shape.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - New shape (must have same total elements)
    ///
    /// # Errors
    ///
    /// If `new_shape.iter().product() != self.data.len()`.
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> CnnResult<()> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.data.len() {
            return Err(CnnError::InvalidShape {
                expected: format!("numel {}", self.data.len()),
                got: format!("numel {}", new_numel),
            });
        }

        self.metadata.shape = new_shape;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::QuantizationMode;

    fn create_test_params() -> QuantizationParams {
        QuantizationParams::from_minmax(-10.0, 10.0, QuantizationMode::Symmetric).unwrap()
    }

    #[test]
    fn test_metadata_creation() {
        let meta = QuantizationMetadata::new(0.1, 0, vec![2, 3, 4]);
        assert_eq!(meta.scale, 0.1);
        assert_eq!(meta.zero_point, 0);
        assert_eq!(meta.shape, vec![2, 3, 4]);
        assert_eq!(meta.numel(), 24);
    }

    #[test]
    fn test_metadata_validation() {
        let meta = QuantizationMetadata::new(0.1, 0, vec![2, 3]);
        assert!(meta.validate().is_ok());

        let invalid = QuantizationMetadata::new(-0.1, 0, vec![2, 3]);
        assert!(invalid.validate().is_err());

        let empty_shape = QuantizationMetadata::new(0.1, 0, vec![]);
        assert!(empty_shape.validate().is_err());

        let zero_dim = QuantizationMetadata::new(0.1, 0, vec![2, 0, 3]);
        assert!(zero_dim.validate().is_err());
    }

    #[test]
    fn test_quantize_dequantize() {
        let fp32_data = vec![1.0, 2.0, -1.0, 0.5, -5.0, 5.0];
        let shape = vec![6];
        let params = create_test_params();

        let quantized = QuantizedTensor::quantize(&fp32_data, &shape, &params).unwrap();
        assert_eq!(quantized.data().len(), 6);
        assert_eq!(quantized.shape(), &[6]);

        let dequantized = quantized.dequantize().unwrap();
        assert_eq!(dequantized.len(), 6);

        // Check quantization error is reasonable
        for (original, restored) in fp32_data.iter().zip(dequantized.iter()) {
            assert!((original - restored).abs() < 0.2);
        }
    }

    #[test]
    fn test_quantize_shape_mismatch() {
        let fp32_data = vec![1.0, 2.0, 3.0];
        let wrong_shape = vec![2, 2]; // 4 elements expected, but 3 provided
        let params = create_test_params();

        let result = QuantizedTensor::quantize(&fp32_data, &wrong_shape, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_with_invalid_length() {
        let data = vec![1i8, 2, 3];
        let metadata = QuantizationMetadata::new(0.1, 0, vec![2, 2]); // Expects 4 elements

        let result = QuantizedTensor::new(data, metadata);
        assert!(result.is_err());
    }

    #[test]
    fn test_bounds_check() {
        let data = vec![0i8, 50, -50, 127, -127];
        let metadata = QuantizationMetadata::new(0.1, 0, vec![5]);
        let tensor = QuantizedTensor::new(data, metadata).unwrap();

        assert!(tensor.check_bounds(-127, 127));
        assert!(!tensor.check_bounds(-50, 50));
    }

    #[test]
    fn test_validate_invariants() {
        let fp32_data = vec![1.0, 2.0, 3.0];
        let shape = vec![3];
        let params = create_test_params();

        let tensor = QuantizedTensor::quantize(&fp32_data, &shape, &params).unwrap();

        // Should pass all invariants
        assert!(tensor.validate().is_ok());
    }

    #[test]
    fn test_reshape() {
        let fp32_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![6];
        let params = create_test_params();

        let mut tensor = QuantizedTensor::quantize(&fp32_data, &shape, &params).unwrap();

        // Reshape to 2x3
        tensor.reshape(vec![2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);

        // Invalid reshape
        assert!(tensor.reshape(vec![2, 2]).is_err());
    }

    #[test]
    fn test_zero_value() {
        let fp32_data = vec![0.0, 0.0, 0.0];
        let shape = vec![3];
        let params = create_test_params();

        let quantized = QuantizedTensor::quantize(&fp32_data, &shape, &params).unwrap();
        let dequantized = quantized.dequantize().unwrap();

        for &val in &dequantized {
            assert!((val).abs() < 0.01);
        }
    }

    #[test]
    fn test_asymmetric_quantization() {
        let fp32_data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let shape = vec![6];
        let params = QuantizationParams::from_minmax(0.0, 5.0, QuantizationMode::Asymmetric)
            .unwrap();

        let quantized = QuantizedTensor::quantize(&fp32_data, &shape, &params).unwrap();
        assert!(quantized.validate().is_ok());

        let dequantized = quantized.dequantize().unwrap();
        for (i, (original, restored)) in fp32_data.iter().zip(dequantized.iter()).enumerate() {
            let error = (original - restored).abs();
            // Asymmetric quantization maps [0,5] to [-128,127] (255 bins)
            // Scale = 5.0/255 ~= 0.0196, max quantization error ~= scale
            assert!(
                error < 0.6,
                "Value mismatch at index {}: original={}, restored={}, error={}",
                i, original, restored, error
            );
        }
    }

    #[test]
    fn test_getters() {
        let fp32_data = vec![1.0, 2.0];
        let shape = vec![2];
        let params = create_test_params();

        let tensor = QuantizedTensor::quantize(&fp32_data, &shape, &params).unwrap();

        assert_eq!(tensor.data().len(), 2);
        assert_eq!(tensor.shape(), &[2]);
        assert!(tensor.scale() > 0.0);
        assert_eq!(tensor.zero_point(), 0); // Symmetric mode
    }
}
