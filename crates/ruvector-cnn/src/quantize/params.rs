//! Quantization parameters for INT8 quantization.
//!
//! This module defines the core quantization parameters used for both
//! symmetric and asymmetric quantization schemes.

use crate::error::{CnnError, CnnResult};
use serde::{Deserialize, Serialize};

/// Quantization parameters for a tensor or tensor slice.
///
/// Defines the mapping between floating-point values and quantized integers:
/// - **Symmetric**: `x_q = round(x / scale)`
/// - **Asymmetric**: `x_q = round(x / scale) + zero_point`
///
/// ## Invariants
///
/// - `scale > 0.0` (enforced at construction)
/// - `qmin <= zero_point <= qmax`
/// - For symmetric mode: `zero_point == 0`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factor for quantization.
    /// Maps FP32 range to INT8 range.
    pub scale: f32,

    /// Zero point for asymmetric quantization.
    /// Always 0 for symmetric quantization.
    pub zero_point: i32,

    /// Minimum quantized value (typically -128 for i8).
    pub qmin: i8,

    /// Maximum quantized value (typically 127 for i8).
    pub qmax: i8,
}

impl QuantizationParams {
    /// Create symmetric quantization parameters from min/max values.
    ///
    /// Symmetric quantization uses `zero_point = 0` and maps the range
    /// `[-max_abs, max_abs]` to `[-127, 127]`.
    ///
    /// # Arguments
    ///
    /// * `min_val` - Minimum value in the FP32 tensor
    /// * `max_val` - Maximum value in the FP32 tensor
    ///
    /// # Returns
    ///
    /// Quantization parameters with `zero_point = 0`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let params = QuantizationParams::from_minmax(-10.0, 10.0, QuantizationMode::Symmetric);
    /// assert_eq!(params.zero_point, 0);
    /// assert!(params.scale > 0.0);
    /// ```
    pub fn from_minmax(min_val: f32, max_val: f32, mode: QuantizationMode) -> CnnResult<Self> {
        if min_val > max_val {
            return Err(CnnError::InvalidParameter(format!(
                "min_val ({}) must be <= max_val ({})",
                min_val, max_val
            )));
        }

        match mode {
            QuantizationMode::Symmetric => {
                // Symmetric: zero_point = 0, scale based on max absolute value
                let max_abs = min_val.abs().max(max_val.abs());
                let scale = if max_abs > 0.0 {
                    max_abs / 127.0
                } else {
                    1.0 // Prevent division by zero
                };

                Ok(Self {
                    scale,
                    zero_point: 0,
                    qmin: -127,
                    qmax: 127,
                })
            }
            QuantizationMode::Asymmetric => {
                // Asymmetric: Map [min_val, max_val] to [-127, 127] (255 bins)
                // to maintain compatibility with i8 storage
                let scale = if max_val > min_val {
                    (max_val - min_val) / 254.0  // Use 254 to avoid clipping at edges
                } else {
                    1.0
                };

                let zero_point = if scale > 0.0 {
                    // Map min_val to qmin=-127
                    ((-min_val / scale).round() - 127.0).clamp(-127.0, 127.0) as i32
                } else {
                    0
                };

                Ok(Self {
                    scale,
                    zero_point,
                    qmin: -127,
                    qmax: 127,
                })
            }
        }
    }

    /// Create quantization parameters from percentile values.
    ///
    /// Used during calibration to exclude outliers that would skew the
    /// quantization range. Typically uses 0.001 and 0.999 percentiles.
    ///
    /// # Arguments
    ///
    /// * `percentile_min` - Lower percentile value (e.g., -10.0)
    /// * `percentile_max` - Upper percentile value (e.g., 10.0)
    /// * `mode` - Quantization mode (symmetric or asymmetric)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Use 99.8% of the data range, excluding outliers
    /// let params = QuantizationParams::from_percentile(
    ///     -9.5, 9.5, QuantizationMode::Symmetric
    /// );
    /// ```
    pub fn from_percentile(
        percentile_min: f32,
        percentile_max: f32,
        mode: QuantizationMode,
    ) -> CnnResult<Self> {
        Self::from_minmax(percentile_min, percentile_max, mode)
    }

    /// Validate that the parameters satisfy invariants.
    ///
    /// # Invariants
    ///
    /// - `scale > 0.0`
    /// - `qmin <= qmax`
    /// - `qmin <= zero_point <= qmax`
    /// - For symmetric mode: `zero_point == 0`
    pub fn validate(&self) -> CnnResult<()> {
        if self.scale <= 0.0 {
            return Err(CnnError::QuantizationError(format!(
                "scale must be positive, got {}",
                self.scale
            )));
        }

        if self.qmin > self.qmax {
            return Err(CnnError::QuantizationError(format!(
                "qmin ({}) must be <= qmax ({})",
                self.qmin, self.qmax
            )));
        }

        if self.zero_point < self.qmin as i32 || self.zero_point > self.qmax as i32 {
            return Err(CnnError::QuantizationError(format!(
                "zero_point ({}) must be in range [{}, {}]",
                self.zero_point, self.qmin, self.qmax
            )));
        }

        Ok(())
    }

    /// Quantize a single FP32 value to INT8.
    ///
    /// Formula:
    /// - Symmetric: `x_q = round(x / scale)`
    /// - Asymmetric: `x_q = round(x / scale) + zero_point`
    ///
    /// Result is clamped to `[qmin, qmax]`.
    #[inline]
    pub fn quantize_value(&self, value: f32) -> i8 {
        let q = (value / self.scale).round() + self.zero_point as f32;
        q.clamp(self.qmin as f32, self.qmax as f32) as i8
    }

    /// Dequantize a single INT8 value to FP32.
    ///
    /// Formula:
    /// - Symmetric: `x = x_q * scale`
    /// - Asymmetric: `x = (x_q - zero_point) * scale`
    #[inline]
    pub fn dequantize_value(&self, value: i8) -> f32 {
        (value as f32 - self.zero_point as f32) * self.scale
    }
}

/// Quantization scheme granularity.
///
/// Determines whether a single scale factor applies to the entire tensor
/// or per output channel (for weights).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// Single scale factor for entire tensor.
    /// Used for activations.
    PerTensor,

    /// Scale factor per output channel.
    /// Used for Conv2d weights to preserve accuracy.
    PerChannel,
}

/// Quantization mode (symmetric vs asymmetric).
///
/// - **Symmetric**: Zero point is always 0. Good for weights centered around 0.
/// - **Asymmetric**: Zero point computed to maximize range utilization. Good for ReLU activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// Symmetric quantization: `x_q = round(x / scale)`.
    ///
    /// Zero point is always 0. Simpler computation, but may waste range
    /// if data is not centered around 0.
    Symmetric,

    /// Asymmetric quantization: `x_q = round(x / scale) + zero_point`.
    ///
    /// Full range utilization for asymmetric distributions (e.g., ReLU outputs).
    Asymmetric,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_minmax() {
        let params = QuantizationParams::from_minmax(-10.0, 10.0, QuantizationMode::Symmetric)
            .unwrap();

        assert_eq!(params.zero_point, 0);
        assert!(params.scale > 0.0);
        assert_eq!(params.qmin, -127);
        assert_eq!(params.qmax, 127);

        // Validate
        params.validate().unwrap();
    }

    #[test]
    fn test_asymmetric_minmax() {
        let params = QuantizationParams::from_minmax(0.0, 10.0, QuantizationMode::Asymmetric)
            .unwrap();

        // For [0, 10] range, zero_point should map 0.0 to a quantized value
        assert!(params.scale > 0.0);
        assert!(params.zero_point >= -128);
        assert!(params.zero_point <= 127);

        params.validate().unwrap();
    }

    #[test]
    fn test_quantize_dequantize_symmetric() {
        let params = QuantizationParams::from_minmax(-10.0, 10.0, QuantizationMode::Symmetric)
            .unwrap();

        let value = 5.0f32;
        let quantized = params.quantize_value(value);
        let dequantized = params.dequantize_value(quantized);

        // Should be close (within quantization error)
        assert!((dequantized - value).abs() < 0.1);
    }

    #[test]
    fn test_quantize_dequantize_asymmetric() {
        let params = QuantizationParams::from_minmax(0.0, 10.0, QuantizationMode::Asymmetric)
            .unwrap();

        let value = 5.0f32;
        let quantized = params.quantize_value(value);
        let dequantized = params.dequantize_value(quantized);

        assert!((dequantized - value).abs() < 0.1);
    }

    #[test]
    fn test_zero_value_quantization() {
        let params = QuantizationParams::from_minmax(-10.0, 10.0, QuantizationMode::Symmetric)
            .unwrap();

        let quantized = params.quantize_value(0.0);
        assert_eq!(quantized, 0);

        let dequantized = params.dequantize_value(0);
        assert_eq!(dequantized, 0.0);
    }

    #[test]
    fn test_clipping() {
        let params = QuantizationParams::from_minmax(-1.0, 1.0, QuantizationMode::Symmetric)
            .unwrap();

        // Values outside range should be clipped
        let large = params.quantize_value(1000.0);
        assert_eq!(large, 127);

        let small = params.quantize_value(-1000.0);
        assert_eq!(small, -127);
    }

    #[test]
    fn test_invalid_range() {
        let result = QuantizationParams::from_minmax(10.0, -10.0, QuantizationMode::Symmetric);
        assert!(result.is_err());
    }

    #[test]
    fn test_percentile_constructor() {
        let params = QuantizationParams::from_percentile(-9.5, 9.5, QuantizationMode::Symmetric)
            .unwrap();

        assert_eq!(params.zero_point, 0);
        params.validate().unwrap();
    }

    #[test]
    fn test_validation_negative_scale() {
        let mut params = QuantizationParams::from_minmax(-10.0, 10.0, QuantizationMode::Symmetric)
            .unwrap();

        params.scale = -1.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validation_zero_scale() {
        let mut params = QuantizationParams::from_minmax(-10.0, 10.0, QuantizationMode::Symmetric)
            .unwrap();

        params.scale = 0.0;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validation_invalid_qmin_qmax() {
        let mut params = QuantizationParams::from_minmax(-10.0, 10.0, QuantizationMode::Symmetric)
            .unwrap();

        params.qmin = 127;
        params.qmax = -127;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validation_zero_point_out_of_range() {
        let mut params = QuantizationParams::from_minmax(-10.0, 10.0, QuantizationMode::Symmetric)
            .unwrap();

        params.zero_point = 200;
        assert!(params.validate().is_err());
    }
}
