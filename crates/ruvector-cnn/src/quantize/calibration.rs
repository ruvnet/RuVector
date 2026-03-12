//! Calibration and Quantization Parameters (ADR-091 Phase 2)
//!
//! This module provides histogram-based calibration for INT8 quantization.

use std::collections::HashMap;

/// Histogram for calibration data collection
#[derive(Debug, Clone)]
pub struct CalibrationHistogram {
    pub min_val: f32,
    pub max_val: f32,
    pub num_bins: usize,
    pub bins: Vec<u64>,
}

impl CalibrationHistogram {
    pub fn new(min_val: f32, max_val: f32, num_bins: usize) -> Self {
        Self {
            min_val,
            max_val,
            num_bins,
            bins: vec![0; num_bins],
        }
    }

    /// Add a value to the histogram
    pub fn add(&mut self, value: f32) {
        if value < self.min_val || value > self.max_val {
            return;
        }

        let bin_width = (self.max_val - self.min_val) / self.num_bins as f32;
        let bin_idx = ((value - self.min_val) / bin_width) as usize;
        let bin_idx = bin_idx.min(self.num_bins - 1);
        self.bins[bin_idx] += 1;
    }

    /// Compute quantization parameters from histogram
    pub fn compute_quantization_params(&self) -> QuantizationParams {
        // Use min/max for symmetric quantization
        let abs_max = self.max_val.abs().max(self.min_val.abs());
        let scale = abs_max / 127.0;
        let zero_point = 0; // Symmetric quantization

        QuantizationParams {
            scale,
            zero_point,
            min_val: self.min_val,
            max_val: self.max_val,
            num_bins: self.num_bins,
        }
    }
}

/// Quantization parameters for a tensor
#[derive(Debug, Clone, Copy)]
pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i32,
    pub min_val: f32,
    pub max_val: f32,
    pub num_bins: usize,
}

/// Quantizer for converting FP32 to INT8 and back
pub struct Quantizer {
    params: HashMap<String, QuantizationParams>,
}

impl Quantizer {
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    pub fn add_params(&mut self, name: String, params: QuantizationParams) {
        self.params.insert(name, params);
    }

    pub fn quantize(&self, name: &str, values: &[f32]) -> Vec<i8> {
        let params = self.params.get(name).expect("No params for tensor");
        values
            .iter()
            .map(|&v| {
                let q = (v / params.scale).round() as i32 + params.zero_point;
                q.clamp(-128, 127) as i8
            })
            .collect()
    }

    pub fn dequantize(&self, name: &str, values: &[i8]) -> Vec<f32> {
        let params = self.params.get(name).expect("No params for tensor");
        values
            .iter()
            .map(|&v| (v as i32 - params.zero_point) as f32 * params.scale)
            .collect()
    }
}

impl Default for Quantizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_calibration() {
        let mut hist = CalibrationHistogram::new(-10.0, 10.0, 100);

        // Add some values
        for _ in 0..100 {
            hist.add(5.0);
        }
        for _ in 0..50 {
            hist.add(-5.0);
        }

        let params = hist.compute_quantization_params();
        assert!((params.scale - 10.0 / 127.0).abs() < 0.01);
        assert_eq!(params.zero_point, 0);
    }

    #[test]
    fn test_quantizer() {
        let mut quantizer = Quantizer::new();
        quantizer.add_params(
            "test".to_string(),
            QuantizationParams {
                scale: 0.1,
                zero_point: 0,
                min_val: -12.8,
                max_val: 12.7,
                num_bins: 256,
            },
        );

        let values = vec![0.0, 1.0, -1.0, 12.7, -12.8];
        let quantized = quantizer.quantize("test", &values);
        let dequantized = quantizer.dequantize("test", &quantized);

        for (orig, deq) in values.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.2); // Allow some quantization error
        }
    }
}
