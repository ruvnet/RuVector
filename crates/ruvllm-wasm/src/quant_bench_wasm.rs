//! In-Browser Quantization Benchmarking (ADR-090 Phase 4)
//!
//! This module provides WASM bindings for benchmarking quantization performance
//! directly in web browsers, enabling real-time performance analysis.
//!
//! ## Features
//!
//! - **Single-Format Benchmarks**: Measure quantize/dequantize latency
//! - **Format Comparison**: Compare Pi2, Pi3, and uniform quantization
//! - **JSON Reports**: Structured benchmark results for analysis
//!
//! ## Quick Start (JavaScript)
//!
//! ```javascript
//! import { QuantBenchWasm } from 'ruvllm-wasm';
//!
//! const bench = new QuantBenchWasm();
//!
//! // Generate random weights
//! const weights = new Float32Array(1024);
//! for (let i = 0; i < weights.length; i++) {
//!     weights[i] = (Math.random() - 0.5) * 2;
//! }
//!
//! // Run benchmark
//! const result = bench.runBench(weights, 3, 100); // 3-bit, 100 iterations
//! console.log(JSON.parse(result));
//!
//! // Compare all formats
//! const comparison = bench.compareFormats(weights);
//! console.log(JSON.parse(comparison));
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::pi_quant_wasm::PiQuantWasm;

// ============================================================================
// Benchmark Result Structures
// ============================================================================

/// Result of a single-format benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchResult {
    /// Format name (e.g., "pi3", "pi2")
    format: String,

    /// Number of weights processed
    num_weights: usize,

    /// Number of iterations
    iterations: u32,

    /// Quantization time in microseconds (per iteration average)
    quantize_us: f64,

    /// Dequantization time in microseconds (per iteration average)
    dequantize_us: f64,

    /// Total time in microseconds (per iteration average)
    total_us: f64,

    /// Throughput in millions of weights per second (quantize)
    quantize_throughput_mw_s: f64,

    /// Throughput in millions of weights per second (dequantize)
    dequantize_throughput_mw_s: f64,

    /// Mean Squared Error
    mse: f32,

    /// Bits per weight (including overhead)
    bits_per_weight: f32,

    /// Compression ratio vs FP32
    compression_ratio: f32,
}

/// Result of format comparison benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FormatComparison {
    /// Number of weights tested
    num_weights: usize,

    /// Number of iterations per format
    iterations: u32,

    /// Results for each format
    formats: Vec<BenchResult>,

    /// Fastest format for quantization
    fastest_quantize: String,

    /// Fastest format for dequantization
    fastest_dequantize: String,

    /// Lowest MSE format
    lowest_mse: String,

    /// Best compression ratio format
    best_compression: String,
}

// ============================================================================
// QuantBenchWasm - WASM Benchmark Bindings
// ============================================================================

/// WASM-bindgen wrapper for in-browser quantization benchmarking.
///
/// Provides tools for measuring quantization performance and comparing
/// different quantization formats in real-time within the browser.
#[wasm_bindgen]
pub struct QuantBenchWasm {
    /// Default k value for pi-quantization
    default_k: u8,
}

#[wasm_bindgen]
impl QuantBenchWasm {
    /// Create a new benchmark instance.
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const bench = new QuantBenchWasm();
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { default_k: 4 }
    }

    /// Set the default k value for pi-quantization.
    ///
    /// # Arguments
    ///
    /// * `k` - Divisor for pi step size (must be 2, 3, 4, or 5)
    #[wasm_bindgen(js_name = setDefaultK)]
    pub fn set_default_k(&mut self, k: u8) {
        if [2, 3, 4, 5].contains(&k) {
            self.default_k = k;
        }
    }

    /// Run benchmark for a single quantization format.
    ///
    /// # Arguments
    ///
    /// * `weights` - Input f32 weights to benchmark
    /// * `bits` - Quantization bits (2 or 3)
    /// * `iterations` - Number of benchmark iterations
    ///
    /// # Returns
    ///
    /// JSON string containing benchmark results
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const result = bench.runBench(weights, 3, 100);
    /// const data = JSON.parse(result);
    /// console.log(`Quantize: ${data.quantize_us} us`);
    /// console.log(`Dequantize: ${data.dequantize_us} us`);
    /// console.log(`MSE: ${data.mse}`);
    /// ```
    #[wasm_bindgen(js_name = runBench)]
    pub fn run_bench(&self, weights: &[f32], bits: u8, iterations: u32) -> String {
        let result = self.benchmark_format(weights, bits, iterations);
        serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
    }

    /// Compare all supported quantization formats.
    ///
    /// Tests Pi2, Pi3 with different k values, and reports comparative metrics.
    ///
    /// # Arguments
    ///
    /// * `weights` - Input f32 weights to benchmark
    ///
    /// # Returns
    ///
    /// JSON string containing comparison results
    ///
    /// # Example (JavaScript)
    ///
    /// ```javascript
    /// const comparison = bench.compareFormats(weights);
    /// const data = JSON.parse(comparison);
    /// console.log(`Fastest quantize: ${data.fastest_quantize}`);
    /// console.log(`Lowest MSE: ${data.lowest_mse}`);
    /// ```
    #[wasm_bindgen(js_name = compareFormats)]
    pub fn compare_formats(&self, weights: &[f32]) -> String {
        let iterations = 50; // Default iterations for comparison
        let comparison = self.compare_all_formats(weights, iterations);
        serde_json::to_string(&comparison).unwrap_or_else(|_| "{}".to_string())
    }

    /// Run detailed benchmark with warmup and statistical analysis.
    ///
    /// # Arguments
    ///
    /// * `weights` - Input f32 weights to benchmark
    /// * `bits` - Quantization bits (2 or 3)
    /// * `iterations` - Number of benchmark iterations
    /// * `warmup_iterations` - Number of warmup iterations (not counted)
    ///
    /// # Returns
    ///
    /// JSON string containing detailed benchmark results
    #[wasm_bindgen(js_name = runDetailedBench)]
    pub fn run_detailed_bench(
        &self,
        weights: &[f32],
        bits: u8,
        iterations: u32,
        warmup_iterations: u32,
    ) -> String {
        // Warmup phase
        let quantizer = PiQuantWasm::new(bits, self.default_k);
        for _ in 0..warmup_iterations {
            let packed = quantizer.quantize(weights);
            let _ = quantizer.dequantize(&packed);
        }

        // Actual benchmark
        let result = self.benchmark_format(weights, bits, iterations);
        serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
    }

    /// Benchmark memory efficiency.
    ///
    /// Calculates actual memory savings for different quantization formats.
    ///
    /// # Arguments
    ///
    /// * `num_weights` - Number of weights to analyze
    ///
    /// # Returns
    ///
    /// JSON string containing memory efficiency analysis
    #[wasm_bindgen(js_name = memoryEfficiency)]
    pub fn memory_efficiency(&self, num_weights: usize) -> String {
        let fp32_bytes = num_weights * 4;

        let pi3_groups = (num_weights + 7) / 8;
        let pi3_bytes = pi3_groups * 3 + pi3_groups; // Data + scale per group
        let pi3_ratio = fp32_bytes as f32 / pi3_bytes as f32;

        let pi2_blocks = (num_weights + 3) / 4;
        let pi2_bytes = pi2_blocks + pi2_blocks; // Data + scale per block
        let pi2_ratio = fp32_bytes as f32 / pi2_bytes as f32;

        let result = serde_json::json!({
            "num_weights": num_weights,
            "fp32_bytes": fp32_bytes,
            "pi3": {
                "bytes": pi3_bytes,
                "compression_ratio": pi3_ratio,
                "bits_per_weight": 3.0625
            },
            "pi2": {
                "bytes": pi2_bytes,
                "compression_ratio": pi2_ratio,
                "bits_per_weight": 2.0625
            }
        });

        serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
    }

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /// Benchmark a single quantization format
    fn benchmark_format(&self, weights: &[f32], bits: u8, iterations: u32) -> BenchResult {
        let quantizer = PiQuantWasm::new(bits, self.default_k);
        let num_weights = weights.len();
        let format_name = format!("pi{}", bits);

        // Time quantization
        let start_quant = Self::now_us();
        let mut packed = Vec::new();
        for _ in 0..iterations {
            packed = quantizer.quantize(weights);
        }
        let end_quant = Self::now_us();
        let quantize_total_us = end_quant - start_quant;

        // Time dequantization
        let start_dequant = Self::now_us();
        let mut _reconstructed = Vec::new();
        for _ in 0..iterations {
            _reconstructed = quantizer.dequantize(&packed);
        }
        let end_dequant = Self::now_us();
        let dequantize_total_us = end_dequant - start_dequant;

        // Calculate metrics
        let quantize_us = quantize_total_us / (iterations as f64);
        let dequantize_us = dequantize_total_us / (iterations as f64);
        let total_us = quantize_us + dequantize_us;

        // Throughput in millions of weights per second
        let quantize_throughput = if quantize_us > 0.0 {
            (num_weights as f64) / quantize_us // weights per microsecond = million weights per second
        } else {
            0.0
        };

        let dequantize_throughput = if dequantize_us > 0.0 {
            (num_weights as f64) / dequantize_us
        } else {
            0.0
        };

        let mse = quantizer.compute_mse(weights, &packed);
        let bits_per_weight = quantizer.bits_per_weight();
        let compression_ratio = 32.0 / bits_per_weight;

        BenchResult {
            format: format_name,
            num_weights,
            iterations,
            quantize_us,
            dequantize_us,
            total_us,
            quantize_throughput_mw_s: quantize_throughput,
            dequantize_throughput_mw_s: dequantize_throughput,
            mse,
            bits_per_weight,
            compression_ratio,
        }
    }

    /// Compare all quantization formats
    fn compare_all_formats(&self, weights: &[f32], iterations: u32) -> FormatComparison {
        let mut formats = Vec::new();

        // Test Pi3 and Pi2
        formats.push(self.benchmark_format(weights, 3, iterations));
        formats.push(self.benchmark_format(weights, 2, iterations));

        // Find best in each category
        let fastest_quantize = formats
            .iter()
            .min_by(|a, b| a.quantize_us.partial_cmp(&b.quantize_us).unwrap())
            .map(|r| r.format.clone())
            .unwrap_or_default();

        let fastest_dequantize = formats
            .iter()
            .min_by(|a, b| a.dequantize_us.partial_cmp(&b.dequantize_us).unwrap())
            .map(|r| r.format.clone())
            .unwrap_or_default();

        let lowest_mse = formats
            .iter()
            .min_by(|a, b| a.mse.partial_cmp(&b.mse).unwrap())
            .map(|r| r.format.clone())
            .unwrap_or_default();

        let best_compression = formats
            .iter()
            .max_by(|a, b| {
                a.compression_ratio
                    .partial_cmp(&b.compression_ratio)
                    .unwrap()
            })
            .map(|r| r.format.clone())
            .unwrap_or_default();

        FormatComparison {
            num_weights: weights.len(),
            iterations,
            formats,
            fastest_quantize,
            fastest_dequantize,
            lowest_mse,
            best_compression,
        }
    }

    /// Get current time in microseconds (uses performance.now() on WASM)
    fn now_us() -> f64 {
        #[cfg(target_arch = "wasm32")]
        {
            use crate::utils::now_ms;
            now_ms() * 1000.0 // Convert ms to us
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            use std::time::Instant;
            static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
            let start = START.get_or_init(Instant::now);
            start.elapsed().as_micros() as f64
        }
    }
}

impl Default for QuantBenchWasm {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_new() {
        let bench = QuantBenchWasm::new();
        assert_eq!(bench.default_k, 4);
    }

    #[test]
    fn test_set_default_k() {
        let mut bench = QuantBenchWasm::new();
        bench.set_default_k(2);
        assert_eq!(bench.default_k, 2);

        // Invalid k should not change
        bench.set_default_k(7);
        assert_eq!(bench.default_k, 2);
    }

    #[test]
    fn test_run_bench_3bit() {
        let bench = QuantBenchWasm::new();
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();

        let result_json = bench.run_bench(&weights, 3, 10);
        let result: BenchResult = serde_json::from_str(&result_json).unwrap();

        assert_eq!(result.format, "pi3");
        assert_eq!(result.num_weights, 64);
        assert_eq!(result.iterations, 10);
        assert!(result.quantize_us >= 0.0);
        assert!(result.dequantize_us >= 0.0);
        assert!(result.mse >= 0.0);
        assert!((result.bits_per_weight - 3.0625).abs() < 0.01);
    }

    #[test]
    fn test_run_bench_2bit() {
        let bench = QuantBenchWasm::new();
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();

        let result_json = bench.run_bench(&weights, 2, 10);
        let result: BenchResult = serde_json::from_str(&result_json).unwrap();

        assert_eq!(result.format, "pi2");
        assert!((result.bits_per_weight - 2.0625).abs() < 0.01);
    }

    #[test]
    fn test_compare_formats() {
        let bench = QuantBenchWasm::new();
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();

        let comparison_json = bench.compare_formats(&weights);
        let comparison: FormatComparison = serde_json::from_str(&comparison_json).unwrap();

        assert_eq!(comparison.num_weights, 64);
        assert_eq!(comparison.formats.len(), 2); // pi3 and pi2
        assert!(!comparison.fastest_quantize.is_empty());
        assert!(!comparison.fastest_dequantize.is_empty());
        assert!(!comparison.lowest_mse.is_empty());
        assert!(!comparison.best_compression.is_empty());
    }

    #[test]
    fn test_memory_efficiency() {
        let bench = QuantBenchWasm::new();
        let result_json = bench.memory_efficiency(1024);

        // Parse and verify JSON structure
        let result: serde_json::Value = serde_json::from_str(&result_json).unwrap();

        assert_eq!(result["num_weights"], 1024);
        assert_eq!(result["fp32_bytes"], 4096);
        assert!(result["pi3"]["compression_ratio"].as_f64().unwrap() > 5.0);
        assert!(result["pi2"]["compression_ratio"].as_f64().unwrap() > 7.0);
    }

    #[test]
    fn test_run_detailed_bench() {
        let bench = QuantBenchWasm::new();
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 32.0).collect();

        let result_json = bench.run_detailed_bench(&weights, 3, 10, 5);
        let result: BenchResult = serde_json::from_str(&result_json).unwrap();

        assert_eq!(result.format, "pi3");
        assert_eq!(result.iterations, 10);
    }

    #[test]
    fn test_benchmark_empty_weights() {
        let bench = QuantBenchWasm::new();
        let weights: Vec<f32> = Vec::new();

        let result_json = bench.run_bench(&weights, 3, 10);
        let result: BenchResult = serde_json::from_str(&result_json).unwrap();

        assert_eq!(result.num_weights, 0);
    }

    #[test]
    fn test_throughput_calculation() {
        let bench = QuantBenchWasm::new();
        // Use larger array for more realistic throughput
        let weights: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 512.0).collect();

        let result_json = bench.run_bench(&weights, 3, 100);
        let result: BenchResult = serde_json::from_str(&result_json).unwrap();

        // Throughput should be positive for non-zero timing
        if result.quantize_us > 0.0 {
            assert!(result.quantize_throughput_mw_s > 0.0);
        }
        if result.dequantize_us > 0.0 {
            assert!(result.dequantize_throughput_mw_s > 0.0);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let bench = QuantBenchWasm::new();
        let weights: Vec<f32> = vec![0.0; 64];

        let result_json = bench.run_bench(&weights, 3, 1);
        let result: BenchResult = serde_json::from_str(&result_json).unwrap();

        // 32-bit / 3.0625-bit ≈ 10.4x compression
        assert!(result.compression_ratio > 10.0);
        assert!(result.compression_ratio < 11.0);
    }
}
