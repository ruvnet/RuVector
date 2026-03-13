//! Pi-Quantization Benchmarks (ADR-090 Implementation)
//!
//! Comprehensive benchmarks for Pi-constant quantization measuring:
//! - Quantization throughput (target: >1 GB/s)
//! - Dequantization SIMD performance (target: >10 GB/s)
//! - Hadamard transform performance (target: <50μs for 4096)
//! - QAT forward/backward passes
//! - Quality metrics computation
//!
//! Performance targets from ADR-090:
//! - Pi-quantize 3-bit: >1 GB/s
//! - Pi-quantize 2-bit: >1 GB/s
//! - Dequantize NEON: >10 GB/s
//! - Dequantize AVX2: >10 GB/s
//! - Hadamard 4096: <50μs
//!
//! Run with: cargo bench -p ruvllm --bench pi_quant_bench

#![allow(unused_imports, dead_code, unused_variables)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use std::f32::consts::PI;

// ============================================================================
// Inline implementations for benchmark (avoids import issues)
// ============================================================================

/// Pi-quantization constants
const PI3_BLOCK_WEIGHTS: usize = 8;
const PI3_BLOCK_BYTES: usize = 3;
const PI2_BLOCK_WEIGHTS: usize = 4;
const PI2_BLOCK_BYTES: usize = 1;

/// Packed 3-bit block
#[derive(Debug, Clone, Copy, Default)]
struct Pi3BitBlock {
    data: [u8; 3],
}

impl Pi3BitBlock {
    fn new() -> Self {
        Self { data: [0; 3] }
    }

    fn pack(&mut self, values: &[i8; 8]) {
        let mut u = [0u8; 8];
        for i in 0..8 {
            let v = values[i].clamp(-4, 3);
            u[i] = (v + 4) as u8;
        }

        self.data[0] = (u[0] & 0x07) | ((u[1] & 0x07) << 3) | ((u[2] & 0x07) << 6);

        self.data[1] = ((u[2] >> 2) & 0x01)
            | ((u[3] & 0x07) << 1)
            | ((u[4] & 0x07) << 4)
            | ((u[5] & 0x07) << 7);

        self.data[2] = ((u[5] >> 1) & 0x03) | ((u[6] & 0x07) << 2) | ((u[7] & 0x07) << 5);
    }

    fn unpack(&self) -> [i8; 8] {
        let d = self.data;

        let u0 = d[0] & 0x07;
        let u1 = (d[0] >> 3) & 0x07;
        let u2 = ((d[0] >> 6) & 0x03) | ((d[1] & 0x01) << 2);
        let u3 = (d[1] >> 1) & 0x07;
        let u4 = (d[1] >> 4) & 0x07;
        let u5 = ((d[1] >> 7) & 0x01) | ((d[2] & 0x03) << 1);
        let u6 = (d[2] >> 2) & 0x07;
        let u7 = (d[2] >> 5) & 0x07;

        [
            (u0 as i8) - 4,
            (u1 as i8) - 4,
            (u2 as i8) - 4,
            (u3 as i8) - 4,
            (u4 as i8) - 4,
            (u5 as i8) - 4,
            (u6 as i8) - 4,
            (u7 as i8) - 4,
        ]
    }
}

/// Packed 2-bit block
#[derive(Debug, Clone, Copy, Default)]
struct Pi2BitBlock {
    data: u8,
}

impl Pi2BitBlock {
    fn new() -> Self {
        Self { data: 0 }
    }

    fn pack(&mut self, values: &[i8; 4]) {
        let u0 = ((values[0].clamp(-2, 1) + 2) as u8) & 0x03;
        let u1 = ((values[1].clamp(-2, 1) + 2) as u8) & 0x03;
        let u2 = ((values[2].clamp(-2, 1) + 2) as u8) & 0x03;
        let u3 = ((values[3].clamp(-2, 1) + 2) as u8) & 0x03;

        self.data = u0 | (u1 << 2) | (u2 << 4) | (u3 << 6);
    }

    fn unpack(&self) -> [i8; 4] {
        let d = self.data;
        [
            ((d & 0x03) as i8) - 2,
            (((d >> 2) & 0x03) as i8) - 2,
            (((d >> 4) & 0x03) as i8) - 2,
            (((d >> 6) & 0x03) as i8) - 2,
        ]
    }
}

/// Pi-quantizer configuration
struct PiQuantizer {
    bits: u8,
    k: u8,
    alpha: f32,
    half_range: i8,
    base_step: f32,
}

impl PiQuantizer {
    fn new(bits: u8, k: u8, alpha: f32) -> Self {
        let half_range = 1i8 << (bits - 1);
        let base_step = PI / (k as f32);
        Self {
            bits,
            k,
            alpha,
            half_range,
            base_step,
        }
    }

    #[inline(always)]
    fn step_size(&self) -> f32 {
        self.alpha * self.base_step
    }

    #[inline(always)]
    fn quantize_to_int(&self, w: f32) -> i8 {
        let step = self.step_size();
        if step <= 0.0 {
            return 0;
        }
        let q = (w / step).round() as i32;
        let half = self.half_range as i32;
        q.clamp(-half, half - 1) as i8
    }

    fn quantize_block_3bit(&self, weights: &[f32; 8]) -> Pi3BitBlock {
        let mut block = Pi3BitBlock::new();
        let mut q_values = [0i8; 8];
        for (i, &w) in weights.iter().enumerate() {
            q_values[i] = self.quantize_to_int(w);
        }
        block.pack(&q_values);
        block
    }

    fn quantize_block_2bit(&self, weights: &[f32; 4]) -> Pi2BitBlock {
        let mut block = Pi2BitBlock::new();
        let mut q_values = [0i8; 4];
        for (i, &w) in weights.iter().enumerate() {
            q_values[i] = self.quantize_to_int(w);
        }
        block.pack(&q_values);
        block
    }
}

// ============================================================================
// SIMD Dequantization Implementations
// ============================================================================

/// Scalar dequantization (baseline)
fn pi_dequantize_scalar(packed: &[u8], scale: f32, output: &mut [f32]) {
    let num_groups = packed.len() / PI3_BLOCK_BYTES;

    for group in 0..num_groups {
        let byte_offset = group * PI3_BLOCK_BYTES;
        let out_offset = group * PI3_BLOCK_WEIGHTS;

        let b0 = packed[byte_offset] as u32;
        let b1 = packed[byte_offset + 1] as u32;
        let b2 = packed[byte_offset + 2] as u32;
        let combined = b0 | (b1 << 8) | (b2 << 16);

        for i in 0..8 {
            let shift = i * 3;
            let raw = ((combined >> shift) & 0x7) as i32;
            let signed = raw - 4;
            output[out_offset + i] = (signed as f32) * scale;
        }
    }
}

/// NEON dequantization (ARM64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pi_dequantize_neon(packed: &[u8], scale: f32, output: &mut [f32]) {
    use core::arch::aarch64::*;

    let num_groups = packed.len() / PI3_BLOCK_BYTES;
    if num_groups == 0 {
        return;
    }

    let scale_vec = vdupq_n_f32(scale);
    let bias_vec = vdupq_n_s32(-4);

    let simd_groups = num_groups / 4;
    let mut group = 0usize;

    while group < simd_groups * 4 {
        let byte_offset = group * PI3_BLOCK_BYTES;
        let out_offset = group * PI3_BLOCK_WEIGHTS;

        for g in 0..4 {
            let gb = byte_offset + g * 3;
            let go = out_offset + g * 8;

            let b0 = *packed.get_unchecked(gb) as u32;
            let b1 = *packed.get_unchecked(gb + 1) as u32;
            let b2 = *packed.get_unchecked(gb + 2) as u32;
            let combined = b0 | (b1 << 8) | (b2 << 16);

            let mut raw_vals = [0i32; 8];
            for i in 0..8 {
                let shift = i * 3;
                raw_vals[i] = ((combined >> shift) & 0x7) as i32;
            }

            let raw_lo = vld1q_s32(raw_vals.as_ptr());
            let raw_hi = vld1q_s32(raw_vals.as_ptr().add(4));

            let signed_lo = vaddq_s32(raw_lo, bias_vec);
            let signed_hi = vaddq_s32(raw_hi, bias_vec);

            let float_lo = vcvtq_f32_s32(signed_lo);
            let float_hi = vcvtq_f32_s32(signed_hi);

            let result_lo = vmulq_f32(float_lo, scale_vec);
            let result_hi = vmulq_f32(float_hi, scale_vec);

            vst1q_f32(output.as_mut_ptr().add(go), result_lo);
            vst1q_f32(output.as_mut_ptr().add(go + 4), result_hi);
        }

        group += 4;
    }

    // Handle remaining groups
    while group < num_groups {
        let byte_offset = group * PI3_BLOCK_BYTES;
        let out_offset = group * PI3_BLOCK_WEIGHTS;

        let b0 = *packed.get_unchecked(byte_offset) as u32;
        let b1 = *packed.get_unchecked(byte_offset + 1) as u32;
        let b2 = *packed.get_unchecked(byte_offset + 2) as u32;
        let combined = b0 | (b1 << 8) | (b2 << 16);

        for i in 0..8 {
            let shift = i * 3;
            let raw = ((combined >> shift) & 0x7) as i32;
            let signed = raw - 4;
            *output.get_unchecked_mut(out_offset + i) = (signed as f32) * scale;
        }

        group += 1;
    }
}

/// AVX2 dequantization (x86_64)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pi_dequantize_avx2(packed: &[u8], scale: f32, output: &mut [f32]) {
    use core::arch::x86_64::*;

    let num_groups = packed.len() / PI3_BLOCK_BYTES;
    if num_groups == 0 {
        return;
    }

    let scale_vec = _mm256_set1_ps(scale);
    let bias_vec = _mm256_set1_epi32(-4);

    let simd_groups = num_groups / 4;
    let mut group = 0usize;

    while group < simd_groups * 4 {
        let byte_offset = group * PI3_BLOCK_BYTES;
        let out_offset = group * PI3_BLOCK_WEIGHTS;

        for g in 0..4 {
            let gb = byte_offset + g * 3;
            let go = out_offset + g * 8;

            let b0 = *packed.get_unchecked(gb) as u32;
            let b1 = *packed.get_unchecked(gb + 1) as u32;
            let b2 = *packed.get_unchecked(gb + 2) as u32;
            let combined = b0 | (b1 << 8) | (b2 << 16);

            let v0 = (combined & 0x7) as i32;
            let v1 = ((combined >> 3) & 0x7) as i32;
            let v2 = ((combined >> 6) & 0x7) as i32;
            let v3 = ((combined >> 9) & 0x7) as i32;
            let v4 = ((combined >> 12) & 0x7) as i32;
            let v5 = ((combined >> 15) & 0x7) as i32;
            let v6 = ((combined >> 18) & 0x7) as i32;
            let v7 = ((combined >> 21) & 0x7) as i32;

            let raw_vec = _mm256_setr_epi32(v0, v1, v2, v3, v4, v5, v6, v7);
            let signed_vec = _mm256_add_epi32(raw_vec, bias_vec);
            let float_vec = _mm256_cvtepi32_ps(signed_vec);
            let result_vec = _mm256_mul_ps(float_vec, scale_vec);

            _mm256_storeu_ps(output.as_mut_ptr().add(go), result_vec);
        }

        group += 4;
    }

    // Handle remaining groups
    while group < num_groups {
        let byte_offset = group * PI3_BLOCK_BYTES;
        let out_offset = group * PI3_BLOCK_WEIGHTS;

        let b0 = *packed.get_unchecked(byte_offset) as u32;
        let b1 = *packed.get_unchecked(byte_offset + 1) as u32;
        let b2 = *packed.get_unchecked(byte_offset + 2) as u32;
        let combined = b0 | (b1 << 8) | (b2 << 16);

        for i in 0..8 {
            let shift = i * 3;
            let raw = ((combined >> shift) & 0x7) as i32;
            let signed = raw - 4;
            *output.get_unchecked_mut(out_offset + i) = (signed as f32) * scale;
        }

        group += 1;
    }
}

// ============================================================================
// Hadamard Transform Implementation
// ============================================================================

/// Walsh-Hadamard transform (in-place)
fn hadamard_transform(data: &mut [f32]) {
    let n = data.len();
    if n == 0 || (n & (n - 1)) != 0 {
        return; // Must be power of 2
    }

    let mut h = 1;
    while h < n {
        let mut j = 0;
        while j < n {
            for k in 0..h {
                let a = data[j + k];
                let b = data[j + k + h];
                data[j + k] = a + b;
                data[j + k + h] = a - b;
            }
            j += h * 2;
        }
        h *= 2;
    }

    // Normalize
    let norm = 1.0 / (n as f32).sqrt();
    for d in data.iter_mut() {
        *d *= norm;
    }
}

/// SIMD-optimized Hadamard transform
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hadamard_transform_neon(data: &mut [f32]) {
    use core::arch::aarch64::*;

    let n = data.len();
    if n == 0 || (n & (n - 1)) != 0 {
        return;
    }

    let mut h = 1;

    while h < n {
        if h >= 4 {
            let mut j = 0;
            while j < n {
                let mut k = 0;
                while k + 4 <= h {
                    let ptr_a = data.as_mut_ptr().add(j + k);
                    let ptr_b = data.as_mut_ptr().add(j + k + h);

                    let a = vld1q_f32(ptr_a);
                    let b = vld1q_f32(ptr_b);

                    let sum = vaddq_f32(a, b);
                    let diff = vsubq_f32(a, b);

                    vst1q_f32(ptr_a, sum);
                    vst1q_f32(ptr_b, diff);

                    k += 4;
                }
                while k < h {
                    let a = data[j + k];
                    let b = data[j + k + h];
                    data[j + k] = a + b;
                    data[j + k + h] = a - b;
                    k += 1;
                }
                j += h * 2;
            }
        } else {
            let mut j = 0;
            while j < n {
                for k in 0..h {
                    let a = data[j + k];
                    let b = data[j + k + h];
                    data[j + k] = a + b;
                    data[j + k + h] = a - b;
                }
                j += h * 2;
            }
        }
        h *= 2;
    }

    // Normalize with SIMD
    let norm = 1.0 / (n as f32).sqrt();
    let norm_vec = vdupq_n_f32(norm);
    let chunks = n / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let v = vld1q_f32(data.as_ptr().add(idx));
        let result = vmulq_f32(v, norm_vec);
        vst1q_f32(data.as_mut_ptr().add(idx), result);
    }

    for i in (chunks * 4)..n {
        data[i] *= norm;
    }
}

// ============================================================================
// QAT Forward/Backward Pass
// ============================================================================

/// STE variants
enum SteVariant {
    Standard,
    Clipped { clip_val: f32 },
    Ewgs { lambda: f32 },
}

impl SteVariant {
    #[inline]
    fn backward(&self, w: f32, q: f32, grad_out: f32) -> f32 {
        match self {
            SteVariant::Standard => grad_out,
            SteVariant::Clipped { clip_val } => {
                if w.abs() <= *clip_val {
                    grad_out
                } else {
                    0.0
                }
            }
            SteVariant::Ewgs { lambda } => grad_out * (1.0 + lambda * (w - q).abs()),
        }
    }
}

/// QAT forward pass (quantize + dequantize)
fn qat_forward(weights: &[f32], quantizer: &PiQuantizer, output: &mut [f32]) {
    let step = quantizer.step_size();

    for (i, &w) in weights.iter().enumerate() {
        let q_int = quantizer.quantize_to_int(w);
        output[i] = (q_int as f32) * step;
    }
}

/// QAT backward pass with STE
fn qat_backward_ste(
    weights: &[f32],
    quantized: &[f32],
    grad_out: &[f32],
    grad_w: &mut [f32],
    ste: &SteVariant,
) {
    for i in 0..weights.len() {
        grad_w[i] = ste.backward(weights[i], quantized[i], grad_out[i]);
    }
}

// ============================================================================
// Quality Metrics
// ============================================================================

/// Compute MSE
fn compute_mse(original: &[f32], quantized: &[f32]) -> f64 {
    if original.len() != quantized.len() || original.is_empty() {
        return 0.0;
    }

    let sum: f64 = original
        .iter()
        .zip(quantized.iter())
        .map(|(&o, &q)| {
            let diff = (o - q) as f64;
            diff * diff
        })
        .sum();

    sum / (original.len() as f64)
}

/// Compute spectral distortion
fn compute_spectral_distortion(original: &[f32], quantized: &[f32]) -> f64 {
    if original.len() != quantized.len() || original.is_empty() {
        return f64::NEG_INFINITY;
    }

    let signal_power: f64 = original.iter().map(|&x| (x as f64).powi(2)).sum();
    if signal_power == 0.0 {
        return 0.0;
    }

    let mse = compute_mse(original, quantized);
    10.0 * (mse / (signal_power / original.len() as f64)).log10()
}

// ============================================================================
// Benchmark Helper Functions
// ============================================================================

fn random_weights(size: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn random_packed_3bit(num_weights: usize) -> Vec<u8> {
    let mut rng = thread_rng();
    let num_bytes = (num_weights / 8) * 3;
    (0..num_bytes).map(|_| rng.gen()).collect()
}

// ============================================================================
// Benchmarks
// ============================================================================

/// Benchmark: Pi-Quantization 3-bit throughput (original implementation)
/// Target: >1 GB/s
fn bench_pi_quantize_3bit(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi_quantize_3bit");
    group.sample_size(100);

    let quantizer = PiQuantizer::new(3, 4, 1.0);

    for &size in &[256, 4096, 4096 * 11008] {
        let weights = random_weights(size);
        let num_blocks = size / 8;
        let output_bytes = num_blocks * 3;

        group.throughput(Throughput::Bytes(output_bytes as u64));
        group.bench_with_input(BenchmarkId::new("size", size), &weights, |b, w| {
            b.iter(|| {
                let mut blocks = Vec::with_capacity(num_blocks);
                for chunk in w.chunks_exact(8) {
                    let arr: [f32; 8] = chunk.try_into().unwrap();
                    blocks.push(black_box(quantizer.quantize_block_3bit(&arr)));
                }
                blocks
            })
        });
    }

    group.finish();
}

// ============================================================================
// NEW: High-Performance Quantization Benchmarks (>1 GB/s Target)
// ============================================================================

/// Optimized scalar 3-bit quantization with pre-allocated buffer
fn quantize_3bit_fast(weights: &[f32], step: f32, output: &mut [u8]) -> usize {
    let num_blocks = weights.len() / 8;
    if num_blocks == 0 {
        return 0;
    }

    let inv_step = if step.abs() > 1e-10 { 1.0 / step } else { 0.0 };

    unsafe {
        let weights_ptr = weights.as_ptr();
        let output_ptr = output.as_mut_ptr();

        for block in 0..num_blocks {
            let w_offset = block * 8;
            let o_offset = block * 3;

            let mut combined: u32 = 0;
            for i in 0..8 {
                let w = *weights_ptr.add(w_offset + i);
                let q = (w * inv_step).round() as i32;
                let clamped = q.clamp(-4, 3);
                let unsigned = (clamped + 4) as u32;
                combined |= (unsigned & 0x7) << (i * 3);
            }

            *output_ptr.add(o_offset) = (combined & 0xFF) as u8;
            *output_ptr.add(o_offset + 1) = ((combined >> 8) & 0xFF) as u8;
            *output_ptr.add(o_offset + 2) = ((combined >> 16) & 0xFF) as u8;
        }
    }

    num_blocks * 3
}

/// Optimized scalar 2-bit quantization with pre-allocated buffer
fn quantize_2bit_fast(weights: &[f32], step: f32, output: &mut [u8]) -> usize {
    let num_blocks = weights.len() / 4;
    if num_blocks == 0 {
        return 0;
    }

    let inv_step = if step.abs() > 1e-10 { 1.0 / step } else { 0.0 };

    unsafe {
        let weights_ptr = weights.as_ptr();
        let output_ptr = output.as_mut_ptr();

        for block in 0..num_blocks {
            let w_offset = block * 4;

            let w0 = *weights_ptr.add(w_offset);
            let w1 = *weights_ptr.add(w_offset + 1);
            let w2 = *weights_ptr.add(w_offset + 2);
            let w3 = *weights_ptr.add(w_offset + 3);

            let q0 = ((w0 * inv_step).round() as i32).clamp(-2, 1);
            let q1 = ((w1 * inv_step).round() as i32).clamp(-2, 1);
            let q2 = ((w2 * inv_step).round() as i32).clamp(-2, 1);
            let q3 = ((w3 * inv_step).round() as i32).clamp(-2, 1);

            *output_ptr.add(block) = ((q0 + 2) as u8 & 0x03)
                | (((q1 + 2) as u8 & 0x03) << 2)
                | (((q2 + 2) as u8 & 0x03) << 4)
                | (((q3 + 2) as u8 & 0x03) << 6);
        }
    }

    num_blocks
}

/// NEON 3-bit quantization
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_3bit_neon(weights: &[f32], step: f32, output: &mut [u8]) -> usize {
    use core::arch::aarch64::*;

    let num_blocks = weights.len() / 8;
    if num_blocks == 0 {
        return 0;
    }

    let inv_step = if step.abs() > 1e-10 { 1.0 / step } else { 0.0 };
    let inv_step_vec = vdupq_n_f32(inv_step);
    let min_val = vdupq_n_s32(-4);
    let max_val = vdupq_n_s32(3);
    let offset = vdupq_n_s32(4);

    let weights_ptr = weights.as_ptr();
    let output_ptr = output.as_mut_ptr();

    let simd_iterations = num_blocks / 4;
    let mut block = 0usize;

    while block < simd_iterations * 4 {
        for inner in 0..4 {
            let b = block + inner;
            let w_offset = b * 8;
            let o_offset = b * 3;

            let w_lo = vld1q_f32(weights_ptr.add(w_offset));
            let w_hi = vld1q_f32(weights_ptr.add(w_offset + 4));

            let scaled_lo = vmulq_f32(w_lo, inv_step_vec);
            let scaled_hi = vmulq_f32(w_hi, inv_step_vec);

            let rounded_lo = vrndnq_f32(scaled_lo);
            let rounded_hi = vrndnq_f32(scaled_hi);

            let q_lo = vcvtq_s32_f32(rounded_lo);
            let q_hi = vcvtq_s32_f32(rounded_hi);

            let clamped_lo = vminq_s32(vmaxq_s32(q_lo, min_val), max_val);
            let clamped_hi = vminq_s32(vmaxq_s32(q_hi, min_val), max_val);

            let unsigned_lo = vaddq_s32(clamped_lo, offset);
            let unsigned_hi = vaddq_s32(clamped_hi, offset);

            let mut vals = [0u32; 8];
            vst1q_s32(vals.as_mut_ptr() as *mut i32, unsigned_lo);
            vst1q_s32(vals.as_mut_ptr().add(4) as *mut i32, unsigned_hi);

            let mut combined: u32 = 0;
            for i in 0..8 {
                combined |= (vals[i] & 0x7) << (i * 3);
            }

            *output_ptr.add(o_offset) = (combined & 0xFF) as u8;
            *output_ptr.add(o_offset + 1) = ((combined >> 8) & 0xFF) as u8;
            *output_ptr.add(o_offset + 2) = ((combined >> 16) & 0xFF) as u8;
        }
        block += 4;
    }

    while block < num_blocks {
        let w_offset = block * 8;
        let o_offset = block * 3;

        let mut combined: u32 = 0;
        for i in 0..8 {
            let w = *weights_ptr.add(w_offset + i);
            let q = (w * inv_step).round() as i32;
            let clamped = q.clamp(-4, 3);
            let unsigned = (clamped + 4) as u32;
            combined |= (unsigned & 0x7) << (i * 3);
        }

        *output_ptr.add(o_offset) = (combined & 0xFF) as u8;
        *output_ptr.add(o_offset + 1) = ((combined >> 8) & 0xFF) as u8;
        *output_ptr.add(o_offset + 2) = ((combined >> 16) & 0xFF) as u8;

        block += 1;
    }

    num_blocks * 3
}

/// NEON 2-bit quantization
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_2bit_neon(weights: &[f32], step: f32, output: &mut [u8]) -> usize {
    use core::arch::aarch64::*;

    let num_blocks = weights.len() / 4;
    if num_blocks == 0 {
        return 0;
    }

    let inv_step = if step.abs() > 1e-10 { 1.0 / step } else { 0.0 };
    let inv_step_vec = vdupq_n_f32(inv_step);
    let min_val = vdupq_n_s32(-2);
    let max_val = vdupq_n_s32(1);
    let offset = vdupq_n_s32(2);

    let weights_ptr = weights.as_ptr();
    let output_ptr = output.as_mut_ptr();

    let simd_iterations = num_blocks / 4;
    let mut block = 0usize;

    while block < simd_iterations * 4 {
        let w0 = vld1q_f32(weights_ptr.add(block * 4));
        let w1 = vld1q_f32(weights_ptr.add((block + 1) * 4));
        let w2 = vld1q_f32(weights_ptr.add((block + 2) * 4));
        let w3 = vld1q_f32(weights_ptr.add((block + 3) * 4));

        let scaled0 = vmulq_f32(w0, inv_step_vec);
        let scaled1 = vmulq_f32(w1, inv_step_vec);
        let scaled2 = vmulq_f32(w2, inv_step_vec);
        let scaled3 = vmulq_f32(w3, inv_step_vec);

        let rounded0 = vrndnq_f32(scaled0);
        let rounded1 = vrndnq_f32(scaled1);
        let rounded2 = vrndnq_f32(scaled2);
        let rounded3 = vrndnq_f32(scaled3);

        let q0 = vminq_s32(vmaxq_s32(vcvtq_s32_f32(rounded0), min_val), max_val);
        let q1 = vminq_s32(vmaxq_s32(vcvtq_s32_f32(rounded1), min_val), max_val);
        let q2 = vminq_s32(vmaxq_s32(vcvtq_s32_f32(rounded2), min_val), max_val);
        let q3 = vminq_s32(vmaxq_s32(vcvtq_s32_f32(rounded3), min_val), max_val);

        let u0 = vaddq_s32(q0, offset);
        let u1 = vaddq_s32(q1, offset);
        let u2 = vaddq_s32(q2, offset);
        let u3 = vaddq_s32(q3, offset);

        let mut vals0 = [0i32; 4];
        let mut vals1 = [0i32; 4];
        let mut vals2 = [0i32; 4];
        let mut vals3 = [0i32; 4];

        vst1q_s32(vals0.as_mut_ptr(), u0);
        vst1q_s32(vals1.as_mut_ptr(), u1);
        vst1q_s32(vals2.as_mut_ptr(), u2);
        vst1q_s32(vals3.as_mut_ptr(), u3);

        *output_ptr.add(block) = ((vals0[0] as u8) & 0x03)
            | (((vals0[1] as u8) & 0x03) << 2)
            | (((vals0[2] as u8) & 0x03) << 4)
            | (((vals0[3] as u8) & 0x03) << 6);

        *output_ptr.add(block + 1) = ((vals1[0] as u8) & 0x03)
            | (((vals1[1] as u8) & 0x03) << 2)
            | (((vals1[2] as u8) & 0x03) << 4)
            | (((vals1[3] as u8) & 0x03) << 6);

        *output_ptr.add(block + 2) = ((vals2[0] as u8) & 0x03)
            | (((vals2[1] as u8) & 0x03) << 2)
            | (((vals2[2] as u8) & 0x03) << 4)
            | (((vals2[3] as u8) & 0x03) << 6);

        *output_ptr.add(block + 3) = ((vals3[0] as u8) & 0x03)
            | (((vals3[1] as u8) & 0x03) << 2)
            | (((vals3[2] as u8) & 0x03) << 4)
            | (((vals3[3] as u8) & 0x03) << 6);

        block += 4;
    }

    while block < num_blocks {
        let w_offset = block * 4;

        let w0 = *weights_ptr.add(w_offset);
        let w1 = *weights_ptr.add(w_offset + 1);
        let w2 = *weights_ptr.add(w_offset + 2);
        let w3 = *weights_ptr.add(w_offset + 3);

        let q0 = ((w0 * inv_step).round() as i32).clamp(-2, 1);
        let q1 = ((w1 * inv_step).round() as i32).clamp(-2, 1);
        let q2 = ((w2 * inv_step).round() as i32).clamp(-2, 1);
        let q3 = ((w3 * inv_step).round() as i32).clamp(-2, 1);

        *output_ptr.add(block) = ((q0 + 2) as u8 & 0x03)
            | (((q1 + 2) as u8 & 0x03) << 2)
            | (((q2 + 2) as u8 & 0x03) << 4)
            | (((q3 + 2) as u8 & 0x03) << 6);

        block += 1;
    }

    num_blocks
}

/// AVX2 3-bit quantization
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_3bit_avx2(weights: &[f32], step: f32, output: &mut [u8]) -> usize {
    use core::arch::x86_64::*;

    let num_blocks = weights.len() / 8;
    if num_blocks == 0 {
        return 0;
    }

    let inv_step = if step.abs() > 1e-10 { 1.0 / step } else { 0.0 };
    let inv_step_vec = _mm256_set1_ps(inv_step);
    let min_val = _mm256_set1_epi32(-4);
    let max_val = _mm256_set1_epi32(3);
    let offset = _mm256_set1_epi32(4);

    let weights_ptr = weights.as_ptr();
    let output_ptr = output.as_mut_ptr();

    for block in 0..num_blocks {
        let w_offset = block * 8;
        let o_offset = block * 3;

        let w = _mm256_loadu_ps(weights_ptr.add(w_offset));
        let scaled = _mm256_mul_ps(w, inv_step_vec);
        let rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let q = _mm256_cvtps_epi32(rounded);
        let clamped = _mm256_min_epi32(_mm256_max_epi32(q, min_val), max_val);
        let unsigned = _mm256_add_epi32(clamped, offset);

        let mut vals = [0i32; 8];
        _mm256_storeu_si256(vals.as_mut_ptr() as *mut __m256i, unsigned);

        let mut combined: u32 = 0;
        for i in 0..8 {
            combined |= ((vals[i] as u32) & 0x7) << (i * 3);
        }

        *output_ptr.add(o_offset) = (combined & 0xFF) as u8;
        *output_ptr.add(o_offset + 1) = ((combined >> 8) & 0xFF) as u8;
        *output_ptr.add(o_offset + 2) = ((combined >> 16) & 0xFF) as u8;
    }

    num_blocks * 3
}

/// AVX2 2-bit quantization
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_2bit_avx2(weights: &[f32], step: f32, output: &mut [u8]) -> usize {
    use core::arch::x86_64::*;

    let num_blocks = weights.len() / 4;
    if num_blocks == 0 {
        return 0;
    }

    let inv_step = if step.abs() > 1e-10 { 1.0 / step } else { 0.0 };
    let inv_step_vec = _mm_set1_ps(inv_step);
    let min_val = _mm_set1_epi32(-2);
    let max_val = _mm_set1_epi32(1);
    let offset = _mm_set1_epi32(2);

    let weights_ptr = weights.as_ptr();
    let output_ptr = output.as_mut_ptr();

    for block in 0..num_blocks {
        let w_offset = block * 4;

        let w = _mm_loadu_ps(weights_ptr.add(w_offset));
        let scaled = _mm_mul_ps(w, inv_step_vec);
        let rounded = _mm_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        let q = _mm_cvtps_epi32(rounded);
        let clamped = _mm_min_epi32(_mm_max_epi32(q, min_val), max_val);
        let unsigned = _mm_add_epi32(clamped, offset);

        let mut vals = [0i32; 4];
        _mm_storeu_si128(vals.as_mut_ptr() as *mut __m128i, unsigned);

        *output_ptr.add(block) = ((vals[0] as u8) & 0x03)
            | (((vals[1] as u8) & 0x03) << 2)
            | (((vals[2] as u8) & 0x03) << 4)
            | (((vals[3] as u8) & 0x03) << 6);
    }

    num_blocks
}

/// Dispatch to best quantization kernel
fn quantize_3bit_dispatch(weights: &[f32], step: f32, output: &mut [u8]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return quantize_3bit_neon(weights, step, output);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return quantize_3bit_avx2(weights, step, output);
            }
        }
    }

    quantize_3bit_fast(weights, step, output)
}

fn quantize_2bit_dispatch(weights: &[f32], step: f32, output: &mut [u8]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            return quantize_2bit_neon(weights, step, output);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return quantize_2bit_avx2(weights, step, output);
            }
        }
    }

    quantize_2bit_fast(weights, step, output)
}

/// Benchmark: Optimized 3-bit quantization (scalar, pre-allocated)
/// Target: >1 GB/s
fn bench_pi_quantize_3bit_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi_quantize_3bit_fast");
    group.sample_size(100);

    let step = PI / 4.0;

    for &size in &[256, 4096, 4096 * 11008] {
        let weights = random_weights(size);
        let num_blocks = size / 8;
        let output_bytes = num_blocks * 3;
        let mut output = vec![0u8; output_bytes];

        group.throughput(Throughput::Bytes(output_bytes as u64));
        group.bench_with_input(BenchmarkId::new("size", size), &weights, |b, w| {
            b.iter(|| quantize_3bit_fast(black_box(w), step, black_box(&mut output)))
        });
    }

    group.finish();
}

/// Benchmark: Optimized 2-bit quantization (scalar, pre-allocated)
/// Target: >1 GB/s
fn bench_pi_quantize_2bit_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi_quantize_2bit_fast");
    group.sample_size(100);

    let step = PI / 4.0;

    for &size in &[256, 4096, 4096 * 11008] {
        let weights = random_weights(size);
        let num_blocks = size / 4;
        let mut output = vec![0u8; num_blocks];

        group.throughput(Throughput::Bytes(num_blocks as u64));
        group.bench_with_input(BenchmarkId::new("size", size), &weights, |b, w| {
            b.iter(|| quantize_2bit_fast(black_box(w), step, black_box(&mut output)))
        });
    }

    group.finish();
}

/// Benchmark: SIMD dispatched 3-bit quantization
/// Target: >1 GB/s
fn bench_pi_quantize_3bit_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi_quantize_3bit_simd");
    group.sample_size(100);

    let step = PI / 4.0;

    for &size in &[256, 4096, 4096 * 11008] {
        let weights = random_weights(size);
        let num_blocks = size / 8;
        let output_bytes = num_blocks * 3;
        let mut output = vec![0u8; output_bytes];

        group.throughput(Throughput::Bytes(output_bytes as u64));
        group.bench_with_input(BenchmarkId::new("size", size), &weights, |b, w| {
            b.iter(|| quantize_3bit_dispatch(black_box(w), step, black_box(&mut output)))
        });
    }

    group.finish();
}

/// Benchmark: SIMD dispatched 2-bit quantization
/// Target: >1 GB/s
fn bench_pi_quantize_2bit_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi_quantize_2bit_simd");
    group.sample_size(100);

    let step = PI / 4.0;

    for &size in &[256, 4096, 4096 * 11008] {
        let weights = random_weights(size);
        let num_blocks = size / 4;
        let mut output = vec![0u8; num_blocks];

        group.throughput(Throughput::Bytes(num_blocks as u64));
        group.bench_with_input(BenchmarkId::new("size", size), &weights, |b, w| {
            b.iter(|| quantize_2bit_dispatch(black_box(w), step, black_box(&mut output)))
        });
    }

    group.finish();
}

/// Benchmark: NEON 3-bit quantization specifically
#[cfg(target_arch = "aarch64")]
fn bench_pi_quantize_3bit_neon(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi_quantize_3bit_neon");
    group.sample_size(100);

    let step = PI / 4.0;

    for &size in &[256, 4096, 4096 * 1024, 4096 * 11008] {
        let weights = random_weights(size);
        let num_blocks = size / 8;
        let output_bytes = num_blocks * 3;
        let mut output = vec![0u8; output_bytes];

        group.throughput(Throughput::Bytes(output_bytes as u64));
        group.bench_with_input(BenchmarkId::new("weights", size), &weights, |b, w| {
            b.iter(|| unsafe { quantize_3bit_neon(black_box(w), step, black_box(&mut output)) })
        });
    }

    group.finish();
}

/// Benchmark: NEON 2-bit quantization specifically
#[cfg(target_arch = "aarch64")]
fn bench_pi_quantize_2bit_neon(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi_quantize_2bit_neon");
    group.sample_size(100);

    let step = PI / 4.0;

    for &size in &[256, 4096, 4096 * 1024, 4096 * 11008] {
        let weights = random_weights(size);
        let num_blocks = size / 4;
        let mut output = vec![0u8; num_blocks];

        group.throughput(Throughput::Bytes(num_blocks as u64));
        group.bench_with_input(BenchmarkId::new("weights", size), &weights, |b, w| {
            b.iter(|| unsafe { quantize_2bit_neon(black_box(w), step, black_box(&mut output)) })
        });
    }

    group.finish();
}

/// Benchmark: AVX2 3-bit quantization specifically
#[cfg(target_arch = "x86_64")]
fn bench_pi_quantize_3bit_avx2(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let mut group = c.benchmark_group("pi_quantize_3bit_avx2");
    group.sample_size(100);

    let step = PI / 4.0;

    for &size in &[256, 4096, 4096 * 1024, 4096 * 11008] {
        let weights = random_weights(size);
        let num_blocks = size / 8;
        let output_bytes = num_blocks * 3;
        let mut output = vec![0u8; output_bytes];

        group.throughput(Throughput::Bytes(output_bytes as u64));
        group.bench_with_input(BenchmarkId::new("weights", size), &weights, |b, w| {
            b.iter(|| unsafe { quantize_3bit_avx2(black_box(w), step, black_box(&mut output)) })
        });
    }

    group.finish();
}

/// Benchmark: AVX2 2-bit quantization specifically
#[cfg(target_arch = "x86_64")]
fn bench_pi_quantize_2bit_avx2(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let mut group = c.benchmark_group("pi_quantize_2bit_avx2");
    group.sample_size(100);

    let step = PI / 4.0;

    for &size in &[256, 4096, 4096 * 1024, 4096 * 11008] {
        let weights = random_weights(size);
        let num_blocks = size / 4;
        let mut output = vec![0u8; num_blocks];

        group.throughput(Throughput::Bytes(num_blocks as u64));
        group.bench_with_input(BenchmarkId::new("weights", size), &weights, |b, w| {
            b.iter(|| unsafe { quantize_2bit_avx2(black_box(w), step, black_box(&mut output)) })
        });
    }

    group.finish();
}

/// Benchmark: Pi-Quantization 2-bit throughput
/// Target: >1 GB/s
fn bench_pi_quantize_2bit(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi_quantize_2bit");
    group.sample_size(100);

    let quantizer = PiQuantizer::new(2, 4, 1.0);

    for &size in &[256, 4096, 4096 * 11008] {
        let weights = random_weights(size);
        let num_blocks = size / 4;
        let output_bytes = num_blocks;

        group.throughput(Throughput::Bytes(output_bytes as u64));
        group.bench_with_input(BenchmarkId::new("size", size), &weights, |b, w| {
            b.iter(|| {
                let mut blocks = Vec::with_capacity(num_blocks);
                for chunk in w.chunks_exact(4) {
                    let arr: [f32; 4] = chunk.try_into().unwrap();
                    blocks.push(black_box(quantizer.quantize_block_2bit(&arr)));
                }
                blocks
            })
        });
    }

    group.finish();
}

/// Benchmark: Scalar dequantization (baseline)
fn bench_pi_dequantize_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi_dequantize_scalar");
    group.sample_size(100);

    let scale = PI / 4.0;

    for &num_weights in &[256, 4096, 4096 * 1024] {
        let packed = random_packed_3bit(num_weights);
        let mut output = vec![0.0f32; num_weights];
        let input_bytes = packed.len();

        group.throughput(Throughput::Bytes(input_bytes as u64));
        group.bench_with_input(BenchmarkId::new("weights", num_weights), &packed, |b, p| {
            b.iter(|| {
                pi_dequantize_scalar(black_box(p), scale, black_box(&mut output));
            })
        });
    }

    group.finish();
}

/// Benchmark: NEON dequantization
/// Target: >10 GB/s on ARM
#[cfg(target_arch = "aarch64")]
fn bench_pi_dequantize_neon(c: &mut Criterion) {
    let mut group = c.benchmark_group("pi_dequantize_neon");
    group.sample_size(100);

    let scale = PI / 4.0;

    for &num_weights in &[256, 4096, 4096 * 1024, 4096 * 11008] {
        let packed = random_packed_3bit(num_weights);
        let mut output = vec![0.0f32; num_weights];
        let input_bytes = packed.len();

        group.throughput(Throughput::Bytes(input_bytes as u64));
        group.bench_with_input(BenchmarkId::new("weights", num_weights), &packed, |b, p| {
            b.iter(|| unsafe {
                pi_dequantize_neon(black_box(p), scale, black_box(&mut output));
            })
        });
    }

    group.finish();
}

/// Benchmark: AVX2 dequantization
/// Target: >10 GB/s on x86
#[cfg(target_arch = "x86_64")]
fn bench_pi_dequantize_avx2(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    let mut group = c.benchmark_group("pi_dequantize_avx2");
    group.sample_size(100);

    let scale = PI / 4.0;

    for &num_weights in &[256, 4096, 4096 * 1024, 4096 * 11008] {
        let packed = random_packed_3bit(num_weights);
        let mut output = vec![0.0f32; num_weights];
        let input_bytes = packed.len();

        group.throughput(Throughput::Bytes(input_bytes as u64));
        group.bench_with_input(BenchmarkId::new("weights", num_weights), &packed, |b, p| {
            b.iter(|| unsafe {
                pi_dequantize_avx2(black_box(p), scale, black_box(&mut output));
            })
        });
    }

    group.finish();
}

/// Benchmark: Hadamard transform (scalar)
/// Target: <50μs for 4096
fn bench_hadamard_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("hadamard_scalar");
    group.sample_size(100);

    for &size in &[256, 4096, 16384] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) / size as f32).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("size", size), &size, |b, _| {
            b.iter(|| {
                let mut work = data.clone();
                hadamard_transform(black_box(&mut work));
                work
            })
        });
    }

    group.finish();
}

/// Benchmark: Hadamard transform (NEON)
#[cfg(target_arch = "aarch64")]
fn bench_hadamard_neon(c: &mut Criterion) {
    let mut group = c.benchmark_group("hadamard_neon");
    group.sample_size(100);

    for &size in &[256, 4096, 16384] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) / size as f32).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("size", size), &size, |b, _| {
            b.iter(|| {
                let mut work = data.clone();
                unsafe {
                    hadamard_transform_neon(black_box(&mut work));
                }
                work
            })
        });
    }

    group.finish();
}

/// Benchmark: Hadamard for typical layer sizes
fn bench_hadamard_layer_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("hadamard_layers");
    group.sample_size(50);

    // Common layer dimensions (rounded to power of 2)
    for &size in &[256, 4096, 8192, 16384] {
        let data: Vec<f32> = (0..size)
            .map(|i| (i as f32 - size as f32 / 2.0) / 100.0)
            .collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("dim", size), &data, |b, d| {
            b.iter(|| {
                let mut work = d.clone();
                hadamard_transform(black_box(&mut work));
                work
            })
        });
    }

    group.finish();
}

/// Benchmark: QAT forward pass
fn bench_qat_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("qat_forward");
    group.sample_size(100);

    let quantizer = PiQuantizer::new(3, 4, 1.0);

    for &size in &[256, 4096, 4096 * 1024] {
        let weights = random_weights(size);
        let mut output = vec![0.0f32; size];

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("weights", size), &weights, |b, w| {
            b.iter(|| {
                qat_forward(black_box(w), &quantizer, black_box(&mut output));
            })
        });
    }

    group.finish();
}

/// Benchmark: QAT backward (STE)
fn bench_qat_backward_ste(c: &mut Criterion) {
    let mut group = c.benchmark_group("qat_backward_ste");
    group.sample_size(100);

    let ste_variants = [
        ("standard", SteVariant::Standard),
        ("clipped", SteVariant::Clipped { clip_val: 1.0 }),
        ("ewgs", SteVariant::Ewgs { lambda: 0.1 }),
    ];

    for &size in &[256, 4096, 4096 * 1024] {
        let weights = random_weights(size);
        let quantized: Vec<f32> = weights.iter().map(|&w| (w * 4.0).round() / 4.0).collect();
        let grad_out: Vec<f32> = random_weights(size);
        let mut grad_w = vec![0.0f32; size];

        for (name, ref ste) in &ste_variants {
            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{}_weights", name), size),
                &(&weights, &quantized, &grad_out),
                |b, (w, q, g)| {
                    b.iter(|| {
                        qat_backward_ste(
                            black_box(*w),
                            black_box(*q),
                            black_box(*g),
                            black_box(&mut grad_w),
                            ste,
                        );
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark: MSE computation
fn bench_mse_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_mse");
    group.sample_size(100);

    for &size in &[256, 4096, 4096 * 1024] {
        let original = random_weights(size);
        let quantized: Vec<f32> = original.iter().map(|&w| (w * 4.0).round() / 4.0).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("weights", size),
            &(&original, &quantized),
            |b, (o, q)| b.iter(|| compute_mse(black_box(*o), black_box(*q))),
        );
    }

    group.finish();
}

/// Benchmark: Spectral distortion
fn bench_spectral_distortion(c: &mut Criterion) {
    let mut group = c.benchmark_group("quality_spectral");
    group.sample_size(100);

    for &size in &[256, 4096, 4096 * 1024] {
        let original = random_weights(size);
        let quantized: Vec<f32> = original.iter().map(|&w| (w * 4.0).round() / 4.0).collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("weights", size),
            &(&original, &quantized),
            |b, (o, q)| b.iter(|| compute_spectral_distortion(black_box(*o), black_box(*q))),
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

#[cfg(target_arch = "aarch64")]
criterion_group!(
    benches,
    // Original (Vec-allocating) benchmarks
    bench_pi_quantize_3bit,
    bench_pi_quantize_2bit,
    // NEW: Optimized scalar benchmarks (pre-allocated)
    bench_pi_quantize_3bit_fast,
    bench_pi_quantize_2bit_fast,
    // NEW: SIMD dispatched benchmarks
    bench_pi_quantize_3bit_simd,
    bench_pi_quantize_2bit_simd,
    // NEW: Architecture-specific NEON benchmarks
    bench_pi_quantize_3bit_neon,
    bench_pi_quantize_2bit_neon,
    // Dequantization benchmarks
    bench_pi_dequantize_scalar,
    bench_pi_dequantize_neon,
    // Hadamard benchmarks
    bench_hadamard_scalar,
    bench_hadamard_neon,
    bench_hadamard_layer_sizes,
    // QAT benchmarks
    bench_qat_forward,
    bench_qat_backward_ste,
    // Quality metrics
    bench_mse_computation,
    bench_spectral_distortion,
);

#[cfg(target_arch = "x86_64")]
criterion_group!(
    benches,
    // Original (Vec-allocating) benchmarks
    bench_pi_quantize_3bit,
    bench_pi_quantize_2bit,
    // NEW: Optimized scalar benchmarks (pre-allocated)
    bench_pi_quantize_3bit_fast,
    bench_pi_quantize_2bit_fast,
    // NEW: SIMD dispatched benchmarks
    bench_pi_quantize_3bit_simd,
    bench_pi_quantize_2bit_simd,
    // NEW: Architecture-specific AVX2 benchmarks
    bench_pi_quantize_3bit_avx2,
    bench_pi_quantize_2bit_avx2,
    // Dequantization benchmarks
    bench_pi_dequantize_scalar,
    bench_pi_dequantize_avx2,
    // Hadamard benchmarks
    bench_hadamard_scalar,
    bench_hadamard_layer_sizes,
    // QAT benchmarks
    bench_qat_forward,
    bench_qat_backward_ste,
    // Quality metrics
    bench_mse_computation,
    bench_spectral_distortion,
);

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
criterion_group!(
    benches,
    // Original (Vec-allocating) benchmarks
    bench_pi_quantize_3bit,
    bench_pi_quantize_2bit,
    // NEW: Optimized scalar benchmarks (pre-allocated)
    bench_pi_quantize_3bit_fast,
    bench_pi_quantize_2bit_fast,
    // NEW: SIMD dispatched benchmarks
    bench_pi_quantize_3bit_simd,
    bench_pi_quantize_2bit_simd,
    // Dequantization benchmarks
    bench_pi_dequantize_scalar,
    // Hadamard benchmarks
    bench_hadamard_scalar,
    bench_hadamard_layer_sizes,
    // QAT benchmarks
    bench_qat_forward,
    bench_qat_backward_ste,
    // Quality metrics
    bench_mse_computation,
    bench_spectral_distortion,
);

criterion_main!(benches);
