# CNN Implementation Patterns in Rust

## Executive Summary

This document covers Rust-specific patterns for implementing CNNs with SIMD acceleration. It covers available crates, SIMD intrinsics usage, memory layout optimization, and practical implementation patterns suitable for integration with RuVector.

## Rust Deep Learning Ecosystem

### Available Frameworks

| Crate | Description | SIMD Support | Maturity |
|-------|-------------|--------------|----------|
| `tch-rs` | PyTorch bindings | Via LibTorch | Production |
| `burn` | Native Rust ML | Backend-dependent | Growing |
| `candle` | Native Rust inference | SIMD + Metal | Production |
| `ndarray` | N-dimensional arrays | Via BLAS | Mature |
| `neuronika` | Autograd + NN layers | Experimental | Research |
| `tract` | ONNX inference | SIMD kernels | Production |

### Recommended Stack for RuVector

For CPU-focused CNN inference with SIMD:

```toml
[dependencies]
# Core tensor operations
ndarray = { version = "0.16", features = ["blas"] }

# SIMD support
simdeez = "1.0"  # Cross-platform SIMD

# Optional: ONNX model loading
tract-onnx = "0.21"

# Linear algebra
nalgebra = { version = "0.33", features = ["simd"] }
```

## SIMD in Rust: `std::arch` vs `portable_simd`

### Using `std::arch` (Stable)

Architecture-specific intrinsics are available in `std::arch`:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 vectorized dot product
/// Safety: Must be called on AVX2-capable CPUs
#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    let sum128 = _mm_add_ps(
        _mm256_extractf128_ps(sum, 0),
        _mm256_extractf128_ps(sum, 1),
    );
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result: f32 = 0.0;
    _mm_store_ss(&mut result, sum32);

    // Handle remainder
    for i in (chunks * 8)..a.len() {
        result += a[i] * b[i];
    }

    result
}
```

### Runtime Feature Detection

```rust
use std::arch::is_x86_feature_detected;

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { dot_product_sse41(a, b) };
        }
    }

    // Scalar fallback
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
```

### Using `portable_simd` (Nightly)

Cross-platform SIMD without architecture-specific code:

```rust
#![feature(portable_simd)]
use std::simd::{f32x8, SimdFloat, Simd};

pub fn dot_product_portable(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::splat(0.0);
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let va = f32x8::from_slice(&a[i * 8..]);
        let vb = f32x8::from_slice(&b[i * 8..]);
        sum += va * vb;
    }

    let mut result = sum.reduce_sum();

    // Remainder
    for i in (chunks * 8)..a.len() {
        result += a[i] * b[i];
    }

    result
}
```

### Using SIMDeez (Stable + Portable)

```rust
use simdeez::*;
use simdeez::prelude::*;

simd_runtime_generate!(
    fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = S::zerof32();
        let chunks = a.len() / S::F32_WIDTH;

        for i in 0..chunks {
            let va = S::loadu_ps(&a[i * S::F32_WIDTH]);
            let vb = S::loadu_ps(&b[i * S::F32_WIDTH]);
            sum = S::fmadd_ps(va, vb, sum);
        }

        let mut result = S::horizontal_add_ps(sum);

        for i in (chunks * S::F32_WIDTH)..a.len() {
            result += a[i] * b[i];
        }

        result
    }
);

// Usage - automatically selects best available
let result = dot_product_simd(&a, &b);
```

## CNN Layer Implementations

### Convolution 2D

```rust
use ndarray::{Array4, ArrayView4, s};

/// Memory layout: NHWC (batch, height, width, channels)
/// More SIMD-friendly than NCHW for channel-wise operations
pub struct Conv2d {
    weights: Array4<f32>,  // [out_channels, kh, kw, in_channels]
    bias: Option<Vec<f32>>,
    stride: (usize, usize),
    padding: (usize, usize),
}

impl Conv2d {
    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        let (batch, h_in, w_in, c_in) = input.dim();
        let (c_out, kh, kw, _) = self.weights.dim();

        let h_out = (h_in + 2 * self.padding.0 - kh) / self.stride.0 + 1;
        let w_out = (w_in + 2 * self.padding.1 - kw) / self.stride.1 + 1;

        let mut output = Array4::zeros((batch, h_out, w_out, c_out));

        // For each output position
        for b in 0..batch {
            for y in 0..h_out {
                for x in 0..w_out {
                    let y_in = y * self.stride.0;
                    let x_in = x * self.stride.1;

                    // Extract patch
                    let patch = self.extract_patch(input, b, y_in, x_in);

                    // Compute all output channels (SIMD-friendly)
                    for c in 0..c_out {
                        let kernel = self.weights.slice(s![c, .., .., ..]);
                        let dot = dot_product(
                            patch.as_slice().unwrap(),
                            kernel.as_slice().unwrap(),
                        );
                        output[[b, y, x, c]] = dot + self.bias.as_ref().map_or(0.0, |b| b[c]);
                    }
                }
            }
        }

        output
    }
}
```

### Depthwise Separable Convolution

```rust
/// MobileNet-style depthwise separable convolution
/// = Depthwise (spatial) + Pointwise (1x1)
pub struct DepthwiseSeparableConv {
    depthwise: Array4<f32>,  // [1, kh, kw, channels]
    pointwise: Array2<f32>,  // [out_channels, in_channels]
    dw_bias: Option<Vec<f32>>,
    pw_bias: Option<Vec<f32>>,
}

impl DepthwiseSeparableConv {
    #[target_feature(enable = "avx2")]
    #[cfg(target_arch = "x86_64")]
    unsafe fn depthwise_avx2(
        &self,
        input: &Array4<f32>,
        output: &mut Array4<f32>,
    ) {
        let (batch, h, w, c) = input.dim();
        let (_, kh, kw, _) = self.depthwise.dim();

        // Process 8 channels at a time
        for b in 0..batch {
            for y in 1..(h - 1) {
                for x in 1..(w - 1) {
                    for c_base in (0..c).step_by(8) {
                        let mut acc = _mm256_setzero_ps();

                        // 3x3 kernel
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let y_in = y + ky - 1;
                                let x_in = x + kx - 1;

                                let inp_ptr = input
                                    .as_ptr()
                                    .add(((b * h + y_in) * w + x_in) * c + c_base);
                                let ker_ptr = self.depthwise
                                    .as_ptr()
                                    .add((ky * kw + kx) * c + c_base);

                                let inp = _mm256_loadu_ps(inp_ptr);
                                let ker = _mm256_loadu_ps(ker_ptr);
                                acc = _mm256_fmadd_ps(inp, ker, acc);
                            }
                        }

                        let out_ptr = output
                            .as_mut_ptr()
                            .add(((b * h + y) * w + x) * c + c_base);
                        _mm256_storeu_ps(out_ptr, acc);
                    }
                }
            }
        }
    }
}
```

### Batch Normalization

```rust
pub struct BatchNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    epsilon: f32,
}

impl BatchNorm {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg(target_arch = "x86_64")]
    unsafe fn forward_avx2(&self, data: &mut [f32], channels: usize) {
        let spatial_size = data.len() / channels;

        for c in 0..channels {
            // Pre-compute: scale = gamma / sqrt(var + eps)
            let inv_std = 1.0 / (self.running_var[c] + self.epsilon).sqrt();
            let scale = self.gamma[c] * inv_std;
            let shift = self.beta[c] - self.running_mean[c] * scale;

            let scale_v = _mm256_set1_ps(scale);
            let shift_v = _mm256_set1_ps(shift);

            let channel_start = c * spatial_size;
            let channel_data = &mut data[channel_start..channel_start + spatial_size];

            // Process 8 elements at a time
            for chunk in channel_data.chunks_exact_mut(8) {
                let x = _mm256_loadu_ps(chunk.as_ptr());
                // y = x * scale + shift
                let y = _mm256_fmadd_ps(x, scale_v, shift_v);
                _mm256_storeu_ps(chunk.as_mut_ptr(), y);
            }

            // Handle remainder
            let remainder_start = (spatial_size / 8) * 8;
            for i in remainder_start..spatial_size {
                channel_data[i] = channel_data[i] * scale + shift;
            }
        }
    }
}
```

### Activation Functions

```rust
/// SIMD-accelerated activation functions
pub mod activations {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn relu_avx2(data: &mut [f32]) {
        let zero = _mm256_setzero_ps();

        for chunk in data.chunks_exact_mut(8) {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y = _mm256_max_ps(x, zero);
            _mm256_storeu_ps(chunk.as_mut_ptr(), y);
        }

        // Scalar remainder
        let start = (data.len() / 8) * 8;
        for x in &mut data[start..] {
            *x = x.max(0.0);
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn relu6_avx2(data: &mut [f32]) {
        let zero = _mm256_setzero_ps();
        let six = _mm256_set1_ps(6.0);

        for chunk in data.chunks_exact_mut(8) {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y = _mm256_min_ps(_mm256_max_ps(x, zero), six);
            _mm256_storeu_ps(chunk.as_mut_ptr(), y);
        }
    }

    /// Swish: x * sigmoid(x)
    /// Using polynomial approximation for sigmoid
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn swish_avx2(data: &mut [f32]) {
        for chunk in data.chunks_exact_mut(8) {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let sig = sigmoid_approx_avx2(x);
            let y = _mm256_mul_ps(x, sig);
            _mm256_storeu_ps(chunk.as_mut_ptr(), y);
        }
    }

    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn sigmoid_approx_avx2(x: __m256) -> __m256 {
        // Piecewise linear approximation
        // sigmoid(x) ~ 0.5 + 0.25*x for |x| < 2
        //            ~ 0 for x < -4
        //            ~ 1 for x > 4

        let half = _mm256_set1_ps(0.5);
        let quarter = _mm256_set1_ps(0.25);
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();

        // Linear region: 0.5 + 0.25*x
        let linear = _mm256_fmadd_ps(x, quarter, half);

        // Clamp to [0, 1]
        _mm256_min_ps(_mm256_max_ps(linear, zero), one)
    }
}
```

### Global Average Pooling

```rust
/// Global Average Pooling: reduce spatial dimensions
#[target_feature(enable = "avx2")]
#[cfg(target_arch = "x86_64")]
pub unsafe fn global_avg_pool_avx2(
    input: &[f32],     // [H, W, C]
    output: &mut [f32], // [C]
    h: usize,
    w: usize,
    c: usize,
) {
    let spatial = h * w;
    let inv_spatial = _mm256_set1_ps(1.0 / spatial as f32);

    // Process 8 channels at a time
    for c_base in (0..c).step_by(8) {
        let mut sum = _mm256_setzero_ps();

        // Sum all spatial positions for these 8 channels
        for s in 0..spatial {
            let ptr = input.as_ptr().add(s * c + c_base);
            let val = _mm256_loadu_ps(ptr);
            sum = _mm256_add_ps(sum, val);
        }

        // Divide by spatial size
        let avg = _mm256_mul_ps(sum, inv_spatial);
        _mm256_storeu_ps(output.as_mut_ptr().add(c_base), avg);
    }
}
```

## Memory Layout Optimization

### NHWC vs NCHW

```rust
/// NCHW: Batch, Channels, Height, Width (PyTorch default)
/// - Good for channel-wise operations
/// - Less SIMD-friendly for spatial operations

/// NHWC: Batch, Height, Width, Channels (TensorFlow default)
/// - SIMD-friendly for channel-wise vectorization
/// - Better cache locality for channel operations
/// - Recommended for CPU inference

pub enum Layout {
    NCHW,  // [N, C, H, W]
    NHWC,  // [N, H, W, C]
}

/// Convert NCHW to NHWC
pub fn nchw_to_nhwc(input: &Array4<f32>) -> Array4<f32> {
    input.permuted_axes([0, 2, 3, 1]).to_owned()
}
```

### Aligned Memory Allocation

```rust
use std::alloc::{alloc_zeroed, Layout};

/// Allocate 32-byte aligned memory for AVX2
pub fn alloc_aligned_f32(count: usize) -> Vec<f32> {
    let layout = Layout::from_size_align(
        count * std::mem::size_of::<f32>(),
        32, // 32-byte alignment for AVX2
    ).unwrap();

    unsafe {
        let ptr = alloc_zeroed(layout) as *mut f32;
        Vec::from_raw_parts(ptr, count, count)
    }
}

/// Aligned tensor wrapper
#[repr(align(32))]
pub struct AlignedTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}
```

## INT8 Quantization

### Quantization Infrastructure

```rust
pub struct QuantizedTensor {
    data: Vec<i8>,
    scale: f32,
    zero_point: i8,
    shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Quantize from f32 to i8
    pub fn from_f32(input: &[f32]) -> Self {
        let min = input.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let scale = (max - min) / 255.0;
        let zero_point = (-128.0 - min / scale).round() as i8;

        let data: Vec<i8> = input
            .iter()
            .map(|&x| ((x / scale).round() as i32 + zero_point as i32).clamp(-128, 127) as i8)
            .collect();

        Self {
            data,
            scale,
            zero_point,
            shape: vec![input.len()],
        }
    }

    /// Dequantize back to f32
    pub fn to_f32(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&q| (q as i32 - self.zero_point as i32) as f32 * self.scale)
            .collect()
    }
}
```

### INT8 Dot Product with VNNI

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni")]
pub unsafe fn dot_product_int8_vnni(
    a: &[u8],  // Unsigned activations
    b: &[i8],  // Signed weights
) -> i32 {
    use std::arch::x86_64::*;

    let mut acc = _mm512_setzero_si512();
    let chunks = a.len() / 64;

    for i in 0..chunks {
        let va = _mm512_loadu_si512(a.as_ptr().add(i * 64) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i * 64) as *const __m512i);

        // VNNI: 4-way byte dot product with accumulation
        acc = _mm512_dpbusd_epi32(acc, va, vb);
    }

    // Horizontal sum
    let sum256 = _mm256_add_epi32(
        _mm512_extracti64x4_epi64(acc, 0),
        _mm512_extracti64x4_epi64(acc, 1),
    );
    // ... continue reduction

    todo!("complete horizontal sum")
}
```

## Embedding Extraction

### Feature Extraction from CNN

```rust
/// Extract embeddings from penultimate layer
pub struct FeatureExtractor {
    backbone: MobileNetV3,
    embedding_dim: usize,
}

impl FeatureExtractor {
    pub fn extract(&self, image: &Array4<f32>) -> Vec<f32> {
        // Forward through backbone (excluding final classifier)
        let features = self.backbone.forward_features(image);

        // Global average pooling
        let (batch, h, w, c) = features.dim();
        let mut embedding = vec![0.0f32; batch * c];

        unsafe {
            global_avg_pool_avx2(
                features.as_slice().unwrap(),
                &mut embedding,
                h, w, c,
            );
        }

        // L2 normalize for cosine similarity
        l2_normalize(&mut embedding);

        embedding
    }
}

fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}
```

## Performance Benchmarks

### Expected Performance (Relative to Scalar)

| Operation | Scalar | AVX2 | AVX-512 |
|-----------|--------|------|---------|
| Dot Product | 1x | 8-10x | 14-16x |
| Conv 3x3 | 1x | 4-6x | 8-12x |
| BatchNorm | 1x | 6-8x | 10-14x |
| ReLU | 1x | 7-8x | 15-16x |

### Memory Bandwidth Considerations

```rust
/// Memory bandwidth is often the bottleneck
/// Roofline model considerations:
///
/// Arithmetic Intensity = FLOPs / Bytes loaded
///
/// Conv 3x3 (3x3x64x64 kernel):
///   FLOPs per output = 2 * 3 * 3 * 64 = 1152
///   Bytes per output = 4 (f32)
///   AI = 288 FLOPs/Byte (compute bound)
///
/// BatchNorm:
///   FLOPs per element = ~5
///   Bytes per element = 4 + 4 (read + write)
///   AI = 0.625 FLOPs/Byte (memory bound)
```

## WASM Compilation (RuVector Pattern)

Following the existing `ruvector-wasm` and `ruvector-*-wasm` crate patterns:

### Cargo.toml Configuration

```toml
[package]
name = "ruvector-cnn-wasm"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[features]
default = ["console_error_panic_hook"]
simd = []  # Enable WASM SIMD128

[dependencies]
ruvector-cnn = { path = "../ruvector-cnn", default-features = false }
wasm-bindgen = "0.2"
js-sys = "0.3"
console_error_panic_hook = { version = "0.1", optional = true }

[target.'cfg(target_arch = "wasm32")'.dependencies]
getrandom = { version = "0.2", features = ["js"] }
```

### WASM SIMD128 Implementation

```rust
#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

/// WASM SIMD128: 4x f32 per register (matches ARM NEON width)
#[cfg(target_arch = "wasm32")]
pub fn dot_product_wasm_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x4_splat(0.0);
    let chunks = a.len() / 4;

    for i in 0..chunks {
        let va = v128_load(a.as_ptr().add(i * 4) as *const v128);
        let vb = v128_load(b.as_ptr().add(i * 4) as *const v128);
        sum = f32x4_add(sum, f32x4_mul(va, vb));
    }

    // Horizontal sum
    let sum_arr: [f32; 4] = std::mem::transmute(sum);
    let mut result = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

    // Remainder
    for i in (chunks * 4)..a.len() {
        result += a[i] * b[i];
    }

    result
}
```

### wasm-bindgen Exports

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmCnnEmbedder {
    inner: MobileNetEmbedder,
}

#[wasm_bindgen]
impl WasmCnnEmbedder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmCnnEmbedder, JsValue> {
        console_error_panic_hook::set_once();
        let inner = MobileNetEmbedder::new_v3_small()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self { inner })
    }

    /// Extract embedding from image bytes (RGBA)
    #[wasm_bindgen]
    pub fn extract(&self, image_data: &[u8], width: u32, height: u32) -> Result<Vec<f32>, JsValue> {
        self.inner
            .extract(image_data, width, height)
            .map(|v| v.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Embedding dimension
    #[wasm_bindgen(getter)]
    pub fn embedding_dim(&self) -> usize {
        self.inner.embedding_dim()
    }
}
```

### Build Commands

```bash
# Standard WASM build
wasm-pack build --target web --release

# With SIMD128 enabled (requires browser support)
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --release

# For Node.js
wasm-pack build --target nodejs --release
```

## NAPI-RS Bindings (Node.js)

Following `ruvector-node`, `ruvector-attention-node`, etc.:

### Cargo.toml

```toml
[package]
name = "ruvector-cnn-node"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
ruvector-cnn = { path = "../ruvector-cnn" }
napi = { version = "2", features = ["async"] }
napi-derive = "2"

[build-dependencies]
napi-build = "2"
```

### NAPI Implementation

```rust
#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use ruvector_cnn::{EmbeddingExtractor, MobileNetEmbedder, EmbeddingConfig};

#[napi]
pub struct CnnEmbedder {
    inner: MobileNetEmbedder,
}

#[napi]
impl CnnEmbedder {
    #[napi(constructor)]
    pub fn new(config: Option<CnnConfig>) -> Result<Self> {
        let cfg = config.map(|c| c.into()).unwrap_or_default();
        let inner = MobileNetEmbedder::new(cfg)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Extract embedding from image buffer
    #[napi]
    pub fn extract(&self, image_data: Buffer, width: u32, height: u32) -> Result<Vec<f64>> {
        let embedding = self.inner
            .extract(image_data.as_ref(), width, height)
            .map_err(|e| Error::from_reason(e.to_string()))?;

        // Convert f32 to f64 for JS compatibility
        Ok(embedding.iter().map(|&x| x as f64).collect())
    }

    /// Batch extract embeddings (async for large batches)
    #[napi]
    pub async fn extract_batch(&self, images: Vec<ImageInput>) -> Result<Vec<Vec<f64>>> {
        let results: Vec<_> = images
            .into_iter()
            .map(|img| {
                self.inner
                    .extract(&img.data, img.width, img.height)
                    .map(|e| e.iter().map(|&x| x as f64).collect())
                    .map_err(|e| Error::from_reason(e.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(results)
    }

    #[napi(getter)]
    pub fn embedding_dim(&self) -> u32 {
        self.inner.embedding_dim() as u32
    }
}

#[napi(object)]
pub struct CnnConfig {
    pub input_size: Option<u32>,
    pub embedding_dim: Option<u32>,
    pub normalize: Option<bool>,
    pub quantized: Option<bool>,
}

#[napi(object)]
pub struct ImageInput {
    pub data: Buffer,
    pub width: u32,
    pub height: u32,
}

impl From<CnnConfig> for EmbeddingConfig {
    fn from(cfg: CnnConfig) -> Self {
        EmbeddingConfig {
            input_size: cfg.input_size.unwrap_or(224),
            embedding_dim: cfg.embedding_dim.unwrap_or(512) as usize,
            normalize: cfg.normalize.unwrap_or(true),
            quantized: cfg.quantized.unwrap_or(false),
        }
    }
}
```

### TypeScript Definitions (auto-generated)

```typescript
// index.d.ts (generated by napi-rs)
export class CnnEmbedder {
  constructor(config?: CnnConfig);
  extract(imageData: Buffer, width: number, height: number): number[];
  extractBatch(images: ImageInput[]): Promise<number[][]>;
  get embeddingDim(): number;
}

export interface CnnConfig {
  inputSize?: number;
  embeddingDim?: number;
  normalize?: boolean;
  quantized?: boolean;
}

export interface ImageInput {
  data: Buffer;
  width: number;
  height: number;
}
```

### Usage in Node.js

```javascript
const { CnnEmbedder } = require('ruvector-cnn-node');
const sharp = require('sharp');

async function embedImage(imagePath) {
    const embedder = new CnnEmbedder({ embeddingDim: 512, normalize: true });

    // Load and preprocess image
    const { data, info } = await sharp(imagePath)
        .resize(224, 224)
        .raw()
        .toBuffer({ resolveWithObject: true });

    // Extract embedding
    const embedding = embedder.extract(data, info.width, info.height);
    console.log(`Embedding dimension: ${embedder.embeddingDim}`);
    console.log(`First 5 values: ${embedding.slice(0, 5)}`);

    return embedding;
}
```

## Platform-Specific Compilation

### Unified Dispatch (per ADR-003)

```rust
/// Unified dispatch following ruvector-core SIMD pattern
pub mod dispatch {
    use super::*;

    pub fn conv_3x3_depthwise(input: &[f32], kernel: &[f32], output: &mut [f32], params: &ConvParams) {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { conv_3x3_depthwise_neon(input, kernel, output, params) }
        }

        #[cfg(all(target_arch = "x86_64"))]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                unsafe { conv_3x3_depthwise_avx2(input, kernel, output, params) }
            } else {
                conv_3x3_depthwise_scalar(input, kernel, output, params)
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            conv_3x3_depthwise_wasm_simd(input, kernel, output, params)
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64", target_arch = "wasm32")))]
        {
            conv_3x3_depthwise_scalar(input, kernel, output, params)
        }
    }
}
```

### Feature Flags

```toml
[features]
default = ["std"]
std = []

# SIMD backends (auto-detected at runtime on x86_64)
avx2 = []
avx512 = []
neon = []  # Always enabled on aarch64

# WASM SIMD (requires explicit opt-in)
wasm-simd = []

# Quantization
int8 = []
int4 = []

# ONNX model loading
onnx = ["tract-onnx"]

# Bindings
wasm = ["wasm-bindgen", "js-sys"]
napi = ["napi", "napi-derive"]
```

## References

1. [Are We Learning Yet?](https://www.arewelearningyet.com/neural-networks/)
2. [rust_cnn - CNN from scratch](https://github.com/goldstraw/rust_cnn)
3. [cnn-rs - CNN implementation](https://github.com/mikecvet/cnn-rs)
4. [Neuronika - Rust autograd](https://users.rust-lang.org/t/announcing-neuronika-a-deep-learning-framework-in-rust/61137)
5. [Rust SIMD RFC](https://rust-lang.github.io/rfcs/2325-stable-simd.html)
6. [std::arch documentation](https://doc.rust-lang.org/std/arch/index.html)
7. [SIMDeez library](https://docs.rs/simdeez/)
8. [Convolutions in Rust for Deep Learning](https://athemathmo.github.io/2016/04/29/convolutions-deep-learning.html)
9. [Rust SIMD intrinsics guide](https://www.slingacademy.com/article/using-simd-intrinsics-for-high-performance-math-in-rust/)
10. [Awesome-Rust-MachineLearning](https://github.com/vaaaaanquish/Awesome-Rust-MachineLearning)
11. [wasm-bindgen Guide](https://rustwasm.github.io/wasm-bindgen/)
12. [NAPI-RS Documentation](https://napi.rs/)
13. [RuVector ADR-003: SIMD Optimization Strategy](../../adr/ADR-003-simd-optimization-strategy.md)
14. [RuVector ADR-005: WASM Runtime Integration](../../adr/ADR-005-wasm-runtime-integration.md)
