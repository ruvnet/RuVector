# SIMD/CPU Optimization Techniques for CNNs

## Executive Summary

This document covers CPU optimization techniques for CNN inference, focusing on SIMD vectorization (AVX2, AVX-512, NEON), efficient convolution algorithms (Winograd, FFT), quantization strategies, and structured pruning. These techniques are essential for achieving high-performance CNN inference without GPU acceleration.

## SIMD Instruction Sets Overview

### AVX2 (256-bit)

**Availability**: Intel Haswell (2013+), AMD Zen+

**Register Width**: 256 bits = 8x float32 or 32x int8

**Key Instructions**:
- `_mm256_loadu_ps`: Load 8 floats
- `_mm256_mul_ps`: Multiply 8 floats
- `_mm256_fmadd_ps`: Fused multiply-add (8 floats)
- `_mm256_storeu_ps`: Store 8 floats

**Performance**: ~10x faster than scalar for vectorizable operations

### AVX-512 (512-bit)

**Availability**: Intel Skylake-X (2017+), Ice Lake, AMD Zen 4+

**Register Width**: 512 bits = 16x float32 or 64x int8

**Key Extensions**:
- **AVX-512F**: Foundation (basic operations)
- **AVX-512VNNI**: Vector Neural Network Instructions (INT8 dot products)
- **AVX-512BF16**: BFloat16 support

**Performance vs AVX2**:
- 15% faster indexing
- 13% faster vector search
- 2x theoretical throughput

### ARM NEON (128-bit)

**Availability**: ARM Cortex-A, Apple Silicon

**Register Width**: 128 bits = 4x float32 or 16x int8

**Key Instructions**:
- `vld1q_f32`: Load 4 floats
- `vmulq_f32`: Multiply 4 floats
- `vmlaq_f32`: Multiply-accumulate

**Apple Silicon (AMX)**: Additional matrix coprocessor for ML

## Convolution Optimization Techniques

### 1. Winograd Convolution

**Principle**: Transform-based algorithm minimizing multiplications

**Theory**: For input size n and kernel size k:
- Direct: n × k multiplications
- Winograd: n + k - 1 multiplications (theoretical minimum)

**F(2×2, 3×3) Algorithm**:
```
Output tile: 2×2
Kernel: 3×3
Transformed operations: 4×4 element-wise multiply

Speedup: ~2.25x over direct for 3×3 kernels
```

**Implementation Steps**:
1. Transform input tile: `d = B^T × input × B`
2. Transform kernel (once): `g = G × kernel × G^T`
3. Element-wise multiply: `m = d ⊙ g`
4. Transform output: `output = A^T × m × A`

**Performance Results** (vs cuDNN FFT):
- VGG network: 1.48x faster at batch=64, 2.26x faster at batch=1
- Throughput: 9.49 TFLOPS at batch=16

**Trade-offs**:
- Best for: 3×3 and 5×5 kernels (CNN standard)
- Additional memory for transformation matrices
- Numerical precision concerns for larger tiles

**Rust Pseudocode**:
```rust
fn winograd_conv_3x3(input: &[f32], kernel: &[f32], output: &mut [f32]) {
    // Pre-computed transformation matrices
    const BT: [[f32; 4]; 4] = /* B transpose */;
    const G: [[f32; 3]; 4] = /* G matrix */;
    const AT: [[f32; 4]; 2] = /* A transpose */;

    // Transform kernel (done once)
    let g_kernel = transform_kernel(kernel, &G);

    // For each 4x4 input tile producing 2x2 output:
    for tile in input.tiles(4, 4) {
        let d = bt_transform(tile, &BT);
        let m = elementwise_mul(&d, &g_kernel);
        let out = at_transform(&m, &AT);
        // Store 2x2 output
    }
}
```

### 2. FFT Convolution

**Principle**: Convolution becomes element-wise multiply in frequency domain

**Algorithm**:
1. FFT(input) and FFT(kernel)
2. Element-wise complex multiply
3. IFFT(result)

**Best for**: Large kernels (7×7+)

**Limitation**: Overhead for small 3×3 kernels common in modern CNNs

### 3. im2col + GEMM

**Principle**: Convert convolution to matrix multiplication

**Algorithm**:
1. Unfold input patches into columns (im2col)
2. Reshape kernel to matrix
3. Matrix multiply (BLAS GEMM)

**Advantages**:
- Leverages highly optimized BLAS libraries
- Consistent performance across kernel sizes
- Easy to parallelize

**Disadvantage**: Memory overhead for unfolded matrix

### 4. Direct Convolution with SIMD

**Register Blocking Strategy**:
```rust
// AVX-512: 16 output channels in ZMM registers
fn conv_direct_avx512(
    input: &[f32],      // [H, W, C_in]
    kernel: &[f32],     // [C_out, K, K, C_in]
    output: &mut [f32], // [H', W', C_out]
) {
    // Load 16 output channel accumulators
    let mut acc = [_mm512_setzero_ps(); 16];

    // For each spatial position:
    for ky in 0..K {
        for kx in 0..K {
            for c_in in 0..C_in {
                // Broadcast input value
                let inp = _mm512_set1_ps(input[...]);
                // Load 16 kernel weights
                let w = _mm512_load_ps(&kernel[...]);
                // FMA: acc += inp * w
                acc[0] = _mm512_fmadd_ps(inp, w, acc[0]);
            }
        }
    }
    // Store 16 outputs
}
```

## INT8 Quantization

### Post-Training Quantization (PTQ)

**Process**:
1. Collect activation statistics on calibration set
2. Compute scale and zero-point for each layer
3. Quantize weights: `q = round(w / scale) + zero_point`
4. Run inference with INT8 operations

**Model Size Reduction**: 4x (FP32 -> INT8)

**Accuracy Loss**: Typically <1% with proper calibration

### Quantization-Aware Training (QAT)

**Process**:
1. Insert fake quantization nodes during training
2. Forward: quantize -> dequantize (simulate INT8)
3. Backward: straight-through estimator (STE)
4. Export quantized model

**Benefits**:
- <1% accuracy gap from FP32
- Model learns to be robust to quantization

**Trade-off**: Requires retraining (hundreds of epochs)

### INT8 Convolution with VNNI

**AVX-512 VNNI Instructions**:
```rust
// Dot product of 4 INT8 values with accumulation
// _mm512_dpbusd_epi32: u8 × i8 -> i32 accumulate
fn conv_int8_vnni(
    input: &[u8],       // Unsigned activations
    kernel: &[i8],      // Signed weights
    output: &mut [i32], // 32-bit accumulator
) {
    // Process 64 int8 values per instruction
    let inp = _mm512_load_si512(input.as_ptr());
    let w = _mm512_load_si512(kernel.as_ptr());
    acc = _mm512_dpbusd_epi32(acc, inp, w);
}
```

**Performance**: 2x faster than FP32 on VNNI-capable CPUs

### INT4 Quantization

**Challenges**:
- Very limited range [-8, 7]
- Problematic for vision models
- Often requires replacing ReLU with bounded activations

**Best Practices**:
- Use for weight-only quantization
- Keep activations at INT8 or higher
- Apply after extensive QAT

## Structured Pruning

### Channel Pruning

**Strategy**: Remove entire output channels

**Benefits**:
- Direct reduction in FLOPS and parameters
- No specialized sparse kernels needed
- Works with standard convolution libraries

**Implementation**:
```rust
struct PrunedConv {
    // Original: [C_out, K, K, C_in]
    // Pruned: [C_out - n_pruned, K, K, C_in - m_pruned]
    weights: Array4<f32>,
    channel_mask: Vec<bool>,
}
```

### Filter Pruning Criteria

| Criterion | Description | Effectiveness |
|-----------|-------------|---------------|
| L1 Norm | Sum of absolute weights | Baseline |
| L2 Norm | Euclidean norm | Standard |
| BN Scale | Batch norm gamma values | Very effective |
| Taylor | First-order importance | Best accuracy |
| Geometric Median | Redundancy-based | Good compression |

### Dynamic Structured Pruning (DSP)

**Recent Approach** (2024):
1. Add instance-wise sparsity loss during training
2. Analyze global activations to identify redundant filters
3. Prune without predefined ratio

**Results**:
- 58% compression for LITETime
- 75% compression for InceptionTime
- Maintains classification accuracy

### 2:4 Sparsity Pattern

**NVIDIA Ampere Feature**: Hardware-accelerated 2:4 sparsity

**Pattern**: 2 non-zeros per 4 consecutive elements

**Benefits**:
- 2x inference speedup
- Minimal accuracy loss
- Direct hardware support

## Memory-Efficient Training

### Gradient Checkpointing

**Principle**: Trade compute for memory

**Standard**: Store all activations for backward pass

**Checkpointed**: Store only checkpoint activations, recompute others

**Memory Reduction**: O(n) -> O(sqrt(n)) for n layers

**Overhead**: ~20-30% longer training time

**Implementation**:
```rust
fn forward_checkpointed(layers: &[Layer], input: Tensor) -> Tensor {
    let checkpoint_interval = (layers.len() as f32).sqrt() as usize;
    let mut checkpoints = vec![];

    let mut x = input;
    for (i, layer) in layers.iter().enumerate() {
        if i % checkpoint_interval == 0 {
            checkpoints.push(x.clone());
        }
        x = layer.forward(x);
    }
    x
}

fn backward_checkpointed(/* ... */) {
    // Recompute activations between checkpoints
}
```

### Mixed Precision Training

**BFloat16 Format**:
- Same dynamic range as FP32 (8-bit exponent)
- Reduced precision (7-bit mantissa vs 23-bit)
- 2x memory savings

**Strategy**:
- Forward/backward in BF16
- Master weights in FP32
- Loss scaling for gradient stability

### CPU Offloading

**ZeRO-Offload**: Offload optimizer states and gradients to CPU

**Benefits**: 10x larger models on single GPU

**Consideration**: PCIe bandwidth becomes bottleneck

## Practical SIMD Patterns for CNN Operations

### Vectorized ReLU (AVX2)

```rust
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2(data: &mut [f32]) {
    let zero = _mm256_setzero_ps();
    for chunk in data.chunks_exact_mut(8) {
        let x = _mm256_loadu_ps(chunk.as_ptr());
        let result = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(chunk.as_mut_ptr(), result);
    }
}
```

### Vectorized Batch Normalization

```rust
#[target_feature(enable = "avx2")]
unsafe fn batch_norm_avx2(
    data: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    epsilon: f32,
) {
    let eps = _mm256_set1_ps(epsilon);

    for c in 0..channels {
        let g = _mm256_set1_ps(gamma[c]);
        let b = _mm256_set1_ps(beta[c]);
        let m = _mm256_set1_ps(mean[c]);
        let v = _mm256_set1_ps(var[c]);

        // inv_std = 1 / sqrt(var + eps)
        let inv_std = _mm256_rsqrt_ps(_mm256_add_ps(v, eps));

        for spatial in channel_data.chunks_exact_mut(8) {
            let x = _mm256_loadu_ps(spatial.as_ptr());
            // y = gamma * (x - mean) * inv_std + beta
            let centered = _mm256_sub_ps(x, m);
            let normed = _mm256_mul_ps(centered, inv_std);
            let scaled = _mm256_fmadd_ps(normed, g, b);
            _mm256_storeu_ps(spatial.as_mut_ptr(), scaled);
        }
    }
}
```

### Vectorized Depthwise Convolution

```rust
#[target_feature(enable = "avx2")]
unsafe fn depthwise_conv_3x3_avx2(
    input: &[f32],   // [H, W, C]
    kernel: &[f32],  // [3, 3, C]
    output: &mut [f32],
    h: usize, w: usize, c: usize,
) {
    // Process 8 channels at a time
    for ch in (0..c).step_by(8) {
        for y in 1..h-1 {
            for x in 1..w-1 {
                let mut acc = _mm256_setzero_ps();

                for ky in 0..3 {
                    for kx in 0..3 {
                        let inp_idx = ((y + ky - 1) * w + (x + kx - 1)) * c + ch;
                        let ker_idx = (ky * 3 + kx) * c + ch;

                        let inp = _mm256_loadu_ps(&input[inp_idx]);
                        let ker = _mm256_loadu_ps(&kernel[ker_idx]);
                        acc = _mm256_fmadd_ps(inp, ker, acc);
                    }
                }

                let out_idx = (y * w + x) * c + ch;
                _mm256_storeu_ps(&mut output[out_idx], acc);
            }
        }
    }
}
```

## Performance Comparison

### Convolution Algorithm Selection

| Kernel Size | Best Algorithm | Speedup vs Direct |
|-------------|----------------|-------------------|
| 1×1 | Direct/GEMM | 1x (baseline) |
| 3×3 | Winograd F(2×2,3×3) | 2-2.5x |
| 5×5 | Winograd F(4×4,5×5) | 2-3x |
| 7×7+ | FFT | 2-4x |

### Quantization Impact

| Precision | Model Size | Inference Speed | Accuracy Drop |
|-----------|------------|-----------------|---------------|
| FP32 | 100% | 1x | 0% |
| FP16 | 50% | 1.5-2x | <0.1% |
| INT8 | 25% | 2-4x | <1% |
| INT4 | 12.5% | 3-6x | 1-5% |

## References

1. [Deep Learning with Intel AVX-512 and DL Boost](https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html)
2. [Optimizing CNN Model Inference on CPUs](https://arxiv.org/pdf/1809.02697)
3. [Fast Algorithms for Convolutional Neural Networks](https://arxiv.org/abs/1509.09308)
4. [PyTorch Inductor CPU Optimizations](https://pytorch.org/blog/accelerated-cpu-inference/)
5. [OpenVINO CPU Plugin](https://ovino.readthedocs.io/en/latest/IE_DG/supported_plugins/CPU/)
6. [Efficient Quantized Winograd Convolution](https://dl.acm.org/doi/10.1145/3632956)
7. [Structured Pruning Survey](https://www.researchgate.net/publication/376011921_Structured_Pruning_for_Deep_Convolutional_Neural_Networks_A_Survey)
8. [Gradient Checkpointing](https://github.com/cybertronai/gradient-checkpointing)
9. [Model Quantization Guide](https://www.edge-ai-vision.com/2024/02/quantization-of-convolutional-neural-networks-model-quantization/)
10. [LoWino: Quantized Winograd](https://dl.acm.org/doi/10.1145/3632956)
