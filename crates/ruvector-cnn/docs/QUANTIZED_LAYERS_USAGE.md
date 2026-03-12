# Quantized Layers Usage Guide

This guide demonstrates how to use the INT8 quantized layers in ruvector-cnn for efficient inference.

## Quick Start

### 1. Quantize a Conv2d Layer

```rust
use ruvector_cnn::layers::{Conv2dBuilder, QuantizedConv2d};

// Create FP32 convolution
let conv = Conv2dBuilder::new(16, 32, 3)
    .stride(1)
    .padding(1)
    .build()?;

// Convert to INT8 quantized convolution
let input_scale = 0.01f32;
let input_zero_point = 128i32;
let qconv = QuantizedConv2d::from_fp32(&conv, input_scale, input_zero_point);

// Run INT8 inference
let input_u8 = vec![128u8; 1 * 8 * 8 * 16]; // Quantized input
let input_shape = &[1, 8, 8, 16];
let output = qconv.forward_int8(&input_u8, input_shape, input_scale, 128)?;
```

### 2. Quantize a Linear Layer

```rust
use ruvector_cnn::layers::{Linear, QuantizedLinear};

// Create FP32 linear layer
let linear = Linear::new(256, 128, true)?;

// Convert to INT8
let qlinear = QuantizedLinear::from_fp32(&linear, 0.01);

// Run INT8 inference
let batch_size = 4;
let input_u8 = vec![128u8; batch_size * 256];
let output = qlinear.forward_int8(&input_u8, batch_size, 0.01, 128)?;
```

### 3. Depthwise Convolution

```rust
use ruvector_cnn::layers::QuantizedDepthwiseConv2d;

let channels = 32;
let kernel_size = 3;
let weights = vec![0.1f32; channels * kernel_size * kernel_size];
let bias = vec![0.0f32; channels];

let qdw = QuantizedDepthwiseConv2d::from_fp32(
    channels,
    kernel_size,
    &weights,
    Some(&bias),
    1, // stride
    1, // padding
    0.01, // input_scale
);

let input_u8 = vec![128u8; 1 * 8 * 8 * channels];
let input_shape = &[1, 8, 8, channels];
let output = qdw.forward_int8(&input_u8, input_shape, 0.01, 128)?;
```

### 4. Pooling Operations

```rust
use ruvector_cnn::layers::{QuantizedMaxPool2d, QuantizedAvgPool2d};

// Max pooling (no scale change)
let qmax = QuantizedMaxPool2d::new(2, 2, 0);
let (output, shape, scale, zp) = qmax.forward_int8(
    &input_u8,
    &input_shape,
    0.01, // input scale
    128,  // input zero point
)?;

// Average pooling (uses i16 intermediate precision)
let qavg = QuantizedAvgPool2d::new(2, 2, 0);
let (output, shape, scale, zp) = qavg.forward_int8(
    &input_u8,
    &input_shape,
    0.01,
    128,
)?;
```

### 5. Residual Connections

```rust
use ruvector_cnn::layers::QuantizedResidualAdd;

// Add two quantized tensors with potentially different scales
let scale1 = 0.01f32;
let scale2 = 0.015f32;
let residual = QuantizedResidualAdd::new(scale1, scale2);

let input1 = vec![128u8; 16];
let input2 = vec![138u8; 16];
let shape = &[4, 4];

let (output, out_scale, out_zp) = residual.forward_int8(
    &input1, scale1, 128,
    &input2, scale2, 128,
    shape,
)?;

// For higher precision, use fixed-point arithmetic
let (output, out_scale, out_zp) = residual.forward_int8_i16(
    &input1, scale1, 128,
    &input2, scale2, 128,
    shape,
)?;
```

## Complete Example: Quantized Network

```rust
use ruvector_cnn::layers::{
    Conv2dBuilder, Linear,
    QuantizedConv2d, QuantizedLinear, QuantizedMaxPool2d,
};

// 1. Build FP32 network
let conv1 = Conv2dBuilder::new(3, 16, 3).padding(1).build()?;
let conv2 = Conv2dBuilder::new(16, 32, 3).padding(1).build()?;
let linear = Linear::new(32 * 8 * 8, 10, true)?;

// 2. Quantize network
let input_scale = 0.01f32;
let qconv1 = QuantizedConv2d::from_fp32(&conv1, input_scale, 128);
let qconv2 = QuantizedConv2d::from_fp32(&conv2, input_scale, 128);
let qlinear = QuantizedLinear::from_fp32(&linear, input_scale);
let qpool = QuantizedMaxPool2d::new(2, 2, 0);

// 3. Run INT8 inference
let input_u8 = vec![128u8; 1 * 32 * 32 * 3]; // 1x32x32x3
let mut x = &input_u8[..];
let mut x_shape = vec![1, 32, 32, 3];

// Conv1 + Pool
let out1 = qconv1.forward_int8(x, &x_shape, input_scale, 128)?;
let (out1_u8, x_shape, scale, zp) = qpool.forward_int8(
    quantize_tensor(&out1, input_scale)?,
    &out1.shape(),
    input_scale,
    128,
)?;

// Conv2 + Pool
let out2 = qconv2.forward_int8(&out1_u8, &x_shape, input_scale, 128)?;
let (out2_u8, x_shape, scale, zp) = qpool.forward_int8(
    quantize_tensor(&out2, input_scale)?,
    &out2.shape(),
    input_scale,
    128,
)?;

// Flatten and Linear
let batch_size = 1;
let flat_size = 32 * 8 * 8;
let output = qlinear.forward_int8(&out2_u8, batch_size, input_scale, 128)?;
```

## Quantization Workflow

### Step 1: Prepare FP32 Model

Train your model in FP32 as usual.

### Step 2: Calibrate Quantization Parameters

Use a representative dataset to determine optimal scales and zero-points:

```rust
// Collect activation statistics
let mut min_val = f32::MAX;
let mut max_val = f32::MIN;

for sample in calibration_data {
    let activations = model.forward(&sample)?;
    min_val = min_val.min(activations.min());
    max_val = max_val.max(activations.max());
}

// Compute scale and zero-point
let scale = (max_val - min_val) / 255.0;
let zero_point = ((-min_val / scale).round().clamp(0.0, 255.0)) as u8;
```

### Step 3: Quantize Model

Convert all layers to INT8 using the calibrated parameters.

### Step 4: Validate Accuracy

Compare INT8 model output with FP32 baseline:

```rust
let fp32_output = fp32_model.forward(&test_input)?;
let int8_output = quantized_model.forward(&test_input)?;

let mse = compute_mse(&fp32_output, &int8_output);
println!("Quantization error (MSE): {:.6}", mse);
```

## Performance Tips

1. **Use per-channel quantization for weights**: Provides ~0.5% better accuracy
2. **Calibrate on diverse data**: Use 100-1000 representative samples
3. **Fuse operations**: Combine Conv+BatchNorm+ReLU before quantization
4. **Use asymmetric quantization for activations**: Better for ReLU outputs
5. **Symmetric quantization for weights**: Simpler and often sufficient

## Memory Requirements

INT8 quantization reduces memory by 4x compared to FP32:

| Layer Type | FP32 Memory | INT8 Memory | Reduction |
|------------|-------------|-------------|-----------|
| Conv2d (3x3, 16→32) | 18 KB | 4.5 KB | 4x |
| Linear (256→128) | 128 KB | 32 KB | 4x |
| MobileNetV3-Small | ~6 MB | ~1.5 MB | 4x |

## Expected Performance

On AVX2-capable x86_64 CPUs:

| Operation | FP32 Time | INT8 Time | Speedup |
|-----------|-----------|-----------|---------|
| Conv2d 3x3 | 1.0x | 0.33x | 3x |
| Conv2d 1x1 | 1.0x | 0.25x | 4x |
| Linear | 1.0x | 0.33x | 3x |
| **Full Model** | 1.0x | **0.25-0.5x** | **2-4x** |

## Troubleshooting

### High Quantization Error

- Increase calibration dataset size
- Use per-channel quantization for weights
- Try different calibration methods (percentile vs min-max)

### Numerical Instability

- Check for extreme outliers in activations
- Use clipping or percentile-based calibration
- Verify scale factors are not too small (<1e-10)

### Performance Not Improving

- Ensure SIMD instructions are available (`is_x86_feature_detected!("avx2")`)
- Check that batch size is large enough for SIMD efficiency
- Profile to identify bottlenecks

## API Reference

### QuantizedConv2d

- `from_fp32(conv, input_scale, input_zero_point)` - Create from FP32
- `forward_int8(input, shape, scale, zp)` - INT8 forward pass
- `in_channels()`, `out_channels()`, `kernel_size()` - Getters

### QuantizedDepthwiseConv2d

- `from_fp32(channels, kernel_size, weights, bias, stride, padding, input_scale)`
- `forward_int8(input, shape, scale, zp)`

### QuantizedLinear

- `from_fp32(linear, input_scale)`
- `forward_int8(input, batch_size, scale, zp)`
- `in_features()`, `out_features()`

### QuantizedMaxPool2d / QuantizedAvgPool2d

- `new(kernel_size, stride, padding)`
- `forward_int8(input, shape, scale, zp)` → `(output, shape, scale, zp)`

### QuantizedResidualAdd

- `new(scale1, scale2)`
- `forward_int8(input1, scale1, zp1, input2, scale2, zp2, shape)`
- `forward_int8_i16(...)` - Higher precision variant
- `output_scale()`, `output_zero_point()`

## Further Reading

- [INT8 Quantization Design Document](INT8_QUANTIZATION_DESIGN.md)
- [ADR-091 Phase 4 Implementation](ADR-091-PHASE-4-IMPLEMENTATION.md)
- [Quantization and Training of Neural Networks (Google, 2018)](https://arxiv.org/abs/1712.05877)
