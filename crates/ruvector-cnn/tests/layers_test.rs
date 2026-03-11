//! Tests for CNN layer implementations
//!
//! Tests cover:
//! - Conv2d forward pass shapes
//! - BatchNorm statistics
//! - Activation functions (ReLU, ReLU6, Swish, HardSwish)
//! - Pooling operations

use ruvector_cnn::layers::{
    Activation, ActivationType, BatchNorm, Conv2d, DepthwiseSeparableConv,
    GlobalAvgPool, HardSwish, Layer, MaxPool2d, AvgPool2d, ReLU, ReLU6,
    Sigmoid, Swish, TensorShape,
};
use ruvector_cnn::{simd, Tensor};

// ============================================================================
// Conv2d Tests
// ============================================================================

#[test]
fn test_conv2d_output_shape_no_padding() {
    // Input: [batch=1, height=8, width=8, channels=3] (NHWC format)
    // Kernel: 3x3, stride=1, no padding
    // Expected output: [1, 6, 6, 16]
    let conv = Conv2d::new(3, 16, 3, 1, 0);

    let input = Tensor::ones(&[1, 8, 8, 3]);
    let output = conv.forward(&input).unwrap();

    // Output shape calculation: (h_in - kh + 2*pad_h) / stride_h + 1
    // (8 - 3 + 0) / 1 + 1 = 6
    assert_eq!(output.shape(), &[1, 6, 6, 16]);
}

#[test]
fn test_conv2d_output_shape_with_padding() {
    // Input: [1, 8, 8, 3]
    // Kernel: 3x3, stride=1, padding=1 (same padding)
    // Expected output: [1, 8, 8, 16]
    let conv = Conv2d::new(3, 16, 3, 1, 1);

    let input = Tensor::ones(&[1, 8, 8, 3]);
    let output = conv.forward(&input).unwrap();

    // (8 + 2*1 - 3) / 1 + 1 = 8
    assert_eq!(output.shape(), &[1, 8, 8, 16]);
}

#[test]
fn test_conv2d_output_shape_with_stride() {
    // Input: [1, 8, 8, 3]
    // Kernel: 3x3, stride=2, padding=1
    // Expected output: [1, 4, 4, 16]
    let conv = Conv2d::new(3, 16, 3, 2, 1);

    let input = Tensor::ones(&[1, 8, 8, 3]);
    let output = conv.forward(&input).unwrap();

    // (8 + 2*1 - 3) / 2 + 1 = 4
    assert_eq!(output.shape(), &[1, 4, 4, 16]);
}

#[test]
fn test_conv2d_batch_processing() {
    // Verify batch dimension is handled correctly
    let conv = Conv2d::new(3, 8, 3, 1, 1);

    // Batch of 4 images
    let input = Tensor::ones(&[4, 16, 16, 3]);
    let output = conv.forward(&input).unwrap();

    assert_eq!(output.shape(), &[4, 16, 16, 8]);
}

#[test]
fn test_conv2d_1x1_pointwise() {
    // 1x1 convolution (pointwise) - commonly used in MobileNet
    let conv = Conv2d::new(64, 128, 1, 1, 0);

    let input = Tensor::ones(&[1, 7, 7, 64]);
    let output = conv.forward(&input).unwrap();

    // 1x1 conv preserves spatial dimensions
    assert_eq!(output.shape(), &[1, 7, 7, 128]);
}

#[test]
fn test_conv2d_output_shape_method() {
    let conv = Conv2d::new(3, 64, 3, 1, 1);
    let shape = conv.output_shape(&[1, 224, 224, 3]).unwrap();
    assert_eq!(shape, vec![1, 224, 224, 64]);
}

#[test]
fn test_conv2d_output_shape_stride2() {
    let conv = Conv2d::new(3, 64, 3, 2, 1);
    let shape = conv.output_shape(&[1, 224, 224, 3]).unwrap();
    assert_eq!(shape, vec![1, 112, 112, 64]);
}

#[test]
fn test_depthwise_separable_conv_shape() {
    // MobileNet-style depthwise separable convolution
    let dw_conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1);

    let input = Tensor::ones(&[1, 14, 14, 32]);
    let output = dw_conv.forward(&input).unwrap();

    // Depthwise + pointwise should produce [1, 14, 14, 64]
    assert_eq!(output.shape(), &[1, 14, 14, 64]);
}

#[test]
fn test_depthwise_separable_conv_params() {
    let conv = DepthwiseSeparableConv::new(16, 32, 3, 1, 1);

    // depthwise: 16 * 3 * 3 = 144
    // pointwise: 32 * 16 = 512
    // total: 656
    assert_eq!(conv.num_params(), 144 + 512);
}

// ============================================================================
// BatchNorm Tests
// ============================================================================

#[test]
fn test_batchnorm_output_shape() {
    let bn = BatchNorm::new(64);

    // Input: [batch=2, height=8, width=8, channels=64]
    let input = Tensor::ones(&[2, 8, 8, 64]);
    let output = bn.forward(&input).unwrap();

    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_batchnorm_creation() {
    let bn = BatchNorm::new(64);
    assert_eq!(bn.num_features(), 64);
    assert_eq!(bn.gamma().len(), 64);
    assert_eq!(bn.beta().len(), 64);
    assert_eq!(bn.num_params(), 128);
}

#[test]
fn test_batchnorm_default_params() {
    let bn = BatchNorm::new(4);

    // Default: gamma=1, beta=0
    for i in 0..4 {
        assert!((bn.gamma()[i] - 1.0).abs() < 1e-6);
        assert!((bn.beta()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_batchnorm_with_running_stats() {
    let mut bn = BatchNorm::new(2);

    // Set mean=[1, 2], var=[1, 4]
    bn.set_running_stats(vec![1.0, 2.0], vec![1.0, 4.0]).unwrap();

    let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 1, 2]).unwrap();
    let output = bn.forward(&input).unwrap();

    // For channel 0: (x - 1) / sqrt(1 + eps) approx (x - 1)
    // For channel 1: (x - 2) / sqrt(4 + eps) approx (x - 2) / 2
    assert!(output.data()[0].abs() < 0.01); // (1-1)/1 = 0
    assert!(output.data()[1].abs() < 0.01); // (2-2)/2 = 0
    assert!((output.data()[2] - 2.0).abs() < 0.01); // (3-1)/1 = 2
    assert!((output.data()[3] - 1.0).abs() < 0.01); // (4-2)/2 = 1
}

#[test]
fn test_batchnorm_numerical_stability() {
    let mut bn = BatchNorm::new(4);

    // Set very small variance to test numerical stability
    bn.set_running_stats(vec![0.0; 4], vec![1e-10; 4]).unwrap();

    let input = Tensor::ones(&[1, 2, 2, 4]);
    let output = bn.forward(&input).unwrap();

    // Should not produce NaN or Inf
    for &val in output.data() {
        assert!(val.is_finite(), "Output should be finite, got {}", val);
    }
}

#[test]
fn test_batchnorm_invalid_channels() {
    let bn = BatchNorm::new(4);
    let input = Tensor::ones(&[1, 8, 8, 8]); // Wrong number of channels

    let result = bn.forward(&input);
    assert!(result.is_err());
}

// ============================================================================
// Activation Function Tests
// ============================================================================

#[test]
fn test_relu_positive_unchanged() {
    let relu = ReLU::new();
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
    let output = relu.forward(&input).unwrap();

    assert_eq!(output.data(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_relu_negative_zeroed() {
    let relu = ReLU::new();
    let input = Tensor::from_data(vec![-1.0, -2.0, -3.0, -4.0], &[4]).unwrap();
    let output = relu.forward(&input).unwrap();

    assert_eq!(output.data(), &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_relu_mixed() {
    let relu = ReLU::new();
    let input = Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
    let output = relu.forward(&input).unwrap();

    assert_eq!(output.data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_relu6_clamps_at_6() {
    let relu6 = ReLU6::new();
    let input = Tensor::from_data(vec![-1.0, 0.0, 3.0, 6.0, 10.0], &[5]).unwrap();
    let output = relu6.forward(&input).unwrap();

    assert_eq!(output.data(), &[0.0, 0.0, 3.0, 6.0, 6.0]);
}

#[test]
fn test_swish_properties() {
    // Swish: x * sigmoid(x)
    let swish = Swish::new();
    let input = Tensor::from_data(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    let output = swish.forward(&input).unwrap();

    // swish(0) = 0 * 0.5 = 0
    assert!(output.data()[0].abs() < 0.001);
    // swish(1) = 1 * sigmoid(1) approx 0.731
    assert!((output.data()[1] - 0.731).abs() < 0.01);
}

#[test]
fn test_hard_swish() {
    // HardSwish: x * relu6(x + 3) / 6
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

#[test]
fn test_sigmoid_at_zero() {
    let sigmoid = Sigmoid::new();
    let input = Tensor::from_data(vec![0.0], &[1]).unwrap();
    let output = sigmoid.forward(&input).unwrap();

    // sigmoid(0) = 0.5
    assert!((output.data()[0] - 0.5).abs() < 0.001);
}

#[test]
fn test_activation_generic() {
    let activation = Activation::new(ActivationType::ReLU);
    let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    activation.apply_inplace(&mut data);

    assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_activation_identity() {
    let activation = Activation::new(ActivationType::Identity);
    let mut data = vec![-2.0, 0.0, 2.0];
    activation.apply_inplace(&mut data);

    assert_eq!(data, vec![-2.0, 0.0, 2.0]);
}

// ============================================================================
// Pooling Tests
// ============================================================================

#[test]
fn test_global_avg_pool_output_shape() {
    // Input: [batch=1, height=7, width=7, channels=512]
    // Output should be: [batch=1, 1, 1, channels=512]
    let pool = GlobalAvgPool::new();
    let input = Tensor::ones(&[1, 7, 7, 512]);
    let output = pool.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 1, 512]);
}

#[test]
fn test_global_avg_pool_computes_average() {
    let pool = GlobalAvgPool::new();

    // Create input where channel 0 = 1, channel 1 = 2
    let mut data = vec![0.0; 2 * 2 * 2];
    for i in 0..4 {
        data[i * 2] = 1.0;     // channel 0
        data[i * 2 + 1] = 2.0; // channel 1
    }
    let input = Tensor::from_data(data, &[1, 2, 2, 2]).unwrap();

    let output = pool.forward(&input).unwrap();

    assert!((output.data()[0] - 1.0).abs() < 0.001);
    assert!((output.data()[1] - 2.0).abs() < 0.001);
}

#[test]
fn test_global_avg_pool_batch() {
    let pool = GlobalAvgPool::new();
    let input = Tensor::ones(&[3, 4, 4, 8]);
    let output = pool.forward(&input).unwrap();

    assert_eq!(output.shape(), &[3, 1, 1, 8]);

    // All ones averaged = 1
    for &val in output.data() {
        assert!((val - 1.0).abs() < 0.001);
    }
}

#[test]
fn test_max_pool_2d_output_shape() {
    let pool = MaxPool2d::new(2, 2, 0);

    // Input: [1, 8, 8, 32]
    let input = Tensor::ones(&[1, 8, 8, 32]);
    let output = pool.forward(&input).unwrap();

    // Output should be [1, 4, 4, 32]
    assert_eq!(output.shape(), &[1, 4, 4, 32]);
}

#[test]
fn test_max_pool_2d_finds_maximum() {
    let pool = MaxPool2d::new(2, 2, 0);

    // 2x2 input, 1 channel: [[1, 2], [3, 4]]
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Tensor::from_data(data, &[1, 2, 2, 1]).unwrap();

    let output = pool.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 1, 1]);
    assert_eq!(output.data()[0], 4.0);
}

#[test]
fn test_max_pool_2d_output_shape_method() {
    let pool = MaxPool2d::new(2, 2, 0);
    let shape = pool.output_shape(&[1, 224, 224, 64]).unwrap();
    assert_eq!(shape, vec![1, 112, 112, 64]);
}

#[test]
fn test_avg_pool_2d_output_shape() {
    let pool = AvgPool2d::new(2, 2, 0);
    let input = Tensor::ones(&[1, 8, 8, 4]);
    let output = pool.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 4, 4, 4]);
}

#[test]
fn test_avg_pool_2d_computes_average() {
    let pool = AvgPool2d::new(2, 2, 0);

    // 2x2 input, 1 channel: [[1, 2], [3, 4]]
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let input = Tensor::from_data(data, &[1, 2, 2, 1]).unwrap();

    let output = pool.forward(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1, 1, 1]);
    assert!((output.data()[0] - 2.5).abs() < 0.001); // (1+2+3+4)/4 = 2.5
}

#[test]
fn test_max_pool_with_stride1() {
    let pool = MaxPool2d::new(2, 1, 0);
    let shape = pool.output_shape(&[1, 4, 4, 1]).unwrap();
    assert_eq!(shape, vec![1, 3, 3, 1]);
}

// ============================================================================
// TensorShape Tests
// ============================================================================

#[test]
fn test_tensor_shape() {
    let shape = TensorShape::new(2, 64, 7, 7);
    assert_eq!(shape.n, 2);
    assert_eq!(shape.c, 64);
    assert_eq!(shape.h, 7);
    assert_eq!(shape.w, 7);
    assert_eq!(shape.numel(), 2 * 64 * 7 * 7);
}

// ============================================================================
// SIMD Functions Tests
// ============================================================================

#[test]
fn test_simd_relu() {
    let input = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
    let mut output = vec![0.0; 8];

    simd::relu_simd(&input, &mut output);

    assert_eq!(output, vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0]);
}

#[test]
fn test_simd_relu6() {
    let input = vec![-1.0, 2.0, 7.0, 4.0, -5.0, 10.0, 3.0, 8.0];
    let mut output = vec![0.0; 8];

    simd::relu6_simd(&input, &mut output);

    assert_eq!(output, vec![0.0, 2.0, 6.0, 4.0, 0.0, 6.0, 3.0, 6.0]);
}

#[test]
fn test_simd_dot_product() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    let result = simd::dot_product_simd(&a, &b);
    let expected = simd::scalar::dot_product_scalar(&a, &b);

    assert!((result - expected).abs() < 0.001);
}

// ============================================================================
// Integration Tests for Layer Combinations
// ============================================================================

#[test]
fn test_conv_bn_relu_pipeline() {
    // Common pattern in CNNs: Conv -> BN -> ReLU
    let conv = Conv2d::new(3, 32, 3, 1, 1);
    let bn = BatchNorm::new(32);
    let relu = ReLU::new();

    let input = Tensor::full(&[1, 32, 32, 3], 0.5);

    // Conv
    let conv_out = conv.forward(&input).unwrap();
    assert_eq!(conv_out.shape(), &[1, 32, 32, 32]);

    // BN
    let bn_out = bn.forward(&conv_out).unwrap();
    assert_eq!(bn_out.shape(), conv_out.shape());

    // ReLU
    let relu_out = relu.forward(&bn_out).unwrap();
    assert_eq!(relu_out.shape(), bn_out.shape());

    // All values should be >= 0 after ReLU
    for &val in relu_out.data() {
        assert!(val >= 0.0);
    }
}
