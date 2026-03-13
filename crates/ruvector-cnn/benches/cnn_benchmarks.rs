//! CNN Layer Benchmarks
//!
//! Comprehensive benchmarks for ruvector-cnn components:
//! - SIMD Operations (dot product, ReLU, ReLU6, BatchNorm)
//! - Convolutional layers (Conv2d, DepthwiseSeparable)
//! - Activations (ReLU, ReLU6, Swish, HardSwish)
//! - Pooling (GlobalAvgPool, MaxPool2d)
//! - Full MobileNet-style blocks
//!
//! Run with: cargo bench --package ruvector-cnn
//! View HTML report: open target/criterion/report/index.html

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvector_cnn::{
    layers::{
        BatchNorm, Conv2d, DepthwiseSeparableConv, GlobalAvgPool, HardSwish, Layer, MaxPool2d,
        ReLU, ReLU6, Swish,
    },
    simd, Tensor,
};

// ============================================================================
// SIMD Operations Benchmarks
// ============================================================================

fn bench_simd_relu(c: &mut Criterion) {
    let sizes = [1024, 4096, 16384, 65536, 262144];

    let mut group = c.benchmark_group("simd/relu");
    group.sample_size(100);

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        let input = vec![1.0f32; size];
        let mut output = vec![0.0f32; size];

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |b, _| {
            b.iter(|| {
                simd::relu_simd(black_box(&input), black_box(&mut output));
            })
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                simd::scalar::relu_scalar(black_box(&input), black_box(&mut output));
            })
        });
    }

    group.finish();
}

fn bench_simd_relu6(c: &mut Criterion) {
    let sizes = [1024, 4096, 16384, 65536];

    let mut group = c.benchmark_group("simd/relu6");
    group.sample_size(100);

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        let input = vec![3.0f32; size]; // Mixed values in [0, 6] range
        let mut output = vec![0.0f32; size];

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |b, _| {
            b.iter(|| {
                simd::relu6_simd(black_box(&input), black_box(&mut output));
            })
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                simd::scalar::relu6_scalar(black_box(&input), black_box(&mut output));
            })
        });
    }

    group.finish();
}

fn bench_simd_dot_product(c: &mut Criterion) {
    let sizes = [64, 256, 1024, 4096, 16384];

    let mut group = c.benchmark_group("simd/dot_product");
    group.sample_size(100);

    for &size in &sizes {
        group.throughput(Throughput::Elements(size as u64));

        let a = vec![1.0f32; size];
        let b = vec![2.0f32; size];

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |b_iter, _| {
            b_iter.iter(|| black_box(simd::dot_product_simd(black_box(&a), black_box(&b))))
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b_iter, _| {
            b_iter.iter(|| {
                black_box(simd::scalar::dot_product_scalar(
                    black_box(&a),
                    black_box(&b),
                ))
            })
        });
    }

    group.finish();
}

fn bench_simd_batch_norm(c: &mut Criterion) {
    // (height, width, channels)
    let configs = [(8, 8, 64), (28, 28, 128), (56, 56, 64), (112, 112, 32)];

    let mut group = c.benchmark_group("simd/batch_norm");
    group.sample_size(50);

    for (h, w, ch) in configs {
        let size = h * w * ch;
        group.throughput(Throughput::Elements(size as u64));

        let input = vec![1.0f32; size];
        let mut output = vec![0.0f32; size];
        let gamma = vec![1.0f32; ch];
        let beta = vec![0.0f32; ch];
        let mean = vec![0.0f32; ch];
        let var = vec![1.0f32; ch];
        let epsilon = 1e-5f32;

        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}x{}", h, w, ch)),
            &(h, w, ch),
            |b, _| {
                b.iter(|| {
                    simd::batch_norm_simd(
                        black_box(&input),
                        black_box(&mut output),
                        &gamma,
                        &beta,
                        &mean,
                        &var,
                        epsilon,
                        ch,
                    );
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}x{}", h, w, ch)),
            &(h, w, ch),
            |b, _| {
                b.iter(|| {
                    simd::scalar::batch_norm_scalar(
                        black_box(&input),
                        black_box(&mut output),
                        &gamma,
                        &beta,
                        &mean,
                        &var,
                        epsilon,
                        ch,
                    );
                })
            },
        );
    }

    group.finish();
}

fn bench_simd_conv_3x3(c: &mut Criterion) {
    // (height, width, in_channels, out_channels)
    let configs = [
        (8, 8, 3, 16),
        (32, 32, 3, 32),
        (56, 56, 16, 64),
        (28, 28, 64, 128),
    ];

    let mut group = c.benchmark_group("simd/conv_3x3");
    group.sample_size(30);

    for (h, w, in_c, out_c) in configs {
        let input_size = h * w * in_c;
        let kernel_size = out_c * 9 * in_c; // [out_c, 3, 3, in_c]
        let out_h = h; // padding=1, stride=1
        let out_w = w;
        let output_size = out_h * out_w * out_c;

        group.throughput(Throughput::Elements(output_size as u64));

        let input = vec![1.0f32; input_size];
        let kernel = vec![0.01f32; kernel_size];
        let mut output = vec![0.0f32; output_size];

        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}x{}->{}", h, w, in_c, out_c)),
            &(h, w, in_c, out_c),
            |b, _| {
                b.iter(|| {
                    simd::conv_3x3_simd(
                        black_box(&input),
                        black_box(&kernel),
                        black_box(&mut output),
                        h,
                        w,
                        in_c,
                        out_c,
                        1, // stride
                        1, // padding
                    );
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}x{}->{}", h, w, in_c, out_c)),
            &(h, w, in_c, out_c),
            |b, _| {
                b.iter(|| {
                    simd::scalar::conv_3x3_scalar(
                        black_box(&input),
                        black_box(&kernel),
                        black_box(&mut output),
                        h,
                        w,
                        in_c,
                        out_c,
                        1,
                        1,
                    );
                })
            },
        );
    }

    group.finish();
}

fn bench_simd_global_avg_pool(c: &mut Criterion) {
    let configs = [
        (7, 7, 576), // MobileNetV3-Small final
        (7, 7, 960), // MobileNetV3-Large final
        (14, 14, 256),
        (28, 28, 128),
    ];

    let mut group = c.benchmark_group("simd/global_avg_pool");
    group.sample_size(100);

    for (h, w, ch) in configs {
        let size = h * w * ch;
        group.throughput(Throughput::Elements(size as u64));

        let input = vec![1.0f32; size];
        let mut output = vec![0.0f32; ch];

        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}x{}", h, w, ch)),
            &(h, w, ch),
            |b, _| {
                b.iter(|| {
                    simd::global_avg_pool_simd(black_box(&input), black_box(&mut output), h, w, ch);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}x{}", h, w, ch)),
            &(h, w, ch),
            |b, _| {
                b.iter(|| {
                    simd::scalar::global_avg_pool_scalar(
                        black_box(&input),
                        black_box(&mut output),
                        h,
                        w,
                        ch,
                    );
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Layer Benchmarks (using Layer trait with Tensor)
// ============================================================================

fn bench_conv2d_layer(c: &mut Criterion) {
    // (batch, height, width, in_channels, out_channels)
    let configs = [
        (1, 8, 8, 3, 16),     // Small
        (1, 32, 32, 3, 32),   // Medium
        (1, 56, 56, 16, 64),  // Large (MobileNet-style)
        (1, 112, 112, 3, 32), // Initial conv
    ];

    let mut group = c.benchmark_group("layers/conv2d_3x3");
    group.sample_size(30);

    for (n, h, w, in_c, out_c) in configs {
        let elements = n * h * w * out_c;
        group.throughput(Throughput::Elements(elements as u64));

        let conv = Conv2d::new(in_c, out_c, 3, 1, 1);
        let input = Tensor::ones(&[n, h, w, in_c]);

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}x{}->{}", h, w, in_c, out_c)),
            &(n, h, w, in_c, out_c),
            |b, _| b.iter(|| black_box(conv.forward(black_box(&input)).unwrap())),
        );
    }

    group.finish();
}

fn bench_depthwise_separable(c: &mut Criterion) {
    let configs = [
        (1, 32, 32, 16, 32),
        (1, 56, 56, 32, 64),
        (1, 28, 28, 64, 128),
        (1, 14, 14, 128, 256),
    ];

    let mut group = c.benchmark_group("layers/depthwise_separable");
    group.sample_size(30);

    for (n, h, w, in_c, out_c) in configs {
        let elements = n * h * w * out_c;
        group.throughput(Throughput::Elements(elements as u64));

        let conv = DepthwiseSeparableConv::new(in_c, out_c, 3, 1, 1);
        let input = Tensor::ones(&[n, h, w, in_c]);

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}x{}->{}", h, w, in_c, out_c)),
            &(n, h, w, in_c, out_c),
            |b, _| b.iter(|| black_box(conv.forward(black_box(&input)).unwrap())),
        );
    }

    group.finish();
}

fn bench_batch_norm_layer(c: &mut Criterion) {
    let configs = [
        (1, 8, 8, 64),
        (1, 28, 28, 128),
        (1, 56, 56, 64),
        (1, 112, 112, 32),
    ];

    let mut group = c.benchmark_group("layers/batch_norm");
    group.sample_size(50);

    for (n, h, w, ch) in configs {
        let elements = n * h * w * ch;
        group.throughput(Throughput::Elements(elements as u64));

        let bn = BatchNorm::new(ch);
        let input = Tensor::ones(&[n, h, w, ch]);

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}x{}", h, w, ch)),
            &(n, h, w, ch),
            |b, _| b.iter(|| black_box(bn.forward(black_box(&input)).unwrap())),
        );
    }

    group.finish();
}

// ============================================================================
// Activation Benchmarks
// ============================================================================

fn bench_activations(c: &mut Criterion) {
    let configs = [(1, 56, 56, 64), (1, 28, 28, 128), (1, 14, 14, 256)];

    let mut group = c.benchmark_group("layers/activations");
    group.sample_size(50);

    for (n, h, w, ch) in configs {
        let elements = n * h * w * ch;
        let size_label = format!("{}x{}x{}", h, w, ch);

        group.throughput(Throughput::Elements(elements as u64));

        let input = Tensor::ones(&[n, h, w, ch]);

        // ReLU
        let relu = ReLU::new();
        group.bench_with_input(
            BenchmarkId::new("relu", &size_label),
            &(n, h, w, ch),
            |b, _| b.iter(|| black_box(relu.forward(black_box(&input)).unwrap())),
        );

        // ReLU6
        let relu6 = ReLU6::new();
        group.bench_with_input(
            BenchmarkId::new("relu6", &size_label),
            &(n, h, w, ch),
            |b, _| b.iter(|| black_box(relu6.forward(black_box(&input)).unwrap())),
        );

        // Swish
        let swish = Swish::new();
        group.bench_with_input(
            BenchmarkId::new("swish", &size_label),
            &(n, h, w, ch),
            |b, _| b.iter(|| black_box(swish.forward(black_box(&input)).unwrap())),
        );

        // HardSwish
        let hard_swish = HardSwish::new();
        group.bench_with_input(
            BenchmarkId::new("hard_swish", &size_label),
            &(n, h, w, ch),
            |b, _| b.iter(|| black_box(hard_swish.forward(black_box(&input)).unwrap())),
        );
    }

    group.finish();
}

// ============================================================================
// Pooling Benchmarks
// ============================================================================

fn bench_pooling(c: &mut Criterion) {
    let configs = [
        (1, 8, 8, 64),
        (1, 28, 28, 128),
        (1, 56, 56, 64),
        (1, 7, 7, 576), // MobileNetV3-Small final
        (1, 7, 7, 960), // MobileNetV3-Large final
    ];

    let mut group = c.benchmark_group("layers/pooling");
    group.sample_size(100);

    for (n, h, w, ch) in configs {
        let size_label = format!("{}x{}x{}", h, w, ch);

        // GlobalAvgPool
        let gap = GlobalAvgPool::new();
        let input = Tensor::ones(&[n, h, w, ch]);
        let input_elements = (n * h * w * ch) as u64;

        group.throughput(Throughput::Elements(input_elements));
        group.bench_with_input(
            BenchmarkId::new("global_avg", &size_label),
            &(n, h, w, ch),
            |b, _| b.iter(|| black_box(gap.forward(black_box(&input)).unwrap())),
        );

        // MaxPool2d (only for sizes >= 4)
        if h >= 4 && w >= 4 {
            let maxpool = MaxPool2d::new(2, 2, 0);
            group.bench_with_input(
                BenchmarkId::new("max_2x2", &size_label),
                &(n, h, w, ch),
                |b, _| b.iter(|| black_box(maxpool.forward(black_box(&input)).unwrap())),
            );
        }
    }

    group.finish();
}

// ============================================================================
// Full Block Benchmarks (MobileNet-style)
// ============================================================================

fn bench_full_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("blocks/mobilenet_style");
    group.sample_size(20);

    // MobileNet-style block: DepthwiseSeparable -> BatchNorm -> ReLU6
    let input = Tensor::ones(&[1, 56, 56, 32]);
    let dw_conv = DepthwiseSeparableConv::new(32, 64, 3, 1, 1);
    let bn = BatchNorm::new(64);
    let relu6 = ReLU6::new();

    let elements = 56 * 56 * 64;
    group.throughput(Throughput::Elements(elements as u64));

    group.bench_function("dw_conv_bn_relu6_56x56x32->64", |b| {
        b.iter(|| {
            let x = dw_conv.forward(black_box(&input)).unwrap();
            let x = bn.forward(&x).unwrap();
            black_box(relu6.forward(&x).unwrap())
        })
    });

    // Larger block
    let input2 = Tensor::ones(&[1, 28, 28, 64]);
    let dw_conv2 = DepthwiseSeparableConv::new(64, 128, 3, 1, 1);
    let bn2 = BatchNorm::new(128);

    let elements2 = 28 * 28 * 128;
    group.throughput(Throughput::Elements(elements2 as u64));

    group.bench_function("dw_conv_bn_relu6_28x28x64->128", |b| {
        b.iter(|| {
            let x = dw_conv2.forward(black_box(&input2)).unwrap();
            let x = bn2.forward(&x).unwrap();
            black_box(relu6.forward(&x).unwrap())
        })
    });

    // With stride=2 (downsampling)
    let input3 = Tensor::ones(&[1, 56, 56, 64]);
    let dw_conv3 = DepthwiseSeparableConv::new(64, 128, 3, 2, 1);
    let bn3 = BatchNorm::new(128);

    let elements3 = 28 * 28 * 128; // Output is halved
    group.throughput(Throughput::Elements(elements3 as u64));

    group.bench_function("dw_conv_bn_relu6_56x56x64->28x28x128_stride2", |b| {
        b.iter(|| {
            let x = dw_conv3.forward(black_box(&input3)).unwrap();
            let x = bn3.forward(&x).unwrap();
            black_box(relu6.forward(&x).unwrap())
        })
    });

    group.finish();
}

// ============================================================================
// Batch Size Scaling Benchmarks
// ============================================================================

fn bench_batch_scaling(c: &mut Criterion) {
    let batch_sizes = [1, 2, 4, 8, 16];
    let h = 56;
    let w = 56;
    let in_c = 32;
    let out_c = 64;

    let mut group = c.benchmark_group("scaling/batch_size");
    group.sample_size(20);

    let conv = Conv2d::new(in_c, out_c, 3, 1, 1);

    for batch in batch_sizes {
        let elements = batch * h * w * out_c;
        group.throughput(Throughput::Elements(elements as u64));

        let input = Tensor::ones(&[batch, h, w, in_c]);

        group.bench_with_input(BenchmarkId::new("conv2d", batch), &batch, |b, _| {
            b.iter(|| black_box(conv.forward(black_box(&input)).unwrap()))
        });
    }

    group.finish();
}

// ============================================================================
// Memory Layout Comparison
// ============================================================================

fn bench_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor/operations");
    group.sample_size(50);

    // Test tensor creation
    let shapes = [(1, 224, 224, 3), (1, 56, 56, 64), (16, 28, 28, 128)];

    for (n, h, w, c) in shapes {
        let elements = n * h * w * c;
        let label = format!("{}x{}x{}x{}", n, h, w, c);

        group.throughput(Throughput::Elements(elements as u64));

        group.bench_with_input(BenchmarkId::new("zeros", &label), &(n, h, w, c), |b, _| {
            b.iter(|| black_box(Tensor::zeros(&[n, h, w, c])))
        });

        group.bench_with_input(BenchmarkId::new("ones", &label), &(n, h, w, c), |b, _| {
            b.iter(|| black_box(Tensor::ones(&[n, h, w, c])))
        });

        let tensor = Tensor::ones(&[n, h, w, c]);
        group.bench_with_input(BenchmarkId::new("clone", &label), &(n, h, w, c), |b, _| {
            b.iter(|| black_box(tensor.clone()))
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    simd_benches,
    bench_simd_relu,
    bench_simd_relu6,
    bench_simd_dot_product,
    bench_simd_batch_norm,
    bench_simd_conv_3x3,
    bench_simd_global_avg_pool,
);

criterion_group!(
    layer_benches,
    bench_conv2d_layer,
    bench_depthwise_separable,
    bench_batch_norm_layer,
    bench_activations,
    bench_pooling,
);

criterion_group!(block_benches, bench_full_block, bench_batch_scaling,);

criterion_group!(misc_benches, bench_tensor_operations,);

criterion_main!(simd_benches, layer_benches, block_benches, misc_benches);
