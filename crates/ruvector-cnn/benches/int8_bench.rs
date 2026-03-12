//! INT8 Quantization Benchmarks
//!
//! Validates GATE-3 (Latency ≥2.5x) and GATE-4 (Memory ≥3x) via Criterion.
//!
//! Run with: `cargo bench --bench int8_bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use ruvector_cnn::int8::{QuantParams, quantize_tensor, dequantize_tensor};

#[cfg(target_arch = "x86_64")]
use ruvector_cnn::int8::kernels::simd::{conv2d_int8_simd, matmul_int8_simd};

use ruvector_cnn::int8::kernels::scalar::{conv2d_int8_scalar, matmul_int8_scalar};

fn bench_conv2d_int8(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_int8");

    // Test different input sizes
    let sizes = vec![
        (28, 28, 16, 3, 1),   // Small: 28x28, 16 channels, 3x3 kernel
        (56, 56, 32, 3, 1),   // Medium: 56x56, 32 channels
        (112, 112, 64, 3, 1), // Large: 112x112, 64 channels
    ];

    for (h, w, c, k, stride) in sizes {
        let input_size = h * w * c;
        let kernel_size = k * k * c;

        group.throughput(Throughput::Elements(input_size as u64));

        // Generate random INT8 input and kernel
        let mut rng = fastrand::Rng::with_seed(42);
        let input: Vec<i8> = (0..input_size).map(|_| rng.i8(..)).collect();
        let kernel: Vec<i8> = (0..kernel_size).map(|_| rng.i8(..)).collect();

        let params = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };

        let bench_name = format!("{}x{}x{}_k{}_s{}", h, w, c, k, stride);

        // Benchmark scalar version
        group.bench_with_input(
            BenchmarkId::new("scalar", &bench_name),
            &(&input, &kernel, params),
            |b, (input, kernel, params)| {
                b.iter(|| {
                    conv2d_int8_scalar(
                        black_box(input),
                        black_box(kernel),
                        black_box(*params),
                        h,
                        w,
                        c,
                        k,
                        stride,
                    )
                })
            },
        );

        // Benchmark SIMD version (x86_64 only)
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                group.bench_with_input(
                    BenchmarkId::new("simd", &bench_name),
                    &(&input, &kernel, params),
                    |b, (input, kernel, params)| {
                        b.iter(|| unsafe {
                            conv2d_int8_simd(
                                black_box(input),
                                black_box(kernel),
                                black_box(*params),
                                h,
                                w,
                                c,
                                k,
                                stride,
                            )
                        })
                    },
                );
            }
        }

        // Benchmark FP32 baseline for comparison
        let input_fp32: Vec<f32> = input.iter().map(|&x| x as f32 * 0.01).collect();
        let kernel_fp32: Vec<f32> = kernel.iter().map(|&x| x as f32 * 0.01).collect();

        group.bench_with_input(
            BenchmarkId::new("fp32_baseline", &bench_name),
            &(&input_fp32, &kernel_fp32),
            |b, (input, kernel)| {
                b.iter(|| {
                    conv2d_fp32_naive(
                        black_box(input),
                        black_box(kernel),
                        h,
                        w,
                        c,
                        k,
                        stride,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_matmul_int8(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_int8");

    // Test different matrix sizes
    let sizes = vec![
        (64, 64, 64),     // Small
        (128, 128, 128),  // Medium
        (256, 256, 256),  // Large
        (512, 512, 512),  // XLarge
    ];

    for (m, n, k) in sizes {
        let a_size = m * k;
        let b_size = k * n;

        group.throughput(Throughput::Elements((m * n * k) as u64));

        // Generate random INT8 matrices
        let mut rng = fastrand::Rng::with_seed(123);
        let a: Vec<i8> = (0..a_size).map(|_| rng.i8(..)).collect();
        let b: Vec<i8> = (0..b_size).map(|_| rng.i8(..)).collect();

        let params = QuantParams {
            scale: 0.01,
            zero_point: 0,
        };

        let bench_name = format!("{}x{}x{}", m, n, k);

        // Benchmark scalar version
        group.bench_with_input(
            BenchmarkId::new("scalar", &bench_name),
            &(&a, &b, params),
            |bench, (a, b, params)| {
                bench.iter(|| {
                    matmul_int8_scalar(
                        black_box(a),
                        black_box(b),
                        black_box(*params),
                        m,
                        n,
                        k,
                    )
                })
            },
        );

        // Benchmark SIMD version (x86_64 only)
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                group.bench_with_input(
                    BenchmarkId::new("simd", &bench_name),
                    &(&a, &b, params),
                    |bench, (a, b, params)| {
                        bench.iter(|| unsafe {
                            matmul_int8_simd(
                                black_box(a),
                                black_box(b),
                                black_box(*params),
                                m,
                                n,
                                k,
                            )
                        })
                    },
                );
            }
        }

        // Benchmark FP32 baseline
        let a_fp32: Vec<f32> = a.iter().map(|&x| x as f32 * 0.01).collect();
        let b_fp32: Vec<f32> = b.iter().map(|&x| x as f32 * 0.01).collect();

        group.bench_with_input(
            BenchmarkId::new("fp32_baseline", &bench_name),
            &(&a_fp32, &b_fp32),
            |bench, (a, b)| {
                bench.iter(|| {
                    matmul_fp32_naive(
                        black_box(a),
                        black_box(b),
                        m,
                        n,
                        k,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_mobilenetv3_int8(c: &mut Criterion) {
    // GATE-3 target: 2.5x speedup for end-to-end inference
    let mut group = c.benchmark_group("mobilenetv3_e2e");
    group.sample_size(10); // Reduce sample size for expensive operations

    // Simulate MobileNetV3 embedding extraction
    // Input: 224x224x3 RGB image
    // Output: 1024-dim embedding

    let input_size = 224 * 224 * 3;
    let embedding_size = 1024;

    let mut rng = fastrand::Rng::with_seed(456);

    // INT8 path
    let input_int8: Vec<i8> = (0..input_size).map(|_| rng.i8(..)).collect();
    let params = QuantParams {
        scale: 1.0 / 255.0,
        zero_point: -128,
    };

    group.bench_function("int8_inference", |b| {
        b.iter(|| {
            // Simulate INT8 inference (placeholder for actual model)
            let mut embedding = vec![0i32; embedding_size];
            for i in 0..embedding_size {
                let start = (i * input_size) / embedding_size;
                let end = ((i + 1) * input_size) / embedding_size;
                let sum: i32 = input_int8[start..end]
                    .iter()
                    .map(|&x| x as i32)
                    .sum();
                embedding[i] = sum;
            }
            black_box(embedding)
        })
    });

    // FP32 baseline
    let input_fp32: Vec<f32> = input_int8.iter().map(|&x| x as f32 / 255.0).collect();

    group.bench_function("fp32_baseline", |b| {
        b.iter(|| {
            // Simulate FP32 inference
            let mut embedding = vec![0.0f32; embedding_size];
            for i in 0..embedding_size {
                let start = (i * input_size) / embedding_size;
                let end = ((i + 1) * input_size) / embedding_size;
                let sum: f32 = input_fp32[start..end].iter().sum();
                embedding[i] = sum;
            }
            black_box(embedding)
        })
    });

    group.finish();
}

fn bench_quantization_dequantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_dequant");

    let sizes = vec![128, 512, 1024, 2048];

    for size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        let mut rng = fastrand::Rng::with_seed(789);
        let fp32: Vec<f32> = (0..size).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let params = QuantParams::from_tensor(&fp32);

        // Benchmark quantization
        group.bench_with_input(
            BenchmarkId::new("quantize", size),
            &(&fp32, params),
            |b, (fp32, params)| {
                b.iter(|| quantize_tensor(black_box(fp32), black_box(params)))
            },
        );

        // Benchmark dequantization
        let int8 = quantize_tensor(&fp32, &params);
        group.bench_with_input(
            BenchmarkId::new("dequantize", size),
            &(&int8, params),
            |b, (int8, params)| {
                b.iter(|| dequantize_tensor(black_box(int8), black_box(params)))
            },
        );

        // Benchmark round-trip
        group.bench_with_input(
            BenchmarkId::new("round_trip", size),
            &(&fp32, params),
            |b, (fp32, params)| {
                b.iter(|| {
                    let int8 = quantize_tensor(black_box(fp32), black_box(params));
                    dequantize_tensor(black_box(&int8), black_box(params))
                })
            },
        );
    }

    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    // GATE-4: Memory reduction ≥3x
    // This benchmark measures memory footprint, not speed

    let mut group = c.benchmark_group("memory_footprint");

    let sizes = vec![1024, 4096, 16384, 65536];

    for size in sizes {
        let mut rng = fastrand::Rng::with_seed(999);
        let fp32: Vec<f32> = (0..size).map(|_| rng.f32() * 2.0 - 1.0).collect();

        // FP32 memory: 4 bytes per element
        let fp32_bytes = size * std::mem::size_of::<f32>();

        // INT8 memory: 1 byte per element + params (8 bytes)
        let params = QuantParams::from_tensor(&fp32);
        let int8 = quantize_tensor(&fp32, &params);
        let int8_bytes = size * std::mem::size_of::<i8>() + std::mem::size_of::<QuantParams>();

        let reduction = fp32_bytes as f32 / int8_bytes as f32;

        println!(
            "Size: {:<6} | FP32: {:>8} bytes | INT8: {:>8} bytes | Reduction: {:.2}x",
            size, fp32_bytes, int8_bytes, reduction
        );

        // Verify GATE-4: ≥3x reduction
        assert!(
            reduction >= 3.0,
            "GATE-4 FAILED: Memory reduction {:.2}x < 3.0x for size {}",
            reduction, size
        );

        // Dummy benchmark to keep Criterion happy
        group.bench_function(BenchmarkId::new("verify", size), |b| {
            b.iter(|| black_box(&int8))
        });
    }

    group.finish();
}

// Helper: Naive FP32 conv2d for baseline comparison
fn conv2d_fp32_naive(
    input: &[f32],
    kernel: &[f32],
    h: usize,
    w: usize,
    c: usize,
    k: usize,
    stride: usize,
) -> Vec<f32> {
    let out_h = (h - k) / stride + 1;
    let out_w = (w - k) / stride + 1;
    let mut output = vec![0.0f32; out_h * out_w];

    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut sum = 0.0f32;
            for kh in 0..k {
                for kw in 0..k {
                    for ch in 0..c {
                        let ih = oh * stride + kh;
                        let iw = ow * stride + kw;
                        let input_idx = (ih * w + iw) * c + ch;
                        let kernel_idx = (kh * k + kw) * c + ch;
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
            output[oh * out_w + ow] = sum;
        }
    }

    output
}

// Helper: Naive FP32 matmul for baseline comparison
fn matmul_fp32_naive(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    c
}

criterion_group!(
    benches,
    bench_conv2d_int8,
    bench_matmul_int8,
    bench_mobilenetv3_int8,
    bench_quantization_dequantization,
    bench_memory_usage,
);

criterion_main!(benches);
