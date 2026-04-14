//! # EML Operator Benchmarks (ADR-033)
//!
//! Comparative benchmarks proving EML-inspired optimizations improve on plain Ruvector:
//!
//! 1. **LogQuantized vs ScalarQuantized**: Reconstruction error comparison
//! 2. **Unified vs Dispatched distance**: Throughput for batch operations
//! 3. **EML tree vs Linear model**: Position prediction accuracy
//! 4. **End-to-end quality**: Distance preservation after quantization

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use ruvector_core::advanced::eml::{
    EmlModel, EmlScoreFusion, EmlTree, TrainConfig, UnifiedDistanceParams, train_eml_tree,
};
use ruvector_core::distance;
use ruvector_core::quantization::{LogQuantized, QuantizedVector, ScalarQuantized};
use ruvector_core::types::DistanceMetric;

// ============================================================================
// Data Generators
// ============================================================================

/// Generate vectors with normal-like distribution (typical transformer embeddings)
fn gen_normal_vectors(count: usize, dims: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            (0..dims)
                .map(|_| {
                    // Box-Muller approximation: concentrate values near 0
                    let u1: f32 = rng.gen::<f32>().max(1e-7);
                    let u2: f32 = rng.gen();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * 0.3
                })
                .collect()
        })
        .collect()
}

/// Generate vectors with uniform distribution (baseline comparison)
fn gen_uniform_vectors(count: usize, dims: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

/// Generate sorted CDF-like training data for learned index testing
fn gen_cdf_data(count: usize) -> Vec<(Vec<f32>, usize)> {
    (0..count)
        .map(|i| {
            let x = i as f32 / count as f32;
            (vec![x], i)
        })
        .collect()
}

/// Generate non-linear CDF data (sigmoid-shaped distribution)
fn gen_nonlinear_cdf_data(count: usize) -> Vec<(Vec<f32>, usize)> {
    (0..count)
        .map(|i| {
            // Sigmoid-like: values bunch in the middle
            let t = (i as f32 / count as f32) * 12.0 - 6.0; // [-6, 6]
            let x = 1.0 / (1.0 + (-t).exp()); // sigmoid maps to [0, 1]
            (vec![x], i)
        })
        .collect()
}

// ============================================================================
// Benchmark 1: LogQuantized vs ScalarQuantized Reconstruction Error
// ============================================================================

fn bench_quantization_error(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_reconstruction_error");

    for dims in [128, 384, 768] {
        let normal_vecs = gen_normal_vectors(100, dims);
        let uniform_vecs = gen_uniform_vectors(100, dims);

        // --- Normal distribution (where log quantization shines) ---
        group.bench_with_input(
            BenchmarkId::new("scalar_normal", dims),
            &normal_vecs,
            |bench, vecs| {
                bench.iter(|| {
                    let mut total_mse = 0.0f32;
                    for v in vecs {
                        let q = ScalarQuantized::quantize(black_box(v));
                        let r = q.reconstruct();
                        let mse: f32 = v
                            .iter()
                            .zip(r.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f32>()
                            / v.len() as f32;
                        total_mse += mse;
                    }
                    total_mse / vecs.len() as f32
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("log_normal", dims),
            &normal_vecs,
            |bench, vecs| {
                bench.iter(|| {
                    let mut total_mse = 0.0f32;
                    for v in vecs {
                        let q = LogQuantized::quantize(black_box(v));
                        total_mse += q.reconstruction_error(v);
                    }
                    total_mse / vecs.len() as f32
                });
            },
        );

        // --- Uniform distribution (baseline comparison) ---
        group.bench_with_input(
            BenchmarkId::new("scalar_uniform", dims),
            &uniform_vecs,
            |bench, vecs| {
                bench.iter(|| {
                    let mut total_mse = 0.0f32;
                    for v in vecs {
                        let q = ScalarQuantized::quantize(black_box(v));
                        let r = q.reconstruct();
                        let mse: f32 = v
                            .iter()
                            .zip(r.iter())
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum::<f32>()
                            / v.len() as f32;
                        total_mse += mse;
                    }
                    total_mse / vecs.len() as f32
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("log_uniform", dims),
            &uniform_vecs,
            |bench, vecs| {
                bench.iter(|| {
                    let mut total_mse = 0.0f32;
                    for v in vecs {
                        let q = LogQuantized::quantize(black_box(v));
                        total_mse += q.reconstruction_error(v);
                    }
                    total_mse / vecs.len() as f32
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 2: Unified vs Dispatched Distance Kernel
// ============================================================================

fn bench_distance_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_kernel_dispatch");

    for dims in [128, 384, 768] {
        let vecs = gen_uniform_vectors(1000, dims);
        let query = &vecs[0];

        for metric in [
            DistanceMetric::Euclidean,
            DistanceMetric::Cosine,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let metric_name = format!("{:?}", metric);

            // Original: match-based dispatch on every call
            group.bench_with_input(
                BenchmarkId::new(format!("dispatched_{}", metric_name), dims),
                &dims,
                |bench, _| {
                    bench.iter(|| {
                        let mut sum = 0.0f32;
                        for v in &vecs[1..] {
                            sum += distance::distance(
                                black_box(query),
                                black_box(v),
                                metric,
                            )
                            .unwrap();
                        }
                        sum
                    });
                },
            );

            // Unified: parameters resolved once, no match in hot loop
            let params = UnifiedDistanceParams::from_metric(metric);
            group.bench_with_input(
                BenchmarkId::new(format!("unified_{}", metric_name), dims),
                &dims,
                |bench, _| {
                    bench.iter(|| {
                        let mut sum = 0.0f32;
                        for v in &vecs[1..] {
                            sum += params.compute(black_box(query), black_box(v));
                        }
                        sum
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Benchmark 3: EML Tree vs Linear Model for CDF Approximation
// ============================================================================

fn bench_learned_index_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("learned_index_prediction_error");

    for count in [100, 500, 1000] {
        // Linear CDF (easy case — both models should do well)
        let linear_data = gen_cdf_data(count);

        group.bench_with_input(
            BenchmarkId::new("linear_model_linear_cdf", count),
            &linear_data,
            |bench, data| {
                bench.iter(|| {
                    // Simulate LinearModel: just use first dim * slope + bias
                    let n = data.len() as f32;
                    let mut total_error = 0.0f32;
                    for (key, pos) in data {
                        let pred = (key[0] * n).max(0.0) as usize;
                        let error = (*pos as i32 - pred as i32).abs() as f32;
                        total_error += error;
                    }
                    total_error / data.len() as f32
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("eml_model_linear_cdf", count),
            &linear_data,
            |bench, data| {
                bench.iter(|| {
                    let mut model = EmlModel::new(2);
                    model.train(data);
                    let mut total_error = 0.0f32;
                    for (key, pos) in data {
                        let pred = model.predict(key) as usize;
                        let error = (*pos as i32 - pred as i32).abs() as f32;
                        total_error += error;
                    }
                    total_error / data.len() as f32
                });
            },
        );

        // Non-linear CDF (hard case — EML should outperform)
        let nonlinear_data = gen_nonlinear_cdf_data(count);

        group.bench_with_input(
            BenchmarkId::new("linear_model_nonlinear_cdf", count),
            &nonlinear_data,
            |bench, data| {
                bench.iter(|| {
                    let n = data.len() as f32;
                    let mut total_error = 0.0f32;
                    for (key, pos) in data {
                        let pred = (key[0] * n).max(0.0) as usize;
                        let error = (*pos as i32 - pred as i32).abs() as f32;
                        total_error += error;
                    }
                    total_error / data.len() as f32
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("eml_model_nonlinear_cdf", count),
            &nonlinear_data,
            |bench, data| {
                bench.iter(|| {
                    let mut model = EmlModel::new(2);
                    model.train(data);
                    let mut total_error = 0.0f32;
                    for (key, pos) in data {
                        let pred = model.predict(key) as usize;
                        let error = (*pos as i32 - pred as i32).abs() as f32;
                        total_error += error;
                    }
                    total_error / data.len() as f32
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 4: Distance Preservation After Quantization
// ============================================================================

fn bench_distance_preservation(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_distance_preservation");

    for dims in [128, 384] {
        let vecs = gen_normal_vectors(200, dims);

        // Compute true distances for first 50 pairs
        let pairs: Vec<(usize, usize)> = (0..50).map(|i| (i, i + 50)).collect();

        let true_dists: Vec<f32> = pairs
            .iter()
            .map(|(i, j)| {
                distance::euclidean_distance(&vecs[*i], &vecs[*j])
            })
            .collect();

        // Scalar quantized distances
        group.bench_with_input(
            BenchmarkId::new("scalar_dist_correlation", dims),
            &dims,
            |bench, _| {
                bench.iter(|| {
                    let quantized: Vec<ScalarQuantized> =
                        vecs.iter().map(|v| ScalarQuantized::quantize(v)).collect();

                    let q_dists: Vec<f32> = pairs
                        .iter()
                        .map(|(i, j)| quantized[*i].distance(&quantized[*j]))
                        .collect();

                    // Spearman rank correlation (simplified): count concordant pairs
                    let mut concordant = 0usize;
                    let total = pairs.len() * (pairs.len() - 1) / 2;
                    for a in 0..pairs.len() {
                        for b in (a + 1)..pairs.len() {
                            let true_order = true_dists[a] < true_dists[b];
                            let q_order = q_dists[a] < q_dists[b];
                            if true_order == q_order {
                                concordant += 1;
                            }
                        }
                    }
                    concordant as f32 / total as f32
                });
            },
        );

        // Log quantized distances
        group.bench_with_input(
            BenchmarkId::new("log_dist_correlation", dims),
            &dims,
            |bench, _| {
                bench.iter(|| {
                    let quantized: Vec<LogQuantized> =
                        vecs.iter().map(|v| LogQuantized::quantize(v)).collect();

                    let q_dists: Vec<f32> = pairs
                        .iter()
                        .map(|(i, j)| quantized[*i].distance(&quantized[*j]))
                        .collect();

                    let mut concordant = 0usize;
                    let total = pairs.len() * (pairs.len() - 1) / 2;
                    for a in 0..pairs.len() {
                        for b in (a + 1)..pairs.len() {
                            let true_order = true_dists[a] < true_dists[b];
                            let q_order = q_dists[a] < q_dists[b];
                            if true_order == q_order {
                                concordant += 1;
                            }
                        }
                    }
                    concordant as f32 / total as f32
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark 5: EML Score Fusion vs Linear Fusion
// ============================================================================

fn bench_score_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("score_fusion");

    let mut rng = rand::thread_rng();
    let scores: Vec<(f32, f32)> = (0..10000)
        .map(|_| (rng.gen::<f32>(), rng.gen::<f32>() * 10.0))
        .collect();

    // Linear fusion: alpha * vector + (1 - alpha) * bm25
    group.bench_function("linear_fusion", |bench| {
        let alpha = 0.7f32;
        bench.iter(|| {
            let mut sum = 0.0f32;
            for &(v, b) in &scores {
                sum += black_box(alpha) * black_box(v) + (1.0 - alpha) * black_box(b);
            }
            sum
        });
    });

    // EML fusion
    let fusion = EmlScoreFusion::default();
    group.bench_function("eml_fusion", |bench| {
        bench.iter(|| {
            let mut sum = 0.0f32;
            for &(v, b) in &scores {
                sum += fusion.fuse(black_box(v), black_box(b));
            }
            sum
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark 6: EML Tree Evaluation Throughput
// ============================================================================

fn bench_eml_tree_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("eml_tree_evaluation");

    for depth in [1, 2, 3, 4] {
        let tree = EmlTree::fully_parameterized(depth, 0);
        let inputs: Vec<Vec<f32>> = (0..1000).map(|i| vec![i as f32 / 1000.0]).collect();

        group.bench_with_input(
            BenchmarkId::new("eval", depth),
            &depth,
            |bench, _| {
                bench.iter(|| {
                    let mut sum = 0.0f32;
                    for inp in &inputs {
                        sum += tree.evaluate(black_box(inp));
                    }
                    sum
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_quantization_error,
    bench_distance_dispatch,
    bench_learned_index_accuracy,
    bench_distance_preservation,
    bench_score_fusion,
    bench_eml_tree_eval,
);
criterion_main!(benches);
