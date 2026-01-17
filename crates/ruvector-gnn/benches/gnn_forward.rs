//! Benchmarks for GNN forward pass operations.
//!
//! Performance targets from ADR:
//! - NQED: GNN forward pass < 100us for d=11
//! - AV-QKCM: Kernel update < 10us per sample

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ruvector_gnn::{
    layer::RuvectorLayer,
    search::{cosine_similarity, differentiable_search, hierarchical_forward},
    tensor::Tensor,
    training::{info_nce_loss, local_contrastive_loss, sgd_step, Loss, LossType, Optimizer, OptimizerType},
    ewc::ElasticWeightConsolidation,
};
use ndarray::Array2;

/// Benchmark cosine similarity for various dimensions
fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");

    for dim in [11, 64, 128, 256, 512, 768, 1024].iter() {
        let vec_a: Vec<f32> = (0..*dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let vec_b: Vec<f32> = (0..*dim).map(|i| (i as f32 * 0.01).cos()).collect();

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |b, _| {
            b.iter(|| {
                black_box(cosine_similarity(black_box(&vec_a), black_box(&vec_b)))
            });
        });
    }

    group.finish();
}

/// Benchmark differentiable search (soft attention)
fn bench_differentiable_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("differentiable_search");

    // Test with d=11 (NQED target dimension)
    let dim = 11;
    let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();

    for num_candidates in [10, 50, 100, 500, 1000].iter() {
        let candidates: Vec<Vec<f32>> = (0..*num_candidates)
            .map(|j| (0..dim).map(|i| ((i + j) as f32 * 0.01).cos()).collect())
            .collect();

        group.throughput(Throughput::Elements(*num_candidates as u64));
        group.bench_with_input(
            BenchmarkId::new("d11", num_candidates),
            num_candidates,
            |b, _| {
                b.iter(|| {
                    differentiable_search(
                        black_box(&query),
                        black_box(&candidates),
                        5,  // top-k
                        1.0 // temperature
                    )
                });
            }
        );
    }

    // Higher dimension test (d=64)
    let dim = 64;
    let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();

    for num_candidates in [100, 500].iter() {
        let candidates: Vec<Vec<f32>> = (0..*num_candidates)
            .map(|j| (0..dim).map(|i| ((i + j) as f32 * 0.01).cos()).collect())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("d64", num_candidates),
            num_candidates,
            |b, _| {
                b.iter(|| {
                    differentiable_search(
                        black_box(&query),
                        black_box(&candidates),
                        5,
                        1.0
                    )
                });
            }
        );
    }

    group.finish();
}

/// Benchmark RuvectorLayer forward pass - Target: <100us for d=11
fn bench_ruvector_layer_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("ruvector_layer_forward");

    // NQED target: d=11
    for (input_dim, hidden_dim, heads) in [(11, 16, 2), (11, 32, 4), (64, 128, 4), (128, 256, 8)].iter() {
        let layer = RuvectorLayer::new(*input_dim, *hidden_dim, *heads, 0.1);

        // Create test data
        let node_embedding: Vec<f32> = (0..*input_dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let neighbor_embeddings: Vec<Vec<f32>> = (0..5)
            .map(|j| (0..*input_dim).map(|i| ((i + j) as f32 * 0.01).cos()).collect())
            .collect();
        let edge_weights = vec![0.2, 0.3, 0.15, 0.2, 0.15];

        group.bench_with_input(
            BenchmarkId::new(format!("d{}_h{}_heads{}", input_dim, hidden_dim, heads), 5),
            &5,
            |b, _| {
                b.iter(|| {
                    layer.forward(
                        black_box(&node_embedding),
                        black_box(&neighbor_embeddings),
                        black_box(&edge_weights)
                    )
                });
            }
        );
    }

    // Vary number of neighbors
    let layer = RuvectorLayer::new(11, 16, 2, 0.1);
    let node_embedding: Vec<f32> = (0..11).map(|i| (i as f32 * 0.1).sin()).collect();

    for num_neighbors in [1, 5, 10, 20].iter() {
        let neighbor_embeddings: Vec<Vec<f32>> = (0..*num_neighbors)
            .map(|j| (0..11).map(|i| ((i + j) as f32 * 0.01).cos()).collect())
            .collect();
        let edge_weights: Vec<f32> = (0..*num_neighbors).map(|i| 1.0 / *num_neighbors as f32).collect();

        group.bench_with_input(
            BenchmarkId::new("d11_neighbors", num_neighbors),
            num_neighbors,
            |b, _| {
                b.iter(|| {
                    layer.forward(
                        black_box(&node_embedding),
                        black_box(&neighbor_embeddings),
                        black_box(&edge_weights)
                    )
                });
            }
        );
    }

    group.finish();
}

/// Benchmark hierarchical forward pass through multiple GNN layers
fn bench_hierarchical_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("hierarchical_forward");

    // Create layers
    let num_layers = 3;
    let gnn_layers: Vec<RuvectorLayer> = (0..num_layers)
        .map(|_| RuvectorLayer::new(11, 11, 1, 0.0))
        .collect();

    let query: Vec<f32> = (0..11).map(|i| (i as f32 * 0.1).sin()).collect();

    for nodes_per_layer in [10, 50, 100].iter() {
        let layer_embeddings: Vec<Vec<Vec<f32>>> = (0..num_layers)
            .map(|_| {
                (0..*nodes_per_layer)
                    .map(|j| (0..11).map(|i| ((i + j) as f32 * 0.01).cos()).collect())
                    .collect()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("3_layers", nodes_per_layer),
            nodes_per_layer,
            |b, _| {
                b.iter(|| {
                    hierarchical_forward(
                        black_box(&query),
                        black_box(&layer_embeddings),
                        black_box(&gnn_layers)
                    )
                });
            }
        );
    }

    group.finish();
}

/// Benchmark InfoNCE loss computation - Target: <10us per sample
fn bench_info_nce_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("info_nce_loss");

    // Test with typical embedding dimensions
    for dim in [11, 64, 128, 256].iter() {
        let anchor: Vec<f32> = (0..*dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let positive: Vec<f32> = (0..*dim).map(|i| (i as f32 * 0.01).sin() + 0.1).collect();

        // Test with varying numbers of negatives
        for num_negatives in [10, 64, 128].iter() {
            let negatives: Vec<Vec<f32>> = (0..*num_negatives)
                .map(|j| (0..*dim).map(|i| ((i + j * 10) as f32 * 0.01).cos()).collect())
                .collect();
            let neg_refs: Vec<&[f32]> = negatives.iter().map(|v| v.as_slice()).collect();

            group.bench_with_input(
                BenchmarkId::new(format!("d{}_neg{}", dim, num_negatives), *num_negatives),
                num_negatives,
                |b, _| {
                    b.iter(|| {
                        info_nce_loss(
                            black_box(&anchor),
                            black_box(&[positive.as_slice()]),
                            black_box(&neg_refs),
                            0.07
                        )
                    });
                }
            );
        }
    }

    group.finish();
}

/// Benchmark local contrastive loss for graph-based learning
fn bench_local_contrastive_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("local_contrastive_loss");

    let dim = 64;
    let node: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();

    for num_neighbors in [5, 10, 20].iter() {
        let neighbors: Vec<Vec<f32>> = (0..*num_neighbors)
            .map(|j| (0..dim).map(|i| (i as f32 * 0.1).sin() + (j as f32 * 0.01)).collect())
            .collect();
        let non_neighbors: Vec<Vec<f32>> = (0..50)
            .map(|j| (0..dim).map(|i| ((i + j * 10) as f32 * 0.01).cos()).collect())
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(num_neighbors),
            num_neighbors,
            |b, _| {
                b.iter(|| {
                    local_contrastive_loss(
                        black_box(&node),
                        black_box(&neighbors),
                        black_box(&non_neighbors),
                        0.07
                    )
                });
            }
        );
    }

    group.finish();
}

/// Benchmark SGD step (kernel update) - Target: <10us per sample
fn bench_sgd_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgd_step");

    for dim in [11, 64, 128, 256, 512].iter() {
        let mut embedding: Vec<f32> = (0..*dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let gradient: Vec<f32> = (0..*dim).map(|i| (i as f32 * 0.001).cos()).collect();

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(dim),
            dim,
            |b, _| {
                b.iter(|| {
                    sgd_step(
                        black_box(&mut embedding.clone()),
                        black_box(&gradient),
                        0.001
                    )
                });
            }
        );
    }

    group.finish();
}

/// Benchmark Adam optimizer step
fn bench_adam_optimizer(c: &mut Criterion) {
    let mut group = c.benchmark_group("adam_optimizer");

    for size in [100, 500, 1000, 5000].iter() {
        let mut optimizer = Optimizer::new(OptimizerType::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        });

        let mut params = Array2::from_elem((1, *size), 0.5f32);
        let grads = Array2::from_elem((1, *size), 0.01f32);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    optimizer.step(
                        black_box(&mut params.clone()),
                        black_box(&grads)
                    )
                });
            }
        );
    }

    group.finish();
}

/// Benchmark EWC penalty computation
fn bench_ewc_penalty(c: &mut Criterion) {
    let mut group = c.benchmark_group("ewc_penalty");

    for num_weights in [100, 1000, 10000, 50000].iter() {
        // Setup EWC
        let mut ewc = ElasticWeightConsolidation::new(1000.0);
        let gradients: Vec<Vec<f32>> = (0..10)
            .map(|j| (0..*num_weights).map(|i| ((i + j * 100) as f32 * 0.001).sin()).collect())
            .collect();
        let grad_refs: Vec<&[f32]> = gradients.iter().map(|v| v.as_slice()).collect();
        ewc.compute_fisher(&grad_refs, 10);

        let anchor_weights: Vec<f32> = (0..*num_weights).map(|i| (i as f32 * 0.01).sin()).collect();
        ewc.consolidate(&anchor_weights);

        let current_weights: Vec<f32> = (0..*num_weights).map(|i| (i as f32 * 0.01).sin() + 0.1).collect();

        group.throughput(Throughput::Elements(*num_weights as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_weights),
            num_weights,
            |b, _| {
                b.iter(|| {
                    ewc.penalty(black_box(&current_weights))
                });
            }
        );
    }

    group.finish();
}

/// Benchmark Loss computation (MSE, CrossEntropy, BCE)
fn bench_loss_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("loss_computation");

    for batch_size in [32, 64, 128, 256].iter() {
        for output_dim in [10, 100, 1000].iter() {
            let predictions = Array2::from_elem((*batch_size, *output_dim), 0.5f32);
            let targets = Array2::from_elem((*batch_size, *output_dim), 0.7f32);

            group.bench_with_input(
                BenchmarkId::new(format!("mse_b{}_d{}", batch_size, output_dim), batch_size),
                batch_size,
                |b, _| {
                    b.iter(|| {
                        Loss::compute(
                            LossType::Mse,
                            black_box(&predictions),
                            black_box(&targets)
                        )
                    });
                }
            );

            group.bench_with_input(
                BenchmarkId::new(format!("bce_b{}_d{}", batch_size, output_dim), batch_size),
                batch_size,
                |b, _| {
                    b.iter(|| {
                        Loss::compute(
                            LossType::BinaryCrossEntropy,
                            black_box(&predictions),
                            black_box(&targets)
                        )
                    });
                }
            );
        }
    }

    group.finish();
}

/// Benchmark Tensor operations
fn bench_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");

    // Matrix multiplication
    for size in [32, 64, 128, 256].iter() {
        let a_data: Vec<f32> = (0..*size * *size).map(|i| (i as f32 * 0.001).sin()).collect();
        let b_data: Vec<f32> = (0..*size * *size).map(|i| (i as f32 * 0.001).cos()).collect();

        let a = Tensor::new(a_data, vec![*size, *size]).unwrap();
        let b = Tensor::new(b_data, vec![*size, *size]).unwrap();

        group.throughput(Throughput::Elements((*size * *size) as u64));
        group.bench_with_input(
            BenchmarkId::new("matmul", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    a.matmul(black_box(&b))
                });
            }
        );
    }

    // L2 norm computation
    for size in [128, 512, 1024, 4096].iter() {
        let data: Vec<f32> = (0..*size).map(|i| (i as f32 * 0.001).sin()).collect();
        let tensor = Tensor::from_vec(data);

        group.bench_with_input(
            BenchmarkId::new("l2_norm", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(&tensor).l2_norm()
                });
            }
        );
    }

    // Normalization
    for size in [128, 512, 1024].iter() {
        let data: Vec<f32> = (0..*size).map(|i| (i as f32 * 0.001).sin() + 1.0).collect();
        let tensor = Tensor::from_vec(data);

        group.bench_with_input(
            BenchmarkId::new("normalize", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    black_box(&tensor).normalize()
                });
            }
        );
    }

    // Activation functions
    let data: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01) - 5.0).collect();
    let tensor = Tensor::from_vec(data);

    group.bench_function("relu_1024", |b| {
        b.iter(|| black_box(&tensor).relu())
    });

    group.bench_function("sigmoid_1024", |b| {
        b.iter(|| black_box(&tensor).sigmoid())
    });

    group.bench_function("tanh_1024", |b| {
        b.iter(|| black_box(&tensor).tanh())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_differentiable_search,
    bench_ruvector_layer_forward,
    bench_hierarchical_forward,
    bench_info_nce_loss,
    bench_local_contrastive_loss,
    bench_sgd_step,
    bench_adam_optimizer,
    bench_ewc_penalty,
    bench_loss_computation,
    bench_tensor_operations,
);

criterion_main!(benches);
