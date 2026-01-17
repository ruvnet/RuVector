//! Benchmarks for the Neural Quantum Error Decoder
//!
//! Measures performance of:
//! - GNN encoding
//! - Mamba decoding
//! - Feature fusion
//! - Full decode pipeline

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array2;
use ruvector_neural_decoder::{
    gnn::{GNNConfig, GNNEncoder},
    mamba::{MambaConfig, MambaDecoder},
    graph::GraphBuilder,
    NeuralDecoder, DecoderConfig,
};

fn bench_gnn_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("GNN Encoding");

    for distance in [3, 5, 7, 9].iter() {
        let config = GNNConfig {
            input_dim: 5,
            embed_dim: 64,
            hidden_dim: 128,
            num_layers: 3,
            num_heads: 4,
            dropout: 0.0,
        };
        let encoder = GNNEncoder::new(config);

        let graph = GraphBuilder::from_surface_code(*distance)
            .build()
            .unwrap();

        group.bench_with_input(
            BenchmarkId::new("encode", format!("d={}", distance)),
            &distance,
            |b, _| {
                b.iter(|| {
                    black_box(encoder.encode(&graph).unwrap())
                })
            },
        );
    }

    group.finish();
}

fn bench_mamba_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mamba Decoding");

    for size in [9, 25, 49, 81].iter() {
        let config = MambaConfig {
            input_dim: 128,
            state_dim: 64,
            output_dim: *size,
        };
        let mut decoder = MambaDecoder::new(config);

        let embeddings = Array2::from_shape_fn((*size, 128), |(i, j)| {
            ((i + j) as f32) / 100.0
        });

        group.bench_with_input(
            BenchmarkId::new("decode", format!("n={}", size)),
            &size,
            |b, _| {
                b.iter(|| {
                    decoder.reset();
                    black_box(decoder.decode(&embeddings).unwrap())
                })
            },
        );
    }

    group.finish();
}

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full Pipeline");

    for distance in [3, 5, 7].iter() {
        let config = DecoderConfig {
            distance: *distance,
            embed_dim: 64,
            hidden_dim: 128,
            num_gnn_layers: 2,
            num_heads: 4,
            mamba_state_dim: 64,
            use_mincut_fusion: false,
            dropout: 0.0,
        };
        let mut decoder = NeuralDecoder::new(config);

        let syndrome = vec![false; distance * distance];

        group.bench_with_input(
            BenchmarkId::new("decode", format!("d={}", distance)),
            &distance,
            |b, _| {
                b.iter(|| {
                    decoder.reset();
                    black_box(decoder.decode(&syndrome).unwrap())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gnn_encoding,
    bench_mamba_decoding,
    bench_full_pipeline,
);
criterion_main!(benches);
