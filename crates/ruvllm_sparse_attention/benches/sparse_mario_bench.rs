// sparse_mario_bench — benchmark the retrieval workload used by
// `examples/sparse_mario.rs` against three attention paths:
//
//   1. dense_attention             — O(N²) baseline
//   2. sparse forward()            — O(N log N) with non-causal window+log-stride+landmarks
//   3. sparse forward_gated_with_fastgrnn() — near-linear with FastGRNN salience gate
//
// Tensor shape mirrors sparse-mario: heads=1, head_dim=64, non-causal,
// window=256, block=64. Sequence lengths 256/512/1024/2048 cover the
// realistic range of corpus_len + prefix_len in the example (2.1K–2.9K).
//
// Run with:
//   cargo bench --bench sparse_mario_bench --features parallel \
//       -- --warm-up-time 1 --measurement-time 3 --sample-size 20

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvllm_sparse_attention::{
    dense_attention, AttentionBackend, FastGrnnGate, SparseAttentionConfig,
    SubquadraticSparseAttention, Tensor3,
};

const HEAD_DIM: usize = 64;
const N_HEADS: usize = 1;
const SEQS: &[usize] = &[256, 512, 1024, 2048];

fn random_tensor(seq: usize, seed: u64) -> Tensor3 {
    let mut rng = StdRng::seed_from_u64(seed);
    let len = seq * N_HEADS * HEAD_DIM;
    let data: Vec<f32> = (0..len).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect();
    Tensor3::from_vec(data, seq, N_HEADS, HEAD_DIM).unwrap()
}

fn mario_config() -> SparseAttentionConfig {
    SparseAttentionConfig {
        window: 256,
        block_size: 64,
        global_tokens: vec![0],
        causal: false,
        use_log_stride: true,
        use_landmarks: true,
        sort_candidates: false,
    }
}

fn bench_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_mario/dense");
    for &seq in SEQS {
        let q = random_tensor(seq, 1);
        let k = random_tensor(seq, 2);
        let v = random_tensor(seq, 3);
        group.bench_function(format!("seq_{}", seq), |b| {
            b.iter(|| {
                // sparse-mario uses non-causal attention; dense_attention's
                // last arg is the causal flag.
                dense_attention(black_box(&q), black_box(&k), black_box(&v), false).unwrap()
            })
        });
    }
    group.finish();
}

fn bench_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_mario/sparse");
    let attention = SubquadraticSparseAttention::new(mario_config()).unwrap();
    for &seq in SEQS {
        let q = random_tensor(seq, 4);
        let k = random_tensor(seq, 5);
        let v = random_tensor(seq, 6);
        group.bench_function(format!("seq_{}", seq), |b| {
            b.iter(|| {
                attention
                    .forward(black_box(&q), black_box(&k), black_box(&v))
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_sparse_fastgrnn(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_mario/sparse_fastgrnn");
    let attention = SubquadraticSparseAttention::new(mario_config()).unwrap();
    let gate = FastGrnnGate::new(HEAD_DIM, 32);

    for &seq in SEQS {
        let q = random_tensor(seq, 7);
        let k = random_tensor(seq, 8);
        let v = random_tensor(seq, 9);
        // Keep top 25% of long-range candidates — FastGRNN drops the rest.
        let gate_top_k = (seq / 4).max(8);
        group.bench_function(format!("seq_{}", seq), |b| {
            b.iter(|| {
                attention
                    .forward_gated_with_fastgrnn(
                        black_box(&q),
                        black_box(&k),
                        black_box(&v),
                        &gate,
                        gate_top_k,
                    )
                    .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dense, bench_sparse, bench_sparse_fastgrnn);
criterion_main!(benches);
