#![cfg(feature = "gpu-cuda")]
//! Criterion benchmark: GPU vs CPU SDPA batch throughput (ADR-154 §6.4).
//!
//! Compiled only under `--features gpu-cuda`. If `CudaBackend::new()`
//! fails at runtime (no CUDA runtime, no driver, wrong toolkit) we skip
//! the GPU arm — the CPU number still publishes.

use connectome_fly::analysis::gpu::{ComputeBackend, CpuBackend, CudaBackend, Dims};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn make_batch(batch: usize, kv_len: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let total_q = batch * d;
    let total_k = batch * kv_len * d;
    let mut q = Vec::with_capacity(total_q);
    let mut k = Vec::with_capacity(total_k);
    let mut v = Vec::with_capacity(total_k);
    // Deterministic fill — no RNG.
    for i in 0..total_q {
        q.push(((i as f32) * 0.013).sin());
    }
    for i in 0..total_k {
        k.push(((i as f32) * 0.007).cos());
        v.push(((i as f32) * 0.019).sin());
    }
    (q, k, v)
}

fn bench(c: &mut Criterion) {
    let batch = 10_000;
    let kv_len = 10;
    let d = 64;
    let dims = Dims {
        q_len: 1,
        kv_len,
        d,
        batch,
    };
    let (q, k, v) = make_batch(batch, kv_len, d);

    let mut group = c.benchmark_group("gpu_sdpa_10k");
    group.sample_size(10);

    let cpu = CpuBackend;
    group.bench_function("cpu", |b| {
        b.iter(|| black_box(cpu.sdpa_batch(&q, &k, &v, dims)));
    });

    match CudaBackend::new() {
        Ok(gpu) => {
            // Warm-up outside the measured loop.
            let _ = gpu.sdpa_batch(&q, &k, &v, dims);
            group.bench_function("cuda", |b| {
                b.iter(|| black_box(gpu.sdpa_batch(&q, &k, &v, dims)));
            });
        }
        Err(e) => {
            eprintln!("gpu_sdpa: CUDA backend unavailable: {e}");
            eprintln!("gpu_sdpa: publishing CPU number only; see GPU.md");
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
