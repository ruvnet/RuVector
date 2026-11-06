//! Times the attention kernels alone (no NAPI), dim 384, to separate kernel
//! cost from Node.js marshaling cost.
//!
//! `cargo run --release -p ruvector-attention --example kernel_timing`
use ruvector_attention::{
    attention::MultiHeadAttention, sparse::FlashAttention, traits::Attention,
};
use std::time::Instant;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn median_us<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    for _ in 0..iters / 10 + 1 {
        f();
    }
    let mut samples: Vec<f64> = (0..iters)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    samples.sort_by(|a, b| a.total_cmp(b));
    samples[samples.len() / 2]
}

fn main() {
    let dim = 384;
    let mut seed = 42u64;
    let query: Vec<f32> = (0..dim).map(|_| lcg(&mut seed)).collect();
    for &n in &[20usize, 200, 1000] {
        let keys: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| lcg(&mut seed)).collect())
            .collect();
        let values: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| lcg(&mut seed)).collect())
            .collect();
        let k: Vec<&[f32]> = keys.iter().map(|v| v.as_slice()).collect();
        let v: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
        let flash = FlashAttention::new(dim, 64);
        let iters = if n >= 1000 { 2000 } else { 10000 };
        let t = median_us(
            || {
                std::hint::black_box(flash.compute(&query, &k, &v).unwrap());
            },
            iters,
        );
        println!("flash keys={n:<5} median_us={t:.2}");
        if n == 20 {
            let mh = MultiHeadAttention::new(dim, 8);
            let t = median_us(
                || {
                    std::hint::black_box(mh.compute(&query, &k, &v).unwrap());
                },
                iters,
            );
            println!("multi_head(8) keys={n:<5} median_us={t:.2}");
        }
    }
}
