//! Micro-benchmark: scalar reference cosine vs SimSIMD cosine over full-dim
//! rerank calls. This isolates the kernel that `EmlHnsw::search_with_rerank`
//! invokes once per candidate — 500 calls per dim, two dims (128 and 384),
//! matches a realistic `fetch_k`.
//!
//! Keeps Criterion optional: uses plain `std::time::Instant` so the numbers
//! show up in `cargo bench -- --nocapture` and in the commit body without
//! requiring gnuplot or html reports.

use ruvector_eml_hnsw::cosine_decomp::cosine_distance_f32;
use ruvector_eml_hnsw::selected_distance::cosine_distance_simd;
use std::time::Instant;

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn gen_vec(&mut self, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| self.next_f32()).collect()
    }
}

fn make_pairs(n: usize, dim: usize, seed: u64) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| (r.gen_vec(dim), r.gen_vec(dim))).collect()
}

fn bench_kernel<F: Fn(&[f32], &[f32]) -> f32>(
    label: &str,
    pairs: &[(Vec<f32>, Vec<f32>)],
    f: F,
) -> f64 {
    // Warmup: 3 passes so steady-state (caches + branch predictors) is hit.
    let mut sink = 0.0f32;
    for _ in 0..3 {
        for (a, b) in pairs {
            sink += f(a, b);
        }
    }
    std::hint::black_box(sink);

    // Measured: 5 passes, take the minimum — noise is strictly additive at
    // this granularity so min removes scheduler jitter.
    let mut best_ns = u128::MAX;
    for _ in 0..5 {
        let t0 = Instant::now();
        let mut s = 0.0f32;
        for (a, b) in pairs {
            s += f(a, b);
        }
        std::hint::black_box(s);
        let ns = t0.elapsed().as_nanos();
        if ns < best_ns {
            best_ns = ns;
        }
    }
    let per_call_ns = best_ns as f64 / pairs.len() as f64;
    println!(
        "{:<35} {:>6} calls   total {:>7} ns   per-call {:>7.1} ns",
        label,
        pairs.len(),
        best_ns,
        per_call_ns
    );
    per_call_ns
}

fn run_dim(dim: usize, n_calls: usize) {
    let pairs = make_pairs(n_calls, dim, 0xDEAD_BEEF ^ dim as u64);

    println!("\n== rerank kernel, dim={} ({} calls) ==", dim, n_calls);
    let scalar_ns = bench_kernel(
        &format!("cosine_distance_f32 (scalar) d{}", dim),
        &pairs,
        cosine_distance_f32,
    );
    let simd_ns = bench_kernel(
        &format!("cosine_distance_simd (SIMD)  d{}", dim),
        &pairs,
        cosine_distance_simd,
    );
    let speedup = scalar_ns / simd_ns.max(1e-6);
    println!("speedup: {:.2}x (dim={})", speedup, dim);
}

fn main() {
    // 500 calls per dim per the Tier 1B spec; matches a realistic fetch_k
    // worst case the parallel work pushes to.
    run_dim(128, 500);
    run_dim(384, 500);
}
