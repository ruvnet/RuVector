use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_consciousness::phi::{auto_compute_phi, ExactPhiEngine, SpectralPhiEngine};
use ruvector_consciousness::traits::PhiEngine;
use ruvector_consciousness::types::{ComputeBudget, TransitionMatrix};

fn make_tpm(n: usize) -> TransitionMatrix {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            let val: f64 = rng.random();
            data[i * n + j] = val;
            row_sum += val;
        }
        for j in 0..n {
            data[i * n + j] /= row_sum;
        }
    }
    TransitionMatrix::new(n, data)
}

fn bench_phi_exact_4(c: &mut Criterion) {
    let tpm = make_tpm(4);
    let budget = ComputeBudget::exact();
    c.bench_function("phi_exact_4_states", |b| {
        b.iter(|| ExactPhiEngine.compute_phi(black_box(&tpm), Some(0), &budget))
    });
}

fn bench_phi_exact_8(c: &mut Criterion) {
    let tpm = make_tpm(8);
    let budget = ComputeBudget::exact();
    c.bench_function("phi_exact_8_states", |b| {
        b.iter(|| ExactPhiEngine.compute_phi(black_box(&tpm), Some(0), &budget))
    });
}

fn bench_phi_spectral_16(c: &mut Criterion) {
    let tpm = make_tpm(16);
    let budget = ComputeBudget::fast();
    c.bench_function("phi_spectral_16_states", |b| {
        b.iter(|| {
            SpectralPhiEngine::default().compute_phi(black_box(&tpm), Some(0), &budget)
        })
    });
}

fn bench_phi_auto_4(c: &mut Criterion) {
    let tpm = make_tpm(4);
    let budget = ComputeBudget::exact();
    c.bench_function("phi_auto_4_states", |b| {
        b.iter(|| auto_compute_phi(black_box(&tpm), Some(0), &budget))
    });
}

criterion_group!(benches, bench_phi_exact_4, bench_phi_exact_8, bench_phi_spectral_16, bench_phi_auto_4);
criterion_main!(benches);
