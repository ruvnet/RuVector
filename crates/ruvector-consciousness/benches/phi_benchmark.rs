use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ruvector_consciousness::collapse::QuantumCollapseEngine;
use ruvector_consciousness::emergence::CausalEmergenceEngine;
use ruvector_consciousness::geomip::GeoMipPhiEngine;
use ruvector_consciousness::phi::{
    auto_compute_phi, ExactPhiEngine, GreedyBisectionPhiEngine, HierarchicalPhiEngine,
    SpectralPhiEngine, StochasticPhiEngine,
};
use ruvector_consciousness::rsvd_emergence::RsvdEmergenceEngine;
use ruvector_consciousness::traits::{ConsciousnessCollapse, EmergenceEngine, PhiEngine};
use ruvector_consciousness::types::{ComputeBudget, TransitionMatrix};

fn make_tpm(n: usize) -> TransitionMatrix {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            let val: f64 = rng.gen();
            data[i * n + j] = val;
            row_sum += val;
        }
        for j in 0..n {
            data[i * n + j] /= row_sum;
        }
    }
    TransitionMatrix::new(n, data)
}

// --- Exact ---

fn bench_phi_exact_4(c: &mut Criterion) {
    let tpm = make_tpm(4);
    let budget = ComputeBudget::exact();
    c.bench_function("phi_exact_n4", |b| {
        b.iter(|| ExactPhiEngine.compute_phi(black_box(&tpm), Some(0), &budget))
    });
}

fn bench_phi_exact_8(c: &mut Criterion) {
    let tpm = make_tpm(8);
    let budget = ComputeBudget::exact();
    c.bench_function("phi_exact_n8", |b| {
        b.iter(|| ExactPhiEngine.compute_phi(black_box(&tpm), Some(0), &budget))
    });
}

// --- GeoMIP ---

fn bench_phi_geomip_4(c: &mut Criterion) {
    let tpm = make_tpm(4);
    let budget = ComputeBudget::exact();
    c.bench_function("phi_geomip_n4", |b| {
        b.iter(|| GeoMipPhiEngine::default().compute_phi(black_box(&tpm), Some(0), &budget))
    });
}

fn bench_phi_geomip_8(c: &mut Criterion) {
    let tpm = make_tpm(8);
    let budget = ComputeBudget::exact();
    c.bench_function("phi_geomip_n8", |b| {
        b.iter(|| GeoMipPhiEngine::default().compute_phi(black_box(&tpm), Some(0), &budget))
    });
}

// --- Spectral ---

fn bench_phi_spectral_16(c: &mut Criterion) {
    let tpm = make_tpm(16);
    let budget = ComputeBudget::fast();
    c.bench_function("phi_spectral_n16", |b| {
        b.iter(|| SpectralPhiEngine::default().compute_phi(black_box(&tpm), Some(0), &budget))
    });
}

// --- Greedy Bisection ---

fn bench_phi_greedy_8(c: &mut Criterion) {
    let tpm = make_tpm(8);
    let budget = ComputeBudget::fast();
    c.bench_function("phi_greedy_n8", |b| {
        b.iter(|| {
            GreedyBisectionPhiEngine::default().compute_phi(black_box(&tpm), Some(0), &budget)
        })
    });
}

// --- Stochastic ---

fn bench_phi_stochastic_16(c: &mut Criterion) {
    let tpm = make_tpm(16);
    let budget = ComputeBudget::fast();
    c.bench_function("phi_stochastic_n16_1k", |b| {
        b.iter(|| StochasticPhiEngine::new(1000, 42).compute_phi(black_box(&tpm), Some(0), &budget))
    });
}

// --- Hierarchical ---

fn bench_phi_hierarchical_16(c: &mut Criterion) {
    let tpm = make_tpm(16);
    let budget = ComputeBudget::fast();
    c.bench_function("phi_hierarchical_n16", |b| {
        b.iter(|| HierarchicalPhiEngine::new(8).compute_phi(black_box(&tpm), Some(0), &budget))
    });
}

// --- Collapse ---

fn bench_phi_collapse_8(c: &mut Criterion) {
    let tpm = make_tpm(8);
    c.bench_function("phi_collapse_n8_reg128", |b| {
        b.iter(|| QuantumCollapseEngine::new(128).collapse_to_mip(black_box(&tpm), 10, 42))
    });
}

// --- Emergence ---

fn bench_emergence_8(c: &mut Criterion) {
    let tpm = make_tpm(8);
    let budget = ComputeBudget::fast();
    c.bench_function("emergence_n8", |b| {
        b.iter(|| CausalEmergenceEngine::default().compute_emergence(black_box(&tpm), &budget))
    });
}

fn bench_rsvd_emergence_16(c: &mut Criterion) {
    let tpm = make_tpm(16);
    let budget = ComputeBudget::fast();
    c.bench_function("rsvd_emergence_n16_k5", |b| {
        b.iter(|| RsvdEmergenceEngine::new(5, 3, 42).compute(black_box(&tpm), &budget))
    });
}

// --- Auto ---

fn bench_phi_auto_4(c: &mut Criterion) {
    let tpm = make_tpm(4);
    let budget = ComputeBudget::exact();
    c.bench_function("phi_auto_n4", |b| {
        b.iter(|| auto_compute_phi(black_box(&tpm), Some(0), &budget))
    });
}

criterion_group!(
    benches,
    bench_phi_exact_4,
    bench_phi_exact_8,
    bench_phi_geomip_4,
    bench_phi_geomip_8,
    bench_phi_spectral_16,
    bench_phi_greedy_8,
    bench_phi_stochastic_16,
    bench_phi_hierarchical_16,
    bench_phi_collapse_8,
    bench_emergence_8,
    bench_rsvd_emergence_16,
    bench_phi_auto_4,
);
criterion_main!(benches);
