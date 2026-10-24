//! Shared bench helpers: deterministic graph generators and timing.
#![allow(dead_code)]

use ruvector_cohomology::dynamic::{CohomologyEdit, DynamicCohomology, DynamicSheafEngine, EngineConfig};
use ruvector_cohomology::operator::RestrictionMap;
use std::time::Instant;

/// Deterministic pseudo-random stream for reproducible benchmarks.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Build a ring of `n` vertices with stalk dimension `dim`, identity maps,
/// coherent observations, plus `extra_chords` random chords.
pub fn build_ring_engine(n: u64, dim: usize, extra_chords: u64, seed: u64) -> DynamicSheafEngine {
    let mut engine = DynamicSheafEngine::new(EngineConfig::default());
    let mut rng = Rng(seed);
    for i in 0..n {
        assert!(engine
            .apply_edit(CohomologyEdit::InsertVertex { id: i, dim })
            .accepted);
    }
    let add_edge = |engine: &mut DynamicSheafEngine, a: u64, b: u64| {
        let receipt = engine.apply_edit(CohomologyEdit::InsertEdge {
            a,
            b,
            rho_a: RestrictionMap::Identity { dim },
            rho_b: RestrictionMap::Identity { dim },
            weight: 1.0,
            observation: vec![0.0; dim],
        });
        receipt.accepted
    };
    for i in 0..n {
        add_edge(&mut engine, i, (i + 1) % n);
    }
    let mut added = 0;
    while added < extra_chords {
        let a = rng.next_u64() % n;
        let b = rng.next_u64() % n;
        if a != b && add_edge(&mut engine, a, b) {
            added += 1;
        }
    }
    engine
}

/// Time a closure, returning (result, elapsed ms).
pub fn time_ms<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let out = f();
    (out, start.elapsed().as_secs_f64() * 1e3)
}

/// Report a labeled percentile summary of per-op latencies (ms).
pub fn report(label: &str, mut samples: Vec<f64>) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| samples[((samples.len() - 1) as f64 * p) as usize];
    println!(
        "{label}: n={} median={:.4}ms p99={:.4}ms max={:.4}ms",
        samples.len(),
        pct(0.5),
        pct(0.99),
        samples[samples.len() - 1]
    );
}
