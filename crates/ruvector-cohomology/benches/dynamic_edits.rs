//! Dynamic edit latency benchmark: median/p99 local edit and estimate cost.

mod common;

use common::{build_ring_engine, report, time_ms, Rng};
use ruvector_cohomology::dynamic::{CohomologyEdit, DynamicCohomology};

fn main() {
    let n = 10_000u64;
    let dim = 8usize;
    let mut engine = build_ring_engine(n, dim, 5_000, 7);
    engine.flush().expect("initial flush");

    let mut rng = Rng(99);
    let mut edit_samples = Vec::new();
    let mut estimate_samples = Vec::new();
    for _ in 0..10_000 {
        let a = rng.next_u64() % n;
        let b = (a + 1) % n;
        let value: Vec<f64> = (0..dim).map(|_| rng.unit() * 0.01).collect();
        let (receipt, ms) = time_ms(|| {
            engine.apply_edit(CohomologyEdit::UpdateObservation { a, b, value: value.clone() })
        });
        assert!(receipt.accepted);
        edit_samples.push(ms);
        let (_, ems) = time_ms(|| engine.estimate());
        estimate_samples.push(ems);
    }
    report("local edit (UpdateObservation)", edit_samples);
    report("estimate()", estimate_samples);

    let (result, ms) = time_ms(|| engine.flush().expect("flush"));
    println!("exact flush after 10k edits: {:.2}ms energy={:.3e}", ms, result.energy);
}
