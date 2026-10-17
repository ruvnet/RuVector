//! Planted-contradiction localization benchmark: precision/recall of the
//! ranked syndrome support against known planted inconsistent edges.

mod common;

use common::{build_ring_engine, Rng};
use ruvector_cohomology::dynamic::{CohomologyEdit, DynamicCohomology};
use ruvector_cohomology::EdgeKey;

fn main() {
    let n = 500u64;
    let dim = 4usize;
    let planted = 5usize;
    // Localization precision depends on cycle redundancy: on a bare ring the
    // syndrome is a constant harmonic 1-form (no localization is possible);
    // with ~2 chords per vertex the residual concentrates on planted edges.
    let mut engine = build_ring_engine(n, dim, 1000, 11);
    let mut rng = Rng(23);

    // Plant contradictions on ring edges (guaranteed to lie on cycles).
    let mut planted_edges = Vec::new();
    while planted_edges.len() < planted {
        let a = rng.next_u64() % n;
        let b = (a + 1) % n;
        let key = EdgeKey::new(a, b).unwrap();
        if planted_edges.contains(&key) {
            continue;
        }
        let value: Vec<f64> = (0..dim).map(|_| 1.0 + rng.unit()).collect();
        assert!(engine
            .apply_edit(CohomologyEdit::UpdateObservation { a, b, value })
            .accepted);
        planted_edges.push(key);
    }

    let result = engine.flush().expect("flush");
    let predicted: Vec<EdgeKey> = result
        .ranked_support
        .iter()
        .take(planted)
        .map(|c| c.edge)
        .collect();
    let hits = predicted.iter().filter(|e| planted_edges.contains(e)).count();
    let precision = hits as f64 / predicted.len() as f64;
    let recall = hits as f64 / planted_edges.len() as f64;
    println!(
        "planted_cycles: n={n} planted={planted} energy={:.3e} precision={precision:.2} recall={recall:.2}",
        result.energy
    );
}
