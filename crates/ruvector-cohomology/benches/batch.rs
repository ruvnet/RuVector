//! Batch syndrome benchmark: assembly + full solve at increasing scale.

mod common;

use common::{build_ring_engine, time_ms};
use ruvector_cohomology::dynamic::DynamicCohomology;

fn main() {
    for &(n, dim, chords) in &[(100u64, 4usize, 50u64), (1_000, 4, 500), (5_000, 8, 2_500)] {
        let mut engine = build_ring_engine(n, dim, chords, 42);
        let (result, ms) = time_ms(|| engine.flush().expect("flush"));
        println!(
            "batch: n={n} dim={dim} edges={} energy={:.3e} flush={:.2}ms",
            n + chords,
            result.energy,
            ms
        );
    }
}
