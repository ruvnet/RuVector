//! Group-sparse repair benchmark: latency and post-repair energy.

mod common;

use common::{build_ring_engine, time_ms};
use ruvector_cohomology::dynamic::{CohomologyEdit, DynamicCohomology};
use ruvector_cohomology::repair::RepairPolicy;

fn main() {
    for &(n, dim) in &[(50u64, 4usize), (200, 4), (500, 8)] {
        let mut engine = build_ring_engine(n, dim, n / 2, 5);
        // One planted contradiction.
        assert!(engine
            .apply_edit(CohomologyEdit::UpdateObservation {
                a: 0,
                b: 1,
                value: vec![2.0; dim],
            })
            .accepted);
        let policy = RepairPolicy { lambda: 0.05, ..RepairPolicy::default() };
        let (plan, ms) = time_ms(|| engine.propose_repair(policy).expect("repair"));
        println!(
            "repair: n={n} dim={dim} support={} pre={:.3e} post={:.3e} obj={:.3e} {:.2}ms",
            plan.corrections.len(),
            plan.pre_energy,
            plan.post_energy,
            plan.objective,
            ms
        );
    }
}
