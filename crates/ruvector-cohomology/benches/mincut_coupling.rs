//! Combined syndrome → tension → quarantine → repair benchmark.

mod common;

use common::{build_ring_engine, time_ms};
use ruvector_cohomology::dynamic::{CohomologyEdit, DynamicCohomology};
use ruvector_cohomology::mincut_bridge::{run_containment, ContainmentPolicy, TensionWeights};
use ruvector_cohomology::repair::RepairPolicy;

fn main() {
    for policy in [ContainmentPolicy::IsolateFirst, ContainmentPolicy::RepairFirst] {
        let mut engine = build_ring_engine(200, 4, 100, 3);
        assert!(engine
            .apply_edit(CohomologyEdit::UpdateObservation {
                a: 10,
                b: 11,
                value: vec![3.0; 4],
            })
            .accepted);
        let (receipt, ms) = time_ms(|| {
            run_containment(
                &mut engine,
                &TensionWeights::default(),
                &RepairPolicy { lambda: 0.05, ..RepairPolicy::default() },
                policy,
                0.5,
                1e-9,
            )
            .expect("containment")
        });
        println!(
            "containment {:?}: pre={:.3e} post={:.3e} cut_edges={} isolated={} verified={} {:.2}ms",
            policy,
            receipt.pre_energy,
            receipt.post_energy,
            receipt.boundary.cut_edges.len(),
            receipt.boundary.isolated_vertices.len(),
            receipt.verified,
            ms
        );
    }
}
