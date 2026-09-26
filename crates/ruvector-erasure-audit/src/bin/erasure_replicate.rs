//! Confirmatory replication of the *unregistered* finding from the headline
//! run: that `EagerRepair` — not `Tombstone` — is the erasure strategy a
//! black-box adversary can detect.
//!
//! ```text
//! cargo run --release -p ruvector-erasure-audit --bin erasure-replicate
//! ```
//!
//! ## Why this binary exists
//!
//! The headline run's primary hypothesis ("tombstones leak") was not confirmed,
//! and the EagerRepair result was found by looking at the rest of the table —
//! which is exactly the situation where a number should not be trusted. The
//! confirmation hypothesis below was fixed *before* any of these three
//! replicate seeds was run, and each replicate uses a completely fresh corpus,
//! fresh targets, fresh decoys and fresh probe jitter.
//!
//! **Confirmation hypothesis (pre-registered for this binary):**
//! in all three independent replicates, `EagerRepair` at churn=0 has a
//! held-out distinguishing accuracy whose 95% Wilson lower bound exceeds 0.50,
//! while `LocalRebuild`'s does not.

use ruvector_erasure_audit::audit::ProbeConfig;
use ruvector_erasure_audit::harness::{run_leak_audit, ExperimentConfig};

/// Three independent seed offsets, fixed before this binary was first run.
const SEED_OFFSETS: [u64; 3] = [0x1111_1111, 0x2222_2222, 0x3333_3333];
const CHURN: usize = 0;

fn main() {
    println!("=== erasure-audit confirmatory replication ===");
    println!(
        "confirmation hypothesis: in ALL {} replicates, EagerRepair CI_low > 0.50 \
         and LocalRebuild CI_low <= 0.50",
        SEED_OFFSETS.len()
    );

    let probe = ProbeConfig::default();
    let mut eager_confirms = 0usize;
    let mut rebuild_clean = 0usize;
    let mut tombstone_leaks = 0usize;

    println!(
        "\n{:>10} {:<13} {:>18} {:>9} {:>17} {:>6}",
        "replicate", "mode", "feature", "hold_acc", "95% CI", "leak?"
    );
    for (i, &off) in SEED_OFFSETS.iter().enumerate() {
        let cfg = ExperimentConfig {
            seed_offset: off,
            ..Default::default()
        };
        let base = cfg.build_base();
        for mode in cfg.modes() {
            let r = run_leak_audit(&cfg, mode, CHURN, &base, &probe);
            println!(
                "{:>10} {:<13} {:>18} {:>9.4}  [{:.3}, {:.3}] {:>6}",
                i + 1,
                r.mode,
                r.feature,
                r.holdout_acc,
                r.ci.0,
                r.ci.1,
                if r.leaks() { "YES" } else { "no" }
            );
            match r.mode {
                "EagerRepair" if r.leaks() => eager_confirms += 1,
                "LocalRebuild" if !r.leaks() => rebuild_clean += 1,
                "Tombstone" if r.leaks() => tombstone_leaks += 1,
                _ => {}
            }
        }
    }

    let n = SEED_OFFSETS.len();
    println!("\n--- Replication verdict ---");
    println!("EagerRepair leaked in   {eager_confirms}/{n} replicates");
    println!("LocalRebuild clean in   {rebuild_clean}/{n} replicates");
    println!("Tombstone leaked in     {tombstone_leaks}/{n} replicates");
    let confirmed = eager_confirms == n && rebuild_clean == n;
    println!(
        "\nCONFIRMATION: {}",
        if confirmed {
            "CONFIRMED — the EagerRepair leak reproduces on independent seeds"
        } else {
            "NOT CONFIRMED — treat the headline EagerRepair number as exploratory only"
        }
    );
}
