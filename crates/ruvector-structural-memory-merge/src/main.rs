//! Benchmark: does structural-time + coherence conflict resolution beat
//! wall-clock and vector-clock last-write-wins on concurrent multi-agent
//! memory writes?
//!
//! Formal hypothesis (fixed before this binary was run; see
//! `docs/research/nightly/2026-08-14-structural-time-memory-merge/README.md`):
//!
//! Given 2000 genuinely concurrent (causally-unordered) synthetic memory-write
//! conflicts across 6 agents with a hidden ground-truth quality label,
//!
//! when `StructuralCoherenceMerge` resolves each conflict instead of
//! `LwwWallClock` (400ms per-agent clock skew) or `LwwVectorClock`
//! (agent-id tiebreak),
//!
//! then its correct-resolution rate (picks the higher hidden-quality write)
//! should exceed both baselines by >= 10 percentage points,
//!
//! subject to: zero causal-order violations on a 500-case causally-ordered
//! control set (all three policies), and throughput within 5x of the
//! wall-clock baseline.

use ruvector_structural_memory_merge::scenario::{generate, ConflictCase, ScenarioConfig};
use ruvector_structural_memory_merge::{
    LwwVectorClock, LwwWallClock, MergePolicy, StructuralCoherenceMerge, Winner,
};
use std::time::Instant;

struct PolicyResult {
    name: &'static str,
    correct_rate: f64,
    mean_quality_regret: f64,
    causal_violations: usize,
    causal_total: usize,
    merges_per_sec: f64,
}

fn eval_policy(
    policy: &dyn MergePolicy,
    conflicts: &[ConflictCase],
    controls: &[ConflictCase],
) -> PolicyResult {
    let mut correct = 0usize;
    let mut regret_sum = 0.0f64;
    for c in conflicts {
        let d = policy.resolve(&c.a, &c.b, &c.context);
        let (winner_q, loser_q) = match d.winner {
            Winner::A => (c.quality_a, c.quality_b),
            Winner::B => (c.quality_b, c.quality_a),
        };
        if winner_q >= loser_q {
            correct += 1;
        }
        regret_sum += (loser_q - winner_q).max(0.0);
    }
    let correct_rate = correct as f64 / conflicts.len().max(1) as f64;
    let mean_quality_regret = regret_sum / conflicts.len().max(1) as f64;

    let mut causal_violations = 0usize;
    for c in controls {
        let d = policy.resolve(&c.a, &c.b, &c.context);
        // Control pairs are constructed so `a` always causally precedes `b`;
        // the only correct winner is `b`. Anything else is a causal-order
        // violation, independent of quality.
        if d.winner != Winner::B {
            causal_violations += 1;
        }
    }

    let warmup = conflicts.iter().chain(controls.iter()).take(200);
    for c in warmup {
        std::hint::black_box(policy.resolve(&c.a, &c.b, &c.context));
    }
    let reps = 20usize;
    let start = Instant::now();
    for _ in 0..reps {
        for c in conflicts.iter().chain(controls.iter()) {
            std::hint::black_box(policy.resolve(&c.a, &c.b, &c.context));
        }
    }
    let elapsed = start.elapsed();
    let total_ops = reps * (conflicts.len() + controls.len());
    let merges_per_sec = total_ops as f64 / elapsed.as_secs_f64();

    PolicyResult {
        name: policy.name(),
        correct_rate,
        mean_quality_regret,
        causal_violations,
        causal_total: controls.len(),
        merges_per_sec,
    }
}

fn run_at_skew(skew_ms: f64) -> Vec<PolicyResult> {
    let cfg = ScenarioConfig {
        num_conflicts: 2000,
        num_causal_controls: 500,
        dim: 16,
        num_agents: 6,
        wall_skew_ms: skew_ms,
        jitter_ms: 30.0,
        seed: 0xA6E17,
    };
    let cases = generate(&cfg);
    let (controls, conflicts): (Vec<_>, Vec<_>) = cases.into_iter().partition(|c| !c.is_concurrent);

    let policies: Vec<Box<dyn MergePolicy>> = vec![
        Box::new(LwwWallClock),
        Box::new(LwwVectorClock),
        Box::new(StructuralCoherenceMerge::default()),
    ];
    policies
        .iter()
        .map(|p| eval_policy(p.as_ref(), &conflicts, &controls))
        .collect()
}

fn print_table(title: &str, results: &[PolicyResult]) {
    println!("\n### {title}\n");
    println!("| Policy | Correct-resolution rate | Mean quality regret | Causal violations | Merges/sec |");
    println!("|---|---|---|---|---|");
    for r in results {
        println!(
            "| {} | {:.1}% | {:.4} | {}/{} | {:.0} |",
            r.name,
            r.correct_rate * 100.0,
            r.mean_quality_regret,
            r.causal_violations,
            r.causal_total,
            r.merges_per_sec
        );
    }
}

fn main() {
    println!("# Structural-Time Memory Merge — benchmark\n");
    println!(
        "Rust {}, release build, deterministic seed 0xA6E17.",
        env!("CARGO_PKG_VERSION")
    );

    let zero_skew = run_at_skew(0.0);
    print_table("Skew = 0ms (best case for wall-clock LWW)", &zero_skew);

    let realistic_skew = run_at_skew(400.0);
    print_table(
        "Skew = 400ms per-agent bias (realistic, unsynchronized edge agents)",
        &realistic_skew,
    );

    let heavy_skew = run_at_skew(2000.0);
    print_table("Skew = 2000ms per-agent bias (severe drift)", &heavy_skew);

    // --- Acceptance check against the pre-registered hypothesis -----------
    let wall = &realistic_skew[0];
    let vclock = &realistic_skew[1];
    let structural = &realistic_skew[2];

    let beats_wall = structural.correct_rate - wall.correct_rate >= 0.10;
    let beats_vclock = structural.correct_rate - vclock.correct_rate >= 0.10;
    let no_causal_violations = structural.causal_violations == 0 && vclock.causal_violations == 0;
    let throughput_ok = structural.merges_per_sec >= wall.merges_per_sec / 5.0;

    println!("\n### Acceptance check (realistic skew = 400ms)\n");
    println!(
        "- beats LwwWallClock by >=10pp: {} ({:+.1}pp)",
        beats_wall,
        (structural.correct_rate - wall.correct_rate) * 100.0
    );
    println!(
        "- beats LwwVectorClock by >=10pp: {} ({:+.1}pp)",
        beats_vclock,
        (structural.correct_rate - vclock.correct_rate) * 100.0
    );
    println!(
        "- zero causal-order violations (vclock & structural): {} (vclock={}, structural={})",
        no_causal_violations, vclock.causal_violations, structural.causal_violations
    );
    println!(
        "- throughput within 5x of wall-clock baseline: {} ({:.0} vs {:.0} merges/sec)",
        throughput_ok, structural.merges_per_sec, wall.merges_per_sec
    );

    let verdict = if beats_wall && beats_vclock && no_causal_violations && throughput_ok {
        "ACCEPT"
    } else if !no_causal_violations || !beats_wall || !beats_vclock {
        "REJECT"
    } else {
        "INCONCLUSIVE"
    };
    println!("\n### VERDICT: {verdict}\n");
}
