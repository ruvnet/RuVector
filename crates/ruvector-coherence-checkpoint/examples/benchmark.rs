//! Baseline vs. drift-triggered vs. capped-drift-triggered checkpoint
//! scheduling, benchmarked on a deterministic bursty-drift workload.
//!
//! Usage: cargo run --release -p ruvector-coherence-checkpoint --example benchmark \
//!          -- [n_events] [dims] [seed] [drift_threshold]

use ruvector_coherence_checkpoint::{
    generate_workload, run_checkpoint_policy, verify_exact_replay, CheckpointRun, DriftTriggered,
    DriftTriggeredCapped, FixedInterval, WorkloadConfig,
};
use ruvector_proof_gate::HashChainGate;
use std::time::{Duration, Instant};

fn snapshot_storage_bytes(run: &CheckpointRun) -> usize {
    run.snapshots
        .iter()
        .map(|s| s.entries.len() * s.entries.first().map(|e| e.len()).unwrap_or(0) * 4)
        .sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_events: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6000);
    let dims: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(48);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2026);
    let threshold: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.08);

    let cfg = WorkloadConfig {
        dims,
        n_events,
        calm_phase_len: 250,
        burst_phase_len: 50,
        noise: 0.04,
    };
    let events = generate_workload(&cfg, seed);

    println!("=== ruvector-coherence-checkpoint benchmark ===");
    println!(
        "events={n_events} dims={dims} seed={seed} drift_threshold={threshold} \
         calm_phase_len={} burst_phase_len={} noise={}",
        cfg.calm_phase_len, cfg.burst_phase_len, cfg.noise
    );
    println!();

    // Candidate A runs first: its emergent snapshot count sets the storage
    // budget every other variant is matched against for a fair comparison.
    let t0 = Instant::now();
    let (run_a, gate_a) = run_checkpoint_policy(&events, dims, DriftTriggered { threshold });
    let elapsed_a = t0.elapsed();
    let budget = run_a.snapshots.len();

    let interval = (n_events / budget.max(1)).max(1);
    let t0 = Instant::now();
    let (run_baseline, gate_baseline) =
        run_checkpoint_policy(&events, dims, FixedInterval { interval });
    let elapsed_baseline = t0.elapsed();

    let max_interval = cfg.calm_phase_len * 2;
    let t0 = Instant::now();
    let (run_b, gate_b) = run_checkpoint_policy(
        &events,
        dims,
        DriftTriggeredCapped {
            threshold,
            max_interval,
        },
    );
    let elapsed_b = t0.elapsed();

    let runs: [(&str, &CheckpointRun, &HashChainGate, Duration); 3] = [
        (
            "baseline (FixedInterval)",
            &run_baseline,
            &gate_baseline,
            elapsed_baseline,
        ),
        ("candidate_A (DriftTriggered)", &run_a, &gate_a, elapsed_a),
        (
            "candidate_B (DriftTriggeredCapped)",
            &run_b,
            &gate_b,
            elapsed_b,
        ),
    ];

    println!(
        "{:<36} {:>10} {:>9} {:>9} {:>9} {:>13} {:>10}",
        "variant", "snapshots", "max_gap", "mean_gap", "p95_gap", "storage_KB", "time_ms"
    );
    for (name, run, _gate, elapsed) in &runs {
        println!(
            "{:<36} {:>10} {:>9} {:>9.1} {:>9} {:>13.1} {:>10.3}",
            name,
            run.snapshots.len(),
            run.max_gap(),
            run.mean_gap(),
            run.p95_gap(),
            snapshot_storage_bytes(run) as f64 / 1024.0,
            elapsed.as_secs_f64() * 1000.0,
        );
    }
    println!(
        "(baseline FixedInterval was tuned to interval={interval} to match candidate_A's \
         emergent snapshot budget of {budget}; candidate_B's max_interval={max_interval})"
    );
    println!();

    println!("=== Correctness: exact-replay + witness verification ===");
    let sample_step = (n_events / 40).max(1);
    let mut all_correct = true;
    let mut all_gate_ok = true;
    for (name, run, gate, _e) in &runs {
        let mut ok = 0usize;
        let mut total = 0usize;
        for target in (0..n_events).step_by(sample_step) {
            total += 1;
            if verify_exact_replay(run, &events, target) {
                ok += 1;
            }
        }
        let structural_ok = run.all_receipts_structurally_consistent(gate);
        println!(
            "{name}: exact_replay={ok}/{total}  chain_rederivation_ok={}  \
             receipt_structural_ok={structural_ok}",
            run.gate_integrity_ok
        );
        all_correct &= ok == total;
        all_gate_ok &= run.gate_integrity_ok && structural_ok;
    }
    println!();

    // Acceptance threshold fixed before this benchmark ran (nightly README /
    // ADR-305): at equal (±1) snapshot budget, candidate_A's max replay gap
    // must be at least 20% lower than baseline's, with 100% exact-replay
    // correctness and 100% witness-chain integrity across every variant.
    let budget_diff = (run_baseline.snapshots.len() as i64 - run_a.snapshots.len() as i64).abs();
    let gap_improvement = 1.0 - (run_a.max_gap() as f64 / run_baseline.max_gap().max(1) as f64);

    println!("=== Acceptance ===");
    println!("snapshot budget diff (baseline vs candidate_A): {budget_diff}");
    println!(
        "candidate_A max_gap: {}  baseline max_gap: {}  reduction: {:.1}%",
        run_a.max_gap(),
        run_baseline.max_gap(),
        gap_improvement * 100.0
    );
    println!("all variants exact-replay correct: {all_correct}");
    println!("all variants witness-chain valid: {all_gate_ok}");

    let accept = budget_diff <= 1 && gap_improvement >= 0.20 && all_correct && all_gate_ok;
    println!(
        "ACCEPTANCE_RESULT: {}",
        if accept { "ACCEPT" } else { "REJECT" }
    );
}
