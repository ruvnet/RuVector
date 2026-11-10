//! Real-run benchmark for the verified-partition-import hypothesis.
//!
//! Measures, for `N in {500, 2000, 8000, 20000}`:
//!
//! - `baseline`: recompute the global minimum cut from scratch.
//! - `candidate_a`: verify Ed25519 signature, accept partition if valid.
//! - `candidate_b`: verify + local-optimality spot-check at several
//!   sampling fractions.
//!
//! Also runs two adversarial checks for real, at the largest practical
//! size: (1) tamper-in-transit detection rate for `candidate_a`/`candidate_b`
//! (must be 100%); (2) signed-but-suboptimal detection rate for
//! `candidate_b` at each sampled fraction `k` (reported as measured, no
//! assumed value).
//!
//! Run with:
//! `cargo run --release -p ruvector-partition-import --bin benchmark`

use ruvector_partition_import::candidates::{
    candidate_a, candidate_b, local_optimality_spot_check, ImportOutcome,
};
use ruvector_partition_import::certificate::{verify_against, Signer};
use ruvector_partition_import::global_min_cut;
use ruvector_partition_import::graph::{generate_graph, Partition, WeightedGraph};
use std::time::{Duration, Instant};

const SEED: u64 = 20261022;
const AVG_DEGREE: usize = 8;
const SIGNER_SEED: [u8; 32] = [0x42; 32];
const SPOT_CHECK_FRACTIONS: [f64; 3] = [0.10, 0.25, 0.50];

struct Stats {
    mean_us: f64,
    p50_us: f64,
    p95_us: f64,
}

fn stats(mut samples: Vec<Duration>) -> Stats {
    samples.sort();
    let n = samples.len();
    let to_us = |d: Duration| d.as_secs_f64() * 1_000_000.0;
    let mean_us = samples.iter().map(|d| to_us(*d)).sum::<f64>() / n as f64;
    let p50_us = to_us(samples[n * 50 / 100]);
    let p95_us = to_us(samples[(n * 95 / 100).min(n - 1)]);
    Stats {
        mean_us,
        p50_us,
        p95_us,
    }
}

fn time_trials<F: FnMut()>(trials: usize, mut f: F) -> Vec<Duration> {
    let mut out = Vec::with_capacity(trials);
    for _ in 0..trials {
        let start = Instant::now();
        f();
        out.push(start.elapsed());
    }
    out
}

/// Flip `count` distinct vertices' side, deterministically, until the
/// resulting partition's cut value is strictly worse than the original.
/// Returns the perturbed partition and its cut value.
fn perturb_worse(
    graph: &WeightedGraph,
    original: &Partition,
    original_cut: f64,
    count: usize,
    seed: u64,
) -> (Partition, f64, usize) {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let indices: Vec<usize> = (0..graph.n).collect();
    for attempt in 0..200u64 {
        let mut rng2 = StdRng::seed_from_u64(seed.wrapping_add(attempt));
        let flips: Vec<usize> = indices.choose_multiple(&mut rng2, count).copied().collect();
        let mut side = original.side.clone();
        for &v in &flips {
            side[v] = !side[v];
        }
        let candidate = Partition::new(side);
        let cut = candidate.cut_value(graph);
        if cut > original_cut {
            return (candidate, cut, flips.len());
        }
    }
    // Fallback: flip every vertex in `indices[..count]` even if we did not
    // confirm strict worsening within the attempt budget (rare for
    // reasonably sized graphs); the caller still records the real
    // resulting cut value.
    let mut side = original.side.clone();
    for &v in indices.iter().take(count) {
        side[v] = !side[v];
    }
    let candidate = Partition::new(side);
    let cut = candidate.cut_value(graph);
    (candidate, cut, count)
}

fn run_size(n: usize, baseline_trials: usize, fast_trials: usize) {
    println!("\n=== N = {n} (avg_degree={AVG_DEGREE}, seed={SEED}) ===");
    let graph = generate_graph(n, SEED, AVG_DEGREE);
    println!(
        "graph: n={} edges={} wire_bytes={}",
        graph.n,
        graph.edge_count(),
        graph.wire_size_bytes()
    );

    let gen_start = Instant::now();
    let mincut = global_min_cut(&graph);
    let gen_elapsed = gen_start.elapsed();
    println!(
        "ground-truth min cut: value={:.6} computed_in={:.3}ms",
        mincut.cut_value,
        gen_elapsed.as_secs_f64() * 1000.0
    );

    let signer = Signer::from_seed(SIGNER_SEED);
    let cert = signer.sign(&graph, &mincut.partition, mincut.cut_value);
    assert!(verify_against(
        &signer.verifying_key,
        &graph,
        &mincut.partition,
        &cert
    ));
    let payload_bytes_base = graph.wire_size_bytes();
    let payload_bytes_candidates =
        graph.wire_size_bytes() + mincut.partition.wire_size_bytes() + cert.wire_size_bytes();

    // --- baseline: recompute from scratch every trial ---
    let baseline_samples = time_trials(baseline_trials, || {
        let r = global_min_cut(&graph);
        assert!(
            (r.cut_value - mincut.cut_value).abs() < 1e-6,
            "baseline correctness regression"
        );
    });
    let baseline_stats = stats(baseline_samples);
    println!(
        "baseline      trials={baseline_trials:<5} mean={:>10.3}us p50={:>10.3}us p95={:>10.3}us payload_bytes={payload_bytes_base}",
        baseline_stats.mean_us, baseline_stats.p50_us, baseline_stats.p95_us
    );

    // --- candidate_a: verify + trust ---
    let a_samples = time_trials(fast_trials, || {
        let d = candidate_a(&graph, &mincut.partition, &cert, &signer.verifying_key);
        assert_eq!(d.outcome, ImportOutcome::AcceptedTrusted);
        assert!(
            (d.cut_value - mincut.cut_value).abs() < 1e-9,
            "candidate_a correctness regression"
        );
    });
    let a_stats = stats(a_samples);
    println!(
        "candidate_a    trials={fast_trials:<5} mean={:>10.3}us p50={:>10.3}us p95={:>10.3}us payload_bytes={payload_bytes_candidates} speedup_vs_baseline={:.2}x",
        a_stats.mean_us, a_stats.p50_us, a_stats.p95_us,
        baseline_stats.mean_us / a_stats.mean_us
    );

    // --- candidate_b at each spot-check fraction ---
    for &frac in &SPOT_CHECK_FRACTIONS {
        let b_samples = time_trials(fast_trials, || {
            let d = candidate_b(
                &graph,
                &mincut.partition,
                &cert,
                &signer.verifying_key,
                frac,
                7,
            );
            assert_eq!(d.outcome, ImportOutcome::AcceptedSpotChecked);
            assert!(
                (d.cut_value - mincut.cut_value).abs() < 1e-9,
                "candidate_b correctness regression"
            );
        });
        let b_stats = stats(b_samples);
        println!(
            "candidate_b(k={frac:.2}) trials={fast_trials:<5} mean={:>10.3}us p50={:>10.3}us p95={:>10.3}us payload_bytes={payload_bytes_candidates} speedup_vs_baseline={:.2}x",
            b_stats.mean_us, b_stats.p50_us, b_stats.p95_us,
            baseline_stats.mean_us / b_stats.mean_us
        );
    }

    // --- tamper-in-transit detection (single-byte-equivalent weight flip),
    // N trials. candidate_a and candidate_b share the exact same first
    // step (`verify_against`) before they diverge, so both candidates'
    // detection rate is measured by calling `verify_against` directly
    // rather than through the full candidate wrapper — which, on a
    // detected tamper, would otherwise trigger a full (and, at this
    // crate's N, expensive) baseline recomputation whose cost has no
    // bearing on whether detection itself succeeded.
    let tamper_trials = fast_trials.clamp(50, 500);
    let mut detected = 0usize;
    for i in 0..tamper_trials {
        let mut tampered_graph = graph.clone();
        let edge_idx = i % tampered_graph.edges.len();
        tampered_graph.edges[edge_idx].2 += 0.000001 + (i as f64) * 1e-9;

        let ok = verify_against(
            &signer.verifying_key,
            &tampered_graph,
            &mincut.partition,
            &cert,
        );
        if !ok {
            detected += 1;
        }
    }
    let a_detected = detected;
    let b_detected = detected;
    println!(
        "tamper-in-transit detection: candidate_a={}/{} ({:.1}%) candidate_b={}/{} ({:.1}%)",
        a_detected,
        tamper_trials,
        100.0 * a_detected as f64 / tamper_trials as f64,
        b_detected,
        tamper_trials,
        100.0 * b_detected as f64 / tamper_trials as f64
    );
    // Confirm, once, that the full candidate API actually resolves a
    // tampered input to RejectedSignatureInvalid end to end (this call's
    // own latency is not part of the detection-rate measurement above).
    {
        let mut tampered_graph = graph.clone();
        tampered_graph.edges[0].2 += 0.5;
        let a = candidate_a(
            &tampered_graph,
            &mincut.partition,
            &cert,
            &signer.verifying_key,
        );
        let b = candidate_b(
            &tampered_graph,
            &mincut.partition,
            &cert,
            &signer.verifying_key,
            0.1,
            7,
        );
        assert_eq!(a.outcome, ImportOutcome::RejectedSignatureInvalid);
        assert_eq!(b.outcome, ImportOutcome::RejectedSignatureInvalid);
    }

    // --- signed-but-suboptimal adversarial test ---
    // Perturb a small, fixed fraction of vertices (2%) to build a
    // deliberately-worse, honestly-signed partition P', then measure what
    // fraction of trials each candidate accepts it.
    let perturb_count = ((n as f64 * 0.02).round() as usize).clamp(1, n);
    let adversarial_trials = fast_trials.clamp(30, 200);
    let mut a_wrongly_accepted = 0usize;
    let mut b_caught: Vec<usize> = vec![0; SPOT_CHECK_FRACTIONS.len()];
    let mut worse_cuts_sum = 0.0f64;
    for trial in 0..adversarial_trials {
        let (bad_partition, bad_cut, flips) = perturb_worse(
            &graph,
            &mincut.partition,
            mincut.cut_value,
            perturb_count,
            1000 + trial as u64,
        );
        worse_cuts_sum += bad_cut;
        let bad_cert = signer.sign(&graph, &bad_partition, bad_cut);

        // candidate_a has no further check after a valid signature, so
        // this call is cheap (no fallback) and is run through the full
        // API for a faithful measurement.
        let a = candidate_a(&graph, &bad_partition, &bad_cert, &signer.verifying_key);
        if a.outcome == ImportOutcome::AcceptedTrusted {
            a_wrongly_accepted += 1;
        }

        // candidate_b's detection outcome is exactly
        // `local_optimality_spot_check`'s boolean result (after an
        // already-confirmed-valid signature); calling it directly avoids
        // paying for the full baseline-recompute fallback that the real
        // `candidate_b` would (correctly) trigger on every catch at this
        // crate's N, which would measure recompute cost, not detection.
        for (fi, &frac) in SPOT_CHECK_FRACTIONS.iter().enumerate() {
            let passed = local_optimality_spot_check(&graph, &bad_partition, frac, trial as u64);
            if !passed {
                b_caught[fi] += 1;
            }
        }
        let _ = flips;
    }

    // Confirm, once, that the full candidate_b API actually resolves a
    // caught signed-but-suboptimal partition to RejectedSpotCheckFailed
    // end to end (this call's own recompute-fallback latency is not part
    // of any measurement above).
    {
        let (bad_partition, bad_cut, _) = perturb_worse(
            &graph,
            &mincut.partition,
            mincut.cut_value,
            perturb_count,
            999,
        );
        let bad_cert = signer.sign(&graph, &bad_partition, bad_cut);
        if !local_optimality_spot_check(&graph, &bad_partition, 0.5, 999) {
            let b = candidate_b(
                &graph,
                &bad_partition,
                &bad_cert,
                &signer.verifying_key,
                0.5,
                999,
            );
            assert_eq!(b.outcome, ImportOutcome::RejectedSpotCheckFailed);
        }
    }

    println!(
        "signed-but-suboptimal adversarial test: perturbed_vertices={perturb_count}/{n} ({:.2}%) trials={adversarial_trials} mean_worse_cut={:.4} (honest cut={:.4})",
        100.0 * perturb_count as f64 / n as f64,
        worse_cuts_sum / adversarial_trials as f64,
        mincut.cut_value
    );
    println!(
        "  candidate_a wrongly accepted: {}/{} ({:.1}%)",
        a_wrongly_accepted,
        adversarial_trials,
        100.0 * a_wrongly_accepted as f64 / adversarial_trials as f64
    );
    for (fi, &frac) in SPOT_CHECK_FRACTIONS.iter().enumerate() {
        println!(
            "  candidate_b(k={:.2}) caught: {}/{} ({:.1}%)",
            frac,
            b_caught[fi],
            adversarial_trials,
            100.0 * b_caught[fi] as f64 / adversarial_trials as f64
        );
    }
}

fn main() {
    println!("ruvector-partition-import benchmark");
    println!(
        "SEED={SEED} AVG_DEGREE={AVG_DEGREE} SIGNER_SEED=0x{:02x}..",
        SIGNER_SEED[0]
    );

    // (N, baseline_trials, fast_trials) — a pre-flight scaling probe
    // measured one global_min_cut call at ~56ms (N=500), ~974ms (N=2000),
    // ~23.1s (N=8000), ~152.6s (N=20000) on this container. baseline_trials
    // is set from those real numbers to keep total wall time for this
    // binary in the tens-of-minutes range rather than silently reducing
    // precision without saying so; see the research report's Methodology
    // section for the exact reasoning.
    run_size(500, 200, 2000);
    run_size(2000, 50, 1000);
    run_size(8000, 5, 300);
    run_size(20000, 3, 100);

    println!("\nbenchmark complete.");
}
