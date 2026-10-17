//! Nightly research benchmark (2026-10-10, follow-up to 2026-09-16's
//! `witness-signer-agent-memory` run, Next Research item #1): measures the
//! real wall-clock cost of *assembling* a `SigningStrategy::BatchTail` span
//! from a live, possibly-stalling write stream — the gap that run named but
//! explicitly did not model (it measured only CPU signing cost against an
//! already-full batch; see `witness_signing_bench.rs`).
//!
//! Three variants, same methodology `ruvector-retrieval-receipt`'s
//! `bin/batch_latency.rs` used to bound anchor-batch latency (ADR-340's own
//! Next Research item #1) — the pattern is ported here, not reused as a
//! dependency, because it drives real `SignedWitnessSink::emit_batch` /
//! `check_timeout` calls (real Ed25519 signs over real witness-record
//! digests) rather than a generic `BatchScheduler` + external signer:
//!
//!   baseline    — `SigningStrategy::BatchTail { batch_size }` (today's
//!                 shipped strategy: closes only when the batch fills, no
//!                 wall-clock bound)
//!   candidate_a — `SigningStrategy::BatchTailTimeout { batch_size,
//!                 max_wait_ns }` (this run's addition: also closes once
//!                 `max_wait_ns` elapses since the oldest pending record)
//!   candidate_b — `SigningStrategy::PerRecord` (reference upper bound:
//!                 every record's signature is available immediately, at
//!                 the cost of one signature per record)
//!
//! Three arrival regimes, deterministic seeded synthetic timestamps (no
//! real sleeping — the simulation drives virtual arrival times through the
//! sink and only measures real wall-clock cost for the signing operation
//! itself, via `std::time::Instant`):
//!
//!   target_load   — fast enough that `batch_size` fills well inside
//!                   `max_wait_ns` (the timeout should almost never fire)
//!   light_load    — slow enough that `batch_size` cannot fill within
//!                   `max_wait_ns` (mean gap x batch_size >> max_wait_ns),
//!                   forcing baseline's pending span to sit open far longer
//!                   than candidate_a's
//!   bursty        — on/off traffic (agent tool-call bursts), not a smooth
//!                   Poisson rate
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example witness_signing_batch_fill_latency

use ruvector_agent_memory::{
    verify_signed_chain, EvidenceGrade, LedgerWitnessRecord, MemoryWitnessLog, SignedAnchor,
    SignedWitnessSink, SigningStrategy, WitnessSink,
};
use rvf_types::ed25519::Ed25519Keypair;
use std::time::Instant;

const SECRET: [u8; 32] = [21u8; 32];
const NS_PER_MS: u64 = 1_000_000;

/// Deterministic xorshift — same construction as
/// `ruvector-retrieval-receipt::bin::batch_latency`'s, so arrival timing is
/// reproducible without an external RNG dependency.
struct Xorshift64(u64);
impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_unit_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
}

#[derive(Clone, Copy)]
struct Arrival {
    sequence: u64,
    arrived_at_ns: u64,
}

fn poisson_arrivals(count: usize, rate_per_sec: f64, seed: u64) -> Vec<Arrival> {
    let mut rng = Xorshift64(seed);
    let mean_gap_ns = 1.0e9 / rate_per_sec;
    let mut t_ns = 0.0f64;
    (0..count)
        .map(|i| {
            let gap = -mean_gap_ns * rng.next_unit_open().ln();
            t_ns += gap;
            Arrival {
                sequence: i as u64,
                arrived_at_ns: t_ns as u64,
            }
        })
        .collect()
}

/// Bursty on/off Poisson process: alternates `on_ns` at `on_rate_per_sec`
/// with `off_ns` of silence. Models an agent's clustered tool-call bursts.
fn bursty_arrivals(
    count: usize,
    on_rate_per_sec: f64,
    on_ns: u64,
    off_ns: u64,
    seed: u64,
) -> Vec<Arrival> {
    let mut rng = Xorshift64(seed);
    let mean_gap_ns = 1.0e9 / on_rate_per_sec;
    let mut cycle_start_ns = 0u64;
    let mut t_ns = 0.0f64;
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let gap = -mean_gap_ns * rng.next_unit_open().ln();
        t_ns += gap;
        if t_ns - cycle_start_ns as f64 >= on_ns as f64 {
            cycle_start_ns += on_ns + off_ns;
            t_ns = cycle_start_ns as f64;
            continue;
        }
        out.push(Arrival {
            sequence: out.len() as u64,
            arrived_at_ns: t_ns as u64,
        });
    }
    out
}

fn minimal_record(sequence: u64, timestamp_ns: u64) -> LedgerWitnessRecord {
    // `verify_chain` cross-checks the evidence-grade nibble packed into
    // `flags` (bits 12-15, ADR-134 §3 / ADR-322C) against `evidence_grade`
    // itself; `flags: 0` would silently disagree with `Recomputed`'s code
    // and fail `verify_chain` regardless of signing correctness.
    let flags = (EvidenceGrade::Recomputed.code() as u16) << 12;
    LedgerWitnessRecord {
        sequence,
        timestamp_ns,
        action_kind: 0,
        proof_tier: 0,
        flags,
        actor_partition_id: 0,
        target_object_id: 0,
        capability_hash: 0,
        payload: sequence,
        prev_hash: 0,
        record_hash: 0,
        aux: 0,
        evidence_grade: EvidenceGrade::Recomputed,
    }
}

fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

struct RegimeStats {
    regime: &'static str,
    policy: &'static str,
    num_records: usize,
    num_spans: usize,
    mean_span_size: f64,
    latency_mean_ns: f64,
    latency_p50_ns: u64,
    latency_p95_ns: u64,
    latency_p99_ns: u64,
    latency_max_ns: u64,
    sign_amortized_ns: f64,
    chain_verified: bool,
}

/// Drive one (regime, strategy) combination through a real
/// `SignedWitnessSink`, timing every call that performs a real Ed25519
/// signature (span close, whether triggered by `batch_size`, `check_timeout`,
/// or end-of-stream `seal`). `arrived_at_ns + real_sign_elapsed_ns` is each
/// covered record's availability time, so both the virtual batch-fill wait
/// and the real cryptographic cost are represented in the latency reported.
fn run_policy(
    regime: &'static str,
    policy_name: &'static str,
    strategy: SigningStrategy,
    arrivals: &[Arrival],
) -> RegimeStats {
    let mut sink = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &Ed25519Keypair::from_secret(&SECRET),
        strategy,
    )
    .expect("valid strategy");

    let mut availability_ns = vec![0u64; arrivals.len()];
    let mut sign_total_ns = 0u128;
    let mut num_spans = 0usize;
    // FNV-1a hash-chain state (ADR-134 §3): a real deployment's ledger
    // fixes these up before `emit_batch` (see `ledger.rs::stage`'s
    // "fixed up during emit_and_apply chaining" comment); this simulation
    // bypasses the ledger for deterministic synthetic timestamps, so it
    // must replicate that chaining itself or `verify_chain` rejects every
    // record as a broken chain regardless of signing correctness.
    let mut prev_hash = 0u64;

    let mut record_new_spans = |sink: &SignedWitnessSink<MemoryWitnessLog>,
                                spans_before: usize,
                                close_at_ns: u64,
                                sign_elapsed_ns: u128,
                                availability_ns: &mut [u64]| {
        for span in &sink.spans()[spans_before..] {
            for seq in span.covers_from_seq..=span.covers_to_seq {
                availability_ns[seq as usize] = close_at_ns + sign_elapsed_ns as u64;
            }
            sign_total_ns += sign_elapsed_ns;
            num_spans += 1;
        }
    };

    let mut i = 0usize;
    while i < arrivals.len() {
        let deadline = if let SigningStrategy::BatchTailTimeout { max_wait_ns, .. } = strategy {
            sink.oldest_pending_arrival_ns().map(|t| t + max_wait_ns)
        } else {
            None
        };
        let next_arrival_ns = arrivals[i].arrived_at_ns;

        if let Some(d) = deadline {
            if d <= next_arrival_ns {
                let spans_before = sink.spans().len();
                let t0 = Instant::now();
                let sealed = sink.check_timeout(d);
                let elapsed = t0.elapsed().as_nanos();
                assert!(sealed, "deadline implies a pending span to seal");
                record_new_spans(&sink, spans_before, d, elapsed, &mut availability_ns);
                continue; // re-evaluate deadline vs. the same next arrival
            }
        }

        let a = arrivals[i];
        i += 1;
        let mut rec = minimal_record(a.sequence, a.arrived_at_ns);
        rec.prev_hash = prev_hash;
        rec.record_hash = rec.compute_record_hash();
        prev_hash = rec.chain_hash();
        let spans_before = sink.spans().len();
        let t0 = Instant::now();
        sink.emit_batch(&[rec]).expect("contiguous sequence");
        let elapsed = t0.elapsed().as_nanos();
        record_new_spans(
            &sink,
            spans_before,
            a.arrived_at_ns,
            elapsed,
            &mut availability_ns,
        );
    }

    // End of stream: seal whatever partial span remains. This is a
    // simulation-boundary artifact (the earliest a real deployment could
    // know no more records are coming in this window), not a claim about
    // production shutdown behavior.
    let close_at_ns = arrivals.last().map(|a| a.arrived_at_ns).unwrap_or(0);
    let spans_before = sink.spans().len();
    let t0 = Instant::now();
    sink.seal();
    let elapsed = t0.elapsed().as_nanos();
    record_new_spans(
        &sink,
        spans_before,
        close_at_ns,
        elapsed,
        &mut availability_ns,
    );

    let pk = sink.public_key();
    let chain_verified =
        verify_signed_chain(sink.inner(), sink.spans(), &pk, &SignedAnchor::genesis()).is_ok();

    let mut latencies: Vec<u64> = arrivals
        .iter()
        .map(|a| availability_ns[a.sequence as usize].saturating_sub(a.arrived_at_ns))
        .collect();
    latencies.sort_unstable();
    let latency_mean_ns = latencies.iter().map(|&l| l as f64).sum::<f64>() / latencies.len() as f64;

    RegimeStats {
        regime,
        policy: policy_name,
        num_records: arrivals.len(),
        num_spans,
        mean_span_size: arrivals.len() as f64 / num_spans.max(1) as f64,
        latency_mean_ns,
        latency_p50_ns: percentile(&latencies, 50.0),
        latency_p95_ns: percentile(&latencies, 95.0),
        latency_p99_ns: percentile(&latencies, 99.0),
        latency_max_ns: latencies.last().copied().unwrap_or(0),
        sign_amortized_ns: sign_total_ns as f64 / arrivals.len() as f64,
        chain_verified,
    }
}

fn fmt_ms(ns: f64) -> String {
    format!("{:.3}ms", ns / NS_PER_MS as f64)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_records: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4000);

    println!("=== ruvector-agent-memory witness-signing batch-fill latency simulation ===");
    println!("records_per_regime={num_records}\n");

    const BATCH_SIZE: usize = 32;
    const MAX_WAIT_NS: u64 = 50 * NS_PER_MS;

    let strategies: [(&str, SigningStrategy); 3] = [
        (
            "baseline_batchtail32",
            SigningStrategy::BatchTail {
                batch_size: BATCH_SIZE,
            },
        ),
        (
            "candidate_a_hybrid50ms",
            SigningStrategy::BatchTailTimeout {
                batch_size: BATCH_SIZE,
                max_wait_ns: MAX_WAIT_NS,
            },
        ),
        ("candidate_b_per_record", SigningStrategy::PerRecord),
    ];

    // Regime 1: fast enough that a 32-record batch fills far inside 50ms
    // (mean gap 0.5ms x 32 = 16ms). Regime 2: slow enough it cannot (mean
    // gap 20ms x 32 = 640ms >> 50ms), forcing candidate_a's timeout to
    // fire well before baseline would ever close on size alone. Regime 3:
    // bursty on/off traffic rather than a smooth rate.
    let regimes: Vec<(&'static str, Vec<Arrival>)> = vec![
        (
            "target_load_2000rps",
            poisson_arrivals(num_records, 2000.0, 0x1111_2222_3333_4444),
        ),
        (
            "light_load_50rps",
            poisson_arrivals(num_records, 50.0, 0x5555_6666_7777_8888),
        ),
        (
            "bursty_on1500rps_off400ms",
            bursty_arrivals(
                num_records,
                1500.0,
                100 * NS_PER_MS,
                400 * NS_PER_MS,
                0x9999_AAAA_BBBB_CCCC,
            ),
        ),
    ];

    let mut all_stats = Vec::new();
    for (regime_name, arrivals) in &regimes {
        for (policy_name, strategy) in &strategies {
            all_stats.push(run_policy(regime_name, policy_name, *strategy, arrivals));
        }
    }

    println!(
        "{:<28} {:<24} {:>8} {:>8} {:>9} {:>10} {:>10} {:>10} {:>10} {:>10} {:>14} {:>9}",
        "regime",
        "policy",
        "records",
        "spans",
        "mean_sz",
        "lat_mean",
        "lat_p50",
        "lat_p95",
        "lat_p99",
        "lat_max",
        "sign_amort_ns",
        "verified"
    );
    for s in &all_stats {
        println!(
            "{:<28} {:<24} {:>8} {:>8} {:>9.1} {:>10} {:>10} {:>10} {:>10} {:>10} {:>14.1} {:>9}",
            s.regime,
            s.policy,
            s.num_records,
            s.num_spans,
            s.mean_span_size,
            fmt_ms(s.latency_mean_ns),
            fmt_ms(s.latency_p50_ns as f64),
            fmt_ms(s.latency_p95_ns as f64),
            fmt_ms(s.latency_p99_ns as f64),
            fmt_ms(s.latency_max_ns as f64),
            s.sign_amortized_ns,
            s.chain_verified,
        );
    }

    // ── Acceptance evaluation (thresholds fixed before this run; mirrors
    // ruvector-retrieval-receipt::bin::batch_latency's acceptance shape) ──
    let find = |regime: &str, policy: &str| -> &RegimeStats {
        all_stats
            .iter()
            .find(|s| s.regime == regime && s.policy == policy)
            .expect("regime/policy combination was run")
    };

    let all_verified = all_stats.iter().all(|s| s.chain_verified);

    let hybrid_target = find("target_load_2000rps", "candidate_a_hybrid50ms");
    let baseline_target = find("target_load_2000rps", "baseline_batchtail32");
    let hybrid_light = find("light_load_50rps", "candidate_a_hybrid50ms");
    let baseline_light = find("light_load_50rps", "baseline_batchtail32");
    let hybrid_bursty = find("bursty_on1500rps_off400ms", "candidate_a_hybrid50ms");

    // Bound = 50ms fill-timeout + 20ms fixed slack for real sign cost
    // (microseconds, three orders of magnitude below the bound) and
    // simulation-boundary effects.
    let bound_ns = (MAX_WAIT_NS + 20 * NS_PER_MS) as f64;
    let hybrid_bounded = (hybrid_target.latency_p99_ns as f64) <= bound_ns
        && (hybrid_light.latency_p99_ns as f64) <= bound_ns
        && (hybrid_bursty.latency_p99_ns as f64) <= bound_ns;

    let baseline_unbounded_at_light_load =
        baseline_light.latency_p99_ns > 2 * hybrid_light.latency_p99_ns.max(1);

    let amortization_preserved_at_target_load =
        hybrid_target.sign_amortized_ns <= 2.0 * baseline_target.sign_amortized_ns.max(1.0);

    println!("\n=== acceptance ===");
    println!(
        "all closed spans verify under verify_signed_chain, every regime/policy: {all_verified}"
    );
    println!(
        "candidate_a p99 latency bounded by {}: target={} light={} bursty={} -> {hybrid_bounded}",
        fmt_ms(bound_ns),
        fmt_ms(hybrid_target.latency_p99_ns as f64),
        fmt_ms(hybrid_light.latency_p99_ns as f64),
        fmt_ms(hybrid_bursty.latency_p99_ns as f64)
    );
    println!(
        "baseline p99 at light load exceeds 2x candidate_a's p99 (demonstrates the unbounded-tail failure mode): baseline={} candidate_a={} -> {baseline_unbounded_at_light_load}",
        fmt_ms(baseline_light.latency_p99_ns as f64),
        fmt_ms(hybrid_light.latency_p99_ns as f64)
    );
    println!(
        "candidate_a amortized signing cost at target load within 2x of baseline: candidate_a={:.1}ns baseline={:.1}ns -> {amortization_preserved_at_target_load}",
        hybrid_target.sign_amortized_ns, baseline_target.sign_amortized_ns
    );

    let verdict = if all_verified
        && hybrid_bounded
        && baseline_unbounded_at_light_load
        && amortization_preserved_at_target_load
    {
        "ACCEPT"
    } else if all_verified && hybrid_bounded {
        "INCONCLUSIVE"
    } else {
        "REJECT"
    };
    println!("\nBATCH-FILL LATENCY ACCEPTANCE RESULT: {verdict}");
}
