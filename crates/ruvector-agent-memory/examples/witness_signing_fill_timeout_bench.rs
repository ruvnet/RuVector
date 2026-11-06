//! Nightly research benchmark (2026-10-23): closes ADR-347's Next
//! Research item 1 — "a wall-clock batch-fill timeout for `BatchTail`...
//! so a deployment gets a bounded worst-case signature-availability
//! latency instead of whenever the batch happens to fill" — by measuring
//! `SigningStrategy::batch_tail_timed`'s actual effect on that latency,
//! and separately confirming it does not regress `BatchTail`'s signing
//! amortization under normal (non-idle) load.
//!
//! ## What "signature-availability latency" means here
//!
//! For a witness record `r`, it is the elapsed time between `r`'s own
//! `timestamp_ns` and the moment the `SignedWitnessSink` span covering it
//! closes (and is therefore independently verifiable via
//! `verify_signed_chain`). Under the original `BatchTail` (no timeout),
//! that moment is whenever `batch_size` more records happen to arrive —
//! unbounded if they never do. `batch_tail_timed` adds a second trigger:
//! `max_wait_ns` after the span's oldest pending record.
//!
//! ## Methodology and its one deliberate simplification
//!
//! Inter-record arrival times are *synthetic, injected* `timestamp_ns`
//! values (built directly via [`Stream::next_record`], bypassing
//! `TransactionalLedger`'s real-wall-clock `now_ns()`), not real
//! `std::thread::sleep` — this experiment needs multi-second idle gaps,
//! and simulating those with real sleeps would make the benchmark
//! impractically slow without adding information (the signing/hashing
//! code path does not care what clock produced `timestamp_ns`). What IS
//! measured for real: every `SignedWitnessSink::emit_batch`,
//! `seal_on_timeout`, and `close()` call below runs the actual Ed25519
//! signing and SHA-256 hashing code, timed with `Instant`. The discrete-
//! event driver also assumes a scheduler that wakes at exactly
//! `pending_deadline_ns()` — the best case the timeout mechanism can
//! achieve; a real deployment's own scheduler wake-up jitter adds on top
//! of these numbers, not instead of them (see `witness_signing`'s module
//! docs, "The fill-timeout needs a driver").
//!
//! ## Experiment A — idle-gap availability latency
//!
//! Four 10-record bursts (50us apart within a burst), separated by 2s
//! idle gaps, batch_size=64 (so a burst alone never fills a span by size:
//! 10 < 64). `baseline` never gets a second trigger, so its pending span
//! only grows across bursts; `candidate_a`/`candidate_b` close on every
//! idle gap via `seal_on_timeout`.
//!
//! ## Experiment B — high-load amortization regression check
//!
//! 2,000 records back-to-back, 1us apart, batch_size=64: every span
//! should close by size long before any millisecond-scale timeout could
//! matter, so `candidate_a`/`candidate_b` must show essentially the same
//! signature count and per-op cost as `baseline` — the timeout must cost
//! nothing when it is not needed.
//!
//! ## Acceptance thresholds (fixed before running)
//!
//! A1. In Experiment A, `candidate_a`'s and `candidate_b`'s maximum
//!     observed per-record availability latency must be `<= max_wait_ns`.
//! A2. `baseline`'s pending staleness at the final checkpoint (4th idle
//!     gap's end) must exceed both candidates' `max_wait_ns` by at least
//!     10x, demonstrating the unbounded-wait problem this run fixes.
//! A3. `verify_signed_chain` must return `Ok` for all three variants
//!     after a final `seal()` (confirming the timeout path produces
//!     exactly as-valid a chain as the size-triggered path, per
//!     `witness_signing_tests::timeout_closed_span_verifies_exactly_like_a_size_closed_one`).
//! B1. In Experiment B, `candidate_a`'s and `candidate_b`'s signature
//!     count and mean per-op latency must be within 5% of `baseline`'s.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example witness_signing_fill_timeout_bench

use ruvector_agent_memory::{
    verify_signed_chain, EvidenceGrade, LedgerWitnessRecord, MemoryWitnessLog, SignedWitnessSink,
    SigningStrategy, WitnessSink,
};
use rvf_types::ed25519::Ed25519Keypair;
use std::time::Instant;

const SECRET: [u8; 32] = [21u8; 32];
const BATCH_SIZE: usize = 64;

/// Builds a correctly-chained synthetic record stream with fully
/// controlled `timestamp_ns` values — see the module docs' "one
/// deliberate simplification".
struct Stream {
    next_seq: u64,
    prev_hash: u64,
}

impl Stream {
    fn new() -> Self {
        Self {
            next_seq: 0,
            prev_hash: 0,
        }
    }

    fn next_record(&mut self, timestamp_ns: u64) -> LedgerWitnessRecord {
        let mut r = LedgerWitnessRecord {
            sequence: self.next_seq,
            timestamp_ns,
            action_kind: 0,
            proof_tier: 0,
            flags: (EvidenceGrade::Recomputed.code() as u16) << 12,
            actor_partition_id: 0,
            target_object_id: 0,
            capability_hash: 0,
            payload: self.next_seq,
            prev_hash: self.prev_hash,
            record_hash: 0,
            aux: 0,
            evidence_grade: EvidenceGrade::Recomputed,
        };
        r.record_hash = r.compute_record_hash();
        self.prev_hash = r.chain_hash();
        self.next_seq += 1;
        r
    }
}

fn new_sink(strategy: SigningStrategy) -> SignedWitnessSink<MemoryWitnessLog> {
    SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &Ed25519Keypair::from_secret(&SECRET),
        strategy,
    )
    .expect("valid strategy")
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn report_latency(label: &str, mut latencies_ns: Vec<u64>, max_wait_ns: Option<u64>) {
    latencies_ns.sort_unstable();
    let max = *latencies_ns.last().unwrap_or(&0);
    let mean: f64 = latencies_ns.iter().sum::<u64>() as f64 / latencies_ns.len().max(1) as f64;
    println!(
        "  {label:<12} n={:>4}  mean={:>12.1}ns  p50={:>10}ns  p99={:>10}ns  max={:>12}ns  bound={}",
        latencies_ns.len(),
        mean,
        percentile(&latencies_ns, 0.50),
        percentile(&latencies_ns, 0.99),
        max,
        max_wait_ns.map_or("none (unbounded)".to_string(), |w| format!(
            "{w}ns ({})",
            if max <= w { "HELD" } else { "VIOLATED" }
        )),
    );
}

/// Experiment A: four 10-record bursts separated by 2s idle gaps.
/// Returns (per-record availability latencies for every CLOSED span,
/// final pending staleness if anything is still open after the last
/// checkpoint, the sink for a final correctness check).
fn run_experiment_a(
    strategy: SigningStrategy,
    max_wait_ns: Option<u64>,
) -> (Vec<u64>, Option<u64>, SignedWitnessSink<MemoryWitnessLog>) {
    const BURSTS: usize = 4;
    const BURST_SIZE: u64 = 10;
    const BURST_GAP_NS: u64 = 50_000; // 50us between records within a burst
    const IDLE_GAP_NS: u64 = 2_000_000_000; // 2s between bursts

    let mut sink = new_sink(strategy);
    let mut stream = Stream::new();
    let mut latencies = Vec::new();
    let mut t = 0u64;

    for burst in 0..BURSTS {
        for i in 0..BURST_SIZE {
            let r = stream.next_record(t);
            sink.emit_batch(std::slice::from_ref(&r)).unwrap();
            t += BURST_GAP_NS;
            let _ = i;
        }
        // Checkpoint: either a timeout wakes exactly at the deadline
        // before the next burst starts, or (fixed-size-only) nothing
        // happens and the pending span just keeps growing.
        if let Some(deadline) = sink.pending_deadline_ns() {
            let closed_before = sink.spans().len();
            let fired = sink.seal_on_timeout(deadline);
            assert!(fired || burst + 1 == BURSTS, "deadline must fire mid-run");
            if sink.spans().len() > closed_before {
                let span = sink.spans().last().unwrap();
                for seq in span.covers_from_seq..=span.covers_to_seq {
                    let arrival = sink.inner().records[seq as usize].timestamp_ns;
                    latencies.push(deadline - arrival);
                }
            }
            t = t.max(deadline) + IDLE_GAP_NS;
        } else {
            t += IDLE_GAP_NS;
        }
    }

    // Final checkpoint for whatever is still pending (models an external
    // timer still watching after the last write — not a crash).
    let final_checkpoint = t;
    let final_pending_staleness = if sink.unsigned_pending() > 0 {
        let oldest_seq = sink.inner().records.len() - sink.unsigned_pending();
        Some(final_checkpoint - sink.inner().records[oldest_seq].timestamp_ns)
    } else {
        None
    };
    if let Some(deadline) = sink.pending_deadline_ns() {
        // candidates: this fires too (the 4th burst's own timeout).
        sink.seal_on_timeout(deadline.max(final_checkpoint));
        let span = sink.spans().last().unwrap();
        for seq in span.covers_from_seq..=span.covers_to_seq {
            let arrival = sink.inner().records[seq as usize].timestamp_ns;
            latencies.push(final_checkpoint.max(deadline) - arrival);
        }
    }
    let _ = max_wait_ns;
    (latencies, final_pending_staleness, sink)
}

/// Experiment B: 2,000 records back-to-back, 1us apart, no idle gaps —
/// every span should close by `batch_size` long before any timeout.
fn run_experiment_b(strategy: SigningStrategy) -> (usize, f64) {
    const N: usize = 2_000;
    const GAP_NS: u64 = 1_000;

    let mut sink = new_sink(strategy);
    let mut stream = Stream::new();
    let mut per_op_ns = Vec::with_capacity(N);
    let mut t = 0u64;
    for _ in 0..N {
        let r = stream.next_record(t);
        let op0 = Instant::now();
        sink.emit_batch(std::slice::from_ref(&r)).unwrap();
        per_op_ns.push(op0.elapsed().as_nanos() as u64);
        t += GAP_NS;
    }
    sink.seal();
    let mean = per_op_ns.iter().sum::<u64>() as f64 / per_op_ns.len() as f64;
    (sink.spans().len(), mean)
}

fn main() {
    println!("ruvector-agent-memory BatchTail fill-timeout benchmark (ADR-347 Next Research #1)\n");

    let variants: [(&str, SigningStrategy, Option<u64>); 3] = [
        ("baseline", SigningStrategy::batch_tail(BATCH_SIZE), None),
        (
            "candidate_a",
            SigningStrategy::batch_tail_timed(BATCH_SIZE, 10_000_000),
            Some(10_000_000),
        ),
        (
            "candidate_b",
            SigningStrategy::batch_tail_timed(BATCH_SIZE, 100_000_000),
            Some(100_000_000),
        ),
    ];

    println!("Experiment A: idle-gap signature-availability latency (4 bursts of 10, 2s apart)");
    let mut sinks_a = Vec::new();
    for (label, strategy, max_wait_ns) in variants {
        let (latencies, final_pending_staleness, sink) = run_experiment_a(strategy, max_wait_ns);
        report_latency(label, latencies, max_wait_ns);
        if let Some(staleness) = final_pending_staleness {
            println!(
                "    (+ still pending at final checkpoint: staleness={staleness}ns, unsigned_pending={})",
                sink.unsigned_pending()
            );
        }
        sinks_a.push((label, sink, final_pending_staleness));
    }

    println!("\nExperiment A correctness (verify_signed_chain after a final seal()):");
    for (label, mut sink, _) in sinks_a {
        sink.seal();
        let pk = sink.public_key();
        let ok = verify_signed_chain(sink.inner(), sink.spans(), &pk, &sink.anchor()).is_ok();
        println!(
            "  {label:<12} records={:<4} spans={:<3} correctness={}",
            sink.inner().records.len(),
            sink.spans().len(),
            if ok { "PASS" } else { "FAIL" }
        );
    }

    println!("\nExperiment B: high-load amortization regression check (2,000 records, 1us apart, no idle gaps)");
    let mut baseline_mean = 0.0;
    for (label, strategy, _) in variants {
        let (signatures, mean_ns) = run_experiment_b(strategy);
        if label == "baseline" {
            baseline_mean = mean_ns;
        }
        let regression_pct = (mean_ns - baseline_mean) / baseline_mean.max(1.0) * 100.0;
        println!(
            "  {label:<12} signatures={:<4} mean_per_op={:>8.1}ns  vs_baseline={:+.2}%",
            signatures, mean_ns, regression_pct
        );
    }
}
