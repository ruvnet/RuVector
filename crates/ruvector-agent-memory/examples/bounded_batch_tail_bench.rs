//! Nightly research benchmark (2026-10-14, closing 2026-09-16's Next
//! Research item 1): measures whether `SigningStrategy::BoundedBatchTail`
//! actually bounds the TARL witness chain's worst-case
//! signature-availability latency, and whether it costs anything in
//! steady-state throughput relative to the existing fixed-size-only
//! `BatchTail`.
//!
//! Two independent experiments:
//!
//! 1. **Steady-state throughput** (dense arrivals, no idle gaps): N
//!    sequential `add`+`accept` pairs through a real `TransactionalLedger`,
//!    same methodology as the 2026-09-16 `witness_signing_bench` — baseline
//!    (unsigned), `BatchTail{64}`, `BoundedBatchTail{64, max_wait_ns=5ms}`.
//!    Wall-clock gaps between real `add`/`accept` calls in this tight loop
//!    are microseconds, far below `max_wait_ns`, so this isolates whether
//!    the extra inline deadline check costs anything when it essentially
//!    never fires.
//!
//! 2. **Worst-case availability latency** (bursty-then-idle, synthetic
//!    simulated clock — no real sleeping): `CYCLES` cycles of a short burst
//!    of arrivals followed by a long idle gap with zero further arrivals.
//!    `batch_size` is set deliberately huge in THIS experiment only (so it
//!    never triggers by size), isolating the timeout mechanism from the
//!    size trigger. A simulated periodic timer calls `seal_if_due` on both
//!    candidates during every idle gap (harmless no-op on plain
//!    `BatchTail`, confirmed by this crate's
//!    `seal_if_due_is_a_no_op_for_plain_batch_tail_and_per_record` unit
//!    test) — the same driver code runs for both, so any difference in
//!    outcome comes from the sink's own behavior, not the driver.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example bounded_batch_tail_bench

use ruvector_agent_memory::ops::pack_flags;
use ruvector_agent_memory::{
    verify_signed_chain, AlwaysAdmitGate, EvidenceGrade, LedgerWitnessRecord, MemoryWitnessLog,
    SignedWitnessSink, SigningStrategy, TransactionalLedger, WitnessSink,
};
use rvf_types::ed25519::Ed25519Keypair;
use std::time::{Duration, Instant};

const SECRET: [u8; 32] = [31u8; 32];

// ---------------------------------------------------------- experiment 1

const N_ENTRIES: usize = 20_000;
const STEADY_BATCH_SIZE: usize = 64;
const STEADY_MAX_WAIT_NS: u64 = 5_000_000; // 5ms

fn percentile(sorted_ns: &[u64], p: f64) -> f64 {
    if sorted_ns.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ns.len() - 1) as f64 * p).round() as usize;
    sorted_ns[idx] as f64 / 1000.0 // microseconds
}

struct ThroughputResult {
    label: &'static str,
    total: Duration,
    per_op_ns: Vec<u64>,
    signatures_produced: usize,
    correctness_ok: bool,
}

fn report_throughput(r: &ThroughputResult) {
    let mut sorted = r.per_op_ns.clone();
    sorted.sort_unstable();
    let mean: f64 = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64 / 1000.0;
    let throughput = N_ENTRIES as f64 / r.total.as_secs_f64();
    println!(
        "{:<16} total={:>9.3}ms  mean={:>8.3}us  p50={:>8.3}us  p95={:>8.3}us  p99={:>8.3}us  \
         throughput={:>10.1} ops/s  signatures={:>7}  correctness={}",
        r.label,
        r.total.as_secs_f64() * 1000.0,
        mean,
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.95),
        percentile(&sorted, 0.99),
        throughput,
        r.signatures_produced,
        if r.correctness_ok { "PASS" } else { "FAIL" },
    );
}

fn run_baseline() -> ThroughputResult {
    let mut ledger =
        TransactionalLedger::new(MemoryWitnessLog::default(), AlwaysAdmitGate::default());
    let mut per_op = Vec::with_capacity(N_ENTRIES);
    let t0 = Instant::now();
    for i in 0..N_ENTRIES {
        let op0 = Instant::now();
        let id = ledger
            .add(format!("memory entry {i}"), &[], "bench", "synthetic")
            .expect("add");
        ledger
            .accept(id, "bench", "synthetic verification")
            .expect("accept");
        per_op.push(op0.elapsed().as_nanos() as u64);
    }
    let total = t0.elapsed();
    let log = ledger.into_witness_sink();
    ThroughputResult {
        label: "baseline",
        total,
        per_op_ns: per_op,
        signatures_produced: 0,
        correctness_ok: log.verify_chain(),
    }
}

fn run_signed_throughput(label: &'static str, strategy: SigningStrategy) -> ThroughputResult {
    let sink = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &Ed25519Keypair::from_secret(&SECRET),
        strategy,
    )
    .expect("valid strategy");
    let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
    let mut per_op = Vec::with_capacity(N_ENTRIES);
    let t0 = Instant::now();
    for i in 0..N_ENTRIES {
        let op0 = Instant::now();
        let id = ledger
            .add(format!("memory entry {i}"), &[], "bench", "synthetic")
            .expect("add");
        ledger
            .accept(id, "bench", "synthetic verification")
            .expect("accept");
        per_op.push(op0.elapsed().as_nanos() as u64);
    }
    let total = t0.elapsed();
    let mut sink = ledger.into_witness_sink();
    sink.seal();
    let pk = sink.public_key();
    let correctness_ok =
        verify_signed_chain(sink.inner(), sink.spans(), &pk, &sink.anchor()).is_ok();
    ThroughputResult {
        label,
        total,
        per_op_ns: per_op,
        signatures_produced: sink.spans().len(),
        correctness_ok,
    }
}

// ---------------------------------------------------------- experiment 2

const CYCLES: usize = 10;
const BURST_SIZE: u64 = 5;
const BURST_GAP_NS: u64 = 1_000; // 1us between records within a burst
const IDLE_GAP_NS: u64 = 20_000_000; // 20ms of silence between bursts
const LATENCY_MAX_WAIT_NS: u64 = 5_000_000; // 5ms bound under test
const POLL_INTERVAL_NS: u64 = 1_000_000; // 1ms simulated timer tick
const NEVER_TRIGGER_BY_SIZE: usize = 1_000_000; // isolates the timeout path

/// Build a chained, verify_chain-valid record at `sequence`/`timestamp_ns`.
fn chained_record(prev: &mut u64, sequence: u64, timestamp_ns: u64) -> LedgerWitnessRecord {
    let mut r = LedgerWitnessRecord {
        sequence,
        timestamp_ns,
        action_kind: 0,
        proof_tier: 0,
        flags: pack_flags(None, None, EvidenceGrade::Recomputed),
        actor_partition_id: 0,
        target_object_id: 0,
        capability_hash: 0,
        payload: sequence,
        prev_hash: *prev,
        record_hash: 0,
        aux: 0,
        evidence_grade: EvidenceGrade::Recomputed,
    };
    r.record_hash = r.compute_record_hash();
    *prev = r.chain_hash();
    r
}

enum Event {
    Arrival(LedgerWitnessRecord),
    Poll,
}

/// Merge `CYCLES` [burst, idle-gap] cycles into one time-ordered event
/// trace, plus a steady stream of `Poll` ticks spanning the whole run.
fn build_trace() -> (Vec<(u64, Event)>, u64) {
    let mut events: Vec<(u64, Event)> = Vec::new();
    let mut prev_hash = 0u64;
    let mut seq = 0u64;
    let mut t = 0u64;
    for _ in 0..CYCLES {
        for _ in 0..BURST_SIZE {
            events.push((t, Event::Arrival(chained_record(&mut prev_hash, seq, t))));
            seq += 1;
            t += BURST_GAP_NS;
        }
        t += IDLE_GAP_NS;
    }
    let end = t;
    let mut poll_t = 0u64;
    while poll_t <= end {
        events.push((poll_t, Event::Poll));
        poll_t += POLL_INTERVAL_NS;
    }
    events.sort_by_key(|(t, e)| (*t, matches!(e, Event::Poll) as u8)); // arrivals before polls at equal t
    (events, end)
}

struct LatencyResult {
    label: &'static str,
    max_latency_ns: u64,
    mean_latency_ns: f64,
    spans_produced: usize,
    correctness_ok: bool,
}

fn run_latency_sim(label: &'static str, strategy: SigningStrategy) -> LatencyResult {
    let (events, end) = build_trace();
    let n_records = (CYCLES as u64) * BURST_SIZE;
    let mut sink = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &Ed25519Keypair::from_secret(&SECRET),
        strategy,
    )
    .expect("valid strategy");
    let mut record_ts = vec![0u64; n_records as usize];
    let mut close_time: Vec<Option<u64>> = vec![None; n_records as usize];
    let mut covered = 0usize; // spans fully accounted for so far

    let note_new_spans = |sink: &SignedWitnessSink<MemoryWitnessLog>,
                          covered: &mut usize,
                          close_time: &mut [Option<u64>],
                          now_ns: u64| {
        for span in &sink.spans()[*covered..] {
            for seq in span.covers_from_seq..=span.covers_to_seq {
                close_time[seq as usize] = Some(now_ns);
            }
        }
        *covered = sink.spans().len();
    };

    for (now, event) in events {
        match event {
            Event::Arrival(r) => {
                record_ts[r.sequence as usize] = r.timestamp_ns;
                sink.emit_batch(&[r]).expect("sequential arrival");
                note_new_spans(&sink, &mut covered, &mut close_time, now);
            }
            Event::Poll => {
                if sink.seal_if_due(now) {
                    note_new_spans(&sink, &mut covered, &mut close_time, now);
                }
            }
        }
    }
    // Deployment shutdown: sign whatever is still open.
    sink.seal();
    note_new_spans(&sink, &mut covered, &mut close_time, end);

    let latencies: Vec<u64> = (0..n_records as usize)
        .map(|i| {
            close_time[i].expect("every record covered by a span after final seal") - record_ts[i]
        })
        .collect();
    let max_latency_ns = *latencies.iter().max().unwrap();
    let mean_latency_ns = latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;

    let pk = sink.public_key();
    let correctness_ok =
        verify_signed_chain(sink.inner(), sink.spans(), &pk, &sink.anchor()).is_ok();

    LatencyResult {
        label,
        max_latency_ns,
        mean_latency_ns,
        spans_produced: sink.spans().len(),
        correctness_ok,
    }
}

fn report_latency(r: &LatencyResult) {
    println!(
        "{:<16} max_latency={:>12.3}ms  mean_latency={:>10.3}ms  spans_produced={:>4}  correctness={}",
        r.label,
        r.max_latency_ns as f64 / 1_000_000.0,
        r.mean_latency_ns / 1_000_000.0,
        r.spans_produced,
        if r.correctness_ok { "PASS" } else { "FAIL" },
    );
}

fn main() {
    println!("ruvector-agent-memory BoundedBatchTail benchmark\n");

    println!(
        "== Experiment 1: steady-state throughput (N_ENTRIES={N_ENTRIES}, dense real-clock \
         arrivals, batch_size={STEADY_BATCH_SIZE}, max_wait_ns={STEADY_MAX_WAIT_NS} for the \
         bounded variant) =="
    );
    let baseline = run_baseline();
    let batch_tail = run_signed_throughput(
        "candidate_a",
        SigningStrategy::BatchTail {
            batch_size: STEADY_BATCH_SIZE,
        },
    );
    let bounded = run_signed_throughput(
        "candidate_b",
        SigningStrategy::BoundedBatchTail {
            batch_size: STEADY_BATCH_SIZE,
            max_wait_ns: STEADY_MAX_WAIT_NS,
        },
    );
    report_throughput(&baseline);
    report_throughput(&batch_tail);
    report_throughput(&bounded);

    let a_mean: f64 =
        batch_tail.per_op_ns.iter().sum::<u64>() as f64 / batch_tail.per_op_ns.len() as f64;
    let b_mean: f64 = bounded.per_op_ns.iter().sum::<u64>() as f64 / bounded.per_op_ns.len() as f64;
    let throughput_overhead_pct = (b_mean - a_mean) / a_mean * 100.0;
    println!(
        "\ncandidate_a mean/op = {a_mean:.1}ns, candidate_b mean/op = {b_mean:.1}ns, \
         candidate_b overhead vs candidate_a = {throughput_overhead_pct:+.2}%"
    );

    println!(
        "\n== Experiment 2: worst-case signature-availability latency (CYCLES={CYCLES}, \
         burst={BURST_SIZE} records/{BURST_GAP_NS}ns apart, idle_gap={}ms, \
         batch_size={NEVER_TRIGGER_BY_SIZE} [never triggers by size, isolates the timeout \
         path], candidate_b max_wait_ns={}ms, simulated timer poll every {}ms) ==",
        IDLE_GAP_NS / 1_000_000,
        LATENCY_MAX_WAIT_NS / 1_000_000,
        POLL_INTERVAL_NS / 1_000_000,
    );
    let legacy = run_latency_sim(
        "candidate_a",
        SigningStrategy::BatchTail {
            batch_size: NEVER_TRIGGER_BY_SIZE,
        },
    );
    let bounded_latency = run_latency_sim(
        "candidate_b",
        SigningStrategy::BoundedBatchTail {
            batch_size: NEVER_TRIGGER_BY_SIZE,
            max_wait_ns: LATENCY_MAX_WAIT_NS,
        },
    );
    report_latency(&legacy);
    report_latency(&bounded_latency);

    println!("\n== Acceptance gates ==");
    let steady_state_ok = batch_tail.correctness_ok && bounded.correctness_ok;
    println!(
        "  [{}] steady-state verify_signed_chain correctness (both variants)",
        if steady_state_ok { "PASS" } else { "FAIL" }
    );
    let overhead_ok = throughput_overhead_pct.abs() <= 20.0;
    println!(
        "  [{}] candidate_b steady-state overhead vs candidate_a <= 20% (measured {:+.2}%)",
        if overhead_ok { "PASS" } else { "FAIL" },
        throughput_overhead_pct
    );
    let unbounded_ok = legacy.max_latency_ns >= IDLE_GAP_NS * (CYCLES as u64 - 1);
    println!(
        "  [{}] candidate_a (legacy BatchTail) worst-case latency grows with total idle time \
         (>= {:.1}ms, measured {:.1}ms) — demonstrates the unbounded failure mode",
        if unbounded_ok { "PASS" } else { "FAIL" },
        (IDLE_GAP_NS * (CYCLES as u64 - 1)) as f64 / 1_000_000.0,
        legacy.max_latency_ns as f64 / 1_000_000.0,
    );
    let bound = LATENCY_MAX_WAIT_NS + POLL_INTERVAL_NS + BURST_GAP_NS * BURST_SIZE;
    let bounded_ok = bounded_latency.max_latency_ns <= bound;
    println!(
        "  [{}] candidate_b worst-case latency bounded by max_wait_ns + poll_interval + \
         burst_span (<= {:.3}ms, measured {:.3}ms)",
        if bounded_ok { "PASS" } else { "FAIL" },
        bound as f64 / 1_000_000.0,
        bounded_latency.max_latency_ns as f64 / 1_000_000.0,
    );
    let latency_correctness_ok = legacy.correctness_ok && bounded_latency.correctness_ok;
    println!(
        "  [{}] latency-experiment verify_signed_chain correctness (both variants)",
        if latency_correctness_ok {
            "PASS"
        } else {
            "FAIL"
        }
    );

    let all_pass =
        steady_state_ok && overhead_ok && unbounded_ok && bounded_ok && latency_correctness_ok;
    println!(
        "\nOVERALL: {}",
        if all_pass {
            "ACCEPT (all gates pass)"
        } else {
            "REJECT (one or more gates failed)"
        }
    );
}
