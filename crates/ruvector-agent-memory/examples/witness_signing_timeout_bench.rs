//! Nightly research benchmark (2026-10-11, ADR-348): measures whether
//! `SigningStrategy::BatchTailTimeout` actually bounds the worst-case
//! signature-availability latency that plain `BatchTail` leaves open — the
//! gap the 2026-09-16 nightly (ADR-347) named as its first "Next Research"
//! item, quoting `ruvector-retrieval-receipt::batch_fill`'s hybrid
//! count-or-timeout policy as the pattern to port.
//!
//! Two independent measurements, both against the real wall clock (no
//! synthetic/simulated time):
//!
//! 1. **Dense-burst overhead.** N_ENTRIES back-to-back add+accept pairs,
//!    no idle gaps, comparing `BatchTail{64}` against
//!    `BatchTailTimeout{64, 5ms}`. Hypothesis: with arrivals dense enough
//!    that the count-close always wins, the timeout variant's overhead
//!    over plain `BatchTail` is negligible (<=10% mean per-op latency,
//!    <=10% fewer signatures-per-record amortization).
//!
//! 2. **Sparse-arrival worst-case latency.** A short burst well under
//!    `batch_size`, then a real `std::thread::sleep` idle gap longer than
//!    `max_wait_ns`, simulating arrivals stopping. Hypothesis: plain
//!    `BatchTail` leaves the burst permanently unsigned (no timeout
//!    concept exists to bound it — confirmed structurally via
//!    `next_timeout_deadline_ns() == None`), while `BatchTailTimeout`,
//!    polled by a lightweight periodic tick (`check_timeout`), force-seals
//!    it within `max_wait_ns` plus the poll interval's slack.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example witness_signing_timeout_bench

use ruvector_agent_memory::{
    AlwaysAdmitGate, MemoryWitnessLog, SignedWitnessSink, SigningStrategy, TransactionalLedger,
};
use rvf_types::ed25519::Ed25519Keypair;
use std::time::{Duration, Instant};

const N_ENTRIES: usize = 20_000;
const SECRET: [u8; 32] = [21u8; 32];
const BATCH_SIZE: usize = 64;
const MAX_WAIT_NS: u64 = 5_000_000; // 5ms
const POLL_INTERVAL: Duration = Duration::from_millis(1);
const POLL_SAFETY_CUTOFF: Duration = Duration::from_millis(200);

fn percentile(sorted_ns: &[u64], p: f64) -> f64 {
    if sorted_ns.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ns.len() - 1) as f64 * p).round() as usize;
    sorted_ns[idx] as f64 / 1000.0 // microseconds
}

struct DenseResult {
    label: &'static str,
    total: Duration,
    per_op_us: Vec<u64>,
    signatures_produced: usize,
}

fn report_dense(r: &DenseResult) {
    let mut sorted = r.per_op_us.clone();
    sorted.sort_unstable();
    let mean: f64 = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64 / 1000.0;
    let throughput = N_ENTRIES as f64 / r.total.as_secs_f64();
    println!(
        "{:<16} total={:>9.3}ms  mean={:>8.3}us  p50={:>8.3}us  p95={:>8.3}us  p99={:>8.3}us  \
         throughput={:>10.1} ops/s  signatures={:>7}",
        r.label,
        r.total.as_secs_f64() * 1000.0,
        mean,
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.95),
        percentile(&sorted, 0.99),
        throughput,
        r.signatures_produced,
    );
}

fn run_dense(label: &'static str, strategy: SigningStrategy) -> DenseResult {
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
    DenseResult {
        label,
        total,
        per_op_us: per_op,
        signatures_produced: sink.spans().len(),
    }
}

/// Sparse-arrival scenario: `n_pairs` add+accept pairs (2*n_pairs records,
/// well under `BATCH_SIZE`), then a real idle gap, modeling arrivals that
/// stop before the batch fills.
fn sparse_burst(strategy: SigningStrategy, n_pairs: usize) -> SignedWitnessSink<MemoryWitnessLog> {
    let sink = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &Ed25519Keypair::from_secret(&SECRET),
        strategy,
    )
    .expect("valid strategy");
    let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
    for i in 0..n_pairs {
        let id = ledger
            .add(format!("sparse entry {i}"), &[], "bench", "synthetic")
            .expect("add");
        ledger
            .accept(id, "bench", "synthetic verification")
            .expect("accept");
    }
    ledger.into_witness_sink()
}

fn main() {
    println!("ruvector-agent-memory witness-signing bounded-timeout benchmark (ADR-348)");
    println!(
        "BATCH_SIZE={BATCH_SIZE}  MAX_WAIT_NS={MAX_WAIT_NS} ({:.1}ms)  N_ENTRIES={N_ENTRIES}\n",
        MAX_WAIT_NS as f64 / 1_000_000.0
    );

    // ---- Measurement 1: dense-burst overhead ----------------------------
    println!("== Measurement 1: dense-burst overhead (no idle gaps) ==");
    let batch_tail = run_dense(
        "batch_tail_64",
        SigningStrategy::BatchTail {
            batch_size: BATCH_SIZE,
        },
    );
    let batch_tail_timeout = run_dense(
        "batch_tail_timeout_64_5ms",
        SigningStrategy::BatchTailTimeout {
            batch_size: BATCH_SIZE,
            max_wait_ns: MAX_WAIT_NS,
        },
    );
    report_dense(&batch_tail);
    report_dense(&batch_tail_timeout);

    let bt_mean: f64 =
        batch_tail.per_op_us.iter().sum::<u64>() as f64 / batch_tail.per_op_us.len() as f64;
    let btt_mean: f64 = batch_tail_timeout.per_op_us.iter().sum::<u64>() as f64
        / batch_tail_timeout.per_op_us.len() as f64;
    let overhead_pct = (btt_mean - bt_mean) / bt_mean * 100.0;
    let sig_delta_pct = (batch_tail_timeout.signatures_produced as f64
        - batch_tail.signatures_produced as f64)
        / batch_tail.signatures_produced as f64
        * 100.0;
    println!(
        "\nMean-latency overhead of BatchTailTimeout over BatchTail: {overhead_pct:+.2}%  \
         (threshold: within +/-10%)"
    );
    println!(
        "Signature-count: batch_tail={}, batch_tail_timeout={}, delta={sig_delta_pct:+.2}% \
         (threshold: within +/-10%)",
        batch_tail.signatures_produced, batch_tail_timeout.signatures_produced,
    );
    let dense_gate_pass = overhead_pct.abs() <= 10.0 && sig_delta_pct.abs() <= 10.0;
    println!(
        "Dense-burst overhead gate: {}",
        if dense_gate_pass { "PASS" } else { "FAIL" }
    );

    // ---- Measurement 2: sparse-arrival worst-case latency ---------------
    println!("\n== Measurement 2: sparse-arrival worst-case latency (real idle gap) ==");
    let idle_gap = Duration::from_millis(3 * (MAX_WAIT_NS / 1_000_000));
    println!(
        "Burst = 10 pairs (20 records, well under batch_size={BATCH_SIZE}), then a real \
         {idle_gap:?} idle gap with zero further arrivals.\n"
    );

    // BatchTail: structurally, there is no timeout to bound the wait at all.
    let bt_sparse = sparse_burst(
        SigningStrategy::BatchTail {
            batch_size: BATCH_SIZE,
        },
        10,
    );
    let bt_pending_before = bt_sparse.unsigned_pending();
    std::thread::sleep(idle_gap);
    let bt_pending_after = bt_sparse.unsigned_pending();
    let bt_has_timeout_bound = bt_sparse.next_timeout_deadline_ns().is_some();
    println!(
        "batch_tail_64:          pending_before_gap={bt_pending_before}  \
         pending_after_gap={bt_pending_after}  has_timeout_bound={bt_has_timeout_bound}"
    );

    // BatchTailTimeout: a lightweight periodic tick (check_timeout) must
    // force-seal it within max_wait_ns + poll slack.
    let mut btt_sparse = sparse_burst(
        SigningStrategy::BatchTailTimeout {
            batch_size: BATCH_SIZE,
            max_wait_ns: MAX_WAIT_NS,
        },
        10,
    );
    let pending_before = btt_sparse.unsigned_pending();
    let t_arrival = Instant::now();
    let mut sealed_at = None;
    while t_arrival.elapsed() < POLL_SAFETY_CUTOFF {
        if btt_sparse.check_timeout() {
            sealed_at = Some(t_arrival.elapsed());
            break;
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    let pending_after = btt_sparse.unsigned_pending();
    println!(
        "batch_tail_timeout_64:  pending_before_gap={pending_before}  \
         pending_after_poll_loop={pending_after}  sealed_after={sealed_at:?}  \
         bound=max_wait_ns({:.1}ms)+poll({:.1}ms)",
        MAX_WAIT_NS as f64 / 1_000_000.0,
        POLL_INTERVAL.as_secs_f64() * 1000.0,
    );

    let sparse_gate_pass = bt_pending_after > 0
        && !bt_has_timeout_bound
        && pending_after == 0
        && sealed_at.is_some_and(|d| {
            d.as_nanos() as u64 <= MAX_WAIT_NS + 3 * POLL_INTERVAL.as_nanos() as u64
        });
    println!(
        "Sparse-arrival bounded-latency gate: {}",
        if sparse_gate_pass { "PASS" } else { "FAIL" }
    );

    // ---- Overall acceptance ----------------------------------------------
    println!(
        "\nAcceptance: {}",
        if dense_gate_pass && sparse_gate_pass {
            "ACCEPT"
        } else {
            "REJECT"
        }
    );
}
