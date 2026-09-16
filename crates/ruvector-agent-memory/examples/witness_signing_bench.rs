//! Nightly research benchmark (2026-09-16, PIR follow-up to ADR-134 §9):
//! measures the real latency/throughput cost of closing the `WitnessSigner`
//! gap named in `ops.rs` and `ledger.rs`, comparing three variants:
//!
//!   baseline    — unsigned `MemoryWitnessLog` (today's shipped behavior)
//!   candidate_a — `SignedWitnessSink` with `SigningStrategy::PerRecord`
//!   candidate_b — `SignedWitnessSink` with `SigningStrategy::BatchTail`
//!                 at a few batch sizes
//!
//! Workload: N sequential `add` + `accept` pairs (2N witness records),
//! deterministic content, single-threaded, release build. Reports mean/p50/
//! p95/p99 per-`add`+`accept` wall latency, total run time, signatures
//! produced, and a correctness gate (every produced signature verifies,
//! every constructed diligent-forgery attempt is rejected).
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example witness_signing_bench

use ruvector_agent_memory::{
    verify_signed_chain, AlwaysAdmitGate, MemoryWitnessLog, SignedWitnessSink, SigningStrategy,
    TransactionalLedger,
};
use rvf_types::ed25519::Ed25519Keypair;
use std::time::{Duration, Instant};

const N_ENTRIES: usize = 20_000;
const SECRET: [u8; 32] = [11u8; 32];

fn percentile(sorted_ns: &[u64], p: f64) -> f64 {
    if sorted_ns.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ns.len() - 1) as f64 * p).round() as usize;
    sorted_ns[idx] as f64 / 1000.0 // microseconds
}

struct RunResult {
    label: &'static str,
    total: Duration,
    per_op_us: Vec<u64>,
    signatures_produced: usize,
    correctness_ok: bool,
}

fn report(r: &RunResult) {
    let mut sorted = r.per_op_us.clone();
    sorted.sort_unstable();
    let mean: f64 = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64 / 1000.0;
    let throughput = N_ENTRIES as f64 / r.total.as_secs_f64();
    println!(
        "{:<14} total={:>9.3}ms  mean={:>8.3}us  p50={:>8.3}us  p95={:>8.3}us  p99={:>8.3}us  \
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

fn run_baseline() -> RunResult {
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
    RunResult {
        label: "baseline",
        total,
        per_op_us: per_op,
        signatures_produced: 0,
        correctness_ok: log.verify_chain(),
    }
}

fn run_signed(label: &'static str, strategy: SigningStrategy) -> RunResult {
    let sink = SignedWitnessSink::new(
        MemoryWitnessLog::default(),
        Ed25519Keypair::from_secret(&SECRET),
        strategy,
    );
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
    sink.flush();
    let pk = sink.public_key();
    let correctness_ok = verify_signed_chain(sink.inner(), sink.spans(), &pk);
    RunResult {
        label,
        total,
        per_op_us: per_op,
        signatures_produced: sink.spans().len(),
        correctness_ok,
    }
}

/// Construct a diligent, fully-recomputed forgery (see
/// `witness_signing::tests::diligent_forgery_defeats_chain_walk_alone_but_not_signatures`)
/// against a signed run and confirm it is rejected. Returns `true` iff the
/// forgery is correctly rejected (the desired, secure outcome).
fn diligent_forgery_is_rejected(strategy: SigningStrategy) -> bool {
    let sink = SignedWitnessSink::new(
        MemoryWitnessLog::default(),
        Ed25519Keypair::from_secret(&SECRET),
        strategy,
    );
    let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
    for i in 0..200 {
        let id = ledger.add(format!("m{i}"), &[], "bench", "r").unwrap();
        ledger.accept(id, "bench", "r").unwrap();
    }
    let mut sink = ledger.into_witness_sink();
    sink.flush();
    let pk = sink.public_key();

    let mut forged = sink.inner().clone();
    let tamper_at = forged.records.len() / 2;
    forged.records[tamper_at].payload ^= 0xDEAD_BEEF_u64;
    let mut prev_hash = if tamper_at == 0 {
        0
    } else {
        forged.records[tamper_at - 1].chain_hash()
    };
    for r in forged.records.iter_mut().skip(tamper_at) {
        r.prev_hash = prev_hash;
        r.record_hash = r.compute_record_hash();
        prev_hash = r.chain_hash();
    }
    forged.committed_head = prev_hash;
    forged.committed_count = forged.records.len() as u64;

    let chain_walk_alone_passes = forged.verify_chain();
    let signed_check_passes = verify_signed_chain(&forged, sink.spans(), &pk);
    // The whole point of signing: the unsigned chain walk is fooled, the
    // signed check is not.
    chain_walk_alone_passes && !signed_check_passes
}

fn main() {
    println!("ruvector-agent-memory witness signing benchmark");
    println!("N_ENTRIES={N_ENTRIES} (each = 1 add + 1 accept = 2 witness records)\n");

    let baseline = run_baseline();
    let per_record = run_signed("candidate_a", SigningStrategy::PerRecord);
    let batch_16 = run_signed(
        "candidate_b16",
        SigningStrategy::BatchTail { batch_size: 16 },
    );
    let batch_64 = run_signed(
        "candidate_b64",
        SigningStrategy::BatchTail { batch_size: 64 },
    );
    let batch_256 = run_signed(
        "candidate_b256",
        SigningStrategy::BatchTail { batch_size: 256 },
    );

    report(&baseline);
    report(&per_record);
    report(&batch_16);
    report(&batch_64);
    report(&batch_256);

    println!("\nDiligent-forgery rejection (chain-walk-alone fooled, signed check must reject):");
    for (label, strategy) in [
        ("candidate_a", SigningStrategy::PerRecord),
        (
            "candidate_b64",
            SigningStrategy::BatchTail { batch_size: 64 },
        ),
    ] {
        let ok = diligent_forgery_is_rejected(strategy);
        println!(
            "  {label:<16} forgery_rejected={}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    let per_record_mean: f64 =
        per_record.per_op_us.iter().sum::<u64>() as f64 / per_record.per_op_us.len() as f64;
    let batch64_mean: f64 =
        batch_64.per_op_us.iter().sum::<u64>() as f64 / batch_64.per_op_us.len() as f64;
    println!(
        "\nAmortization: candidate_a mean/op = {:.1}ns, candidate_b64 mean/op = {:.1}ns, ratio = {:.2}x",
        per_record_mean,
        batch64_mean,
        per_record_mean / batch64_mean.max(1.0)
    );
    println!(
        "Signature count: candidate_a={} candidate_b64={} (expected ratio ~{}x)",
        per_record.signatures_produced, batch_64.signatures_produced, 64
    );
}
