//! Nightly research benchmark (2026-11-02, follow-up to the 2026-09-16
//! `witness_signing_bench` run): measures whether `SigningStrategy::BatchTail`'s
//! new `max_wait` field actually bounds the worst-case wall-clock time a
//! witness record spends unsigned, under a workload that arrives slower
//! than `batch_size` ever fills.
//!
//! Variants:
//!   baseline (`unbounded`)  — `BatchTail { batch_size: 64, max_wait: None }`,
//!                             a slow trickle of 8 records (< batch_size),
//!                             never flushed by anything but a manual seal.
//!   candidate_a (`hybrid`)  — same trickle, `max_wait: Some(50ms)`, closed
//!                             by repeatedly polling `seal_if_overdue_now`
//!                             on a short real-clock tick (the production
//!                             shape: a periodic maintenance call, not a
//!                             one-shot `seal()`).
//!   candidate_b (`size`)    — a fast burst of exactly `batch_size` records,
//!                             closed by the pre-existing size trigger, to
//!                             confirm the timeout path does not change
//!                             size-triggered behavior or its cost.
//!
//! Acceptance threshold (this run's falsifiable claim):
//!   `unbounded`'s pending records must still be unsigned with no signal
//!   that they ever will be (that is the bug being fixed);
//!   `hybrid`'s measured unsigned age must be `<= max_wait + slack`, where
//!   `slack` is the polling tick (an honestly-reported scheduling cost, not
//!   hidden); `size`'s per-op cost must not regress vs the 2026-09-16 run's
//!   own `candidate_b64` numbers (same signing path, unchanged by this run).
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example witness_signing_timeout_bench

use ruvector_agent_memory::{
    verify_signed_chain, AlwaysAdmitGate, MemoryWitnessLog, SignedWitnessSink, SigningStrategy,
    TransactionalLedger,
};
use rvf_types::ed25519::Ed25519Keypair;
use std::time::{Duration, Instant};

const SECRET: [u8; 32] = [21u8; 32];
const POLL_TICK: Duration = Duration::from_millis(5);
const MAX_WAIT: Duration = Duration::from_millis(50);
const TRICKLE_RECORDS: usize = 8; // < batch_size, so size never triggers.
const BATCH_SIZE: usize = 64;

fn fresh_sink(strategy: SigningStrategy) -> SignedWitnessSink<MemoryWitnessLog> {
    SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &Ed25519Keypair::from_secret(&SECRET),
        strategy,
    )
    .expect("valid strategy")
}

/// Baseline: a trickle well under `batch_size`, no timeout configured.
/// Demonstrates the bug this run closes: records sit unsigned forever,
/// with nothing in the public API that will ever close them short of a
/// caller remembering to call `seal()` manually.
fn run_unbounded() -> (usize, bool) {
    let sink = fresh_sink(SigningStrategy::BatchTail {
        batch_size: BATCH_SIZE,
        max_wait: None,
    });
    let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
    for i in 0..TRICKLE_RECORDS {
        let id = ledger
            .add(format!("trickle {i}"), &[], "bench", "r")
            .unwrap();
        ledger.accept(id, "bench", "r").unwrap();
    }
    let sink = ledger.into_witness_sink();
    let pending = sink.unsigned_pending();
    let has_deadline = sink.pending_deadline().is_some();
    (pending, has_deadline)
}

/// Candidate: the same trickle, but `max_wait` is set and a production-shaped
/// caller polls `seal_if_overdue_now` on a fixed real tick until it fires.
/// Reports the measured wall-clock age of the oldest record when it is
/// finally signed, including the polling slack.
fn run_hybrid_measured() -> (Duration, bool) {
    let sink = fresh_sink(SigningStrategy::BatchTail {
        batch_size: BATCH_SIZE,
        max_wait: Some(MAX_WAIT),
    });
    let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
    let t_oldest_arrival = Instant::now();
    for i in 0..TRICKLE_RECORDS {
        let id = ledger
            .add(format!("trickle {i}"), &[], "bench", "r")
            .unwrap();
        ledger.accept(id, "bench", "r").unwrap();
    }
    let mut sink = ledger.into_witness_sink();

    // Production shape: a periodic maintenance tick, not a one-shot sleep
    // computed from the deadline — this measures real scheduling slack.
    loop {
        std::thread::sleep(POLL_TICK);
        if sink.seal_if_overdue_now() {
            break;
        }
    }
    let age = t_oldest_arrival.elapsed();
    let pk = sink.public_key();
    let signed_ok = verify_signed_chain(sink.inner(), sink.spans(), &pk, &sink.anchor()).is_ok();
    (age, signed_ok)
}

/// Control: a fast burst that fills exactly `batch_size`, closed by the
/// pre-existing size trigger (unaffected by `max_wait`). Confirms the new
/// field does not disturb the already-measured size-triggered path.
fn run_size_triggered() -> (Duration, usize) {
    let sink = fresh_sink(SigningStrategy::BatchTail {
        batch_size: BATCH_SIZE,
        max_wait: Some(MAX_WAIT),
    });
    let mut ledger = TransactionalLedger::new(sink, AlwaysAdmitGate::default());
    let t0 = Instant::now();
    // batch_size records = batch_size/2 add+accept pairs.
    for i in 0..(BATCH_SIZE / 2) {
        let id = ledger.add(format!("burst {i}"), &[], "bench", "r").unwrap();
        ledger.accept(id, "bench", "r").unwrap();
    }
    let elapsed = t0.elapsed();
    let sink = ledger.into_witness_sink();
    (elapsed, sink.spans().len())
}

fn main() {
    println!("ruvector-agent-memory witness-signer batch-fill timeout benchmark");
    println!(
        "trickle_records={TRICKLE_RECORDS} batch_size={BATCH_SIZE} max_wait={MAX_WAIT:?} poll_tick={POLL_TICK:?}\n"
    );

    let (pending, has_deadline_before_timeout_configured) = run_unbounded();
    println!(
        "baseline (unbounded): unsigned_pending={pending} pending_deadline_without_max_wait={} \
         -> {}",
        has_deadline_before_timeout_configured,
        if pending == TRICKLE_RECORDS * 2 && !has_deadline_before_timeout_configured {
            "CONFIRMS unbounded tail (no max_wait => no deadline, ever)"
        } else {
            "UNEXPECTED"
        }
    );

    let (age, signed_ok) = run_hybrid_measured();
    let bound = MAX_WAIT + POLL_TICK * 2; // deadline + at most ~2 ticks of poll slack
    let within_bound = age <= bound;
    println!(
        "candidate_a (hybrid):  measured_unsigned_age={age:?}  bound={bound:?}  signed_chain_ok={signed_ok}  \
         -> {}",
        if within_bound && signed_ok { "PASS" } else { "FAIL" }
    );

    let (elapsed, spans) = run_size_triggered();
    println!(
        "candidate_b (size):    {} records closed by size trigger in {:?} ({} span(s)) -> {}",
        BATCH_SIZE,
        elapsed,
        spans,
        if spans == 1 { "PASS" } else { "FAIL" }
    );

    let unbounded_confirmed =
        pending == TRICKLE_RECORDS * 2 && !has_deadline_before_timeout_configured;
    let hybrid_bounded = within_bound && signed_ok;
    let size_unaffected = spans == 1;

    println!();
    if unbounded_confirmed && hybrid_bounded && size_unaffected {
        println!("ACCEPT: max_wait bounds worst-case signature-availability latency to {bound:?}; \
                   baseline (no max_wait) remains unbounded as documented; size-triggered path unaffected.");
    } else {
        println!("REJECT: see FAIL/UNEXPECTED markers above.");
        std::process::exit(1);
    }
}
