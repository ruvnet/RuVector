//! Nightly research benchmark (2026-09-29, ADR-352): does
//! `SigningStrategy::BatchTailTimeout` bound worst-case signature-availability
//! latency without costing steady-state throughput?
//!
//! Part A — dense workload (batches fill long before any timeout):
//!   `BatchTail{64}` vs `BatchTailTimeout{64, 10ms}`, N_DENSE add+accept pairs,
//!   REPS interleaved repetitions each, compared by median total run time.
//!
//! Part B — bursty/slow workload (real `thread::sleep` gaps, deterministic
//!   schedule from a fixed xorshift seed): real ledger-generated records are
//!   emitted into the signed sink one per `emit_batch` call; per-record
//!   latency runs from "witnessed" (instant BEFORE the `emit_batch` call —
//!   conservative) to "signature exists" (instant after the call / poll that
//!   closed its span).
//!   Variants: plain `BatchTail{64}`; `BatchTailTimeout{64,10ms}` exactly as
//!   specified (write-driven checks only); and, as a supplementary row that is
//!   NOT part of the hypothesis, the same strategy plus a caller-owned 1 ms
//!   `seal_expired()` poll during idle gaps.
//!
//! Acceptance gates were fixed in this source before the first run (ADR-352).
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example bounded_batch_fill_bench

use ruvector_agent_memory::{
    verify_signed_chain, AlwaysAdmitGate, LedgerWitnessRecord, MemoryWitnessLog, SignedAnchor,
    SignedWitnessSink, SigningStrategy, TransactionalLedger, WitnessSink,
};
use rvf_types::ed25519::Ed25519Keypair;
use std::time::{Duration, Instant};

const SECRET: [u8; 32] = [11u8; 32];
const BATCH: usize = 64;
const MAX_WAIT: Duration = Duration::from_millis(10);
const N_DENSE: usize = 20_000;
const REPS: usize = 7;
const BURSTS: usize = 80;
const SEED: u64 = 0x5EED_2026_0929;
const POLL: Duration = Duration::from_millis(1);
/// Allowance over `max_wait` for one Ed25519 sign + clock/scheduler jitter.
const TOLERANCE: Duration = Duration::from_millis(1);
/// Gate A1: timeout strategy's median dense run time / plain's.
const MAX_DENSE_RATIO: f64 = 1.10;

type Sink = SignedWitnessSink<MemoryWitnessLog>;
type Ledger = TransactionalLedger<Sink, AlwaysAdmitGate>;

fn plain() -> SigningStrategy {
    SigningStrategy::BatchTail { batch_size: BATCH }
}
fn timed() -> SigningStrategy {
    SigningStrategy::BatchTailTimeout {
        batch_size: BATCH,
        max_wait: MAX_WAIT,
    }
}

fn ledger(strategy: SigningStrategy) -> Ledger {
    let sink = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &Ed25519Keypair::from_secret(&SECRET),
        strategy,
    )
    .expect("valid strategy");
    TransactionalLedger::new(sink, AlwaysAdmitGate::default())
}

fn verified(sink: &Sink) -> bool {
    verify_signed_chain(
        sink.inner(),
        sink.spans(),
        &sink.public_key(),
        &sink.anchor(),
    )
    .is_ok()
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

// ------------------------------------------------------------- Part A

struct DenseRun {
    total: Duration,
    signatures: usize,
    ok: bool,
}

fn dense(strategy: SigningStrategy) -> DenseRun {
    let mut l = ledger(strategy);
    let t0 = Instant::now();
    for i in 0..N_DENSE {
        let id = l
            .add(format!("memory entry {i}"), &[], "bench", "s")
            .unwrap();
        l.accept(id, "bench", "v").unwrap();
    }
    let total = t0.elapsed();
    let mut sink = l.into_witness_sink();
    sink.seal();
    DenseRun {
        total,
        signatures: sink.spans().len(),
        ok: verified(&sink),
    }
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

// ------------------------------------------------------------- Part B

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
}

/// (ops in burst, idle gap after it). Same schedule for every variant.
fn schedule() -> Vec<(usize, Duration)> {
    let mut rng = XorShift(SEED);
    (0..BURSTS)
        .map(|_| {
            let ops = rng.range(1, 6) as usize;
            let gap = Duration::from_millis(rng.range(2, 30));
            (ops, gap)
        })
        .collect()
}

struct Tracker {
    witnessed: Vec<Instant>,
    latency: Vec<Duration>,
    signed: usize,
}

impl Tracker {
    /// Stamp newly witnessed records with `before`, then stamp every record
    /// newly covered by a closed span with "now".
    fn observe(&mut self, sink: &Sink, before: Instant) {
        let len = sink.inner().records.len();
        self.witnessed.resize(len, before);
        let covered = sink.anchor().record_count as usize;
        if covered > self.signed {
            let now = Instant::now();
            for t in &self.witnessed[self.signed..covered] {
                self.latency.push(now - *t);
            }
            self.signed = covered;
        }
    }
}

struct BurstRun {
    label: &'static str,
    latency: Vec<Duration>,
    unsigned_at_end: usize,
    oldest_unsigned_age: Duration,
    signatures: usize,
    ok: bool,
    sink: Sink,
}

/// Honest witness records produced by the real ledger (2 per add+accept
/// pair). Part B replays them into the signed sink one record per
/// `emit_batch` call — exactly how the ledger emits them — so a caller-owned
/// timer can hold `&mut` to the sink (the ledger only lends `&S`).
fn honest_records(pairs: usize) -> Vec<LedgerWitnessRecord> {
    let mut l = TransactionalLedger::new(MemoryWitnessLog::default(), AlwaysAdmitGate::default());
    for i in 0..pairs {
        let id = l
            .add(format!("burst entry {i}"), &[], "bench", "s")
            .unwrap();
        l.accept(id, "bench", "v").unwrap();
    }
    l.into_witness_sink().records
}

fn bursty(label: &'static str, strategy: SigningStrategy, poll: bool) -> BurstRun {
    let sched = schedule();
    let recs = honest_records(sched.iter().map(|s| s.0).sum());
    let mut sink = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &Ed25519Keypair::from_secret(&SECRET),
        strategy,
    )
    .expect("valid strategy");
    let mut tr = Tracker {
        witnessed: Vec::new(),
        latency: Vec::new(),
        signed: 0,
    };
    let mut next = 0usize;
    for (ops, gap) in sched {
        for r in &recs[next..next + 2 * ops] {
            let t = Instant::now();
            sink.emit_batch(std::slice::from_ref(r)).unwrap();
            tr.observe(&sink, t);
        }
        next += 2 * ops;
        let idle_end = Instant::now() + gap;
        if poll {
            loop {
                let now = Instant::now();
                if now >= idle_end {
                    break;
                }
                std::thread::sleep(POLL.min(idle_end - now));
                let t = Instant::now();
                if sink.seal_expired() {
                    tr.observe(&sink, t);
                }
            }
        } else {
            std::thread::sleep(gap);
        }
    }
    let end = Instant::now();
    let unsigned_at_end = sink.unsigned_pending();
    let oldest_unsigned_age = tr
        .witnessed
        .get(tr.signed)
        .map_or(Duration::ZERO, |t| end - *t);
    sink.seal(); // for the correctness gate only; not counted as latency
    BurstRun {
        label,
        latency: tr.latency,
        unsigned_at_end,
        oldest_unsigned_age,
        signatures: sink.spans().len(),
        ok: verified(&sink),
        sink,
    }
}

fn pct(sorted: &[Duration], p: f64) -> Duration {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

fn report_burst(r: &BurstRun) -> Duration {
    let mut s = r.latency.clone();
    s.sort_unstable();
    let max = *s.last().unwrap();
    println!(
        "{:<22} signed={:>4} p50={:>7.3}ms p95={:>7.3}ms p99={:>7.3}ms max={:>7.3}ms  \
         unsigned_at_end={:>3} (oldest age {:>7.3}ms)  signatures={:>3}  correctness={}",
        r.label,
        s.len(),
        ms(pct(&s, 0.50)),
        ms(pct(&s, 0.95)),
        ms(pct(&s, 0.99)),
        ms(max),
        r.unsigned_at_end,
        ms(r.oldest_unsigned_age),
        r.signatures,
        if r.ok { "PASS" } else { "FAIL" },
    );
    max.max(r.oldest_unsigned_age)
}

fn forgery_rejected(sink: &Sink) -> bool {
    let mut forged = sink.inner().clone();
    let at = forged.records.len() / 2;
    forged.records[at].payload ^= 0xDEAD_BEEF;
    let mut prev = forged.records[at - 1].chain_hash();
    for r in forged.records.iter_mut().skip(at) {
        r.prev_hash = prev;
        r.record_hash = r.compute_record_hash();
        prev = r.chain_hash();
    }
    forged.committed_head = prev;
    forged.committed_count = forged.records.len() as u64;
    let fooled = forged.verify_chain();
    let rejected = verify_signed_chain(
        &forged,
        sink.spans(),
        &sink.public_key(),
        &SignedAnchor::genesis(),
    )
    .is_err();
    fooled && rejected
}

fn gate(name: &str, pass: bool) -> bool {
    println!("  {:<64} {}", name, if pass { "PASS" } else { "FAIL" });
    pass
}

fn main() {
    println!("ruvector-agent-memory bounded batch-fill signing benchmark (ADR-352)");
    println!(
        "batch_size={BATCH} max_wait={}ms tolerance={}ms\n",
        ms(MAX_WAIT),
        ms(TOLERANCE)
    );

    println!("Part A: dense workload, N={N_DENSE} add+accept pairs, {REPS} interleaved reps");
    let (mut pt, mut tt) = (Vec::new(), Vec::new());
    let (mut psig, mut tsig, mut dense_ok) = (0, 0, true);
    for rep in 0..REPS {
        let p = dense(plain());
        let t = dense(timed());
        println!(
            "  rep {rep}: plain_b64 total={:>8.3}ms sigs={}  timeout_b64 total={:>8.3}ms sigs={}",
            ms(p.total),
            p.signatures,
            ms(t.total),
            t.signatures
        );
        dense_ok &= p.ok && t.ok;
        (psig, tsig) = (p.signatures, t.signatures);
        pt.push(p.total);
        tt.push(t.total);
    }
    let (pm, tm) = (median(pt), median(tt));
    let ratio = tm.as_secs_f64() / pm.as_secs_f64();
    println!(
        "  median: plain_b64={:.3}ms ({:.3}us/pair)  timeout_b64={:.3}ms ({:.3}us/pair)  ratio={ratio:.3}\n",
        ms(pm),
        pm.as_secs_f64() * 1e6 / N_DENSE as f64,
        ms(tm),
        tm.as_secs_f64() * 1e6 / N_DENSE as f64,
    );

    let sched = schedule();
    let ops: usize = sched.iter().map(|s| s.0).sum();
    let idle: Duration = sched.iter().map(|s| s.1).sum();
    let long_gaps = sched.iter().filter(|s| s.1 > MAX_WAIT).count();
    println!(
        "Part B: bursty workload, seed={SEED:#x}, {BURSTS} bursts, {ops} add+accept pairs \
         ({} records), idle {:.0}ms total, {long_gaps}/{BURSTS} gaps > max_wait",
        2 * ops,
        ms(idle)
    );
    let b_plain = bursty("plain_b64", plain(), false);
    let b_timed = bursty("timeout_b64_10ms", timed(), false);
    let b_poll = bursty("timeout_b64_10ms+poll", timed(), true);
    let max_plain = report_burst(&b_plain);
    let max_timed = report_burst(&b_timed);
    let max_poll = report_burst(&b_poll);

    let bound = MAX_WAIT + TOLERANCE;
    println!("\nAcceptance (gates fixed before the first run):");
    let mut accept = true;
    accept &= gate(
        "A1 dense median ratio timeout/plain <= 1.10",
        ratio <= MAX_DENSE_RATIO,
    );
    accept &= gate(
        "A2 dense signature counts identical (timeout never fired)",
        psig == tsig,
    );
    accept &= gate(
        "A3 verify_signed_chain PASS, all dense + bursty runs",
        dense_ok && b_plain.ok && b_timed.ok && b_poll.ok,
    );
    accept &= gate(
        "A4 diligent forgery rejected (timeout_b64_10ms bursty log)",
        forgery_rejected(&b_timed.sink),
    );
    let b1 = gate(
        &format!(
            "B1 HYPOTHESIS: timeout_b64_10ms max latency {:.3}ms <= {:.1}ms",
            ms(max_timed),
            ms(bound)
        ),
        max_timed <= bound,
    );
    let valid = gate(
        &format!(
            "B2 workload validity: plain_b64 max latency {:.3}ms > {:.1}ms",
            ms(max_plain),
            ms(bound)
        ),
        max_plain > bound,
    );
    println!(
        "  (supplementary, not a gate) timeout+{}ms poll max latency {:.3}ms vs max_wait+poll+tol {:.1}ms",
        ms(POLL),
        ms(max_poll),
        ms(bound + POLL)
    );
    let verdict = if !valid {
        "INCONCLUSIVE"
    } else if accept && b1 {
        "ACCEPT"
    } else {
        "REJECT"
    };
    println!("\nBOUNDED BATCH-FILL ACCEPTANCE RESULT: {verdict}");
}
