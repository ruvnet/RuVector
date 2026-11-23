//! Nightly research benchmark (PIR follow-up to the 2026-09-16 nightly's
//! Next Research item 1): measures whether wrapping `SigningStrategy::
//! BatchTail` in a wall-clock fill timeout (`TimedBatchTailSink`) actually
//! bounds worst-case signature-availability latency under slow/bursty
//! arrival, the way today's shipped `BatchTail` (size-only) cannot.
//!
//! Methodology ported (not copied) from
//! `ruvector-retrieval-receipt/src/bin/batch_latency.rs`'s discrete-event
//! simulation: deterministic xorshift arrival processes (Poisson + bursty
//! on/off), a virtual clock driving the batch-fill *decision*, real
//! Ed25519 signing cost measured with `Instant` and added on top of the
//! virtual close time, and a fixed-before-running acceptance gate.
//!
//! Three variants, one witness record per synthetic arrival:
//!   baseline    — today's shipped `SignedWitnessSink` with
//!                 `SigningStrategy::BatchTail{batch_size:64}`, no wrapper,
//!                 no timeout: closes only when 64 records have arrived.
//!   candidate_a — `TimedBatchTailSink` wrapping the same, max_wait_ns=50ms.
//!   candidate_b — `TimedBatchTailSink` wrapping the same, max_wait_ns=10ms
//!                 (tighter bound, more timeout-triggered seals: the
//!                 latency/signing-cost tradeoff curve).
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example batch_tail_timeout_latency

use ruvector_agent_memory::{
    verify_signed_chain, EvidenceGrade, LedgerWitnessRecord, LogicalClock, MemoryWitnessLog,
    SignedWitnessSink, SigningStrategy, TimedBatchTailSink, WitnessSink,
};
use rvf_types::ed25519::Ed25519Keypair;
use std::time::Instant;

const SECRET: [u8; 32] = [23u8; 32];
const NS_PER_MS: u64 = 1_000_000;
const BATCH_SIZE: usize = 64;

/// Deterministic xorshift, independently seeded per regime — same pattern
/// `ruvector-retrieval-receipt`'s `bin/batch_latency.rs` uses, so neither
/// benchmark depends on an external RNG crate for its arrival process.
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
    index: usize,
    arrived_at_ns: u64,
}

fn poisson_arrivals(count: usize, rate_per_sec: f64, seed: u64) -> Vec<Arrival> {
    let mut rng = Xorshift64(seed);
    let mean_gap_ns = 1.0e9 / rate_per_sec;
    let mut t_ns = 0.0f64;
    (0..count)
        .map(|i| {
            t_ns += -mean_gap_ns * rng.next_unit_open().ln();
            Arrival {
                index: i,
                arrived_at_ns: t_ns as u64,
            }
        })
        .collect()
}

/// Bursty on/off Poisson process: `on_rate_per_sec` for `on_ns`, then
/// `off_ns` of silence, repeating until `count` arrivals exist.
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
        t_ns += -mean_gap_ns * rng.next_unit_open().ln();
        if t_ns - cycle_start_ns as f64 >= on_ns as f64 {
            cycle_start_ns += on_ns + off_ns;
            t_ns = cycle_start_ns as f64;
            continue;
        }
        out.push(Arrival {
            index: out.len(),
            arrived_at_ns: t_ns as u64,
        });
    }
    out
}

fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn fmt_ms(ns: f64) -> String {
    format!("{:.3}ms", ns / NS_PER_MS as f64)
}

/// Build the next witness record in the chain: real ADR-134 chaining
/// (`prev_hash` from the previous record, `record_hash` computed), a
/// distinct deterministic `payload` per arrival, graded `Recomputed` (no
/// signature exists yet when a record is created — that is what this
/// benchmark adds).
fn next_record(seq: u64, prev_hash: u64, payload: u64) -> LedgerWitnessRecord {
    let grade = EvidenceGrade::Recomputed;
    let mut r = LedgerWitnessRecord {
        sequence: seq,
        timestamp_ns: 0,
        action_kind: 0,
        proof_tier: 0,
        flags: ruvector_agent_memory::ops::pack_flags(None, None, grade),
        actor_partition_id: 0,
        target_object_id: u32::MAX,
        capability_hash: 0,
        payload,
        prev_hash,
        record_hash: 0,
        aux: 0,
        evidence_grade: grade,
    };
    r.record_hash = r.compute_record_hash();
    r
}

struct RegimeStats {
    regime: &'static str,
    policy: &'static str,
    num_arrivals: usize,
    spans_closed: usize,
    timeout_seals: usize,
    availability_mean_ns: f64,
    availability_p50_ns: u64,
    availability_p95_ns: u64,
    availability_p99_ns: u64,
    availability_max_ns: u64,
    sign_amortized_ns: f64,
    verified: bool,
}

/// Drive `arrivals` through a freshly built `SignedWitnessSink` (baseline,
/// `max_wait_ns = None`) or `TimedBatchTailSink` (`max_wait_ns = Some(_)`),
/// using a `LogicalClock` for virtual arrival/deadline time and a real
/// `Instant` for the Ed25519 signing cost that gets added on top of it.
fn run_policy(
    regime: &'static str,
    policy: &'static str,
    max_wait_ns: Option<u64>,
    arrivals: &[Arrival],
) -> RegimeStats {
    let clock = LogicalClock::new(0);
    let keypair = Ed25519Keypair::from_secret(&SECRET);
    let inner = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &keypair,
        SigningStrategy::BatchTail {
            batch_size: BATCH_SIZE,
        },
    )
    .expect("valid strategy");

    // `None` (baseline) wraps in nothing at all: the literal shipped sink.
    // `Some(w)` wraps in `TimedBatchTailSink`; the wrapper is still driven
    // through its own `WitnessSink::emit_batch`, so both paths exercise
    // real production code, not a simulation-only stand-in. The two cases
    // are handled in separate branches below (rather than a shared enum)
    // because `TimedBatchTailSink<_, &LogicalClock>` borrows `clock`, and
    // threading that borrow through a shared match arm gains nothing here.
    let mut availability_ns = vec![0u64; arrivals.len()];
    let mut spans_closed = 0usize;
    let mut timeout_seals = 0usize;
    let mut sign_total_ns: u128 = 0;
    let mut prev_hash = 0u64;
    let mut next_seq = 0u64;
    let verified_chain;

    // One real signature-count snapshot + a timed call, used identically
    // for both the baseline and the timed sink via a small closure pair
    // defined per-branch below (kept inline rather than behind the `Sink`
    // enum above, which Rust's borrow checker cannot express cleanly with
    // a borrowed `LogicalClock`; the duplication is the price of avoiding
    // `unsafe`/`'static` lifetime tricks for a benchmark binary).
    if let Some(max_wait_ns) = max_wait_ns {
        let mut sink = TimedBatchTailSink::new(inner, max_wait_ns, &clock);
        let mut i = 0usize;
        while i < arrivals.len() {
            let deadline = sink.pending_deadline_ns();
            let next_arrival_ns = arrivals[i].arrived_at_ns;
            if let Some(d) = deadline {
                if d <= next_arrival_ns {
                    clock.set_ns(d);
                    let before = sink.spans().len();
                    let t0 = Instant::now();
                    let fired = sink.check_timeout();
                    let sign_elapsed = t0.elapsed().as_nanos();
                    assert!(fired, "deadline reached implies a pending span to seal");
                    let span = *sink.spans().last().expect("just closed one");
                    debug_assert_eq!(sink.spans().len(), before + 1);
                    let avail = d + sign_elapsed as u64;
                    for seq in span.covers_from_seq..=span.covers_to_seq {
                        availability_ns[seq as usize] = avail;
                    }
                    sign_total_ns += sign_elapsed;
                    spans_closed += 1;
                    timeout_seals += 1;
                    continue; // re-evaluate against the same next arrival
                }
            }
            let a = arrivals[i];
            i += 1;
            clock.set_ns(a.arrived_at_ns);
            let record = next_record(next_seq, prev_hash, a.index as u64);
            prev_hash = record.chain_hash();
            next_seq += 1;
            let before = sink.spans().len();
            let t0 = Instant::now();
            sink.emit(&record).expect("emit");
            let sign_elapsed = t0.elapsed().as_nanos();
            if sink.spans().len() > before {
                let span = *sink.spans().last().expect("just closed one");
                let avail = a.arrived_at_ns + sign_elapsed as u64;
                for seq in span.covers_from_seq..=span.covers_to_seq {
                    availability_ns[seq as usize] = avail;
                }
                sign_total_ns += sign_elapsed;
                spans_closed += 1;
            }
        }
        if sink.unsigned_pending() > 0 {
            let close_at = arrivals.last().map(|a| a.arrived_at_ns).unwrap_or(0);
            let t0 = Instant::now();
            sink.seal();
            let sign_elapsed = t0.elapsed().as_nanos();
            let span = *sink.spans().last().expect("flush closed one");
            let avail = close_at + sign_elapsed as u64;
            for seq in span.covers_from_seq..=span.covers_to_seq {
                availability_ns[seq as usize] = avail;
            }
            sign_total_ns += sign_elapsed;
            spans_closed += 1;
        }
        let pk = sink.public_key();
        verified_chain =
            verify_signed_chain(sink.inner_log(), sink.spans(), &pk, &sink.anchor()).is_ok();
    } else {
        let mut sink = inner;
        for a in arrivals {
            clock.set_ns(a.arrived_at_ns);
            let record = next_record(next_seq, prev_hash, a.index as u64);
            prev_hash = record.chain_hash();
            next_seq += 1;
            let before = sink.spans().len();
            let t0 = Instant::now();
            sink.emit(&record).expect("emit");
            let sign_elapsed = t0.elapsed().as_nanos();
            if sink.spans().len() > before {
                let span = *sink.spans().last().expect("just closed one");
                let avail = a.arrived_at_ns + sign_elapsed as u64;
                for seq in span.covers_from_seq..=span.covers_to_seq {
                    availability_ns[seq as usize] = avail;
                }
                sign_total_ns += sign_elapsed;
                spans_closed += 1;
            }
        }
        if sink.unsigned_pending() > 0 {
            let close_at = arrivals.last().map(|a| a.arrived_at_ns).unwrap_or(0);
            let t0 = Instant::now();
            sink.seal();
            let sign_elapsed = t0.elapsed().as_nanos();
            let span = *sink.spans().last().expect("flush closed one");
            let avail = close_at + sign_elapsed as u64;
            for seq in span.covers_from_seq..=span.covers_to_seq {
                availability_ns[seq as usize] = avail;
            }
            sign_total_ns += sign_elapsed;
            spans_closed += 1;
        }
        let pk = sink.public_key();
        verified_chain =
            verify_signed_chain(sink.inner(), sink.spans(), &pk, &sink.anchor()).is_ok();
    }

    let mut latencies: Vec<u64> = arrivals
        .iter()
        .map(|a| availability_ns[a.index].saturating_sub(a.arrived_at_ns))
        .collect();
    latencies.sort_unstable();
    let mean = latencies.iter().map(|&l| l as f64).sum::<f64>() / latencies.len() as f64;

    RegimeStats {
        regime,
        policy,
        num_arrivals: arrivals.len(),
        spans_closed,
        timeout_seals,
        availability_mean_ns: mean,
        availability_p50_ns: percentile(&latencies, 50.0),
        availability_p95_ns: percentile(&latencies, 95.0),
        availability_p99_ns: percentile(&latencies, 99.0),
        availability_max_ns: latencies.last().copied().unwrap_or(0),
        sign_amortized_ns: sign_total_ns as f64 / arrivals.len() as f64,
        verified: verified_chain,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_arrivals: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4000);

    println!("=== ruvector-agent-memory BatchTail fill-timeout latency simulation ===");
    println!("batch_size={BATCH_SIZE} arrivals_per_regime={num_arrivals}\n");

    let regimes: Vec<(&'static str, Vec<Arrival>)> = vec![
        (
            "target_load_5000qps",
            poisson_arrivals(num_arrivals, 5000.0, 0x1111_2222_3333_4444),
        ),
        (
            "light_load_50qps",
            poisson_arrivals(num_arrivals, 50.0, 0x5555_6666_7777_8888),
        ),
        (
            "bursty_on2000qps_off400ms",
            bursty_arrivals(
                num_arrivals,
                2000.0,
                100 * NS_PER_MS,
                400 * NS_PER_MS,
                0x9999_AAAA_BBBB_CCCC,
            ),
        ),
    ];

    let policies: [(&'static str, Option<u64>); 3] = [
        ("baseline_no_timeout", None),
        ("candidate_a_50ms", Some(50 * NS_PER_MS)),
        ("candidate_b_10ms", Some(10 * NS_PER_MS)),
    ];

    let mut all_stats = Vec::new();
    for (regime_name, arrivals) in &regimes {
        for (policy_name, max_wait_ns) in policies {
            all_stats.push(run_policy(regime_name, policy_name, max_wait_ns, arrivals));
        }
    }

    println!(
        "{:<26} {:<20} {:>5} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10} {:>14} {:>9}",
        "regime",
        "policy",
        "n",
        "spans",
        "t/o",
        "avail_mean",
        "avail_p50",
        "avail_p95",
        "avail_p99",
        "avail_max",
        "sign_amort_ns",
        "verified"
    );
    for s in &all_stats {
        println!(
            "{:<26} {:<20} {:>5} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10} {:>14.1} {:>9}",
            s.regime,
            s.policy,
            s.num_arrivals,
            s.spans_closed,
            s.timeout_seals,
            fmt_ms(s.availability_mean_ns),
            fmt_ms(s.availability_p50_ns as f64),
            fmt_ms(s.availability_p95_ns as f64),
            fmt_ms(s.availability_p99_ns as f64),
            fmt_ms(s.availability_max_ns as f64),
            s.sign_amortized_ns,
            s.verified
        );
    }

    // ── Acceptance evaluation (thresholds fixed before this run) ──────────
    let find = |regime: &str, policy: &str| -> &RegimeStats {
        all_stats
            .iter()
            .find(|s| s.regime == regime && s.policy == policy)
            .expect("regime/policy combination was run")
    };

    let all_verified = all_stats.iter().all(|s| s.verified);

    let a_light = find("light_load_50qps", "candidate_a_50ms");
    let b_light = find("light_load_50qps", "candidate_b_10ms");
    let a_bursty = find("bursty_on2000qps_off400ms", "candidate_a_50ms");
    let baseline_light = find("light_load_50qps", "baseline_no_timeout");

    let bound_a_ns = 50 * NS_PER_MS + 5 * NS_PER_MS;
    let bound_b_ns = 10 * NS_PER_MS + 5 * NS_PER_MS;
    let bounded = a_light.availability_p99_ns <= bound_a_ns
        && a_bursty.availability_p99_ns <= bound_a_ns
        && b_light.availability_p99_ns <= bound_b_ns;

    let baseline_unbounded_at_light_load =
        baseline_light.availability_p99_ns > 5 * a_light.availability_p99_ns.max(1);

    let baseline_target = find("target_load_5000qps", "baseline_no_timeout");
    let a_target = find("target_load_5000qps", "candidate_a_50ms");
    let amortization_preserved_at_target_load =
        a_target.sign_amortized_ns <= 2.0 * baseline_target.sign_amortized_ns.max(1.0);

    println!("\n=== acceptance ===");
    println!("all closed spans verify (every regime/policy): {all_verified}");
    println!(
        "candidate_a p99 bounded by {}: light={} bursty={} -> {}",
        fmt_ms(bound_a_ns as f64),
        fmt_ms(a_light.availability_p99_ns as f64),
        fmt_ms(a_bursty.availability_p99_ns as f64),
        a_light.availability_p99_ns <= bound_a_ns && a_bursty.availability_p99_ns <= bound_a_ns
    );
    println!(
        "candidate_b p99 bounded by {}: light={} -> {}",
        fmt_ms(bound_b_ns as f64),
        fmt_ms(b_light.availability_p99_ns as f64),
        b_light.availability_p99_ns <= bound_b_ns
    );
    println!(
        "baseline p99 at light load exceeds 5x candidate_a's p99 (demonstrates the unbounded-tail failure mode this fixes): baseline={} candidate_a={} -> {}",
        fmt_ms(baseline_light.availability_p99_ns as f64),
        fmt_ms(a_light.availability_p99_ns as f64),
        baseline_unbounded_at_light_load
    );
    println!(
        "candidate_a amortized signing cost at target load within 2x of baseline's: candidate_a={:.1}ns baseline={:.1}ns -> {}",
        a_target.sign_amortized_ns, baseline_target.sign_amortized_ns, amortization_preserved_at_target_load
    );
    println!(
        "timeout-triggered seals (tradeoff curve): candidate_a_light={} candidate_b_light={} candidate_a_target={} (expect b >= a under slow load, both ~0 under healthy load)",
        a_light.timeout_seals, b_light.timeout_seals, a_target.timeout_seals
    );

    let verdict = if !all_verified {
        "REJECT"
    } else if bounded && baseline_unbounded_at_light_load && amortization_preserved_at_target_load {
        "ACCEPT"
    } else if bounded {
        "INCONCLUSIVE"
    } else {
        "REJECT"
    };
    println!("\nBATCH-TAIL FILL-TIMEOUT LATENCY ACCEPTANCE RESULT: {verdict}");
}
