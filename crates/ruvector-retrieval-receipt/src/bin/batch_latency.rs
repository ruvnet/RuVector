//! End-to-end signed-receipt batch-fill latency simulation.
//!
//! ADR-340 measured the CPU cost of signing/verifying an *already
//! assembled* batch of `MerkleReceipt` roots and explicitly left wall-clock
//! batch-fill latency unmodeled (its Next Research item #1). This binary
//! closes that gap: a discrete-event simulation drives real query arrivals
//! under several load regimes through the real `BatchScheduler` +
//! `Issuer`/`BatchAnchor` signing path (real Ed25519 signs, real SHA-256
//! Merkle trees — nothing here is a stand-in number), and reports the
//! resulting end-to-end receipt-availability latency: the time from a
//! query's arrival until its signed batch anchor exists.
//!
//! Usage:
//!   cargo run --release -p ruvector-retrieval-receipt --bin batch_latency
//!   cargo run --release -p ruvector-retrieval-receipt --bin batch_latency -- 2000 64 10 3000

use std::time::Instant;

use ruvector_retrieval_receipt::{
    query_hash, synthetic_queries, verify_root, AnchorContext, AnchorPurpose, BatchAnchor,
    BatchFillPolicy, BatchScheduler, Issuer, ReceiptVariant, RetrievalIndex, RetrievalReceipt,
};

const ISSUED_AT_UNIX_MS: u64 = 1_788_134_400_000;
const NS_PER_MS: u64 = 1_000_000;

/// Deterministic xorshift, seeded independently per regime/run so arrival
/// timing is reproducible without pulling in an external RNG dependency —
/// same pattern as `bin/benchmark.rs`'s own `Xorshift64`.
struct Xorshift64(u64);
impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform(0,1) open interval, avoiding 0.0 so `ln` never sees zero.
    fn next_unit_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
    }
}

fn percentile(sorted: &[u64], pct: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0) as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// One arrival: which synthetic query it uses, and its virtual arrival
/// time in nanoseconds since the regime's simulation start.
#[derive(Clone, Copy)]
struct Arrival {
    query_index: usize,
    arrived_at_ns: u64,
}

/// Generate `count` arrivals as a Poisson process at `rate_per_sec`,
/// exponential interarrival times, deterministic seed.
fn poisson_arrivals(count: usize, rate_per_sec: f64, seed: u64) -> Vec<Arrival> {
    let mut rng = Xorshift64(seed);
    let mean_gap_ns = 1.0e9 / rate_per_sec;
    let mut t_ns = 0.0f64;
    (0..count)
        .map(|i| {
            let gap = -mean_gap_ns * rng.next_unit_open().ln();
            t_ns += gap;
            Arrival {
                query_index: i,
                arrived_at_ns: t_ns as u64,
            }
        })
        .collect()
}

/// Bursty on/off Poisson process: alternates `on_ns` at `on_rate_per_sec`
/// with `off_ns` of silence, repeating until `count` arrivals are
/// produced. Models an agent's clustered tool-call / query bursts rather
/// than a smooth arrival rate.
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
            // Advance past the off-period into the next on-period.
            cycle_start_ns += on_ns + off_ns;
            t_ns = cycle_start_ns as f64;
            continue;
        }
        out.push(Arrival {
            query_index: out.len(),
            arrived_at_ns: t_ns as u64,
        });
    }
    out
}

struct RegimeStats {
    regime: &'static str,
    policy: &'static str,
    num_queries: usize,
    num_batches: usize,
    mean_batch_size: f64,
    latency_mean_ns: f64,
    latency_p50_ns: u64,
    latency_p95_ns: u64,
    latency_p99_ns: u64,
    latency_max_ns: u64,
    sign_amortized_ns: f64,
    all_batches_verified: bool,
}

/// Accumulated outcome of closing one batch: real signing/verification
/// work plus the bookkeeping `run_policy` folds into its running stats.
struct ClosedBatch {
    available_at_ns: u64,
    member_query_indices: Vec<usize>,
    sign_elapsed_ns: u128,
    verified: bool,
}

/// Sign and verify one closed batch for real: builds the `BatchAnchor`,
/// signs its root with `issuer`, times that with a real `Instant`, and
/// verifies every member's inclusion proof under the resulting signature.
/// `close_at_ns` is the virtual simulation time at which the batch-fill
/// *decision* fired (size reached or timeout elapsed); the real measured
/// signing wall time is added on top to get each member's availability
/// time, so batch-fill queueing and real cryptographic cost are both
/// represented in the latency this produces.
fn sign_and_verify_batch(
    members: &[ruvector_retrieval_receipt::PendingMember],
    close_at_ns: u64,
    roots: &[[u8; 32]],
    issuer: &Issuer,
    context: AnchorContext,
) -> ClosedBatch {
    let batch_roots: Vec<[u8; 32]> = members.iter().map(|m| roots[m.query_index]).collect();

    let t0 = Instant::now();
    let anchor = BatchAnchor::build(&batch_roots).expect("nonempty closed batch");
    let signed = issuer.sign_root(context, anchor.root(), ISSUED_AT_UNIX_MS);
    let sign_elapsed = t0.elapsed();

    let verified = verify_root(&issuer.verifying_key, context, &signed);
    let batch_ok = verified.as_ref().is_some_and(|trusted| {
        (0..batch_roots.len()).all(|i| {
            let proof = anchor.proof_for(i).expect("in-bounds index");
            BatchAnchor::verify_inclusion(batch_roots[i], &proof, trusted)
        })
    });

    ClosedBatch {
        available_at_ns: close_at_ns + sign_elapsed.as_nanos() as u64,
        member_query_indices: members.iter().map(|m| m.query_index).collect(),
        sign_elapsed_ns: sign_elapsed.as_nanos(),
        verified: batch_ok,
    }
}

/// Drive one (regime, policy) combination through the scheduler, closing
/// batches with real `BatchAnchor` + `Issuer::sign_root` calls, and
/// computing end-to-end availability latency per query. `roots` must be
/// precomputed once per regime (same query content across all policies
/// tested against that regime) so the comparison is fair and no policy
/// benefits from cheaper/hotter receipt generation than another.
fn run_policy(
    regime: &'static str,
    policy_name: &'static str,
    policy: BatchFillPolicy,
    arrivals: &[Arrival],
    roots: &[[u8; 32]],
    issuer: &Issuer,
    scope_hash: [u8; 32],
) -> RegimeStats {
    let context = AnchorContext::new(AnchorPurpose::Batch, scope_hash);
    let mut scheduler = BatchScheduler::new(policy);
    let mut availability_ns = vec![0u64; arrivals.len()];
    let mut num_batches = 0usize;
    let mut sign_total_ns = 0u128;
    let mut all_verified = true;

    let mut record = |closed: ClosedBatch| {
        for qi in &closed.member_query_indices {
            availability_ns[*qi] = closed.available_at_ns;
        }
        all_verified &= closed.verified;
        num_batches += 1;
        sign_total_ns += closed.sign_elapsed_ns;
    };

    let mut i = 0usize;
    while i < arrivals.len() {
        let deadline = policy
            .max_wait_ns
            .and_then(|w| scheduler.oldest_pending_arrival_ns().map(|t| t + w));
        let next_arrival_ns = arrivals[i].arrived_at_ns;

        if let Some(d) = deadline {
            if d <= next_arrival_ns {
                let members = scheduler
                    .close_on_timeout()
                    .expect("deadline implies pending members");
                record(sign_and_verify_batch(&members, d, roots, issuer, context));
                continue; // re-evaluate deadline vs. same next arrival
            }
        }

        let a = arrivals[i];
        i += 1;
        if let Some(members) = scheduler.arrive(a.query_index, a.arrived_at_ns) {
            record(sign_and_verify_batch(
                &members,
                a.arrived_at_ns,
                roots,
                issuer,
                context,
            ));
        }
    }

    // End of stream: flush whatever partial batch remains. Its close time
    // is the last arrival's time — the earliest a real deployment could
    // know no more queries are coming in this simulation window. This is a
    // simulation-boundary artifact, not a claim about production shutdown
    // behavior (see Limitations in the nightly report).
    if let Some(members) = scheduler.flush() {
        let close_at_ns = arrivals.last().map(|a| a.arrived_at_ns).unwrap_or(0);
        record(sign_and_verify_batch(
            &members,
            close_at_ns,
            roots,
            issuer,
            context,
        ));
    }

    let mut latencies: Vec<u64> = arrivals
        .iter()
        .map(|a| availability_ns[a.query_index].saturating_sub(a.arrived_at_ns))
        .collect();
    latencies.sort_unstable();

    let latency_mean_ns = latencies.iter().map(|&l| l as f64).sum::<f64>() / latencies.len() as f64;

    RegimeStats {
        regime,
        policy: policy_name,
        num_queries: arrivals.len(),
        num_batches,
        mean_batch_size: arrivals.len() as f64 / num_batches as f64,
        latency_mean_ns,
        latency_p50_ns: percentile(&latencies, 50.0),
        latency_p95_ns: percentile(&latencies, 95.0),
        latency_p99_ns: percentile(&latencies, 99.0),
        latency_max_ns: latencies.last().copied().unwrap_or(0),
        sign_amortized_ns: sign_total_ns as f64 / arrivals.len() as f64,
        all_batches_verified: all_verified,
    }
}

fn fmt_ms(ns: f64) -> String {
    format!("{:.3}ms", ns / NS_PER_MS as f64)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let dims: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let num_queries: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3000);

    println!("=== ruvector-retrieval-receipt batch-fill latency simulation ===");
    println!("n={n} dims={dims} k={k} queries_per_regime={num_queries}");

    let index = RetrievalIndex::ingest(n, dims, 0xC0FF_EE01_D00D);
    assert!(index.verify_write_history(), "write history must verify");
    let index_root = index.index_state_root();
    let queries = synthetic_queries(num_queries.max(256), dims, 0xA5A5_5A5A_1111);
    let issuer = Issuer::generate();

    let policies: [(&str, BatchFillPolicy); 3] = [
        ("B1_baseline", BatchFillPolicy::fixed_size(1)),
        ("B32_fixed_only", BatchFillPolicy::fixed_size(32)),
        (
            "B32_hybrid_50ms",
            BatchFillPolicy::hybrid(32, 50 * NS_PER_MS),
        ),
    ];

    // Regimes chosen to cover: (1) load high enough to fill a 32-batch far
    // inside a 50ms window, (2) load low enough that a 32-batch cannot fill
    // within 50ms (mean gap 20ms x 32 = 640ms), forcing the hybrid policy's
    // timeout to fire, and (3) bursty on/off traffic, the realistic shape of
    // agent tool-call query streams rather than a smooth Poisson rate.
    let regimes: Vec<(&'static str, Vec<Arrival>)> = vec![
        (
            "target_load_1000qps",
            poisson_arrivals(num_queries, 1000.0, 0x1111_2222_3333_4444),
        ),
        (
            "light_load_50qps",
            poisson_arrivals(num_queries, 50.0, 0x5555_6666_7777_8888),
        ),
        (
            "bursty_on2000qps_off400ms",
            bursty_arrivals(
                num_queries,
                2000.0,
                100 * NS_PER_MS,
                400 * NS_PER_MS,
                0x9999_AAAA_BBBB_CCCC,
            ),
        ),
    ];

    let mut all_stats = Vec::new();
    for (regime_name, arrivals) in &regimes {
        // Real receipt roots, computed once per regime and reused across
        // every policy tested against it — real RetrievalIndex search +
        // real MerkleReceipt construction, not synthetic hashes.
        let roots: Vec<[u8; 32]> = arrivals
            .iter()
            .map(|a| {
                let q = &queries[a.query_index % queries.len()];
                let results = index.search(q, k);
                let receipt = RetrievalReceipt::build(
                    ReceiptVariant::Merkle,
                    query_hash(q),
                    index_root,
                    &results,
                );
                receipt.root().expect("Merkle receipt always has a root")
            })
            .collect();

        for (policy_name, policy) in &policies {
            let stats = run_policy(
                regime_name,
                policy_name,
                *policy,
                arrivals,
                &roots,
                &issuer,
                index_root,
            );
            all_stats.push(stats);
        }
    }

    println!(
        "\n{:<28} {:<18} {:>7} {:>9} {:>9} {:>12} {:>12} {:>12} {:>12} {:>12} {:>16} {:>10}",
        "regime",
        "policy",
        "queries",
        "batches",
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
            "{:<28} {:<18} {:>7} {:>9} {:>9.1} {:>12} {:>12} {:>12} {:>12} {:>12} {:>16.1} {:>10}",
            s.regime,
            s.policy,
            s.num_queries,
            s.num_batches,
            s.mean_batch_size,
            fmt_ms(s.latency_mean_ns),
            fmt_ms(s.latency_p50_ns as f64),
            fmt_ms(s.latency_p95_ns as f64),
            fmt_ms(s.latency_p99_ns as f64),
            fmt_ms(s.latency_max_ns as f64),
            s.sign_amortized_ns,
            s.all_batches_verified
        );
    }

    // ── Acceptance evaluation (thresholds fixed before this run; see the
    // nightly research README for the formalized hypothesis) ──────────────
    let find = |regime: &str, policy: &str| -> &RegimeStats {
        all_stats
            .iter()
            .find(|s| s.regime == regime && s.policy == policy)
            .expect("regime/policy combination was run")
    };

    let hybrid_target = find("target_load_1000qps", "B32_hybrid_50ms");
    let fixed_target = find("target_load_1000qps", "B32_fixed_only");
    let hybrid_light = find("light_load_50qps", "B32_hybrid_50ms");
    let fixed_light = find("light_load_50qps", "B32_fixed_only");
    let hybrid_bursty = find("bursty_on2000qps_off400ms", "B32_hybrid_50ms");

    let all_verified = all_stats.iter().all(|s| s.all_batches_verified);

    // Bound = 50ms fill-timeout + generous slack for real sign cost and
    // simulation-boundary flush effects (sign cost is tens of microseconds,
    // three orders of magnitude below the bound; slack is a fixed 20ms, not
    // tuned post-hoc).
    let bound_ns = 50 * NS_PER_MS + 20 * NS_PER_MS;
    let hybrid_bounded = hybrid_target.latency_p99_ns <= bound_ns
        && hybrid_light.latency_p99_ns <= bound_ns
        && hybrid_bursty.latency_p99_ns <= bound_ns;

    let fixed_size_unbounded_at_light_load =
        fixed_light.latency_p99_ns > 2 * hybrid_light.latency_p99_ns.max(1);

    let amortization_preserved_at_target_load =
        hybrid_target.sign_amortized_ns <= 2.0 * fixed_target.sign_amortized_ns.max(1.0);

    println!("\n=== acceptance ===");
    println!(
        "all closed batches verify (signature + inclusion), every regime/policy: {all_verified}"
    );
    println!(
        "hybrid p99 latency bounded by {}: target={} light={} bursty={} -> {hybrid_bounded}",
        fmt_ms(bound_ns as f64),
        fmt_ms(hybrid_target.latency_p99_ns as f64),
        fmt_ms(hybrid_light.latency_p99_ns as f64),
        fmt_ms(hybrid_bursty.latency_p99_ns as f64)
    );
    println!(
        "fixed-size-only p99 at light load exceeds 2x hybrid's p99 (demonstrates the unbounded-tail failure mode): fixed={} hybrid={} -> {fixed_size_unbounded_at_light_load}",
        fmt_ms(fixed_light.latency_p99_ns as f64),
        fmt_ms(hybrid_light.latency_p99_ns as f64)
    );
    println!(
        "hybrid amortized signing cost at target load within 2x of fixed-size-only: hybrid={:.1}ns fixed={:.1}ns -> {amortization_preserved_at_target_load}",
        hybrid_target.sign_amortized_ns, fixed_target.sign_amortized_ns
    );

    let verdict = if all_verified
        && hybrid_bounded
        && fixed_size_unbounded_at_light_load
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
