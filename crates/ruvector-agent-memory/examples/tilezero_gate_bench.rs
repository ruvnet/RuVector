//! Nightly research benchmark (2026-11-10,
//! `docs/research/nightly/2026-11-10-tilezero-gated-agent-memory-admission/`):
//! measures whether routing `TransactionalLedger` acceptance through
//! `TileZeroGate` (content-inspecting, built on `cognitum-gate-tilezero`'s
//! `ThreeFilterDecision`) Denies/Defers a measurably higher fraction of
//! synthetically "incoherent" candidate entries than the two existing gates
//! (`AlwaysAdmitGate`, and `WriteGateAdapter`, both of which admit every
//! candidate regardless of content), while still Permitting coherent
//! entries at a rate not meaningfully worse than the baselines — and runs a
//! bounded ("Darwin-lite") `GateThresholds` parameter sweep.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --features tilezero-gate \
//!     --example tilezero_gate_bench
//!
//! ## Synthetic workload
//!
//! "Coherent" candidates are drawn from a small set of templated sentences
//! (shared structure, randomized fill-in numbers/topic words) — the kind of
//! repetitive, on-topic statement a well-behaved agent would actually write
//! to memory. "Incoherent" candidates are random printable-ASCII strings
//! with no template structure at all — a crude stand-in for
//! poisoning/injection/drift content that shares essentially no byte
//! n-grams with recent history. See `src/tilezero_gate.rs`'s module doc for
//! why this is a legitimate but narrowly-scoped test of the wiring (byte
//! n-gram separability), not a claim about real semantic-embedding-space
//! poisoning detection.
//!
//! `add()`'s cost is identical across every gate variant tested (it never
//! consults the `ProofGate`), so only `accept()` — the call that actually
//! invokes `ProofGate::admit` — is timed, to isolate the gate's own added
//! cost rather than diluting it with constant ledger bookkeeping.

use cognitum_gate_tilezero::GateThresholds;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    AlwaysAdmitGate, LedgerState, MemoryWitnessLog, ProofGate, TileZeroGate, TransactionalLedger,
};
use std::time::{Duration, Instant};

const SEED: u64 = 42;
const N_WARM: usize = 64;
const N_TRIALS: usize = 500;
const WINDOW_SIZE: usize = 64;

const TOPICS: &[&str] = &["billing", "shipping", "inventory", "support", "security"];
const TEMPLATES: &[&str] = &[
    "memory entry {id:04} regarding {topic} status update for account {acct:05}",
    "routine log {id:04}: {topic} subsystem nominal, account {acct:05} checked",
    "agent observation {id:04} on {topic} queue depth for account {acct:05}",
    "scheduled note {id:04} -- {topic} reconciliation complete, account {acct:05}",
];

/// A templated, on-topic "coherent" statement: high byte n-gram overlap
/// with every other call's output, by construction.
fn coherent_content(rng: &mut StdRng) -> String {
    let t = TEMPLATES[rng.gen_range(0..TEMPLATES.len())];
    let topic = TOPICS[rng.gen_range(0..TOPICS.len())];
    let id: u32 = rng.gen_range(0..10_000);
    let acct: u32 = rng.gen_range(0..100_000);
    t.replace("{id:04}", &format!("{id:04}"))
        .replace("{topic}", topic)
        .replace("{acct:05}", &format!("{acct:05}"))
}

/// A random printable-ASCII "incoherent" statement: no shared template
/// structure with [`coherent_content`]'s output, simulating
/// poisoning/drift/injection content far from the ledger's recent cluster.
fn incoherent_content(rng: &mut StdRng) -> String {
    let len = rng.gen_range(20..80);
    (0..len)
        .map(|_| char::from(rng.gen_range(33u8..=126u8)))
        .collect()
}

/// Pre-generate the fixed warm-up + trial streams once, from one seeded
/// RNG, so every gate variant and every Darwin-lite candidate sees
/// byte-identical input (apples-to-apples comparison, and a prerequisite
/// for the reproducibility check below).
struct Streams {
    warm: Vec<String>,
    coherent_trials: Vec<String>,
    incoherent_trials: Vec<String>,
}

fn generate_streams(seed: u64) -> Streams {
    let mut rng = StdRng::seed_from_u64(seed);
    let warm = (0..N_WARM).map(|_| coherent_content(&mut rng)).collect();
    let coherent_trials = (0..N_TRIALS).map(|_| coherent_content(&mut rng)).collect();
    let incoherent_trials = (0..N_TRIALS)
        .map(|_| incoherent_content(&mut rng))
        .collect();
    Streams {
        warm,
        coherent_trials,
        incoherent_trials,
    }
}

struct ConditionResult {
    permitted: usize,
    refused: usize,
    warm_permitted: usize,
    lat_ns: Vec<u64>,
    elapsed: Duration,
    any_non_finite: bool,
}

/// Warm a fresh ledger (gate `gate`, sink `MemoryWitnessLog`) with
/// `streams.warm`, then run `trials` through `add` + (timed) `accept`,
/// recording the admission outcome and `accept()`'s own wall latency.
fn run_condition<G: ProofGate>(gate: G, warm: &[String], trials: &[String]) -> ConditionResult {
    let mut ledger = TransactionalLedger::new(MemoryWitnessLog::default(), gate);
    let mut warm_permitted = 0usize;
    for w in warm {
        if let Ok(id) = ledger.add(w.clone(), &[], "warm", "warm-up") {
            if ledger.accept(id, "warm", "warm-up").is_ok() {
                warm_permitted += 1;
            }
        }
    }

    let mut permitted = 0usize;
    let mut refused = 0usize;
    let mut lat_ns = Vec::with_capacity(trials.len());
    let mut any_non_finite = false;
    let t0 = Instant::now();
    for (i, content) in trials.iter().enumerate() {
        let id = ledger
            .add(content.clone(), &[], "bench", &format!("trial {i}"))
            .expect("fresh id, no deps, add never fails here");
        let op0 = Instant::now();
        let res = ledger.accept(id, "bench", "trial");
        lat_ns.push(op0.elapsed().as_nanos() as u64);
        match res {
            Ok(_receipt) => permitted += 1,
            Err(_) => refused += 1,
        }
        if ledger.state_of(id) != Some(LedgerState::Accepted)
            && ledger.state_of(id) != Some(LedgerState::Pending)
        {
            any_non_finite = true; // should never happen; flags a bug if it does
        }
    }
    let elapsed = t0.elapsed();
    ConditionResult {
        permitted,
        refused,
        warm_permitted,
        lat_ns,
        elapsed,
        any_non_finite,
    }
}

fn percentile(sorted_ns: &[u64], p: f64) -> f64 {
    if sorted_ns.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_ns.len() - 1) as f64 * p).round() as usize;
    sorted_ns[idx] as f64 / 1000.0 // microseconds
}

fn report_latency_row(label: &str, r: &ConditionResult) {
    let mut sorted = r.lat_ns.clone();
    sorted.sort_unstable();
    let mean: f64 = sorted.iter().sum::<u64>() as f64 / sorted.len().max(1) as f64 / 1000.0;
    let throughput = sorted.len() as f64 / r.elapsed.as_secs_f64();
    println!(
        "{:<28} permit={:4} refused={:4} warm_permit={:3}/{:<3} mean={:>8.3}us p50={:>8.3}us p95={:>8.3}us p99={:>8.3}us throughput={:>10.1} ops/s",
        label,
        r.permitted,
        r.refused,
        r.warm_permitted,
        N_WARM,
        mean,
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.95),
        percentile(&sorted, 0.99),
        throughput,
    );
}

// ── Darwin-lite GateThresholds sweep ─────────────────────────────────────────

#[derive(Clone, Debug)]
struct Bounds {
    tau_deny: (f64, f64),
    tau_permit: (f64, f64),
    min_cut: (f64, f64),
    max_shift: (f64, f64),
    theta_uncertainty: (f64, f64),
    theta_confidence: (f64, f64),
}

const BOUNDS: Bounds = Bounds {
    tau_deny: (0.001, 10.0),
    tau_permit: (10.0, 150.0),
    min_cut: (0.5, 9.5),
    max_shift: (0.05, 0.95),
    theta_uncertainty: (1.0, 50.0),
    theta_confidence: (0.5, 20.0),
};

fn clamp(v: f64, bounds: (f64, f64)) -> f64 {
    v.max(bounds.0).min(bounds.1)
}

/// Mutate every field of `base` by an independent relative jitter in
/// `[-30%, +30%]`, clamped into [`BOUNDS`]. Simple, fully deterministic
/// given `rng`'s state — see the nightly doc for why this (not a more
/// elaborate crossover/selection scheme) is an adequate "Darwin-lite" for a
/// 6-dimensional, 12-candidate bounded search.
fn mutate(base: &GateThresholds, rng: &mut StdRng) -> GateThresholds {
    let jitter = |v: f64, rng: &mut StdRng| v * (1.0 + rng.gen_range(-0.3..0.3));
    GateThresholds {
        tau_deny: clamp(jitter(base.tau_deny, rng), BOUNDS.tau_deny),
        tau_permit: clamp(jitter(base.tau_permit, rng), BOUNDS.tau_permit),
        min_cut: clamp(jitter(base.min_cut, rng), BOUNDS.min_cut),
        max_shift: clamp(jitter(base.max_shift, rng), BOUNDS.max_shift),
        permit_ttl_ns: base.permit_ttl_ns,
        theta_uncertainty: clamp(
            jitter(base.theta_uncertainty, rng),
            BOUNDS.theta_uncertainty,
        ),
        theta_confidence: clamp(jitter(base.theta_confidence, rng), BOUNDS.theta_confidence),
    }
}

struct CandidateMetrics {
    label: String,
    thresholds: GateThresholds,
    coherent_permit_rate: f64,
    incoherent_permit_rate: f64,
    separation: f64,
    mean_latency_ns: f64,
    throughput_ops_s: f64,
    correctness: f64,
    reproducibility: f64,
}

fn well_formed(t: &GateThresholds) -> bool {
    t.tau_permit > t.tau_deny
        && t.min_cut > 0.0
        && t.max_shift > 0.0
        && t.tau_deny.is_finite()
        && t.tau_permit.is_finite()
        && t.min_cut.is_finite()
        && t.max_shift.is_finite()
}

/// Evaluate one `GateThresholds` config against the fixed streams, once.
fn evaluate_candidate(
    label: &str,
    thresholds: GateThresholds,
    streams: &Streams,
) -> CandidateMetrics {
    let coherent = run_condition(
        TileZeroGate::with_window_size(thresholds.clone(), WINDOW_SIZE),
        &streams.warm,
        &streams.coherent_trials,
    );
    let incoherent = run_condition(
        TileZeroGate::with_window_size(thresholds.clone(), WINDOW_SIZE),
        &streams.warm,
        &streams.incoherent_trials,
    );
    let coherent_permit_rate = coherent.permitted as f64 / N_TRIALS as f64;
    let incoherent_permit_rate = incoherent.permitted as f64 / N_TRIALS as f64;
    let total_trials = (coherent.lat_ns.len() + incoherent.lat_ns.len()) as f64;
    let total_ns: u64 = coherent.lat_ns.iter().chain(incoherent.lat_ns.iter()).sum();
    let mean_latency_ns = total_ns as f64 / total_trials;
    let combined_elapsed = coherent.elapsed + incoherent.elapsed;
    let throughput_ops_s = total_trials / combined_elapsed.as_secs_f64();

    // Correctness hard gate. Beyond plain numeric well-formedness, this
    // sweep empirically discovered a real trap: `ReducedGraph::new()`'s
    // cold-start default (`aggregate_evidence = 100.0`) only clears the
    // evidence filter's `>= tau_permit` check while `tau_permit <= 100.0`.
    // Any config with `tau_permit > 100.0` makes even a config's OWN
    // warm-up admissions Defer at cold start, so the sliding window never
    // fills and the gate permanently defers every admission on both
    // streams (cheaply, since an empty window skips the cosine-similarity
    // scan — which is exactly why such a config would otherwise look
    // attractive on the latency/throughput terms alone). A gate that
    // cannot even admit its own warm-up is not a working admission gate,
    // so `warm_permitted == 0` is treated as a correctness failure,
    // matching the pre-registered "all tests still pass with this config"
    // hard gate in spirit (see the nightly doc's Darwin-lite section for
    // why this was added after, not before, observing it).
    let correctness = if well_formed(&thresholds)
        && !coherent.any_non_finite
        && !incoherent.any_non_finite
        && coherent.warm_permitted > 0
        && incoherent.warm_permitted > 0
    {
        1.0
    } else {
        0.0
    };

    // Reproducibility: re-run the exact same config against the exact same
    // streams and confirm identical decision counts.
    let coherent2 = run_condition(
        TileZeroGate::with_window_size(thresholds.clone(), WINDOW_SIZE),
        &streams.warm,
        &streams.coherent_trials,
    );
    let incoherent2 = run_condition(
        TileZeroGate::with_window_size(thresholds.clone(), WINDOW_SIZE),
        &streams.warm,
        &streams.incoherent_trials,
    );
    let reproducibility = if coherent2.permitted == coherent.permitted
        && coherent2.refused == coherent.refused
        && incoherent2.permitted == incoherent.permitted
        && incoherent2.refused == incoherent.refused
    {
        1.0
    } else {
        0.0
    };

    CandidateMetrics {
        label: label.to_string(),
        thresholds,
        coherent_permit_rate,
        incoherent_permit_rate,
        separation: coherent_permit_rate - incoherent_permit_rate,
        mean_latency_ns,
        throughput_ops_s,
        correctness,
        reproducibility,
    }
}

/// Min-max normalize `metric` to `[0, 1]` across `pool` ("higher better" if
/// `higher_is_better`); a degenerate (constant) population normalizes to
/// `1.0` for every member (stated explicitly rather than dividing by zero).
fn normalize(values: &[f64], higher_is_better: bool) -> Vec<f64> {
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-12 {
        return values.iter().map(|_| 1.0).collect();
    }
    values
        .iter()
        .map(|&v| {
            let n = (v - min) / (max - min);
            if higher_is_better {
                n
            } else {
                1.0 - n
            }
        })
        .collect()
}

fn run_darwin_lite(streams: &Streams) -> Vec<CandidateMetrics> {
    let default = GateThresholds::default();
    let mut all = vec![evaluate_candidate("default", default.clone(), streams)];

    let mut rng = StdRng::seed_from_u64(1337);
    let mut best_so_far = default.clone();
    let mut best_fitness_so_far = f64::NEG_INFINITY;

    for gen in 1..=3 {
        let mut gen_candidates = Vec::new();
        for c in 1..=4 {
            let thresholds = mutate(&best_so_far, &mut rng);
            let label = format!("gen{gen}_cand{c}");
            gen_candidates.push(evaluate_candidate(&label, thresholds, streams));
        }
        // Normalize fitness within the running population seen so far
        // (default + every candidate evaluated up to and including this
        // generation) so each generation's elitist pick is comparable
        // across the whole sweep, not just within one generation.
        all.extend(gen_candidates);
        let fitness = compute_fitness(&all);
        for (m, f) in all.iter().zip(fitness.iter()) {
            if m.correctness > 0.5 && *f > best_fitness_so_far {
                best_fitness_so_far = *f;
                best_so_far = m.thresholds.clone();
            }
        }
    }
    all
}

/// `fitness = 0.30*separation + 0.25*latency + 0.15*throughput +
/// 0.10*memory + 0.10*correctness + 0.10*reproducibility`, each term (other
/// than the two 0/1 hard-gate terms) min-max normalized across `pool`.
/// `memory` is constant across every candidate here (see the nightly doc:
/// `GateThresholds` does not change `TileZeroGate`'s allocations —
/// `window_size`/`TILEZERO_EMBED_DIM` are fixed, not part of this sweep —
/// so every candidate's approximate per-admission memory footprint is
/// identical: `window_size * TILEZERO_EMBED_DIM * size_of::<f32>()` bytes
/// for the sliding window plus one stack-allocated `ReducedGraph`), so its
/// normalized term is `1.0` for every candidate by the degenerate-population
/// rule above; it is included in the formula for fidelity to the
/// pre-registered spec, not because it discriminates among candidates here.
fn compute_fitness(pool: &[CandidateMetrics]) -> Vec<f64> {
    let sep: Vec<f64> = pool.iter().map(|m| m.separation).collect();
    let lat: Vec<f64> = pool.iter().map(|m| m.mean_latency_ns).collect();
    let thr: Vec<f64> = pool.iter().map(|m| m.throughput_ops_s).collect();
    let mem: Vec<f64> = pool.iter().map(|_| 1.0).collect(); // constant, see doc above

    let sep_n = normalize(&sep, true);
    let lat_n = normalize(&lat, false);
    let thr_n = normalize(&thr, true);
    let mem_n = normalize(&mem, true);

    pool.iter()
        .enumerate()
        .map(|(i, m)| {
            if m.correctness < 0.5 {
                return 0.0; // hard-rejected regardless of other terms
            }
            0.30 * sep_n[i]
                + 0.25 * lat_n[i]
                + 0.15 * thr_n[i]
                + 0.10 * mem_n[i]
                + 0.10 * m.correctness
                + 0.10 * m.reproducibility
        })
        .collect()
}

fn main() {
    println!("ruvector-agent-memory TileZeroGate admission benchmark");
    println!(
        "SEED={SEED} N_WARM={N_WARM} N_TRIALS={N_TRIALS} WINDOW_SIZE={WINDOW_SIZE} EMBED_DIM={}\n",
        ruvector_agent_memory::TILEZERO_EMBED_DIM
    );

    let streams = generate_streams(SEED);

    println!("== Baseline vs candidate, default GateThresholds ==");
    let always_coherent = run_condition(
        AlwaysAdmitGate::default(),
        &streams.warm,
        &streams.coherent_trials,
    );
    let always_incoherent = run_condition(
        AlwaysAdmitGate::default(),
        &streams.warm,
        &streams.incoherent_trials,
    );
    let tz_coherent = run_condition(
        TileZeroGate::with_window_size(GateThresholds::default(), WINDOW_SIZE),
        &streams.warm,
        &streams.coherent_trials,
    );
    let tz_incoherent = run_condition(
        TileZeroGate::with_window_size(GateThresholds::default(), WINDOW_SIZE),
        &streams.warm,
        &streams.incoherent_trials,
    );
    report_latency_row("always_admit / coherent", &always_coherent);
    report_latency_row("always_admit / incoherent", &always_incoherent);
    report_latency_row("tilezero_default / coherent", &tz_coherent);
    report_latency_row("tilezero_default / incoherent", &tz_incoherent);

    let always_incoherent_deny_defer_rate = always_incoherent.refused as f64 / N_TRIALS as f64;
    let tz_incoherent_deny_defer_rate = tz_incoherent.refused as f64 / N_TRIALS as f64;
    let tz_coherent_permit_rate = tz_coherent.permitted as f64 / N_TRIALS as f64;
    let deny_defer_delta_pp =
        (tz_incoherent_deny_defer_rate - always_incoherent_deny_defer_rate) * 100.0;

    println!("\n== Pre-registered acceptance computation (default GateThresholds) ==");
    println!(
        "  Criterion 1: TileZeroGate incoherent Deny+Defer rate ({:.1}%) - AlwaysAdmitGate incoherent Deny+Defer rate ({:.1}%) = {:.1}pp (need >= 50.0pp) -> {}",
        tz_incoherent_deny_defer_rate * 100.0,
        always_incoherent_deny_defer_rate * 100.0,
        deny_defer_delta_pp,
        if deny_defer_delta_pp >= 50.0 { "PASS" } else { "FAIL" }
    );
    println!(
        "  Criterion 2: TileZeroGate coherent Permit rate = {:.1}% (need >= 85.0%) -> {}",
        tz_coherent_permit_rate * 100.0,
        if tz_coherent_permit_rate >= 0.85 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let added_latency_mean_us: f64 = {
        let tz_mean: f64 = tz_coherent
            .lat_ns
            .iter()
            .chain(tz_incoherent.lat_ns.iter())
            .sum::<u64>() as f64
            / (2 * N_TRIALS) as f64
            / 1000.0;
        let baseline_mean: f64 = always_coherent
            .lat_ns
            .iter()
            .chain(always_incoherent.lat_ns.iter())
            .sum::<u64>() as f64
            / (2 * N_TRIALS) as f64
            / 1000.0;
        tz_mean - baseline_mean
    };
    println!(
        "  Criterion 5 (informational, no threshold): mean added per-admission latency = {:.3}us (TileZeroGate accept() vs AlwaysAdmitGate accept(), pooled over both streams)",
        added_latency_mean_us
    );

    println!("\n== Darwin-lite GateThresholds sweep (3 generations x 4 candidates + default = 13 configs) ==");
    let lineage = run_darwin_lite(&streams);
    let fitness = compute_fitness(&lineage);
    println!(
        "{:<12} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}   {:>10} {:>10} {:>9} {:>6} {:>5} {:>5} {:>8}",
        "label",
        "tau_deny",
        "tau_perm",
        "min_cut",
        "max_shft",
        "th_unc",
        "th_conf",
        "coh_perm%",
        "inc_perm%",
        "sep",
        "lat_us",
        "corr",
        "repr",
        "fitness"
    );
    let mut best_idx = 0usize;
    for (i, (m, f)) in lineage.iter().zip(fitness.iter()).enumerate() {
        println!(
            "{:<12} {:>9.4} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3}   {:>10.1} {:>10.1} {:>9.3} {:>6.2} {:>5.1} {:>5.1} {:>8.4}",
            m.label,
            m.thresholds.tau_deny,
            m.thresholds.tau_permit,
            m.thresholds.min_cut,
            m.thresholds.max_shift,
            m.thresholds.theta_uncertainty,
            m.thresholds.theta_confidence,
            m.coherent_permit_rate * 100.0,
            m.incoherent_permit_rate * 100.0,
            m.separation,
            m.mean_latency_ns / 1000.0,
            m.correctness,
            m.reproducibility,
            f,
        );
        if *f > fitness[best_idx] {
            best_idx = i;
        }
    }
    let default_fitness = fitness[0];
    let best = &lineage[best_idx];
    println!(
        "\nDefault fitness = {default_fitness:.4}; best-in-sweep = '{}' fitness = {:.4}",
        best.label, fitness[best_idx]
    );
    let promote = best_idx != 0
        && fitness[best_idx] > default_fitness
        && best.correctness > 0.5
        && best.reproducibility > 0.5;
    if promote {
        println!(
            "PROMOTION: '{}' strictly beats the hand-picked default on fitness and passes every hard constraint -- recommend as the new default.",
            best.label
        );
    } else {
        println!("PROMOTION: no Darwin-lite candidate strictly improved on the hand-picked default; default GateThresholds is retained.");
    }

    println!("\n== Reproducibility spot-check (re-run default config twice) ==");
    let rerun_coherent = run_condition(
        TileZeroGate::with_window_size(GateThresholds::default(), WINDOW_SIZE),
        &streams.warm,
        &streams.coherent_trials,
    );
    let rerun_incoherent = run_condition(
        TileZeroGate::with_window_size(GateThresholds::default(), WINDOW_SIZE),
        &streams.warm,
        &streams.incoherent_trials,
    );
    let reproducible = rerun_coherent.permitted == tz_coherent.permitted
        && rerun_coherent.refused == tz_coherent.refused
        && rerun_incoherent.permitted == tz_incoherent.permitted
        && rerun_incoherent.refused == tz_incoherent.refused;
    println!(
        "  First run:  coherent permit={} refused={}  incoherent permit={} refused={}",
        tz_coherent.permitted, tz_coherent.refused, tz_incoherent.permitted, tz_incoherent.refused
    );
    println!(
        "  Second run: coherent permit={} refused={}  incoherent permit={} refused={}",
        rerun_coherent.permitted,
        rerun_coherent.refused,
        rerun_incoherent.permitted,
        rerun_incoherent.refused
    );
    println!(
        "  Reproducible (identical decision counts): {}",
        if reproducible { "PASS" } else { "FAIL" }
    );
}
