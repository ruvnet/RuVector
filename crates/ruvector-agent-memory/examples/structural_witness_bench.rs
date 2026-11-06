//! Nightly research benchmark (2026-10-12): signed structural forgetting
//! receipts. Composes ADR-345 (mincut-gated forgetting) + ADR-347 (Ed25519
//! `WitnessSigner`) into a new eviction-witness payload — see
//! `src/structural_witness.rs` module docs and
//! `docs/research/nightly/2026-10-12_signed-structural-forgetting-receipts/README.md`.
//!
//! Pre-registered hypothesis (fixed before this run; do not edit after
//! reading results):
//!
//! Given `MincutGatedForgetting`'s eviction witnesses carry no structural
//! justification (`aux = 0`) today,
//!
//! when evictions are annotated via `compact_witnessed_structural` (mincut
//! partition computed exactly once per decision, identical cost to plain
//! `compact_witnessed` on the same policy) and optionally routed through
//! `SignedWitnessSink`,
//!
//! then (1) `audit_structural_evictions` run over only the resulting witness
//! records (no vectors, no `MemoryStore`) recovers the exact ground-truth
//! boundary-eviction count in every trial; (2) the structural-encoding
//! overhead vs. plain `compact_witnessed` on the same decision stays within
//! 10% mean wall-clock; (3) a naive (non-recomputing) tamper of the boundary
//! bit is caught by the existing unsigned chain walk in every trial, while a
//! diligent (fully forward-recomputed) tamper is NOT caught by the unsigned
//! walk but IS caught by `verify_signed_chain` in every trial,
//!
//! subject to: zero false negatives/positives in the audit-accuracy and
//! forgery-detection checks across all trials; every number below comes from
//! this exact `cargo run --release` execution.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory --example structural_witness_bench --features mincut-forget

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    audit_structural_evictions, compact_witnessed, compact_witnessed_structural, is_hard_mode,
    verify_signed_chain, CoherenceWeights, EvictionWitnessChain, MemoryStore, MemoryWitnessLog,
    MincutGatedForgetting, SignedWitnessSink, SigningStrategy, AUX_BOUNDARY_BIT,
};
use rvf_types::ed25519::Ed25519Keypair;
use std::collections::HashSet;
use std::time::{Duration, Instant};

// ── Dataset parameters (same shape as the 2026-09-05 mincut_gated_forgetting
// nightly's corpus, which measured RuVectorGraphAnalyzer::partition() cost
// at this scale; reused rather than re-deriving) ──────────────────────────
const N_CLUSTERS: usize = 6;
const PER_CLUSTER: usize = 12;
const N_CORE: usize = N_CLUSTERS * PER_CLUSTER; // 72
const N_BRIDGES: usize = 12;
const N_MEMORIES: usize = N_CORE + N_BRIDGES; // 84
const DIMS: usize = 32;
const TARGET_SIZE: usize = N_MEMORIES / 2; // 42, 50% compaction
const STRUCTURAL_BONUS: f32 = 0.5;
const MINCUT_TRIALS: usize = 1; // see 2026-09-05 nightly: retries re-pay full mincut cost

const N_LATENCY_TRIALS: usize = 20;
const N_FORGERY_TRIALS: usize = 10;
const MAX_STRUCTURAL_OVERHEAD_FRACTION: f64 = 0.10; // acceptance gate 2

const SECRET: [u8; 32] = [19u8; 32];
const SIGN_BATCH_SIZE: usize = 16;

fn unit_gaussian(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}
fn add_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}
fn scale_vec(v: &[f32], s: f32) -> Vec<f32> {
    v.iter().map(|x| x * s).collect()
}
fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}
fn perturb(centroid: &[f32], noise: f32, rng: &mut StdRng) -> Vec<f32> {
    let n = unit_gaussian(rng, centroid.len());
    normalize_vec(&add_vecs(centroid, &scale_vec(&n, noise)))
}
fn midpoint(a: &[f32], b: &[f32]) -> Vec<f32> {
    normalize_vec(&add_vecs(a, b))
}

/// Deterministic corpus: `N_CLUSTERS` tight clusters plus `N_BRIDGES`
/// cross-cluster interpolated "bridge" memories, mirroring the 2026-09-05
/// nightly's dataset exactly (same constants) so its measured mincut cost
/// transfers to this run.
fn generate_corpus(seed: u64) -> MemoryStore {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut store = MemoryStore::new(DIMS);
    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| unit_gaussian(&mut rng, DIMS))
        .collect();
    for centroid in &centroids {
        for _ in 0..PER_CLUSTER {
            store.insert(perturb(centroid, 0.35, &mut rng));
        }
    }
    for _ in 0..N_BRIDGES {
        let a = rng.gen_range(0..N_CLUSTERS);
        let mut b = rng.gen_range(0..N_CLUSTERS);
        while b == a {
            b = rng.gen_range(0..N_CLUSTERS);
        }
        let mid = midpoint(&centroids[a], &centroids[b]);
        store.insert(perturb(&mid, 0.15, &mut rng));
    }
    store
}

fn policy() -> MincutGatedForgetting {
    let mut p = MincutGatedForgetting::soft(CoherenceWeights::default(), STRUCTURAL_BONUS);
    p.mincut_trials = MINCUT_TRIALS;
    p
}

struct LatencyRow {
    label: &'static str,
    mean_us: f64,
    total: Duration,
}

/// Time `N_LATENCY_TRIALS` independent compaction calls (fresh corpus each
/// trial, same seed sequence for both variants so the mincut partition cost
/// paid is as close to identical as the documented `partition()`
/// non-determinism (2026-09-05/09-11 nightlies) allows) and return the mean
/// per-call wall-clock.
fn measure_latency<F: Fn(&mut MemoryStore, &MincutGatedForgetting) -> usize>(
    label: &'static str,
    base_seed: u64,
    run: F,
) -> LatencyRow {
    let mut micros = Vec::with_capacity(N_LATENCY_TRIALS);
    let t0 = Instant::now();
    for trial in 0..N_LATENCY_TRIALS {
        let mut store = generate_corpus(base_seed + trial as u64);
        let p = policy();
        let call_t0 = Instant::now();
        let n = run(&mut store, &p);
        micros.push(call_t0.elapsed().as_micros() as u64);
        assert_eq!(
            n,
            N_MEMORIES - TARGET_SIZE,
            "every trial evicts the same count"
        );
    }
    let total = t0.elapsed();
    let mean_us = micros.iter().sum::<u64>() as f64 / micros.len() as f64;
    LatencyRow {
        label,
        mean_us,
        total,
    }
}

/// `decide()` exactly once, apply it via `compact_witnessed_structural`, and
/// hand back the resulting records plus the `ForgetDecision` that produced
/// them -- the decision IS the ground truth; nothing is recomputed
/// separately (see `structural_witness_tests.rs::decide_and_apply`, which
/// this mirrors, for why a second, independent `decide()` call would risk
/// disagreeing with the first).
fn decide_and_apply(
    store: &mut MemoryStore,
    p: &MincutGatedForgetting,
    seed: u64,
    sink: &mut dyn ruvector_agent_memory::WitnessSink,
) -> (
    Vec<ruvector_agent_memory::LedgerWitnessRecord>,
    ruvector_agent_memory::ForgetDecision,
) {
    let mut chain = EvictionWitnessChain::new();
    let entries = store.entries().to_vec();
    let decision = p.decide(&entries, TARGET_SIZE, &[]);
    let records = compact_witnessed_structural(
        store,
        &decision,
        is_hard_mode(p.mode),
        "bench",
        seed,
        &mut chain,
        sink,
    )
    .expect("structural compaction succeeds");
    (records, decision)
}

/// Ground truth IS the `ForgetDecision` that produced these exact records
/// (see `decide_and_apply`) -- compare it against what
/// `audit_structural_evictions` (the actual auditor-facing function under
/// test) recovers from the witness records alone, with no vectors and no
/// store.
fn audit_accuracy_trial(seed: u64) -> bool {
    let mut store = generate_corpus(seed);
    let p = policy();
    let entries = store.entries().to_vec();
    let mut log = MemoryWitnessLog::default();
    let (records, decision) = decide_and_apply(&mut store, &p, seed, &mut log);
    assert!(log.verify_chain());

    let survivor_set: HashSet<usize> = decision.survivors.iter().copied().collect();
    let ground_truth_boundary_count = (0..entries.len())
        .filter(|i| !survivor_set.contains(i) && decision.boundary.contains(i))
        .count();
    let ground_truth_ids: HashSet<u64> = (0..entries.len())
        .filter(|i| !survivor_set.contains(i) && decision.boundary.contains(i))
        .map(|i| entries[i].id)
        .collect();

    let report = audit_structural_evictions(&records);
    let recovered_ids: HashSet<u64> = records
        .iter()
        .filter(|r| ruvector_agent_memory::decode_structural_aux(r.aux).0)
        .map(|r| r.target_object_id as u64)
        .collect();

    report.evicted == records.len()
        && report.boundary_evicted == ground_truth_boundary_count
        && recovered_ids == ground_truth_ids
}

/// Naive tamper (flip the boundary bit on one evicted record, nothing else)
/// must break the unsigned chain walk; a diligent tamper (flip + forward
/// recompute every downstream hash) must not, but `verify_signed_chain`
/// over the same honestly-signed spans must reject it.
fn forgery_trial(seed: u64) -> (bool, bool) {
    let mut store = generate_corpus(seed);
    let p = policy();
    let key = Ed25519Keypair::from_secret(&SECRET);
    let mut signed = SignedWitnessSink::from_keypair(
        MemoryWitnessLog::default(),
        &key,
        SigningStrategy::BatchTail {
            batch_size: SIGN_BATCH_SIZE,
        },
    )
    .unwrap();
    decide_and_apply(&mut store, &p, seed, &mut signed);
    signed.seal();

    let victim = 0usize; // first evicted record is deterministic across identical runs

    // Naive tamper on a plain clone of the inner log.
    let mut naive = signed.inner().clone();
    naive.records[victim].aux ^= AUX_BOUNDARY_BIT;
    let naive_caught = !naive.verify_chain();

    // Diligent tamper: flip + forward-recompute.
    let mut diligent = signed.inner().clone();
    diligent.records[victim].aux ^= AUX_BOUNDARY_BIT;
    diligent.records[victim].record_hash = diligent.records[victim].compute_record_hash();
    let mut prev = if victim == 0 {
        0
    } else {
        diligent.records[victim - 1].chain_hash()
    };
    for r in diligent.records.iter_mut().skip(victim) {
        r.prev_hash = prev;
        r.record_hash = r.compute_record_hash();
        prev = r.chain_hash();
    }
    diligent.committed_head = prev;
    let unsigned_fooled = diligent.verify_chain();
    let anchor = signed.anchor();
    let signed_rejects =
        verify_signed_chain(&diligent, signed.spans(), &signed.public_key(), &anchor).is_err();

    (naive_caught, unsigned_fooled && signed_rejects)
}

fn main() {
    println!("ruvector-agent-memory — Signed Structural Forgetting Receipts bench");
    println!(
        "corpus: {N_MEMORIES} memories ({N_CLUSTERS} clusters x {PER_CLUSTER} + {N_BRIDGES} bridges), \
         dims={DIMS}, target_size={TARGET_SIZE}, mincut_trials={MINCUT_TRIALS}\n"
    );

    let baseline = measure_latency("baseline(compact_witnessed)", 1_000, |store, p| {
        let mut chain = EvictionWitnessChain::new();
        let mut log = MemoryWitnessLog::default();
        compact_witnessed(store, p, TARGET_SIZE, &[], "bench", 0, &mut chain, &mut log)
            .unwrap()
            .len()
    });

    let candidate_a = measure_latency("candidate_A(structural, unsigned)", 1_000, |store, p| {
        let mut log = MemoryWitnessLog::default();
        decide_and_apply(store, p, 0, &mut log).0.len()
    });

    let candidate_b = measure_latency(
        "candidate_B(structural, signed BatchTail16)",
        1_000,
        |store, p| {
            let mut sink = SignedWitnessSink::from_keypair(
                MemoryWitnessLog::default(),
                &Ed25519Keypair::from_secret(&SECRET),
                SigningStrategy::BatchTail {
                    batch_size: SIGN_BATCH_SIZE,
                },
            )
            .unwrap();
            let n = decide_and_apply(store, p, 0, &mut sink).0.len();
            sink.seal();
            n
        },
    );

    println!("{:<42} {:>12} {:>14}", "Variant", "mean (us)", "total (ms)");
    println!("{}", "-".repeat(70));
    for row in [&baseline, &candidate_a, &candidate_b] {
        println!(
            "{:<42} {:>12.1} {:>14.3}",
            row.label,
            row.mean_us,
            row.total.as_secs_f64() * 1000.0
        );
    }
    println!();

    let structural_overhead = (candidate_a.mean_us - baseline.mean_us) / baseline.mean_us;
    let signing_overhead_ratio = candidate_b.mean_us / candidate_a.mean_us.max(1.0);
    println!(
        "Structural-encoding overhead (candidate_A vs baseline): {:+.2}%  (gate: <= {:.0}%)",
        structural_overhead * 100.0,
        MAX_STRUCTURAL_OVERHEAD_FRACTION * 100.0
    );
    println!(
        "Signing overhead (candidate_B vs candidate_A, BatchTail{{{SIGN_BATCH_SIZE}}}): {:.3}x (descriptive, no fixed gate)\n",
        signing_overhead_ratio
    );

    {
        let mut demo_store = generate_corpus(2_000);
        let mut demo_log = MemoryWitnessLog::default();
        let (demo_records, _decision) =
            decide_and_apply(&mut demo_store, &policy(), 2_000, &mut demo_log);
        let demo_report = audit_structural_evictions(&demo_records);
        println!(
            "Example StructuralAuditReport (seed=2000, no vectors/store involved): \
             evicted={} boundary_evicted={} hard_mode_evicted={}\n",
            demo_report.evicted, demo_report.boundary_evicted, demo_report.hard_mode_evicted
        );
    }

    println!("Audit-accuracy trials ({N_LATENCY_TRIALS} independently seeded corpora):");
    let mut audit_pass = 0usize;
    for trial in 0..N_LATENCY_TRIALS {
        if audit_accuracy_trial(2_000 + trial as u64) {
            audit_pass += 1;
        }
    }
    println!("  {audit_pass}/{N_LATENCY_TRIALS} trials: audited boundary set == ground truth\n");

    println!("Forgery-detection trials ({N_FORGERY_TRIALS} independently seeded corpora):");
    let mut naive_pass = 0usize;
    let mut diligent_pass = 0usize;
    for trial in 0..N_FORGERY_TRIALS {
        let (naive_caught, diligent_caught) = forgery_trial(3_000 + trial as u64);
        if naive_caught {
            naive_pass += 1;
        }
        if diligent_caught {
            diligent_pass += 1;
        }
    }
    println!("  naive tamper caught by unsigned walk    : {naive_pass}/{N_FORGERY_TRIALS}");
    println!("  diligent tamper: unsigned fooled AND signed chain rejects : {diligent_pass}/{N_FORGERY_TRIALS}\n");

    println!("Acceptance test");
    let gate_overhead = structural_overhead <= MAX_STRUCTURAL_OVERHEAD_FRACTION;
    let gate_audit = audit_pass == N_LATENCY_TRIALS;
    let gate_naive = naive_pass == N_FORGERY_TRIALS;
    let gate_diligent = diligent_pass == N_FORGERY_TRIALS;
    println!(
        "  Structural overhead <= {:.0}%          ({:+.2}%)         : {}",
        MAX_STRUCTURAL_OVERHEAD_FRACTION * 100.0,
        structural_overhead * 100.0,
        if gate_overhead { "PASS" } else { "FAIL" }
    );
    println!(
        "  Audit accuracy 100%                  ({audit_pass}/{N_LATENCY_TRIALS})           : {}",
        if gate_audit { "PASS" } else { "FAIL" }
    );
    println!(
        "  Naive tamper detection 100%           ({naive_pass}/{N_FORGERY_TRIALS})           : {}",
        if gate_naive { "PASS" } else { "FAIL" }
    );
    println!(
        "  Diligent forgery rejected by signing 100% ({diligent_pass}/{N_FORGERY_TRIALS})       : {}",
        if gate_diligent { "PASS" } else { "FAIL" }
    );

    let all_pass = gate_overhead && gate_audit && gate_naive && gate_diligent;
    if all_pass {
        println!(
            "\n=> ACCEPT: signed structural forgetting receipts are auditable, cheap to add, and \
             cryptographically tamper-evident against diligent forgery."
        );
    } else {
        println!("\n=> REJECT: one or more mandatory acceptance gates failed (see above).");
        std::process::exit(1);
    }
}
