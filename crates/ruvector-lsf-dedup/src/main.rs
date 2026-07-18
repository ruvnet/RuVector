//! Benchmark: LSF-based semantic deduplication across three strategies.
//!
//! Dataset: 2 000 synthetic 128-dim unit vectors.
//! Composition: 60 % unique, 25 % near-duplicates (cosine ≥ 0.95), 15 % exact clones.
//! Ground truth is derived from actual cosine similarity, not from the labels.
//!
//! Each strategy is run three times and the best elapsed time is kept.
//!
//! Acceptance criteria:
//!   SimHash  recall > 0.55, precision > 0.70
//!   MinHash  recall > 0.50, precision > 0.65
//!   Hybrid   recall > 0.70, precision > 0.85

use ruvector_lsf_dedup::{
    cosine_similarity,
    dedup_store::{DedupStats, DedupStore, HybridDedupStore},
    minhash::MinHasher,
    normalize,
    simhash::SimHasher,
    DedupMethod, DedupResult, SemanticFingerprinter,
};
use std::time::Instant;

const DIM: usize = 128;
const N_BASE: usize = 1_200; // unique base vectors
const NEAR_DUP_FRACTION: f32 = 0.25; // target fraction of near-dups in dataset
const EXACT_FRACTION: f32 = 0.15; // target fraction of exact clones
const COSINE_DUP_THRESHOLD: f32 = 0.93; // ground-truth duplicate boundary
const RUNS: usize = 3; // repeat runs to get stable timing

fn main() {
    print_header();
    let (vectors, labels) = build_dataset();
    let total = vectors.len();

    // Strategy 1: SimHash
    let (sim_stats, sim_ns) = run_simhash(&vectors, &labels);
    // Strategy 2: MinHash
    let (min_stats, min_ns) = run_minhash(&vectors, &labels);
    // Strategy 3: Hybrid (SimHash pre-filter + exact cosine)
    let (hyb_stats, hyb_ns) = run_hybrid(&vectors, &labels);

    print_results(
        total, &sim_stats, sim_ns, &min_stats, min_ns, &hyb_stats, hyb_ns,
    );
    acceptance_check(&sim_stats, &min_stats, &hyb_stats);
}

// ─── dataset ─────────────────────────────────────────────────────────────────

fn build_dataset() -> (Vec<Vec<f32>>, Vec<bool>) {
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<bool> = Vec::new(); // true = is a near-duplicate

    let near_count = (N_BASE as f32 * NEAR_DUP_FRACTION / (1.0 - NEAR_DUP_FRACTION)) as usize;
    let exact_count =
        (N_BASE as f32 * EXACT_FRACTION / (1.0 - NEAR_DUP_FRACTION - EXACT_FRACTION)) as usize;

    // unique base vectors
    for i in 0..N_BASE {
        let v = unit_vec(DIM, (i as u64) * 7919 + 1);
        vectors.push(v);
        labels.push(false);
    }

    // near-duplicates (noise 0.05)
    let nd_seed_offset = 1_000_000u64;
    for i in 0..near_count {
        let base_idx = i % N_BASE;
        let base = vectors[base_idx].clone();
        let v = perturb(&base, 0.05, nd_seed_offset + i as u64);
        vectors.push(v);
        labels.push(true);
    }

    // exact clones (noise 1e-5, effectively identical after normalisation)
    let ex_seed_offset = 2_000_000u64;
    for i in 0..exact_count {
        let base_idx = i % N_BASE;
        let base = vectors[base_idx].clone();
        let v = perturb(&base, 0.001, ex_seed_offset + i as u64);
        vectors.push(v);
        labels.push(true);
    }

    // shuffle deterministically
    let mut indices: Vec<usize> = (0..vectors.len()).collect();
    fisher_yates(&mut indices, 42);
    let shuffled_vecs: Vec<_> = indices.iter().map(|&i| vectors[i].clone()).collect();
    let shuffled_labels: Vec<_> = indices.iter().map(|&i| labels[i]).collect();

    // post-shuffle: recompute labels against actual cosine similarity
    // (labels based on perturb noise may slightly differ from cosine threshold)
    recompute_labels(&shuffled_vecs, &shuffled_labels)
}

fn recompute_labels(vectors: &[Vec<f32>], initial_labels: &[bool]) -> (Vec<Vec<f32>>, Vec<bool>) {
    let n = vectors.len();
    let mut labels = vec![false; n];

    // For each vector, check if it is "similar" (cosine ≥ threshold) to any
    // earlier vector in the sequence — this mirrors what a sequential dedup store does.
    for i in 1..n {
        'outer: for j in 0..i {
            if cosine_similarity(&vectors[i], &vectors[j]) >= COSINE_DUP_THRESHOLD {
                labels[i] = true;
                break 'outer;
            }
        }
    }
    // For vectors labeled true by initial_labels but not caught by cosine check,
    // keep the initial label (the perturb was very strong, might still be close).
    // Actually, cosine-derived labels are the ground truth; initial labels are discarded.
    let _ = initial_labels;
    (vectors.to_vec(), labels)
}

// ─── strategies ──────────────────────────────────────────────────────────────

fn run_simhash(vectors: &[Vec<f32>], labels: &[bool]) -> (DedupStats, u64) {
    let hasher = SimHasher::new(DIM, 64, 0xabc123);
    let mut best_ns = u64::MAX;
    let mut final_stats = DedupStats::default();

    for _ in 0..RUNS {
        let mut store = DedupStore::new(
            SimHasher::new(DIM, 64, 0xabc123),
            0.88,
            DedupMethod::SimHash,
        );
        let t0 = Instant::now();
        for v in vectors.iter() {
            store.insert(v.clone(), None);
        }
        let elapsed = t0.elapsed().as_nanos() as u64;
        if elapsed < best_ns {
            best_ns = elapsed;
            final_stats = DedupStats::compute(store.decisions(), labels);
        }
        let _ = hasher.fingerprint(&vectors[0]); // avoid dead-code elimination
    }
    (final_stats, best_ns)
}

fn run_minhash(vectors: &[Vec<f32>], labels: &[bool]) -> (DedupStats, u64) {
    let mut best_ns = u64::MAX;
    let mut final_stats = DedupStats::default();

    for _ in 0..RUNS {
        let mut store = DedupStore::new(
            MinHasher::new(DIM, 128, 8, 0xdef456),
            0.75,
            DedupMethod::MinHash,
        );
        let t0 = Instant::now();
        for v in vectors.iter() {
            store.insert(v.clone(), None);
        }
        let elapsed = t0.elapsed().as_nanos() as u64;
        if elapsed < best_ns {
            best_ns = elapsed;
            final_stats = DedupStats::compute(store.decisions(), labels);
        }
    }
    (final_stats, best_ns)
}

fn run_hybrid(vectors: &[Vec<f32>], labels: &[bool]) -> (DedupStats, u64) {
    let mut best_ns = u64::MAX;
    let mut final_stats = DedupStats::default();

    for _ in 0..RUNS {
        let mut store = HybridDedupStore::new(
            SimHasher::new(DIM, 64, 0x789abc),
            0.70,
            COSINE_DUP_THRESHOLD,
        );
        let t0 = Instant::now();
        for v in vectors.iter() {
            store.insert(v.clone(), None);
        }
        let elapsed = t0.elapsed().as_nanos() as u64;
        if elapsed < best_ns {
            best_ns = elapsed;
            let decisions = store.decisions();
            final_stats = DedupStats::compute(decisions, labels);
        }
    }
    (final_stats, best_ns)
}

// ─── output ──────────────────────────────────────────────────────────────────

fn print_header() {
    println!("══════════════════════════════════════════════════════════════");
    println!("  ruvector-lsf-dedup  │  Semantic Memory Deduplication Bench");
    println!("══════════════════════════════════════════════════════════════");
    println!("  DIM           = {DIM}");
    println!("  N_BASE        = {N_BASE}");
    println!(
        "  Near-dup frac = {:.0}%  (cosine noise 0.05)",
        NEAR_DUP_FRACTION * 100.0
    );
    println!(
        "  Exact frac    = {:.0}%  (cosine noise 0.001)",
        EXACT_FRACTION * 100.0
    );
    println!("  Dup threshold = {COSINE_DUP_THRESHOLD} cosine (ground truth)");
    println!("  Runs          = {RUNS}");
    println!("──────────────────────────────────────────────────────────────");
}

fn print_results(
    total: usize,
    sim: &DedupStats,
    sim_ns: u64,
    min: &DedupStats,
    min_ns: u64,
    hyb: &DedupStats,
    hyb_ns: u64,
) {
    println!("\nDataset  : {total} vectors total");
    println!(
        "  Unique   : {} ({:.0}%)",
        sim.unique_stored,
        sim.unique_stored as f32 / total as f32 * 100.0
    );
    println!(
        "  Dup pool : {} ({:.0}%)",
        sim.duplicates_rejected,
        sim.duplicates_rejected as f32 / total as f32 * 100.0
    );

    println!(
        "\n{:<12} {:>8} {:>9} {:>9} {:>8} {:>10} {:>10}",
        "Strategy", "Stored", "Rejected", "Prec", "Recall", "FPR", "Time(ms)"
    );
    println!("{}", "─".repeat(70));

    print_row("SimHash", sim, sim_ns);
    print_row("MinHash", min, min_ns);
    print_row("Hybrid", hyb, hyb_ns);
    println!();
}

fn print_row(name: &str, s: &DedupStats, ns: u64) {
    let ms = ns as f64 / 1_000_000.0;
    println!(
        "{:<12} {:>8} {:>9} {:>8.3} {:>9.3} {:>9.3} {:>10.2}",
        name,
        s.unique_stored,
        s.duplicates_rejected,
        s.precision,
        s.recall,
        s.false_positive_rate,
        ms
    );
}

fn acceptance_check(sim: &DedupStats, min: &DedupStats, hyb: &DedupStats) {
    println!("══════════════════════════════════════════════════════════════");
    println!("  Acceptance Criteria");
    println!("──────────────────────────────────────────────────────────────");

    let mut passed = true;

    let checks: &[(&str, f32, f32, f32, f32)] = &[
        ("SimHash  recall", sim.recall, 0.55, sim.recall, 0.55),
        (
            "SimHash  precision",
            sim.precision,
            0.70,
            sim.precision,
            0.70,
        ),
        ("MinHash  recall", min.recall, 0.50, min.recall, 0.50),
        (
            "MinHash  precision",
            min.precision,
            0.65,
            min.precision,
            0.65,
        ),
        ("Hybrid   recall", hyb.recall, 0.70, hyb.recall, 0.70),
        (
            "Hybrid   precision",
            hyb.precision,
            0.85,
            hyb.precision,
            0.85,
        ),
    ];

    for (label, val, threshold, _, _) in checks {
        let ok = *val >= *threshold;
        if !ok {
            passed = false;
        }
        println!(
            "  {:<24} {:.3}  ≥ {:.2}  {}",
            label,
            val,
            threshold,
            if ok { "PASS ✓" } else { "FAIL ✗" }
        );
    }

    println!("──────────────────────────────────────────────────────────────");
    if passed {
        println!("  RESULT: ALL ACCEPTANCE CHECKS PASSED");
    } else {
        println!("  RESULT: ONE OR MORE CHECKS FAILED");
    }
    println!("══════════════════════════════════════════════════════════════");

    // Exit non-zero only for Hybrid — the core research claim
    if hyb.recall < 0.70 || hyb.precision < 0.85 {
        std::process::exit(1);
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn unit_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0xfeedcafe;
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let hi = (state >> 33) as u32;
            f32::from_bits(0x3f80_0000 | (hi & 0x7fffff)) * 2.0 - 3.0
        })
        .collect();
    normalize(&mut v);
    v
}

fn perturb(base: &[f32], noise: f32, seed: u64) -> Vec<f32> {
    let mut state = seed ^ 0xbaddecaf;
    let mut v: Vec<f32> = base
        .iter()
        .map(|&x| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let hi = (state >> 33) as u32;
            let n = f32::from_bits(0x3f80_0000 | (hi & 0x7fffff)) * 2.0 - 3.0;
            x + noise * n
        })
        .collect();
    normalize(&mut v);
    v
}

fn fisher_yates(arr: &mut Vec<usize>, seed: u64) {
    let mut state = seed;
    let n = arr.len();
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        arr.swap(i, j);
    }
}

// Expose DedupResult for use in this file (suppress unused import warning)
fn _use_dedup_result(_: DedupResult) {}
