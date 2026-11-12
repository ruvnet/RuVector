//! Benchmark binary: coherence-weighted agent memory compaction.
//!
//! Simulates an agent accumulating 2 000 memories organised in 20 topic
//! clusters, running biased access patterns (5 hot clusters get 6× more
//! accesses), then compacting to 50% capacity and measuring Recall@10 for
//! 50 test queries from the hot clusters.
//!
//! Run:
//!   cargo run --release -p ruvector-agent-memory

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_agent_memory::{
    compact, recall_at_k, CoherencePolicy, CompactionPolicy, LfuPolicy, LruPolicy, MemoryStore,
};
use std::time::{Duration, Instant};

// ── Dataset parameters ────────────────────────────────────────────────────────
const N_MEMORIES: usize = 2_000;
const N_CLUSTERS: usize = 20;
const N_HOT_CLUSTERS: usize = 5;
const DIMS: usize = 64;
const N_QUERIES: usize = 50;
const K: usize = 10;
const TARGET_SIZE: usize = N_MEMORIES / 2; // compact to 50%
const CONTEXT_WINDOW_SIZE: usize = 20;

// Access simulation
const N_COLD_ERA_ACCESSES: usize = 200; // random across all memories
const N_HOT_ERA_ACCESSES: usize = 600; // 90% to hot clusters
const HOT_ERA_HOT_FRAC: f64 = 0.90;

// ── Utilities ─────────────────────────────────────────────────────────────────

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

// ── Dataset generation ────────────────────────────────────────────────────────

struct Dataset {
    centroids: Vec<Vec<f32>>,
    // cluster assignment per memory (index into store matches index here)
    cluster_of: Vec<usize>,
    // test queries: each pair of (query_vec, true_top_k neighbor ids in full store)
    queries: Vec<(Vec<f32>, Vec<u64>)>,
}

fn generate_dataset(store: &mut MemoryStore, rng: &mut StdRng) -> Dataset {
    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS).map(|_| unit_gaussian(rng, DIMS)).collect();

    let per_cluster = N_MEMORIES / N_CLUSTERS;
    let mut cluster_of = Vec::with_capacity(N_MEMORIES);

    for (c, centroid) in centroids.iter().enumerate() {
        for _ in 0..per_cluster {
            let v = perturb(centroid, 0.35, rng);
            store.insert(v);
            cluster_of.push(c);
        }
    }

    // Generate 50 test queries near hot clusters (0..N_HOT_CLUSTERS)
    let mut queries = Vec::with_capacity(N_QUERIES);
    for i in 0..N_QUERIES {
        let hot_cluster = i % N_HOT_CLUSTERS;
        let q = perturb(&centroids[hot_cluster], 0.30, rng);

        // True top-K = brute force over all entries
        let results = store.search(&q, K);
        let truth: Vec<u64> = results.iter().map(|r| r.id).collect();
        queries.push((q, truth));
    }

    Dataset {
        centroids,
        cluster_of,
        queries,
    }
}

// ── Access simulation ─────────────────────────────────────────────────────────

/// Returns the context window (last CONTEXT_WINDOW_SIZE query vectors).
fn simulate_accesses(
    store: &mut MemoryStore,
    dataset: &Dataset,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let per_cluster = N_MEMORIES / N_CLUSTERS;

    // Cold era: uniform random accesses
    for _ in 0..N_COLD_ERA_ACCESSES {
        let idx = rng.gen_range(0..N_MEMORIES);
        store.access_by_index(idx);
    }

    // Hot era: biased toward hot clusters (0..N_HOT_CLUSTERS)
    let mut context_accesses: Vec<Vec<f32>> = Vec::new();
    for _ in 0..N_HOT_ERA_ACCESSES {
        let idx = if rng.gen_bool(HOT_ERA_HOT_FRAC) {
            // access a random memory in a hot cluster
            let hot_c = rng.gen_range(0..N_HOT_CLUSTERS);
            let offset = rng.gen_range(0..per_cluster);
            hot_c * per_cluster + offset
        } else {
            // access a random cold memory
            let cold_c = rng.gen_range(N_HOT_CLUSTERS..N_CLUSTERS);
            let offset = rng.gen_range(0..per_cluster);
            cold_c * per_cluster + offset
        };
        store.access_by_index(idx);
        // Log query vector for context window (approximate with centroid)
        let cluster = dataset.cluster_of[idx];
        context_accesses.push(dataset.centroids[cluster].clone());
    }

    // Context window = last CONTEXT_WINDOW_SIZE access centroids
    let start = context_accesses.len().saturating_sub(CONTEXT_WINDOW_SIZE);
    context_accesses[start..].to_vec()
}

// ── Compaction + evaluation ───────────────────────────────────────────────────

fn measure_recall(original_queries: &[(Vec<f32>, Vec<u64>)], store: &MemoryStore) -> f32 {
    let mut total = 0.0f32;
    for (q, truth) in original_queries {
        let candidates: Vec<u64> = store.search(q, K).into_iter().map(|r| r.id).collect();
        total += recall_at_k(truth, &candidates);
    }
    total / original_queries.len() as f32
}

fn run_policy(
    policy: &dyn CompactionPolicy,
    context_window: &[Vec<f32>],
    queries: &[(Vec<f32>, Vec<u64>)],
    rng_seed: u64,
) -> (f32, Duration) {
    // Rebuild a fresh store with the same RNG seed so all policies see identical data.
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut store = MemoryStore::new(DIMS);
    let dataset = generate_dataset(&mut store, &mut rng);
    let mut rng2 = StdRng::seed_from_u64(rng_seed + 1);
    simulate_accesses(&mut store, &dataset, &mut rng2);

    // Sanity: store should still have N_MEMORIES entries before compaction.
    assert_eq!(store.len(), N_MEMORIES);

    let t0 = Instant::now();
    compact(&mut store, policy, TARGET_SIZE, context_window);
    let compaction_time = t0.elapsed();

    assert_eq!(store.len(), TARGET_SIZE, "store size after compaction");

    let recall = measure_recall(queries, &store);
    (recall, compaction_time)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let seed: u64 = 42;
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    ruvector-agent-memory — Compaction Benchmark              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Print env info
    println!("Platform  : {}", std::env::consts::OS);
    println!("Arch      : {}", std::env::consts::ARCH);
    println!("Rust      : {}", rustc_version_string());
    println!();

    // Dataset
    println!("Dataset");
    println!("  Memories        : {N_MEMORIES}");
    println!("  Clusters        : {N_CLUSTERS}");
    println!("  Hot clusters    : {N_HOT_CLUSTERS}");
    println!("  Dimensions      : {DIMS}");
    println!("  Test queries    : {N_QUERIES}");
    println!("  K               : {K}");
    println!("  Target size     : {TARGET_SIZE} (50% compaction)");
    println!("  Context window  : {CONTEXT_WINDOW_SIZE} entries");
    println!("  Cold era accesses: {N_COLD_ERA_ACCESSES}");
    println!(
        "  Hot era accesses : {N_HOT_ERA_ACCESSES} ({:.0}% hot-cluster bias)",
        HOT_ERA_HOT_FRAC * 100.0
    );
    println!();

    // Build ground truth and context window from a reference store
    let mut rng_ref = StdRng::seed_from_u64(seed);
    let mut ref_store = MemoryStore::new(DIMS);
    let dataset = generate_dataset(&mut ref_store, &mut rng_ref);
    let queries = dataset.queries.clone();

    let mut rng_acc = StdRng::seed_from_u64(seed + 1);
    let context_window = simulate_accesses(&mut ref_store, &dataset, &mut rng_acc);
    println!(
        "Context window built: {} vectors from hot-era accesses\n",
        context_window.len()
    );

    // Memory estimate (f32 per float, 4 bytes)
    let bytes_full = N_MEMORIES * DIMS * 4;
    let bytes_compact = TARGET_SIZE * DIMS * 4;
    println!("Memory estimate");
    println!(
        "  Full store     : {} KB ({} vectors × {} dims × 4 B)",
        bytes_full / 1024,
        N_MEMORIES,
        DIMS
    );
    println!(
        "  After compaction: {} KB ({} vectors × {} dims × 4 B)",
        bytes_compact / 1024,
        TARGET_SIZE,
        DIMS
    );
    println!();

    // --- Baseline recall BEFORE compaction ---
    let recall_before = measure_recall(&queries, &ref_store);
    println!(
        "Recall@{K} BEFORE compaction: {:.1}%\n",
        recall_before * 100.0
    );

    // --- Run all three policies ---
    struct Result {
        name: String,
        recall: f32,
        compaction_us: u64,
    }
    let mut results: Vec<Result> = Vec::new();

    let cow = CoherencePolicy::default();
    let policies: Vec<(&dyn CompactionPolicy, &str)> = vec![
        (&LruPolicy as &dyn CompactionPolicy, "LRU"),
        (&LfuPolicy as &dyn CompactionPolicy, "LFU"),
        (&cow as &dyn CompactionPolicy, "CoherenceWeighted"),
    ];

    for (policy, _name) in &policies {
        let (recall, dur) = run_policy(*policy, &context_window, &queries, seed);
        results.push(Result {
            name: policy.name().to_string(),
            recall,
            compaction_us: dur.as_micros() as u64,
        });
    }

    // Print results table
    println!(
        "{:<22} {:>12} {:>18} {:>14}",
        "Policy", "Recall@10", "Compaction (µs)", "vs LRU (pp)"
    );
    println!("{}", "-".repeat(70));
    let lru_recall = results[0].recall;
    for r in &results {
        let delta = (r.recall - lru_recall) * 100.0;
        let delta_str = if r.name == "LRU" {
            "—".to_string()
        } else if delta >= 0.0 {
            format!("+{delta:.1}")
        } else {
            format!("{delta:.1}")
        };
        println!(
            "{:<22} {:>11.1}% {:>17} {:>14}",
            r.name,
            r.recall * 100.0,
            r.compaction_us,
            delta_str
        );
    }
    println!();

    // --- Acceptance test ---
    let lfu_recall = results[1].recall;
    let cow_recall = results[2].recall;

    println!("Acceptance test");
    let threshold_pp = 2.0_f32; // CoW must beat LRU by at least 2 pp
    let pass = cow_recall > lru_recall + threshold_pp / 100.0;
    let lfu_pass = lfu_recall > lru_recall - 0.05; // LFU should not be much worse than LRU
    println!(
        "  CoW recall ({:.1}%) > LRU recall ({:.1}%) + {threshold_pp:.0}pp : {}",
        cow_recall * 100.0,
        lru_recall * 100.0,
        if pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!(
        "  LFU recall ({:.1}%) within 5pp of LRU ({:.1}%)         : {}",
        lfu_recall * 100.0,
        lru_recall * 100.0,
        if lfu_pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!();

    if pass && lfu_pass {
        println!("→ BENCHMARK PASSED");
    } else {
        println!("→ BENCHMARK FAILED");
        std::process::exit(1);
    }

    run_arbitration_bench();
}

/// CAMA-pattern arbitration bench (ADR-330, PIR WP27): naive vote counting
/// vs provenance-clustered arbitration over a synthetic 1 000-memory causal
/// graph (100 independent origins, 900 derived memories). The correctness
/// assertion — arbitration collapses the 1 000 memories to exactly the 100
/// origin lineages, at a score no higher than the naive vote — is checked
/// here as well as in the test suite.
///
/// Every derived memory here **declares** its parent, so this measures the
/// clustering math on the mechanism's best case. It is not an adversarial
/// benchmark: content-copying memories that declare no parent mint their own
/// lineages by design (see `arbitration`'s "Known limitation" docs).
fn run_arbitration_bench() {
    use rand::rngs::OsRng;
    use ruvector_agent_memory::{
        arbitrate, ArbitrationConfig, AtomicObservation, CausalEpisodicGraph, NodeRef,
        ObservationId, ObservationSource, ReliabilityModel, SourceKind, Tenant,
    };
    use rvf_types::Ed25519Keypair;

    const N_ROOTS: usize = 100;
    const N_DERIVED: usize = 900;

    println!("\nArbitration bench (ADR-330): naive vote vs provenance-clustered arbitration");
    let keypair = Ed25519Keypair::generate(&mut OsRng);
    let mut rng = StdRng::seed_from_u64(330);
    let mut graph = CausalEpisodicGraph::new(Tenant::new("bench"));

    let sign = |payload: &[u8], t: u64, parents: Vec<ObservationId>, kp: &Ed25519Keypair| {
        AtomicObservation::new_signed(
            ObservationSource {
                kind: SourceKind::AgentObservation,
                id: "bench".to_string(),
                public_key: [0u8; 32],
            },
            t,
            0.9,
            Tenant::new("bench"),
            parents,
            payload.to_vec(),
            kp,
        )
        .unwrap()
    };

    let roots: Vec<ObservationId> = (0..N_ROOTS)
        .map(|i| {
            graph
                .ingest(sign(&(i as u32).to_le_bytes(), 0, vec![], &keypair))
                .unwrap()
        })
        .collect();
    // Per-root latest descendant, so derivation chains deepen over time.
    let mut latest: Vec<ObservationId> = roots.clone();
    let mut memories: Vec<NodeRef> = Vec::with_capacity(N_DERIVED);
    for i in 0..N_DERIVED {
        let r = rng.gen_range(0..N_ROOTS);
        let parent = latest[r];
        let id = graph
            .ingest(sign(
                &(1_000_000 + i as u32).to_le_bytes(),
                1 + i as u64,
                vec![parent],
                &keypair,
            ))
            .unwrap();
        latest[r] = id;
        memories.push(NodeRef::Observation(id));
    }

    let config = ArbitrationConfig {
        now_ns: 1_000_000,
        half_life_ns: u64::MAX,
        sufficiency_threshold: 1,
        reliability: ReliabilityModel::default(),
    };

    // Naive vote: sum of raw confidences (the correlation-blind baseline).
    let t0 = Instant::now();
    let naive: f64 = memories
        .iter()
        .map(|m| match m {
            NodeRef::Observation(id) => graph.observation(*id).unwrap().confidence as f64,
            NodeRef::Cluster(_) => unreachable!(),
        })
        .sum();
    let naive_us = t0.elapsed().as_micros();

    let t1 = Instant::now();
    let outcome = arbitrate(&graph, &memories, &config).expect("arbitration succeeds");
    let arb_us = t1.elapsed().as_micros();
    let verdict = outcome.verdict();

    println!(
        "  Memories        : {} ({} origins, {} derived)",
        N_DERIVED, N_ROOTS, N_DERIVED
    );
    println!("  Naive vote      : {naive:.1} supporting confidence ({naive_us} µs)");
    println!(
        "  Arbitrated      : {:.1} effective confidence across {} lineages ({} µs)",
        verdict.arbitrated_confidence,
        verdict.independent_lineages(),
        arb_us
    );
    assert_eq!(verdict.independent_lineages(), N_ROOTS);
    assert!(verdict.arbitrated_confidence <= verdict.naive_confidence + 1e-9);
    println!(
        "  → {N_DERIVED} declared-derivation memories collapse to {N_ROOTS} independent lineages"
    );
    println!("    (declared-parent chains — the mechanism's best case, not an adversarial test)");
}

fn rustc_version_string() -> String {
    // Populated at compile time via RUSTC_VERSION env set in build.rs; fall back if unavailable.
    option_env!("CARGO_PKG_RUST_VERSION")
        .unwrap_or("unknown")
        .to_string()
}

#[cfg(test)]
mod bench_tests {
    use super::*;
    use ruvector_agent_memory::{CoherencePolicy, LruPolicy};

    #[test]
    fn coherence_beats_lru_acceptance() {
        let seed = 42u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut ref_store = MemoryStore::new(DIMS);
        let dataset = generate_dataset(&mut ref_store, &mut rng);
        let queries = dataset.queries.clone();
        let mut rng2 = StdRng::seed_from_u64(seed + 1);
        let context_window = simulate_accesses(&mut ref_store, &dataset, &mut rng2);

        let (lru_recall, _) = run_policy(&LruPolicy, &context_window, &queries, seed);
        let (cow_recall, _) =
            run_policy(&CoherencePolicy::default(), &context_window, &queries, seed);

        assert!(
            cow_recall > lru_recall + 0.02,
            "CoW recall {:.1}% should exceed LRU recall {:.1}% by >2pp",
            cow_recall * 100.0,
            lru_recall * 100.0
        );
    }
}
