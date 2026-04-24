//! Save -> ship -> warm-restart end-to-end demo for ruLake.
//!
//! Shows the three operator phases a real warm-handoff goes through:
//!
//!   1. PUBLISHER: prime a RuLake cache from a backend, snapshot the
//!      compressed index + witnessed bundle sidecar to a directory.
//!   2. READER:    a brand-new RuLake process (no backend registered)
//!      calls `warm_from_dir` on that same directory and is serving
//!      queries in milliseconds, bypassing the backend round-trip and
//!      RaBitQ compression entirely.
//!   3. COLD:      for comparison, another fresh RuLake with the
//!      backend registered but no warm snapshot pays the full
//!      prime-from-backend cost on its first query.
//!
//! The summary at the end reports the speedup of warm-from-disk vs.
//! cold-prime-from-backend plus steady-state QPS for both — the QPS
//! should be nearly identical because they end up with the same
//! compressed index; only the first-query cost differs.
//!
//! Run with:
//!   cargo run -p ruvector-rulake --release --example warm_restart

use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use ruvector_rulake::cache::Consistency;
use ruvector_rulake::{LocalBackend, RuLake};

const BACKEND_ID: &str = "prod-warehouse";
const COLLECTION: &str = "memories";
const N: usize = 5_000;
const D: usize = 128;
const RERANK: usize = 20;
const SEED: u64 = 42;
const NUM_QUERIES: usize = 100;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  ruLake: save → ship → warm-restart flow                         ║");
    println!("║                                                                  ║");
    println!("║  Phase 1 — Publisher primes cache, snapshots to disk             ║");
    println!("║  Phase 2 — Fresh Reader warms from disk (NO backend round-trip)  ║");
    println!("║  Phase 3 — Cold Reader re-primes from backend (for comparison)   ║");
    println!("║                                                                  ║");
    println!("║  Corpus: {N} vectors, D={D}, rerank={RERANK}, seed={SEED}                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Target directory for the snapshot. Using PID keeps parallel runs
    // of this example isolated. `std::env::temp_dir()` is
    // cross-platform; `/tmp/...` in the task spec just happens to be
    // where that resolves on Linux.
    let snapshot_dir = std::env::temp_dir().join(format!(
        "rulake-warm-demo-{}-{}",
        std::process::id(),
        Instant::now().elapsed().as_nanos()
    ));
    println!("Snapshot dir: {}\n", snapshot_dir.display());

    // Shared corpus — both the publisher and the cold-comparison
    // reader need an identical-shaped backend, so we build the
    // vectors once.
    let (ids, vectors) = build_clustered_corpus(N, D, SEED);

    // ---------- Phase 1: Publisher ----------
    println!("── Phase 1: Publisher ──────────────────────────────────────────────");
    let pub_backend = Arc::new(LocalBackend::new(BACKEND_ID));
    pub_backend
        .put_collection(COLLECTION, D, ids.clone(), vectors.clone())
        .expect("put_collection");

    let publisher = RuLake::new(RERANK, SEED);
    publisher
        .register_backend(pub_backend.clone())
        .expect("register backend");

    let key = (BACKEND_ID.to_string(), COLLECTION.to_string());

    // Prime the cache with one query so there's a primed entry to
    // snapshot.
    let warm_query = query_vector(D, 0xC0FFEE);
    let t_prime = Instant::now();
    let _ = publisher
        .search_one(BACKEND_ID, COLLECTION, &warm_query, 10)
        .expect("prime query");
    let prime_ms = t_prime.elapsed().as_secs_f64() * 1e3;
    println!(
        "  primed cache from backend in {:.2} ms ({} vectors × D={})",
        prime_ms, N, D
    );

    // Save -> the snapshot dir will hold index.rbpx + table.rulake.json.
    let t_save = Instant::now();
    let written_path = publisher
        .save_cache_to_dir(&key, &snapshot_dir)
        .expect("save_cache_to_dir");
    let save_ms = t_save.elapsed().as_secs_f64() * 1e3;

    let idx_size = std::fs::metadata(&written_path)
        .map(|m| m.len())
        .unwrap_or(0);
    let bundle_path = snapshot_dir.join("table.rulake.json");
    let bundle_size = std::fs::metadata(&bundle_path)
        .map(|m| m.len())
        .unwrap_or(0);

    println!("  save_cache_to_dir took {:.2} ms", save_ms);
    println!("  wrote {} ({} bytes)", written_path.display(), idx_size);
    println!(
        "  wrote {} ({} bytes)\n",
        bundle_path.display(),
        bundle_size
    );

    // ---------- Phase 2: Fresh reader, warm from disk ----------
    println!("── Phase 2: Reader process (warm from disk) ────────────────────────");
    // BRAND NEW RuLake — no backend registered. This is the whole
    // point: warm-restart must not require the backend to be wired.
    // We pick `Consistency::Frozen` so queries serve directly out of
    // the installed entry without a backend coherence check — the
    // whole raison d'être of warm-restart. A real deployment might
    // flip to `Eventual` once the backend controller comes online.
    let warm_reader = RuLake::new(RERANK, SEED).with_consistency(Consistency::Frozen);

    let t_warm = Instant::now();
    let n_loaded = warm_reader
        .warm_from_dir(&key, &snapshot_dir)
        .expect("warm_from_dir");
    let warm_ms = t_warm.elapsed().as_secs_f64() * 1e3;

    println!(
        "  warm_from_dir loaded n={} in {:.2} ms (no backend round-trip)",
        n_loaded, warm_ms
    );

    // Confirm the cache bookkeeping matches the warm-install contract.
    let warm_stats = warm_reader.cache_stats();
    assert_eq!(
        warm_stats.warm_installs, 1,
        "expected exactly one warm_install after warm_from_dir"
    );
    assert_eq!(
        warm_stats.primes, 0,
        "warm path must NOT have run a backend prime"
    );
    println!(
        "  cache_stats: warm_installs={} primes={} (as expected)\n",
        warm_stats.warm_installs, warm_stats.primes
    );

    // Steady-state workload against the warm reader.
    let queries = build_query_set(NUM_QUERIES, D, 0xBEEF);
    let warm_qps_ms = time_queries(&warm_reader, &queries);
    let warm_qps = (NUM_QUERIES as f64) / (warm_qps_ms / 1e3);
    println!(
        "  {} queries served in {:.2} ms → {:.0} QPS",
        NUM_QUERIES, warm_qps_ms, warm_qps
    );

    // ---------- Phase 3: Cold reader, re-prime from backend ----------
    println!("\n── Phase 3: Cold reader (prime from backend) ───────────────────────");
    let cold_backend = Arc::new(LocalBackend::new(BACKEND_ID));
    cold_backend
        .put_collection(COLLECTION, D, ids.clone(), vectors.clone())
        .expect("cold put_collection");

    let cold_reader = RuLake::new(RERANK, SEED);
    cold_reader
        .register_backend(cold_backend.clone())
        .expect("register cold backend");

    let t_cold = Instant::now();
    let _ = cold_reader
        .search_one(BACKEND_ID, COLLECTION, &warm_query, 10)
        .expect("cold prime query");
    let cold_prime_ms = t_cold.elapsed().as_secs_f64() * 1e3;
    println!(
        "  first query primed cache from backend in {:.2} ms",
        cold_prime_ms
    );

    let cold_qps_ms = time_queries(&cold_reader, &queries);
    let cold_qps = (NUM_QUERIES as f64) / (cold_qps_ms / 1e3);
    println!(
        "  {} queries served in {:.2} ms → {:.0} QPS",
        NUM_QUERIES, cold_qps_ms, cold_qps
    );

    // ---------- Summary ----------
    let speedup = if warm_ms > 0.0 {
        cold_prime_ms / warm_ms
    } else {
        f64::INFINITY
    };
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY                                                         ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!(
        "  Cold start (prime from backend):   {:>9.2} ms",
        cold_prime_ms
    );
    println!("  Warm start (load from disk):       {:>9.2} ms", warm_ms);
    println!(
        "  Speedup: {:.2} ms / {:.2} ms ≈ {:.1}x",
        cold_prime_ms, warm_ms, speedup
    );
    println!(
        "  Query QPS  cold: {:>8.0}   warm: {:>8.0}   (should be similar)",
        cold_qps, warm_qps
    );
    println!(
        "  Snapshot footprint: index.rbpx={} bytes + table.rulake.json={} bytes",
        idx_size, bundle_size
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(&snapshot_dir);
    println!("\n✓ cleaned up {}", snapshot_dir.display());
}

/// Build a clustered-Gaussian corpus. Five Gaussian clusters with
/// unit-ish variance — enough structure that ANN recall is
/// meaningful, small enough that 5k × D=128 primes in tens of ms.
fn build_clustered_corpus(n: usize, d: usize, seed: u64) -> (Vec<u64>, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let num_clusters = 5usize;

    // Cluster centroids spread on a shell so they're well-separated.
    let centroids: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| {
            let v: Vec<f32> = (0..d).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect();
            v
        })
        .collect();

    let jitter = Normal::new(0.0_f32, 0.5_f32).expect("valid normal");

    let ids: Vec<u64> = (0..n as u64).collect();
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let c = &centroids[i % num_clusters];
            c.iter()
                .map(|x| x + jitter.sample(&mut rng))
                .collect::<Vec<f32>>()
        })
        .collect();

    (ids, vectors)
}

/// Deterministic per-query vector generator so warm & cold paths see
/// the exact same query stream (makes QPS comparison apples-to-apples).
fn query_vector(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..d).map(|_| rng.gen_range(-3.0_f32..3.0_f32)).collect()
}

fn build_query_set(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n as u64)
        .map(|i| query_vector(d, seed.wrapping_add(i)))
        .collect()
}

/// Run the query batch sequentially, return total wall time in ms.
/// Uses `search_one` (not the batch API) because it's the more common
/// serving path and stresses the per-query cache fast-path.
fn time_queries(lake: &RuLake, queries: &[Vec<f32>]) -> f64 {
    let t = Instant::now();
    for q in queries {
        let _hits = lake
            .search_one(BACKEND_ID, COLLECTION, q, 10)
            .expect("search_one");
    }
    t.elapsed().as_secs_f64() * 1e3
}
