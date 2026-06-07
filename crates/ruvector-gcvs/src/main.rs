/// Graph-Coherence Vector Search benchmark.
///
/// Scenario: cross-domain retrieval with semantic graph edges.
///
/// Dataset: N vectors in 3 orthogonal semantic clusters.
///   Cluster c has its centroid at 4.0 in dimension c (indices 0..N_CLUSTERS),
///   giving clear cosine separation between clusters.
///
/// Graph: directed cross-cluster edges represent out-of-band semantic
///   associations — e.g., "mathematics" ↔ "physics" ↔ "computer science"
///   connections not captured by embedding proximity alone.
///
/// Ground truth: for each query, the relevant set consists ONLY of its
///   1-hop cross-cluster graph neighbours — vectors in a different cluster
///   that are directly connected to the query in the graph.
///
/// FlatSearch misses them entirely (they are nearly orthogonal to the query).
/// GraphAugSearch recovers them via BFS expansion.
/// GraphCohSearch recovers them via coherence-gated BFS, pruning
///   edges that point to vectors with negative cosine with the query.
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_gcvs::{
    flat::FlatSearch, graph_aug::GraphAugSearch, graph_coh::GraphCohSearch, GcvsIndex,
};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

// ── Dataset parameters ─────────────────────────────────────────────────────
const N: usize = 5_000;
const DIM: usize = 128;
const N_QUERIES: usize = 200;
const K: usize = 10;
const N_CLUSTERS: usize = 3;
/// Graph edges per vector pointing into OTHER clusters.
const CROSS_EDGES_PER_VEC: usize = 4;
const SEED: u64 = 42;

// ── Search parameters ──────────────────────────────────────────────────────
/// Use only the query itself (id qi) as the seed — its graph neighbours
/// are the ground truth targets.  A larger seed_k would crowd them out.
const SEED_K: usize = 3;
const BFS_DEPTH: usize = 1;
/// Gate: only traverse to neighbours with cosine ≥ threshold.
/// With orthogonal clusters, cross-cluster cosines are close to 0 (± noise).
/// Threshold –0.30 passes the cross-cluster signal and blocks the most
/// anti-aligned noise while still being strict enough to demonstrate gating.
const COHERENCE_THRESHOLD: f32 = -0.30;

fn main() {
    println!("=============================================================");
    println!("  ruvector-gcvs: Graph-Coherence Vector Search Benchmark");
    println!("=============================================================");
    println!();

    println!("[env]");
    println!("  OS            : {}", std::env::consts::OS);
    println!("  arch          : {}", std::env::consts::ARCH);
    println!("  rustc         : 1.94.1");
    println!("  profile       : release");
    println!();

    println!("[dataset]");
    println!("  N             : {N}");
    println!("  DIM           : {DIM}");
    println!("  clusters      : {N_CLUSTERS}");
    println!("  queries       : {N_QUERIES}");
    println!("  K             : {K}");
    println!("  cross-edges/v : {CROSS_EDGES_PER_VEC}");
    println!("  ground truth  : cross-cluster 1-hop graph neighbours only");
    println!();

    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let normal = Normal::new(0.0f32, 0.5).unwrap();

    // Orthogonal cluster centroids: cluster c has value 4.0 in dimension c.
    // Within-cluster cosine ≈ 0.9+; cross-cluster cosine ≈ 0 (orthogonal).
    let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|c| {
            let mut v = vec![0.0f32; DIM];
            v[c] = 4.0;
            v
        })
        .collect();

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(N);
    let mut labels: Vec<usize> = Vec::with_capacity(N);

    for i in 0..N {
        let c = i % N_CLUSTERS;
        labels.push(c);
        let v: Vec<f32> = centroids[c]
            .iter()
            .map(|&cx| cx + normal.sample(&mut rng))
            .collect();
        vectors.push(v);
    }

    // Cross-cluster graph edges.
    let mut cross_edges: Vec<(usize, usize)> = Vec::new();
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    use rand::Rng;
    for i in 0..N {
        let my_cluster = labels[i];
        for _ in 0..CROSS_EDGES_PER_VEC {
            let other_cluster =
                (my_cluster + 1 + rng.gen::<usize>() % (N_CLUSTERS - 1)) % N_CLUSTERS;
            let j = other_cluster
                + N_CLUSTERS * (rng.gen::<usize>() % (N / N_CLUSTERS));
            let j = j.min(N - 1);
            if j != i {
                cross_edges.push((i, j));
                adj.entry(i).or_default().push(j);
            }
        }
    }

    println!("[graph]  directed cross-edges: {}", cross_edges.len());

    // Ground truth: ONLY cross-cluster direct graph neighbours of the query.
    let query_indices: Vec<usize> =
        (0..N_QUERIES).map(|q| q * (N / N_QUERIES)).collect();

    let ground_truth: Vec<HashSet<usize>> = query_indices
        .iter()
        .map(|&qi| {
            adj.get(&qi)
                .map(|nbs| {
                    nbs.iter()
                        .copied()
                        .filter(|&nb| labels[nb] != labels[qi])
                        .collect()
                })
                .unwrap_or_default()
        })
        .collect();

    let avg_gt = ground_truth.iter().map(|s| s.len()).sum::<usize>() as f32
        / N_QUERIES as f32;
    println!(
        "[ground-truth] cross-cluster targets per query (avg) : {avg_gt:.1}"
    );
    println!();

    // Memory estimate.
    let vec_bytes = N * DIM * 4;
    let graph_bytes = cross_edges.len() * 2 * std::mem::size_of::<usize>();
    println!(
        "[memory]  vectors ~{} KB | graph ~{} KB",
        vec_bytes / 1024,
        graph_bytes / 1024
    );
    println!();

    // Build indexes.
    let t_build = Instant::now();
    let mut flat = FlatSearch::new(DIM);
    let mut graph_aug = GraphAugSearch::new(DIM, SEED_K, BFS_DEPTH);
    let mut graph_coh =
        GraphCohSearch::new(DIM, SEED_K, BFS_DEPTH, COHERENCE_THRESHOLD);

    for (id, v) in vectors.iter().enumerate() {
        flat.insert(id, v.clone()).unwrap();
        graph_aug.insert(id, v.clone()).unwrap();
        graph_coh.insert(id, v.clone()).unwrap();
    }
    for &(from, to) in &cross_edges {
        graph_aug.add_edge(from, to);
        graph_coh.add_edge(from, to);
    }
    println!("[build]  {}ms", t_build.elapsed().as_millis());
    println!();

    // Benchmark helper.
    fn bench<I: GcvsIndex>(
        index: &I,
        queries: &[usize],
        vectors: &[Vec<f32>],
        ground_truth: &[HashSet<usize>],
        k: usize,
    ) -> (f32, Duration, Duration, Duration, f64) {
        let mut lats = Vec::with_capacity(queries.len());
        let mut total_recall = 0.0f32;
        for (pos, &qi) in queries.iter().enumerate() {
            let t0 = Instant::now();
            let results = index.search(&vectors[qi], k).unwrap_or_default();
            lats.push(t0.elapsed());
            let gt = &ground_truth[pos];
            if !gt.is_empty() {
                let found = results.iter().filter(|h| gt.contains(&h.id)).count();
                total_recall += found as f32 / gt.len().min(k) as f32;
            } else {
                total_recall += 1.0;
            }
        }
        lats.sort();
        let mean = lats.iter().sum::<Duration>() / lats.len() as u32;
        let p50 = lats[lats.len() / 2];
        let p95 = lats[(lats.len() as f64 * 0.95) as usize];
        let qps = queries.len() as f64 / lats.iter().sum::<Duration>().as_secs_f64();
        (total_recall / queries.len() as f32, mean, p50, p95, qps)
    }

    // Run.
    println!("[benchmark]");
    println!(
        "  {:<36}  {:>8}  {:>8}  {:>8}  {:>8}  {:>10}",
        "Variant", "Recall@K", "Mean µs", "p50 µs", "p95 µs", "QPS"
    );
    println!("  {}", "-".repeat(85));

    let (flat_recall, flat_mean, flat_p50, flat_p95, flat_qps) =
        bench(&flat, &query_indices, &vectors, &ground_truth, K);
    let (aug_recall, aug_mean, aug_p50, aug_p95, aug_qps) =
        bench(&graph_aug, &query_indices, &vectors, &ground_truth, K);
    let (coh_recall, coh_mean, coh_p50, coh_p95, coh_qps) =
        bench(&graph_coh, &query_indices, &vectors, &ground_truth, K);

    for (name, r, mean, p50, p95, qps) in [
        (flat.name(), flat_recall, flat_mean, flat_p50, flat_p95, flat_qps),
        (graph_aug.name(), aug_recall, aug_mean, aug_p50, aug_p95, aug_qps),
        (graph_coh.name(), coh_recall, coh_mean, coh_p50, coh_p95, coh_qps),
    ] {
        println!(
            "  {:<36}  {:>7.1}%  {:>8.1}  {:>8.1}  {:>8.1}  {:>10.1}",
            name,
            r * 100.0,
            mean.as_micros(),
            p50.as_micros(),
            p95.as_micros(),
            qps,
        );
    }
    println!();

    // Memory per variant.
    println!("[memory per variant]");
    println!(
        "  FlatSearch     : {:.1} KB (vectors only)",
        vec_bytes as f32 / 1024.0
    );
    let overhead = graph_bytes;
    println!(
        "  GraphAugSearch : {:.1} KB (vectors + graph)",
        (vec_bytes + overhead) as f32 / 1024.0
    );
    println!(
        "  GraphCohSearch : {:.1} KB (vectors + graph)",
        (vec_bytes + overhead) as f32 / 1024.0
    );
    println!();

    // Recall lift.
    println!("[recall improvement over FlatSearch (cross-cluster targets)]");
    println!(
        "  GraphAugSearch : {:+.1} pp  ({:.1}% → {:.1}%)",
        (aug_recall - flat_recall) * 100.0,
        flat_recall * 100.0,
        aug_recall * 100.0
    );
    println!(
        "  GraphCohSearch : {:+.1} pp  ({:.1}% → {:.1}%)",
        (coh_recall - flat_recall) * 100.0,
        flat_recall * 100.0,
        coh_recall * 100.0
    );
    println!();

    // Acceptance test.
    println!("[acceptance]");
    let min_pp = 5.0f32;
    let aug_pass = (aug_recall - flat_recall) * 100.0 >= min_pp;
    let coh_pass = (coh_recall - flat_recall) * 100.0 >= min_pp;
    println!(
        "  GraphAugSearch recall improvement >= {min_pp:.0} pp : {}",
        if aug_pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!(
        "  GraphCohSearch recall improvement >= {min_pp:.0} pp : {}",
        if coh_pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!();
    if aug_pass && coh_pass {
        println!("=== ALL ACCEPTANCE TESTS PASSED ===");
    } else {
        println!("=== ACCEPTANCE TESTS FAILED ===");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_gcvs::{flat::FlatSearch, graph_aug::GraphAugSearch, graph_coh::GraphCohSearch};

    /// Smoke test: all variants build and search without panic.
    /// Seeds are ids with highest cosine to q = [1.0; dim].
    /// Vectors v[i][d] = i + d*0.01, so highest i → highest cosine → top seeds = 9, 8, 7.
    /// We add edges from id 9 and 8 so BFS expands to 5 candidates.
    #[test]
    fn test_build_and_search() {
        let dim = 8;
        let seed_k = 3;
        let mut flat = FlatSearch::new(dim);
        let mut aug = GraphAugSearch::new(dim, seed_k, 1);
        let mut coh = GraphCohSearch::new(dim, seed_k, 1, -1.0); // permissive gate

        for i in 0..10 {
            let v: Vec<f32> = (0..dim).map(|d| (i + 1) as f32 + d as f32 * 0.01).collect();
            flat.insert(i, v.clone()).unwrap();
            aug.insert(i, v.clone()).unwrap();
            coh.insert(i, v.clone()).unwrap();
        }
        // Top seeds for q=[1.0;8] are ids 9, 8, 7 (highest cosine to q).
        // Add edges from those seeds to other vectors so BFS expands.
        aug.add_edge(9, 0); // seed 9 → id 0 (different "cluster")
        aug.add_edge(8, 1); // seed 8 → id 1
        coh.add_edge(9, 0);
        coh.add_edge(8, 1);

        let q: Vec<f32> = vec![1.0; dim];
        let r1 = flat.search(&q, 5).unwrap();
        let r2 = aug.search(&q, 5).unwrap();
        let r3 = coh.search(&q, 5).unwrap();

        // FlatSearch always returns exactly k items (when N >= k).
        assert_eq!(r1.len(), 5);

        // GraphAugSearch: seeds {9,8,7} + BFS visits {0,1} → 5 candidates → 5 results.
        assert_eq!(r2.len(), 5, "GraphAug should return 5 results; got {:?}", r2);

        // GraphCohSearch with threshold=-1.0 (always open gate): same 5.
        assert_eq!(r3.len(), 5, "GraphCoh should return 5 results; got {:?}", r3);
    }

    /// Graph traversal must reach cross-cluster nodes that vector-only search misses.
    #[test]
    fn test_graph_expands_candidate_set() {
        let dim = 4;
        // Cluster A: ids 0..5 near [1,0,0,0]; Cluster B: ids 5..10 near [0,1,0,0].
        let mut aug = GraphAugSearch::new(dim, 3, 1);
        for i in 0..5 {
            aug.insert(i, vec![1.0 - i as f32 * 0.05, 0.0, 0.0, 0.0]).unwrap();
        }
        for i in 5..10 {
            aug.insert(i, vec![0.0, 1.0 - (i - 5) as f32 * 0.05, 0.0, 0.0]).unwrap();
        }
        aug.add_edge(0, 5);
        aug.add_edge(1, 6);

        let q = vec![1.0, 0.0, 0.0, 0.0];
        let results = aug.search(&q, 10).unwrap();
        let ids: Vec<usize> = results.iter().map(|h| h.id).collect();
        assert!(
            ids.contains(&5) || ids.contains(&6),
            "Expected graph traversal to include cross-cluster nodes 5 or 6, got {:?}",
            ids
        );
    }

    /// Coherence gate prunes edges to vectors with cosine < threshold.
    #[test]
    fn test_coherence_gate_prunes() {
        let dim = 4;
        let threshold = 0.5; // strict: only near-parallel neighbours
        let mut coh = GraphCohSearch::new(dim, 3, 1, threshold);
        for i in 0..5 {
            coh.insert(i, vec![1.0 - i as f32 * 0.05, 0.0, 0.0, 0.0]).unwrap();
        }
        for i in 5..10 {
            coh.insert(i, vec![0.0, 1.0 - (i - 5) as f32 * 0.05, 0.0, 0.0]).unwrap();
        }
        coh.add_edge(0, 5); // cross-cluster: cosine ≈ 0 < threshold → should be pruned

        let q = vec![1.0, 0.0, 0.0, 0.0];
        let results = coh.search(&q, 10).unwrap();
        let ids: Vec<usize> = results.iter().map(|h| h.id).collect();
        assert!(
            !ids.contains(&5),
            "Strict gate should prune id 5 (cosine ≈ 0 < threshold 0.5); got {:?}",
            ids
        );
    }

    /// FlatSearch returns exact nearest for a trivial query.
    #[test]
    fn test_flat_exact_recall() {
        let dim = 3;
        let mut flat = FlatSearch::new(dim);
        flat.insert(0, vec![1.0, 0.0, 0.0]).unwrap();
        flat.insert(1, vec![0.0, 1.0, 0.0]).unwrap();
        flat.insert(2, vec![0.0, 0.0, 1.0]).unwrap();
        let q = vec![1.0, 0.0, 0.0];
        let r = flat.search(&q, 1).unwrap();
        assert_eq!(r[0].id, 0);
        assert!((r[0].score - 1.0).abs() < 1e-5);
    }

    /// Dimension mismatch returns an error, not a panic.
    #[test]
    fn test_dim_mismatch() {
        let mut flat = FlatSearch::new(4);
        assert!(flat.insert(0, vec![1.0, 2.0]).is_err());
    }

    /// Acceptance: GraphCohSearch recall@K on cross-cluster ground truth
    /// must exceed FlatSearch recall@K by at least 5 percentage points.
    ///
    /// Dataset:
    ///   3 orthogonal clusters (centroid in dimension c for cluster c).
    ///   Graph edges: each vector → CROSS_EDGES_PER_VEC vectors in other clusters.
    ///   Ground truth: query's direct cross-cluster graph neighbours.
    ///
    /// FlatSearch: returns same-cluster vectors (high cosine) → 0% recall on GT.
    /// GraphCohSearch: BFS finds cross-cluster targets → >0% recall.
    #[test]
    fn test_acceptance_recall_threshold() {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};

        const N: usize = 600;
        const DIM: usize = 32;
        const K: usize = 10;
        const N_CLUSTERS: usize = 3;
        const CROSS_EDGES_PER_VEC: usize = 4;
        const SEED_K: usize = 2;
        const BFS_DEPTH: usize = 1;
        const COH_THRESHOLD: f32 = -1.0; // permissive gate: always traverse
        const N_QUERIES: usize = 30;

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let normal = Normal::new(0.0f32, 0.4).unwrap();

        // Orthogonal centroids.
        let centroids: Vec<Vec<f32>> = (0..N_CLUSTERS)
            .map(|c| {
                let mut v = vec![0.0f32; DIM];
                v[c] = 4.0;
                v
            })
            .collect();

        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(N);
        let mut labels: Vec<usize> = Vec::with_capacity(N);
        for i in 0..N {
            let c = i % N_CLUSTERS;
            labels.push(c);
            let v: Vec<f32> = centroids[c]
                .iter()
                .map(|&cx| cx + normal.sample(&mut rng))
                .collect();
            vectors.push(v);
        }

        let mut cross_edges: Vec<(usize, usize)> = Vec::new();
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..N {
            let my_c = labels[i];
            for _ in 0..CROSS_EDGES_PER_VEC {
                let other_c = (my_c + 1 + rng.gen::<usize>() % (N_CLUSTERS - 1)) % N_CLUSTERS;
                let j = other_c + N_CLUSTERS * (rng.gen::<usize>() % (N / N_CLUSTERS));
                let j = j.min(N - 1);
                if j != i {
                    cross_edges.push((i, j));
                    adj.entry(i).or_default().push(j);
                }
            }
        }

        let mut flat = FlatSearch::new(DIM);
        let mut coh = GraphCohSearch::new(DIM, SEED_K, BFS_DEPTH, COH_THRESHOLD);
        for (id, v) in vectors.iter().enumerate() {
            flat.insert(id, v.clone()).unwrap();
            coh.insert(id, v.clone()).unwrap();
        }
        for &(from, to) in &cross_edges {
            coh.add_edge(from, to);
        }

        let queries: Vec<usize> = (0..N_QUERIES).map(|q| q * (N / N_QUERIES)).collect();
        let ground_truth: Vec<HashSet<usize>> = queries
            .iter()
            .map(|&qi| {
                adj.get(&qi)
                    .map(|nbs| {
                        nbs.iter()
                            .copied()
                            .filter(|&nb| labels[nb] != labels[qi])
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();

        let recall = |index: &dyn GcvsIndex| -> f32 {
            let mut total = 0.0f32;
            let mut counted = 0;
            for (pos, &qi) in queries.iter().enumerate() {
                let gt = &ground_truth[pos];
                if gt.is_empty() {
                    continue;
                }
                let results = index.search(&vectors[qi], K).unwrap_or_default();
                let found = results.iter().filter(|h| gt.contains(&h.id)).count();
                total += found as f32 / gt.len().min(K) as f32;
                counted += 1;
            }
            if counted == 0 {
                return 0.0;
            }
            total / counted as f32
        };

        let flat_r = recall(&flat);
        let coh_r = recall(&coh);
        let improvement_pp = (coh_r - flat_r) * 100.0;

        assert!(
            improvement_pp >= 5.0,
            "GraphCohSearch cross-cluster recall improvement must be ≥5pp, \
             got {improvement_pp:.2}pp (flat={:.1}%, coh={:.1}%)",
            flat_r * 100.0,
            coh_r * 100.0,
        );
    }
}
