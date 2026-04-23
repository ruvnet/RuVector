//! End-to-end smoke tests for the ruLake federation intermediary.
//!
//! Acceptance gates for M1 in `docs/research/ruLake/07-implementation-plan.md`:
//!
//! 1. A query through `RuLake::search_one` over `LocalBackend` returns
//!    the same top-k ids as a direct `RabitqPlusIndex::search` on the
//!    same data, modulo tie ordering.
//! 2. Cache coherence — mutating the backend bumps the generation;
//!    the next search re-primes the cache automatically.
//! 3. Federated search — fanning out across two backends and merging
//!    by score gives the globally-correct top-k.
//! 4. Cache-hit path is significantly faster than cache-miss
//!    (measurement-level check, not just a correctness check).

use std::sync::Arc;
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

use ruvector_rabitq::{AnnIndex, RabitqPlusIndex};
use ruvector_rulake::{cache::Consistency, LocalBackend, RuLake};

fn clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let centroid = Uniform::new(-2.0f32, 2.0);
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..d).map(|_| centroid.sample(&mut rng)).collect())
        .collect();
    let noise = Normal::new(0.0f64, 0.6).unwrap();
    (0..n)
        .map(|_| {
            let c = &centroids[rng.gen_range(0..n_clusters)];
            c.iter()
                .map(|&x| x + noise.sample(&mut rng) as f32)
                .collect()
        })
        .collect()
}

#[test]
fn rulake_matches_direct_rabitq_on_local_backend() {
    // M1 acceptance #1: ruLake's search output matches a direct call to
    // RabitqPlusIndex::search, given the same seed + rerank factor.
    let d = 64;
    let n = 1_000;
    let rerank = 20;
    let seed = 42;

    let data = clustered(n, d, 10, seed);
    let ids: Vec<u64> = (0..n as u64).collect();

    // Reference: direct RaBitQ.
    let mut direct = RabitqPlusIndex::new(d, seed, rerank);
    for (i, v) in data.iter().enumerate() {
        direct.add(i, v.clone()).unwrap();
    }

    // ruLake with a LocalBackend.
    let backend = Arc::new(LocalBackend::new("local-a"));
    backend
        .put_collection("demo", d, ids, data.clone())
        .unwrap();
    let lake = RuLake::new(rerank, seed);
    lake.register_backend(backend).unwrap();

    // Prime the cache + query.
    let query = clustered(1, d, 10, 999)[0].clone();
    let direct_hits = direct.search(&query, 10).unwrap();
    let lake_hits = lake.search_one("local-a", "demo", &query, 10).unwrap();
    assert_eq!(lake_hits.len(), direct_hits.len());

    // Ids should match (positions are dense 0..n, so direct's usize
    // matches lake's u64 after the pos_to_id mapping).
    let direct_ids: Vec<usize> = direct_hits.iter().map(|r| r.id).collect();
    let lake_ids: Vec<u64> = lake_hits.iter().map(|r| r.id).collect();
    let direct_as_u64: Vec<u64> = direct_ids.iter().map(|&x| x as u64).collect();
    assert_eq!(direct_as_u64, lake_ids);

    // Scores should match too — same seed + same rerank factor means
    // the two caches are byte-identical.
    for (a, b) in direct_hits.iter().zip(lake_hits.iter()) {
        assert!(
            (a.score - b.score).abs() < 1e-4,
            "direct {} vs lake {} — estimator divergence",
            a.score,
            b.score
        );
    }
}

#[test]
fn rulake_recomputes_on_backend_generation_bump() {
    // M1 acceptance #2: mutating the backend bumps the generation; the
    // next search observes the new state.
    let d = 32;
    let rerank = 20;
    let seed = 7;

    let backend = Arc::new(LocalBackend::new("local-mut"));
    backend
        .put_collection("c1", d, vec![10], vec![vec![1.0; d]])
        .unwrap();

    let lake = RuLake::new(rerank, seed);
    lake.register_backend(backend.clone()).unwrap();

    // First search — miss, prime.
    let q = vec![1.0; d];
    let r1 = lake.search_one("local-mut", "c1", &q, 1).unwrap();
    assert_eq!(r1[0].id, 10);
    let s1 = lake.cache_stats();
    assert_eq!(s1.primes, 1);
    assert_eq!(s1.misses, 1);
    assert_eq!(s1.hits, 0);

    // Second search, same data — hit.
    lake.search_one("local-mut", "c1", &q, 1).unwrap();
    let s2 = lake.cache_stats();
    assert_eq!(s2.primes, 1, "no re-prime on cache hit");
    assert_eq!(s2.hits, 1);

    // Mutate the backend — append a *closer* vector with a new id.
    backend.append("c1", 42, vec![1.0; d]).unwrap();

    // Third search — backend generation bumped → cache invalidated +
    // re-primed. The new vector is a tie with the old one (same
    // coordinates), so at k=2 both are returned.
    let r3 = lake.search_one("local-mut", "c1", &q, 2).unwrap();
    let ids: std::collections::HashSet<u64> = r3.iter().map(|r| r.id).collect();
    assert!(ids.contains(&42), "new vector not observed after bump");
    let s3 = lake.cache_stats();
    assert_eq!(s3.primes, 2, "re-prime on generation bump");
}

#[test]
fn rulake_federates_across_two_backends() {
    // M1 acceptance #3: fan-out across two backends, merge by score.
    let d = 16;
    let rerank = 20;
    let seed = 3;

    // Build a fake global index by concatenating the two backends' data,
    // so we know the correct federated top-k.
    let a_data: Vec<Vec<f32>> = (0..50)
        .map(|i| (0..d).map(|j| (i + j) as f32).collect())
        .collect();
    let b_data: Vec<Vec<f32>> = (0..50)
        .map(|i| (0..d).map(|j| ((i * 2) + j) as f32).collect())
        .collect();

    let ba = Arc::new(LocalBackend::new("bq-like"));
    ba.put_collection("t1", d, (0..a_data.len() as u64).collect(), a_data.clone())
        .unwrap();

    let bb = Arc::new(LocalBackend::new("snowflake-like"));
    bb.put_collection(
        "t2",
        d,
        (1_000..1_000 + b_data.len() as u64).collect(),
        b_data.clone(),
    )
    .unwrap();

    let lake = RuLake::new(rerank, seed);
    lake.register_backend(ba).unwrap();
    lake.register_backend(bb).unwrap();

    // Query close to a specific a_data[7] row.
    let query = (0..d).map(|j| (7 + j) as f32 + 0.1).collect::<Vec<_>>();
    let hits = lake
        .search_federated(&[("bq-like", "t1"), ("snowflake-like", "t2")], &query, 5)
        .unwrap();
    assert_eq!(hits.len(), 5);
    // Top hit should be from `bq-like/t1` with id 7 (closest).
    assert_eq!(hits[0].backend, "bq-like");
    assert_eq!(hits[0].collection, "t1");
    assert_eq!(hits[0].id, 7);

    // Results must be sorted by score ascending.
    for w in hits.windows(2) {
        assert!(w[0].score <= w[1].score);
    }
}

#[test]
fn cache_hit_is_faster_than_miss() {
    // M1 acceptance #4: cache hits are meaningfully cheaper than misses.
    // We don't assert a specific speedup (CI noise) — we assert that
    // miss_time is greater than hit_time, both sub-second, at n=500.
    let d = 64;
    let n = 500;
    let rerank = 20;
    let seed = 13;

    let data = clustered(n, d, 10, seed);
    let backend = Arc::new(LocalBackend::new("perf-demo"));
    backend
        .put_collection("c", d, (0..n as u64).collect(), data.clone())
        .unwrap();

    // Eventual consistency so we don't round-trip to the backend on
    // every hit.
    let lake = RuLake::new(rerank, seed).with_consistency(Consistency::Eventual { ttl_ms: 60_000 });
    lake.register_backend(backend).unwrap();

    let query = clustered(1, d, 10, 99)[0].clone();

    // First search — cache miss.
    let t = Instant::now();
    let _ = lake.search_one("perf-demo", "c", &query, 10).unwrap();
    let t_miss = t.elapsed();

    // Subsequent searches — cache hits.
    let t = Instant::now();
    for _ in 0..20 {
        let _ = lake.search_one("perf-demo", "c", &query, 10).unwrap();
    }
    let t_hit_20 = t.elapsed();
    let t_hit_avg = t_hit_20 / 20;

    eprintln!(
        "miss={:?}  hit_avg={:?}  ratio={:.1}×",
        t_miss,
        t_hit_avg,
        t_miss.as_nanos() as f64 / t_hit_avg.as_nanos().max(1) as f64
    );
    // Miss includes building the RabitqPlusIndex over n rows; hit is
    // just a search. Miss must dominate.
    assert!(
        t_miss > t_hit_avg,
        "expected miss > hit_avg; got miss={:?} hit_avg={:?}",
        t_miss,
        t_hit_avg
    );

    let stats = lake.cache_stats();
    assert_eq!(stats.primes, 1);
    assert!(stats.hits >= 20, "want ≥20 hits, got {}", stats.hits);
}

#[test]
fn two_backends_share_cache_when_witness_matches() {
    // The reviewer's "use RVF witness chain hash as cache-key anchor"
    // fix: two backends advertising the same logical dataset (same
    // data_ref, same seed, same rerank, same generation) should share
    // ONE compressed index in the cache, not build two.
    //
    // We use a thin `SharedLocalBackend` shim that overrides
    // `current_bundle()` to report a shared `data_ref`, so two distinct
    // `LocalBackend` instances look like two pointers into the same
    // logical dataset.
    use ruvector_rulake::{BackendAdapter, Generation, PulledBatch, RuLakeBundle, RuLakeError};
    use std::sync::RwLock;

    struct SharedLocalBackend {
        inner: LocalBackend,
        shared_data_ref: String,
    }
    impl BackendAdapter for SharedLocalBackend {
        fn id(&self) -> &str {
            self.inner.id()
        }
        fn list_collections(&self) -> Result<Vec<String>, RuLakeError> {
            self.inner.list_collections()
        }
        fn pull_vectors(&self, c: &str) -> Result<PulledBatch, RuLakeError> {
            self.inner.pull_vectors(c)
        }
        fn generation(&self, c: &str) -> Result<u64, RuLakeError> {
            self.inner.generation(c)
        }
        fn current_bundle(
            &self,
            collection: &str,
            rotation_seed: u64,
            rerank_factor: usize,
        ) -> Result<RuLakeBundle, RuLakeError> {
            // Both backends report the SAME data_ref → same witness.
            let batch = self.inner.pull_vectors(collection)?;
            Ok(RuLakeBundle::new(
                self.shared_data_ref.clone(),
                batch.dim,
                rotation_seed,
                rerank_factor,
                Generation::Num(batch.generation),
            ))
        }
    }

    let d = 32;
    let n = 200;
    let data = clustered(n, d, 10, 5);

    // Two backends. Both load the SAME data (deterministically — same
    // seed and same vectors). Both report the same `data_ref`.
    let a = LocalBackend::new("region-us");
    a.put_collection("c", d, (0..n as u64).collect(), data.clone())
        .unwrap();
    let b = LocalBackend::new("region-eu");
    b.put_collection("c", d, (0..n as u64).collect(), data.clone())
        .unwrap();

    // Ensure both have generation=1 so their bundles match exactly.
    // (put_collection bumps generation from 0 to 1.)
    let a_shim = Arc::new(SharedLocalBackend {
        inner: a,
        shared_data_ref: "gs://shared/dataset.parquet".to_string(),
    });
    let b_shim = Arc::new(SharedLocalBackend {
        inner: b,
        shared_data_ref: "gs://shared/dataset.parquet".to_string(),
    });

    // Sanity: the _lock_ is required because BackendAdapter's default
    // current_bundle() pulls vectors; we don't want the test to be
    // racy. RwLock is just here to shut the borrow checker up on the
    // unused field — the real backends are the shims above.
    let _locked: RwLock<()> = RwLock::new(());

    let lake = RuLake::new(20, 42);
    lake.register_backend(a_shim).unwrap();
    lake.register_backend(b_shim).unwrap();

    let q = vec![0.5_f32; d];
    let _ = lake.search_one("region-us", "c", &q, 5).unwrap();
    let _ = lake.search_one("region-eu", "c", &q, 5).unwrap();

    let stats = lake.cache_stats();
    assert_eq!(stats.primes, 1, "second backend should not re-prime");
    assert_eq!(
        stats.shared_hits, 1,
        "second backend should resolve to the shared witness"
    );

    // Both pointers must exist but they both resolve to the same
    // witness, so only ONE compressed entry should be in the pool.
    let wa = lake
        .cache_witness_of(&("region-us".to_string(), "c".to_string()))
        .unwrap();
    let wb = lake
        .cache_witness_of(&("region-eu".to_string(), "c".to_string()))
        .unwrap();
    assert_eq!(wa, wb, "witnesses must match");
    assert_eq!(lake.cache_entry_count(), 1);
    assert_eq!(lake.cache_refcount_of(&wa), 2);
}

#[test]
fn dimension_mismatch_returns_error() {
    let d = 8;
    let backend = Arc::new(LocalBackend::new("tiny"));
    backend
        .put_collection("c", d, vec![1], vec![vec![0.0; d]])
        .unwrap();
    let lake = RuLake::new(20, 0);
    lake.register_backend(backend).unwrap();
    let bad_query = vec![0.0; d + 1];
    let err = lake.search_one("tiny", "c", &bad_query, 1).unwrap_err();
    assert!(matches!(
        err,
        ruvector_rulake::RuLakeError::DimensionMismatch { .. }
    ));
}

#[test]
fn unknown_backend_returns_error() {
    let lake = RuLake::new(20, 0);
    let err = lake.search_one("nope", "nope", &[0.0; 4], 1).unwrap_err();
    assert!(matches!(
        err,
        ruvector_rulake::RuLakeError::UnknownBackend(_)
    ));
}

#[test]
fn rulake_recall_at_10_above_90pct_vs_brute_force() {
    // The MVP's real recall gate: against brute-force top-10 on the
    // same query set, does ruLake return the same neighbours? We use
    // rerank×20 + clustered Gaussian — the regime `ruvector-rabitq`
    // documents as 100 % recall@10 at n ≥ 5 k. The intermediary
    // doesn't change the estimator, so this should pass with room.
    // Match BENCHMARK.md methodology: generate n+nq from the same seed
    // and split so queries share the data's cluster centroids.
    // Different seeds would test a distribution-shift regime — valuable
    // but distinct from the "recall against known-good RaBitQ"
    // correctness gate this test is meant to hold.
    let d = 128;
    let n = 5_000;
    let nq = 50;
    let rerank = 20;
    let seed = 101;

    let all = clustered(n + nq, d, 100, seed);
    let data = all[..n].to_vec();
    let queries = all[n..].to_vec();

    let backend = Arc::new(LocalBackend::new("recall-test"));
    backend
        .put_collection("c", d, (0..n as u64).collect(), data.clone())
        .unwrap();
    let lake = RuLake::new(rerank, seed);
    lake.register_backend(backend).unwrap();

    let mut hits = 0usize;
    for q in &queries {
        // Ground truth: brute-force top-10 L2².
        let mut scored: Vec<(usize, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let s: f32 = q
                    .iter()
                    .zip(v.iter())
                    .map(|(&a, &b)| (a - b) * (a - b))
                    .sum();
                (i, s)
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        let truth: std::collections::HashSet<u64> =
            scored.iter().take(10).map(|(i, _)| *i as u64).collect();

        let got = lake.search_one("recall-test", "c", q, 10).unwrap();
        hits += got.iter().filter(|r| truth.contains(&r.id)).count();
    }
    let recall = hits as f64 / (nq * 10) as f64;
    assert!(
        recall > 0.90,
        "ruLake recall@10 = {:.1}% — expected > 90 % on clustered/rerank×20",
        recall * 100.0
    );
}

#[test]
fn lru_eviction_caps_entry_count_when_pointers_dropped() {
    // With max_entries=2 and three distinct witnesses primed + then
    // invalidated (dropping the pointer → refcount=0), LRU should
    // keep the pool at ≤ 2 entries.
    let d = 8;

    let a = Arc::new(LocalBackend::new("a"));
    let b = Arc::new(LocalBackend::new("b"));
    let c = Arc::new(LocalBackend::new("c"));
    for (i, back) in [&a, &b, &c].iter().enumerate() {
        back.put_collection("c", d, vec![i as u64], vec![vec![(i as f32); d]])
            .unwrap();
    }
    let lake = RuLake::new(5, 42).with_max_cache_entries(2);
    lake.register_backend(a).unwrap();
    lake.register_backend(b).unwrap();
    lake.register_backend(c).unwrap();

    let q = vec![0.0; d];
    lake.search_one("a", "c", &q, 1).unwrap();
    lake.search_one("b", "c", &q, 1).unwrap();
    lake.search_one("c", "c", &q, 1).unwrap();
    // Three pointers → three entries (all pinned).
    assert_eq!(lake.cache_entry_count(), 3);

    // Drop pointer `a` → its entry becomes unpinned.
    lake.invalidate_cache(&("a".to_string(), "c".to_string()));
    // Still 2 pinned (b, c); the now-unpinned `a`-witness entry would
    // put us at 3 entries, but max_entries=2 evicts it.
    // Re-prime `b` to trigger the cap check.
    lake.search_one("b", "c", &q, 1).unwrap();
    assert!(
        lake.cache_entry_count() <= 2,
        "lru cap violated: entry_count={}",
        lake.cache_entry_count()
    );
}

#[test]
fn unknown_collection_returns_error() {
    let backend = Arc::new(LocalBackend::new("b"));
    let lake = RuLake::new(20, 0);
    lake.register_backend(backend).unwrap();
    let err = lake.search_one("b", "missing", &[0.0; 4], 1).unwrap_err();
    // Error surfaces via the backend's generation() call.
    assert!(matches!(
        err,
        ruvector_rulake::RuLakeError::UnknownCollection { .. }
    ));
}
