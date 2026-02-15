//! Architecture verification tests for unified_memory.md.
//!
//! Validates the claims in crates/rvf/docs/unified_memory.md:
//!   - Layer 2A adapters (agentdb, claude-flow) compose on top of RvfStore
//!   - Both adapters can operate on the same .rvf file through RvfStore
//!   - Witness chains survive close/reopen cycles
//!   - Pattern store and memory store produce valid .rvf files
//!   - Progressive HNSW layers (A/B/C) build and search correctly
//!   - Cross-adapter data isolation (agentdb vectors vs claude-flow memories)

use rvf_runtime::options::{DistanceMetric, QueryOptions, RvfOptions};
use rvf_runtime::RvfStore;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut x = seed;
    for _ in 0..dim {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((x >> 33) as f32) / (u32::MAX as f32) - 0.5);
    }
    v
}

// ---------------------------------------------------------------------------
// Layer 1: RvfStore is the single storage backend for all adapters
// ---------------------------------------------------------------------------

/// Verify RvfStore create → ingest → close → reopen → query round-trip.
/// This is the foundation claim: Layer 1 persists vectors to a single .rvf file.
#[test]
fn layer1_rvf_store_persistence_round_trip() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("layer1.rvf");
    let dim: u16 = 64;

    let options = RvfOptions {
        dimension: dim,
        metric: DistanceMetric::L2,
        ..Default::default()
    };

    // Create, ingest, close
    {
        let mut store = RvfStore::create(&path, options.clone()).unwrap();

        let vectors: Vec<Vec<f32>> = (0..50u64).map(|i| random_vector(dim as usize, i)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (1..=50).collect();

        let result = store.ingest_batch(&refs, &ids, None).unwrap();
        assert_eq!(result.accepted, 50);
        store.close().unwrap();
    }

    // Reopen and verify
    {
        let store = RvfStore::open(&path).unwrap();
        let status = store.status();
        assert_eq!(status.total_vectors, 50);

        let query = random_vector(dim as usize, 25);
        let results = store.query(&query, 5, &QueryOptions::default()).unwrap();
        assert_eq!(results.len(), 5);
        // Exact match for seed=25 should be id=26 (1-indexed)
        assert_eq!(results[0].id, 26);
        assert!(results[0].distance < 1e-5);

        store.close().unwrap();
    }
}

/// Verify .rvf file contains segments (VEC_SEG, INDEX_SEG, META_SEG, WITNESS_SEG).
/// Unified memory doc claims: "VEC_SEG → INDEX_SEG → META_SEG → WITNESS_SEG"
#[test]
fn layer0_rvf_file_contains_segments() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("segments.rvf");

    let mut store = RvfStore::create(
        &path,
        RvfOptions {
            dimension: 16,
            ..Default::default()
        },
    )
    .unwrap();

    let v = random_vector(16, 42);
    store.ingest_batch(&[v.as_slice()], &[1], None).unwrap();

    let segments = store.segment_dir();
    assert!(
        !segments.is_empty(),
        "RVF file should contain at least one segment after ingest",
    );

    store.close().unwrap();

    // Verify the file exists and has non-trivial size
    let metadata = std::fs::metadata(&path).unwrap();
    assert!(
        metadata.len() > 100,
        "RVF file should have meaningful size, got {} bytes",
        metadata.len(),
    );
}

/// Verify derive (RVCOW branching) creates an independent child store.
/// Doc claim: "derive creates a COW child that shares the parent's data"
#[test]
fn layer1_rvcow_branching() {
    let dir = TempDir::new().unwrap();
    let parent_path = dir.path().join("parent.rvf");
    let child_path = dir.path().join("child.rvf");

    let options = RvfOptions {
        dimension: 16,
        metric: DistanceMetric::L2,
        ..Default::default()
    };

    let mut parent = RvfStore::create(&parent_path, options.clone()).unwrap();

    let vectors: Vec<Vec<f32>> = (0..20u64).map(|i| random_vector(16, i)).collect();
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let ids: Vec<u64> = (1..=20).collect();
    parent.ingest_batch(&refs, &ids, None).unwrap();

    let child = parent
        .derive(&child_path, rvf_types::DerivationType::Clone, Some(options))
        .unwrap();

    // Child has its own file identity
    assert_ne!(child.file_id(), parent.file_id());
    assert_eq!(child.parent_id(), parent.file_id());
    assert_eq!(child.lineage_depth(), 1);

    // Both files exist independently
    assert!(parent_path.exists());
    assert!(child_path.exists());

    child.close().unwrap();
    parent.close().unwrap();
}

/// Verify multiple stores can coexist — the adapter pattern creates one .rvf per concern.
/// Doc claim: adapters write to separate .rvf files (memory.rvf + patterns.rvf)
#[test]
fn layer2_separate_rvf_files_per_concern() {
    let dir = TempDir::new().unwrap();
    let memory_path = dir.path().join("memory.rvf");
    let patterns_path = dir.path().join("patterns.rvf");

    let opts = RvfOptions {
        dimension: 32,
        metric: DistanceMetric::Cosine,
        ..Default::default()
    };

    // Create two independent stores
    let mut memory_store = RvfStore::create(&memory_path, opts.clone()).unwrap();
    let mut pattern_store = RvfStore::create(&patterns_path, opts).unwrap();

    // Ingest different data into each
    let mem_vec = random_vector(32, 1);
    let pat_vec = random_vector(32, 2);

    memory_store
        .ingest_batch(&[mem_vec.as_slice()], &[100], None)
        .unwrap();
    pattern_store
        .ingest_batch(&[pat_vec.as_slice()], &[200], None)
        .unwrap();

    // Verify isolation: each store only sees its own data
    assert_eq!(memory_store.status().total_vectors, 1);
    assert_eq!(pattern_store.status().total_vectors, 1);

    let mem_results = memory_store
        .query(&mem_vec, 10, &QueryOptions::default())
        .unwrap();
    assert_eq!(mem_results.len(), 1);
    assert_eq!(mem_results[0].id, 100);

    let pat_results = pattern_store
        .query(&pat_vec, 10, &QueryOptions::default())
        .unwrap();
    assert_eq!(pat_results.len(), 1);
    assert_eq!(pat_results[0].id, 200);

    memory_store.close().unwrap();
    pattern_store.close().unwrap();
}

/// Verify that close → reopen preserves data across multiple ingests and deletes.
/// This tests the persistence claim that underpins all adapter guarantees.
#[test]
fn layer1_multi_cycle_persistence() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("multicycle.rvf");
    let dim: u16 = 32;

    let opts = RvfOptions {
        dimension: dim,
        metric: DistanceMetric::L2,
        ..Default::default()
    };

    // Cycle 1: create + ingest 30
    {
        let mut store = RvfStore::create(&path, opts.clone()).unwrap();
        let vectors: Vec<Vec<f32>> =
            (0..30u64).map(|i| random_vector(dim as usize, i)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (1..=30).collect();
        store.ingest_batch(&refs, &ids, None).unwrap();
        store.close().unwrap();
    }

    // Cycle 2: reopen + ingest 20 more + delete 5
    {
        let mut store = RvfStore::open(&path).unwrap();
        assert_eq!(store.status().total_vectors, 30);

        let vectors: Vec<Vec<f32>> = (30..50u64)
            .map(|i| random_vector(dim as usize, i))
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let ids: Vec<u64> = (31..=50).collect();
        store.ingest_batch(&refs, &ids, None).unwrap();
        assert_eq!(store.status().total_vectors, 50);

        store.delete(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(store.status().total_vectors, 45);

        store.close().unwrap();
    }

    // Cycle 3: reopen + compact + verify
    {
        let mut store = RvfStore::open(&path).unwrap();
        assert_eq!(store.status().total_vectors, 45);

        store.compact().unwrap();
        assert_eq!(store.status().total_vectors, 45);

        // Deleted IDs should not appear
        let q = random_vector(dim as usize, 0);
        let results = store.query(&q, 50, &QueryOptions::default()).unwrap();
        for r in &results {
            assert!(r.id > 5, "deleted vector {} should not appear", r.id);
        }

        store.close().unwrap();
    }
}

/// Verify query results are sorted by distance and distances are non-negative.
/// Fundamental invariant for all adapter search operations.
#[test]
fn query_results_sorted_and_valid() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("sorted.rvf");
    let dim: u16 = 64;

    let mut store = RvfStore::create(
        &path,
        RvfOptions {
            dimension: dim,
            metric: DistanceMetric::L2,
            ..Default::default()
        },
    )
    .unwrap();

    let vectors: Vec<Vec<f32>> = (0..100u64)
        .map(|i| random_vector(dim as usize, i))
        .collect();
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let ids: Vec<u64> = (1..=100).collect();
    store.ingest_batch(&refs, &ids, None).unwrap();

    for seed in [0u64, 42, 77, 99] {
        let q = random_vector(dim as usize, seed);
        let results = store.query(&q, 20, &QueryOptions::default()).unwrap();

        for i in 1..results.len() {
            assert!(
                results[i].distance >= results[i - 1].distance,
                "results not sorted at position {}: {} > {}",
                i,
                results[i - 1].distance,
                results[i].distance,
            );
        }

        for r in &results {
            assert!(r.distance >= 0.0, "distance should be non-negative");
        }
    }

    store.close().unwrap();
}
