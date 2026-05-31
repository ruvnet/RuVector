//! Deterministic synthetic dataset generation for benchmarking.
//!
//! Two dataset shapes are provided:
//! - `generate_store` / `generate_queries`: clustered Gaussian data where queries
//!   share cluster centres with the store (meaningful recall@K measurement).
//! - `generate_redundant_store`: simulates realistic agent memory — each
//!   semantic concept has many near-duplicate copies (high redundancy), which is
//!   the primary use case where graph-cut compaction outperforms simpler strategies.

use crate::compactor::{l2_normalize, MemoryEntry, MemoryStore};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Parameters for generating a clustered Gaussian dataset.
pub struct DatasetParams {
    pub n: usize,
    pub dims: usize,
    pub n_clusters: usize,
    /// Within-cluster standard deviation.
    pub cluster_std: f32,
    pub seed: u64,
}

impl Default for DatasetParams {
    fn default() -> Self {
        Self {
            n: 5_000,
            dims: 128,
            n_clusters: 20,
            cluster_std: 0.25,
            seed: 42,
        }
    }
}

/// Build a `MemoryStore` from clustered Gaussian vectors.
///
/// Cluster centres are shared with `generate_queries` (same `params.seed`) so
/// that recall@K measures how well compaction preserves same-cluster retrieval.
pub fn generate_store(params: &DatasetParams) -> MemoryStore {
    let mut rng_centres = StdRng::seed_from_u64(params.seed);
    let centres: Vec<Vec<f32>> = (0..params.n_clusters)
        .map(|_| random_unit_vector(&mut rng_centres, params.dims))
        .collect();

    let mut rng_noise = StdRng::seed_from_u64(params.seed.wrapping_add(1));
    let mut store = MemoryStore::new(params.dims);

    for i in 0..params.n {
        let cluster_id = i % params.n_clusters;
        let centre = &centres[cluster_id];
        let mut v: Vec<f32> = centre
            .iter()
            .map(|&c| c + (rng_noise.gen::<f32>() * 2.0 - 1.0) * params.cluster_std)
            .collect();
        l2_normalize(&mut v);

        store.push(MemoryEntry {
            id: i as u64,
            vector: v,
            age_tick: i as u64,
            access_count: rng_noise.gen_range(0..10),
        });
    }
    store
}

/// Generate query vectors from the same cluster distribution as `generate_store`.
pub fn generate_queries(params: &DatasetParams, n_queries: usize) -> Vec<Vec<f32>> {
    let mut rng_centres = StdRng::seed_from_u64(params.seed);
    let centres: Vec<Vec<f32>> = (0..params.n_clusters)
        .map(|_| random_unit_vector(&mut rng_centres, params.dims))
        .collect();

    let mut rng_noise = StdRng::seed_from_u64(params.seed.wrapping_add(999_999));
    (0..n_queries)
        .map(|i| {
            let cluster_id = i % params.n_clusters;
            let centre = &centres[cluster_id];
            let mut v: Vec<f32> = centre
                .iter()
                .map(|&c| c + (rng_noise.gen::<f32>() * 2.0 - 1.0) * params.cluster_std)
                .collect();
            l2_normalize(&mut v);
            v
        })
        .collect()
}

/// Parameters for the high-redundancy agent-memory dataset.
pub struct RedundantParams {
    /// Number of distinct semantic concepts.
    pub n_concepts: usize,
    /// Number of near-duplicate copies per concept.
    pub copies_per_concept: usize,
    pub dims: usize,
    /// Very small noise for near-duplicates (σ ≈ 0.03 simulates OCR/re-embed noise).
    pub dup_noise: f32,
    pub seed: u64,
}

impl Default for RedundantParams {
    fn default() -> Self {
        Self {
            n_concepts: 50,
            copies_per_concept: 20,
            dims: 64,
            dup_noise: 0.04,
            seed: 1337,
        }
    }
}

/// Build a high-redundancy store: each semantic concept has `copies_per_concept`
/// nearly-identical vector embeddings — exactly the redundancy that accumulates
/// in long-running agent memory loops.
pub fn generate_redundant_store(params: &RedundantParams) -> MemoryStore {
    let mut rng_centres = StdRng::seed_from_u64(params.seed);
    let centres: Vec<Vec<f32>> = (0..params.n_concepts)
        .map(|_| random_unit_vector(&mut rng_centres, params.dims))
        .collect();

    let mut rng_noise = StdRng::seed_from_u64(params.seed.wrapping_add(7));
    let mut store = MemoryStore::new(params.dims);
    let mut id = 0u64;

    for (concept, centre) in centres.iter().enumerate() {
        for copy in 0..params.copies_per_concept {
            let mut v: Vec<f32> = centre
                .iter()
                .map(|&c| c + (rng_noise.gen::<f32>() * 2.0 - 1.0) * params.dup_noise)
                .collect();
            l2_normalize(&mut v);
            store.push(MemoryEntry {
                id,
                vector: v,
                // Older copies have lower ticks (concept-then-copy ordering).
                age_tick: (concept * params.copies_per_concept + copy) as u64,
                access_count: rng_noise.gen_range(0..5),
            });
            id += 1;
        }
    }
    store
}

/// For the redundant store, query using the concept centres directly.
pub fn generate_concept_queries(params: &RedundantParams) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(params.seed);
    (0..params.n_concepts)
        .map(|_| random_unit_vector(&mut rng, params.dims))
        .collect()
}

fn random_unit_vector(rng: &mut StdRng, dims: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    l2_normalize(&mut v);
    v
}
