//! Deterministic synthetic agent-memory corpus with unequal-size semantic
//! clusters (including a minority cluster), decoupled recency/frequency
//! signals, and a recency-biased "focus cluster" standing in for the
//! agent's most recent working topic.

use crate::search::{cosine, normalize, top_k_by_cosine};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub id: u64,
    pub embedding: Vec<f32>,
    /// Ground-truth cluster label. Not visible to any compaction policy —
    /// used only for evaluation.
    pub cluster: usize,
    /// 0 = oldest, n-1 = most recently accessed.
    pub last_accessed_tick: u32,
    pub access_count: u32,
}

#[derive(Debug, Clone)]
pub struct Query {
    pub embedding: Vec<f32>,
    pub cluster: usize,
    /// True top-k nearest memory ids in the *full, uncompacted* corpus.
    pub ground_truth: Vec<u64>,
}

pub struct Corpus {
    pub records: Vec<MemoryRecord>,
    pub queries: Vec<Query>,
    pub cluster_sizes: Vec<usize>,
    pub focus_cluster: usize,
    pub dims: usize,
}

pub struct CorpusConfig {
    pub n: usize,
    pub dims: usize,
    /// Fractional size of each cluster; renormalized to sum to 1 and
    /// rounded to integer counts that sum to exactly `n`.
    pub cluster_fracs: Vec<f64>,
    pub queries_per_cluster: usize,
    pub k: usize,
    pub noise_std: f32,
    pub seed: u64,
}

impl Default for CorpusConfig {
    fn default() -> Self {
        Self {
            n: 4000,
            dims: 64,
            // 6 clusters: majority, three mid-size, and two minorities
            // (5% and 2% of the corpus) that a global top-score compactor
            // is free to drop entirely.
            cluster_fracs: vec![0.35, 0.25, 0.18, 0.15, 0.05, 0.02],
            queries_per_cluster: 25,
            k: 10,
            // Calibrated (see mincut_exact.rs / examples used during this
            // nightly's development): at noise_std=0.35 the kNN graph's
            // true global min cut degenerately isolates a single outlier
            // vertex (normalized_cut ~0.76, no real topic boundary is the
            // weakest link). At 0.25 the min cut cleanly isolates one whole
            // semantic cluster (normalized_cut ~0.07) — a graph structure
            // this crate's hypothesis can actually be tested against.
            noise_std: 0.25,
            seed: 42,
        }
    }
}

fn random_unit_vector(rng: &mut StdRng, dims: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    normalize(&mut v);
    v
}

/// Sample `k` centroids on the unit sphere, retrying (bounded) whenever a
/// new centroid is too close to an existing one so clusters stay separable.
fn well_separated_centroids(
    rng: &mut StdRng,
    dims: usize,
    k: usize,
    max_sim: f32,
) -> Vec<Vec<f32>> {
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    for _ in 0..k {
        let mut best = random_unit_vector(rng, dims);
        for _attempt in 0..200 {
            let worst_sim = centroids
                .iter()
                .map(|c| cosine(&best, c))
                .fold(f32::NEG_INFINITY, f32::max);
            if centroids.is_empty() || worst_sim <= max_sim {
                break;
            }
            best = random_unit_vector(rng, dims);
        }
        centroids.push(best);
    }
    centroids
}

fn cluster_counts(n: usize, fracs: &[f64]) -> Vec<usize> {
    let total: f64 = fracs.iter().sum();
    let mut counts: Vec<usize> = fracs
        .iter()
        .map(|f| ((f / total) * n as f64).round() as usize)
        .collect();
    let assigned: usize = counts.iter().sum();
    if assigned != n {
        // Reconcile rounding drift against the largest cluster so every
        // other cluster keeps its intended (possibly minority) size.
        let (largest_idx, _) = counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| **c)
            .expect("cluster_fracs must be non-empty");
        let diff = n as i64 - assigned as i64;
        counts[largest_idx] = (counts[largest_idx] as i64 + diff).max(0) as usize;
    }
    counts
}

pub fn generate(cfg: &CorpusConfig) -> Corpus {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let num_clusters = cfg.cluster_fracs.len();
    let centroids = well_separated_centroids(&mut rng, cfg.dims, num_clusters, 0.35);
    let counts = cluster_counts(cfg.n, &cfg.cluster_fracs);
    // The largest cluster stands in for "what the agent was just working
    // on" — recency is biased toward it below, decoupled from frequency.
    let focus_cluster = counts
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| **c)
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut records: Vec<MemoryRecord> = Vec::with_capacity(cfg.n);
    let mut id = 0u64;
    for (cluster, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            let mut emb: Vec<f32> = centroids[cluster]
                .iter()
                .map(|c| c + rng.gen_range(-cfg.noise_std..cfg.noise_std))
                .collect();
            normalize(&mut emb);
            let access_count = 1 + (rng.gen::<f32>().powf(3.0) * 50.0) as u32;
            records.push(MemoryRecord {
                id,
                embedding: emb,
                cluster,
                last_accessed_tick: 0, // assigned below
                access_count,
            });
            id += 1;
        }
    }

    // Recency: sample a tick-order key per record, biased for the focus
    // cluster, then rank into a dense 0..n-1 permutation so ticks stay
    // decoupled from frequency and only softly correlated with cluster.
    let mut keyed: Vec<(usize, f64)> = records
        .iter()
        .enumerate()
        .map(|(idx, r)| {
            let key = if r.cluster == focus_cluster {
                rng.gen_range(0.5f64..1.0)
            } else {
                rng.gen_range(0.0f64..0.85)
            };
            (idx, key)
        })
        .collect();
    keyed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (tick, (idx, _)) in keyed.into_iter().enumerate() {
        records[idx].last_accessed_tick = tick as u32;
    }

    // Out-of-sample queries per cluster, with brute-force ground truth
    // against the *full* corpus computed once, up front.
    let refs: Vec<(u64, &[f32])> = records
        .iter()
        .map(|r| (r.id, r.embedding.as_slice()))
        .collect();
    let mut queries = Vec::with_capacity(num_clusters * cfg.queries_per_cluster);
    for (cluster, centroid) in centroids.iter().enumerate() {
        for _ in 0..cfg.queries_per_cluster {
            let mut emb: Vec<f32> = centroid
                .iter()
                .map(|c| c + rng.gen_range(-cfg.noise_std * 0.5..cfg.noise_std * 0.5))
                .collect();
            normalize(&mut emb);
            let ground_truth = top_k_by_cosine(&emb, &refs, cfg.k);
            queries.push(Query {
                embedding: emb,
                cluster,
                ground_truth,
            });
        }
    }

    Corpus {
        records,
        queries,
        cluster_sizes: counts,
        focus_cluster,
        dims: cfg.dims,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_requested_total_and_cluster_counts() {
        let cfg = CorpusConfig {
            n: 500,
            ..CorpusConfig::default()
        };
        let corpus = generate(&cfg);
        assert_eq!(corpus.records.len(), 500);
        assert_eq!(corpus.cluster_sizes.iter().sum::<usize>(), 500);
        assert_eq!(corpus.cluster_sizes.len(), cfg.cluster_fracs.len());
    }

    #[test]
    fn minority_cluster_is_nonempty() {
        let cfg = CorpusConfig {
            n: 1000,
            ..CorpusConfig::default()
        };
        let corpus = generate(&cfg);
        let minority = *corpus.cluster_sizes.last().unwrap();
        assert!(
            minority >= 15,
            "minority cluster too small to test: {minority}"
        );
    }

    #[test]
    fn is_deterministic_for_fixed_seed() {
        let cfg = CorpusConfig {
            n: 200,
            ..CorpusConfig::default()
        };
        let a = generate(&cfg);
        let b = generate(&cfg);
        assert_eq!(a.records[10].embedding, b.records[10].embedding);
        assert_eq!(a.queries[0].ground_truth, b.queries[0].ground_truth);
    }

    #[test]
    fn ground_truth_query_finds_own_cluster_neighbours() {
        let cfg = CorpusConfig {
            n: 400,
            ..CorpusConfig::default()
        };
        let corpus = generate(&cfg);
        let q = &corpus.queries[0];
        let hit_own_cluster = q
            .ground_truth
            .iter()
            .filter(|id| corpus.records[**id as usize].cluster == q.cluster)
            .count();
        // Centroids are separated (cosine <= 0.35) so a query's true top-k
        // should be dominated by its own cluster, not scattered randomly.
        assert!(
            hit_own_cluster >= cfg.k / 2,
            "got {hit_own_cluster}/{}",
            cfg.k
        );
    }
}
