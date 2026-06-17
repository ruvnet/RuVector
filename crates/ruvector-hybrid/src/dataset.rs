//! Deterministic dataset generation for hybrid search benchmarks.
//!
//! Two generators:
//! - [`generate`]: fully random tokens, independent of vectors. Used for
//!   latency and throughput measurements.  With random tokens the BM25 signal
//!   is noise relative to the dense ground truth, so HybridRRF recall < 100%
//!   even though DenseFlat is exact — this is the correct, expected behaviour
//!   when lexical and semantic signals are uncorrelated.
//! - [`generate_structured`]: topic-model corpus.  Both the dense vector and
//!   the sparse tokens for each document derive from the same set of latent
//!   topic vectors.  Documents sharing the same topics are both semantically
//!   similar (high cosine) and lexically similar (shared token IDs), so BM25
//!   and cosine rank the same documents highly.  RRF recall is therefore high.

use rand::{rngs::StdRng, Rng, SeedableRng};

/// A single indexed document.
pub struct Document {
    /// Caller-assigned document identifier.
    pub id: u32,
    /// Unit-normalised dense embedding.
    pub vector: Vec<f32>,
    /// Sparse token IDs (vocabulary-mapped, no raw text).
    pub tokens: Vec<u32>,
}

/// A single query for evaluation.
pub struct Query {
    /// Unit-normalised query embedding.
    pub vector: Vec<f32>,
    /// Sparse query token IDs.
    pub tokens: Vec<u32>,
    /// Exact top-K document IDs by cosine similarity (dense ground truth).
    pub ground_truth: Vec<u32>,
}

/// Parameters controlling dataset size and structure.
pub struct DatasetConfig {
    /// Number of documents to generate.
    pub n_docs: usize,
    /// Embedding dimensionality.
    pub dim: usize,
    /// Vocabulary size.  For [`generate_structured`] this is the number of
    /// latent topics (token IDs in `[0, vocab_size)`).
    pub vocab_size: u32,
    /// Average tokens per document (used only by [`generate`]).
    pub tokens_per_doc: usize,
    /// Number of queries to generate.
    pub n_queries: usize,
    /// RNG seed for deterministic generation.
    pub seed: u64,
    /// Top-k for ground-truth and search.
    pub k: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            n_docs: 10_000,
            dim: 128,
            vocab_size: 1_000,
            tokens_per_doc: 20,
            n_queries: 200,
            seed: 42,
            k: 10,
        }
    }
}

fn rand_unit(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

fn rand_tokens(rng: &mut StdRng, vocab: u32, avg_len: usize) -> Vec<u32> {
    let lo = avg_len / 2;
    let hi = (3 * avg_len / 2).max(lo + 1);
    let n = rng.gen_range(lo..=hi);
    (0..n).map(|_| rng.gen_range(0..vocab)).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Sample k distinct indices from [0, n) without replacement.
fn sample_k(rng: &mut StdRng, n: usize, k: usize) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.gen_range(0..n - i);
        pool.swap(i, j);
    }
    pool[..k].to_vec()
}

fn normalise(v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / norm).collect()
}

/// Generate a corpus with fully random (uncorrelated) tokens.
///
/// Ground truth = exact cosine top-k.  With random tokens the BM25 signal is
/// independent of cosine rank, so HybridRRF recall < DenseFlat recall.
/// Use this generator for latency / throughput benchmarks.
pub fn generate(cfg: &DatasetConfig) -> (Vec<Document>, Vec<Query>) {
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let docs: Vec<Document> = (0..cfg.n_docs as u32)
        .map(|id| Document {
            id,
            vector: rand_unit(&mut rng, cfg.dim),
            tokens: rand_tokens(&mut rng, cfg.vocab_size, cfg.tokens_per_doc),
        })
        .collect();

    let queries: Vec<Query> = (0..cfg.n_queries)
        .map(|_| {
            let qv = rand_unit(&mut rng, cfg.dim);
            let qt = rand_tokens(&mut rng, cfg.vocab_size, cfg.tokens_per_doc / 2);
            let mut sims: Vec<(f32, u32)> =
                docs.iter().map(|d| (dot(&qv, &d.vector), d.id)).collect();
            sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let ground_truth: Vec<u32> = sims.iter().take(cfg.k).map(|&(_, id)| id).collect();
            Query {
                vector: qv,
                tokens: qt,
                ground_truth,
            }
        })
        .collect();

    (docs, queries)
}

/// Generate a topic-model corpus where BM25 and cosine signals are correlated.
///
/// A set of `n_topics` random unit vectors serve as latent topic bases.
/// Each document samples `k_doc` topics; its embedding is the normalised sum of
/// those topic vectors and its tokens are the chosen topic IDs.  Each query
/// samples `k_query` topics similarly.
///
/// Two documents sharing the same subset of topics will have:
/// 1. High cosine similarity (their embeddings are sums of the same vectors).
/// 2. Identical or highly overlapping token sets (same topic IDs).
///
/// Therefore BM25 and cosine agree on which documents are relevant, and RRF
/// fusion achieves recall close to the DenseFlat oracle.
///
/// `cfg.vocab_size` controls `n_topics` (token IDs ∈ `[0, vocab_size)`).
pub fn generate_structured(cfg: &DatasetConfig) -> (Vec<Document>, Vec<Query>) {
    let n_topics = cfg.vocab_size as usize; // token IDs = topic IDs
    let k_doc: usize = 3; // topics sampled per document
    let k_query: usize = 2; // topics sampled per query

    assert!(n_topics >= k_doc.max(k_query), "vocab_size too small");
    assert!(cfg.k <= cfg.n_docs, "k must be <= n_docs");

    let mut rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(1));

    // Generate one unit vector per topic.
    let topic_vecs: Vec<Vec<f32>> = (0..n_topics)
        .map(|_| rand_unit(&mut rng, cfg.dim))
        .collect();

    // Generate documents: sum of k_doc topic vectors, then normalise.
    let docs: Vec<Document> = (0..cfg.n_docs as u32)
        .map(|id| {
            let topic_ids = sample_k(&mut rng, n_topics, k_doc);
            let raw: Vec<f32> = (0..cfg.dim)
                .map(|d| topic_ids.iter().map(|&t| topic_vecs[t][d]).sum::<f32>())
                .collect();
            let tokens: Vec<u32> = topic_ids.iter().map(|&t| t as u32).collect();
            Document {
                id,
                vector: normalise(raw),
                tokens,
            }
        })
        .collect();

    // Generate queries: sum of k_query topics, normalised.
    let queries: Vec<Query> = (0..cfg.n_queries)
        .map(|_| {
            let topic_ids = sample_k(&mut rng, n_topics, k_query);
            let raw: Vec<f32> = (0..cfg.dim)
                .map(|d| topic_ids.iter().map(|&t| topic_vecs[t][d]).sum::<f32>())
                .collect();
            let qv = normalise(raw);
            let qt: Vec<u32> = topic_ids.iter().map(|&t| t as u32).collect();

            let mut sims: Vec<(f32, u32)> =
                docs.iter().map(|d| (dot(&qv, &d.vector), d.id)).collect();
            sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let ground_truth: Vec<u32> = sims.iter().take(cfg.k).map(|&(_, id)| id).collect();

            Query {
                vector: qv,
                tokens: qt,
                ground_truth,
            }
        })
        .collect();

    (docs, queries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_counts() {
        let cfg = DatasetConfig {
            n_docs: 50,
            n_queries: 5,
            ..Default::default()
        };
        let (docs, queries) = generate(&cfg);
        assert_eq!(docs.len(), 50);
        assert_eq!(queries.len(), 5);
    }

    #[test]
    fn ground_truth_length() {
        let cfg = DatasetConfig {
            n_docs: 100,
            k: 10,
            ..Default::default()
        };
        let (_, queries) = generate(&cfg);
        for q in &queries {
            assert_eq!(q.ground_truth.len(), 10);
        }
    }

    #[test]
    fn vectors_are_unit() {
        let cfg = DatasetConfig {
            n_docs: 20,
            n_queries: 2,
            dim: 32,
            ..Default::default()
        };
        let (docs, _) = generate(&cfg);
        for d in &docs {
            let norm: f32 = d.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "vector not unit: norm={norm}");
        }
    }

    #[test]
    fn deterministic() {
        let cfg = DatasetConfig {
            n_docs: 30,
            n_queries: 3,
            ..Default::default()
        };
        let (d1, q1) = generate(&cfg);
        let (d2, q2) = generate(&cfg);
        assert_eq!(d1[0].vector, d2[0].vector);
        assert_eq!(q1[0].ground_truth, q2[0].ground_truth);
    }

    #[test]
    fn structured_docs_unit() {
        let cfg = DatasetConfig {
            n_docs: 50,
            n_queries: 5,
            dim: 16,
            vocab_size: 8,
            k: 5,
            ..Default::default()
        };
        let (docs, _) = generate_structured(&cfg);
        for d in &docs {
            let norm: f32 = d.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "structured doc not unit: norm={norm}"
            );
        }
    }

    #[test]
    fn structured_tokens_in_vocab() {
        let cfg = DatasetConfig {
            n_docs: 50,
            n_queries: 5,
            dim: 16,
            vocab_size: 8,
            k: 5,
            ..Default::default()
        };
        let (docs, _) = generate_structured(&cfg);
        for d in &docs {
            for &t in &d.tokens {
                assert!(t < 8, "token {t} out of vocab");
            }
        }
    }
}
