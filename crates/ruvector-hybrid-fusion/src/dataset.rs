//! Deterministic corpus generator for hybrid retrieval benchmarks.
//!
//! Produces a corpus where 50 % of documents are *text-dominant* (rich in
//! topic keywords, moderate vector alignment) and 50 % are *vector-dominant*
//! (sparse keywords, tight vector alignment).  Queries are mixed across three
//! types (Hybrid / KeywordHeavy / VectorHeavy) so that a coherence-adaptive
//! fuser that correctly senses each leg's signal strength outperforms RRF.
//!
//! All randomness is seeded so every run produces identical numbers.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// How textually / vectorially rich a document is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocType {
    TextDominant,
    VectorDominant,
}

/// What mix of keyword vs. vector signal a query carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    Hybrid,
    KeywordHeavy,
    VectorHeavy,
}

#[derive(Debug, Clone)]
pub struct Document {
    pub id: usize,
    pub tokens: Vec<String>,
    pub vector: Vec<f32>,
    pub topic: usize,
    pub doc_type: DocType,
}

#[derive(Debug, Clone)]
pub struct Query {
    pub tokens: Vec<String>,
    pub vector: Vec<f32>,
    pub topic: usize,
    pub query_type: QueryType,
}

/// The assembled benchmark corpus.
pub struct Corpus {
    pub docs: Vec<Document>,
    pub queries: Vec<Query>,
    /// ground_truth[i] = ordered doc-ids for query i, by combined oracle score.
    pub ground_truth: Vec<Vec<usize>>,
    pub num_topics: usize,
    pub dim: usize,
}

// ── public entry point ────────────────────────────────────────────────────────

pub const NUM_TOPICS: usize = 10;
pub const DOCS_PER_TOPIC: usize = 300; // 150 TextDominant + 150 VectorDominant
pub const DIM: usize = 128;
pub const QUERIES_PER_TOPIC: usize = 20; // 10 Hybrid + 5 KeywordHeavy + 5 VectorHeavy
pub const CORE_TERMS_PER_TOPIC: usize = 20;
pub const NOISE_TERMS: usize = 30;

/// Generate a reproducible corpus from `seed`.
pub fn generate_corpus(seed: u64) -> Corpus {
    let mut rng = StdRng::seed_from_u64(seed);

    // Build vocabulary: core terms per topic + shared noise terms
    let vocab = build_vocab();

    // Generate random unit centroids for each topic
    let centroids: Vec<Vec<f32>> = (0..NUM_TOPICS)
        .map(|_| unit_rand_vec(&mut rng, DIM))
        .collect();

    // Generate documents
    let mut docs: Vec<Document> = Vec::with_capacity(NUM_TOPICS * DOCS_PER_TOPIC);
    for topic in 0..NUM_TOPICS {
        let core_start = topic * CORE_TERMS_PER_TOPIC;
        let core_end = core_start + CORE_TERMS_PER_TOPIC;
        let core: Vec<&str> = vocab[core_start..core_end].iter().map(|s| s.as_str()).collect();
        let noise: Vec<&str> = vocab[NUM_TOPICS * CORE_TERMS_PER_TOPIC..].iter().map(|s| s.as_str()).collect();

        for local_i in 0..DOCS_PER_TOPIC {
            let id = topic * DOCS_PER_TOPIC + local_i;
            let doc_type = if local_i < DOCS_PER_TOPIC / 2 {
                DocType::TextDominant
            } else {
                DocType::VectorDominant
            };

            let tokens = match doc_type {
                DocType::TextDominant => {
                    // 12-15 core terms + 3-5 noise terms
                    let n_core = rng.gen_range(12..=15usize);
                    let n_noise = rng.gen_range(3..=5usize);
                    sample_terms(&mut rng, &core, n_core)
                        .into_iter()
                        .chain(sample_terms(&mut rng, &noise, n_noise))
                        .collect()
                }
                DocType::VectorDominant => {
                    // 2-4 core terms + 10-14 noise terms
                    let n_core = rng.gen_range(2..=4usize);
                    let n_noise = rng.gen_range(10..=14usize);
                    sample_terms(&mut rng, &core, n_core)
                        .into_iter()
                        .chain(sample_terms(&mut rng, &noise, n_noise))
                        .collect()
                }
            };

            let vector = match doc_type {
                DocType::TextDominant => {
                    // Centroid + moderate noise (σ=0.35)
                    perturb_unit(&centroids[topic], 0.35, &mut rng)
                }
                DocType::VectorDominant => {
                    // Centroid + tight noise (σ=0.10)
                    perturb_unit(&centroids[topic], 0.10, &mut rng)
                }
            };

            docs.push(Document { id, tokens, vector, topic, doc_type });
        }
    }

    // Generate queries
    let mut queries: Vec<Query> = Vec::with_capacity(NUM_TOPICS * QUERIES_PER_TOPIC);
    for topic in 0..NUM_TOPICS {
        let core_start = topic * CORE_TERMS_PER_TOPIC;
        let core_end = core_start + CORE_TERMS_PER_TOPIC;
        let core: Vec<&str> = vocab[core_start..core_end].iter().map(|s| s.as_str()).collect();
        for qi in 0..QUERIES_PER_TOPIC {
            let query_type = if qi < 10 {
                QueryType::Hybrid
            } else if qi < 15 {
                QueryType::KeywordHeavy
            } else {
                QueryType::VectorHeavy
            };

            let (tokens, vector) = match query_type {
                QueryType::Hybrid => {
                    // 7-10 core terms + tight vector
                    let n = rng.gen_range(7..=10usize);
                    let toks = sample_terms(&mut rng, &core, n);
                    let vec = perturb_unit(&centroids[topic], 0.12, &mut rng);
                    (toks, vec)
                }
                QueryType::KeywordHeavy => {
                    // 12-16 core terms + diffuse vector
                    let n = rng.gen_range(12..=16usize);
                    let toks = sample_terms(&mut rng, &core, n);
                    let vec = perturb_unit(&centroids[topic], 0.60, &mut rng);
                    (toks, vec)
                }
                QueryType::VectorHeavy => {
                    // 2-4 core terms + very tight vector
                    let n = rng.gen_range(2..=4usize);
                    let toks = sample_terms(&mut rng, &core, n);
                    let vec = perturb_unit(&centroids[topic], 0.05, &mut rng);
                    (toks, vec)
                }
            };

            queries.push(Query { tokens, vector, topic, query_type });
        }
    }

    // Compute ground truth by combined oracle: normalised BM25 + cosine
    let ground_truth = compute_ground_truth(&docs, &queries);

    Corpus {
        docs,
        queries,
        ground_truth,
        num_topics: NUM_TOPICS,
        dim: DIM,
    }
}

// ── private helpers ───────────────────────────────────────────────────────────

fn build_vocab() -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for t in 0..NUM_TOPICS {
        for i in 0..CORE_TERMS_PER_TOPIC {
            v.push(format!("t{}_w{}", t, i));
        }
    }
    for i in 0..NOISE_TERMS {
        v.push(format!("noise_{}", i));
    }
    v
}

fn sample_terms<R: Rng>(rng: &mut R, pool: &[&str], n: usize) -> Vec<String> {
    let n = n.min(pool.len());
    let mut indices: Vec<usize> = (0..pool.len()).collect();
    // Fisher-Yates partial shuffle to pick n unique items
    for i in 0..n {
        let j = rng.gen_range(i..pool.len());
        indices.swap(i, j);
    }
    indices[..n].iter().map(|&i| pool[i].to_string()).collect()
}

fn unit_rand_vec<R: Rng>(rng: &mut R, dim: usize) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
    unit_normalise(v)
}

fn perturb_unit<R: Rng>(base: &[f32], sigma: f32, rng: &mut R) -> Vec<f32> {
    let v: Vec<f32> = base
        .iter()
        .map(|&x| x + sigma * (rng.gen::<f32>() * 2.0 - 1.0))
        .collect();
    unit_normalise(v)
}

fn unit_normalise(v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 { return v; }
    v.into_iter().map(|x| x / norm).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn compute_ground_truth(docs: &[Document], queries: &[Query]) -> Vec<Vec<usize>> {
    // Oracle: top-5 by keyword (BM25 approx) ∪ top-5 by cosine.
    //
    // This bimodal oracle models the hybrid relevance criterion explicitly:
    // the best hybrid retriever is one that finds both the strongest keyword
    // matches AND the strongest semantic matches.  Single-leg retrievers can
    // only capture one half, giving them ~50% recall against this oracle.
    //
    // BM25 here is a IDF-weighted term-overlap sum (no TF / length normalisation)
    // which is sufficient to rank the text-dominant docs above vector-dominant
    // docs within the same topic.

    let n = docs.len() as f32;
    let mut df: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for doc in docs {
        let terms: std::collections::HashSet<&String> = doc.tokens.iter().collect();
        for t in terms {
            *df.entry(t.clone()).or_insert(0.0) += 1.0;
        }
    }

    queries
        .iter()
        .map(|q| {
            // ── BM25 leg ──────────────────────────────────────────────────────
            let mut bm25_ranked: Vec<(usize, f32)> = docs
                .iter()
                .map(|doc| {
                    let doc_terms: std::collections::HashSet<&String> =
                        doc.tokens.iter().collect();
                    let score: f32 = q
                        .tokens
                        .iter()
                        .filter(|t| doc_terms.contains(t))
                        .map(|t| {
                            let d = df.get(t).copied().unwrap_or(0.0);
                            ((n - d + 0.5) / (d + 0.5) + 1.0).ln().max(0.0)
                        })
                        .sum();
                    (doc.id, score)
                })
                .collect();
            bm25_ranked
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // ── Cosine leg ────────────────────────────────────────────────────
            let mut cos_ranked: Vec<(usize, f32)> = docs
                .iter()
                .map(|doc| (doc.id, dot(&q.vector, &doc.vector)))
                .collect();
            cos_ranked
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // ── Union of top-5 from each leg ──────────────────────────────────
            let mut result: Vec<usize> = Vec::with_capacity(10);
            let mut seen: std::collections::HashSet<usize> =
                std::collections::HashSet::new();

            for &(id, _) in bm25_ranked.iter().take(5) {
                if seen.insert(id) {
                    result.push(id);
                }
            }
            for &(id, _) in cos_ranked.iter().take(5) {
                if seen.insert(id) {
                    result.push(id);
                }
            }

            // Fill to exactly 10 if early legs had overlap
            let mut b_iter = bm25_ranked.iter().skip(5);
            let mut c_iter = cos_ranked.iter().skip(5);
            while result.len() < 10 {
                let candidate = b_iter.next().or_else(|| c_iter.next());
                match candidate {
                    Some(&(id, _)) => {
                        if seen.insert(id) {
                            result.push(id);
                        }
                    }
                    None => break,
                }
            }
            result.truncate(10);
            result
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_sizes() {
        let c = generate_corpus(1);
        assert_eq!(c.docs.len(), NUM_TOPICS * DOCS_PER_TOPIC);
        assert_eq!(c.queries.len(), NUM_TOPICS * QUERIES_PER_TOPIC);
        assert_eq!(c.ground_truth.len(), c.queries.len());
        for gt in &c.ground_truth {
            assert_eq!(gt.len(), 10);
        }
    }

    #[test]
    fn corpus_deterministic() {
        let c1 = generate_corpus(42);
        let c2 = generate_corpus(42);
        assert_eq!(c1.docs[0].tokens, c2.docs[0].tokens);
        assert_eq!(c1.queries[0].tokens, c2.queries[0].tokens);
        assert_eq!(c1.ground_truth[0], c2.ground_truth[0]);
    }

    #[test]
    fn corpus_different_seeds() {
        let c1 = generate_corpus(1);
        let c2 = generate_corpus(2);
        // Very unlikely to be identical with different seeds
        assert_ne!(c1.docs[0].tokens, c2.docs[0].tokens);
    }

    #[test]
    fn vectors_are_unit_length() {
        let c = generate_corpus(7);
        for doc in &c.docs {
            let norm: f32 = doc.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "doc {} vector norm {}", doc.id, norm);
        }
    }

    #[test]
    fn query_types_distributed() {
        let c = generate_corpus(3);
        let n_hybrid = c.queries.iter().filter(|q| q.query_type == QueryType::Hybrid).count();
        let n_kw = c.queries.iter().filter(|q| q.query_type == QueryType::KeywordHeavy).count();
        let n_vec = c.queries.iter().filter(|q| q.query_type == QueryType::VectorHeavy).count();
        // 10 Hybrid + 5 KW + 5 Vec per topic × 10 topics
        assert_eq!(n_hybrid, 100);
        assert_eq!(n_kw, 50);
        assert_eq!(n_vec, 50);
    }
}
