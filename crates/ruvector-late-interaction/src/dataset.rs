/// Deterministic synthetic dataset generator for MaxSim benchmarks.
///
/// All data is reproducible with a fixed seed.  Embeddings are sampled from
/// a unit Gaussian and L2-normalised, matching the typical ColBERT setup.
use crate::{MultiVecDoc, MultiVecQuery};
use rand::distributions::Standard;
use rand::{Rng, SeedableRng};

pub struct DatasetGen {
    seed: u64,
    pub dim: usize,
}

impl DatasetGen {
    pub fn new(seed: u64, dim: usize) -> Self {
        Self { seed, dim }
    }

    /// Generate `n` documents each with `tokens_per_doc` L2-normalised embeddings.
    pub fn random_docs(&self, n: usize, tokens_per_doc: usize) -> Vec<MultiVecDoc> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        (0..n as u64)
            .map(|id| {
                let tokens = (0..tokens_per_doc)
                    .map(|_| {
                        let mut v: Vec<f32> =
                            (&mut rng).sample_iter(Standard).take(self.dim).collect();
                        normalize_vec(&mut v);
                        v
                    })
                    .collect();
                MultiVecDoc::new(id, tokens)
            })
            .collect()
    }

    /// Generate `n` queries each with `tokens_per_query` L2-normalised embeddings.
    pub fn random_queries(&self, n: usize, tokens_per_query: usize) -> Vec<MultiVecQuery> {
        // Offset seed by large prime so queries differ from docs.
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed.wrapping_add(999_983));
        (0..n)
            .map(|_| {
                let tokens = (0..tokens_per_query)
                    .map(|_| {
                        let mut v: Vec<f32> =
                            (&mut rng).sample_iter(Standard).take(self.dim).collect();
                        normalize_vec(&mut v);
                        v
                    })
                    .collect();
                MultiVecQuery::new(tokens)
            })
            .collect()
    }
}

fn normalize_vec(v: &mut Vec<f32>) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docs_are_normalised() {
        let gen = DatasetGen::new(1, 32);
        let docs = gen.random_docs(10, 4);
        for doc in &docs {
            for tok in &doc.tokens {
                let norm: f32 = tok.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    (norm - 1.0).abs() < 1e-5,
                    "token not normalised: norm={norm}"
                );
            }
        }
    }

    #[test]
    fn queries_differ_from_docs() {
        let gen = DatasetGen::new(1, 8);
        let docs = gen.random_docs(1, 1);
        let queries = gen.random_queries(1, 1);
        // With different seeds the vectors should differ
        let same = docs[0].tokens[0]
            .iter()
            .zip(queries[0].tokens[0].iter())
            .all(|(a, b)| (a - b).abs() < 1e-6);
        assert!(
            !same,
            "docs and queries should use different random streams"
        );
    }

    #[test]
    fn deterministic_regeneration() {
        let gen = DatasetGen::new(77, 16);
        let a = gen.random_docs(5, 3);
        let b = gen.random_docs(5, 3);
        for (da, db) in a.iter().zip(b.iter()) {
            for (ta, tb) in da.tokens.iter().zip(db.tokens.iter()) {
                for (x, y) in ta.iter().zip(tb.iter()) {
                    assert!((x - y).abs() < 1e-9, "generation is not deterministic");
                }
            }
        }
    }
}
