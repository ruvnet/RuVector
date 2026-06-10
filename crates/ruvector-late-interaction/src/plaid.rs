/// PLAID-lite: centroid pre-filter + full MaxSim on shortlist.
///
/// Algorithm (adapted from Santhanam et al., PLAID, EMNLP 2022):
///   Build: cluster all doc token embeddings into K centroids (k-means).
///         For each centroid, store the set of doc IDs whose tokens are
///         assigned to it.
///   Query: for each query token, find the `n_probe` nearest centroids.
///          Union all candidate doc IDs.  Rerank with exact MaxSim.
///
/// Trade-off: speed vs recall.  Recall degrades gracefully as n_probe decreases.
use std::collections::HashSet;

use crate::maxsim::{dot, kmeans_centroids, l2sq, maxsim_score};
use crate::{LiError, MaxSimIndex, MultiVecDoc, MultiVecQuery, Result, ScoredDoc};

pub struct PlaidLiteIndex {
    docs: Vec<MultiVecDoc>,
    dim: usize,
    /// Number of k-means centroids.
    num_centroids: usize,
    /// Centroids (num_centroids × dim).
    centroids: Vec<Vec<f32>>,
    /// centroid_id → set of doc IDs whose tokens are assigned to it.
    centroid_to_docs: Vec<Vec<u64>>,
    /// Number of centroids to probe per query token.
    n_probe: usize,
    built: bool,
}

impl PlaidLiteIndex {
    /// Create a new PLAID-lite index.
    ///
    /// - `num_centroids`: number of k-means clusters (e.g. 32–256)
    /// - `n_probe`: centroids visited per query token (higher = better recall, slower)
    pub fn new(dim: usize, num_centroids: usize, n_probe: usize) -> Self {
        Self {
            docs: Vec::new(),
            dim,
            num_centroids,
            centroids: Vec::new(),
            centroid_to_docs: Vec::new(),
            n_probe: n_probe.max(1),
            built: false,
        }
    }

    /// Find the indices of the `n` nearest centroids to `query_token`.
    fn nearest_centroids(&self, query_token: &[f32], n: usize) -> Vec<usize> {
        let k = self.centroids.len();
        if k == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(f32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (dot(query_token, c), i))
            .collect();
        // Descending by dot product (centroids should be normalised after k-means).
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.iter().take(n).map(|(_, i)| *i).collect()
    }
}

impl MaxSimIndex for PlaidLiteIndex {
    fn name(&self) -> &'static str {
        "plaid-lite-maxsim"
    }

    fn len(&self) -> usize {
        self.docs.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn memory_bytes(&self) -> usize {
        let doc_mem: usize = self
            .docs
            .iter()
            .map(|d| d.tokens.len() * self.dim * 4)
            .sum();
        let centroid_mem = self.centroids.len() * self.dim * 4;
        doc_mem + centroid_mem
    }

    fn insert(&mut self, doc: MultiVecDoc) -> Result<()> {
        if let Some(tok) = doc.tokens.first() {
            if tok.len() != self.dim {
                return Err(LiError::DimMismatch {
                    expected: self.dim,
                    got: tok.len(),
                });
            }
        }
        self.docs.push(doc);
        self.built = false;
        Ok(())
    }

    fn build(&mut self) -> Result<()> {
        if self.docs.is_empty() {
            return Err(LiError::EmptyCorpus);
        }

        // Collect up to MAX_KMEANS_TOKENS token embeddings for k-means.
        // Subsampling keeps build time bounded even for large corpora.
        const MAX_KMEANS_TOKENS: usize = 8_000;
        let all_tokens: Vec<Vec<f32>> = {
            let raw: Vec<Vec<f32>> = self
                .docs
                .iter()
                .flat_map(|d| d.tokens.iter().cloned())
                .collect();
            if raw.len() <= MAX_KMEANS_TOKENS {
                raw
            } else {
                // Uniform stride sample: take every N-th token.
                let stride = raw.len() / MAX_KMEANS_TOKENS;
                raw.into_iter()
                    .step_by(stride.max(1))
                    .take(MAX_KMEANS_TOKENS)
                    .collect()
            }
        };

        let k = self.num_centroids.min(all_tokens.len());
        self.centroids = kmeans_centroids(&all_tokens, k, self.dim, 5, 42);

        // Build centroid → doc inverted index.
        self.centroid_to_docs = vec![Vec::new(); self.centroids.len()];

        for doc in &self.docs {
            // For each doc, find all centroids that any of its tokens are
            // assigned to, then add the doc ID once per centroid.
            let mut centroid_hits: HashSet<usize> = HashSet::new();
            for tok in &doc.tokens {
                let nearest = (0..self.centroids.len())
                    .min_by(|&a, &b| {
                        l2sq(&self.centroids[a], tok)
                            .partial_cmp(&l2sq(&self.centroids[b], tok))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);
                centroid_hits.insert(nearest);
            }
            for c in centroid_hits {
                self.centroid_to_docs[c].push(doc.id);
            }
        }

        self.built = true;
        Ok(())
    }

    fn query(&self, q: &MultiVecQuery, top_k: usize) -> Result<Vec<ScoredDoc>> {
        if !self.built {
            return Err(LiError::NotBuilt);
        }
        if self.docs.is_empty() {
            return Err(LiError::EmptyCorpus);
        }
        if top_k == 0 {
            return Err(LiError::InvalidK);
        }

        // Step 1: collect candidate doc IDs via centroid pre-filter.
        let mut candidate_ids: HashSet<u64> = HashSet::new();
        for qt in &q.tokens {
            let centroids = self.nearest_centroids(qt, self.n_probe);
            for c in centroids {
                for &doc_id in &self.centroid_to_docs[c] {
                    candidate_ids.insert(doc_id);
                }
            }
        }

        // Step 2: rerank candidates with exact MaxSim.
        // Build a lookup: doc_id → &MultiVecDoc.
        let mut scores: Vec<ScoredDoc> = self
            .docs
            .iter()
            .filter(|d| candidate_ids.contains(&d.id))
            .map(|doc| ScoredDoc {
                id: doc.id,
                score: maxsim_score(&q.tokens, &doc.tokens),
            })
            .collect();

        scores.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(top_k);
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute::BruteForceIndex;
    use crate::dataset::DatasetGen;
    use crate::recall_at_k;

    #[test]
    fn plaid_builds_and_queries() {
        let gen = DatasetGen::new(9, 16);
        let docs = gen.random_docs(100, 4);
        let queries = gen.random_queries(5, 3);

        let mut idx = PlaidLiteIndex::new(16, 16, 4);
        for d in &docs {
            idx.insert(d.clone()).unwrap();
        }
        idx.build().unwrap();

        for q in &queries {
            let res = idx.query(q, 5).unwrap();
            assert!(!res.is_empty());
        }
    }

    #[test]
    fn plaid_recall_probe4_above_threshold() {
        let gen = DatasetGen::new(19, 32);
        let docs = gen.random_docs(500, 8);
        let queries = gen.random_queries(20, 4);

        let mut bf = BruteForceIndex::new(32);
        let mut plaid = PlaidLiteIndex::new(32, 32, 4);
        for d in &docs {
            bf.insert(d.clone()).unwrap();
            plaid.insert(d.clone()).unwrap();
        }
        bf.build().unwrap();
        plaid.build().unwrap();

        let total: f32 = queries
            .iter()
            .map(|q| {
                let gt = bf.query(q, 10).unwrap();
                let res = plaid.query(q, 10).unwrap();
                recall_at_k(&res, &gt, 10)
            })
            .sum();
        let avg = total / queries.len() as f32;
        assert!(
            avg >= 0.60,
            "PLAID probe=4 recall@10 = {avg:.3}, want ≥ 0.60"
        );
    }

    #[test]
    fn plaid_probe_increase_improves_recall() {
        let gen = DatasetGen::new(23, 32);
        let docs = gen.random_docs(300, 8);
        let queries = gen.random_queries(10, 4);

        let mut bf = BruteForceIndex::new(32);
        let mut p2 = PlaidLiteIndex::new(32, 32, 2);
        let mut p8 = PlaidLiteIndex::new(32, 32, 8);
        for d in &docs {
            bf.insert(d.clone()).unwrap();
            p2.insert(d.clone()).unwrap();
            p8.insert(d.clone()).unwrap();
        }
        bf.build().unwrap();
        p2.build().unwrap();
        p8.build().unwrap();

        let recall2: f32 = queries
            .iter()
            .map(|q| {
                let gt = bf.query(q, 10).unwrap();
                let res = p2.query(q, 10).unwrap();
                recall_at_k(&res, &gt, 10)
            })
            .sum::<f32>()
            / queries.len() as f32;

        let recall8: f32 = queries
            .iter()
            .map(|q| {
                let gt = bf.query(q, 10).unwrap();
                let res = p8.query(q, 10).unwrap();
                recall_at_k(&res, &gt, 10)
            })
            .sum::<f32>()
            / queries.len() as f32;

        assert!(
            recall8 >= recall2,
            "Higher n_probe should not decrease recall: probe=2: {recall2:.3}, probe=8: {recall8:.3}"
        );
    }
}
