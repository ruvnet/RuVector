use crate::maxsim::maxsim_score;
/// Brute-force MaxSim index — exact O(N · T_d · T_q · D) scan.
///
/// This is the ground-truth baseline.  Every query scans every document.
use crate::{LiError, MaxSimIndex, MultiVecDoc, MultiVecQuery, Result, ScoredDoc};

pub struct BruteForceIndex {
    docs: Vec<MultiVecDoc>,
    dim: usize,
    built: bool,
}

impl BruteForceIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            docs: Vec::new(),
            dim,
            built: false,
        }
    }
}

impl MaxSimIndex for BruteForceIndex {
    fn name(&self) -> &'static str {
        "brute-force-maxsim"
    }

    fn len(&self) -> usize {
        self.docs.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn memory_bytes(&self) -> usize {
        self.docs
            .iter()
            .map(|d| d.tokens.len() * self.dim * 4)
            .sum()
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

        let mut scores: Vec<ScoredDoc> = self
            .docs
            .iter()
            .map(|doc| ScoredDoc {
                id: doc.id,
                score: maxsim_score(&q.tokens, &doc.tokens),
            })
            .collect();

        // Partial sort: only need top_k.
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
    use crate::dataset::DatasetGen;

    #[test]
    fn insert_and_query() {
        let gen = DatasetGen::new(1, 8);
        let docs = gen.random_docs(50, 4);
        let queries = gen.random_queries(3, 3);

        let mut idx = BruteForceIndex::new(8);
        for d in &docs {
            idx.insert(d.clone()).unwrap();
        }
        idx.build().unwrap();

        for q in &queries {
            let res = idx.query(q, 5).unwrap();
            assert_eq!(res.len(), 5);
            // Scores must be in descending order.
            for w in res.windows(2) {
                assert!(w[0].score >= w[1].score);
            }
        }
    }

    #[test]
    fn dim_mismatch_is_rejected() {
        let mut idx = BruteForceIndex::new(8);
        let bad_doc = MultiVecDoc::new(0, vec![vec![1.0; 16]]);
        assert!(matches!(
            idx.insert(bad_doc),
            Err(LiError::DimMismatch { .. })
        ));
    }

    #[test]
    fn top_k_capped_at_corpus_size() {
        let gen = DatasetGen::new(2, 8);
        let docs = gen.random_docs(5, 2);
        let mut idx = BruteForceIndex::new(8);
        for d in &docs {
            idx.insert(d.clone()).unwrap();
        }
        idx.build().unwrap();
        let q = gen.random_queries(1, 2);
        let res = idx.query(&q[0], 100).unwrap();
        assert_eq!(res.len(), 5); // only 5 docs exist
    }
}
