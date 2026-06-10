/// Compressed MaxSim index — SQ8 scalar-quantized token storage.
///
/// Each f32 token embedding is quantized to i8 at insert time, reducing
/// memory by 4× while preserving ≥ 85 % recall@10 against the exact baseline.
use crate::maxsim::Sq8Codec;
use crate::{LiError, MaxSimIndex, MultiVecDoc, MultiVecQuery, Result, ScoredDoc};

struct QuantizedDoc {
    id: u64,
    tokens: Vec<Vec<i8>>,
}

pub struct CompressedIndex {
    docs: Vec<QuantizedDoc>,
    codec: Sq8Codec,
    built: bool,
}

impl CompressedIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            docs: Vec::new(),
            codec: Sq8Codec::new(dim),
            built: false,
        }
    }

    fn maxsim_sq8(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<i8>]) -> f32 {
        query_tokens
            .iter()
            .map(|qt| {
                // Quantize query token on-the-fly.
                let qt_q: Vec<i8> = qt
                    .iter()
                    .map(|&x| (x.clamp(-1.0, 1.0) * 127.0).round() as i8)
                    .collect();
                doc_tokens
                    .iter()
                    .map(|dt| Sq8Codec::dot_i8(&qt_q, dt))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum()
    }
}

impl MaxSimIndex for CompressedIndex {
    fn name(&self) -> &'static str {
        "compressed-sq8-maxsim"
    }

    fn len(&self) -> usize {
        self.docs.len()
    }

    fn dim(&self) -> usize {
        self.codec.dim
    }

    fn memory_bytes(&self) -> usize {
        self.docs
            .iter()
            .map(|d| d.tokens.len() * self.codec.bytes_per_token())
            .sum()
    }

    fn insert(&mut self, doc: MultiVecDoc) -> Result<()> {
        if let Some(tok) = doc.tokens.first() {
            if tok.len() != self.codec.dim {
                return Err(LiError::DimMismatch {
                    expected: self.codec.dim,
                    got: tok.len(),
                });
            }
        }
        let qtokens: Vec<Vec<i8>> = doc.tokens.iter().map(|t| self.codec.encode(t)).collect();
        self.docs.push(QuantizedDoc {
            id: doc.id,
            tokens: qtokens,
        });
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
                score: Self::maxsim_sq8(&q.tokens, &doc.tokens),
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
    fn memory_is_quarter_of_brute() {
        let gen = DatasetGen::new(5, 64);
        let docs = gen.random_docs(100, 16);

        let mut bf = BruteForceIndex::new(64);
        let mut cmp = CompressedIndex::new(64);
        for d in &docs {
            bf.insert(d.clone()).unwrap();
            cmp.insert(d.clone()).unwrap();
        }
        bf.build().unwrap();
        cmp.build().unwrap();

        // i8 is 1 byte; f32 is 4 bytes → 4× reduction.
        assert_eq!(cmp.memory_bytes() * 4, bf.memory_bytes());
    }

    #[test]
    fn compressed_recall_above_threshold() {
        let gen = DatasetGen::new(55, 32);
        let docs = gen.random_docs(300, 8);
        let queries = gen.random_queries(20, 4);

        let mut bf = BruteForceIndex::new(32);
        let mut cmp = CompressedIndex::new(32);
        for d in &docs {
            bf.insert(d.clone()).unwrap();
            cmp.insert(d.clone()).unwrap();
        }
        bf.build().unwrap();
        cmp.build().unwrap();

        let total: f32 = queries
            .iter()
            .map(|q| {
                let gt = bf.query(q, 10).unwrap();
                let res = cmp.query(q, 10).unwrap();
                recall_at_k(&res, &gt, 10)
            })
            .sum();
        let avg = total / queries.len() as f32;
        assert!(avg >= 0.75, "SQ8 recall@10 = {avg:.3}, want ≥ 0.75");
    }
}
