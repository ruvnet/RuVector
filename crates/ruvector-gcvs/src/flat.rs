/// Variant 1 — FlatSearch (baseline).
///
/// Brute-force cosine similarity scan over all stored vectors.
/// O(N·D) per query; exact recall = 100% by definition when no graph is needed.
use crate::{distance::cosine, GcvsError, GcvsIndex, Hit, Result};
use std::collections::HashMap;

pub struct FlatSearch {
    dim: usize,
    vectors: HashMap<usize, Vec<f32>>,
}

impl FlatSearch {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vectors: HashMap::new(),
        }
    }
}

impl GcvsIndex for FlatSearch {
    fn insert(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dim {
            return Err(GcvsError::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        self.vectors.insert(id, vector);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        if k == 0 {
            return Err(GcvsError::InvalidK);
        }
        if self.vectors.is_empty() {
            return Err(GcvsError::Empty);
        }
        let mut scored: Vec<Hit> = self
            .vectors
            .iter()
            .map(|(&id, v)| Hit {
                id,
                score: cosine(query, v),
            })
            .collect();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scored.truncate(k);
        Ok(scored)
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn name(&self) -> &'static str {
        "FlatSearch (baseline)"
    }
}
