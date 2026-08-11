//! StaticPQ: codebook trained once on the initial build batch, never updated.
//!
//! New streaming vectors are encoded using the original codebook. When the
//! embedding distribution shifts (Phase B), the codebook no longer fits,
//! and recall degrades. This is the "no adaptation" baseline.

use crate::{AnnVariant, Hit};
use crate::pq::{Codebook, M, K};

pub struct StaticPq {
    codebook: Option<Codebook>,
    codes: Vec<Vec<u8>>,
    dims: usize,
}

impl StaticPq {
    pub fn new() -> Self {
        Self { codebook: None, codes: Vec::new(), dims: 0 }
    }
}

impl AnnVariant for StaticPq {
    fn build(&mut self, vectors: &[Vec<f32>]) {
        if vectors.is_empty() {
            return;
        }
        self.dims = vectors[0].len();
        let cb = Codebook::train(vectors, M, K, 1);
        self.codes = vectors.iter().map(|v| cb.encode(v)).collect();
        self.codebook = Some(cb);
    }

    fn insert(&mut self, vector: Vec<f32>) {
        if let Some(cb) = &self.codebook {
            self.codes.push(cb.encode(&vector));
        }
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let cb = match &self.codebook {
            Some(c) => c,
            None => return vec![],
        };
        let table = cb.adc_table(query);
        let mut hits: Vec<Hit> = self
            .codes
            .iter()
            .enumerate()
            .map(|(id, code)| Hit { id, dist: Codebook::adc_dist(&table, code) })
            .collect();
        hits.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        hits.truncate(k);
        hits
    }

    fn name(&self) -> &str { "StaticPQ" }
    fn len(&self) -> usize { self.codes.len() }
    fn memory_bytes(&self) -> usize {
        let code_bytes: usize = self.codes.iter().map(|c| c.len()).sum();
        let centroid_bytes = self.codebook.as_ref().map_or(0, |cb| {
            cb.centroids.iter().flat_map(|sub| sub.iter()).map(|c| c.len() * 4).sum()
        });
        code_bytes + centroid_bytes
    }
}
