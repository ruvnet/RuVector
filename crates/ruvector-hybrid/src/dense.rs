// Flat dense index with exact dot-product (inner product) search.
// For L2-normalised vectors, inner product equals cosine similarity.
// Production replacement: plug in ruvector-core HNSW via the HybridSearch trait.

use crate::{DenseVec, HybridDoc, HybridQuery, Scored};

/// Brute-force flat dense index: O(N·D) per query, exact.
/// Drop-in baseline for benchmarking against approximate methods.
pub struct DenseFlatIndex {
    docs: Vec<(u32, Vec<f32>)>,
    dims: usize,
}

impl DenseFlatIndex {
    pub fn new(dims: usize) -> Self {
        Self {
            docs: Vec::new(),
            dims,
        }
    }

    pub fn insert(&mut self, id: u32, v: &DenseVec) {
        assert_eq!(v.dims(), self.dims, "dimension mismatch");
        self.docs.push((id, v.data.clone()));
    }

    pub fn insert_doc(&mut self, doc: &HybridDoc) {
        self.insert(doc.id, &doc.dense);
    }

    /// Exact inner-product search, returns top-k.
    pub fn search(&self, q: &HybridQuery, k: usize) -> Vec<Scored> {
        self.search_dense(&q.dense, k)
    }

    pub fn search_dense(&self, q: &DenseVec, k: usize) -> Vec<Scored> {
        let mut scores: Vec<Scored> = self
            .docs
            .iter()
            .map(|(id, v)| Scored::new(*id, dot(q, v)))
            .collect();
        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores
    }

    pub fn num_docs(&self) -> usize {
        self.docs.len()
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Memory estimate: 4 bytes per f32 component.
    pub fn memory_bytes(&self) -> usize {
        self.docs.len() * self.dims * 4
    }
}

#[inline(always)]
fn dot(q: &DenseVec, v: &[f32]) -> f32 {
    q.data.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
}

/// L2-normalise a vector in place so inner product = cosine similarity.
pub fn l2_normalise(v: &mut DenseVec) {
    let norm: f32 = v.data.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in &mut v.data {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dense(data: Vec<f32>) -> DenseVec {
        DenseVec::new(data)
    }

    #[test]
    fn dot_product_correct() {
        let q = make_dense(vec![1.0, 0.0, 0.0]);
        let v = vec![0.9, 0.1, 0.1];
        assert!((dot(&q, &v) - 0.9).abs() < 1e-6);
    }

    #[test]
    fn flat_top1_correct() {
        let mut idx = DenseFlatIndex::new(3);
        idx.insert(0, &make_dense(vec![1.0, 0.0, 0.0]));
        idx.insert(1, &make_dense(vec![0.0, 1.0, 0.0]));
        idx.insert(2, &make_dense(vec![0.0, 0.0, 1.0]));

        let q = make_dense(vec![0.9, 0.1, 0.05]);
        let r = idx.search_dense(&q, 1);
        assert_eq!(r[0].id, 0, "closest to e_x should be doc 0");
    }

    #[test]
    fn l2_normalise_unit_length() {
        let mut v = make_dense(vec![3.0, 4.0]);
        l2_normalise(&mut v);
        let norm: f32 = v.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm={}", norm);
    }

    #[test]
    fn memory_estimate() {
        let mut idx = DenseFlatIndex::new(128);
        for i in 0..100u32 {
            idx.insert(i, &make_dense(vec![0.0f32; 128]));
        }
        // 100 docs * 128 dims * 4 bytes = 51200
        assert_eq!(idx.memory_bytes(), 51200);
    }
}
