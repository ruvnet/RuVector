// ruvector-hybrid: hybrid sparse-dense vector search
// Combines BM25-style sparse inverted index with dense flat/HNSW search,
// fused via Reciprocal Rank Fusion (RRF) or linear score interpolation.

pub mod dense;
pub mod fusion;
pub mod index;
pub mod sparse;

/// A sparse vector: sorted (term_id, weight) pairs.
/// Represents a document or query in term-weight space (SPLADE / BM25 style).
#[derive(Debug, Clone, Default)]
pub struct SparseVec {
    /// Sorted by term_id ascending for merge-join efficiency.
    pub terms: Vec<(u32, f32)>,
}

impl SparseVec {
    pub fn new(mut terms: Vec<(u32, f32)>) -> Self {
        terms.sort_by_key(|&(t, _)| t);
        terms.dedup_by_key(|&mut (t, _)| t);
        Self { terms }
    }

    pub fn nnz(&self) -> usize {
        self.terms.len()
    }

    /// Dot product with another sparse vector (merge-join, O(nnz_a + nnz_b)).
    pub fn dot(&self, other: &SparseVec) -> f32 {
        let (mut i, mut j) = (0, 0);
        let mut acc = 0.0f32;
        while i < self.terms.len() && j < other.terms.len() {
            let (ta, wa) = self.terms[i];
            let (tb, wb) = other.terms[j];
            match ta.cmp(&tb) {
                std::cmp::Ordering::Equal => {
                    acc += wa * wb;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    j += 1;
                }
            }
        }
        acc
    }
}

/// A dense vector: f32 components, L2-normalised by convention.
#[derive(Debug, Clone)]
pub struct DenseVec {
    pub data: Vec<f32>,
}

impl DenseVec {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn dot(&self, other: &DenseVec) -> f32 {
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }

    pub fn dims(&self) -> usize {
        self.data.len()
    }
}

/// A document combining dense and sparse representations.
#[derive(Debug, Clone)]
pub struct HybridDoc {
    pub id: u32,
    pub dense: DenseVec,
    pub sparse: SparseVec,
}

/// A query combining dense and sparse representations.
#[derive(Debug, Clone)]
pub struct HybridQuery {
    pub dense: DenseVec,
    pub sparse: SparseVec,
}

/// A scored retrieval result.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    pub id: u32,
    pub score: f32,
}

impl Scored {
    pub fn new(id: u32, score: f32) -> Self {
        Self { id, score }
    }
}

/// Core trait for any hybrid search backend.
pub trait HybridSearch {
    fn insert(&mut self, doc: HybridDoc);
    fn search_dense(&self, q: &HybridQuery, k: usize) -> Vec<Scored>;
    fn search_sparse(&self, q: &HybridQuery, k: usize) -> Vec<Scored>;

    /// Hybrid search with RRF fusion.
    fn search_rrf(&self, q: &HybridQuery, k: usize, candidate_k: usize) -> Vec<Scored> {
        let d = self.search_dense(q, candidate_k);
        let s = self.search_sparse(q, candidate_k);
        fusion::rrf(&d, &s, k)
    }

    /// Hybrid search with linear score interpolation.
    fn search_linear(
        &self,
        q: &HybridQuery,
        k: usize,
        candidate_k: usize,
        alpha: f32,
    ) -> Vec<Scored> {
        let d = self.search_dense(q, candidate_k);
        let s = self.search_sparse(q, candidate_k);
        fusion::linear(&d, &s, k, alpha)
    }
}

/// Compute recall@k: fraction of oracle_ids found in retrieved ids.
pub fn recall_at_k(retrieved: &[Scored], oracle: &[Scored]) -> f64 {
    let oracle_set: std::collections::HashSet<u32> = oracle.iter().map(|s| s.id).collect();
    let hits = retrieved
        .iter()
        .filter(|s| oracle_set.contains(&s.id))
        .count();
    if oracle.is_empty() {
        return 1.0;
    }
    hits as f64 / oracle.len() as f64
}
