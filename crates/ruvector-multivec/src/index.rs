//! Multi-vector search index variants.
//!
//! Three structs implement [`MultiVecIndex`]:
//!
//! | Struct              | Scoring       | Memory | Notes |
//! |---------------------|---------------|--------|-------|
//! | `CentroidIndex`     | centroid dot  | O(n×D) | Cheapest; loses token-level signal |
//! | `MaxSimIndex`       | exact MaxSim  | O(n×T×D) | Best recall; O(n×T×D) per query |
//! | `MuveraFdeIndex`    | FDE approx    | O(n×R×M×D) | Sub-linear approx via FDE encoding |

use crate::error::MultivecError;
use crate::scoring::{
    chamfer_score, dot, l2_normalize, maxsim_exact, FdeEncoder,
};

/// Search result: document id + similarity score (higher = better).
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    pub score: f32,
}

/// Common interface for multi-vector index variants.
pub trait MultiVecIndex {
    /// Add a document (list of token vectors) to the index.
    ///
    /// Vectors are L2-normalised on insertion. `id` is user-assigned.
    fn add(&mut self, id: usize, token_vecs: Vec<Vec<f32>>) -> Result<(), MultivecError>;

    /// Search for top-k documents most similar to the query token vectors.
    fn search(
        &self,
        query_tokens: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<SearchResult>, MultivecError>;

    /// Approximate heap memory used.
    fn memory_bytes(&self) -> usize;

    /// Human-readable variant name.
    fn name(&self) -> &'static str;

    /// Number of indexed documents.
    fn len(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Helper: top-k from a scored list (O(n log k) heap)
// ---------------------------------------------------------------------------

fn top_k(scores: Vec<(usize, f32)>, k: usize) -> Vec<SearchResult> {
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    // Min-heap by score so we keep the k largest.
    #[derive(PartialEq)]
    struct Scored(f32, usize);
    impl Eq for Scored {}
    impl PartialOrd for Scored {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Scored {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse so BinaryHeap (max-heap) acts as min-heap for scores.
            other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
        }
    }

    let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(k + 1);
    for (id, score) in scores {
        heap.push(Scored(score, id));
        if heap.len() > k {
            heap.pop();
        }
    }

    let mut results: Vec<SearchResult> = heap
        .into_iter()
        .map(|Scored(score, id)| SearchResult { id, score })
        .collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ---------------------------------------------------------------------------
// Variant 1: CentroidIndex
// ---------------------------------------------------------------------------

/// Baseline: pool each document's token vectors to their centroid, store as
/// a single f32 vector. Query also pooled. Score via dot product.
///
/// Pros: O(n×D) memory, O(n×D) per query.
/// Cons: loses all token-level signal — recall degrades on multi-topic docs.
pub struct CentroidIndex {
    dim: usize,
    ids: Vec<usize>,
    centroids: Vec<Vec<f32>>,
}

impl CentroidIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, ids: Vec::new(), centroids: Vec::new() }
    }

    fn pool(tokens: &[Vec<f32>]) -> Vec<f32> {
        let dim = tokens[0].len();
        let mut c = vec![0.0f32; dim];
        for t in tokens {
            c.iter_mut().zip(t.iter()).for_each(|(a, &b)| *a += b);
        }
        let scale = 1.0 / tokens.len() as f32;
        c.iter_mut().for_each(|x| *x *= scale);
        l2_normalize(&mut c);
        c
    }
}

impl MultiVecIndex for CentroidIndex {
    fn add(&mut self, id: usize, mut token_vecs: Vec<Vec<f32>>) -> Result<(), MultivecError> {
        if token_vecs.is_empty() {
            return Err(MultivecError::EmptyDocument { id });
        }
        for tv in &mut token_vecs {
            if tv.len() != self.dim {
                return Err(MultivecError::DimMismatch {
                    expected: self.dim,
                    actual: tv.len(),
                });
            }
            l2_normalize(tv);
        }
        self.ids.push(id);
        self.centroids.push(Self::pool(&token_vecs));
        Ok(())
    }

    fn search(
        &self,
        query_tokens: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<SearchResult>, MultivecError> {
        if self.ids.is_empty() {
            return Err(MultivecError::EmptyCorpus);
        }
        if k > self.ids.len() {
            return Err(MultivecError::KTooLarge { k, n: self.ids.len() });
        }
        let mut qt_norm: Vec<Vec<f32>> = query_tokens.to_vec();
        let qc = {
            for qt in &mut qt_norm {
                l2_normalize(qt);
            }
            Self::pool(&qt_norm)
        };
        let scores: Vec<(usize, f32)> = self
            .ids
            .iter()
            .zip(self.centroids.iter())
            .map(|(&id, dc)| (id, dot(&qc, dc)))
            .collect();
        Ok(top_k(scores, k))
    }

    fn memory_bytes(&self) -> usize {
        self.centroids.len() * self.dim * 4
    }

    fn name(&self) -> &'static str {
        "CentroidIndex (centroid dot)"
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

// ---------------------------------------------------------------------------
// Variant 2: MaxSimIndex (exact ColBERT-style)
// ---------------------------------------------------------------------------

/// Exact ColBERT MaxSim: store all token vectors per document; score =
/// Σ_i max_j dot(q_i, d_j). Optionally reports Chamfer score instead.
///
/// Pros: highest recall — captures multi-topic documents.
/// Cons: O(n×T_d×T_q×D) per query; memory O(n×T_d×D).
pub struct MaxSimIndex {
    dim: usize,
    ids: Vec<usize>,
    /// Stored token vectors per document, L2-normalised.
    doc_tokens: Vec<Vec<Vec<f32>>>,
    pub use_chamfer: bool,
}

impl MaxSimIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, ids: Vec::new(), doc_tokens: Vec::new(), use_chamfer: false }
    }

    pub fn with_chamfer(mut self) -> Self {
        self.use_chamfer = true;
        self
    }
}

impl MultiVecIndex for MaxSimIndex {
    fn add(&mut self, id: usize, mut token_vecs: Vec<Vec<f32>>) -> Result<(), MultivecError> {
        if token_vecs.is_empty() {
            return Err(MultivecError::EmptyDocument { id });
        }
        for tv in &mut token_vecs {
            if tv.len() != self.dim {
                return Err(MultivecError::DimMismatch {
                    expected: self.dim,
                    actual: tv.len(),
                });
            }
            l2_normalize(tv);
        }
        self.ids.push(id);
        self.doc_tokens.push(token_vecs);
        Ok(())
    }

    fn search(
        &self,
        query_tokens: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<SearchResult>, MultivecError> {
        if self.ids.is_empty() {
            return Err(MultivecError::EmptyCorpus);
        }
        if k > self.ids.len() {
            return Err(MultivecError::KTooLarge { k, n: self.ids.len() });
        }
        let mut qt_norm: Vec<Vec<f32>> = query_tokens.to_vec();
        for qt in &mut qt_norm {
            l2_normalize(qt);
        }

        let scores: Vec<(usize, f32)> = self
            .ids
            .iter()
            .zip(self.doc_tokens.iter())
            .map(|(&id, dt)| {
                let s = if self.use_chamfer {
                    chamfer_score(&qt_norm, dt)
                } else {
                    maxsim_exact(&qt_norm, dt)
                };
                (id, s)
            })
            .collect();
        Ok(top_k(scores, k))
    }

    fn memory_bytes(&self) -> usize {
        self.doc_tokens
            .iter()
            .map(|dt| dt.len() * self.dim * 4)
            .sum()
    }

    fn name(&self) -> &'static str {
        if self.use_chamfer {
            "MaxSimIndex (Chamfer)"
        } else {
            "MaxSimIndex (ColBERT MaxSim)"
        }
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

// ---------------------------------------------------------------------------
// Variant 3: MuveraFdeIndex (approximate via FDE)
// ---------------------------------------------------------------------------

/// Approximate MaxSim via MUVERA Fixed-Dimensional Encoding (FDE).
///
/// Each document is encoded into a single dense vector of length R×M×dim.
/// Query encoded the same way at search time. Score ≈ dot(fde_q, fde_d).
///
/// Pros: O(n × R×M×D) memory; O(n × R×M×D) per query (one dot per doc,
///       but larger vectors). Build is O(n×T×R×M×D).
/// Cons: Approximation introduces ~5-15% recall gap vs exact MaxSim.
///       FDE vector is larger than the original token vectors.
pub struct MuveraFdeIndex {
    dim: usize,
    encoder: FdeEncoder,
    ids: Vec<usize>,
    fde_vecs: Vec<Vec<f32>>,
}

impl MuveraFdeIndex {
    pub fn new(dim: usize, m: usize, r: usize, seed: u64) -> Result<Self, MultivecError> {
        if dim % m != 0 {
            return Err(MultivecError::FdeSubspaceMismatch { m, d: dim });
        }
        if r == 0 {
            return Err(MultivecError::InvalidRepetitions);
        }
        Ok(Self {
            dim,
            encoder: FdeEncoder::new(dim, m, r, seed),
            ids: Vec::new(),
            fde_vecs: Vec::new(),
        })
    }

    pub fn fde_dim(&self) -> usize {
        self.encoder.fde_dim()
    }
}

impl MultiVecIndex for MuveraFdeIndex {
    fn add(&mut self, id: usize, mut token_vecs: Vec<Vec<f32>>) -> Result<(), MultivecError> {
        if token_vecs.is_empty() {
            return Err(MultivecError::EmptyDocument { id });
        }
        for tv in &mut token_vecs {
            if tv.len() != self.dim {
                return Err(MultivecError::DimMismatch {
                    expected: self.dim,
                    actual: tv.len(),
                });
            }
            l2_normalize(tv);
        }
        let fde = self.encoder.encode(&token_vecs);
        self.ids.push(id);
        self.fde_vecs.push(fde);
        Ok(())
    }

    fn search(
        &self,
        query_tokens: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<SearchResult>, MultivecError> {
        if self.ids.is_empty() {
            return Err(MultivecError::EmptyCorpus);
        }
        if k > self.ids.len() {
            return Err(MultivecError::KTooLarge { k, n: self.ids.len() });
        }
        let mut qt_norm: Vec<Vec<f32>> = query_tokens.to_vec();
        for qt in &mut qt_norm {
            l2_normalize(qt);
        }
        let qfde = self.encoder.encode(&qt_norm);

        let scores: Vec<(usize, f32)> = self
            .ids
            .iter()
            .zip(self.fde_vecs.iter())
            .map(|(&id, dfde)| (id, dot(&qfde, dfde)))
            .collect();
        Ok(top_k(scores, k))
    }

    fn memory_bytes(&self) -> usize {
        self.fde_vecs.len() * self.encoder.fde_dim() * 4
    }

    fn name(&self) -> &'static str {
        "MuveraFdeIndex (FDE approx)"
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

// ---------------------------------------------------------------------------
// Variant 4: MuveraFdeRerankIndex (FDE retrieval + exact MaxSim reranking)
// ---------------------------------------------------------------------------

/// Full MUVERA two-stage pipeline:
///   Stage 1 — Scan FDE vectors → fetch top `rerank_factor × k` candidates.
///   Stage 2 — Exact MaxSim rerank of those candidates → return top k.
///
/// This is how MUVERA achieves near-oracle recall (~95%+) at 5-10× the QPS
/// of brute-force MaxSim. The FDE scan is O(n × R×K×D) but at reduced recall;
/// the reranking is O(C × T_q × T_d × D) where C = rerank_factor × k << n.
///
/// Memory: O(n × (R×K×D + T×D)) — stores both FDE and original token vecs.
pub struct MuveraFdeRerankIndex {
    dim: usize,
    encoder: FdeEncoder,
    ids: Vec<usize>,
    fde_vecs: Vec<Vec<f32>>,
    doc_tokens: Vec<Vec<Vec<f32>>>,
    pub rerank_factor: usize,
}

impl MuveraFdeRerankIndex {
    pub fn new(
        dim: usize,
        m: usize,
        r: usize,
        rerank_factor: usize,
        seed: u64,
    ) -> Result<Self, MultivecError> {
        if dim % m != 0 {
            return Err(MultivecError::FdeSubspaceMismatch { m, d: dim });
        }
        if r == 0 {
            return Err(MultivecError::InvalidRepetitions);
        }
        Ok(Self {
            dim,
            encoder: FdeEncoder::new(dim, m, r, seed),
            ids: Vec::new(),
            fde_vecs: Vec::new(),
            doc_tokens: Vec::new(),
            rerank_factor,
        })
    }
}

impl MultiVecIndex for MuveraFdeRerankIndex {
    fn add(&mut self, id: usize, mut token_vecs: Vec<Vec<f32>>) -> Result<(), MultivecError> {
        if token_vecs.is_empty() {
            return Err(MultivecError::EmptyDocument { id });
        }
        for tv in &mut token_vecs {
            if tv.len() != self.dim {
                return Err(MultivecError::DimMismatch {
                    expected: self.dim,
                    actual: tv.len(),
                });
            }
            l2_normalize(tv);
        }
        let fde = self.encoder.encode(&token_vecs);
        self.ids.push(id);
        self.fde_vecs.push(fde);
        self.doc_tokens.push(token_vecs);
        Ok(())
    }

    fn search(
        &self,
        query_tokens: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<SearchResult>, MultivecError> {
        if self.ids.is_empty() {
            return Err(MultivecError::EmptyCorpus);
        }
        if k > self.ids.len() {
            return Err(MultivecError::KTooLarge { k, n: self.ids.len() });
        }
        let mut qt_norm: Vec<Vec<f32>> = query_tokens.to_vec();
        for qt in &mut qt_norm {
            l2_normalize(qt);
        }
        let qfde = self.encoder.encode(&qt_norm);

        // Stage 1: FDE scan → top C candidates.
        let c = (self.rerank_factor * k).min(self.ids.len());
        let fde_scores: Vec<(usize, f32)> = self
            .ids
            .iter()
            .zip(self.fde_vecs.iter())
            .map(|(&id, dfde)| (id, dot(&qfde, dfde)))
            .collect();
        let candidates = top_k(fde_scores, c);

        // Stage 2: Exact MaxSim rerank of C candidates.
        let rerank_scores: Vec<(usize, f32)> = candidates
            .iter()
            .map(|cand| {
                let doc_idx = self.ids.iter().position(|&id| id == cand.id).unwrap();
                let ms = maxsim_exact(&qt_norm, &self.doc_tokens[doc_idx]);
                (cand.id, ms)
            })
            .collect();
        Ok(top_k(rerank_scores, k))
    }

    fn memory_bytes(&self) -> usize {
        let fde_bytes = self.fde_vecs.len() * self.encoder.fde_dim() * 4;
        let token_bytes: usize = self
            .doc_tokens
            .iter()
            .map(|dt| dt.len() * self.dim * 4)
            .sum();
        fde_bytes + token_bytes
    }

    fn name(&self) -> &'static str {
        "MuveraFdeRerank (FDE+MaxSim rerank)"
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_corpus(n: usize, t: usize, dim: usize, seed: u64) -> Vec<(usize, Vec<Vec<f32>>)> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f64, 1.0).unwrap();
        (0..n)
            .map(|id| {
                let tokens = (0..t)
                    .map(|_| (0..dim).map(|_| normal.sample(&mut rng) as f32).collect())
                    .collect();
                (id, tokens)
            })
            .collect()
    }

    fn top1_matches<I: MultiVecIndex>(idx: &I, query: &[Vec<f32>], expected_id: usize) -> bool {
        let res = idx.search(query, 1).unwrap();
        res[0].id == expected_id
    }

    #[test]
    fn centroid_index_self_retrieval() {
        let dim = 32;
        let corpus = make_corpus(10, 5, dim, 0);
        let mut idx = CentroidIndex::new(dim);
        for (id, tokens) in &corpus {
            idx.add(*id, tokens.clone()).unwrap();
        }
        // Each doc should retrieve itself.
        for (id, tokens) in &corpus {
            assert!(top1_matches(&idx, tokens, *id), "doc {id} not self-retrieved");
        }
    }

    #[test]
    fn maxsim_index_self_retrieval() {
        let dim = 32;
        let corpus = make_corpus(10, 5, dim, 1);
        let mut idx = MaxSimIndex::new(dim);
        for (id, tokens) in &corpus {
            idx.add(*id, tokens.clone()).unwrap();
        }
        for (id, tokens) in &corpus {
            assert!(top1_matches(&idx, tokens, *id), "doc {id} not self-retrieved");
        }
    }

    #[test]
    fn chamfer_index_self_retrieval() {
        let dim = 32;
        let corpus = make_corpus(10, 5, dim, 2);
        let mut idx = MaxSimIndex::new(dim).with_chamfer();
        for (id, tokens) in &corpus {
            idx.add(*id, tokens.clone()).unwrap();
        }
        for (id, tokens) in &corpus {
            assert!(top1_matches(&idx, tokens, *id), "doc {id} not self-retrieved");
        }
    }

    #[test]
    fn muvera_fde_self_retrieval() {
        let dim = 32;
        let corpus = make_corpus(10, 5, dim, 3);
        let mut idx = MuveraFdeIndex::new(dim, 4, 2, 42).unwrap();
        for (id, tokens) in &corpus {
            idx.add(*id, tokens.clone()).unwrap();
        }
        // FDE is approximate so we check top-3 contains self
        for (id, tokens) in &corpus {
            let res = idx.search(tokens, 3).unwrap();
            let found = res.iter().any(|r| r.id == *id);
            assert!(found, "doc {id} not in FDE top-3");
        }
    }

    #[test]
    fn memory_bytes_centroid_correct() {
        let dim = 64;
        let mut idx = CentroidIndex::new(dim);
        idx.add(0, vec![vec![1.0f32; dim]]).unwrap();
        idx.add(1, vec![vec![0.5f32; dim]]).unwrap();
        assert_eq!(idx.memory_bytes(), 2 * dim * 4);
    }

    #[test]
    fn error_on_empty_corpus() {
        let idx = MaxSimIndex::new(32);
        let result = idx.search(&[vec![0.0f32; 32]], 1);
        assert!(matches!(result, Err(MultivecError::EmptyCorpus)));
    }

    #[test]
    fn error_on_k_too_large() {
        let dim = 32;
        let corpus = make_corpus(3, 3, dim, 5);
        let mut idx = MaxSimIndex::new(dim);
        for (id, tokens) in &corpus {
            idx.add(*id, tokens.clone()).unwrap();
        }
        let q: Vec<Vec<f32>> = vec![vec![0.1f32; dim]];
        let result = idx.search(&q, 10);
        assert!(matches!(result, Err(MultivecError::KTooLarge { .. })));
    }
}
