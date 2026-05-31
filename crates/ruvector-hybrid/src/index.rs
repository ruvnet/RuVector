// Composite HybridIndex: wraps DenseFlatIndex + SparseIndex and implements
// the HybridSearch trait so callers get all four search modes transparently.

use crate::{
    dense::DenseFlatIndex, sparse::SparseIndex, HybridDoc, HybridQuery, HybridSearch, Scored,
};

pub struct HybridIndex {
    dense: DenseFlatIndex,
    sparse: SparseIndex,
    num_docs: usize,
}

impl HybridIndex {
    pub fn new(dims: usize) -> Self {
        Self {
            dense: DenseFlatIndex::new(dims),
            sparse: SparseIndex::new(),
            num_docs: 0,
        }
    }

    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    /// Combined memory: dense floats + sparse posting lists.
    pub fn memory_bytes(&self) -> usize {
        self.dense.memory_bytes() + self.sparse.memory_bytes()
    }
}

impl HybridSearch for HybridIndex {
    fn insert(&mut self, doc: HybridDoc) {
        self.dense.insert_doc(&doc);
        self.sparse.insert_doc(&doc);
        self.num_docs += 1;
    }

    fn search_dense(&self, q: &HybridQuery, k: usize) -> Vec<Scored> {
        self.dense.search(q, k)
    }

    fn search_sparse(&self, q: &HybridQuery, k: usize) -> Vec<Scored> {
        self.sparse.search(q, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DenseVec, SparseVec};

    fn build_index() -> (HybridIndex, Vec<HybridDoc>) {
        let mut idx = HybridIndex::new(4);
        let docs = vec![
            HybridDoc {
                id: 0,
                dense: DenseVec::new(vec![1.0, 0.0, 0.0, 0.0]),
                sparse: SparseVec::new(vec![(1, 1.0), (2, 0.5)]),
            },
            HybridDoc {
                id: 1,
                dense: DenseVec::new(vec![0.0, 1.0, 0.0, 0.0]),
                sparse: SparseVec::new(vec![(2, 1.0), (3, 0.5)]),
            },
            HybridDoc {
                id: 2,
                dense: DenseVec::new(vec![0.0, 0.0, 1.0, 0.0]),
                sparse: SparseVec::new(vec![(3, 1.0), (4, 0.5)]),
            },
        ];
        for doc in &docs {
            idx.insert(doc.clone());
        }
        (idx, docs)
    }

    #[test]
    fn dense_search_finds_nearest() {
        let (idx, _) = build_index();
        let q = HybridQuery {
            dense: DenseVec::new(vec![0.9, 0.1, 0.0, 0.0]),
            sparse: SparseVec::default(),
        };
        let r = idx.search_dense(&q, 1);
        assert_eq!(r[0].id, 0);
    }

    #[test]
    fn sparse_search_finds_term_match() {
        let (idx, _) = build_index();
        let q = HybridQuery {
            dense: DenseVec::new(vec![0.0; 4]),
            sparse: SparseVec::new(vec![(3, 1.0)]), // term 3 in docs 1 and 2
        };
        let r = idx.search_sparse(&q, 2);
        let ids: Vec<u32> = r.iter().map(|s| s.id).collect();
        assert!(ids.contains(&1) && ids.contains(&2));
    }

    #[test]
    fn rrf_fuses_both_signals() {
        let (idx, _) = build_index();
        // Dense query points at doc 0; sparse query points at doc 2.
        let q = HybridQuery {
            dense: DenseVec::new(vec![1.0, 0.0, 0.0, 0.0]),
            sparse: SparseVec::new(vec![(3, 1.0), (4, 1.0)]),
        };
        let r = idx.search_rrf(&q, 3, 3);
        assert_eq!(r.len(), 3, "should return 3 results");
        // Both doc 0 (dense match) and doc 2 (sparse match) should appear.
        let ids: Vec<u32> = r.iter().map(|s| s.id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
    }

    #[test]
    fn hybrid_index_memory_nonzero() {
        let (idx, _) = build_index();
        assert!(idx.memory_bytes() > 0);
    }
}
