//! Variant 1 — FlatScan: exact brute-force L2 search (baseline / ground truth oracle).

use crate::{CommunitySearch, Hit, VectorMeta, l2_sq};

/// Stores raw f32 vectors and performs an exhaustive linear scan.
/// Used as the oracle for measuring recall of the other two variants.
pub struct FlatScan {
    vectors: Vec<Vec<f32>>,
    metas: Vec<VectorMeta>,
}

impl FlatScan {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            metas: Vec::new(),
        }
    }

    /// Read-only access to metadata (needed for cross-variant community scoring).
    pub fn metas(&self) -> &[VectorMeta] {
        &self.metas
    }
}

impl CommunitySearch for FlatScan {
    fn insert(&mut self, vector: &[f32], community: usize) {
        let id = self.vectors.len();
        self.metas.push(VectorMeta { id, community });
        self.vectors.push(vector.to_vec());
    }

    fn build(&mut self) {
        // No index to build for brute-force scan.
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut heap: Vec<Hit> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(id, v)| Hit { id, distance: l2_sq(query, v) })
            .collect();
        heap.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        heap.truncate(k);
        heap
    }

    fn memory_bytes(&self) -> usize {
        let vec_bytes: usize = self.vectors.iter().map(|v| v.len() * 4).sum();
        let meta_bytes = self.metas.len() * std::mem::size_of::<VectorMeta>();
        vec_bytes + meta_bytes
    }

    fn name(&self) -> &'static str {
        "FlatScan"
    }
}
