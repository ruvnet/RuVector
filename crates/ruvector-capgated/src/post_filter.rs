/// Variant 1: PostFilter (baseline).
///
/// Scan all n vectors, compute distance for every vector regardless of
/// capability, then discard unauthorised results.  Maximum recall, but
/// wastes O(n) distance computations even when few vectors are accessible.
///
/// This is the strategy used by most production vector databases today
/// (Qdrant, pgvector, Weaviate) when filtering is applied after ANN search.
use crate::{dist_sq, CapGatedIndex, CapMask, SearchResult, VecEntry};

pub struct PostFilterIndex {
    entries: Vec<VecEntry>,
    dims: usize,
}

impl PostFilterIndex {
    pub fn new(dims: usize) -> Self {
        PostFilterIndex {
            entries: Vec::new(),
            dims,
        }
    }
}

impl CapGatedIndex for PostFilterIndex {
    fn insert(&mut self, id: usize, vector: Vec<f32>, required: CapMask) {
        assert_eq!(vector.len(), self.dims);
        self.entries.push(VecEntry {
            id,
            vector,
            required,
        });
    }

    fn search(&self, query: &[f32], k: usize, holder: CapMask) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dims);
        // Step 1: compute distance for ALL vectors
        let mut scored: Vec<(f32, usize, CapMask)> = self
            .entries
            .iter()
            .map(|e| (dist_sq(query, &e.vector), e.id, e.required))
            .collect();
        // Step 2: sort by distance
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // Step 3: filter and return top-k authorised
        scored
            .into_iter()
            .filter(|(_, _, req)| holder.satisfies(*req))
            .take(k)
            .map(|(d, id, _)| SearchResult { id, dist_sq: d })
            .collect()
    }

    fn name(&self) -> &'static str {
        "PostFilter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapMask;

    #[test]
    fn post_filter_returns_only_authorised() {
        let mut idx = PostFilterIndex::new(2);
        idx.insert(0, vec![0.0, 0.0], CapMask::single(0));
        idx.insert(1, vec![0.1, 0.0], CapMask::single(1)); // not authorised
        idx.insert(2, vec![0.2, 0.0], CapMask::NONE);

        let results = idx.search(&[0.0, 0.0], 10, CapMask::single(0));
        let ids: Vec<usize> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn post_filter_ordering_correct() {
        let mut idx = PostFilterIndex::new(2);
        idx.insert(0, vec![10.0, 0.0], CapMask::NONE);
        idx.insert(1, vec![1.0, 0.0], CapMask::NONE);
        idx.insert(2, vec![0.5, 0.0], CapMask::NONE);
        let results = idx.search(&[0.0, 0.0], 3, CapMask::NONE);
        assert_eq!(results[0].id, 2);
        assert_eq!(results[1].id, 1);
        assert_eq!(results[2].id, 0);
    }

    #[test]
    fn post_filter_k_limit() {
        let mut idx = PostFilterIndex::new(1);
        for i in 0..20usize {
            idx.insert(i, vec![i as f32], CapMask::NONE);
        }
        let results = idx.search(&[0.0], 5, CapMask::NONE);
        assert_eq!(results.len(), 5);
    }
}
