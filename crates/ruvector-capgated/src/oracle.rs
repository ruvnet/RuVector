/// Brute-force oracle: used as ground truth for recall computation.
///
/// Scans all vectors, filters by capability, returns exact top-k by distance.
use crate::{dist_sq, CapGatedIndex, CapMask, SearchResult, VecEntry};

pub struct Oracle {
    entries: Vec<VecEntry>,
    dims: usize,
}

impl Oracle {
    pub fn new(dims: usize) -> Self {
        Oracle {
            entries: Vec::new(),
            dims,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many vectors are accessible to a given holder?
    pub fn authorized_count(&self, holder: CapMask) -> usize {
        self.entries
            .iter()
            .filter(|e| holder.satisfies(e.required))
            .count()
    }
}

impl CapGatedIndex for Oracle {
    fn insert(&mut self, id: usize, vector: Vec<f32>, required: CapMask) {
        assert_eq!(vector.len(), self.dims, "dimension mismatch on insert");
        self.entries.push(VecEntry {
            id,
            vector,
            required,
        });
    }

    fn search(&self, query: &[f32], k: usize, holder: CapMask) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dims, "dimension mismatch on query");
        let mut scored: Vec<SearchResult> = self
            .entries
            .iter()
            .filter(|e| holder.satisfies(e.required))
            .map(|e| SearchResult {
                id: e.id,
                dist_sq: dist_sq(query, &e.vector),
            })
            .collect();
        scored.sort_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap());
        scored.truncate(k);
        scored
    }

    fn name(&self) -> &'static str {
        "Oracle (brute-force, exact)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapMask;

    fn make_oracle() -> Oracle {
        let mut o = Oracle::new(2);
        // id=0 requires cap bit 0
        o.insert(0, vec![0.0, 0.0], CapMask::single(0));
        // id=1 requires cap bit 1
        o.insert(1, vec![1.0, 0.0], CapMask::single(1));
        // id=2 requires cap bit 0 AND bit 1
        o.insert(2, vec![0.5, 0.0], CapMask(0b11));
        // id=3 requires no capabilities
        o.insert(3, vec![0.2, 0.0], CapMask::NONE);
        o
    }

    #[test]
    fn oracle_filters_by_cap() {
        let o = make_oracle();
        let holder = CapMask::single(0); // holds bit 0 only
        let results = o.search(&[0.0, 0.0], 10, holder);
        let ids: Vec<usize> = results.iter().map(|r| r.id).collect();
        // Should get id=0 (needs bit0), id=3 (needs nothing), NOT id=1 (needs bit1), NOT id=2 (needs both)
        assert!(ids.contains(&0));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&1));
        assert!(!ids.contains(&2));
    }

    #[test]
    fn oracle_sorted_by_distance() {
        let o = make_oracle();
        let holder = CapMask::ALL;
        let results = o.search(&[0.0, 0.0], 10, holder);
        for w in results.windows(2) {
            assert!(w[0].dist_sq <= w[1].dist_sq);
        }
    }

    #[test]
    fn oracle_top_k_respected() {
        let o = make_oracle();
        let results = o.search(&[0.0, 0.0], 2, CapMask::ALL);
        assert!(results.len() <= 2);
    }

    #[test]
    fn oracle_no_access_returns_empty() {
        let mut o = Oracle::new(2);
        o.insert(0, vec![0.0, 0.0], CapMask::single(7)); // requires bit 7
        let holder = CapMask::single(0); // holds bit 0
        let results = o.search(&[0.0, 0.0], 10, holder);
        assert!(results.is_empty());
    }
}
