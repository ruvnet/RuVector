use crate::distance::l2sq;

/// Linear-scan flat segment. O(n) query, O(1) insert.
/// Used as the hot tier in the LSM index: new writes land here instantly.
pub struct FlatSegment {
    ids: Vec<u64>,
    vecs: Vec<Vec<f32>>,
    dims: usize,
}

impl FlatSegment {
    pub fn new(dims: usize) -> Self {
        Self {
            ids: Vec::new(),
            vecs: Vec::new(),
            dims,
        }
    }

    pub fn insert(&mut self, id: u64, vec: Vec<f32>) {
        self.ids.push(id);
        self.vecs.push(vec);
    }

    /// Search for the k nearest neighbours. Always 100% recall (brute force).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(f32, u64)> {
        if self.ids.is_empty() {
            return Vec::new();
        }
        let mut dists: Vec<(f32, u64)> = self
            .ids
            .iter()
            .zip(self.vecs.iter())
            .map(|(&id, v)| (l2sq(query, v), id))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Drain all entries for compaction; leaves the segment empty.
    pub fn drain_all(&mut self) -> Vec<(u64, Vec<f32>)> {
        let ids = std::mem::take(&mut self.ids);
        let vecs = std::mem::take(&mut self.vecs);
        ids.into_iter().zip(vecs).collect()
    }

    /// Estimated heap allocation in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.ids.len() * (8 + self.dims * 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_nearest_is_self() {
        let mut seg = FlatSegment::new(4);
        for i in 0u64..20 {
            seg.insert(i, vec![i as f32, 0.0, 0.0, 0.0]);
        }
        let q = vec![5.0_f32, 0.0, 0.0, 0.0];
        let res = seg.search(&q, 1);
        assert_eq!(res[0].1, 5);
    }

    #[test]
    fn flat_drain_clears() {
        let mut seg = FlatSegment::new(2);
        seg.insert(1, vec![1.0, 2.0]);
        seg.insert(2, vec![3.0, 4.0]);
        let drained = seg.drain_all();
        assert_eq!(drained.len(), 2);
        assert!(seg.is_empty());
    }
}
