//! Ordered candidates and bounded top-k selection.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// One search result: dense id and distance (lower is closer).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Dense internal id (`vectors.iid`).
    pub iid: u32,
    /// Distance under the collection metric. Approximate (code-space) from
    /// the plain `search` methods; exact f32 after a rerank.
    pub distance: f32,
}

/// Candidate ordered by `(distance, id)` with `total_cmp`, so ties break
/// deterministically on every platform.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Cand {
    pub d: f32,
    pub id: u32,
}

impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.d.total_cmp(&o.d).then(self.id.cmp(&o.id))
    }
}

/// Keeps the `k` smallest candidates.
pub(crate) struct TopK {
    k: usize,
    heap: BinaryHeap<Cand>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            // k comes from the request: never pre-allocate from it unbounded.
            heap: BinaryHeap::with_capacity(k.min(1024) + 1),
        }
    }

    /// Current worst kept distance, if full.
    pub fn bound(&self) -> Option<f32> {
        if self.heap.len() >= self.k {
            self.heap.peek().map(|c| c.d)
        } else {
            None
        }
    }

    pub fn push(&mut self, c: Cand) {
        if self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(c);
        } else if let Some(top) = self.heap.peek() {
            if c < *top {
                self.heap.pop();
                self.heap.push(c);
            }
        }
    }

    /// Ascending by `(distance, id)`.
    pub fn into_sorted(self) -> Vec<Cand> {
        self.heap.into_sorted_vec()
    }
}

pub(crate) fn to_hits(c: Vec<Cand>) -> Vec<Hit> {
    c.into_iter()
        .map(|c| Hit {
            iid: c.id,
            distance: c.d,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_smallest_with_id_tiebreak() {
        let mut t = TopK::new(3);
        for (d, id) in [(3.0, 1), (1.0, 9), (1.0, 2), (5.0, 0), (0.5, 7)] {
            t.push(Cand { d, id });
        }
        let ids: Vec<u32> = t.into_sorted().iter().map(|c| c.id).collect();
        assert_eq!(ids, vec![7, 2, 9]);
        let mut z = TopK::new(0);
        z.push(Cand { d: 1.0, id: 1 });
        assert!(z.into_sorted().is_empty());
    }
}
