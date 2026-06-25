/// Variant 2: EagerMask.
///
/// Pre-compute a bitset of authorised vector indices *before* the distance
/// scan.  Distance computation is skipped entirely for unauthorised vectors.
///
/// This is strictly better than PostFilter when the authorised fraction < 50%
/// (distance computation is the dominant cost).  At 100% access ratio both
/// variants are equivalent.
///
/// Production systems with role-based access control typically have low access
/// ratios (a user can see 5-20% of all stored vectors), making EagerMask a
/// practical default for per-vector capability gating.
use crate::{dist_sq, CapGatedIndex, CapMask, SearchResult, VecEntry};

pub struct EagerMaskIndex {
    entries: Vec<VecEntry>,
    dims: usize,
}

impl EagerMaskIndex {
    pub fn new(dims: usize) -> Self {
        EagerMaskIndex {
            entries: Vec::new(),
            dims,
        }
    }
}

impl CapGatedIndex for EagerMaskIndex {
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

        // Step 1: build authorised index set — O(n) bitset operations
        let auth_mask: Vec<bool> = self
            .entries
            .iter()
            .map(|e| holder.satisfies(e.required))
            .collect();

        // Step 2: compute distance ONLY for authorised vectors — O(auth_frac * n * d)
        let mut scored: Vec<(f32, usize)> = self
            .entries
            .iter()
            .zip(auth_mask.iter())
            .filter_map(|(e, &auth)| {
                if auth {
                    Some((dist_sq(query, &e.vector), e.id))
                } else {
                    None
                }
            })
            .collect();

        // Step 3: partial sort to find top-k
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored
            .into_iter()
            .take(k)
            .map(|(d, id)| SearchResult { id, dist_sq: d })
            .collect()
    }

    fn name(&self) -> &'static str {
        "EagerMask"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapMask;

    #[test]
    fn eager_mask_same_results_as_oracle() {
        use crate::oracle::Oracle;
        let mut eager = EagerMaskIndex::new(4);
        let mut oracle = Oracle::new(4);

        let vecs: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
            vec![0.9, 0.9, 0.9, 0.9],
            vec![0.2, 0.1, 0.0, 0.0],
        ];
        let caps = [
            CapMask::NONE,
            CapMask::single(0),
            CapMask::single(1),
            CapMask(0b11),
            CapMask::single(0),
        ];

        for (i, (v, c)) in vecs.iter().zip(caps.iter()).enumerate() {
            eager.insert(i, v.clone(), *c);
            oracle.insert(i, v.clone(), *c);
        }

        let query = vec![0.0f32, 0.0, 0.0, 0.0];
        let holder = CapMask::single(0);

        let er = eager.search(&query, 3, holder);
        let or = oracle.search(&query, 3, holder);

        let e_ids: std::collections::HashSet<usize> = er.iter().map(|r| r.id).collect();
        let o_ids: std::collections::HashSet<usize> = or.iter().map(|r| r.id).collect();
        assert_eq!(e_ids, o_ids);
    }

    #[test]
    fn eager_mask_zero_access() {
        let mut idx = EagerMaskIndex::new(2);
        idx.insert(0, vec![0.0, 0.0], CapMask::single(7));
        let results = idx.search(&[0.0, 0.0], 5, CapMask::single(0));
        assert!(results.is_empty());
    }
}
