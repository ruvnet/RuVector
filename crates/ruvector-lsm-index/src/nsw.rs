use crate::distance::l2sq;
use std::collections::HashSet;

/// Navigable Small World graph (HNSW layer-0).
///
/// Provides approximate nearest-neighbour search after batch construction
/// or incremental online inserts.
///
/// Design notes
/// ─────────────
/// • One layer only (no hierarchical skip graph).
/// • `m` target neighbours; `m_max = 2*m` hard edge cap.
/// • `ef_build` controls search width during construction.
/// • `ef_search` is derived per-query as `max(k, ef_build)`.
/// • Entry point: `sqrt(n)` diversified samples, pick best 3 seeds.
pub struct NswSegment {
    ids: Vec<u64>,
    vecs: Vec<Vec<f32>>,
    neighbors: Vec<Vec<usize>>,
    m: usize,
    m_max: usize,
    ef_build: usize,
    dims: usize,
}

impl NswSegment {
    pub fn new(dims: usize, m: usize, ef_build: usize) -> Self {
        Self {
            ids: Vec::new(),
            vecs: Vec::new(),
            neighbors: Vec::new(),
            m,
            m_max: m * 2,
            ef_build,
            dims,
        }
    }

    /// Batch-build from a list of (id, vec) entries.
    pub fn build_from(
        entries: Vec<(u64, Vec<f32>)>,
        dims: usize,
        m: usize,
        ef_build: usize,
    ) -> Self {
        let mut seg = Self::new(dims, m, ef_build);
        for (id, vec) in entries {
            seg.insert_internal(id, vec);
        }
        seg
    }

    /// Online insert.
    pub fn insert(&mut self, id: u64, vec: Vec<f32>) {
        self.insert_internal(id, vec);
    }

    /// Search for k approximate nearest neighbours.
    ///
    /// `ef_search` overrides the default ef (defaults to `max(k, ef_build * 3)`).
    /// Higher ef → better recall at higher latency cost.
    pub fn search_ef(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(f32, u64)> {
        if self.ids.is_empty() {
            return Vec::new();
        }
        let ef = ef_search.max(k).min(self.ids.len());
        let init = self.pick_entry_points(query, self.ids.len());
        let candidates = self.greedy_search(query, init, ef, self.ids.len());
        candidates
            .into_iter()
            .take(k)
            .map(|(d, i)| (d, self.ids[i]))
            .collect()
    }

    /// Search using the default ef = max(k, ef_build * 3).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(f32, u64)> {
        self.search_ef(query, k, k.max(self.ef_build * 3))
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Drain all entries; resets the graph. Used during LSM tier merge.
    pub fn drain_all(&mut self) -> Vec<(u64, Vec<f32>)> {
        let ids = std::mem::take(&mut self.ids);
        let vecs = std::mem::take(&mut self.vecs);
        self.neighbors.clear();
        ids.into_iter().zip(vecs).collect()
    }

    /// Estimated heap allocation in bytes (vectors + graph edges).
    pub fn memory_bytes(&self) -> usize {
        let vec_bytes = self.ids.len() * (8 + self.dims * 4);
        let edge_bytes: usize = self.neighbors.iter().map(|nb| nb.len() * 8).sum();
        vec_bytes + edge_bytes
    }

    // ─── private ───────────────────────────────────────────────────────────────

    fn insert_internal(&mut self, id: u64, vec: Vec<f32>) {
        let idx = self.ids.len();
        self.ids.push(id);
        self.vecs.push(vec);
        self.neighbors.push(Vec::new());

        if idx == 0 {
            return; // first node: no edges
        }

        let ef = self.ef_build.min(idx);
        let init = self.pick_entry_points(&self.vecs[idx].clone(), idx);
        let candidates = self.greedy_search(&self.vecs[idx].clone(), init, ef, idx);

        // Connect new node to best-M neighbours (simple heuristic selection).
        let connect: Vec<usize> = candidates.iter().take(self.m).map(|(_, i)| *i).collect();
        for &nb in &connect {
            self.neighbors[idx].push(nb);
            if self.neighbors[nb].len() < self.m_max {
                self.neighbors[nb].push(idx);
            }
        }
    }

    /// Pick diverse seed entry points by sampling sqrt(n) nodes evenly,
    /// then keeping the 3 closest to the query.
    fn pick_entry_points(&self, query: &[f32], exclude_from: usize) -> Vec<(f32, usize)> {
        let n = exclude_from.min(self.ids.len());
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![(l2sq(query, &self.vecs[0]), 0)];
        }

        // Sample ~sqrt(n) diverse nodes evenly spaced through the insertion order.
        let n_samples = ((n as f64).sqrt() as usize + 2).min(n);
        let step = n / n_samples;
        let step = step.max(1);

        let mut eps: Vec<(f32, usize)> = (0..n_samples)
            .filter_map(|i| {
                let idx = (i * step).min(n - 1);
                Some((l2sq(query, &self.vecs[idx]), idx))
            })
            .collect();

        eps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // Keep best 8 seeds — more seeds give better graph coverage at ~8x seed init cost.
        eps.truncate(8);
        eps
    }

    /// Greedy beam search returning ≤ef (dist, idx) pairs sorted by distance.
    fn greedy_search(
        &self,
        query: &[f32],
        init: Vec<(f32, usize)>,
        ef: usize,
        exclude_from: usize,
    ) -> Vec<(f32, usize)> {
        let n = self.ids.len().min(exclude_from);
        if n == 0 {
            return Vec::new();
        }

        let mut visited: HashSet<usize> = HashSet::new();
        let mut candidates: Vec<(f32, usize)> = Vec::new(); // work queue
        let mut results: Vec<(f32, usize)> = Vec::new(); // top-ef buffer

        for (d, ep) in init {
            if ep < n && visited.insert(ep) {
                candidates.push((d, ep));
                results.push((d, ep));
            }
        }

        while !candidates.is_empty() {
            // Pop closest candidate.
            let best_pos = candidates
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1 .0
                        .partial_cmp(&b.1 .0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap();
            let (best_dist, best_idx) = candidates.swap_remove(best_pos);

            // Early exit: best remaining candidate is already worse than ef-th result.
            if results.len() >= ef {
                let worst = worst_dist(&results);
                if best_dist > worst {
                    break;
                }
            }

            for &nb in &self.neighbors[best_idx] {
                if nb >= n || !visited.insert(nb) {
                    continue;
                }
                let d = l2sq(query, &self.vecs[nb]);
                let should_add = results.len() < ef || d < worst_dist(&results);
                if should_add {
                    candidates.push((d, nb));
                    results.push((d, nb));
                    if results.len() > ef {
                        trim_worst(&mut results);
                    }
                }
            }
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

#[inline]
fn worst_dist(results: &[(f32, usize)]) -> f32 {
    results
        .iter()
        .map(|(d, _)| *d)
        .fold(f32::NEG_INFINITY, f32::max)
}

fn trim_worst(results: &mut Vec<(f32, usize)>) {
    if let Some(pos) = results
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1 .0
                .partial_cmp(&b.1 .0)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
    {
        results.swap_remove(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatSegment;

    fn make_vecs(n: usize, dims: usize, seed: u32) -> Vec<(u64, Vec<f32>)> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dims)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 17;
                        s ^= s << 5;
                        (s as f64 / u32::MAX as f64) as f32 * 2.0 - 1.0
                    })
                    .collect();
                (i as u64, v)
            })
            .collect()
    }

    #[test]
    fn nsw_recall_at_least_50_pct() {
        let n = 500;
        let dims = 32;
        let k = 10;
        let vecs = make_vecs(n, dims, 42);
        let queries = make_vecs(50, dims, 9999);

        let index = NswSegment::build_from(
            vecs.iter().map(|(id, v)| (*id, v.clone())).collect(),
            dims,
            16,
            40,
        );

        let mut flat = FlatSegment::new(dims);
        for (id, v) in &vecs {
            flat.insert(*id, v.clone());
        }

        let mut hits = 0usize;
        let mut total = 0usize;
        for (_, q) in &queries {
            let gt: HashSet<u64> = flat.search(q, k).iter().map(|(_, id)| *id).collect();
            let res: HashSet<u64> = index.search(q, k).iter().map(|(_, id)| *id).collect();
            hits += gt.intersection(&res).count();
            total += k;
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.50,
            "NSW recall@10 = {:.3}, expected >=0.50",
            recall
        );
    }

    #[test]
    fn nsw_drain_clears_graph() {
        let mut seg = NswSegment::new(4, 8, 20);
        for i in 0u64..50 {
            seg.insert(i, vec![i as f32 / 50.0, 0.0, 0.0, 0.0]);
        }
        assert_eq!(seg.len(), 50);
        let drained = seg.drain_all();
        assert_eq!(drained.len(), 50);
        assert!(seg.is_empty());
    }
}
