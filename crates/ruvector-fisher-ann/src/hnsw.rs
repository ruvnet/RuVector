//! A compact, from-scratch, single-threaded HNSW variant used only for
//! this experiment's metric ablation.
//!
//! This is *not* `ruvector-core`'s production HNSW — it exists so the same
//! graph-construction and greedy-search algorithm can be instantiated with
//! two different distance metrics ([`FisherWeighted`](crate::types::FisherWeighted)
//! for candidate A, [`PlainL2`](crate::types::PlainL2) for candidate B) and
//! compared apples-to-apples. Wiring a Fisher-weighted metric into
//! `ruvector-core`'s actual HNSW (which is metric-fixed today) is listed as
//! future work in the nightly report rather than attempted here.
//!
//! Algorithm: standard multi-layer HNSW (Malkov & Yashunin, 2016) —
//! exponential level sampling, greedy layer descent to find an entry
//! point, bounded best-first search per layer, and simple distance-based
//! neighbor pruning (not the heuristic diversity selection from the paper,
//! which is unnecessary at this scale and would add complexity without
//! changing what this experiment measures).

use crate::types::{TaskMetric, TaskRecord};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy)]
struct Scored(f32, u32);
impl PartialEq for Scored {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for Scored {}
impl Ord for Scored {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Scored {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Reverse ordering wrapper so a `BinaryHeap<MinScored>` pops the
/// *smallest* distance first (a min-heap), used for the exploration
/// frontier during search.
#[derive(Clone, Copy)]
struct MinScored(Scored);
impl PartialEq for MinScored {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for MinScored {}
impl PartialOrd for MinScored {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinScored {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.cmp(&self.0)
    }
}

pub struct HnswIndex<M: TaskMetric> {
    metric: M,
    records: Vec<TaskRecord>,
    /// `neighbors[node][level]` = neighbor ids at that level.
    neighbors: Vec<Vec<Vec<u32>>>,
    entry_point: Option<usize>,
    max_level: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ml: f64,
    rng: StdRng,
}

fn search_layer<M: TaskMetric>(
    metric: &M,
    records: &[TaskRecord],
    neighbors: &[Vec<Vec<u32>>],
    query: &[f32],
    entry_points: &[usize],
    ef: usize,
    level: usize,
) -> Vec<(f32, usize)> {
    let n = records.len();
    let mut visited = vec![false; n];
    let mut candidates: BinaryHeap<MinScored> = BinaryHeap::new();
    let mut result: BinaryHeap<Scored> = BinaryHeap::new();

    for &ep in entry_points {
        if !visited[ep] {
            visited[ep] = true;
            let d = metric.distance(query, &records[ep]);
            candidates.push(MinScored(Scored(d, ep as u32)));
            result.push(Scored(d, ep as u32));
        }
    }

    while let Some(MinScored(Scored(d, c))) = candidates.pop() {
        if result.len() >= ef {
            if let Some(worst) = result.peek() {
                if d > worst.0 {
                    break;
                }
            }
        }
        if level >= neighbors[c as usize].len() {
            continue;
        }
        for &nb in &neighbors[c as usize][level] {
            if !visited[nb as usize] {
                visited[nb as usize] = true;
                let dn = metric.distance(query, &records[nb as usize]);
                let should_add =
                    result.len() < ef || result.peek().map(|w| dn < w.0).unwrap_or(true);
                if should_add {
                    candidates.push(MinScored(Scored(dn, nb)));
                    result.push(Scored(dn, nb));
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }
    }

    let mut out: Vec<(f32, usize)> = result
        .into_sorted_vec()
        .into_iter()
        .map(|s| (s.0, s.1 as usize))
        .collect();
    out.truncate(ef);
    out
}

impl<M: TaskMetric> HnswIndex<M> {
    pub fn new(metric: M, m: usize, ef_construction: usize, seed: u64) -> Self {
        Self {
            metric,
            records: Vec::new(),
            neighbors: Vec::new(),
            entry_point: None,
            max_level: 0,
            m,
            m0: m * 2,
            ef_construction,
            ml: 1.0 / (m.max(2) as f64).ln(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn metric_name(&self) -> &'static str {
        self.metric.name()
    }

    /// Total bytes actually allocated for records + graph edges (measured,
    /// not estimated): `Vec<f32>` payloads plus `u32` neighbor ids.
    pub fn memory_bytes(&self) -> usize {
        let record_bytes: usize = self
            .records
            .iter()
            .map(|r| (r.weights.len() + r.fisher.len()) * std::mem::size_of::<f32>())
            .sum();
        let edge_bytes: usize = self
            .neighbors
            .iter()
            .map(|levels| {
                levels
                    .iter()
                    .map(|l| l.len() * std::mem::size_of::<u32>())
                    .sum::<usize>()
            })
            .sum();
        record_bytes + edge_bytes
    }

    fn sample_level(&mut self) -> usize {
        let r: f64 = self.rng.gen::<f64>().max(1e-12);
        (-r.ln() * self.ml).floor() as usize
    }

    pub fn insert(&mut self, record: TaskRecord) -> u32 {
        let id = self.records.len();
        let level = self.sample_level();
        let query = record.weights.clone();
        self.records.push(record);
        self.neighbors
            .push((0..=level).map(|_| Vec::new()).collect());

        let ep = match self.entry_point {
            None => {
                self.entry_point = Some(id);
                self.max_level = level;
                return id as u32;
            }
            Some(ep) => ep,
        };

        let mut ep_vec = vec![ep];
        for lc in ((level + 1)..=self.max_level).rev() {
            let res = search_layer(
                &self.metric,
                &self.records,
                &self.neighbors,
                &query,
                &ep_vec,
                1,
                lc,
            );
            if let Some(&(_, best)) = res.first() {
                ep_vec = vec![best];
            }
        }

        let top = level.min(self.max_level);
        for lc in (0..=top).rev() {
            let candidates = search_layer(
                &self.metric,
                &self.records,
                &self.neighbors,
                &query,
                &ep_vec,
                self.ef_construction,
                lc,
            );
            let m_max = if lc == 0 { self.m0 } else { self.m };
            let selected: Vec<u32> = candidates
                .iter()
                .take(m_max)
                .map(|&(_, nid)| nid as u32)
                .collect();
            self.neighbors[id][lc] = selected.clone();

            for &nb in &selected {
                if (lc) < self.neighbors[nb as usize].len() {
                    self.neighbors[nb as usize][lc].push(id as u32);
                    if self.neighbors[nb as usize][lc].len() > m_max {
                        let nb_weights = self.records[nb as usize].weights.clone();
                        let metric = &self.metric;
                        let records = &self.records;
                        self.neighbors[nb as usize][lc].sort_by(|&a, &b| {
                            let da = metric.distance(&nb_weights, &records[a as usize]);
                            let db = metric.distance(&nb_weights, &records[b as usize]);
                            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
                        });
                        self.neighbors[nb as usize][lc].truncate(m_max);
                    }
                }
            }
            if !candidates.is_empty() {
                ep_vec = candidates.iter().map(|&(_, nid)| nid).collect();
            }
        }

        if level > self.max_level {
            self.entry_point = Some(id);
            self.max_level = level;
        }
        id as u32
    }

    /// Approximate k-nearest-neighbor search. Returns up to `k` results
    /// sorted ascending by distance.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(f32, usize)> {
        let Some(mut ep) = self.entry_point else {
            return Vec::new();
        };
        for lc in (1..=self.max_level).rev() {
            let res = search_layer(
                &self.metric,
                &self.records,
                &self.neighbors,
                query,
                &[ep],
                1,
                lc,
            );
            if let Some(&(_, best)) = res.first() {
                ep = best;
            }
        }
        let ef = ef_search.max(k);
        let mut out = search_layer(
            &self.metric,
            &self.records,
            &self.neighbors,
            query,
            &[ep],
            ef,
            0,
        );
        out.truncate(k);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baseline::brute_force_topk;
    use crate::dataset::generate_tasks;
    use crate::types::{FisherWeighted, PlainL2};

    #[test]
    fn empty_index_search_returns_empty() {
        let idx = HnswIndex::new(PlainL2, 8, 32, 1);
        assert!(idx.search(&[0.0, 0.0], 5, 20).is_empty());
    }

    #[test]
    fn single_point_index() {
        let mut idx = HnswIndex::new(PlainL2, 8, 32, 1);
        idx.insert(TaskRecord {
            id: 0,
            weights: vec![1.0, 2.0],
            fisher: vec![1.0, 1.0],
            importance: 1.0,
        });
        let res = idx.search(&[1.0, 2.0], 5, 20);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].1, 0);
    }

    #[test]
    fn determinism_same_seed_same_graph() {
        let tasks = generate_tasks(300, 16, 6, 11);
        let build = |seed: u64| {
            let mut idx = HnswIndex::new(PlainL2, 8, 40, seed);
            for t in tasks.clone() {
                idx.insert(t);
            }
            idx
        };
        let a = build(99);
        let b = build(99);
        for n in 0..a.len() {
            assert_eq!(a.neighbors[n], b.neighbors[n], "node {n} diverged");
        }
    }

    #[test]
    fn small_scale_recall_is_high_for_matching_metric() {
        let tasks = generate_tasks(400, 24, 8, 5);
        let mut idx = HnswIndex::new(FisherWeighted, 12, 64, 2);
        for t in tasks.clone() {
            idx.insert(t);
        }
        let queries = crate::dataset::generate_queries(&tasks, 30, 5);
        let mut hits = 0usize;
        for (q, _) in &queries {
            let gt = brute_force_topk(FisherWeighted, &tasks, q, 10);
            let gt_ids: std::collections::HashSet<usize> = gt.iter().map(|&(_, i)| i).collect();
            let approx = idx.search(q, 10, 80);
            hits += approx.iter().filter(|&&(_, i)| gt_ids.contains(&i)).count();
        }
        let recall = hits as f32 / (30.0 * 10.0);
        assert!(recall >= 0.85, "recall@10 too low: {recall}");
    }
}
