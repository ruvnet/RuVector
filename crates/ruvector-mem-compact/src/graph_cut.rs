//! Graph-Cut compactor: k-NN graph clustering with proportional farthest-point sampling.
//!
//! Algorithm (three phases):
//!
//! **Phase 1 — cluster discovery.**
//! Build a k-NN similarity graph and run union-find at `cluster_threshold` to
//! find "redundancy clusters" — groups of nearly-identical or highly-similar
//! vectors.  Unlike the threshold compactor, union-find handles transitivity:
//! A≈B and B≈C are merged even if A and C are only moderately similar.
//!
//! **Phase 2 — proportional sampling.**
//! Each cluster of size `m` contributes `max(1, ⌈m × target_ratio⌉)` kept
//! vectors.  Representatives are chosen by *farthest-point sampling* (FPS) inside
//! the cluster: the first rep is the cluster centroid, each subsequent rep is the
//! member that maximises the minimum cosine distance to all already-chosen reps.
//! This maximises intra-cluster coverage regardless of query direction.
//!
//! **Phase 3 — global trim / pad.**
//! Trim by discarding the globally-most-redundant vectors; pad (if rare) by
//! adding members farthest from all existing reps.
//!
//! Why FPS outperforms centroid-only selection: centroid-only picks one "average"
//! vector per cluster and fills the remaining budget randomly.  FPS spreads the
//! budget evenly across the cluster's spatial extent, so any query landing
//! anywhere in the cluster finds a representative nearby.

use crate::compactor::{cosine_sim, CompactionResult, MemoryCompactor, MemoryStore};
use std::time::Instant;

pub struct GraphCutCompactor {
    /// k for k-NN graph construction.
    pub k_neighbors: usize,
    /// Similarity threshold: edges below this are never included.
    /// Set to the expected cross-cluster similarity ceiling (e.g. 0.70 for
    /// moderately separated clusters).
    pub cluster_threshold: f32,
}

impl GraphCutCompactor {
    pub fn new(k_neighbors: usize, cluster_threshold: f32) -> Self {
        Self {
            k_neighbors,
            cluster_threshold,
        }
    }
}

// ── Union-Find ────────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

// ── Farthest-point sampling ───────────────────────────────────────────────────

/// Select `n_reps` vectors from `members` by farthest-point sampling.
/// Seeds with the member closest to the cluster centroid, then greedily
/// maximises min-distance to all selected reps.
fn fps_select(store: &MemoryStore, members: &[usize], n_reps: usize) -> Vec<usize> {
    let n = members.len();
    if n <= n_reps {
        return members.to_vec();
    }

    let dims = store.dims;

    // Compute cluster centroid (normalised).
    let mut centroid = vec![0.0f32; dims];
    for &m in members {
        for (d, v) in centroid.iter_mut().zip(store.entries[m].vector.iter()) {
            *d += v;
        }
    }
    let inv = 1.0 / n as f32;
    for d in centroid.iter_mut() {
        *d *= inv;
    }
    let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for d in centroid.iter_mut() {
            *d /= norm;
        }
    }

    // Seed: member closest to centroid.
    let seed = members
        .iter()
        .copied()
        .max_by(|&a, &b| {
            cosine_sim(&store.entries[a].vector, &centroid)
                .partial_cmp(&cosine_sim(&store.entries[b].vector, &centroid))
                .unwrap()
        })
        .unwrap();

    let mut selected: Vec<usize> = vec![seed];

    // min_sim_to_selected[i] = cosine similarity of members[i] to nearest selected.
    let mut min_sim: Vec<f32> = members
        .iter()
        .map(|&m| cosine_sim(&store.entries[m].vector, &store.entries[seed].vector))
        .collect();

    while selected.len() < n_reps {
        // Find the member with the LOWEST max-sim-to-selected (most diverse).
        let (next_local, _) = min_sim
            .iter()
            .enumerate()
            .filter(|(i, _)| !selected.contains(&members[*i]))
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let next = members[next_local];
        selected.push(next);

        // Update min_sim: newly added rep might be closer to some members.
        for (i, &m) in members.iter().enumerate() {
            let s = cosine_sim(&store.entries[m].vector, &store.entries[next].vector);
            if s > min_sim[i] {
                min_sim[i] = s;
            }
        }
    }

    selected
}

// ── Main implementation ───────────────────────────────────────────────────────

impl MemoryCompactor for GraphCutCompactor {
    fn name(&self) -> &'static str {
        "GraphCutCompact"
    }

    fn compact(&self, store: &MemoryStore, target_ratio: f32) -> CompactionResult {
        let t = Instant::now();
        let n = store.entries.len();
        let keep_count = ((n as f32) * target_ratio).ceil() as usize;

        if n == 0 {
            return CompactionResult {
                kept_ids: vec![],
                kept_ratio: 0.0,
                duration: t.elapsed(),
            };
        }
        if keep_count >= n {
            let ids = store.entries.iter().map(|e| e.id).collect();
            return CompactionResult {
                kept_ids: ids,
                kept_ratio: 1.0,
                duration: t.elapsed(),
            };
        }

        // ── Phase 1: build k-NN graph and find clusters ───────────────────────
        let mut uf = UnionFind::new(n);
        for i in 0..n {
            let mut row: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    (
                        j,
                        cosine_sim(&store.entries[i].vector, &store.entries[j].vector),
                    )
                })
                .collect();
            row.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (j, sim) in row.into_iter().take(self.k_neighbors) {
                if sim >= self.cluster_threshold {
                    uf.union(i, j);
                }
            }
        }

        let mut components: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..n {
            components.entry(uf.find(i)).or_default().push(i);
        }

        // ── Phase 2: proportional FPS within each cluster ────────────────────
        // Each cluster of size m keeps ⌈m × target_ratio⌉ reps (at least 1).
        let mut reps: Vec<usize> = Vec::with_capacity(keep_count);
        for members in components.values() {
            let n_reps = ((members.len() as f32 * target_ratio).ceil() as usize).max(1);
            reps.extend(fps_select(store, members, n_reps));
        }

        // ── Phase 3: trim / pad to exact keep_count ──────────────────────────
        if reps.len() > keep_count {
            // Trim: remove the globally most-redundant reps (highest max-sim to any neighbour).
            let reps_snapshot = reps.clone();
            reps.sort_unstable_by(|&a, &b| {
                let max_sim = |idx: usize| -> f32 {
                    reps_snapshot
                        .iter()
                        .filter(|&&r| r != idx)
                        .map(|&r| cosine_sim(&store.entries[idx].vector, &store.entries[r].vector))
                        .fold(f32::MIN, f32::max)
                };
                max_sim(b).partial_cmp(&max_sim(a)).unwrap()
            });
            reps.truncate(keep_count);
        } else if reps.len() < keep_count {
            // Pad: add non-rep vectors farthest from all existing reps.
            let kept_set: std::collections::HashSet<usize> = reps.iter().copied().collect();
            let mut candidates: Vec<(usize, f32)> = (0..n)
                .filter(|i| !kept_set.contains(i))
                .map(|i| {
                    let max_sim = reps
                        .iter()
                        .map(|&r| cosine_sim(&store.entries[i].vector, &store.entries[r].vector))
                        .fold(f32::MIN, f32::max);
                    (i, max_sim)
                })
                .collect();
            candidates.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap()); // ascending
            for (idx, _) in candidates.into_iter().take(keep_count - reps.len()) {
                reps.push(idx);
            }
        }

        let kept_ids: Vec<u64> = reps.iter().map(|&i| store.entries[i].id).collect();
        let kept_ratio = kept_ids.len() as f32 / n as f32;
        CompactionResult {
            kept_ids,
            kept_ratio,
            duration: t.elapsed(),
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{generate_store, DatasetParams};

    #[test]
    fn graph_cut_respects_target_ratio() {
        let params = DatasetParams {
            n: 200,
            dims: 32,
            n_clusters: 5,
            cluster_std: 0.2,
            seed: 1,
        };
        let store = generate_store(&params);
        let compactor = GraphCutCompactor::new(8, 0.70);
        let result = compactor.compact(&store, 0.5);
        let ratio = result.kept_ids.len() as f32 / store.len() as f32;
        assert!(
            ratio >= 0.40 && ratio <= 0.65,
            "kept ratio {ratio:.3} outside acceptable range"
        );
    }

    #[test]
    fn all_returned_ids_valid() {
        let params = DatasetParams {
            n: 100,
            dims: 16,
            n_clusters: 4,
            cluster_std: 0.1,
            seed: 7,
        };
        let store = generate_store(&params);
        let compactor = GraphCutCompactor::new(4, 0.60);
        let result = compactor.compact(&store, 0.5);
        let valid: std::collections::HashSet<u64> = store.entries.iter().map(|e| e.id).collect();
        for id in &result.kept_ids {
            assert!(valid.contains(id), "unknown id {id}");
        }
    }
}
