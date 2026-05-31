//! Spectral graph eviction: select which agent memories to evict using the
//! Fiedler vector of the memory similarity graph.
//!
//! ## Algorithm
//!
//! 1. Build a k-NN cosine-similarity graph G = (V, E) on the memory vectors.
//! 2. Compute the normalized graph Laplacian L̃ = I − D^{−½} A D^{−½}.
//! 3. Find the Fiedler vector v₂ (eigenvector of the *second-smallest* eigenvalue
//!    of L̃) via power iteration on (I − L̃) — i.e., the matrix of the random walk.
//! 4. Partition V by the sign of v₂: nodes with v₂[i] < 0 form the "B-side".
//! 5. Evict all nodes on the **smaller** side (minority partition), up to the
//!    configured eviction fraction.
//!
//! The Fiedler vector identifies the minimum-conductance cut — nodes on the
//! minority side are the least structurally embedded in the memory graph and
//! are the safest to evict while preserving high-recall neighbourhoods.
//!
//! ## Complexity
//! Building k-NN: O(N²·D) for the naive implementation used here.
//! Power iteration: O(N·k·iters) — negligible for N ≤ 5K.
//!
//! For production N > 100K, the k-NN step should use the HNSW index already
//! in `ruvector-core`.

use rand::{rngs::SmallRng, Rng, SeedableRng};

/// A single memory entry held by the eviction oracle.
#[derive(Clone, Debug)]
pub struct MemoryEntry {
    /// Numeric ID within the store.
    pub id: usize,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Last-access logical timestamp (higher = more recent).
    pub last_access: u64,
}

/// Which nodes to evict, returned by each [`EvictionPolicy`].
pub struct EvictionPlan {
    /// IDs of entries to remove.
    pub evict: Vec<usize>,
    /// Conductance of the surviving graph (1.0 = completely connected, lower = sparser).
    pub conductance: f64,
}

/// Trait for the three eviction policies benchmarked in this crate.
pub trait EvictionPolicy {
    /// Given the full memory store, select which entries to evict.
    /// `target_size` is the desired post-eviction count.
    fn plan_eviction(&mut self, entries: &[MemoryEntry], target_size: usize) -> EvictionPlan;

    /// Human-readable policy name.
    fn name(&self) -> &str;
}

// ─── Policy 1: Random Eviction ───────────────────────────────────────────────

/// Evict a uniformly random subset — the naive baseline with no graph awareness.
pub struct RandomEviction {
    rng: SmallRng,
}

impl RandomEviction {
    /// Create a random eviction policy with a fixed seed for reproducibility.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl EvictionPolicy for RandomEviction {
    fn plan_eviction(&mut self, entries: &[MemoryEntry], target_size: usize) -> EvictionPlan {
        let n = entries.len();
        if n <= target_size {
            return EvictionPlan {
                evict: vec![],
                conductance: 1.0,
            };
        }
        let keep = target_size;
        let evict_count = n - keep;
        // Reservoir sample without replacement
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..evict_count {
            let j = self.rng.gen_range(i..n);
            indices.swap(i, j);
        }
        let evict: Vec<usize> = indices[..evict_count]
            .iter()
            .map(|&i| entries[i].id)
            .collect();
        EvictionPlan {
            evict,
            conductance: 1.0,
        }
    }

    fn name(&self) -> &str {
        "RandomEviction"
    }
}

// ─── Policy 2: LRU Eviction ──────────────────────────────────────────────────

/// Evict the least-recently-used entries — the production baseline.
pub struct LruEviction;

impl EvictionPolicy for LruEviction {
    fn plan_eviction(&mut self, entries: &[MemoryEntry], target_size: usize) -> EvictionPlan {
        let n = entries.len();
        if n <= target_size {
            return EvictionPlan {
                evict: vec![],
                conductance: 1.0,
            };
        }
        let evict_count = n - target_size;
        // Sort by last_access ascending; lowest timestamps are evicted
        let mut by_age: Vec<(usize, u64)> = entries.iter().map(|e| (e.id, e.last_access)).collect();
        by_age.sort_by_key(|&(_, ts)| ts);
        let evict = by_age[..evict_count].iter().map(|&(id, _)| id).collect();
        EvictionPlan {
            evict,
            conductance: 1.0,
        }
    }

    fn name(&self) -> &str {
        "LruEviction"
    }
}

// ─── Policy 3: Spectral Eviction ─────────────────────────────────────────────

/// Evict the minority partition of the Fiedler (spectral graph) cut.
///
/// Preserves high-recall neighbourhoods by keeping structurally central nodes
/// while evicting semantically peripheral ones regardless of access time.
pub struct SpectralEviction {
    /// k in k-NN graph construction.
    pub knn: usize,
    /// Power iteration steps for Fiedler vector estimation.
    pub iters: usize,
    rng: SmallRng,
}

impl SpectralEviction {
    /// Create a spectral eviction policy.
    ///
    /// # Parameters
    /// * `knn`   – number of neighbours per node in the similarity graph
    /// * `iters` – power-iteration steps (20–40 is sufficient for PoC graphs)
    /// * `seed`  – RNG seed for the initial random vector
    pub fn new(knn: usize, iters: usize, seed: u64) -> Self {
        Self {
            knn,
            iters,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na < 1e-12 || nb < 1e-12 {
            0.0
        } else {
            (dot / (na * nb)).clamp(-1.0, 1.0)
        }
    }

    /// Build k-NN adjacency as edge weights (symmetric, only positive similarities).
    fn build_knn_adj(entries: &[MemoryEntry], k: usize) -> Vec<Vec<(usize, f64)>> {
        let n = entries.len();
        let mut adj = vec![vec![]; n];
        for i in 0..n {
            let mut sims: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, Self::cosine_sim(&entries[i].vector, &entries[j].vector)))
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (j, sim) in sims.into_iter().take(k) {
                let w = sim.max(0.0) as f64;
                adj[i].push((j, w));
                adj[j].push((i, w)); // ensure symmetry
            }
        }
        // Deduplicate (symmetry insertions can create duplicates)
        for row in adj.iter_mut() {
            row.sort_by_key(|&(j, _)| j);
            row.dedup_by_key(|e| e.0);
        }
        adj
    }

    /// Estimate the Fiedler vector via power iteration on the random-walk matrix P = D^{-1}A.
    /// The Fiedler vector corresponds to the *second* eigenvector of P (after the constant
    /// leading vector). We deflate by projecting out the first eigenvector (all-ones, scaled).
    fn fiedler_vector(adj: &[Vec<(usize, f64)>], iters: usize, rng: &mut SmallRng) -> Vec<f64> {
        let n = adj.len();
        // Degree
        let deg: Vec<f64> = adj
            .iter()
            .map(|row| row.iter().map(|&(_, w)| w).sum::<f64>().max(1e-12))
            .collect();

        // Random initialisation
        let mut v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();

        for _ in 0..iters {
            // Deflate: remove component along 1/sqrt(n) (constant eigenvector of P)
            let mean: f64 = v.iter().sum::<f64>() / n as f64;
            for vi in v.iter_mut() {
                *vi -= mean;
            }
            // Apply P = D^{-1} A
            let mut pv = vec![0.0f64; n];
            for i in 0..n {
                for &(j, w) in &adj[i] {
                    pv[i] += w * v[j];
                }
                pv[i] /= deg[i];
            }
            // Normalise
            let norm: f64 = pv.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            for vi in pv.iter_mut() {
                *vi /= norm;
            }
            v = pv;
        }
        v
    }

    /// Graph conductance of the subset S in the graph given by adj and degrees.
    fn conductance(adj: &[Vec<(usize, f64)>], deg: &[f64], side_b: &[bool]) -> f64 {
        let vol_b: f64 = deg
            .iter()
            .zip(side_b.iter())
            .filter(|&(_, &b)| b)
            .map(|(d, _)| d)
            .sum();
        let vol_total: f64 = deg.iter().sum();
        let vol_comp = vol_total - vol_b;
        let denom = vol_b.min(vol_comp).max(1e-12);
        let cut: f64 = adj
            .iter()
            .enumerate()
            .flat_map(|(i, row)| row.iter().map(move |&(j, w)| (i, j, w)))
            .filter(|&(i, j, _)| side_b[i] != side_b[j])
            .map(|(_, _, w)| w)
            .sum::<f64>()
            / 2.0; // each edge counted twice
        cut / denom
    }
}

impl EvictionPolicy for SpectralEviction {
    fn plan_eviction(&mut self, entries: &[MemoryEntry], target_size: usize) -> EvictionPlan {
        let n = entries.len();
        if n <= target_size {
            return EvictionPlan {
                evict: vec![],
                conductance: 1.0,
            };
        }
        let evict_count = n - target_size;
        let adj = Self::build_knn_adj(entries, self.knn);
        let fv = Self::fiedler_vector(&adj, self.iters, &mut self.rng);
        let deg: Vec<f64> = adj
            .iter()
            .map(|row| row.iter().map(|&(_, w)| w).sum::<f64>().max(1e-12))
            .collect();

        // Sort nodes by Fiedler value; most negative = most peripheral
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| fv[a].partial_cmp(&fv[b]).unwrap());

        let evict_ids: Vec<usize> = order[..evict_count]
            .iter()
            .map(|&i| entries[i].id)
            .collect();
        // Rebuild side_b indexed by entry position for conductance computation
        let mut side_b_pos = vec![false; n];
        for &i in &order[..evict_count] {
            side_b_pos[i] = true;
        }
        let cond = Self::conductance(&adj, &deg, &side_b_pos);
        EvictionPlan {
            evict: evict_ids,
            conductance: cond,
        }
    }

    fn name(&self) -> &str {
        "SpectralEviction"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entries(n: usize, dim: usize) -> Vec<MemoryEntry> {
        (0..n)
            .map(|i| MemoryEntry {
                id: i,
                vector: vec![i as f32 / n as f32; dim],
                last_access: i as u64,
            })
            .collect()
    }

    #[test]
    fn random_eviction_correct_count() {
        let entries = make_entries(100, 8);
        let mut policy = RandomEviction::new(42);
        let plan = policy.plan_eviction(&entries, 70);
        assert_eq!(plan.evict.len(), 30);
    }

    #[test]
    fn lru_evicts_oldest() {
        let entries = make_entries(10, 4);
        let mut policy = LruEviction;
        let plan = policy.plan_eviction(&entries, 8);
        // IDs 0 and 1 have the smallest last_access
        let evict_set: std::collections::HashSet<usize> = plan.evict.into_iter().collect();
        assert!(evict_set.contains(&0));
        assert!(evict_set.contains(&1));
    }

    #[test]
    fn spectral_eviction_correct_count() {
        let entries = make_entries(20, 4);
        let mut policy = SpectralEviction::new(3, 15, 42);
        let plan = policy.plan_eviction(&entries, 14);
        assert_eq!(plan.evict.len(), 6);
    }
}
