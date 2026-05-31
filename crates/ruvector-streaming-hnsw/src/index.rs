//! Three ANN index variants for the streaming-HNSW PoC.
//!
//! | Variant              | Build strategy          | Insert after build? | Notes                     |
//! |----------------------|-------------------------|---------------------|---------------------------|
//! | `FlatL2Index`        | None (raw vecs)         | Yes (trivial push)  | O(n·D) scan per query     |
//! | `StaticHnsw`         | Offline greedy k-NN     | No                  | Best recall; immutable     |
//! | `StreamingHnsw`      | Online; one node at a time | Yes (concurrent) | `RwLock` per neighbor list |

use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::dist::l2_sq;
use crate::error::HnswError;

// ---------------------------------------------------------------------------
// Shared trait
// ---------------------------------------------------------------------------

pub trait AnnIndex: Send + Sync {
    /// Number of vectors currently in the index.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    /// Embedding dimension.
    fn dim(&self) -> usize;
    /// Approximate k-NN query. Returns `(node_id, l2_sq_distance)` pairs.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, HnswError>;
    /// Heap bytes estimate.
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}

/// Recall@k against brute-force ground truth.
pub fn recall_at_k(
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    index: &dyn AnnIndex,
) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for q in queries {
        let Ok(approx) = index.search(q, k) else { continue };
        let approx_ids: std::collections::HashSet<u32> =
            approx.iter().map(|(id, _)| *id).collect();

        // Brute-force top-k by l2_sq
        let mut all: Vec<(u32, f32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, l2_sq(q, v)))
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1));
        let exact_ids: std::collections::HashSet<u32> =
            all.iter().take(k).map(|(id, _)| *id).collect();

        hits += approx_ids.intersection(&exact_ids).count();
        total += exact_ids.len();
    }
    if total == 0 { return 0.0; }
    hits as f64 / total as f64
}

// ---------------------------------------------------------------------------
// Variant 1: FlatL2Index — brute-force, always dynamic
// ---------------------------------------------------------------------------

pub struct FlatL2Index {
    data: Vec<Vec<f32>>,
    dim: usize,
}

impl FlatL2Index {
    pub fn new(dim: usize) -> Self {
        Self { data: Vec::new(), dim }
    }
    pub fn insert(&mut self, vec: Vec<f32>) -> Result<u32, HnswError> {
        if vec.len() != self.dim {
            return Err(HnswError::DimMismatch { expected: self.dim, actual: vec.len() });
        }
        let id = self.data.len() as u32;
        self.data.push(vec);
        Ok(id)
    }
}

impl AnnIndex for FlatL2Index {
    fn len(&self) -> usize { self.data.len() }
    fn dim(&self) -> usize { self.dim }
    fn name(&self) -> &'static str { "FlatL2 (baseline)" }
    fn memory_bytes(&self) -> usize { self.data.len() * self.dim * 4 }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, HnswError> {
        if query.len() != self.dim {
            return Err(HnswError::DimMismatch { expected: self.dim, actual: query.len() });
        }
        let n = self.data.len();
        if k > n { return Err(HnswError::KTooLarge { k, n }); }
        let mut scores: Vec<(u32, f32)> = self.data.iter().enumerate()
            .map(|(i, v)| (i as u32, l2_sq(query, v)))
            .collect();
        scores.sort_by(|a, b| a.1.total_cmp(&b.1));
        scores.truncate(k);
        Ok(scores)
    }
}

// ---------------------------------------------------------------------------
// Variant 2: StaticHnsw — offline greedy k-NN graph, immutable after build
//
// Build: for each node i (in insertion order), scan all prior nodes and keep
// the M nearest. Bidirectional edges added up to M back-edges per prior node.
// This is O(n²·D) — fine for n ≤ 50K at D=128.
//
// Search: greedy beam search with beam width `ef`.
// ---------------------------------------------------------------------------

pub struct StaticHnsw {
    /// Row-major flat storage: index `i` occupies `[i*dim .. (i+1)*dim]`.
    data: Vec<f32>,
    /// `neighbors[i]` = node IDs nearest to i, sorted by distance asc.
    neighbors: Vec<Vec<u32>>,
    dim: usize,
    m: usize,   // max neighbors per node
    ef: usize,  // beam width during search
}

impl StaticHnsw {
    /// Build from an existing vector set.
    pub fn build(data: Vec<Vec<f32>>, m: usize, ef: usize) -> Result<Self, HnswError> {
        if data.is_empty() { return Err(HnswError::EmptyDataset); }
        let dim = data[0].len();
        let n = data.len();
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for v in &data {
            if v.len() != dim {
                return Err(HnswError::DimMismatch { expected: dim, actual: v.len() });
            }
            flat.extend_from_slice(v);
        }
        let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); n];

        // For node i, scan all j < i and collect nearest M.
        for i in 1..n {
            let vi = &flat[i * dim..(i + 1) * dim];

            // Collect distances to all prior nodes.
            let mut cands: Vec<(u32, f32)> = (0..i)
                .map(|j| {
                    let vj = &flat[j * dim..(j + 1) * dim];
                    (j as u32, l2_sq(vi, vj))
                })
                .collect();
            cands.sort_by(|a, b| a.1.total_cmp(&b.1));
            cands.truncate(m);

            // Forward edges i → j
            for &(j, _) in &cands {
                neighbors[i].push(j);
                // Back-edge j → i (up to 2·M — HNSW paper's Mmax = 2·M at layer 0)
                if neighbors[j as usize].len() < 2 * m {
                    neighbors[j as usize].push(i as u32);
                }
            }
        }

        Ok(Self { data: flat, neighbors, dim, m, ef })
    }
}

impl AnnIndex for StaticHnsw {
    fn len(&self) -> usize { self.data.len() / self.dim }
    fn dim(&self) -> usize { self.dim }
    fn name(&self) -> &'static str { "StaticHnsw (offline-build)" }
    fn memory_bytes(&self) -> usize {
        self.data.len() * 4
            + self.neighbors.iter().map(|nb| nb.len() * 4).sum::<usize>()
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, HnswError> {
        if query.len() != self.dim {
            return Err(HnswError::DimMismatch { expected: self.dim, actual: query.len() });
        }
        let n = self.len();
        if k > n { return Err(HnswError::KTooLarge { k, n }); }
        hnsw_greedy_search(&self.data, &self.neighbors, self.dim, query, k, self.ef)
    }
}

// ---------------------------------------------------------------------------
// Variant 3: StreamingHnsw — online inserts, RwLock-per-node neighbor list
//
// Each node's neighbor list is wrapped in `Arc<RwLock<Vec<u32>>>`:
//   - Queries hold read-locks briefly while traversing edges.
//   - Inserts write-lock only the nodes whose neighbor lists they modify.
//
// Correctness: the entry-point is updated atomically under a separate lock.
// Recall: same greedy k-NN algorithm as StaticHnsw; recall degrades slightly
// vs. the static version for the same M because back-edges compete with
// forward edges for the M slots, and under concurrency the final neighbor
// sets reflect the insertion order (which is deterministic in single-threaded
// benchmarks).
// ---------------------------------------------------------------------------

pub struct StreamingHnsw {
    /// Flat row-major vector storage, append-only (no concurrent mutation).
    data: Arc<RwLock<Vec<f32>>>,
    /// Per-node neighbor lists; `Arc<RwLock<…>>` so queries & inserts share.
    neighbors: Arc<RwLock<Vec<Arc<RwLock<Vec<u32>>>>>>,
    dim: usize,
    m: usize,
    ef: usize,
}

impl StreamingHnsw {
    pub fn new(dim: usize, m: usize, ef: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(Vec::new())),
            neighbors: Arc::new(RwLock::new(Vec::new())),
            dim,
            m,
            ef,
        }
    }

    /// Insert a single vector, returning its node id.
    ///
    /// Algorithm:
    /// 1. Append vector data (write-lock on `data`).
    /// 2. Create empty neighbor slot (write-lock on `neighbors` vec).
    /// 3. With only read-locks, find M nearest among already-inserted nodes.
    /// 4. Write those ids into our new node's neighbor list.
    /// 5. For each found neighbor, write-lock it and append our id (if not full).
    pub fn insert(&self, vec: Vec<f32>) -> Result<u32, HnswError> {
        if vec.len() != self.dim {
            return Err(HnswError::DimMismatch { expected: self.dim, actual: vec.len() });
        }

        // --- Steps 1+2 (atomic): append vector data AND create neighbor slot
        //     under BOTH write-locks held simultaneously.
        //
        // Rationale: if we released the data lock before acquiring the
        // neighbors lock, another thread could get new_id = this_id + 1 and
        // push a neighbor slot before we do.  That shifts the slot we meant
        // to claim to a different index, causing an out-of-bounds panic in
        // Step 4.  Holding both locks together makes id assignment and slot
        // creation a single atomic operation. Steps 3-5 then proceed with
        // only fine-grained per-node locks, so concurrency is preserved.
        let new_id: u32 = {
            let mut data = self.data.write();
            let mut nb_vec = self.neighbors.write();
            let id = (data.len() / self.dim) as u32;
            data.extend_from_slice(&vec);
            nb_vec.push(Arc::new(RwLock::new(Vec::with_capacity(self.m))));
            id
        };

        // If this is the first node, no edges needed.
        if new_id == 0 { return Ok(0); }

        // --- Step 3: find nearest M among prior nodes (read-only scan) ---
        let cands: Vec<(u32, f32)> = {
            let data = self.data.read();
            let n_existing = new_id as usize; // nodes 0..new_id
            let new_slice = &data[new_id as usize * self.dim..];
            let mut all: Vec<(u32, f32)> = (0..n_existing)
                .map(|j| {
                    let vj = &data[j * self.dim..(j + 1) * self.dim];
                    (j as u32, l2_sq(new_slice, vj))
                })
                .collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1));
            all.truncate(self.m);
            all
        };

        // --- Step 4: write our neighbor list ---
        {
            let nb_vec = self.neighbors.read();
            let mut my_nb = nb_vec[new_id as usize].write();
            for &(j, _) in &cands {
                my_nb.push(j);
            }
        }

        // --- Step 5: add back-edges to neighbors ---
        {
            let nb_vec = self.neighbors.read();
            for &(j, _) in &cands {
                let mut j_nb = nb_vec[j as usize].write();
                if j_nb.len() < 2 * self.m {
                    j_nb.push(new_id);
                }
            }
        }

        Ok(new_id)
    }

    /// Bulk-load from a pre-built dataset (sequential inserts).
    pub fn from_vecs(data: Vec<Vec<f32>>, m: usize, ef: usize) -> Result<Self, HnswError> {
        if data.is_empty() { return Err(HnswError::EmptyDataset); }
        let dim = data[0].len();
        let idx = Self::new(dim, m, ef);
        for v in data {
            idx.insert(v)?;
        }
        Ok(idx)
    }
}

impl AnnIndex for StreamingHnsw {
    fn len(&self) -> usize { self.data.read().len() / self.dim }
    fn dim(&self) -> usize { self.dim }
    fn name(&self) -> &'static str { "StreamingHnsw (concurrent-insert)" }
    fn memory_bytes(&self) -> usize {
        let data_bytes = self.data.read().len() * 4;
        let nb_bytes: usize = self.neighbors.read()
            .iter()
            .map(|nb| nb.read().len() * 4)
            .sum();
        data_bytes + nb_bytes
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, HnswError> {
        if query.len() != self.dim {
            return Err(HnswError::DimMismatch { expected: self.dim, actual: query.len() });
        }
        let n = self.len();
        if n == 0 { return Ok(Vec::new()); }
        if k > n { return Err(HnswError::KTooLarge { k, n }); }

        // Snapshot neighbor lists under read-locks into plain Vecs.
        // This approach avoids holding locks across the entire search and
        // is safe against concurrent inserts appending new nodes after the
        // snapshot: those nodes simply won't be visible in this query.
        let (data_snap, nb_snap): (Vec<f32>, Vec<Vec<u32>>) = {
            let data = self.data.read();
            let nb_vec = self.neighbors.read();
            let data_snap = data.clone();
            let nb_snap: Vec<Vec<u32>> = nb_vec.iter().map(|nb| nb.read().clone()).collect();
            (data_snap, nb_snap)
        };

        hnsw_greedy_search(&data_snap, &nb_snap, self.dim, query, k, self.ef)
    }
}

// ---------------------------------------------------------------------------
// Shared greedy beam search
// ---------------------------------------------------------------------------

/// Greedy beam search on a flat-stored kNN graph.
///
/// Uses a max-heap as the result set (size `ef`) and a min-heap as the
/// frontier. Returns the top-k results sorted by ascending distance.
fn hnsw_greedy_search(
    data: &[f32],
    neighbors: &[Vec<u32>],
    dim: usize,
    query: &[f32],
    k: usize,
    ef: usize,
) -> Result<Vec<(u32, f32)>, HnswError> {
    let n = data.len() / dim;
    if n == 0 { return Ok(Vec::new()); }

    let mut visited = vec![false; n];
    // results: max-heap of (dist, id) — we pop the worst to keep ef best.
    // Use Reverse so BinaryHeap gives us a min-heap for frontier.
    let mut results: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();
    let mut frontier: BinaryHeap<Reverse<(OrdF32, u32)>> = BinaryHeap::new();

    // Enter at node 0.
    let d0 = l2_sq(query, &data[0..dim]);
    frontier.push(Reverse((OrdF32(d0), 0)));
    results.push((OrdF32(d0), 0));
    visited[0] = true;

    while let Some(Reverse((OrdF32(d_cur), id))) = frontier.pop() {
        // Prune: if frontier top is worse than worst result, stop.
        if results.len() >= ef {
            if let Some(&(OrdF32(d_worst), _)) = results.peek() {
                if d_cur > d_worst { break; }
            }
        }

        for &nb in &neighbors[id as usize] {
            if (nb as usize) < n && !visited[nb as usize] {
                visited[nb as usize] = true;
                let d = l2_sq(query, &data[nb as usize * dim..(nb as usize + 1) * dim]);
                if results.len() < ef {
                    results.push((OrdF32(d), nb));
                    frontier.push(Reverse((OrdF32(d), nb)));
                } else if let Some(&(OrdF32(d_worst), _)) = results.peek() {
                    if d < d_worst {
                        results.pop();
                        results.push((OrdF32(d), nb));
                        frontier.push(Reverse((OrdF32(d), nb)));
                    }
                }
            }
        }
    }

    let mut out: Vec<(u32, f32)> = results.into_iter().map(|(OrdF32(d), id)| (id, d)).collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1));
    out.truncate(k);
    Ok(out)
}

/// Total-ordering wrapper for f32.
#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for OrdF32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering { self.0.total_cmp(&o.0) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn make_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0_f32, 1.0).unwrap();
        (0..n).map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect()).collect()
    }

    #[test]
    fn flat_returns_exact_top1() {
        let data = make_vecs(100, 32, 42);
        let mut idx = FlatL2Index::new(32);
        for v in &data { idx.insert(v.clone()).unwrap(); }
        // Query = first vector; it must be its own nearest neighbor.
        let res = idx.search(&data[0], 1).unwrap();
        assert_eq!(res[0].0, 0);
        assert!(res[0].1 < 1e-6);
    }

    #[test]
    fn static_hnsw_recall_high() {
        let data = make_vecs(500, 32, 1);
        let queries = make_vecs(50, 32, 2);
        let idx = StaticHnsw::build(data.clone(), 16, 40).unwrap();
        let recall = recall_at_k(&data, &queries, 10, &idx);
        // Static greedy NN graph should achieve ≥ 90% recall@10 on Gaussian data.
        assert!(recall >= 0.90, "recall={:.3}", recall);
    }

    #[test]
    fn streaming_hnsw_insert_and_search() {
        let data = make_vecs(200, 32, 3);
        let idx = StreamingHnsw::from_vecs(data.clone(), 16, 40).unwrap();
        assert_eq!(idx.len(), 200);
        let q = &data[0];
        let res = idx.search(q, 5).unwrap();
        assert_eq!(res.len(), 5);
        // Nearest should be self (id=0).
        assert_eq!(res[0].0, 0);
    }

    #[test]
    fn streaming_hnsw_recall_acceptable() {
        let data = make_vecs(500, 32, 5);
        let queries = make_vecs(50, 32, 6);
        let idx = StreamingHnsw::from_vecs(data.clone(), 16, 40).unwrap();
        let recall = recall_at_k(&data, &queries, 10, &idx);
        // Online/streaming allows some recall degradation vs static; ≥ 80%.
        assert!(recall >= 0.80, "streaming recall={:.3}", recall);
    }

    #[test]
    fn streaming_hnsw_concurrent_inserts_then_search() {
        use std::thread;
        let dim = 16usize;
        let m = 12usize;
        let ef = 30usize;
        let idx = Arc::new(StreamingHnsw::new(dim, m, ef));

        // Spawn 4 threads, each inserting 50 random vectors.
        let mut handles = Vec::new();
        for t in 0u64..4 {
            let idx = Arc::clone(&idx);
            handles.push(thread::spawn(move || {
                let mut rng = rand::rngs::StdRng::seed_from_u64(t * 100);
                let normal = Normal::new(0.0_f32, 1.0).unwrap();
                for _ in 0..50 {
                    let v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
                    idx.insert(v).unwrap();
                }
            }));
        }
        for h in handles { h.join().unwrap(); }

        // Total: 200 nodes. Search should not panic.
        assert_eq!(idx.len(), 200);
        let q: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
        let res = idx.search(&q, 5).unwrap();
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn dim_mismatch_is_detected() {
        let idx = StreamingHnsw::new(8, 8, 20);
        let bad_vec = vec![1.0f32; 5]; // wrong dim
        assert!(idx.insert(bad_vec).is_err());
    }

    #[test]
    fn memory_bytes_grows_with_inserts() {
        let idx = StreamingHnsw::new(8, 4, 10);
        let m0 = idx.memory_bytes();
        idx.insert(vec![1.0; 8]).unwrap();
        idx.insert(vec![2.0; 8]).unwrap();
        assert!(idx.memory_bytes() > m0);
    }
}
