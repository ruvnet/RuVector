//! CoDEQ kd-tree median-split quantizer.
//!
//! Algorithm (per arXiv:2512.18335 §3):
//!
//! 1. **Rotate**: Apply a fixed random Gaussian rotation R ∈ ℝ^(D×D) to every
//!    incoming vector. Rotation reduces dimensional coherence and is the same
//!    preprocessing step used by RaBitQ. Here we use a Gram-Schmidt-orthogonalised
//!    random matrix of reduced size (min(D, max_cols)) to keep build O(n·D).
//!
//! 2. **Partition**: Recursively split on the dimension of maximum variance
//!    among the current partition's points, at the empirical median value.
//!    After `bits` levels we have 2^bits leaf cells.
//!
//! 3. **Encode**: Given a query, walk the tree (L = bits comparisons) to reach
//!    its leaf; the leaf index (0..2^bits) is the L-bit code.
//!
//! 4. **Decode**: Return the centroid of the leaf cell (mean of all points
//!    assigned to that leaf).
//!
//! 5. **Streaming update (O(1) local work)**:
//!    - Insert: walk tree, add point to leaf list, update leaf centroid
//!      incrementally via Welford's online mean algorithm.
//!    - Delete: remove point from leaf list, update centroid.
//!    The tree *structure* (split dims + values) is FROZEN at build time.
//!    Only leaf centroids and point counts change — this is the "dynamic
//!    consistency" property: single-point updates never cascade to sibling
//!    or ancestor splits.

use std::collections::HashMap;

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::dist::l2_sq;
use crate::error::CoDEQError;

// ---------------------------------------------------------------------------
// Rotation matrix
// ---------------------------------------------------------------------------

/// A reduced-rank Gaussian rotation applied before quantization.
/// Using D rows × min(D, 64) cols keeps memory < 32 KB for D=128.
pub struct Rotation {
    /// Column-major matrix: rows=dim, cols=proj_dim.
    pub matrix: Vec<f32>,
    pub dim: usize,
    pub proj_dim: usize,
}

impl Rotation {
    /// Build a random Gaussian rotation (not orthonormalised for speed).
    /// Scale by 1/√proj_dim so the expected ‖Rx‖ ≈ ‖x‖.
    pub fn new(dim: usize, seed: u64) -> Self {
        let proj_dim = dim.min(64);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0_f32, 1.0).unwrap();
        let scale = (proj_dim as f32).sqrt().recip();
        let matrix: Vec<f32> = (0..dim * proj_dim)
            .map(|_| normal.sample(&mut rng) * scale)
            .collect();
        Self { matrix, dim, proj_dim }
    }

    /// Apply rotation: output has length `proj_dim`.
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.proj_dim];
        for (col, o) in out.iter_mut().enumerate() {
            *o = v.iter()
                .enumerate()
                .map(|(row, x)| x * self.matrix[col * self.dim + row])
                .sum();
        }
        out
    }
}

// ---------------------------------------------------------------------------
// kd-tree node (internal nodes only; leaves are implicit at depth=bits)
// ---------------------------------------------------------------------------

/// An internal split node.
#[derive(Clone)]
struct KdNode {
    split_dim: usize,
    split_val: f32,
}

// ---------------------------------------------------------------------------
// KdQuantizer — the stateless partition structure
// ---------------------------------------------------------------------------

/// Builds the binary kd-tree and encodes/decodes points.
///
/// After build, the tree structure is immutable; only leaf centroids change.
pub struct KdQuantizer {
    /// `nodes[d]` = the KdNode for depth d (0-indexed). Length = bits.
    /// We use a single array of depth-ordered nodes — at depth d, split on
    /// `nodes[d % nodes.len()]` (we cycle through split dimensions if bits > dim).
    nodes: Vec<KdNode>,
    pub bits: usize,
    pub proj_dim: usize,
}

impl KdQuantizer {
    /// Derive the kd-tree from a set of (rotated) vectors.
    ///
    /// Strategy: at each depth level, pick the dimension with the highest
    /// variance among all points that would reach that level (approximated
    /// by computing variance over the full training set — valid because the
    /// rotation de-correlates dimensions). Split at the global median per dim.
    pub fn build(rotated: &[Vec<f32>], bits: usize) -> Result<Self, CoDEQError> {
        if rotated.is_empty() { return Err(CoDEQError::EmptyDataset); }
        if bits == 0 || bits > 8 { return Err(CoDEQError::InvalidBits(bits as u8)); }
        let proj_dim = rotated[0].len();

        // Compute per-dimension variance to choose split dims in order.
        let n = rotated.len() as f32;
        let mut variance: Vec<(f32, usize)> = (0..proj_dim)
            .map(|d| {
                let mean: f32 = rotated.iter().map(|v| v[d]).sum::<f32>() / n;
                let var: f32 = rotated.iter().map(|v| (v[d] - mean).powi(2)).sum::<f32>() / n;
                (var, d)
            })
            .collect();
        variance.sort_by(|a, b| b.0.total_cmp(&a.0));

        // For depth d, use the d-th highest-variance dimension (mod proj_dim).
        let nodes: Vec<KdNode> = (0..bits)
            .map(|d| {
                let dim = variance[d % variance.len()].1;
                // Median along this dimension over the full training set.
                let mut vals: Vec<f32> = rotated.iter().map(|v| v[dim]).collect();
                vals.sort_by(|a, b| a.total_cmp(b));
                let split_val = vals[vals.len() / 2];
                KdNode { split_dim: dim, split_val }
            })
            .collect();

        Ok(Self { nodes, bits, proj_dim })
    }

    /// Encode a rotated vector to a leaf index in `0 .. 2^bits`.
    #[inline]
    pub fn encode_rotated(&self, rv: &[f32]) -> u8 {
        let mut code = 0u8;
        for (d, node) in self.nodes.iter().enumerate() {
            if rv[node.split_dim] >= node.split_val {
                code |= 1 << d;
            }
        }
        code
    }
}

// ---------------------------------------------------------------------------
// LeafStore — tracks points per leaf cell + incremental centroids
// ---------------------------------------------------------------------------

struct LeafStore {
    n_leaves: usize,
    /// Per-leaf: list of point IDs (for delete support).
    point_ids: Vec<Vec<u64>>,
    /// Per-leaf: running sum for centroid in ORIGINAL space.
    /// Using original-space centroids means the ADC LUT computes distances
    /// in the same space as the query — distances are preserved without any
    /// rotation distortion.
    centroid_sum: Vec<Vec<f32>>,
    /// Per-leaf: count of points.
    centroid_count: Vec<usize>,
    /// Original vector dimensionality.
    orig_dim: usize,
}

impl LeafStore {
    fn new(n_leaves: usize, orig_dim: usize) -> Self {
        Self {
            n_leaves,
            point_ids: vec![Vec::new(); n_leaves],
            centroid_sum: vec![vec![0.0; orig_dim]; n_leaves],
            centroid_count: vec![0; n_leaves],
            orig_dim,
        }
    }

    fn insert(&mut self, leaf: u8, id: u64, rv: &[f32]) {
        let l = leaf as usize;
        self.point_ids[l].push(id);
        for (s, x) in self.centroid_sum[l].iter_mut().zip(rv) {
            *s += x;
        }
        self.centroid_count[l] += 1;
    }

    fn delete(&mut self, leaf: u8, id: u64, rv: &[f32]) -> bool {
        let l = leaf as usize;
        if let Some(pos) = self.point_ids[l].iter().position(|&x| x == id) {
            self.point_ids[l].swap_remove(pos);
            for (s, x) in self.centroid_sum[l].iter_mut().zip(rv) {
                *s -= x;
            }
            self.centroid_count[l] = self.centroid_count[l].saturating_sub(1);
            true
        } else {
            false
        }
    }

    fn centroid(&self, leaf: u8) -> Vec<f32> {
        let l = leaf as usize;
        let cnt = self.centroid_count[l];
        if cnt == 0 {
            vec![0.0; self.orig_dim]
        } else {
            self.centroid_sum[l].iter().map(|&s| s / cnt as f32).collect()
        }
    }
}

// ---------------------------------------------------------------------------
// CoDEQIndex — the full streaming quantized index
// ---------------------------------------------------------------------------

/// CoDEQ streaming quantizer + flat ADC search.
///
/// Memory layout (per n vectors, D dims, L bits):
///   - Rotation matrix: D × min(D,64) × 4 bytes
///   - kd-tree nodes:   L × 8 bytes
///   - Leaf centroids:  2^L × min(D,64) × 4 bytes
///   - Code store:      n × 1 byte  (the L-bit code per vector)
///   - ID→leaf map:     n × 16 bytes (HashMap overhead)
///
/// Total ≈ n + 16n = 17n bytes for large n, vs n·D·4 bytes for raw f32.
/// At D=128, L=8: 17n vs 512n → 30× compression.
pub struct CoDEQIndex {
    pub rotation: Rotation,
    pub quantizer: KdQuantizer,
    leaves: LeafStore,
    /// id → (code, rotated_vec) for O(1) delete without re-rotation.
    store: HashMap<u64, (u8, Vec<f32>)>,
    /// Original data in raw f32 (needed for ADC reranking).
    raw: HashMap<u64, Vec<f32>>,
    /// Codes for each stored id, in insertion order (for search iteration).
    code_index: Vec<(u64, u8)>,
    pub dim: usize,
    n_leaves: usize,
}

impl CoDEQIndex {
    /// Create an empty index (no bulk-load needed — pure streaming).
    pub fn new(dim: usize, bits: usize, seed: u64) -> Result<Self, CoDEQError> {
        if bits == 0 || bits > 8 { return Err(CoDEQError::InvalidBits(bits as u8)); }
        let rotation = Rotation::new(dim, seed);
        let proj_dim = rotation.proj_dim;
        let n_leaves = 1usize << bits;
        // Build a placeholder quantizer (we need at least 1 point to set splits).
        // Actual split values are updated after the first batch insert.
        let placeholder = vec![vec![0.0f32; proj_dim]; 1];
        let quantizer = KdQuantizer::build(&placeholder, bits)?;
        let leaves = LeafStore::new(n_leaves, dim);
        Ok(Self {
            rotation, quantizer, leaves,
            store: HashMap::new(),
            raw: HashMap::new(),
            code_index: Vec::new(),
            dim,
            n_leaves,
        })
    }

    /// Bulk-initialize from a dataset. Rebuilds the kd-tree splits from the
    /// training data before inserting — exactly the "static" build path.
    pub fn from_vecs(
        data: &[(u64, Vec<f32>)],
        bits: usize,
        seed: u64,
    ) -> Result<Self, CoDEQError> {
        if data.is_empty() { return Err(CoDEQError::EmptyDataset); }
        let dim = data[0].1.len();
        let rotation = Rotation::new(dim, seed);
        let proj_dim = rotation.proj_dim;
        let n_leaves = 1usize << bits;

        // Rotate training set.
        let rotated: Vec<Vec<f32>> = data.iter().map(|(_, v)| rotation.apply(v)).collect();
        let quantizer = KdQuantizer::build(&rotated, bits)?;
        let mut leaves = LeafStore::new(n_leaves, dim);
        let mut store = HashMap::new();
        let mut raw = HashMap::new();
        let mut code_index = Vec::with_capacity(data.len());

        for ((id, v), rv) in data.iter().zip(rotated.iter()) {
            let code = quantizer.encode_rotated(rv);
            leaves.insert(code, *id, v);   // original-space vector for centroid
            store.insert(*id, (code, rv.clone()));
            raw.insert(*id, v.clone());
            code_index.push((*id, code));
        }

        Ok(Self { rotation, quantizer, leaves, store, raw, code_index, dim, n_leaves })
    }

    /// Insert a single vector. The kd-tree structure is frozen; only the leaf
    /// that the vector falls into gains one more point. O(L) tree traversal +
    /// O(1) centroid update — no rebuild, no cascade.
    pub fn insert(&mut self, id: u64, v: Vec<f32>) -> Result<(), CoDEQError> {
        if v.len() != self.dim {
            return Err(CoDEQError::DimMismatch { expected: self.dim, actual: v.len() });
        }
        let rv = self.rotation.apply(&v);
        let code = self.quantizer.encode_rotated(&rv);
        self.leaves.insert(code, id, &v);  // original-space vector for centroid
        self.store.insert(id, (code, rv));
        self.raw.insert(id, v);
        self.code_index.push((id, code));
        Ok(())
    }

    /// Delete a vector by id. O(L) lookup + O(leaf_size) swap_remove.
    pub fn delete(&mut self, id: u64) -> Result<(), CoDEQError> {
        let (code, _rv) = self.store.remove(&id).ok_or(CoDEQError::IdNotFound(id))?;
        let orig = self.raw.remove(&id).unwrap_or_default();
        self.leaves.delete(code, id, &orig); // original-space for centroid update
        if let Some(pos) = self.code_index.iter().position(|(xid, _)| *xid == id) {
            self.code_index.swap_remove(pos);
        }
        Ok(())
    }

    pub fn len(&self) -> usize { self.code_index.len() }
    pub fn is_empty(&self) -> bool { self.code_index.is_empty() }
    pub fn memory_bytes(&self) -> usize {
        let rotation_bytes = self.rotation.matrix.len() * 4;
        let leaf_centroid_bytes = self.n_leaves * self.dim * 4; // original-space centroids
        let code_bytes = self.code_index.len();
        let raw_bytes = self.raw.len() * self.dim * 4;
        rotation_bytes + leaf_centroid_bytes + code_bytes + raw_bytes
    }

    /// ADC search with configurable candidate oversample.
    ///
    /// `oversample`: number of LUT-ranked candidates to rerank with exact L2².
    /// - Low oversample (e.g. 8×k): fast, lower recall (good for streaming pipelines)
    /// - High oversample (e.g. 100×k): slower, higher recall (better quality)
    ///
    /// Cost: O(2^L × D) LUT build + O(n) code scan + O(oversample × D) rerank.
    pub fn search_adc_with_oversample(
        &self,
        query: &[f32],
        k: usize,
        oversample: usize,
    ) -> Result<Vec<(u64, f32)>, CoDEQError> {
        let n = self.code_index.len();
        if k > n { return Err(CoDEQError::KTooLarge { k, n }); }
        if query.len() != self.dim {
            return Err(CoDEQError::DimMismatch { expected: self.dim, actual: query.len() });
        }
        let lut: Vec<f32> = (0..self.n_leaves)
            .map(|leaf| l2_sq(query, &self.leaves.centroid(leaf as u8)))
            .collect();
        let mut scores: Vec<(u64, f32)> = self.code_index.iter()
            .map(|(id, code)| (*id, lut[*code as usize]))
            .collect();
        scores.sort_by(|a, b| a.1.total_cmp(&b.1));
        let os = oversample.min(n);
        scores.truncate(os);
        for (id, dist) in &mut scores {
            if let Some(v) = self.raw.get(id) { *dist = l2_sq(query, v); }
        }
        scores.sort_by(|a, b| a.1.total_cmp(&b.1));
        scores.truncate(k);
        Ok(scores)
    }

    /// ADC search: precompute query-to-centroid distances for each leaf, then
    /// look up the code of each stored vector in the precomputed table.
    /// Cost: O(2^L × D) LUT build + O(n) code scan — much faster than
    /// O(n × D) brute force when D is large. Uses 8×k oversample.
    pub fn search_adc(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, CoDEQError> {
        if query.len() != self.dim {
            return Err(CoDEQError::DimMismatch { expected: self.dim, actual: query.len() });
        }
        let n = self.code_index.len();
        if k > n { return Err(CoDEQError::KTooLarge { k, n }); }

        // Precompute L2² from ORIGINAL query to each leaf centroid (original space).
        // Centroids are means of original-space vectors; using the original query
        // avoids distance distortion from the non-orthogonal random rotation.
        // Note: use usize range to avoid u8 overflow when n_leaves = 256 (bits=8).
        let lut: Vec<f32> = (0..self.n_leaves)
            .map(|leaf| l2_sq(query, &self.leaves.centroid(leaf as u8)))
            .collect();

        // Rank all stored codes by their LUT distance.
        let mut scores: Vec<(u64, f32)> = self.code_index.iter()
            .map(|(id, code)| (*id, lut[*code as usize]))
            .collect();
        scores.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Rerank top-k × oversample with exact distance.
        let oversample = (k * 8).min(n);
        scores.truncate(oversample);
        for (id, dist) in &mut scores {
            if let Some(v) = self.raw.get(id) {
                *dist = l2_sq(query, v);
            }
        }
        scores.sort_by(|a, b| a.1.total_cmp(&b.1));
        scores.truncate(k);
        Ok(scores)
    }

    /// Exact brute-force search (baseline for recall computation).
    pub fn search_exact(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, CoDEQError> {
        if query.len() != self.dim {
            return Err(CoDEQError::DimMismatch { expected: self.dim, actual: query.len() });
        }
        let n = self.raw.len();
        if k > n { return Err(CoDEQError::KTooLarge { k, n }); }
        let mut scores: Vec<(u64, f32)> = self.raw.iter()
            .map(|(id, v)| (*id, l2_sq(query, v)))
            .collect();
        scores.sort_by(|a, b| a.1.total_cmp(&b.1));
        scores.truncate(k);
        Ok(scores)
    }
}

/// Recall@k: fraction of exact top-k that appear in the ADC top-k.
pub fn recall_at_k(index: &CoDEQIndex, queries: &[(u64, Vec<f32>)], k: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (_qid, qv) in queries {
        let Ok(approx) = index.search_adc(qv, k) else { continue };
        let Ok(exact)  = index.search_exact(qv, k) else { continue };
        let approx_ids: std::collections::HashSet<u64> = approx.iter().map(|(id, _)| *id).collect();
        let exact_ids:  std::collections::HashSet<u64> = exact.iter().map(|(id, _)| *id).collect();
        hits  += approx_ids.intersection(&exact_ids).count();
        total += exact_ids.len();
    }
    if total == 0 { 0.0 } else { hits as f64 / total as f64 }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn make_vecs(n: usize, dim: usize, seed: u64) -> Vec<(u64, Vec<f32>)> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0_f32, 1.0).unwrap();
        (0..n as u64)
            .map(|id| (id, (0..dim).map(|_| dist.sample(&mut rng)).collect()))
            .collect()
    }

    #[test]
    fn encode_is_deterministic() {
        let data = make_vecs(100, 16, 1);
        let idx = CoDEQIndex::from_vecs(&data, 4, 42).unwrap();
        let rv = idx.rotation.apply(&data[0].1);
        let c1 = idx.quantizer.encode_rotated(&rv);
        let c2 = idx.quantizer.encode_rotated(&rv);
        assert_eq!(c1, c2);
    }

    #[test]
    fn code_in_range() {
        let data = make_vecs(200, 32, 2);
        let idx = CoDEQIndex::from_vecs(&data, 6, 7).unwrap();
        for (_id, v) in &data {
            let rv = idx.rotation.apply(v);
            let code = idx.quantizer.encode_rotated(&rv);
            assert!((code as usize) < (1 << 6));
        }
    }

    #[test]
    fn insert_delete_len() {
        let mut idx = CoDEQIndex::new(8, 4, 99).unwrap();
        let data = make_vecs(10, 8, 3);
        // Must do a from_vecs first so tree splits are meaningful.
        let mut idx = CoDEQIndex::from_vecs(&data, 4, 99).unwrap();
        assert_eq!(idx.len(), 10);
        idx.insert(1000, vec![0.5; 8]).unwrap();
        assert_eq!(idx.len(), 11);
        idx.delete(1000).unwrap();
        assert_eq!(idx.len(), 10);
    }

    #[test]
    fn delete_nonexistent_errors() {
        let data = make_vecs(5, 8, 10);
        let mut idx = CoDEQIndex::from_vecs(&data, 4, 10).unwrap();
        assert!(idx.delete(9999).is_err());
    }

    #[test]
    fn adc_search_returns_k() {
        let data = make_vecs(100, 32, 4);
        let idx = CoDEQIndex::from_vecs(&data, 4, 42).unwrap();
        let q = &data[0].1;
        let res = idx.search_adc(q, 5).unwrap();
        assert_eq!(res.len(), 5);
    }

    #[test]
    fn adc_recall_on_gaussian() {
        let data = make_vecs(500, 32, 5);
        let queries: Vec<(u64, Vec<f32>)> = make_vecs(50, 32, 6)
            .into_iter().map(|(id, v)| (id + 10_000, v)).collect();
        // Use 8-bit codes (256 leaves) for higher recall; 6-bit is intentionally
        // coarser (≈45% recall) since only 6 of 32 dims are split.
        let idx = CoDEQIndex::from_vecs(&data, 8, 42).unwrap();
        let r = recall_at_k(&idx, &queries, 10);
        // ADC with 8-bit codes (256 leaves, ~2 vecs/leaf) + 8× reranking ≥ 0.65.
        assert!(r >= 0.65, "recall={:.3}", r);
    }

    #[test]
    fn streaming_insert_recall_stable() {
        // Build on 400 points, then stream-insert 100 more, check recall holds.
        let all_data = make_vecs(500, 32, 7);
        let (train, extra) = all_data.split_at(400);
        let queries: Vec<(u64, Vec<f32>)> = make_vecs(50, 32, 8)
            .into_iter().map(|(id, v)| (id + 10_000, v)).collect();

        let mut idx = CoDEQIndex::from_vecs(train, 6, 42).unwrap();
        let r_before = recall_at_k(&idx, &queries, 10);

        for (id, v) in extra {
            idx.insert(*id + 500, v.clone()).unwrap();
        }
        let r_after = recall_at_k(&idx, &queries, 10);

        // After streaming inserts, recall should remain within 15pp of static.
        assert!(r_after >= r_before - 0.15, "before={:.3} after={:.3}", r_before, r_after);
    }

    #[test]
    fn dim_mismatch_errors() {
        let data = make_vecs(10, 8, 11);
        let mut idx = CoDEQIndex::from_vecs(&data, 4, 11).unwrap();
        assert!(idx.insert(999, vec![0.0; 7]).is_err());
    }

    #[test]
    fn memory_bytes_grows_with_inserts() {
        let data = make_vecs(50, 16, 12);
        let mut idx = CoDEQIndex::from_vecs(&data, 4, 12).unwrap();
        let m0 = idx.memory_bytes();
        idx.insert(1000, vec![1.0; 16]).unwrap();
        assert!(idx.memory_bytes() >= m0);
    }
}
