//! DiskANN/Vamana-style motif index.
//!
//! Self-contained in-memory Vamana graph ANN index for motif
//! embeddings. Follows Subramanya et al., "DiskANN: Fast Accurate
//! Billion-point Nearest Neighbor Search on a Single Node"
//! (NeurIPS 2019) — greedy beam search + α-robust pruning, two build
//! passes (α = 1.0, then α = params.alpha).
//!
//! The workspace already ships `crates/ruvector-diskann`, but that
//! crate is tuned for SSD-resident billion-scale indexes: mmap,
//! rayon, bincode, and nondeterministic `thread_rng()` graph init.
//! This in-example module trades that scale for zero new heavy deps,
//! bit-deterministic graph construction (seeded xoroshiro PRNG), and
//! ≤ 500 LOC with no unsafe.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Type alias for motif embedding vectors.
pub type EmbeddingF32 = Vec<f32>;

/// Vamana construction / query parameters.
#[derive(Clone, Debug)]
pub struct VamanaParams {
    /// Max out-degree R of the Vamana graph.
    pub max_degree: usize,
    /// Build-time beam width L_build (>= `max_degree`).
    pub build_beam: usize,
    /// Query-time beam width L_search (>= k at call sites).
    pub search_beam: usize,
    /// α pruning slack (>= 1.0). Larger keeps more long-range edges.
    pub alpha: f32,
    /// PRNG seed for graph init and build-order shuffle.
    pub seed: u64,
}

impl Default for VamanaParams {
    fn default() -> Self {
        Self {
            max_degree: 32,
            build_beam: 64,
            search_beam: 64,
            alpha: 1.2,
            seed: 0xD15C_A44_5EE_DDEEF_u64,
        }
    }
}

/// In-memory DiskANN/Vamana motif index with deterministic
/// construction. Owns its corpus by value.
pub struct DiskAnnMotifIndex {
    corpus: Vec<EmbeddingF32>,
    dim: usize,
    neighbors: Vec<Vec<u32>>,
    entry: u32,
    search_beam: usize,
}

impl DiskAnnMotifIndex {
    /// Build a Vamana graph over `corpus`.
    ///
    /// Panics if `corpus` is empty or contains a vector with a
    /// dimension different from the first entry.
    pub fn new(corpus: Vec<EmbeddingF32>, params: VamanaParams) -> Self {
        assert!(!corpus.is_empty(), "DiskAnnMotifIndex: empty corpus");
        let dim = corpus[0].len();
        for v in &corpus {
            assert_eq!(v.len(), dim, "DiskAnnMotifIndex: mixed vector dims");
        }
        let n = corpus.len();
        let max_degree = params.max_degree.min(n.saturating_sub(1)).max(1);
        let build_beam = params.build_beam.max(max_degree);
        let alpha = params.alpha.max(1.0);

        let entry = medoid(&corpus);
        let mut neighbors = init_random_graph(n, max_degree, params.seed);

        // Pass 1 — α = 1.0 (shorter edges).
        let order1 = det_permutation(n, params.seed ^ 0xA110C_u64);
        build_pass(
            &corpus,
            &mut neighbors,
            entry,
            build_beam,
            max_degree,
            1.0,
            &order1,
        );
        // Pass 2 — α = params.alpha (longer-range diversification).
        if (alpha - 1.0).abs() > f32::EPSILON {
            let order2 = det_permutation(n, params.seed ^ 0xA110D_u64);
            build_pass(
                &corpus,
                &mut neighbors,
                entry,
                build_beam,
                max_degree,
                alpha,
                &order2,
            );
        }

        Self {
            corpus,
            dim,
            neighbors,
            entry,
            search_beam: params.search_beam.max(max_degree),
        }
    }

    /// Number of vectors indexed.
    pub fn len(&self) -> usize {
        self.corpus.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.corpus.is_empty()
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Top-k nearest neighbours of `q` by Euclidean distance.
    ///
    /// Results are sorted by distance ascending; ties break on
    /// ascending id for determinism.
    pub fn query(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        assert_eq!(q.len(), self.dim, "DiskAnnMotifIndex::query: dim mismatch");
        if k == 0 || self.corpus.is_empty() {
            return Vec::new();
        }
        let beam = self.search_beam.max(k);
        let beamset = greedy_search(&self.corpus, &self.neighbors, self.entry, q, beam);
        let mut out: Vec<(usize, f32)> = beamset
            .into_iter()
            .map(|(id, d2)| (id as usize, d2.sqrt()))
            .collect();
        out.sort_by(cmp_result);
        out.truncate(k);
        out
    }

    /// Class-label precision@k over `queries`.
    ///
    /// Each query is `(embedding, class_label)`. For every query we
    /// retrieve the k nearest neighbours **excluding the query itself
    /// if it is in the corpus** (matched by bit-identical vector), and
    /// count how many share the query's class label. Returns the
    /// fraction of all retrieved slots whose label matches.
    pub fn precision_at_k(&self, queries: &[(EmbeddingF32, usize)], k: usize) -> f32 {
        if queries.is_empty() || k == 0 {
            return 0.0;
        }
        // id -> class map, populated from queries. Corpus vectors not
        // present in `queries` get tag usize::MAX which matches nothing.
        let mut id_class: Vec<usize> = vec![usize::MAX; self.corpus.len()];
        for (qv, cls) in queries {
            if let Some(id) = find_vec(&self.corpus, qv) {
                id_class[id] = *cls;
            }
        }
        let mut matched = 0_usize;
        let mut total = 0_usize;
        let retrieve = k + 1; // pull one extra to drop self-hit.
        for (qv, cls) in queries {
            let self_id = find_vec(&self.corpus, qv);
            let nn = self.query(qv, retrieve);
            let mut taken = 0_usize;
            for (id, _) in nn {
                if Some(id) == self_id {
                    continue;
                }
                if taken == k {
                    break;
                }
                if id_class[id] == *cls {
                    matched += 1;
                }
                total += 1;
                taken += 1;
            }
        }
        if total == 0 {
            0.0
        } else {
            matched as f32 / total as f32
        }
    }
}

// -----------------------------------------------------------------
// Vamana build
// -----------------------------------------------------------------

fn build_pass(
    corpus: &[EmbeddingF32],
    neighbors: &mut [Vec<u32>],
    entry: u32,
    build_beam: usize,
    max_degree: usize,
    alpha: f32,
    order: &[u32],
) {
    for &node in order {
        let q = &corpus[node as usize];
        let visited = greedy_search(corpus, neighbors, entry, q, build_beam);
        let mut cand: Vec<u32> = visited
            .iter()
            .filter_map(|(id, _)| if *id == node { None } else { Some(*id) })
            .collect();
        // Also include current neighbours so pass 2 refines pass 1.
        for &n in &neighbors[node as usize] {
            if n != node && !cand.contains(&n) {
                cand.push(n);
            }
        }
        let pruned = robust_prune(corpus, node, &cand, alpha, max_degree);
        neighbors[node as usize] = pruned.clone();
        // Bidirectional insertion with re-prune on overflow.
        for &nid in &pruned {
            let slot = &mut neighbors[nid as usize];
            if !slot.contains(&node) {
                if slot.len() < max_degree {
                    slot.push(node);
                } else {
                    let mut combined = slot.clone();
                    combined.push(node);
                    neighbors[nid as usize] =
                        robust_prune(corpus, nid, &combined, alpha, max_degree);
                }
            }
        }
    }
}

/// Greedy beam search from `entry` toward `q`. Returns the final
/// closed beam as (id, L2²-distance) pairs in no particular order.
fn greedy_search(
    corpus: &[EmbeddingF32],
    neighbors: &[Vec<u32>],
    entry: u32,
    q: &[f32],
    beam: usize,
) -> Vec<(u32, f32)> {
    let n = corpus.len();
    let mut visited = vec![false; n];
    // frontier: open, min-heap on distance ascending.
    // best:     closed beam, max-heap on distance so we cheaply evict.
    let mut frontier = BinaryHeap::<Min>::new();
    let mut best = BinaryHeap::<Max>::new();
    let entry_u = entry as usize;
    visited[entry_u] = true;
    let d0 = l2_sq(q, &corpus[entry_u]);
    frontier.push(Min { id: entry, d: d0 });
    best.push(Max { id: entry, d: d0 });
    while let Some(cur) = frontier.pop() {
        if best.len() >= beam {
            if let Some(worst) = best.peek() {
                if cur.d > worst.d {
                    break;
                }
            }
        }
        for &nb in &neighbors[cur.id as usize] {
            let nu = nb as usize;
            if nu >= n || visited[nu] {
                continue;
            }
            visited[nu] = true;
            let nd = l2_sq(q, &corpus[nu]);
            let dominated = best.len() >= beam
                && best.peek().map(|w| nd >= w.d).unwrap_or(false);
            if !dominated {
                frontier.push(Min { id: nb, d: nd });
                best.push(Max { id: nb, d: nd });
                if best.len() > beam {
                    best.pop();
                }
            }
        }
    }
    best.into_iter().map(|c| (c.id, c.d)).collect()
}

/// Robust α-prune. Keeps at most `max_degree` neighbours from the
/// distance-sorted candidate list, skipping any candidate dominated
/// by an already-selected neighbour under the α test.
fn robust_prune(
    corpus: &[EmbeddingF32],
    node: u32,
    candidates: &[u32],
    alpha: f32,
    max_degree: usize,
) -> Vec<u32> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let node_v = &corpus[node as usize];
    let mut sorted: Vec<(u32, f32)> = candidates
        .iter()
        .filter(|&&c| c != node)
        .map(|&c| (c, l2_sq(node_v, &corpus[c as usize])))
        .collect();
    sorted.sort_by(cmp_id_asc);
    let mut out: Vec<u32> = Vec::with_capacity(max_degree);
    for (cand_id, cand_d) in &sorted {
        if out.len() >= max_degree {
            break;
        }
        let dominated = out.iter().any(|&sel| {
            let inter = l2_sq(&corpus[sel as usize], &corpus[*cand_id as usize]);
            alpha * inter <= *cand_d
        });
        if !dominated {
            out.push(*cand_id);
        }
    }
    out
}

/// Deterministic medoid: corpus point with smallest summed L2² to
/// every other corpus point. O(n²). Ties break on smaller id.
fn medoid(corpus: &[EmbeddingF32]) -> u32 {
    let n = corpus.len();
    let mut best: u32 = 0;
    let mut best_cost = f32::INFINITY;
    for i in 0..n {
        let mut s = 0.0_f32;
        for j in 0..n {
            if i == j {
                continue;
            }
            s += l2_sq(&corpus[i], &corpus[j]);
        }
        if s < best_cost {
            best_cost = s;
            best = i as u32;
        }
    }
    best
}

fn init_random_graph(n: usize, max_degree: usize, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = Rng::new(seed);
    let mut neighbors = vec![Vec::with_capacity(max_degree); n];
    if n <= 1 {
        return neighbors;
    }
    let degree = max_degree.min(n - 1);
    for i in 0..n {
        let slot = &mut neighbors[i];
        let mut attempts = 0_usize;
        while slot.len() < degree && attempts < degree * 6 {
            let j = (rng.next_u64() % n as u64) as u32;
            if j != i as u32 && !slot.contains(&j) {
                slot.push(j);
            }
            attempts += 1;
        }
    }
    neighbors
}

fn det_permutation(n: usize, seed: u64) -> Vec<u32> {
    let mut v: Vec<u32> = (0..n as u32).collect();
    let mut rng = Rng::new(seed);
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

fn find_vec(corpus: &[EmbeddingF32], needle: &[f32]) -> Option<usize> {
    for (i, v) in corpus.iter().enumerate() {
        if v.len() == needle.len() && v.iter().zip(needle).all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return Some(i);
        }
    }
    None
}

// -----------------------------------------------------------------
// Math / PRNG / heap helpers
// -----------------------------------------------------------------

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    let n = a.len().min(b.len());
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Xoroshiro128++ — tiny, deterministic, no deps.
struct Rng {
    s0: u64,
    s1: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let s0 = splitmix(&mut z);
        let s1 = splitmix(&mut z);
        let s0 = if s0 == 0 { 0xD1B5_4A32_D192_ED03 } else { s0 };
        let s1 = if s1 == 0 { 0x6A09_E667_BB67_AE85 } else { s1 };
        Self { s0, s1 }
    }

    fn next_u64(&mut self) -> u64 {
        let r = self.s0.wrapping_add(self.s1).rotate_left(17).wrapping_add(self.s0);
        let s1 = self.s1 ^ self.s0;
        self.s0 = self.s0.rotate_left(49) ^ s1 ^ (s1 << 21);
        self.s1 = s1.rotate_left(28);
        r
    }
}

fn splitmix(z: &mut u64) -> u64 {
    *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *z;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Min-heap (smaller d pops first) when BinaryHeap normally pops max.
struct Min {
    id: u32,
    d: f32,
}
impl PartialEq for Min {
    fn eq(&self, o: &Self) -> bool {
        self.d == o.d && self.id == o.id
    }
}
impl Eq for Min {}
impl PartialOrd for Min {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Min {
    fn cmp(&self, o: &Self) -> Ordering {
        o.d.partial_cmp(&self.d)
            .unwrap_or(Ordering::Equal)
            .then_with(|| o.id.cmp(&self.id))
    }
}

/// Max-heap (larger d pops first) — used to evict the worst beam.
struct Max {
    id: u32,
    d: f32,
}
impl PartialEq for Max {
    fn eq(&self, o: &Self) -> bool {
        self.d == o.d && self.id == o.id
    }
}
impl Eq for Max {}
impl PartialOrd for Max {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Max {
    fn cmp(&self, o: &Self) -> Ordering {
        self.d
            .partial_cmp(&o.d)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&o.id))
    }
}

fn cmp_result(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    a.1.partial_cmp(&b.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}

fn cmp_id_asc(a: &(u32, f32), b: &(u32, f32)) -> Ordering {
    a.1.partial_cmp(&b.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}

