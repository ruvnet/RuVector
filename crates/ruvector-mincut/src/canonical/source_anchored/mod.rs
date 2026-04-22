//! Source-anchored pseudo-deterministic canonical minimum cut (ADR-117).
//!
//! Given an undirected graph G=(V,E) with integer weights, a fixed source
//! vertex, and a total vertex ordering, returns the unique canonical minimum
//! cut defined by lexicographic tie-breaking:
//!
//!   minimize (λ(S), first_separable_vertex, |S|, π(S))
//!
//! This implements Tier 1 of ADR-117: exact engine using Stoer-Wagner for
//! the global min-cut value, then probing s-t cuts in vertex order to find
//! the first separable vertex.
//!
//! # References
//!
//! Yotam Kenneth-Mordoch, "Faster Pseudo-Deterministic Minimum Cut" (2026).

use super::FixedWeight;
use crate::graph::{DynamicGraph, VertexId, Weight};
use crate::time_compat::PortableTimestamp;

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// SourceAnchoredCut — the canonical cut artifact
// ---------------------------------------------------------------------------

/// A unique canonical minimum cut anchored at a fixed source vertex.
///
/// The cut is uniquely determined by the lexicographic tuple
/// `(lambda, first_separable_vertex, side_size, priority_sum)`.
#[derive(Debug, Clone)]
pub struct SourceAnchoredCut {
    /// The minimum cut value (fixed-point for determinism).
    pub lambda: FixedWeight,
    /// The designated source vertex.
    pub source_vertex: VertexId,
    /// The first vertex in the ordering that participates in a global min-cut.
    pub first_separable_vertex: VertexId,
    /// Sorted vertex IDs on the source side of the canonical cut.
    pub side_vertices: Vec<VertexId>,
    /// Number of vertices on the source side.
    pub side_size: usize,
    /// Sum of vertex priorities on the source side.
    pub priority_sum: u64,
    /// Edges crossing the cut as (u, v) pairs (sorted).
    pub cut_edges: Vec<(VertexId, VertexId)>,
    /// Stable SHA-256 hash of the canonical cut for RVF receipts.
    pub cut_hash: [u8; 32],
}

/// Configuration for the source-anchored canonical min-cut algorithm.
#[derive(Debug, Clone)]
pub struct SourceAnchoredConfig {
    /// The source vertex (default: smallest vertex ID in the graph).
    pub source: Option<VertexId>,
    /// Explicit vertex ordering. If `None`, uses sorted vertex IDs.
    pub vertex_order: Option<Vec<VertexId>>,
    /// Per-vertex priorities for secondary tie-breaking.
    /// If `None`, all priorities default to 1.
    pub vertex_priorities: Option<Vec<(VertexId, u64)>>,
}

impl Default for SourceAnchoredConfig {
    fn default() -> Self {
        Self {
            source: None,
            vertex_order: None,
            vertex_priorities: None,
        }
    }
}

/// Receipt for a source-anchored canonical cut, compatible with RVF witnesses.
#[derive(Debug, Clone)]
pub struct SourceAnchoredReceipt {
    /// Epoch (logical timestamp).
    pub epoch: u64,
    /// The canonical cut hash.
    pub cut_hash: [u8; 32],
    /// The cut value as fixed-point.
    pub lambda: FixedWeight,
    /// Source vertex.
    pub source_vertex: VertexId,
    /// First separable vertex.
    pub first_separable_vertex: VertexId,
    /// Side size.
    pub side_size: usize,
    /// Priority sum.
    pub priority_sum: u64,
    /// Wall-clock timestamp in nanoseconds.
    pub timestamp_ns: u64,
}

// ---------------------------------------------------------------------------
// Internal: adjacency snapshot for pure computation
// ---------------------------------------------------------------------------

/// A compact adjacency snapshot extracted from `DynamicGraph` for
/// deterministic computation without concurrent-map overhead.
struct AdjSnapshot {
    /// Number of vertices.
    n: usize,
    /// Sorted vertex IDs.
    vertices: Vec<VertexId>,
    /// vertex ID -> compact index.
    id_to_idx: HashMap<VertexId, usize>,
    /// Flat adjacency: adj[idx] = vec of (neighbor_idx, weight_fixed).
    adj: Vec<Vec<(usize, FixedWeight)>>,
}

impl AdjSnapshot {
    /// Build from a `DynamicGraph` reference.
    fn from_graph(graph: &DynamicGraph) -> Self {
        let mut vertices = graph.vertices();
        vertices.sort_unstable();

        let id_to_idx: HashMap<VertexId, usize> =
            vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();

        let n = vertices.len();
        let mut adj = vec![Vec::new(); n];

        for edge in graph.edges() {
            if let (Some(&ui), Some(&vi)) =
                (id_to_idx.get(&edge.source), id_to_idx.get(&edge.target))
            {
                let w = FixedWeight::from_f64(edge.weight);
                adj[ui].push((vi, w));
                adj[vi].push((ui, w));
            }
        }

        Self {
            n,
            vertices,
            id_to_idx,
            adj,
        }
    }

    /// Compute global min-cut value using Stoer-Wagner.
    ///
    /// Returns the min-cut value as `FixedWeight`, or `None` if the graph
    /// has fewer than 2 vertices.
    fn global_mincut_value(&self) -> Option<FixedWeight> {
        let n = self.n;
        if n < 2 {
            return None;
        }

        // Dense Stoer-Wagner on compact indices with flat weight matrix.
        let mut w = vec![FixedWeight::zero(); n * n];
        for i in 0..n {
            let row = i * n;
            for &(j, wt) in &self.adj[i] {
                w[row + j] = w[row + j].add(wt);
            }
        }

        let mut active: Vec<bool> = vec![true; n];
        let mut active_list: Vec<usize> = (0..n).collect();
        let mut n_active = n;

        let mut global_min = FixedWeight::from_f64(f64::MAX / 2.0);

        let mut key = vec![FixedWeight::zero(); n];
        let mut in_a = vec![false; n];

        for _phase in 0..(n - 1) {
            if n_active <= 1 {
                break;
            }

            // Reset only active vertices
            for k in 0..n_active {
                let j = active_list[k];
                in_a[j] = false;
                key[j] = FixedWeight::zero();
            }

            let first = active_list[0];
            in_a[first] = true;
            let first_row = first * n;
            for k in 0..n_active {
                let j = active_list[k];
                key[j] = w[first_row + j];
            }

            let mut prev = first;
            let mut last = first;

            for _step in 1..n_active {
                let mut best = usize::MAX;
                let mut best_key = FixedWeight::zero();
                let mut found = false;

                for k in 0..n_active {
                    let j = active_list[k];
                    if !in_a[j] && (!found || key[j] > best_key) {
                        best_key = key[j];
                        best = j;
                        found = true;
                    }
                }

                if !found {
                    break;
                }

                in_a[best] = true;
                prev = last;
                last = best;

                let best_row = best * n;
                for k in 0..n_active {
                    let j = active_list[k];
                    if !in_a[j] {
                        key[j] = key[j].add(w[best_row + j]);
                    }
                }
            }

            let cut_value = key[last];
            if cut_value < global_min {
                global_min = cut_value;
            }

            // Merge last into prev
            let prev_row = prev * n;
            let last_row = last * n;
            for k in 0..n_active {
                let j = active_list[k];
                if j != last {
                    w[prev_row + j] = w[prev_row + j].add(w[last_row + j]);
                    w[j * n + prev] = w[j * n + prev].add(w[j * n + last]);
                }
            }

            active[last] = false;
            // Swap-remove for O(1) instead of retain's O(n)
            if let Some(pos) = active_list[..n_active].iter().position(|&x| x == last) {
                active_list.swap(pos, n_active - 1);
            }
            n_active -= 1;
        }

        Some(global_min)
    }

    /// Compute a minimum s-t cut using Dinic's max-flow.
    ///
    /// Returns `(cut_value, source_side_mask)` where `source_side_mask[i]`
    /// is `true` if compact vertex `i` is on the source side.
    ///
    /// Uses a flat capacity array for cache locality.
    fn min_st_cut(
        &self,
        s_idx: usize,
        t_idx: usize,
        _priority: &[u64],
    ) -> (FixedWeight, Vec<bool>) {
        let n = self.n;
        debug_assert!(s_idx < n && t_idx < n && s_idx != t_idx);

        // Build directed capacity for max-flow (flat array for cache locality)
        let mut cap = vec![FixedWeight::zero(); n * n];
        for i in 0..n {
            let row = i * n;
            for &(j, w) in &self.adj[i] {
                cap[row + j] = cap[row + j].add(w);
            }
        }

        // Dinic's algorithm for max-flow
        let flow = dinic_maxflow(&mut cap, s_idx, t_idx, n);

        // Find source side from residual graph via BFS from s
        let mut source_side = vec![false; n];
        let mut queue = VecDeque::with_capacity(n);
        queue.push_back(s_idx);
        source_side[s_idx] = true;

        while let Some(u) = queue.pop_front() {
            let row = u * n;
            for v in 0..n {
                if !source_side[v] && cap[row + v].raw() > 0 {
                    source_side[v] = true;
                    queue.push_back(v);
                }
            }
        }

        (flow, source_side)
    }
}

// ---------------------------------------------------------------------------
// Dinic's max-flow
// ---------------------------------------------------------------------------

/// Compute max-flow using Dinic's algorithm on a flat capacity array.
///
/// `cap` is a flat n*n array where `cap[u*n + v]` is the capacity from u to v.
/// Modifies `cap` in-place to represent the residual graph.
/// Returns the total flow value as `FixedWeight`.
fn dinic_maxflow(cap: &mut [FixedWeight], s: usize, t: usize, n: usize) -> FixedWeight {
    let mut total_flow = FixedWeight::zero();
    let mut level = vec![-1i32; n];
    let mut queue = VecDeque::with_capacity(n);
    let mut iter = vec![0usize; n];
    let inf = FixedWeight::from_f64(f64::MAX / 2.0);

    loop {
        // BFS to build level graph (reuse allocations)
        for l in level.iter_mut() {
            *l = -1;
        }
        level[s] = 0;
        queue.clear();
        queue.push_back(s);

        while let Some(u) = queue.pop_front() {
            let row = u * n;
            for v in 0..n {
                if level[v] == -1 && cap[row + v].raw() > 0 {
                    level[v] = level[u] + 1;
                    queue.push_back(v);
                }
            }
        }

        if level[t] == -1 {
            break; // No augmenting path
        }

        // DFS to find blocking flows (reuse iter allocation)
        for it in iter.iter_mut() {
            *it = 0;
        }
        loop {
            let pushed = dinic_dfs(cap, &level, &mut iter, s, t, n, inf);
            if pushed.raw() == 0 {
                break;
            }
            total_flow = total_flow.add(pushed);
        }
    }

    total_flow
}

/// DFS for Dinic's blocking flow on a flat capacity array.
fn dinic_dfs(
    cap: &mut [FixedWeight],
    level: &[i32],
    iter: &mut [usize],
    u: usize,
    t: usize,
    n: usize,
    pushed: FixedWeight,
) -> FixedWeight {
    if u == t {
        return pushed;
    }

    let u_row = u * n;
    while iter[u] < n {
        let v = iter[u];
        if level[v] == level[u] + 1 && cap[u_row + v].raw() > 0 {
            let cap_uv = cap[u_row + v];
            let bottleneck = if pushed < cap_uv { pushed } else { cap_uv };
            let d = dinic_dfs(cap, level, iter, v, t, n, bottleneck);
            if d.raw() > 0 {
                cap[u_row + v] = cap[u_row + v].sub(d);
                cap[v * n + u] = cap[v * n + u].add(d);
                return d;
            }
        }
        iter[u] += 1;
    }

    FixedWeight::zero()
}

// ---------------------------------------------------------------------------
// SHA-256 (pure, no_std compatible)
// ---------------------------------------------------------------------------

/// Compute a stable SHA-256 hash of the canonical cut parameters.
///
/// Input layout (little-endian):
///   lambda (8 bytes) ‖ source (8 bytes) ‖ first_v (8 bytes) ‖
///   priority_sum (8 bytes) ‖ side_size (8 bytes) ‖
///   side_vertices (8 bytes each)
///
/// Uses a minimal embedded SHA-256 to avoid external dependency churn.
fn stable_cut_hash(
    lambda: FixedWeight,
    source: VertexId,
    first_v: VertexId,
    side: &[VertexId],
    priority_sum: u64,
) -> [u8; 32] {
    let mut data = Vec::with_capacity(40 + side.len() * 8);
    data.extend_from_slice(&lambda.raw().to_le_bytes());
    data.extend_from_slice(&source.to_le_bytes());
    data.extend_from_slice(&first_v.to_le_bytes());
    data.extend_from_slice(&priority_sum.to_le_bytes());
    data.extend_from_slice(&(side.len() as u64).to_le_bytes());
    for &v in side {
        data.extend_from_slice(&v.to_le_bytes());
    }
    sha256(&data)
}

/// Minimal SHA-256 implementation (FIPS 180-4).
///
/// This is intentionally self-contained to avoid dependency on external
/// crates for a security-critical hash used in witness receipts.
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Pre-processing: padding
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit block
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, &val) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the source-anchored canonical minimum cut of a `DynamicGraph`.
///
/// This is the main entry point for ADR-117 Tier 1.
///
/// # Algorithm
///
/// 1. Snapshot the graph into a compact adjacency structure.
/// 2. Compute the global min-cut value λ* via Stoer-Wagner.
/// 3. Scan vertices in the fixed ordering. For each vertex v (≠ source),
///    compute an exact min s-v cut. If its value equals λ*, v is the
///    first separable vertex. The source side of that cut is the
///    canonical side.
/// 4. Hash the result for RVF witness receipts.
///
/// # Panics
///
/// Does not panic. Returns `None` for trivial or disconnected graphs.
pub fn canonical_mincut(
    graph: &DynamicGraph,
    config: &SourceAnchoredConfig,
) -> Option<SourceAnchoredCut> {
    let snap = AdjSnapshot::from_graph(graph);

    if snap.n < 2 {
        return None;
    }

    // Determine source vertex
    let source = config.source.unwrap_or(snap.vertices[0]);
    let source_idx = *snap.id_to_idx.get(&source)?;

    // Determine vertex ordering
    let order: Vec<VertexId> = match &config.vertex_order {
        Some(o) => o.clone(),
        None => snap.vertices.clone(),
    };

    // Build priority map
    let mut priority = vec![1u64; snap.n];
    if let Some(ref prio) = config.vertex_priorities {
        for &(v, p) in prio {
            if let Some(&idx) = snap.id_to_idx.get(&v) {
                priority[idx] = p;
            }
        }
    }

    // Step 1: Global min-cut value
    let lambda_star = snap.global_mincut_value()?;

    // Disconnected graph: lambda = 0
    if lambda_star.raw() == 0 {
        return None;
    }

    // Step 2: Find first separable vertex
    for &v in &order {
        if v == source {
            continue;
        }

        let v_idx = match snap.id_to_idx.get(&v) {
            Some(&idx) => idx,
            None => continue,
        };

        let (cut_value, source_side_mask) = snap.min_st_cut(source_idx, v_idx, &priority);

        // Compare with tolerance: use raw fixed-point comparison
        if cut_value != lambda_star {
            continue;
        }

        // Found the first separable vertex.
        // Collect the source side vertices.
        let mut side: Vec<VertexId> = source_side_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &in_s)| if in_s { Some(snap.vertices[i]) } else { None })
            .collect();
        side.sort_unstable();

        let side_size = side.len();
        let priority_sum: u64 = source_side_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &in_s)| if in_s { Some(priority[i]) } else { None })
            .sum();

        // Collect cut edges
        let mut cut_edges: Vec<(VertexId, VertexId)> = Vec::new();
        for i in 0..snap.n {
            if !source_side_mask[i] {
                continue;
            }
            for &(j, _w) in &snap.adj[i] {
                if !source_side_mask[j] && snap.vertices[i] < snap.vertices[j] {
                    cut_edges.push((snap.vertices[i], snap.vertices[j]));
                }
            }
        }
        cut_edges.sort_unstable();

        let cut_hash = stable_cut_hash(lambda_star, source, v, &side, priority_sum);

        return Some(SourceAnchoredCut {
            lambda: lambda_star,
            source_vertex: source,
            first_separable_vertex: v,
            side_vertices: side,
            side_size,
            priority_sum,
            cut_edges,
            cut_hash,
        });
    }

    None
}

/// Generate a receipt from a canonical cut result.
pub fn make_receipt(cut: &SourceAnchoredCut, epoch: u64) -> SourceAnchoredReceipt {
    let ts = PortableTimestamp::now().as_secs() * 1_000_000_000;
    SourceAnchoredReceipt {
        epoch,
        cut_hash: cut.cut_hash,
        lambda: cut.lambda,
        source_vertex: cut.source_vertex,
        first_separable_vertex: cut.first_separable_vertex,
        side_size: cut.side_size,
        priority_sum: cut.priority_sum,
        timestamp_ns: ts,
    }
}

/// Verify that two receipts agree on the canonical cut identity.
pub fn receipts_agree(a: &SourceAnchoredReceipt, b: &SourceAnchoredReceipt) -> bool {
    a.cut_hash == b.cut_hash
        && a.lambda == b.lambda
        && a.source_vertex == b.source_vertex
        && a.first_separable_vertex == b.first_separable_vertex
        && a.side_size == b.side_size
        && a.priority_sum == b.priority_sum
}

// ---------------------------------------------------------------------------
// SourceAnchoredMinCut — stateful wrapper with caching
// ---------------------------------------------------------------------------

/// Stateful wrapper that caches the canonical cut and invalidates on mutation.
///
/// Mirrors the `CanonicalMinCutImpl` pattern but uses source-anchored
/// tie-breaking instead of cactus enumeration.
pub struct SourceAnchoredMinCut {
    /// Underlying dynamic min-cut engine.
    inner: crate::algorithm::DynamicMinCut,
    /// Configuration for canonical cut computation.
    config: SourceAnchoredConfig,
    /// Cached result.
    cached: Option<SourceAnchoredCut>,
    /// Logical epoch counter.
    epoch: u64,
    /// Whether the cache is stale.
    dirty: bool,
}

impl SourceAnchoredMinCut {
    /// Create a new instance with default config.
    pub fn new() -> Self {
        Self {
            inner: crate::algorithm::DynamicMinCut::new(crate::MinCutConfig::default()),
            config: SourceAnchoredConfig::default(),
            cached: None,
            epoch: 0,
            dirty: true,
        }
    }

    /// Create with explicit configuration.
    pub fn with_config(config: SourceAnchoredConfig) -> Self {
        Self {
            inner: crate::algorithm::DynamicMinCut::new(crate::MinCutConfig::default()),
            config,
            cached: None,
            epoch: 0,
            dirty: true,
        }
    }

    /// Create from edges.
    pub fn with_edges(
        edges: Vec<(VertexId, VertexId, Weight)>,
        config: SourceAnchoredConfig,
    ) -> crate::Result<Self> {
        let inner = crate::MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build()?;
        Ok(Self {
            inner,
            config,
            cached: None,
            epoch: 0,
            dirty: true,
        })
    }

    /// Compute (or return cached) the canonical cut.
    pub fn canonical_cut(&mut self) -> Option<SourceAnchoredCut> {
        if !self.dirty {
            if let Some(ref c) = self.cached {
                return Some(c.clone());
            }
        }

        let graph = self.inner.graph();
        let g = graph.read();
        let result = canonical_mincut(&g, &self.config);
        drop(g);

        self.cached = result.clone();
        self.dirty = false;
        result
    }

    /// Generate a witness receipt.
    pub fn receipt(&mut self) -> Option<SourceAnchoredReceipt> {
        let cut = self.canonical_cut()?;
        Some(make_receipt(&cut, self.epoch))
    }

    /// Insert an edge and invalidate the cache.
    pub fn insert_edge(&mut self, u: VertexId, v: VertexId, weight: Weight) -> crate::Result<f64> {
        let val = self.inner.insert_edge(u, v, weight)?;
        self.epoch += 1;
        self.dirty = true;
        self.cached = None;
        Ok(val)
    }

    /// Delete an edge and invalidate the cache.
    pub fn delete_edge(&mut self, u: VertexId, v: VertexId) -> crate::Result<f64> {
        let val = self.inner.delete_edge(u, v)?;
        self.epoch += 1;
        self.dirty = true;
        self.cached = None;
        Ok(val)
    }

    /// Get the current min-cut value.
    pub fn min_cut_value(&self) -> f64 {
        self.inner.min_cut_value()
    }

    /// Number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.inner.num_vertices()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Whether the graph is connected.
    pub fn is_connected(&self) -> bool {
        self.inner.is_connected()
    }

    /// Get the current epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Get the config.
    pub fn config(&self) -> &SourceAnchoredConfig {
        &self.config
    }
}

impl Default for SourceAnchoredMinCut {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WASM-compatible exports (no wasm-bindgen, just #[no_mangle] extern "C")
// ---------------------------------------------------------------------------

/// Compact representation for FFI / WASM interop.
///
/// All fields are fixed-size, no heap pointers cross the boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CanonicalMinCutResult {
    /// Min-cut value (fixed-point raw).
    pub lambda_raw: u64,
    /// Source vertex ID.
    pub source_vertex: u64,
    /// First separable vertex ID.
    pub first_separable_vertex: u64,
    /// Number of vertices on the source side.
    pub side_size: u32,
    /// Priority sum.
    pub priority_sum: u64,
    /// Number of cut edges.
    pub cut_edge_count: u32,
    /// SHA-256 hash of the canonical cut.
    pub cut_hash: [u8; 32],
}

impl From<&SourceAnchoredCut> for CanonicalMinCutResult {
    fn from(cut: &SourceAnchoredCut) -> Self {
        Self {
            lambda_raw: cut.lambda.raw(),
            source_vertex: cut.source_vertex,
            first_separable_vertex: cut.first_separable_vertex,
            side_size: cut.side_size as u32,
            priority_sum: cut.priority_sum,
            cut_edge_count: cut.cut_edges.len() as u32,
            cut_hash: cut.cut_hash,
        }
    }
}

#[cfg(test)]
mod tests;
