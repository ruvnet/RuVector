//! Approximate Min-Cut for All Cut Sizes
//!
//! Implementation based on "Approximate Min-Cut in All Cut Sizes"
//! (SODA 2025, arXiv:2412.15069).
//!
//! # Key Innovation
//!
//! Uses spectral sparsification with edge sampling to achieve (1+ε)-approximate
//! minimum cuts for ANY cut size, not just small cuts.
//!
//! # Time Complexity
//!
//! - Preprocessing: O(m log² n / ε²)
//! - Query: O(n polylog n / ε²)
//!
//! # Algorithm Overview
//!
//! 1. Compute effective resistances for all edges
//! 2. Sample edges with probability proportional to resistance × weight
//! 3. Build sparsifier with O(n log n / ε²) edges
//! 4. Run exact min-cut on sparsifier (feasible due to small size)

use crate::graph::VertexId;
use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for approximate min-cut
#[derive(Debug, Clone)]
pub struct ApproxMinCutConfig {
    /// Approximation parameter (0 < ε ≤ 1)
    pub epsilon: f64,
    /// Number of sparsifier samples (higher = more accurate)
    pub num_samples: usize,
    /// Seed for reproducibility
    pub seed: u64,
}

impl Default for ApproxMinCutConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            num_samples: 3,
            seed: 42,
        }
    }
}

/// Edge with weight for the sparsifier
#[derive(Debug, Clone, Copy, PartialEq)]
struct WeightedEdge {
    u: VertexId,
    v: VertexId,
    weight: f64,
}

impl WeightedEdge {
    fn new(u: VertexId, v: VertexId, weight: f64) -> Self {
        let (u, v) = if u < v { (u, v) } else { (v, u) };
        Self { u, v, weight }
    }

    fn endpoints(&self) -> (VertexId, VertexId) {
        (self.u, self.v)
    }
}

/// Spectral sparsifier for approximate min-cut
#[derive(Debug)]
struct SpectralSparsifier {
    /// Edges in the sparsifier
    edges: Vec<WeightedEdge>,
    /// Vertex set
    vertices: HashSet<VertexId>,
    /// Adjacency map
    adj: HashMap<VertexId, Vec<(VertexId, f64)>>,
}

impl SpectralSparsifier {
    fn new() -> Self {
        Self {
            edges: Vec::new(),
            vertices: HashSet::new(),
            adj: HashMap::new(),
        }
    }

    fn add_edge(&mut self, u: VertexId, v: VertexId, weight: f64) {
        self.vertices.insert(u);
        self.vertices.insert(v);
        self.edges.push(WeightedEdge::new(u, v, weight));

        self.adj.entry(u).or_default().push((v, weight));
        self.adj.entry(v).or_default().push((u, weight));
    }

    fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

/// Approximate minimum cut for all cut sizes
///
/// Achieves (1+ε)-approximation for any cut size using spectral sparsification.
///
/// # Example
///
/// ```ignore
/// use ruvector_mincut::algorithm::approximate::ApproxMinCut;
///
/// let mut approx = ApproxMinCut::new(ApproxMinCutConfig::default());
/// approx.insert_edge(0, 1, 1.0);
/// approx.insert_edge(1, 2, 1.0);
/// approx.insert_edge(2, 0, 1.0);
///
/// let result = approx.min_cut();
/// assert!(result.value >= 2.0 * 0.9); // (1-ε) lower bound
/// ```
#[derive(Debug)]
pub struct ApproxMinCut {
    /// All edges in the graph
    edges: Vec<WeightedEdge>,
    /// Vertex set
    vertices: HashSet<VertexId>,
    /// Adjacency list
    adj: HashMap<VertexId, Vec<(VertexId, f64)>>,
    /// Effective resistances (computed lazily)
    resistances: HashMap<(VertexId, VertexId), f64>,
    /// Configuration
    config: ApproxMinCutConfig,
    /// Current minimum cut value
    cached_min_cut: Option<f64>,
    /// Partition achieving `cached_min_cut`, kept in lockstep with it.
    cached_partition: Option<(Vec<VertexId>, Vec<VertexId>)>,
    /// Statistics
    stats: ApproxMinCutStats,
}

/// Statistics for approximate min-cut
#[derive(Debug, Clone, Default)]
pub struct ApproxMinCutStats {
    /// Total insertions
    pub insertions: u64,
    /// Total deletions
    pub deletions: u64,
    /// Total queries
    pub queries: u64,
    /// Number of sparsifier rebuilds
    pub rebuilds: u64,
}

/// Result of approximate min-cut query
#[derive(Debug, Clone)]
pub struct ApproxMinCutResult {
    /// Approximate minimum cut value
    pub value: f64,
    /// Lower bound (value / (1+ε))
    pub lower_bound: f64,
    /// Upper bound (value * (1+ε))
    pub upper_bound: f64,
    /// Partition achieving the cut
    pub partition: Option<(Vec<VertexId>, Vec<VertexId>)>,
    /// Approximation ratio used
    pub epsilon: f64,
}

impl ApproxMinCut {
    /// Create new approximate min-cut structure
    pub fn new(config: ApproxMinCutConfig) -> Self {
        Self {
            edges: Vec::new(),
            vertices: HashSet::new(),
            adj: HashMap::new(),
            resistances: HashMap::new(),
            config,
            cached_min_cut: None,
            cached_partition: None,
            stats: ApproxMinCutStats::default(),
        }
    }

    /// Create with default configuration
    pub fn with_epsilon(epsilon: f64) -> Self {
        Self::new(ApproxMinCutConfig {
            epsilon,
            ..Default::default()
        })
    }

    /// Insert an edge
    pub fn insert_edge(&mut self, u: VertexId, v: VertexId, weight: f64) {
        self.stats.insertions += 1;
        self.cached_min_cut = None; // Invalidate cache
        self.cached_partition = None;

        let edge = WeightedEdge::new(u, v, weight);
        self.edges.push(edge);
        self.vertices.insert(u);
        self.vertices.insert(v);

        self.adj.entry(u).or_default().push((v, weight));
        self.adj.entry(v).or_default().push((u, weight));

        // Clear resistance cache
        self.resistances.clear();
    }

    /// Delete an edge
    pub fn delete_edge(&mut self, u: VertexId, v: VertexId) {
        self.stats.deletions += 1;
        self.cached_min_cut = None;
        self.cached_partition = None;

        let key = if u < v { (u, v) } else { (v, u) };
        self.edges.retain(|e| e.endpoints() != key);

        // Update adjacency
        if let Some(neighbors) = self.adj.get_mut(&u) {
            neighbors.retain(|(neighbor, _)| *neighbor != v);
        }
        if let Some(neighbors) = self.adj.get_mut(&v) {
            neighbors.retain(|(neighbor, _)| *neighbor != u);
        }

        self.resistances.clear();
    }

    /// Query approximate minimum cut
    pub fn min_cut(&mut self) -> ApproxMinCutResult {
        self.stats.queries += 1;

        if self.edges.is_empty() {
            return ApproxMinCutResult {
                value: f64::INFINITY,
                lower_bound: f64::INFINITY,
                upper_bound: f64::INFINITY,
                partition: None,
                epsilon: self.config.epsilon,
            };
        }

        // Use cached value if available. `cached_partition` is invalidated in
        // lockstep with `cached_min_cut` (see `insert_edge`/`delete_edge`), so
        // it is always the partition that achieves this exact cached value.
        if let Some(cached) = self.cached_min_cut {
            let lower = cached / (1.0 + self.config.epsilon);
            let upper = cached * (1.0 + self.config.epsilon);
            return ApproxMinCutResult {
                value: cached,
                lower_bound: lower,
                upper_bound: upper,
                partition: self.cached_partition.clone(),
                epsilon: self.config.epsilon,
            };
        }

        // Build sparsifier and compute min-cut. This also populates
        // `cached_partition` with the partition that actually achieves the
        // returned value (see `compute_min_cut_via_sparsifier`).
        let value = self.compute_min_cut_via_sparsifier();
        self.cached_min_cut = Some(value);

        let lower = value / (1.0 + self.config.epsilon);
        let upper = value * (1.0 + self.config.epsilon);

        ApproxMinCutResult {
            value,
            lower_bound: lower,
            upper_bound: upper,
            partition: self.cached_partition.clone(),
            epsilon: self.config.epsilon,
        }
    }

    /// Get minimum cut value only
    pub fn min_cut_value(&mut self) -> f64 {
        self.min_cut().value
    }

    /// Check if graph is connected
    pub fn is_connected(&self) -> bool {
        if self.vertices.is_empty() {
            return true;
        }

        let start = *self.vertices.iter().next().unwrap();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.adj.get(&current) {
                for &(neighbor, _) in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        visited.len() == self.vertices.len()
    }

    /// Get vertex count
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &ApproxMinCutStats {
        &self.stats
    }

    /// Compute min-cut via spectral sparsification. Always leaves
    /// `self.cached_partition` holding a partition that actually achieves
    /// the returned value (fixes a prior bug where the public `partition`
    /// field was an arbitrary BFS bisection unrelated to the computed cut —
    /// see `docs/research/nightly/2026-10-29-approx-mincut-partition-fix`).
    fn compute_min_cut_via_sparsifier(&mut self) -> f64 {
        self.stats.rebuilds += 1;

        if !self.is_connected() {
            self.cached_partition = Some(self.disconnected_component_partition());
            return 0.0;
        }

        // For small graphs, compute exactly.
        if self.edges.len() <= 50 {
            let (value, partition) = self.stoer_wagner_with_partition(&self.adj, &self.vertices);
            self.cached_partition = Some(partition);
            return value;
        }

        // Compute effective resistances
        self.compute_effective_resistances();

        // Build sparsifier(s) and take the median by value. The matching
        // partition is recovered from the specific sparsifier that produced
        // the median estimate, not recomputed on the full graph: the whole
        // point of sparsification is that the full graph may be too large
        // for a second exact pass.
        let mut estimates: Vec<(f64, SpectralSparsifier)> =
            Vec::with_capacity(self.config.num_samples);
        for i in 0..self.config.num_samples {
            let sparsifier = self.build_sparsifier(self.config.seed + i as u64);
            let cut = self.compute_sparsifier_min_cut(&sparsifier);
            estimates.push((cut, sparsifier));
        }

        estimates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let median = &estimates[estimates.len() / 2];
        let (_, partition) = self.stoer_wagner_with_partition(&median.1.adj, &median.1.vertices);
        self.cached_partition = Some(partition);
        median.0
    }

    /// Partition for a disconnected graph: one connected component versus
    /// everything else. This genuinely achieves the zero cut value
    /// `compute_min_cut_via_sparsifier` returns for the disconnected case.
    fn disconnected_component_partition(&self) -> (Vec<VertexId>, Vec<VertexId>) {
        let Some(&start) = self.vertices.iter().next() else {
            return (Vec::new(), Vec::new());
        };
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);
        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.adj.get(&current) {
                for &(neighbor, _) in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        let rest: Vec<VertexId> = self
            .vertices
            .iter()
            .filter(|v| !visited.contains(v))
            .copied()
            .collect();
        (visited.into_iter().collect(), rest)
    }

    /// Compute effective resistances using BFS approximation
    fn compute_effective_resistances(&mut self) {
        if !self.resistances.is_empty() {
            return;
        }

        // Approximate effective resistance using shortest path heuristic
        // True spectral computation would use Laplacian solver
        for edge in &self.edges {
            let (u, v) = edge.endpoints();

            // Approximate R_eff(u,v) ≈ 1/(min_degree) for connected pairs
            let deg_u = self.adj.get(&u).map_or(1, |n| n.len().max(1));
            let deg_v = self.adj.get(&v).map_or(1, |n| n.len().max(1));
            let approx_resistance = 2.0 / (deg_u + deg_v) as f64;

            self.resistances.insert((u, v), approx_resistance);
        }
    }

    /// Build spectral sparsifier by sampling edges
    fn build_sparsifier(&self, seed: u64) -> SpectralSparsifier {
        let mut sparsifier = SpectralSparsifier::new();
        let n = self.vertices.len();
        let epsilon = self.config.epsilon;

        // Target size: O(n log n / ε²)
        let target_size = ((n as f64) * (n as f64).ln() / (epsilon * epsilon)).ceil() as usize;
        let target_size = target_size.min(self.edges.len()).max(n);

        // Simple deterministic hash for reproducibility
        let hash = |i: usize| -> u64 {
            let mut h = seed.wrapping_add(i as u64);
            h = h.wrapping_mul(0x517cc1b727220a95);
            h ^= h >> 32;
            h
        };

        // Sample each edge with probability proportional to weight × resistance
        let total_weight: f64 = self.edges.iter().map(|e| e.weight).sum();

        for (i, edge) in self.edges.iter().enumerate() {
            let (u, v) = edge.endpoints();
            let resistance = self.resistances.get(&(u, v)).copied().unwrap_or(1.0);

            // Sampling probability
            let prob = (edge.weight * resistance * target_size as f64 / total_weight).min(1.0);

            // Use hash for deterministic sampling
            let rand_val = (hash(i) as f64) / (u64::MAX as f64);

            if rand_val < prob {
                // Rescale weight to preserve expected cut values
                let new_weight = edge.weight / prob;
                sparsifier.add_edge(u, v, new_weight);
            }
        }

        // Ensure connectivity by adding spanning tree
        self.ensure_sparsifier_connectivity(&mut sparsifier);

        sparsifier
    }

    /// Ensure sparsifier is connected
    fn ensure_sparsifier_connectivity(&self, sparsifier: &mut SpectralSparsifier) {
        // BFS to find spanning tree edges
        if self.vertices.is_empty() {
            return;
        }

        let start = *self.vertices.iter().next().unwrap();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.adj.get(&current) {
                for &(neighbor, weight) in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);

                        // Add tree edge to sparsifier if not present
                        if !sparsifier.vertices.contains(&neighbor) {
                            sparsifier.add_edge(current, neighbor, weight);
                        }
                    }
                }
            }
        }
    }

    /// Compute min-cut on sparsifier
    fn compute_sparsifier_min_cut(&self, sparsifier: &SpectralSparsifier) -> f64 {
        if sparsifier.vertex_count() <= 1 {
            return f64::INFINITY;
        }

        // Use Stoer-Wagner on the small sparsifier
        self.stoer_wagner(&sparsifier.adj, &sparsifier.vertices)
    }

    /// Optimized Stoer-Wagner minimum cut algorithm
    /// Uses early termination and degree-based lower bound
    fn stoer_wagner(
        &self,
        adj: &HashMap<VertexId, Vec<(VertexId, f64)>>,
        vertices: &HashSet<VertexId>,
    ) -> f64 {
        if vertices.len() <= 1 {
            return f64::INFINITY;
        }

        // Quick lower bound: minimum weighted degree
        let min_degree: f64 = adj
            .iter()
            .filter(|(v, _)| vertices.contains(v))
            .map(|(_, neighbors)| neighbors.iter().map(|(_, w)| *w).sum::<f64>())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(f64::INFINITY);

        // For very small graphs, use simpler algorithm
        if vertices.len() <= 3 {
            return min_degree;
        }

        // Build adjacency matrix
        let verts: Vec<VertexId> = vertices.iter().copied().collect();
        let n = verts.len();
        let vert_to_idx: HashMap<_, _> = verts.iter().enumerate().map(|(i, &v)| (v, i)).collect();

        let mut weights = vec![vec![0.0; n]; n];
        for (&v, neighbors) in adj {
            if let Some(&i) = vert_to_idx.get(&v) {
                for &(neighbor, weight) in neighbors {
                    if let Some(&j) = vert_to_idx.get(&neighbor) {
                        weights[i][j] += weight;
                    }
                }
            }
        }

        // Stoer-Wagner iterations with optimizations
        let mut min_cut = f64::INFINITY;
        let mut active: Vec<bool> = vec![true; n];

        for phase in 0..n - 1 {
            // Early termination if we found a zero cut
            if min_cut == 0.0 {
                break;
            }

            // Maximum adjacency search with optimized tracking
            let mut in_a = vec![false; n];
            let mut cut_of_phase = vec![0.0; n];

            // Find first active vertex
            let first = match (0..n).find(|&i| active[i]) {
                Some(f) => f,
                None => break,
            };
            in_a[first] = true;

            let mut last = first;
            let mut before_last = first;

            let active_count = active.iter().filter(|&&a| a).count();
            for _ in 1..active_count {
                // Update cut values only for neighbors of last
                for j in 0..n {
                    if active[j] && !in_a[j] && weights[last][j] > 0.0 {
                        cut_of_phase[j] += weights[last][j];
                    }
                }

                // Find vertex with maximum cut value
                before_last = last;
                let mut max_val = f64::NEG_INFINITY;
                for j in 0..n {
                    if active[j] && !in_a[j] && cut_of_phase[j] > max_val {
                        max_val = cut_of_phase[j];
                        last = j;
                    }
                }

                if max_val == f64::NEG_INFINITY {
                    break;
                }
                in_a[last] = true;
            }

            // Update minimum cut
            if cut_of_phase[last] > 0.0 || phase == 0 {
                min_cut = min_cut.min(cut_of_phase[last]);
            }

            // Merge last two vertices
            active[last] = false;
            for j in 0..n {
                weights[before_last][j] += weights[last][j];
                weights[j][before_last] += weights[j][last];
            }
        }

        min_cut
    }

    /// Stoer-Wagner minimum cut, also recovering a partition that actually
    /// achieves the returned value.
    ///
    /// This is the fix for a prior bug: the old `compute_partition` ignored
    /// the computed cut entirely and returned an arbitrary BFS-order
    /// bisection (roughly half the vertices reachable from an arbitrary
    /// start vertex), with no relationship to which edges the min-cut
    /// algorithm actually selected. Any caller that used `partition` as a
    /// structural "which vertices sit on the cut boundary" signal — e.g.
    /// `ruvector-agent-memory`'s `MincutGatedForgetting` — would have been
    /// silently acting on the wrong vertices on every sparsifier-path call.
    /// See `docs/research/nightly/2026-10-29-approx-mincut-partition-fix`.
    ///
    /// For `n <= 3` vertices, isolating the minimum-weighted-degree vertex
    /// is *exactly* the global minimum cut (the only candidate cuts on 2 or
    /// 3 vertices are the `n` singleton-vs-rest cuts, whose values are
    /// exactly each vertex's total degree), so the value and partition
    /// below both stay exact in that regime, matching the pre-existing
    /// `stoer_wagner`'s (value-only) behavior for the same case.
    fn stoer_wagner_with_partition(
        &self,
        adj: &HashMap<VertexId, Vec<(VertexId, f64)>>,
        vertices: &HashSet<VertexId>,
    ) -> (f64, (Vec<VertexId>, Vec<VertexId>)) {
        if vertices.is_empty() {
            return (f64::INFINITY, (Vec::new(), Vec::new()));
        }
        if vertices.len() == 1 {
            return (
                f64::INFINITY,
                (vertices.iter().copied().collect(), Vec::new()),
            );
        }

        let verts: Vec<VertexId> = vertices.iter().copied().collect();
        let n = verts.len();

        if n <= 3 {
            let min_entry = adj
                .iter()
                .filter(|(v, _)| vertices.contains(v))
                .map(|(&v, neighbors)| (v, neighbors.iter().map(|(_, w)| *w).sum::<f64>()))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            return match min_entry {
                Some((min_vertex, min_degree)) => {
                    let rest = verts.iter().copied().filter(|&v| v != min_vertex).collect();
                    (min_degree, (vec![min_vertex], rest))
                }
                None => (f64::INFINITY, (verts, Vec::new())),
            };
        }

        let vert_to_idx: HashMap<_, _> = verts.iter().enumerate().map(|(i, &v)| (v, i)).collect();

        let mut weights = vec![vec![0.0; n]; n];
        for (&v, neighbors) in adj {
            if let Some(&i) = vert_to_idx.get(&v) {
                for &(neighbor, weight) in neighbors {
                    if let Some(&j) = vert_to_idx.get(&neighbor) {
                        weights[i][j] += weight;
                    }
                }
            }
        }

        let mut min_cut = f64::INFINITY;
        let mut best_group: Vec<usize> = Vec::new();
        let mut active: Vec<bool> = vec![true; n];
        // group_members[i]: original vertex indices currently merged into
        // supervertex i (standard Stoer-Wagner contraction bookkeeping,
        // extended to track membership so the cut-of-the-phase partition
        // can be recovered, not just its value).
        let mut group_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        for phase in 0..n - 1 {
            if min_cut == 0.0 {
                break;
            }

            let mut in_a = vec![false; n];
            let mut cut_of_phase = vec![0.0; n];

            let first = match (0..n).find(|&i| active[i]) {
                Some(f) => f,
                None => break,
            };
            in_a[first] = true;

            let mut last = first;
            let mut before_last = first;

            let active_count = active.iter().filter(|&&a| a).count();
            for _ in 1..active_count {
                for j in 0..n {
                    if active[j] && !in_a[j] && weights[last][j] > 0.0 {
                        cut_of_phase[j] += weights[last][j];
                    }
                }

                before_last = last;
                let mut max_val = f64::NEG_INFINITY;
                for j in 0..n {
                    if active[j] && !in_a[j] && cut_of_phase[j] > max_val {
                        max_val = cut_of_phase[j];
                        last = j;
                    }
                }

                if max_val == f64::NEG_INFINITY {
                    break;
                }
                in_a[last] = true;
            }

            // Same update condition as `stoer_wagner`, strengthened with
            // `candidate < min_cut` purely to decide *when to also capture
            // the partition* — `min_cut.min(candidate)` already implies
            // this, so the final `min_cut` value is unchanged from
            // `stoer_wagner`'s.
            let candidate = cut_of_phase[last];
            if (candidate > 0.0 || phase == 0) && candidate < min_cut {
                min_cut = candidate;
                best_group = group_members[last].clone();
            }

            active[last] = false;
            let absorbed = std::mem::take(&mut group_members[last]);
            group_members[before_last].extend(absorbed);
            for j in 0..n {
                weights[before_last][j] += weights[last][j];
                weights[j][before_last] += weights[j][last];
            }
        }

        if best_group.is_empty() {
            // Only reachable if every phase's cut was 0 (fully disconnected
            // within this subgraph, which callers should have already ruled
            // out via `is_connected`) — fall back to the final contraction's
            // grouping so callers still get a valid 2-coloring.
            best_group = (0..n)
                .find(|&i| active[i])
                .map(|i| group_members[i].clone())
                .unwrap_or_default();
        }

        let side_a_set: HashSet<usize> = best_group.into_iter().collect();
        let side_a: Vec<VertexId> = side_a_set.iter().map(|&i| verts[i]).collect();
        let side_b: Vec<VertexId> = (0..n)
            .filter(|i| !side_a_set.contains(i))
            .map(|i| verts[i])
            .collect();

        (min_cut, (side_a, side_b))
    }
}

impl Default for ApproxMinCut {
    fn default() -> Self {
        Self::new(ApproxMinCutConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_approx_min_cut() {
        let mut approx = ApproxMinCut::default();

        approx.insert_edge(0, 1, 1.0);
        approx.insert_edge(1, 2, 1.0);
        approx.insert_edge(2, 0, 1.0);

        let result = approx.min_cut();
        assert!(result.value >= 1.8); // Should be close to 2.0
        assert!(result.value <= 2.2);
    }

    #[test]
    fn test_triangle_graph() {
        let mut approx = ApproxMinCut::with_epsilon(0.1);

        approx.insert_edge(0, 1, 1.0);
        approx.insert_edge(1, 2, 1.0);
        approx.insert_edge(2, 0, 1.0);

        let value = approx.min_cut_value();
        assert!((value - 2.0).abs() < 0.5); // Approximate
    }

    #[test]
    fn test_disconnected_graph() {
        let mut approx = ApproxMinCut::default();

        approx.insert_edge(0, 1, 1.0);
        approx.insert_edge(2, 3, 1.0);

        assert!(!approx.is_connected());
        assert_eq!(approx.min_cut_value(), 0.0);
    }

    #[test]
    fn test_single_edge() {
        let mut approx = ApproxMinCut::default();

        approx.insert_edge(0, 1, 5.0);

        let value = approx.min_cut_value();
        assert!((value - 5.0).abs() < 1.0);
    }

    #[test]
    fn test_path_graph() {
        let mut approx = ApproxMinCut::default();

        // Path: 0 - 1 - 2 - 3
        approx.insert_edge(0, 1, 1.0);
        approx.insert_edge(1, 2, 1.0);
        approx.insert_edge(2, 3, 1.0);

        let value = approx.min_cut_value();
        assert!(value >= 0.5); // Should be close to 1.0
        assert!(value <= 1.5);
    }

    #[test]
    fn test_delete_edge() {
        let mut approx = ApproxMinCut::default();

        approx.insert_edge(0, 1, 1.0);
        approx.insert_edge(1, 2, 1.0);
        approx.insert_edge(2, 0, 1.0);

        assert!(approx.is_connected());

        approx.delete_edge(1, 2);

        assert!(approx.is_connected());
        let value = approx.min_cut_value();
        assert!(value >= 0.5 && value <= 1.5);
    }

    #[test]
    fn test_stats() {
        let mut approx = ApproxMinCut::default();

        approx.insert_edge(0, 1, 1.0);
        approx.insert_edge(1, 2, 1.0);
        approx.delete_edge(0, 1);
        approx.min_cut_value();

        let stats = approx.stats();
        assert_eq!(stats.insertions, 2);
        assert_eq!(stats.deletions, 1);
        assert_eq!(stats.queries, 1);
    }

    #[test]
    fn test_result_bounds() {
        let mut approx = ApproxMinCut::with_epsilon(0.2);

        approx.insert_edge(0, 1, 2.0);
        approx.insert_edge(1, 2, 2.0);
        approx.insert_edge(2, 0, 2.0);

        let result = approx.min_cut();
        assert!(result.lower_bound <= result.value);
        assert!(result.value <= result.upper_bound);
        assert_eq!(result.epsilon, 0.2);
    }

    #[test]
    fn test_larger_graph() {
        let mut approx = ApproxMinCut::with_epsilon(0.15);

        // Create a cycle
        for i in 0..10 {
            approx.insert_edge(i, (i + 1) % 10, 1.0);
        }

        let value = approx.min_cut_value();
        assert!(value >= 1.0); // At least 2.0 theoretically
        assert!(value <= 3.0);
    }

    /// Checks that `partition` is an actual 2-coloring of the cut the
    /// algorithm computed, not an unrelated bisection: every edge strictly
    /// inside one side must sum (with the cross-edges) to the full edge
    /// weight, and the cross-edge weight must equal the reported value.
    fn assert_partition_matches_value(
        approx: &mut ApproxMinCut,
        edges: &[(VertexId, VertexId, f64)],
    ) {
        let result = approx.min_cut();
        let (side_a, side_b) = result
            .partition
            .clone()
            .expect("connected graph with >=2 vertices must return a partition");
        assert!(!side_a.is_empty(), "side A must not be empty");
        assert!(!side_b.is_empty(), "side B must not be empty");
        let side_a_set: HashSet<VertexId> = side_a.into_iter().collect();
        let side_b_set: HashSet<VertexId> = side_b.into_iter().collect();
        assert!(
            side_a_set.is_disjoint(&side_b_set),
            "partition sides must not overlap"
        );

        let cross_weight: f64 = edges
            .iter()
            .filter(|(u, v, _)| {
                side_a_set.contains(u) != side_a_set.contains(v)
                    || side_b_set.contains(u) != side_b_set.contains(v)
            })
            .map(|(_, _, w)| w)
            .sum();

        assert!(
            (cross_weight - result.value).abs() < 1e-6,
            "partition's actual cross-edge weight ({cross_weight}) must equal the reported cut value ({})",
            result.value
        );
    }

    #[test]
    fn partition_matches_reported_value_on_triangle() {
        let edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let mut approx = ApproxMinCut::default();
        for &(u, v, w) in &edges {
            approx.insert_edge(u, v, w);
        }
        assert_partition_matches_value(&mut approx, &edges);
    }

    #[test]
    fn partition_matches_reported_value_on_path() {
        let edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        let mut approx = ApproxMinCut::default();
        for &(u, v, w) in &edges {
            approx.insert_edge(u, v, w);
        }
        assert_partition_matches_value(&mut approx, &edges);
    }

    /// Two 6-cliques joined by a single weak bridge edge: regression test
    /// for the fix. The old stub `compute_partition` returned an arbitrary
    /// BFS-order bisection with no relation to the actual min cut, so it
    /// would essentially never isolate exactly the bridge edge on a graph
    /// this shape. The real minimum cut here is unambiguous: cutting the
    /// lone bridge edge (weight 0.1) is far cheaper than cutting any clique
    /// apart (every intra-clique vertex has total weighted degree >= 5.0).
    #[test]
    fn partition_isolates_the_weak_bridge() {
        let mut edges: Vec<(VertexId, VertexId, f64)> = Vec::new();
        for clique_base in [0u64, 100] {
            for i in 0..6u64 {
                for j in (i + 1)..6u64 {
                    edges.push((clique_base + i, clique_base + j, 1.0));
                }
            }
        }
        edges.push((0, 100, 0.1)); // the weak bridge

        let mut approx = ApproxMinCut::default();
        for &(u, v, w) in &edges {
            approx.insert_edge(u, v, w);
        }

        let result = approx.min_cut();
        assert!(
            (result.value - 0.1).abs() < 1e-6,
            "expected the bridge's weight 0.1 as the global min cut, got {}",
            result.value
        );

        let (side_a, side_b) = result.partition.expect("must return a partition");
        let side_a_set: HashSet<VertexId> = side_a.into_iter().collect();
        let side_b_set: HashSet<VertexId> = side_b.into_iter().collect();
        // The two sides must exactly separate the two cliques (0 is in one
        // side, 100 is in the other) for the cut to actually be the bridge.
        assert_ne!(
            side_a_set.contains(&0),
            side_a_set.contains(&100),
            "the two cliques must land on opposite sides of the partition"
        );
        assert_ne!(side_b_set.contains(&0), side_b_set.contains(&100),);
        assert_partition_matches_value(&mut approx, &edges);
    }

    #[test]
    fn repeated_min_cut_calls_keep_partition_in_sync_with_cached_value() {
        let mut approx = ApproxMinCut::default();
        approx.insert_edge(0, 1, 1.0);
        approx.insert_edge(1, 2, 1.0);
        approx.insert_edge(2, 0, 1.0);

        let first = approx.min_cut();
        // Second call hits the `cached_min_cut` fast path in `min_cut()`:
        // this is the branch that used to unconditionally return
        // `partition: None` even though a value was already cached.
        let second = approx.min_cut();
        assert_eq!(first.value, second.value);
        assert_eq!(first.partition, second.partition);
        assert!(second.partition.is_some());
    }

    #[test]
    fn disconnected_graph_partition_matches_zero_cut() {
        let mut approx = ApproxMinCut::default();
        approx.insert_edge(0, 1, 1.0);
        approx.insert_edge(2, 3, 1.0);

        let result = approx.min_cut();
        assert_eq!(result.value, 0.0);
        let (side_a, side_b) = result
            .partition
            .expect("disconnected graph must still return a partition");
        let side_a_set: HashSet<VertexId> = side_a.into_iter().collect();
        let side_b_set: HashSet<VertexId> = side_b.into_iter().collect();
        assert!(!side_a_set.is_empty() && !side_b_set.is_empty());
        assert!(side_a_set.is_disjoint(&side_b_set));
    }
}
