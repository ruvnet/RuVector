//! In-memory knowledge graph with similarity edges
//!
//! Integrates ruvector-mincut for real graph partitioning,
//! ruvector-solver for PPR-based ranked search, and
//! ruvector-sparsifier for compressed spectral analytics (ADR-116).

use crate::types::*;
use ruvector_mincut::canonical::source_anchored::{self as canonical_sa, SourceAnchoredConfig};
use ruvector_mincut::graph::DynamicGraph;
use ruvector_mincut::{DynamicMinCut, MinCutBuilder};
use ruvector_solver::forward_push::ForwardPushSolver;
use ruvector_solver::types::CsrMatrix;
use ruvector_sparsifier::traits::Sparsifier;
use ruvector_sparsifier::{AdaptiveGeoSpar, SparseGraph, SparsifierConfig};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use uuid::Uuid;

/// Knowledge graph maintaining similarity relationships
pub struct KnowledgeGraph {
    nodes: HashMap<Uuid, GraphNode>,
    edges: Vec<GraphEdge>,
    similarity_threshold: f64,
    /// Real min-cut structure (lazy-initialized)
    mincut: Option<DynamicMinCut>,
    /// CSR cache for solver-based search.
    ///
    /// Behind its own lock, with interior mutability, so the lazy rebuild
    /// does not force `ranked_search` to be `&mut self`. Taking a write lock
    /// on the whole graph to read it was the reason every search after an
    /// inject blocked every other reader.
    csr_cache: parking_lot::RwLock<Option<Arc<CsrMatrix<f64>>>>,
    /// Maps graph indices to memory IDs
    node_ids: Vec<Uuid>,
    /// Reverse index: Uuid → position in node_ids (O(1) lookup)
    node_index: HashMap<Uuid, usize>,
    /// Whether the CSR cache needs rebuilding
    csr_dirty: AtomicBool,
    /// Serialises CSR rebuilds so N concurrent first-searches after an inject
    /// rebuild once between them instead of N times.
    csr_build: parking_lot::Mutex<()>,
    /// Spectral sparsifier for compressed graph analytics (ADR-116)
    sparsifier: Option<AdaptiveGeoSpar>,
    /// Bumped every time `node_index` is REASSIGNED rather than appended to.
    ///
    /// A sparsifier is built against node *positions*, so any reshuffle
    /// invalidates one that is mid-build. Comparing counts is not enough:
    /// a remove plus an add restores the counts while shifting every index
    /// after the removed node, and `rebuild_from_batch` reassigns every
    /// position even when the memory set is identical (it iterates a DashMap,
    /// whose order is arbitrary).
    index_generation: u64,
    /// Mutations recorded while an off-lock rebuild is in flight (ADR-349
    /// item 6). `Some` doubles as the single-flight marker: `begin_rebuild`
    /// refuses while it is set. See `graph/rebuild.rs`.
    rebuild_log: Option<rebuild::RebuildLog>,
    /// Monotonic id handed to each `begin_rebuild`, so an install or abort
    /// from a superseded rebuild can never touch a newer one.
    rebuild_seq: u64,
}

pub mod rebuild;

struct GraphNode {
    embedding: Vec<f32>,
    category: BrainCategory,
    /// Mean quality score at insertion time (ADR-149 P2).
    /// Used to skip low-quality nodes when building edges.
    quality: f64,
}

struct GraphEdge {
    source: Uuid,
    target: Uuid,
    weight: f64,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            similarity_threshold: 0.55,
            mincut: None,
            csr_cache: parking_lot::RwLock::new(None),
            node_ids: Vec::new(),
            node_index: HashMap::new(),
            csr_dirty: AtomicBool::new(false),
            csr_build: parking_lot::Mutex::new(()),
            sparsifier: None,
            index_generation: 0,
            rebuild_log: None,
            rebuild_seq: 0,
        }
    }

    /// Rebuild the entire graph from a batch of memories (ADR-149 P3), in
    /// place, with `&mut self` held for the whole O(n²) edge pass.
    ///
    /// **Do not call this from a request-serving path.** At the live ~60k
    /// nodes it holds whatever lock guards `self` for the full all-pairs
    /// build, which is exactly the ADR-349 item-6 504 mechanism. Serving code
    /// uses [`rebuild::spawn_rebuild`], which builds off-lock and swaps the
    /// result in under a brief write lock. This entry point remains for the
    /// offline worker binary and for tests, where nothing else is waiting.
    ///
    /// Produces exactly the same nodes, positions, edges and edge order as
    /// the off-lock path (both go through [`KnowledgeGraph::build_batch`]).
    /// Cancels any off-lock rebuild that is in flight: its snapshot predates
    /// this one, so its install is refused.
    pub fn rebuild_from_batch(&mut self, memories: &[BrainMemory]) {
        self.rebuild_log = None;
        let build = Self::build_batch(
            memories,
            self.similarity_threshold,
            rebuild::default_build_threads(),
        );
        let (retired, stats) = self.swap_in_batch(build);
        drop(retired);
        tracing::info!(
            nodes = self.nodes.len(),
            edges = self.edges.len(),
            build_ms = stats.elapsed.as_millis() as u64,
            threads = stats.threads,
            "Graph rebuilt from batch (ADR-149 P3)"
        );
    }

    /// Add a memory as a graph node, creating edges to similar nodes
    pub fn add_memory(&mut self, memory: &BrainMemory) {
        let quality = memory.quality_score.mean();
        let new_node = GraphNode {
            embedding: memory.embedding.clone(),
            category: memory.category.clone(),
            quality,
        };
        // An off-lock rebuild is building from a snapshot that may not include
        // this memory; record it so the install can replay it (ADR-349 #6).
        if let Some(log) = self.rebuild_log.as_mut() {
            log.record_add(memory.id, &new_node);
        }
        self.add_node(memory.id, new_node);
    }

    /// Insert one node and its similarity edges. Shared by `add_memory` and
    /// the rebuild-install replay; does not touch the rebuild log.
    fn add_node(&mut self, id: Uuid, new_node: GraphNode) {
        let memory_id = id;

        // ADR-149 P2: quality floor for edge building — skip low-quality nodes
        // to reduce noisy edges and speed up graph operations.
        const EDGE_QUALITY_FLOOR: f64 = 0.01;

        // Compute edges to existing nodes
        let mut new_edges = Vec::new();
        for (existing_id, existing_node) in &self.nodes {
            // Skip low-quality neighbors when building edges
            if existing_node.quality < EDGE_QUALITY_FLOOR {
                continue;
            }
            let sim = cosine_similarity(&new_node.embedding, &existing_node.embedding);
            if sim >= self.similarity_threshold {
                new_edges.push(GraphEdge {
                    source: memory_id,
                    target: *existing_id,
                    weight: sim,
                });
            }
        }

        let new_idx = self.node_ids.len();

        // Insert into DynamicMinCut if initialized
        if let Some(ref mut mincut) = self.mincut {
            let u = new_idx as u64;
            for edge in &new_edges {
                if let Some(&v_pos) = self.node_index.get(&edge.target) {
                    let _ = mincut.insert_edge(u, v_pos as u64, edge.weight);
                }
            }
        }

        self.nodes.insert(memory_id, new_node);
        self.node_index.insert(memory_id, new_idx);
        self.node_ids.push(memory_id);

        // Update sparsifier with new edges (ADR-116)
        if let Some(ref mut spar) = self.sparsifier {
            for edge in &new_edges {
                if let Some(&v_pos) = self.node_index.get(&edge.target) {
                    let _ = spar.insert_edge(new_idx, v_pos, edge.weight);
                }
            }
        }

        self.edges.extend(new_edges);

        // Mark CSR as dirty — deferred rebuild until next query
        self.mark_csr_dirty();
    }

    /// Remove a memory from the graph
    pub fn remove_memory(&mut self, id: &Uuid) {
        // The in-flight rebuild's snapshot may still contain this memory;
        // record the removal so the install can replay it (ADR-349 #6).
        if let Some(log) = self.rebuild_log.as_mut() {
            log.record_remove(*id);
        }
        self.remove_node(id);
    }

    /// Remove one node and its edges. Shared by `remove_memory` and the
    /// rebuild-install replay; does not touch the rebuild log.
    fn remove_node(&mut self, id: &Uuid) {
        // Collect edges to delete from sparsifier before removing them
        if let Some(ref mut spar) = self.sparsifier {
            if let Some(&u_pos) = self.node_index.get(id) {
                for edge in &self.edges {
                    let (src, tgt) = if edge.source == *id {
                        (u_pos, self.node_index.get(&edge.target).copied())
                    } else if edge.target == *id {
                        (u_pos, self.node_index.get(&edge.source).copied())
                    } else {
                        continue;
                    };
                    if let Some(v_pos) = tgt {
                        let _ = spar.delete_edge(src, v_pos);
                    }
                }
            }
        }

        self.nodes.remove(id);
        self.edges.retain(|e| e.source != *id && e.target != *id);
        self.node_ids.retain(|nid| nid != id);
        // Rebuild the index after removal (positions shifted)
        self.node_index.clear();
        for (i, nid) in self.node_ids.iter().enumerate() {
            self.node_index.insert(*nid, i);
        }
        // Invalidate caches — full rebuild needed
        self.mincut = None;
        *self.csr_cache.write() = None;
        self.csr_dirty.store(false, Ordering::Release);
        // Positions after the removed node have all shifted.
        self.index_generation = self.index_generation.wrapping_add(1);
        // Sparsifier indices are now stale after compaction — rebuild lazily
        self.sparsifier = None;
    }

    /// Get top-k similar memories by graph traversal.
    ///
    /// Uses ForwardPushSolver PPR for graph-aware relevance when CSR is
    /// available, merging with cosine similarity scores. Falls back to
    /// brute-force cosine if CSR is unavailable.
    pub fn ranked_search(&self, query_embedding: &[f32], k: usize) -> Vec<(Uuid, f64)> {
        self.ensure_csr();
        // Brute-force cosine scores
        let mut cosine_scores: Vec<(Uuid, f64)> = self
            .nodes
            .iter()
            .map(|(id, node)| (*id, cosine_similarity(query_embedding, &node.embedding)))
            .collect();

        // Boost with PageRank scores when available
        if let Some(ppr_map) = self.pagerank_scores(query_embedding, k) {
            for (id, score) in &mut cosine_scores {
                if let Some(&ppr) = ppr_map.get(id) {
                    *score = *score * 0.6 + ppr * 0.4;
                }
            }
        }

        cosine_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cosine_scores.truncate(k);
        cosine_scores
    }

    /// Compute PageRank-based scores using ForwardPushSolver.
    ///
    /// Builds a CsrMatrix from graph edges and runs PPR from the node
    /// most similar to `query_embedding`. Returns a map of node ID to
    /// PPR score, or `None` if PPR cannot be computed.
    pub fn pagerank_search(&self, query_embedding: &[f32], k: usize) -> Vec<(Uuid, f64)> {
        self.ensure_csr();
        if let Some(ppr_map) = self.pagerank_scores(query_embedding, k) {
            let mut results: Vec<(Uuid, f64)> = ppr_map.into_iter().collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(k);
            results
        } else {
            Vec::new()
        }
    }

    /// Mark the CSR cache stale. Cheap; the rebuild is deferred to the next
    /// query that actually needs it.
    fn mark_csr_dirty(&self) {
        self.csr_dirty.store(true, Ordering::Release);
    }

    /// Ensure the CSR cache is up-to-date (lazy rebuild).
    ///
    /// Takes `&self`: the cache lives behind its own lock, so a search no
    /// longer needs a write lock on the whole graph. Double-checked under
    /// `csr_build` so a burst of concurrent searches after an inject performs
    /// exactly one rebuild rather than one per thread.
    ///
    /// The rebuild is still synchronous, so a search never runs against a
    /// stale CSR — this fixes the lock contention without introducing a
    /// staleness window.
    fn ensure_csr(&self) {
        if !self.csr_dirty.load(Ordering::Acquire) {
            return;
        }
        let _build = self.csr_build.lock();
        // Re-check: another thread may have rebuilt while we waited.
        if !self.csr_dirty.load(Ordering::Acquire) {
            return;
        }
        self.rebuild_csr();
        self.csr_dirty.store(false, Ordering::Release);
    }

    /// Internal: compute raw PPR scores keyed by node ID.
    fn pagerank_scores(&self, query_embedding: &[f32], k: usize) -> Option<HashMap<Uuid, f64>> {
        // Clone the Arc out and release the cache lock immediately, so a
        // concurrent rebuild is never blocked by a long-running PPR solve.
        let csr = self.csr_cache.read().clone()?;
        let csr = csr.as_ref();
        if csr.rows == 0 {
            return None;
        }

        // Find closest node as source for PPR (use index for O(1) lookup)
        let source = self
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                let &pos = self.node_index.get(id)?;
                Some((pos, cosine_similarity(query_embedding, &node.embedding)))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)?;

        if source >= csr.rows {
            return None;
        }

        let solver = ForwardPushSolver::default_params();
        let ppr_results = solver.top_k(csr, source, k * 3).ok()?;

        let mut map = HashMap::new();
        for (idx, ppr_score) in ppr_results {
            if let Some(id) = self.node_ids.get(idx) {
                map.insert(*id, ppr_score);
            }
        }
        Some(map)
    }

    /// Partition with full result including cut_value and edge_strengths.
    ///
    /// Uses DynamicMinCut if available (>= 3 nodes), falls back to Union-Find.
    pub fn partition(&self, min_cluster_size: usize) -> Vec<KnowledgeCluster> {
        self.partition_full(min_cluster_size).0
    }

    /// Partition returning (clusters, cut_value, edge_strengths).
    pub fn partition_full(
        &self,
        min_cluster_size: usize,
    ) -> (Vec<KnowledgeCluster>, f64, Vec<EdgeStrengthInfo>) {
        // Try real MinCut partitioning
        if self.nodes.len() >= 3 {
            if let Some((clusters, cut_val, strengths)) =
                self.partition_via_mincut_full(min_cluster_size)
            {
                if clusters.len() >= 2 {
                    return (clusters, cut_val, strengths);
                }
            }
        }

        // Fallback: Union-Find based clustering
        let clusters = self.partition_union_find(min_cluster_size);
        if clusters.len() >= 2 {
            let strengths = self.compute_edge_strengths(&clusters);
            return (clusters, 0.0, strengths);
        }

        // Final fallback: category-based partitioning
        let clusters = self.partition_by_category(min_cluster_size);
        let strengths = self.compute_edge_strengths(&clusters);
        (clusters, 0.0, strengths)
    }

    /// Category-based partitioning fallback: group nodes by their BrainCategory
    fn partition_by_category(&self, min_cluster_size: usize) -> Vec<KnowledgeCluster> {
        let mut by_category: HashMap<BrainCategory, Vec<Uuid>> = HashMap::new();
        for (&id, node) in &self.nodes {
            by_category
                .entry(node.category.clone())
                .or_default()
                .push(id);
        }

        let mut clusters = Vec::new();
        let mut cluster_id = 0u32;
        for (_, members) in by_category {
            if members.len() >= min_cluster_size {
                clusters.push(self.build_cluster(cluster_id, &members));
                cluster_id += 1;
            }
        }
        clusters
    }

    /// Attempt partitioning via DynamicMinCut (returns clusters, cut_value, edge_strengths).
    ///
    /// When a sparsifier is available and the full graph has > 50 000 edges,
    /// uses the sparsified edge set (~19K edges vs ~1M) for a ~59x speedup
    /// while preserving spectral cut quality (ADR-116).
    fn partition_via_mincut_full(
        &self,
        min_cluster_size: usize,
    ) -> Option<(Vec<KnowledgeCluster>, f64, Vec<EdgeStrengthInfo>)> {
        let use_sparsified = self.sparsifier.is_some() && self.edges.len() > 50_000;

        let edges: Vec<(u64, u64, f64)> = if use_sparsified {
            let spar = self.sparsifier.as_ref().unwrap();
            spar.sparsifier()
                .edges()
                .map(|(u, v, w)| (u as u64, v as u64, w))
                .collect()
        } else {
            self.edges
                .iter()
                .filter_map(|e| {
                    let &u = self.node_index.get(&e.source)?;
                    let &v = self.node_index.get(&e.target)?;
                    Some((u as u64, v as u64, e.weight))
                })
                .collect()
        };

        if use_sparsified {
            tracing::debug!(
                full_edges = self.edges.len(),
                sparsified_edges = edges.len(),
                "partition_via_mincut_full: using sparsified edges"
            );
        }

        let mincut = MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build()
            .ok()?;

        let result = mincut.min_cut();
        let cut_value = result.value;
        let (side_a, side_b) = result.partition?;

        let mut clusters = Vec::new();
        let mut cluster_id = 0u32;

        for side in [side_a, side_b] {
            let members: Vec<Uuid> = side
                .iter()
                .filter_map(|&idx| self.node_ids.get(idx as usize).copied())
                .collect();

            if members.len() < min_cluster_size {
                continue;
            }

            let cluster = self.build_cluster(cluster_id, &members);
            clusters.push(cluster);
            cluster_id += 1;
        }

        if clusters.is_empty() {
            return None;
        }

        let strengths = self.compute_edge_strengths(&clusters);
        Some((clusters, cut_value, strengths))
    }

    /// Union-Find based clustering (fallback)
    fn partition_union_find(&self, min_cluster_size: usize) -> Vec<KnowledgeCluster> {
        let ids: Vec<Uuid> = self.nodes.keys().copied().collect();
        let mut parent: HashMap<Uuid, Uuid> = ids.iter().map(|&id| (id, id)).collect();

        fn find(parent: &mut HashMap<Uuid, Uuid>, x: Uuid) -> Uuid {
            let p = parent[&x];
            if p == x {
                return x;
            }
            let root = find(parent, p);
            parent.insert(x, root);
            root
        }

        fn union(parent: &mut HashMap<Uuid, Uuid>, a: Uuid, b: Uuid) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent.insert(ra, rb);
            }
        }

        for edge in &self.edges {
            union(&mut parent, edge.source, edge.target);
        }

        let mut clusters_map: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
        for &id in &ids {
            let root = find(&mut parent, id);
            clusters_map.entry(root).or_default().push(id);
        }

        let mut clusters = Vec::new();
        let mut cluster_id = 0u32;
        for (_, members) in clusters_map {
            if members.len() < min_cluster_size {
                continue;
            }
            clusters.push(self.build_cluster(cluster_id, &members));
            cluster_id += 1;
        }
        clusters
    }

    /// Build a KnowledgeCluster from member IDs
    fn build_cluster(&self, id: u32, members: &[Uuid]) -> KnowledgeCluster {
        let dim = self
            .nodes
            .values()
            .next()
            .map(|n| n.embedding.len())
            .unwrap_or(0);
        let mut centroid = vec![0.0f32; dim];
        let mut category_counts: HashMap<BrainCategory, usize> = HashMap::new();
        let mut embeddings = Vec::new();
        for mid in members {
            if let Some(node) = self.nodes.get(mid) {
                for (i, &v) in node.embedding.iter().enumerate() {
                    if i < centroid.len() {
                        centroid[i] += v;
                    }
                }
                *category_counts.entry(node.category.clone()).or_default() += 1;
                embeddings.push(node.embedding.clone());
            }
        }
        let n = members.len() as f32;
        for v in &mut centroid {
            *v /= n;
        }
        let dominant = category_counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(cat, _)| cat)
            .unwrap_or(BrainCategory::Pattern);

        // Compute coherence: average cosine similarity of members to centroid
        let coherence = if embeddings.len() < 2 {
            1.0
        } else {
            let avg_sim: f64 = embeddings
                .iter()
                .map(|emb| cosine_similarity(emb, &centroid))
                .sum::<f64>()
                / embeddings.len() as f64;
            avg_sim
        };

        KnowledgeCluster {
            id,
            memory_ids: members.to_vec(),
            centroid,
            dominant_category: dominant,
            size: members.len(),
            coherence,
        }
    }

    /// Compute edge strengths between pairs of clusters
    /// Uses HashSet for O(1) membership lookups instead of Vec::contains O(n)
    fn compute_edge_strengths(&self, clusters: &[KnowledgeCluster]) -> Vec<EdgeStrengthInfo> {
        use std::collections::HashSet;

        // Pre-build HashSets for O(1) membership checks
        let cluster_sets: Vec<HashSet<Uuid>> = clusters
            .iter()
            .map(|c| c.memory_ids.iter().copied().collect())
            .collect();

        let mut strengths = Vec::new();
        for (i, ca) in clusters.iter().enumerate() {
            let set_a = &cluster_sets[i];
            for (j, cb) in clusters.iter().enumerate().skip(i + 1) {
                let set_b = &cluster_sets[j];
                // Sum weights of edges crossing between these two clusters
                let mut cross_weight = 0.0f64;
                let mut cross_count = 0u32;
                for edge in &self.edges {
                    let src_in_a = set_a.contains(&edge.source);
                    let tgt_in_b = set_b.contains(&edge.target);
                    let src_in_b = set_b.contains(&edge.source);
                    let tgt_in_a = set_a.contains(&edge.target);
                    if (src_in_a && tgt_in_b) || (src_in_b && tgt_in_a) {
                        cross_weight += edge.weight;
                        cross_count += 1;
                    }
                }
                if cross_count > 0 {
                    strengths.push(EdgeStrengthInfo {
                        source_cluster: ca.id,
                        target_cluster: cb.id,
                        strength: cross_weight / cross_count as f64,
                    });
                }
            }
        }
        strengths
    }

    /// Partition using source-anchored canonical min-cut (ADR-117).
    ///
    /// Returns deterministic clusters with a stable `cut_hash` suitable for
    /// RVF witnesses. Falls back to standard partition if canonical cut
    /// cannot be computed (disconnected graph, < 3 nodes, etc.).
    pub fn partition_canonical_full(
        &self,
        min_cluster_size: usize,
    ) -> (
        Vec<KnowledgeCluster>,
        f64,
        Vec<EdgeStrengthInfo>,
        Option<String>,
        Option<u64>,
    ) {
        if self.nodes.len() < 3 {
            let (clusters, cut_val, strengths) = self.partition_full(min_cluster_size);
            return (clusters, cut_val, strengths, None, None);
        }

        // Build a DynamicGraph snapshot for canonical computation
        let graph = DynamicGraph::new();
        for edge in &self.edges {
            if let (Some(&u), Some(&v)) = (
                self.node_index.get(&edge.source),
                self.node_index.get(&edge.target),
            ) {
                let _ = graph.insert_edge(u as u64, v as u64, edge.weight);
            }
        }

        let config = SourceAnchoredConfig::default();
        match canonical_sa::canonical_mincut(&graph, &config) {
            Some(cut) => {
                let cut_value = cut.lambda.to_f64();
                let cut_hash_hex = hex::encode(cut.cut_hash);
                let first_sep = cut.first_separable_vertex;

                // Build clusters from the canonical cut's source side
                let source_side: std::collections::HashSet<u64> =
                    cut.side_vertices.iter().copied().collect();

                let side_a: Vec<Uuid> = self
                    .node_ids
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| source_side.contains(&(*i as u64)))
                    .map(|(_, id)| *id)
                    .collect();
                let side_b: Vec<Uuid> = self
                    .node_ids
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !source_side.contains(&(*i as u64)))
                    .map(|(_, id)| *id)
                    .collect();

                let mut clusters = Vec::new();
                let mut cluster_id = 0u32;

                for side in [&side_a, &side_b] {
                    if side.len() >= min_cluster_size {
                        clusters.push(self.build_cluster(cluster_id, side));
                        cluster_id += 1;
                    }
                }

                if clusters.is_empty() {
                    let (cl, cv, st) = self.partition_full(min_cluster_size);
                    return (cl, cv, st, Some(cut_hash_hex), Some(first_sep));
                }

                let strengths = self.compute_edge_strengths(&clusters);
                (
                    clusters,
                    cut_value,
                    strengths,
                    Some(cut_hash_hex),
                    Some(first_sep),
                )
            }
            None => {
                // Canonical cut not available (disconnected graph, etc.)
                let (clusters, cut_val, strengths) = self.partition_full(min_cluster_size);
                (clusters, cut_val, strengths, None, None)
            }
        }
    }

    /// Rebuild the DynamicMinCut from all current edges
    pub fn rebuild_mincut(&mut self) {
        let edges: Vec<(u64, u64, f64)> = self
            .edges
            .iter()
            .filter_map(|e| {
                let &u = self.node_index.get(&e.source)?;
                let &v = self.node_index.get(&e.target)?;
                Some((u as u64, v as u64, e.weight))
            })
            .collect();

        self.mincut = MinCutBuilder::new().exact().with_edges(edges).build().ok();
    }

    /// Rebuild the CsrMatrix from the adjacency list
    pub fn rebuild_csr(&self) {
        let n = self.node_ids.len();
        if n == 0 {
            *self.csr_cache.write() = None;
            return;
        }

        let entries: Vec<(usize, usize, f64)> = self
            .edges
            .iter()
            .filter_map(|e| {
                let &u = self.node_index.get(&e.source)?;
                let &v = self.node_index.get(&e.target)?;
                Some((u, v, e.weight))
            })
            .collect();

        // Build the new matrix before taking the cache write lock, so readers
        // only ever wait for the pointer swap, not for the O(edges) build.
        let built = CsrMatrix::<f64>::from_coo(n, n, entries);
        *self.csr_cache.write() = Some(Arc::new(built));
    }

    /// Get the k nearest graph neighbors for a given memory ID.
    /// Returns (neighbor_id, edge_weight) sorted by descending weight.
    pub fn get_neighbors(&self, id: &Uuid, k: usize) -> Vec<(Uuid, f64)> {
        let mut neighbors: Vec<(Uuid, f64)> = self
            .edges
            .iter()
            .filter_map(|e| {
                if e.source == *id {
                    Some((e.target, e.weight))
                } else if e.target == *id {
                    Some((e.source, e.weight))
                } else {
                    None
                }
            })
            .collect();
        neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(k);
        neighbors
    }

    /// Get graph stats
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The node ids in index order. Exposed for tests that need to name a
    /// specific node (e.g. to drive `remove_memory`).
    pub fn node_ids_snapshot(&self) -> Vec<Uuid> {
        self.node_ids.clone()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    // ----- Sparsifier (ADR-116) -----------------------------------------------

    /// Tuning shared by the in-place and off-lock sparsifier builds.
    fn sparsifier_config() -> SparsifierConfig {
        SparsifierConfig {
            epsilon: 0.2,
            edge_budget_factor: 8,
            audit_interval: 500,
            walk_length: 6,
            num_walks: 10,
            n_audit_probes: 30,
            auto_rebuild_on_audit_failure: true,
            ..Default::default()
        }
    }

    /// Initialize or rebuild the spectral sparsifier from current edges.
    pub fn rebuild_sparsifier(&mut self) {
        if self.node_ids.is_empty() {
            self.sparsifier = None;
            return;
        }

        let mut sg = SparseGraph::with_capacity(self.node_ids.len());
        for edge in &self.edges {
            if let (Some(&u), Some(&v)) = (
                self.node_index.get(&edge.source),
                self.node_index.get(&edge.target),
            ) {
                let _ = sg.insert_or_update_edge(u, v, edge.weight);
            }
        }

        match AdaptiveGeoSpar::build(&sg, Self::sparsifier_config()) {
            Ok(spar) => {
                tracing::info!(
                    full_edges = self.edges.len(),
                    sparsified_edges = spar.sparsifier().num_edges(),
                    compression = %format!("{:.1}x", spar.compression_ratio()),
                    "Sparsifier built"
                );
                self.sparsifier = Some(spar);
            }
            Err(e) => {
                tracing::warn!("Sparsifier build failed: {e}");
                self.sparsifier = None;
            }
        }
    }

    /// Snapshot everything the sparsifier build needs, as plain data.
    ///
    /// Returned under a *read* lock and then used with no lock held at all,
    /// so the expensive `AdaptiveGeoSpar::build` no longer runs inside
    /// `graph.write()`. Returns `(coo_entries, node_count, edge_count, index_generation)`;
    /// the last three are the guard values for `install_sparsifier`.
    pub fn sparsifier_snapshot(&self) -> Option<(Vec<(usize, usize, f64)>, usize, usize, u64)> {
        if self.node_ids.is_empty() || self.edges.is_empty() {
            return None;
        }
        let entries: Vec<(usize, usize, f64)> = self
            .edges
            .iter()
            .filter_map(|e| {
                let &u = self.node_index.get(&e.source)?;
                let &v = self.node_index.get(&e.target)?;
                Some((u, v, e.weight))
            })
            .collect();
        Some((
            entries,
            self.node_ids.len(),
            self.edges.len(),
            self.index_generation,
        ))
    }

    /// Build a sparsifier from a snapshot. Pure — holds no lock, touches no
    /// `self`, and is safe to call from `spawn_blocking`.
    pub fn build_sparsifier_from(
        entries: &[(usize, usize, f64)],
        node_count: usize,
    ) -> Option<AdaptiveGeoSpar> {
        if node_count == 0 || entries.is_empty() {
            return None;
        }
        let mut sg = SparseGraph::with_capacity(node_count);
        for &(u, v, w) in entries {
            let _ = sg.insert_or_update_edge(u, v, w);
        }
        match AdaptiveGeoSpar::build(&sg, Self::sparsifier_config()) {
            Ok(spar) => Some(spar),
            Err(e) => {
                tracing::warn!("Sparsifier build failed: {e}");
                None
            }
        }
    }

    /// Install a sparsifier built from a snapshot, under a brief write lock.
    ///
    /// Refuses the install in three cases, each of which would otherwise
    /// leave a sparsifier that silently describes the wrong graph:
    ///
    /// 1. **Node positions were reassigned** (`index_generation` changed).
    ///    Counts alone do not catch this — a remove plus an add restores them
    ///    while shifting every index past the removed node, and
    ///    `rebuild_from_batch` reassigns all of them even for an identical
    ///    memory set.
    /// 2. **The graph shrank**, a cheap belt-and-braces check on top of (1).
    /// 3. **Someone already installed a sparsifier** while this build ran —
    ///    `rebuild_from_batch` is followed by an inline `rebuild_sparsifier`
    ///    on the small-graph path, and overwriting that fresh one with this
    ///    older build would be a regression.
    ///
    /// Edges appended during the build are replayed, because `add_memory`
    /// only feeds the sparsifier while it is `Some` — and it was `None` for
    /// the whole build window, so those edges are otherwise lost.
    ///
    /// Returns whether the sparsifier was installed.
    pub fn install_sparsifier(
        &mut self,
        spar: AdaptiveGeoSpar,
        snapshot_nodes: usize,
        snapshot_edges: usize,
        snapshot_generation: u64,
    ) -> bool {
        if self.index_generation != snapshot_generation {
            tracing::warn!(
                "Discarding sparsifier build: node positions were reassigned \
                 during build (generation {} -> {})",
                snapshot_generation,
                self.index_generation
            );
            return false;
        }
        if self.node_ids.len() < snapshot_nodes || self.edges.len() < snapshot_edges {
            tracing::warn!(
                "Discarding sparsifier build: graph shrank during build \
                 (nodes {} -> {}, edges {} -> {})",
                snapshot_nodes,
                self.node_ids.len(),
                snapshot_edges,
                self.edges.len()
            );
            return false;
        }
        if self.sparsifier.is_some() {
            tracing::info!(
                "Discarding sparsifier build: a newer sparsifier was installed \
                 while this one was building"
            );
            return false;
        }

        let mut spar = spar;
        let replayed = self.edges.len() - snapshot_edges;
        for edge in &self.edges[snapshot_edges..] {
            if let (Some(&u), Some(&v)) = (
                self.node_index.get(&edge.source),
                self.node_index.get(&edge.target),
            ) {
                let _ = spar.insert_edge(u, v, edge.weight);
            }
        }
        if replayed > 0 {
            tracing::info!("Replayed {replayed} edges added during sparsifier build");
        }

        tracing::info!(
            full_edges = self.edges.len(),
            sparsified_edges = spar.sparsifier().num_edges(),
            compression = %format!("{:.1}x", spar.compression_ratio()),
            "Sparsifier installed"
        );
        self.sparsifier = Some(spar);
        true
    }

    /// Ensure the sparsifier is initialized (lazy build on first access).
    pub fn ensure_sparsifier(&mut self) {
        if self.sparsifier.is_none() && !self.edges.is_empty() {
            self.rebuild_sparsifier();
        }
    }

    /// Get sparsifier stats for monitoring, or None if not initialized.
    pub fn sparsifier_stats(&self) -> Option<SparsifierStatsInfo> {
        let spar = self.sparsifier.as_ref()?;
        let stats = spar.stats();
        Some(SparsifierStatsInfo {
            full_edges: stats.full_edge_count,
            sparsified_edges: stats.edge_count,
            compression_ratio: spar.compression_ratio(),
            insertions: stats.insertions,
            deletions: stats.deletions,
            audits: stats.audit_count,
            audit_pass_rate: if stats.audit_count > 0 {
                stats.audit_pass_count as f64 / stats.audit_count as f64
            } else {
                1.0
            },
            full_rebuilds: stats.full_rebuilds,
        })
    }

    /// Run a spectral audit on the sparsifier, returning pass/fail and error.
    pub fn sparsifier_audit(&self) -> Option<(bool, f64)> {
        let spar = self.sparsifier.as_ref()?;
        let result = spar.audit();
        Some((result.passed, result.max_error))
    }
}

/// Sparsifier stats for the status endpoint (ADR-116).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SparsifierStatsInfo {
    pub full_edges: usize,
    pub sparsified_edges: usize,
    pub compression_ratio: f64,
    pub insertions: u64,
    pub deletions: u64,
    pub audits: u64,
    pub audit_pass_rate: f64,
    pub full_rebuilds: u64,
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// L2-normalize an embedding in place. Safe to call repeatedly (idempotent
/// within float precision).
#[inline]
pub fn normalize_embedding(emb: &mut [f32]) {
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        let inv = 1.0 / norm;
        for x in emb.iter_mut() {
            *x *= inv;
        }
    }
}

/// Fast cosine when BOTH vectors are pre-normalized to unit length.
/// This is just a dot product — ~3x faster than full cosine.
#[inline]
pub fn cosine_similarity_normalized(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let n = a.len();
    let chunks = n / 4;
    let (mut d0, mut d1) = (0.0f64, 0.0f64);
    for c in 0..chunks {
        let i = c * 4;
        d0 += (a[i] as f64) * (b[i] as f64) + (a[i + 2] as f64) * (b[i + 2] as f64);
        d1 += (a[i + 1] as f64) * (b[i + 1] as f64) + (a[i + 3] as f64) * (b[i + 3] as f64);
    }
    let mut sum = d0 + d1;
    for i in (chunks * 4)..n {
        sum += (a[i] as f64) * (b[i] as f64);
    }
    sum
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    // 4x unrolled dot product — compiler auto-vectorizes to SSE/AVX on x86,
    // NEON on ARM. Avoids ruvector-core::simd_intrinsics which is stripped
    // in the Docker build for cross-compilation compatibility.
    let n = a.len();
    let chunks = n / 4;
    let (mut dot0, mut dot1) = (0.0f64, 0.0f64);
    let (mut na0, mut na1) = (0.0f64, 0.0f64);
    let (mut nb0, mut nb1) = (0.0f64, 0.0f64);
    for c in 0..chunks {
        let i = c * 4;
        let (a0, a1, a2, a3) = (
            a[i] as f64,
            a[i + 1] as f64,
            a[i + 2] as f64,
            a[i + 3] as f64,
        );
        let (b0, b1, b2, b3) = (
            b[i] as f64,
            b[i + 1] as f64,
            b[i + 2] as f64,
            b[i + 3] as f64,
        );
        dot0 += a0 * b0 + a2 * b2;
        dot1 += a1 * b1 + a3 * b3;
        na0 += a0 * a0 + a2 * a2;
        na1 += a1 * a1 + a3 * a3;
        nb0 += b0 * b0 + b2 * b2;
        nb1 += b1 * b1 + b3 * b3;
    }
    for i in (chunks * 4)..n {
        let (ai, bi) = (a[i] as f64, b[i] as f64);
        dot0 += ai * bi;
        na0 += ai * ai;
        nb0 += bi * bi;
    }
    let dot = dot0 + dot1;
    let norm_a = (na0 + na1).sqrt();
    let norm_b = (nb0 + nb1).sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
