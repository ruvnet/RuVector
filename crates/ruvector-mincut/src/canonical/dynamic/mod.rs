//! Tier 3: Dynamic/incremental canonical minimum cut maintenance.
//!
//! Wraps the source-anchored `SourceAnchoredMinCut` engine with incremental
//! update logic that avoids full recomputation when possible.
//!
//! # Strategy
//!
//! The key insight is that many graph mutations do not change the minimum
//! cut. Specifically:
//!
//! - **Edge insertion**: If the new edge does not cross the current cut
//!   (both endpoints are on the same side), the cut value is unchanged.
//!   If it does cross the cut, the cut value may increase but the cut
//!   partition might no longer be minimum -- recompute.
//!
//! - **Edge deletion**: If the deleted edge is not in the current cut set,
//!   the cut value is unchanged. If it is in the cut set, the cut value
//!   decreases and we must recompute.
//!
//! - **Staleness**: After many incremental updates, accumulated drift may
//!   cause the cached cut to be incorrect. A configurable threshold
//!   triggers full recomputation.
//!
//! # Complexity
//!
//! - Best case (no recompute): O(1) per update.
//! - Worst case (full recompute): same as Tier 1.
//! - Amortized target: O(V * sqrt(E)) with staleness recomputation.
//!
//! # Epoch Tracking
//!
//! Every mutation increments the epoch counter. The `last_full_epoch`
//! records when the last full recomputation occurred. When
//! `epoch - last_full_epoch > staleness_threshold`, a full recompute
//! is triggered automatically.

use crate::graph::{VertexId, Weight};
use crate::canonical::FixedWeight;
use crate::canonical::source_anchored::{
    canonical_mincut, SourceAnchoredConfig, SourceAnchoredCut, SourceAnchoredReceipt,
    make_receipt,
};

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the dynamic incremental min-cut engine.
#[derive(Debug, Clone)]
pub struct DynamicMinCutConfig {
    /// Source-anchored algorithm configuration.
    pub canonical_config: SourceAnchoredConfig,
    /// Number of incremental updates before forcing a full recomputation.
    /// Set to 0 to disable staleness-based recomputation.
    pub staleness_threshold: u64,
}

impl Default for DynamicMinCutConfig {
    fn default() -> Self {
        Self {
            canonical_config: SourceAnchoredConfig::default(),
            staleness_threshold: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Edge mutation records
// ---------------------------------------------------------------------------

/// A pending edge mutation in a batch.
#[derive(Debug, Clone)]
pub enum EdgeMutation {
    /// Add an edge with the given weight.
    Add(VertexId, VertexId, Weight),
    /// Remove an edge.
    Remove(VertexId, VertexId),
}

// ---------------------------------------------------------------------------
// DynamicMinCut
// ---------------------------------------------------------------------------

/// Dynamic incremental canonical minimum cut engine.
///
/// Maintains the canonical cut across edge insertions and deletions,
/// avoiding full recomputation when the mutation does not affect the
/// current cut.
pub struct DynamicMinCut {
    /// Underlying min-cut engine.
    inner: crate::algorithm::DynamicMinCut,
    /// Configuration.
    config: DynamicMinCutConfig,
    /// Cached canonical cut result.
    cached_cut: Option<SourceAnchoredCut>,
    /// Current epoch (incremented on every mutation).
    epoch: u64,
    /// Epoch at which the last full recomputation occurred.
    last_full_epoch: u64,
    /// Number of incremental updates since last full recomputation.
    incremental_count: u64,
    /// Whether the cache is known to be stale.
    dirty: bool,
    /// Set of edges in the current cut set, for fast lookup.
    cut_edge_set: HashSet<(VertexId, VertexId)>,
    /// Set of vertex IDs on the source side of the current cut.
    source_side_set: HashSet<VertexId>,
}

impl DynamicMinCut {
    /// Create a new dynamic min-cut engine with default configuration.
    pub fn new() -> Self {
        Self {
            inner: crate::algorithm::DynamicMinCut::new(crate::MinCutConfig::default()),
            config: DynamicMinCutConfig::default(),
            cached_cut: None,
            epoch: 0,
            last_full_epoch: 0,
            incremental_count: 0,
            dirty: true,
            cut_edge_set: HashSet::new(),
            source_side_set: HashSet::new(),
        }
    }

    /// Create with explicit configuration.
    pub fn with_config(config: DynamicMinCutConfig) -> Self {
        Self {
            inner: crate::algorithm::DynamicMinCut::new(crate::MinCutConfig::default()),
            config,
            cached_cut: None,
            epoch: 0,
            last_full_epoch: 0,
            incremental_count: 0,
            dirty: true,
            cut_edge_set: HashSet::new(),
            source_side_set: HashSet::new(),
        }
    }

    /// Create from a list of edges.
    pub fn with_edges(
        edges: Vec<(VertexId, VertexId, Weight)>,
        config: DynamicMinCutConfig,
    ) -> crate::Result<Self> {
        let inner = crate::MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build()?;
        Ok(Self {
            inner,
            config,
            cached_cut: None,
            epoch: 0,
            last_full_epoch: 0,
            incremental_count: 0,
            dirty: true,
            cut_edge_set: HashSet::new(),
            source_side_set: HashSet::new(),
        })
    }

    /// Get the current epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Get the epoch of the last full recomputation.
    pub fn last_full_epoch(&self) -> u64 {
        self.last_full_epoch
    }

    /// Number of incremental updates since last full recomputation.
    pub fn incremental_count(&self) -> u64 {
        self.incremental_count
    }

    /// Whether the cached cut is known to be stale.
    pub fn is_stale(&self) -> bool {
        self.dirty
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

    /// Get the current min-cut value from the underlying engine.
    pub fn min_cut_value(&self) -> f64 {
        self.inner.min_cut_value()
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &DynamicMinCutConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Core: canonical cut computation
    // -----------------------------------------------------------------------

    /// Compute (or return cached) the canonical cut.
    ///
    /// Triggers a full recomputation if the cache is stale or if the
    /// staleness threshold has been exceeded.
    pub fn canonical_cut(&mut self) -> Option<SourceAnchoredCut> {
        if !self.dirty && self.cached_cut.is_some() {
            // Check staleness threshold
            if self.config.staleness_threshold > 0
                && self.incremental_count >= self.config.staleness_threshold
            {
                self.force_recompute();
            } else {
                return self.cached_cut.clone();
            }
        }

        self.force_recompute();
        self.cached_cut.clone()
    }

    /// Force a full recomputation of the canonical cut.
    pub fn force_recompute(&mut self) {
        let graph = self.inner.graph();
        let g = graph.read();
        let result = canonical_mincut(&g, &self.config.canonical_config);
        drop(g);

        if let Some(ref cut) = result {
            self.rebuild_cache(cut);
        } else {
            self.cut_edge_set.clear();
            self.source_side_set.clear();
        }

        self.cached_cut = result;
        self.dirty = false;
        self.last_full_epoch = self.epoch;
        self.incremental_count = 0;
    }

    /// Generate a witness receipt for the current canonical cut.
    pub fn receipt(&mut self) -> Option<SourceAnchoredReceipt> {
        let cut = self.canonical_cut()?;
        Some(make_receipt(&cut, self.epoch))
    }

    // -----------------------------------------------------------------------
    // Edge mutations
    // -----------------------------------------------------------------------

    /// Insert an edge and incrementally update the canonical cut.
    ///
    /// If both endpoints are on the same side of the current cut, the
    /// cut value is unchanged and no recomputation is needed.
    /// If the edge crosses the cut, the cut value may increase and
    /// we must recompute.
    pub fn add_edge(
        &mut self,
        u: VertexId,
        v: VertexId,
        weight: Weight,
    ) -> crate::Result<f64> {
        let val = self.inner.insert_edge(u, v, weight)?;
        self.epoch += 1;

        if self.cached_cut.is_some() && !self.dirty {
            let u_in_source = self.source_side_set.contains(&u);
            let v_in_source = self.source_side_set.contains(&v);

            if u_in_source == v_in_source {
                // Both on same side -- cut value unchanged.
                // The new edge doesn't cross the cut.
                self.incremental_count += 1;
            } else {
                // Edge crosses the cut -- cut value increases.
                // The cached cut may no longer be minimum.
                self.dirty = true;
            }
        } else {
            self.dirty = true;
        }

        Ok(val)
    }

    /// Remove an edge and incrementally update the canonical cut.
    ///
    /// If the edge is not in the current cut set, the cut value is
    /// unchanged. If it is in the cut set, the cut value decreases
    /// and we must recompute.
    pub fn remove_edge(
        &mut self,
        u: VertexId,
        v: VertexId,
    ) -> crate::Result<f64> {
        let val = self.inner.delete_edge(u, v)?;
        self.epoch += 1;

        if self.cached_cut.is_some() && !self.dirty {
            let edge_key = normalize_edge(u, v);
            if self.cut_edge_set.contains(&edge_key) {
                // Edge is in the cut -- must recompute.
                self.dirty = true;
            } else {
                // Edge not in the cut -- cut unchanged.
                self.incremental_count += 1;
            }
        } else {
            self.dirty = true;
        }

        Ok(val)
    }

    /// Apply a batch of edge mutations and then recompute if needed.
    ///
    /// This is more efficient than individual mutations when many
    /// edges change at once, because we defer the recomputation
    /// decision until all mutations are applied.
    pub fn apply_batch(
        &mut self,
        mutations: &[EdgeMutation],
    ) -> crate::Result<()> {
        let mut needs_recompute = self.dirty;

        for mutation in mutations {
            match mutation {
                EdgeMutation::Add(u, v, w) => {
                    self.inner.insert_edge(*u, *v, *w)?;
                    self.epoch += 1;

                    if !needs_recompute && self.cached_cut.is_some() {
                        let u_in = self.source_side_set.contains(u);
                        let v_in = self.source_side_set.contains(v);
                        if u_in != v_in {
                            needs_recompute = true;
                        }
                    }
                }
                EdgeMutation::Remove(u, v) => {
                    self.inner.delete_edge(*u, *v)?;
                    self.epoch += 1;

                    if !needs_recompute && self.cached_cut.is_some() {
                        let edge_key = normalize_edge(*u, *v);
                        if self.cut_edge_set.contains(&edge_key) {
                            needs_recompute = true;
                        }
                    }
                }
            }
        }

        self.incremental_count += mutations.len() as u64;

        if needs_recompute {
            self.dirty = true;
        }

        // Check staleness threshold
        if self.config.staleness_threshold > 0
            && self.incremental_count >= self.config.staleness_threshold
        {
            self.force_recompute();
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Rebuild the fast-lookup caches from a canonical cut result.
    fn rebuild_cache(&mut self, cut: &SourceAnchoredCut) {
        self.cut_edge_set.clear();
        for &(u, v) in &cut.cut_edges {
            self.cut_edge_set.insert(normalize_edge(u, v));
        }

        self.source_side_set.clear();
        for &v in &cut.side_vertices {
            self.source_side_set.insert(v);
        }
    }
}

impl Default for DynamicMinCut {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Normalize an edge so (u, v) always has u < v.
fn normalize_edge(u: VertexId, v: VertexId) -> (VertexId, VertexId) {
    if u <= v { (u, v) } else { (v, u) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
