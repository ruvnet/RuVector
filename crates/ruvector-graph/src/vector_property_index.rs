//! Vector-keyed property lookup for graph nodes via RaBitQ codes.
//!
//! Phase 1 / item #2 of the RaBitQ-integration roadmap: lets graph callers
//! ask "find nodes whose vector property is closest to query" without
//! standing up a separate index crate. The index lives alongside the
//! existing property table and is built from the same data the graph
//! already stores — it never owns or mutates the property values; it
//! reads them once at build time and keeps a 1-bit code per node plus
//! a parallel `Vec<NodeId>` for the array-position ↔ NodeId mapping.
//!
//! Memory: a `RabitqPlusIndex` holds the original f32 vectors **and** the
//! 1-bit codes (it needs the originals for the rerank rerank). The 1-bit
//! codes alone are `dim/8` bytes per node — at `dim = 768` that's 96 B vs
//! 3 072 B for an f32 baseline, the 1/16 ratio the acceptance test asks
//! for. The plus-index reports both back through `codes_bytes()` /
//! `original_bytes()` accessors below so callers can verify the ratio
//! independently of the rerank-storage choice.
//!
//! Determinism: `(seed, dim, vectors)` → bit-identical `RabitqPlusIndex`
//! state across runs and platforms (ADR-154 contract upheld by the
//! underlying rabitq crate). Two `VectorPropertyIndex::build` calls with
//! the same seed on the same `GraphDB` must therefore produce
//! byte-identical packed codes — verified in `tests/vector_property_index.rs`.

use crate::error::{GraphError, Result};
use crate::graph::GraphDB;
use crate::types::{NodeId, PropertyValue};
use ruvector_rabitq::{AnnIndex, RabitqPlusIndex};

/// Configuration for building a [`VectorPropertyIndex`].
///
/// The defaults (`seed = 42`, `rerank_factor = 20`) match the acceptance
/// test in the RaBitQ-integration roadmap: at `rerank_factor = 20` the
/// `RabitqPlusIndex` reports recall@10 ≥ 0.95 against brute force on
/// 100 k × 768-d data.
#[derive(Clone, Debug)]
pub struct VectorPropertyIndexConfig {
    /// Seed for the random rotation used by the underlying RaBitQ index.
    pub seed: u64,
    /// Number of 1-bit candidates to rerank per `k` returned. Higher =
    /// higher recall at the cost of one extra exact L2² per candidate.
    pub rerank_factor: u32,
}

impl Default for VectorPropertyIndexConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            rerank_factor: 20,
        }
    }
}

/// Vector-keyed property lookup over the nodes of a [`GraphDB`].
///
/// Built by [`VectorPropertyIndex::build`] from a graph + property name.
/// At query time, [`VectorPropertyIndex::knn`] returns the `k` `NodeId`s
/// whose chosen vector property is closest to the query (squared-L2
/// distance via 1-bit RaBitQ scan + exact rerank).
///
/// The index does **not** track graph mutations — once built it is a
/// snapshot. Callers that need to reflect inserts/deletes should rebuild.
pub struct VectorPropertyIndex {
    inner: RabitqPlusIndex,
    /// Map from rabitq row position (0..n) to the `NodeId` that lived in
    /// the property table at build time. `inner.add(pos, ..)` was called
    /// with the same `pos`, so `result.id == position-in-this-vec`.
    node_id_for_pos: Vec<NodeId>,
    /// Property name this index was built from (kept for diagnostics).
    property: String,
    /// Vector dimension the index expects on `knn`.
    dim: usize,
}

impl VectorPropertyIndex {
    /// Build an index over `graph`'s `property`.
    ///
    /// Reads every node's property table; nodes that lack the property,
    /// or whose value is not a [`PropertyValue::FloatArray`], are
    /// silently skipped. The first vector encountered fixes `dim`; any
    /// later vector with a different length is rejected with
    /// [`GraphError::InvalidEmbedding`].
    ///
    /// Build is O(n · dim) for the rotation+pack step and allocates one
    /// `NodeId` clone per indexed node (the parallel `node_id_for_pos`
    /// vec). The underlying [`RabitqPlusIndex::add`] is amortised O(D).
    ///
    /// Returns [`GraphError::InvalidInput`] if `graph` has no nodes with
    /// a usable vector property of that name.
    pub fn build(
        graph: &GraphDB,
        property: &str,
        config: VectorPropertyIndexConfig,
    ) -> Result<Self> {
        // Snapshot node ids in a deterministic order. DashMap iteration
        // order is shard-dependent so we sort by NodeId to make the
        // `(seed, graph)` → byte-identical-codes contract hold across
        // runs and platforms regardless of insertion order.
        let mut ids = graph.node_ids();
        ids.sort();

        // First pass: collect (NodeId, vector) for nodes that actually
        // carry a `FloatArray` under the requested property. We walk
        // through the sorted ids so the resulting position-to-id map is
        // deterministic.
        let mut pairs: Vec<(NodeId, Vec<f32>)> = Vec::new();
        let mut dim: Option<usize> = None;
        for id in ids {
            let Some(node) = graph.get_node(&id) else {
                continue;
            };
            let Some(value) = node.get_property(property) else {
                continue;
            };
            let PropertyValue::FloatArray(vec) = value else {
                continue;
            };
            if vec.is_empty() {
                continue;
            }
            match dim {
                None => dim = Some(vec.len()),
                Some(d) if d == vec.len() => {}
                Some(d) => {
                    return Err(GraphError::InvalidEmbedding(format!(
                        "vector dimension mismatch on node {id}: expected {d}, got {}",
                        vec.len()
                    )));
                }
            }
            pairs.push((id, vec.clone()));
        }

        let Some(dim) = dim else {
            return Err(GraphError::InvalidInput(format!(
                "no nodes carry a `FloatArray` property named `{property}`"
            )));
        };

        if pairs.is_empty() {
            return Err(GraphError::InvalidInput(format!(
                "property `{property}` produced 0 indexable vectors"
            )));
        }

        let rerank_factor = config.rerank_factor.max(1) as usize;
        let mut inner = RabitqPlusIndex::new(dim, config.seed, rerank_factor);

        let n = pairs.len();
        let mut node_id_for_pos: Vec<NodeId> = Vec::with_capacity(n);
        for (pos, (node_id, vector)) in pairs.into_iter().enumerate() {
            // pos is the row index inside the rabitq SoA — the search()
            // path returns `id` field == this `pos`, and we map it back
            // through node_id_for_pos[pos].
            inner.add(pos, vector)?;
            node_id_for_pos.push(node_id);
        }

        Ok(Self {
            inner,
            node_id_for_pos,
            property: property.to_string(),
            dim,
        })
    }

    /// Find the `k` `NodeId`s whose property vector is closest to `query`.
    ///
    /// Returns pairs of `(NodeId, squared_L2_distance)` sorted ascending
    /// by distance (closest first). Identical to the `RabitqPlusIndex`
    /// score semantics — these are *exact* squared-L2 distances on the
    /// reranked candidates, not the 1-bit estimator.
    pub fn knn(&self, query: &[f32], k: usize) -> Result<Vec<(NodeId, f32)>> {
        if query.len() != self.dim {
            return Err(GraphError::InvalidEmbedding(format!(
                "query dim {} != index dim {}",
                query.len(),
                self.dim
            )));
        }
        if k == 0 {
            return Ok(Vec::new());
        }
        let results = self.inner.search(query, k)?;
        let mut out = Vec::with_capacity(results.len());
        for r in results {
            // r.id is the row position we passed into `inner.add` above.
            let pos = r.id;
            if let Some(node_id) = self.node_id_for_pos.get(pos) {
                out.push((node_id.clone(), r.score));
            }
        }
        Ok(out)
    }

    /// Number of indexed nodes.
    pub fn len(&self) -> usize {
        self.node_id_for_pos.len()
    }

    /// `true` iff the index has zero entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimension the index was built with.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Property name the index was built from.
    pub fn property(&self) -> &str {
        &self.property
    }

    /// Bytes used by the 1-bit codes alone (rotation matrix + packed
    /// codes + cos LUT, no f32 originals). Use this side of the
    /// memory accounting when comparing to an f32 baseline of
    /// `n * dim * 4` bytes — at `dim ≥ 64` this should be ≤ 1/16 of the
    /// baseline plus a constant rotation overhead.
    pub fn codes_bytes(&self) -> usize {
        // RabitqPlusIndex::memory_bytes() is `inner.memory_bytes() + 24
        // + originals_flat.len() * 4`; we want only the codes-side cost.
        // The inner RabitqIndex::memory_bytes() = rotation.bytes() +
        // codes_bytes(). We can't reach that directly without exposing
        // accessors on the rabitq crate, so we compute it as the
        // total-minus-originals — equivalent and uses only the public
        // surface.
        self.inner
            .memory_bytes()
            .saturating_sub(self.original_bytes() + 24)
    }

    /// Bytes used by the f32 originals stored for rerank
    /// (`n * dim * 4`). Reported separately so callers can pick which
    /// side to compare against the f32 baseline.
    pub fn original_bytes(&self) -> usize {
        self.node_id_for_pos.len() * self.dim * 4
    }
}
