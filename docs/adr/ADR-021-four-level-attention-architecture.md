# ADR-021: Four-Level Attention Architecture

**Status**: Proposed
**Date**: 2026-02-08
**Parent**: ADR-018 Visual World Model, ADR-014 Coherence Engine, ADR-019 Three-Cadence Loop Architecture
**Author**: System Architecture Team
**SDK**: Claude-Flow

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-08 | Architecture Team | Initial four-level attention architecture |

---

## Abstract

Attention is the selection function in the visual world model. It determines what gets rendered, what gets updated, and what gets committed. This ADR defines attention at four distinct levels, each operating at a different cadence and serving a different purpose. The four levels -- view, temporal, semantic, and write -- form a pipeline from "what is visible" through "what is relevant" to "what is allowed." The fourth level, write attention, bridges the attention layer and the coherence gate defined in ADR-020.

---

## 1. Context and Motivation

### 1.1 Why Four Levels

A naive attention mechanism selects "important" things from a flat pool. This does not work for a visual world model because importance depends on context, and context operates at multiple timescales:

- **Spatial context** (what is in view right now) changes at 30-60 Hz with camera motion.
- **Temporal context** (what time range matters) changes at 2-10 Hz with user scrubbing or event queries.
- **Semantic context** (what objects or concepts matter) changes at human-interaction timescales, driven by text queries or task intent.
- **Write context** (what updates are safe to commit) changes at 0.1-1 Hz based on structural coherence.

Collapsing these into a single attention mechanism forces the system to recompute everything at the fastest rate. Separating them allows each to run at its natural cadence, matching the three-cadence loop architecture in ADR-019.

### 1.2 Existing Crate Support

The RuVector ecosystem already provides attention primitives:
- `ruvector-attention`: TopologyGatedAttention, MoEAttention, DiffusionAttention, FlashAttention
- `ruvector-attention-wasm`: Browser-side attention for client rendering
- `ruvector-vwm`: Visual world model with coherence modules
- `ruvector-mincut`: Graph cut cost as a write-gating signal

This ADR composes these existing primitives into a four-level architecture with explicit interfaces.

---

## 2. Decision

Implement attention at four levels with strict ownership by loop cadence.

### 2.1 Level 1: View Attention

**What it selects**: Which spatial blocks matter for the current camera pose and fovea region.

**Cadence**: Per-frame, 30-60 Hz (fast loop).

**Mechanism**:
1. Camera pose defines a view frustum in world space.
2. Spatial index query returns candidate tile IDs intersecting the frustum.
3. Foveal weighting: tiles closer to gaze center receive higher priority. Tiles at periphery are eligible for lower LOD or skipping.
4. Budget enforcement: sort candidates by priority, select top-N to fill the per-frame Gaussian budget.

**Output**: An ordered list of `(tile_id, lod_tier, priority)` tuples forming the active block set for the current frame.

**Implementation**:
```rust
/// View attention: frustum + fovea selection
pub struct ViewAttention {
    /// Spatial index for frustum queries
    spatial_index: SpatialIndex,
    /// Foveal falloff parameters
    fovea_params: FoveaParams,
    /// Per-frame Gaussian budget
    frame_budget: u32,
}

impl ViewAttention {
    /// Select active blocks for current pose
    pub fn select(
        &self,
        pose: &CameraPose,
        lod_policy: &LodPolicy,
    ) -> Vec<ActiveBlock> {
        // 1. Frustum query
        let candidates = self.spatial_index.query_frustum(&pose.frustum());

        // 2. Foveal priority
        let mut scored: Vec<(TileId, f32, LodTier)> = candidates
            .iter()
            .map(|tile_id| {
                let center = self.spatial_index.tile_center(*tile_id);
                let angular_dist = pose.angular_distance_to(&center);
                let priority = self.fovea_params.weight(angular_dist);
                let lod = lod_policy.tier_for_distance(pose.distance_to(&center));
                (*tile_id, priority, lod)
            })
            .collect();

        // 3. Sort by priority (descending)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // 4. Budget enforcement
        let mut total_gaussians = 0u32;
        scored
            .into_iter()
            .take_while(|(tile_id, _, lod)| {
                let count = self.spatial_index.gaussian_count(*tile_id, *lod);
                total_gaussians += count;
                total_gaussians <= self.frame_budget
            })
            .map(|(tile_id, priority, lod)| ActiveBlock { tile_id, lod, priority })
            .collect()
    }
}
```

**Runs in**: Browser WASM (`ruvector-attention-wasm`, `ruvector-vwm-wasm`).

**Crate**: `ruvector-attention` :: `ViewAttention` module.

### 2.2 Level 2: Temporal Attention

**What it selects**: Which time slices are relevant for a scrub range or event query.

**Cadence**: 2-10 Hz (medium loop).

**Mechanism**:
1. User scrub position or event query defines a time window `[t_start, t_end]`.
2. Temporal index returns keyframe anchors within the window.
3. Relevance scoring: keyframes with higher delta activity (more Gaussians changed) receive higher weight.
4. Interpolation planning: select keyframe pairs for smooth interpolation, weighted by temporal distance to query time.

**Output**: An ordered list of `(keyframe_id, time, relevance_weight)` tuples, plus interpolation parameters for the current scrub position.

**Implementation**:
```rust
/// Temporal attention: time slice selection
pub struct TemporalAttention {
    /// Temporal index of keyframe anchors
    temporal_index: TemporalIndex,
    /// Maximum keyframes to hold active
    max_active_keyframes: usize,
}

impl TemporalAttention {
    /// Select relevant time slices for a query window
    pub fn select(
        &self,
        time_window: &TimeWindow,
        query_time: f64,
    ) -> TemporalSelection {
        // 1. Find keyframes in window
        let keyframes = self.temporal_index.query_range(
            time_window.start,
            time_window.end,
        );

        // 2. Score by delta activity and temporal proximity
        let mut scored: Vec<(KeyframeId, f64, f32)> = keyframes
            .iter()
            .map(|kf| {
                let time_dist = (kf.time - query_time).abs();
                let temporal_weight = 1.0 / (1.0 + time_dist as f32);
                let activity_weight = kf.delta_count as f32 / kf.total_gaussians as f32;
                let relevance = temporal_weight * 0.7 + activity_weight * 0.3;
                (kf.id, kf.time, relevance)
            })
            .collect();

        // 3. Sort by relevance
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // 4. Select top-N
        let active: Vec<_> = scored.into_iter()
            .take(self.max_active_keyframes)
            .collect();

        // 5. Compute interpolation parameters
        let interp = self.compute_interpolation(&active, query_time);

        TemporalSelection { active_keyframes: active, interpolation: interp }
    }
}
```

**Runs in**: Browser WASM for scrub interaction (`ruvector-attention-wasm`). Server-side for event queries.

**Crate**: `ruvector-attention` :: `TemporalAttention` module, `ruvector-temporal-tensor` for index.

### 2.3 Level 3: Semantic Attention

**What it selects**: Which objects and regions match a text query or task intent.

**Cadence**: On-demand, driven by user queries or agent task assignments.

**Mechanism**:
1. Text query is embedded using the same embedding model that produced entity embeddings.
2. Vector search against entity graph embeddings returns candidate entities ranked by similarity.
3. Spatial expansion: for each matched entity, include its spatial tile neighborhood.
4. Result: a set of entity IDs and tile IDs that represent "what the query is about."

**Output**: A set of `(entity_id, similarity_score)` pairs, plus expanded tile set for rendering.

**Implementation**:
```rust
/// Semantic attention: query-driven entity selection
pub struct SemanticAttention {
    /// Entity embedding index (HNSW-backed)
    entity_index: EntityEmbeddingIndex,
    /// Text-to-embedding model
    embedder: TextEmbedder,
    /// Spatial expansion radius (in tiles)
    expansion_radius: u32,
}

impl SemanticAttention {
    /// Select entities and tiles matching a text query
    pub fn select(
        &self,
        query: &str,
        top_k: usize,
        entity_graph: &EntityGraph,
    ) -> SemanticSelection {
        // 1. Embed query
        let query_embedding = self.embedder.embed(query);

        // 2. Vector search against entity embeddings
        let candidates = self.entity_index.search(&query_embedding, top_k);

        // 3. Spatial expansion
        let mut tile_set: HashSet<TileId> = HashSet::new();
        for (entity_id, _score) in &candidates {
            let entity_tiles = entity_graph.tiles_for_entity(*entity_id);
            for tile_id in entity_tiles {
                tile_set.insert(tile_id);
                // Expand to neighbors within radius
                let neighbors = entity_graph.tile_neighbors(tile_id, self.expansion_radius);
                tile_set.extend(neighbors);
            }
        }

        SemanticSelection {
            matched_entities: candidates,
            active_tiles: tile_set,
        }
    }
}
```

**Runs in**: Browser WASM for client-side entity search (`ruvector-vwm-wasm` with embedded HNSW). Server for full-index queries.

**Crate**: `ruvector-attention` :: `SemanticAttention` module, `ruvector-core` for HNSW search, `ruvector-graph` for entity lookups.

### 2.4 Level 4: Write Attention

**What it selects**: Which updates are allowed to commit to canonical state.

**Cadence**: 0.1-1 Hz (slow loop).

**Mechanism**:
1. Consume promotion candidates from the medium loop.
2. For each candidate batch, compute GNN identity confidence and structural coherence (from `ruvector-gnn`).
3. Compute graph cut cost delta using `ruvector-mincut`.
4. Apply the signal-to-decision mapping defined in ADR-020.
5. Output: Accept, Defer, Freeze, or Rollback per entity/batch.

**Output**: Coherence verdicts that gate which pending updates become canonical.

**Write attention is not traditional attention.** It does not weight inputs for a neural network forward pass. It weights updates for a state machine transition. The "attention score" is a binary gate (accept or not) derived from the continuous signals described in ADR-020. This is the bridge between the attention layer (Levels 1-3, which select what to show) and the coherence gate (which selects what to commit).

**Runs in**: Server-side, within the slow loop.

**Crate**: `ruvector-vwm` :: `governance_loop`, `ruvector-gnn`, `ruvector-mincut`, `ruvector-nervous-system`.

---

## 3. Composition: How the Four Levels Interact

### 3.1 Pipeline

The four levels form a narrowing pipeline:

```
                    All Gaussians in world model
                              │
                    ┌─────────▼──────────┐
                    │  Level 1: View     │  "What is visible?"
                    │  (30-60 Hz)        │  Frustum + fovea + budget
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Level 2: Temporal  │  "What time matters?"
                    │  (2-10 Hz)          │  Scrub range + keyframes
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Level 3: Semantic  │  "What does the query mean?"
                    │  (on-demand)        │  Text query + embeddings
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Level 4: Write    │  "Is this update safe?"
                    │  (0.1-1 Hz)        │  GNN + mincut + coherence
                    └────────────────────┘
```

Levels 1-3 are **read attention**: they select what to show. Level 4 is **write attention**: it selects what to commit. The read path narrows from "everything" to "this subset." The write path gates from "proposed changes" to "accepted changes."

### 3.2 Combined Query Flow

When a user issues a text query (e.g., "show me the red car"):

1. **Semantic attention** (Level 3) identifies the entity (red car) and its tile neighborhood.
2. **Temporal attention** (Level 2) selects keyframes where the red car is active.
3. **View attention** (Level 1) constrains to the current camera frustum, applying foveal priority to the car's tiles.
4. The renderer draws only the intersection of these three selections. Target: max 10% of total Gaussians.

Meanwhile, in the slow loop:
5. **Write attention** (Level 4) evaluates whether new observations of the red car should be committed, based on GNN identity confidence and graph cut cost.

### 3.3 Where Each Level Runs

| Level | Client (WASM) | Server | GPU |
|-------|--------------|--------|-----|
| View | Frustum query, foveal scoring | -- | Draw submission |
| Temporal | Scrub interaction, interpolation | Event query index | -- |
| Semantic | HNSW search on local entity cache | Full-index search | -- |
| Write | -- | GNN, mincut, coherence gate | GNN inference (optional) |

---

## 4. Integration with Existing Crates

| Crate | Attention Level | Usage |
|-------|----------------|-------|
| `ruvector-attention` | 1, 2, 3 | Core attention primitives: TopologyGatedAttention for view scoring, FlashAttention for efficient batch operations |
| `ruvector-attention-wasm` | 1, 2, 3 | Browser-side attention for client rendering and interaction |
| `ruvector-vwm` | 1, 2, 3, 4 | Visual world model hosts all four levels; governance module hosts Level 4 |
| `ruvector-vwm-wasm` | 1, 2, 3 | Browser-side VWM with embedded entity cache for semantic search |
| `ruvector-gnn` | 4 | Identity confidence and structural coherence signals |
| `ruvector-mincut` | 4 | Graph cut cost computation |
| `ruvector-nervous-system` | 4 | CoherenceGatedSystem for hysteresis and broadcast |
| `ruvector-core` | 3 | HNSW vector search for entity embeddings |
| `ruvector-graph` | 3, 4 | Entity graph adjacency, tile-to-entity mapping |
| `ruvector-temporal-tensor` | 2 | Temporal index and keyframe management |
| `cognitum-gate-kernel` | 4 | Tile-level coherence accumulation |

---

## 5. Consequences

### 5.1 Benefits

- **Rate-appropriate selection**: Each level runs at its natural cadence. View attention at 60 Hz does not carry the cost of semantic search. Write attention at 0.1 Hz does not block rendering.
- **Composable narrowing**: The pipeline naturally reduces the active set at each level, making the final render workload a small fraction of the total world model.
- **Unified vocabulary**: All four levels use the same tile/entity/block primitives from the RuVector data model. No impedance mismatch between levels.
- **Write attention as structural guard**: By naming the coherence gate as "attention level 4," it becomes a first-class part of the attention architecture rather than an afterthought governance layer.

### 5.2 Costs

- **Four codepaths to maintain**: Each level has distinct logic, crate dependencies, and test surfaces.
- **Interaction complexity**: The combined query flow (Section 3.2) requires careful ordering. A bug in Level 3 can cause Level 1 to render the wrong tiles.
- **WASM binary size**: Shipping attention primitives + HNSW + entity cache in the browser WASM increases bundle size. Mitigated by lazy loading and code splitting.

### 5.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Semantic attention returns too many tiles, exceeding frame budget | Medium | Low | View attention (Level 1) enforces hard budget regardless of upstream |
| Write attention blocks too aggressively, causing stale rendering | Low | Medium | Defer has a timeout; if deferred too long, auto-accept with degraded confidence flag |
| Temporal attention misses relevant keyframes during fast scrub | Medium | Low | Pre-fetch keyframe window wider than display window; interpolate from nearest |

---

## 6. Acceptance Tests

### Test A: View Budget Enforcement

With 500K total Gaussians and a frame budget of 50K, verify that view attention selects at most 50K Gaussians regardless of frustum size. Verify foveal center tiles are always included before peripheral tiles.

### Test B: Semantic Narrowing

Issue a text query for a specific entity in a scene with 100 entities. Verify semantic attention returns the correct entity in the top-3 results. Verify the expanded tile set covers the entity's spatial extent. Verify the rendered Gaussian count is under 10% of total.

### Test C: Temporal Interpolation

Scrub through a 30-second clip at full speed. Verify temporal attention selects appropriate keyframe pairs for each scrub position. Verify no visual discontinuities (frame-to-frame Gaussian position jumps under 1 pixel at 1080p).

### Test D: Write Gating

Propose a batch of entity updates where 3 entities have high confidence and 1 has low confidence. Verify write attention Accepts the 3 and Defers or Freezes the 1. Verify the accepted updates appear in the next fast loop render. Verify the deferred update does not.

---

## 7. References

- ADR-018: Visual World Model as a Bounded Nervous System
- ADR-019: Three-Cadence Loop Architecture
- ADR-020: GNN-to-Coherence-Gate Feedback Pipeline
- ADR-014: Coherence Engine Architecture
- `ruvector-attention`: Core attention module with TopologyGatedAttention, MoEAttention, FlashAttention
- `ruvector-attention-wasm`: Browser-side attention primitives
