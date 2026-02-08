# ADR-022: Query-First Rendering Pattern

**Status**: Proposed
**Date**: 2026-02-08
**Parent**: ADR-018 Visual World Model, ADR-021 Four-Level Attention Architecture, ADR-019 Three-Cadence Loop Architecture
**Author**: System Architecture Team
**SDK**: Claude-Flow

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-08 | Architecture Team | Initial query-first rendering pattern |

---

## Abstract

Traditional rendering pipelines process all geometry and then search the result. In a world model, most Gaussians are irrelevant to any given query or view. The breakthrough is: retrieval decides what to render, not the renderer. This ADR defines the query-first rendering pattern where user intent hits the RuVector entity graph first, attention selects the active subset, and WebGPU renders only that subset. This turns rendering from a compute problem into a retrieval problem.

---

## 1. Context and Motivation

### 1.1 The Traditional Pipeline

A standard rendering pipeline operates as:

```
All Geometry  ──►  Vertex Processing  ──►  Rasterization  ──►  Fragment Shading  ──►  Output
```

Optimization techniques (frustum culling, occlusion culling, LOD) reduce workload, but the pipeline still starts from "all geometry" and subtracts. The fundamental assumption is: geometry is the input, pixels are the output.

For Gaussian splatting, the equivalent pipeline is:

```
All Gaussians  ──►  Project to Screen  ──►  Sort  ──►  Alpha Blend  ──►  Output
```

At 500K Gaussians, this is tractable. At 5M Gaussians (a multi-room scene over time), it is not. Sorting alone becomes the bottleneck.

### 1.2 The World Model Inversion

In a world model, the user has intent. They are looking at something, searching for something, scrubbing to a time, or asking a question. The intent defines what matters. Everything else is wasted compute.

The inversion is:

```
Intent  ──►  Retrieve  ──►  Select  ──►  Render (only the selected subset)
```

This is not incremental optimization of the traditional pipeline. It is a different pipeline. The starting point is not "all Gaussians" but "what does the user need to see?"

### 1.3 Why This Works for Gaussians

Gaussian splatting has a property that polygon meshes do not: each Gaussian is an independent primitive with a position, covariance, color, and opacity. There is no connectivity (no triangle strips, no mesh topology). This means any subset of Gaussians can be rendered without modifying the others. Subselection is free at the data structure level.

Combined with the entity graph in RuVector (which maps Gaussians to objects, objects to tracks, tracks to tiles), retrieval can produce a rendereable subset directly.

---

## 2. Decision

Adopt the query-first rendering pattern: **retrieve, select, render**.

### 2.1 The Pattern

```
┌─────────────────────────────────────────────────────────────────────┐
│                     QUERY-FIRST RENDERING                            │
│                                                                      │
│   User Intent                                                        │
│       │                                                              │
│       ▼                                                              │
│   ┌──────────────────┐                                               │
│   │  1. RETRIEVE     │  Hit RuVector entity graph                    │
│   │                  │  Return: entity IDs, track segments, tiles    │
│   └────────┬─────────┘                                               │
│            │                                                         │
│            ▼                                                         │
│   ┌──────────────────┐                                               │
│   │  2. SELECT       │  Four-level attention (ADR-021)               │
│   │                  │  Return: active block set, <=10% of total     │
│   └────────┬─────────┘                                               │
│            │                                                         │
│            ▼                                                         │
│   ┌──────────────────┐                                               │
│   │  3. RENDER       │  WebGPU draws ONLY the selected subset        │
│   │                  │  Project, sort, blend on GPU                   │
│   └──────────────────┘                                               │
│                                                                      │
│   NOT: Render ──► Search ──► Filter                                  │
│   YES: Retrieve ──► Select ──► Render                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Step 1: Retrieve

User intent is converted to a retrieval query against the RuVector entity graph:

| Intent Type | Query | RuVector Operation | Result |
|------------|-------|-------------------|--------|
| Camera view | Current pose + frustum | Spatial index tile query | Candidate tile IDs |
| Text search | "red car" | Embedding vector search (HNSW) | Entity IDs + tiles |
| Time scrub | `t = 14.3s` | Temporal index keyframe query | Keyframe IDs + deltas |
| Event query | "when did person enter?" | Event log search + entity tracks | Time ranges + entity IDs |
| Agent reasoning | Task-defined entity set | Direct entity ID lookup | Entity state vectors |

For agent reasoning (the last row), the query never produces pixels. The agent retrieves entity state vectors and reasons over them. Rendering is optional. This enables **intent-driven computation where agents reason without ever drawing**.

**Implementation**:

```rust
/// Query-first retrieval from RuVector entity graph
pub struct QueryRetriever {
    /// Spatial index for frustum queries
    spatial_index: SpatialIndex,
    /// HNSW embedding index for semantic search
    embedding_index: EntityEmbeddingIndex,
    /// Temporal index for time queries
    temporal_index: TemporalIndex,
    /// Entity graph for relationship traversal
    entity_graph: EntityGraph,
}

impl QueryRetriever {
    /// Retrieve entities and tiles matching user intent
    pub fn retrieve(&self, intent: &UserIntent) -> RetrievalResult {
        match intent {
            UserIntent::View { pose } => {
                let tiles = self.spatial_index.query_frustum(&pose.frustum());
                RetrievalResult::Spatial { tiles }
            }
            UserIntent::TextSearch { query, top_k } => {
                let embedding = self.embed(query);
                let entities = self.embedding_index.search(&embedding, *top_k);
                let tiles = self.expand_to_tiles(&entities);
                RetrievalResult::Semantic { entities, tiles }
            }
            UserIntent::TimeScrub { time, window } => {
                let keyframes = self.temporal_index.query_range(
                    time - window / 2.0,
                    time + window / 2.0,
                );
                RetrievalResult::Temporal { keyframes }
            }
            UserIntent::AgentReason { entity_ids } => {
                let states = self.entity_graph.get_states(entity_ids);
                RetrievalResult::StateOnly { states }
                // No tiles needed -- agent does not render
            }
        }
    }
}
```

**Client-side retrieval**: `ruvector-vwm-wasm` embeds a compact HNSW index and spatial index in the browser. This enables sub-millisecond retrieval without a server round-trip for the most common queries (view, text search over local cache).

### 2.3 Step 2: Select

The retrieval result feeds into the four-level attention pipeline (ADR-021):

1. **View attention** constrains to camera frustum and foveal priority.
2. **Temporal attention** selects relevant keyframes.
3. **Semantic attention** narrows to query-matched entities.
4. **Write attention** (slow loop) determines which updates are committed.

The output is the **active block set**: the precise set of Gaussian blocks to render.

**Budget target**: The active block set should contain at most 10% of total Gaussians for a typical query. For highly focused queries (e.g., "show me person #42"), it may be under 1%.

```rust
/// Select active blocks from retrieval results using attention pipeline
pub fn select_active_blocks(
    retrieval: &RetrievalResult,
    view_attention: &ViewAttention,
    temporal_attention: &TemporalAttention,
    semantic_attention: &SemanticAttention,
    pose: &CameraPose,
    time: f64,
    budget: u32,
) -> ActiveBlockSet {
    // Compose attention levels as narrowing filters
    let view_blocks = view_attention.select(pose, &retrieval.lod_policy());
    let temporal_blocks = temporal_attention.select(
        &retrieval.time_window(),
        time,
    );
    let semantic_blocks = match retrieval {
        RetrievalResult::Semantic { entities, tiles } => {
            Some(semantic_attention.select_from_entities(entities, tiles))
        }
        _ => None,
    };

    // Intersect: only blocks that pass all applicable levels
    let mut candidates: Vec<ActiveBlock> = view_blocks
        .iter()
        .filter(|b| temporal_blocks.contains_tile(b.tile_id))
        .filter(|b| semantic_blocks.as_ref().map_or(true, |s| s.contains_tile(b.tile_id)))
        .cloned()
        .collect();

    // Enforce budget
    candidates.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
    candidates.truncate(budget as usize);

    ActiveBlockSet { blocks: candidates }
}
```

### 2.4 Step 3: Render

WebGPU renders only the active block set:

1. Bind active Gaussian blocks to GPU buffers.
2. Project Gaussians to screen space (compute shader).
3. Sort for alpha blending (GPU radix sort on active set only).
4. Rasterize (tile-based splat rasterization).
5. Output frame.

Because the active set is small (target: max 50K Gaussians from 500K total), GPU work is reduced proportionally. Sort complexity drops from O(N log N) on 500K to O(N log N) on 50K -- a 10x reduction in the dominant cost.

---

## 3. Performance Analysis

### 3.1 Benchmark Targets

| Metric | Traditional Pipeline (500K) | Query-First (50K active) | Reduction |
|--------|---------------------------|--------------------------|-----------|
| Projection | 500K Gaussian projections | 50K projections | 90% |
| Sort | O(500K log 500K) | O(50K log 50K) | ~87% |
| Rasterization | 500K splats | 50K splats | 90% |
| Bandwidth (streaming) | ~16 MB keyframe | ~1.6 MB active subset | 90% |
| Memory (GPU) | 500K * 32B = 16 MB | 50K * 32B = 1.6 MB | 90% |

### 3.2 Query Latency Targets

| Operation | Target | Implementation |
|-----------|--------|---------------|
| Frustum query (spatial index) | < 1 ms | `ruvector-vwm-wasm` in-browser spatial index |
| Text search (HNSW) | < 5 ms | `ruvector-vwm-wasm` with 10K entity cache |
| Temporal keyframe lookup | < 1 ms | `ruvector-temporal-tensor` B-tree index |
| Attention composition | < 2 ms | Intersection of pre-sorted block lists |
| **Total retrieve + select** | **< 10 ms** | All client-side for cached scenes |
| GPU render (50K Gaussians) | < 8 ms | WebGPU compute + raster pipeline |
| **Total frame (retrieve + select + render)** | **< 18 ms** | **55+ fps** |

### 3.3 Search-and-Highlight Latency

The key interactive benchmark from ADR-018 Test E: "search and highlight object" queries under 100 ms.

| Step | Time |
|------|------|
| Text embedding (WASM) | ~20 ms |
| HNSW search (WASM, 10K entities) | ~5 ms |
| Tile expansion | ~2 ms |
| Attention selection | ~3 ms |
| GPU render of highlighted subset | ~8 ms |
| Highlight overlay | ~2 ms |
| **Total** | **~40 ms** |

Well within the 100 ms target.

---

## 4. Benefits

### 4.1 Bandwidth

Only stream the relevant subset to the client. A text query for "person at desk" streams the desk region, not the entire floor. For a 500K Gaussian scene, a focused query streams ~50K Gaussians (1.6 MB) instead of the full keyframe (16 MB). Over a bandwidth-constrained connection (5 Mbps), this is the difference between 25 ms and 250 ms for initial load.

### 4.2 Latency

Search-and-highlight in under 100 ms (see benchmarks above). Traditional approach: render all, run object detection on rendered frame, highlight. This takes 200+ ms and requires the full scene to be rendered first. Query-first: search the entity graph, render only the match. The search is faster than the render.

### 4.3 Privacy

Render only what the user is authorized to see. The entity graph carries per-entity permission tags (from ADR-018 Section 8). The retrieval step can filter entities by permission before any Gaussians reach the GPU. A user without clearance for zone B never receives zone B Gaussians -- they are not rendered and then hidden, they are never transmitted.

```rust
/// Privacy-filtered retrieval
pub fn retrieve_with_permissions(
    &self,
    intent: &UserIntent,
    permissions: &PermissionSet,
) -> RetrievalResult {
    let unfiltered = self.retrieve(intent);
    unfiltered.filter_entities(|entity_id| {
        let entity = self.entity_graph.get(entity_id);
        permissions.allows_access(&entity.privacy_tag)
    })
}
```

### 4.4 Cost

90% reduction in GPU work for typical queries. For cloud-rendered scenarios, this translates directly to 90% reduction in GPU cost per frame. For client-side rendering, it means the same scene runs on weaker GPUs (laptops, tablets, phones).

### 4.5 Agent Reasoning Without Drawing

The most radical consequence: agents that consume the world model do not need a renderer. An agent that answers "how many people are in the room?" retrieves entity states, counts entities with type "person," and returns. No Gaussians are projected, sorted, or blended. The world model is the API surface for both humans (who see pixels) and agents (who see entity states).

```rust
/// Agent queries world model without rendering
pub fn agent_query(
    retriever: &QueryRetriever,
    question: &str,
) -> AgentAnswer {
    let intent = UserIntent::AgentReason {
        entity_ids: retriever.entities_matching(question),
    };
    let result = retriever.retrieve(&intent);

    match result {
        RetrievalResult::StateOnly { states } => {
            // Agent reasons over entity states directly
            // No GPU, no rendering, no pixels
            agent_reason(question, &states)
        }
        _ => unreachable!(),
    }
}
```

---

## 5. Role of RuVector WASM for Client-Side Retrieval

### 5.1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      BROWSER                             │
│                                                          │
│  ┌──────────────────────┐   ┌─────────────────────────┐ │
│  │  ruvector-vwm-wasm   │   │    WebGPU Pipeline      │ │
│  │                      │   │                          │ │
│  │  ┌────────────────┐  │   │  Bind ──► Project ──►   │ │
│  │  │ Spatial Index   │  │──►│  Sort ──► Rasterize ──► │ │
│  │  │ HNSW (10K)     │  │   │  Output                 │ │
│  │  │ Temporal Index  │  │   │                          │ │
│  │  │ Entity Cache    │  │   └─────────────────────────┘ │
│  │  └────────────────┘  │                                │
│  │                      │                                │
│  │  retrieve() ──► select() ──► active block set         │
│  └──────────────────────┘                                │
│                                                          │
│  ┌──────────────────────────────────────────────────────┐│
│  │  ruvector-attention-wasm                              ││
│  │  View attention + Temporal attention (WASM)           ││
│  └──────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
         │
         │ (fetch missing blocks from server)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                      SERVER                              │
│                                                          │
│  Full RuVector entity graph + tile store                 │
│  GNN refinement (slow loop)                              │
│  Full HNSW index (all entities)                          │
│  Keyframe + delta storage                                │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Client-Side Capabilities

`ruvector-vwm-wasm` provides:
- **Spatial index**: R-tree or grid-based, for frustum queries. Updated when new tiles are streamed.
- **HNSW index**: Compact, containing embeddings for entities in the current scene cache (up to ~10K entities). Enables sub-5ms text search in the browser.
- **Temporal index**: B-tree over keyframe timestamps for the cached time range.
- **Entity cache**: Recent entity states and relationships, sufficient for semantic attention without server round-trip.

The client-side index is a subset of the full server-side index. Cache misses (e.g., querying for entities not in the local cache) fall through to the server.

### 5.3 Cache Coherence

The client cache is populated by the streaming protocol (ADR-018 Section 6). When the server publishes a keyframe or delta packet, the client-side spatial, HNSW, and temporal indices are updated incrementally. Cache invalidation follows the coherence gate verdicts: when the slow loop issues a Rollback on an entity, the client evicts the stale state.

---

## 6. Consequences

### 6.1 Benefits

- **10x reduction in GPU work** for typical queries (render 10% of total Gaussians)
- **Sub-100ms interactive search** enabled by client-side HNSW in WASM
- **Privacy by architecture**: unauthorized entities never reach the GPU
- **Agent-friendly**: world model is queryable without rendering
- **Bandwidth-efficient**: stream only relevant subsets
- **Scales to large scenes**: 5M Gaussians behaves like 50K at render time

### 6.2 Costs

- **Client-side index maintenance**: WASM binary includes spatial, HNSW, and temporal indices. Memory overhead: ~5-10 MB for 10K cached entities.
- **Cache miss latency**: Queries that miss the local cache require a server round-trip (~50-200 ms depending on network). Mitigated by predictive pre-fetching based on camera trajectory.
- **Retrieval accuracy**: If the entity graph has incorrect embeddings or stale tracks, retrieval returns wrong entities. Mitigated by write attention (ADR-021 Level 4) ensuring only coherent updates are committed.

### 6.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Retrieval misses relevant entities | Medium | Medium | Fall back to broader spatial query; user can widen search |
| Client HNSW index too large for low-memory devices | Low | Medium | Configurable cache size; degrade to server-only search |
| Cache coherence lag causes stale renders | Medium | Low | Client displays cache freshness indicator; force-refresh on stale |
| Agent over-relies on entity states without visual verification | Low | High | Agents can request render of specific entities for verification |

---

## 7. Acceptance Tests

### Test A: 10% Render Budget

Load a scene with 500K Gaussians. Issue a text query for a specific object. Verify the active block set contains fewer than 50K Gaussians. Verify the rendered output correctly shows the queried object.

### Test B: Sub-100ms Search and Highlight

In a browser with `ruvector-vwm-wasm` loaded, issue a text query. Measure wall-clock time from query submission to highlighted object visible on screen. Verify total time is under 100 ms for a 10K entity scene with warm cache.

### Test C: Privacy Enforcement

Create two permission sets: one with access to all entities, one with access to zone A only. Issue the same spatial query with each permission set. Verify the restricted set never receives Gaussians from zone B. Verify the Gaussian block IDs transmitted differ between the two cases.

### Test D: Agent Query Without Render

Issue an agent query ("count people in room"). Verify the retriever returns entity state vectors. Verify no GPU pipeline invocation occurs. Verify the agent produces a correct count from state vectors alone.

### Test E: Cache Miss Fallback

Clear the client-side HNSW cache. Issue a text query. Verify the query falls through to the server. Verify the result is returned within 200 ms over a typical network. Verify the client cache is populated with the returned entities for subsequent queries.

---

## 8. References

- ADR-018: Visual World Model as a Bounded Nervous System
- ADR-019: Three-Cadence Loop Architecture
- ADR-020: GNN-to-Coherence-Gate Feedback Pipeline
- ADR-021: Four-Level Attention Architecture
- ADR-014: Coherence Engine Architecture
- 3D Gaussian Splatting (Kerbl et al.) -- independent primitive property
- AirGS: Streaming optimization for Gaussian splatting
- `ruvector-vwm-wasm`: Browser-side visual world model
- `ruvector-attention-wasm`: Browser-side attention primitives
- `ruvector-core`: HNSW vector search
