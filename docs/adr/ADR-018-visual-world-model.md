# ADR-018: Visual World Model as a Bounded Nervous System

**Status**: Proposed
**Date**: 2026-02-07
**Parent**: ADR-001 RuVector Core Architecture, ADR-014 Coherence Engine, ADR-017 Temporal Tensor Compression
**Author**: System Architecture Team
**SDK**: Claude-Flow

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-07 | Architecture Team | Initial visual world model design proposal |

---

## Abstract

This ADR introduces a **visual world model** that operates as an always-on, mostly silent, event-driven bounded nervous system. RuVector serves as the unified substrate for visual primitives, semantics, structure, and audit. Rust handles all hot loops. WebGPU (or native GPU) is used only for final drawing, never for truth.

The primary representation is a spacetime tile store of 4D Gaussians, plus an entity graph, plus append-only lineage. The runtime produces a packed draw list under strict budgets. Updates and actions are promoted only through coherence gating.

---

## 1. Context and Motivation

### 1.1 Key Terms

- **4D Gaussian Splatting**: A dynamic scene representation where 3D Gaussian primitives vary over time. Introduces a deformation model so Gaussians can move and deform while still rendering fast.
- **World Model**: A persistent, queryable memory of entities, relations, trajectories, constraints, and confidence signals. RuVector is the world model substrate.
- **WebGPU**: Modern browser GPU API designed to map efficiently to contemporary native GPU backends.

### 1.2 25-Year Target Behavior

The system behaves like a visual nervous system:

1. **Perception loop** is continuous and cheap -- converts sensors into sparse events, not frames as product
2. **Memory loop** is persistent and structured -- stores what exists, where it is, how it changed, with provenance
3. **Prediction loop** is short-horizon and bounded -- predicts transitions and expected changes, not fantasies
4. **Action loop** is gated -- triggers only when coherence, permissions, and budgets align

What makes it 25-year class is not photorealism. It is durable, bounded, explainable operation that can run everywhere.

---

## 2. Decision

Build a visual world model with three strict loops.

### 2.1 Loop 1: Render Loop

- **Inputs**: Pose, time, display budget policy
- **Process**: Query hot set from RuVector -> Build packed draw list under per-screen-tile budgets -> Submit to WebGPU or native GPU -> Record metrics
- **Outputs**: Frame, draw list trace and budget telemetry

### 2.2 Loop 2: World Model Update Loop

- **Inputs**: Sensor events, tracks, keyframes, splat deltas
- **Process**: Write pending tile deltas -> Update entity graph -> Append lineage events -> Queue promotion candidates
- **Outputs**: Pending updates, candidate promotions with evidence pointers

### 2.3 Loop 3: Governance Loop

- **Inputs**: Disagreement, drift, sensor confidence, budget pressure, permissions
- **Process**: Run coherence gate -> Promote or defer updates -> Issue rollback when needed -> Adjust LOD and budgets -> Authorize or deny actions
- **Outputs**: Promotion decisions, updated policies and budgets, signed action allowances

---

## 3. Data Model in RuVector

| Collection | Contents |
|-----------|----------|
| `tiles` | Partitioned by time bucket, spatial cell, and LOD |
| `primitive_blocks` | Binary packed blocks with quant tiers, checksums, decode descriptors |
| `entities` | Stable IDs, embeddings, attributes, state |
| `edges` | Typed relations: adjacency, containment, continuity, causality |
| `lineage_events` | Append-only event log with provenance and rollback links |
| `metrics` | Frame time, hot set size, overflow counts, coherence scores |

---

## 4. Packed Draw List Protocol

### 4.1 Structure

- **Header**: Epoch, sequence, budget profile ID, checksum
- **Commands**:
  - `TileBind(tile_id, block_ref, quant_tier)`
  - `SetBudget(screen_tile_id, max_gaussians, max_overdraw)`
  - `DrawBlock(block_ref, sort_key, opacity_mode)`
  - `End`

### 4.2 Rule

The renderer never queries RuVector directly. It only consumes draw lists. This preserves determinism and makes replay possible.

---

## 5. Coherence Gate Policy

### 5.1 Inputs

| Input | Description |
|-------|-------------|
| Per-tile disagreement score | Measures consistency within tile |
| Per-entity continuity score | Tracks identity stability |
| Sensor confidence and freshness | Trust signal from input |
| Budget pressure | Resource constraints |
| Permission context | Authorization scope |

### 5.2 Outputs

| Output | Behavior |
|--------|----------|
| `accept` | Promote pending deltas to canonical state |
| `defer` | Keep pending, request more evidence |
| `freeze` | Stop promotions, render from last coherent state |
| `rollback` | Revert canonical state to prior lineage pointer |

---

## 6. Streaming Format (AirGS-Inspired)

### 6.1 Packet Types

- **Keyframe packet**: Full Gaussian set for a time anchor, quantized
- **Delta packet**: Updated Gaussians + active mask for short time interval
- **Semantic packet**: Object track updates, embeddings, links to Gaussian ID ranges

### 6.2 Bandwidth Estimates

| Parameter | Value |
|-----------|-------|
| Gaussians | 500K at 32 bytes each after quantization |
| Keyframe size | ~16 MB |
| Active mask | 500K bits = ~61 KB per frame |
| Active mask throughput | ~1.8 MB/s at 30 fps |

Requires multi-frame active masks + delta pruning + LOD selection to stay within bandwidth caps.

---

## 7. Rendering Options

| Option | Approach | Performance | Time to Ship | Maintainability | Cross-Platform |
|--------|----------|-------------|-------------|-----------------|----------------|
| A | Rust wgpu -> WASM | 5 | 3 | 4 | 4 |
| B | three.js WebGPURenderer | 3 | 5 | 4 | 5 |
| C | Hybrid (three.js UI + WebGPU pipeline) | 4 | 4 | 3 | 5 |

### 7.1 WebGPU Renderer Steps

1. Select active Gaussians for time *t* (active mask per frame)
2. Project Gaussians to screen space (center, conic matrix, screen-space ellipse)
3. Sort for alpha blending (back-to-front ordering)
4. Rasterize splats (tile-based or instanced quads)
5. Temporal interpolation (keyframes + linear interpolation)

---

## 8. Security and Governance

- Signed lineage events
- Capability-scoped tool access
- Policy allowlists for actions
- Replay protection using epoch and sequence
- Audit queries return both state and provenance

---

## 9. Scope

### 9.1 In Scope

- 4D Gaussian splats for dynamic visual primitives
- Spacetime tiling, adjacency, hot set retrieval
- Tiered quantization and compression as eviction
- Entity graph and semantic embeddings
- Lineage, provenance, and disagreement logs
- Coherence gating for write promotion and action permission
- Packed draw list protocol
- WebGPU renderer and native renderer option

### 9.2 Out of Scope

- Training or global optimization on v0 appliance
- Generative gap filling as authority
- Unbounded sampling methods in the primary loop

---

## 10. Implementation Plan

### Phase 1: Web Viewer Baseline

- Stable 60 fps on a static 3DGS model on laptop GPU

### Phase 2: Segmented Time Playback

- Time as sequence of keyframes + deltas
- Smooth scrubbing of short dynamic clips

### Phase 3: RuVector World Model Integration

- RuVector WASM in browser for local indexing/caching
- Text query -> object track -> filtered Gaussian rendering

### Phase 4: Streaming Optimization

- AirGS-style keyframe + delivery optimization
- 4DGS-1K-style active masks for computation trimming
- Stable playback under bandwidth cap

### Phase 5: Live Capture (Roadmap)

- Gaussian SLAM integration
- RGBD preferred, monocular fallback

---

## 11. Failure Modes

| # | Failure Mode | Description | Mitigation |
|---|-------------|-------------|------------|
| 1 | **Identity drift** | Dynamic Gaussians bleed between objects | Track-level constraints in RuVector |
| 2 | **Bandwidth blowups** | Raw 4D streams are huge | Keyframes + deltas + pruning |
| 3 | **Sorting bottlenecks** | Alpha blending requires sorting | CPU approximate binning -> GPU sorting |
| 4 | **Governance/privacy** | Video-derived models contain sensitive data | Privacy tags per object, redaction at query time |

---

## 12. Acceptance Tests

### Test A: Bounded Operation

Run 10 minutes continuous. 99th percentile frame time under target. Hot set memory under cap. Overflow handled by policy, never crash.

### Test B: Auditability

Select any rendered object and retrieve: tile IDs, primitive block refs, lineage events, promotion decisions and coherence scores.

### Test C: Rollback

Inject contradictory update stream. System freezes promotions, maintains render stability, rolls back to last coherent lineage pointer, logs reason.

### Test D: Portability

Same scene replays identically on two devices given identical draw lists.

### Test E: Interactive Query

Given a 30-second dynamic clip, browser client maintains >= 30 fps at 1080p while scrubbing time, median interaction latency under 100 ms for "search and highlight object" queries, stays under configurable bandwidth cap.

---

## 13. Consequences

### 13.1 Benefits

- Bounded latency and bounded memory by design
- Explainable degradation under load
- Auditability, rollback, and replay
- Works on v0 and scales to v1

### 13.2 Costs

- Engineering complexity in tiling, caching, and policy tuning
- Quality is evidence-limited, not imagination-limited
- Requires disciplined separation of render and truth

---

## 14. References

- 3D Gaussian Splatting (Kerbl et al.)
- 4DGS dynamic scenes
- 4DGS-1K active masks and pruning
- AirGS streaming optimization
- Gaussian Splatting SLAM
- SplaTAM / GS-SLAM
- Language-Guided 4DGS
- wgpu (Rust WebGPU)
- three.js WebGPURenderer
