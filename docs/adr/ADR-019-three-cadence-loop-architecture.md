# ADR-019: Three-Cadence Loop Architecture

**Status**: Proposed
**Date**: 2026-02-08
**Parent**: ADR-018 Visual World Model, ADR-014 Coherence Engine, ADR-017 Temporal Tensor Compression
**Author**: System Architecture Team
**SDK**: Claude-Flow

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-08 | Architecture Team | Initial three-cadence loop design |

---

## Abstract

The visual world model requires strict separation between render rate and learn rate. Real-time rendering at 30+ fps is demonstrated by current 4D Gaussian methods, but live capture optimization and structural refinement run at fundamentally lower cadences. This ADR defines three loops with explicit rate boundaries, latency budgets, and data flow contracts. The key invariant: the world model is the source of truth; the splats are a view of it.

---

## 1. Context and Motivation

### 1.1 The Rate Mismatch Problem

Real-time dynamic 4D Gaussian methods demonstrate rendering at 30+ fps. But the operations that maintain world model integrity -- tracking, optimization, graph refinement, pruning -- operate at different natural cadences. Collapsing all operations into a single loop creates one of two failure modes:

1. **Render stall**: Heavy operations (GNN refinement, consolidation) block the render path, causing frame drops.
2. **Model corruption**: Rushing learning updates to meet frame deadlines produces incoherent state.

The solution is not a priority queue or async task pool. It is a strict architectural separation into three loops with defined interfaces, so that each loop can be reasoned about independently.

### 1.2 The Pattern

Render fast. Learn continuously at a controlled cadence. Refine slowly with structural guarantees.

This mirrors biological nervous systems: reflexes operate at millisecond timescales, perception at tens of milliseconds, and deliberation at seconds or longer. Each timescale has its own state, its own budget, and its own failure mode.

---

## 2. Decision

Implement three loops with strict rate separation and explicit data flow contracts between them.

### 2.1 Fast Loop (Per-Frame, 30-60 Hz)

**Purpose**: Produce frames. Deterministic and bounded.

**Operations**:
- Camera pose tracking (consume IMU/visual odometry output)
- Scene block selection via spatial index query
- Local cache hit/miss resolution
- Packed draw list assembly under per-tile budgets
- WebGPU draw submission
- Frame metric recording (draw time, overdraw, cache hit rate)

**Latency Budget**: 16.6 ms at 60 Hz, 33.3 ms at 30 Hz. Target: complete within 12 ms to leave headroom.

**State Owned**:
- Current camera pose
- Active block set (read-only view of tile cache)
- Draw list (ephemeral, rebuilt each frame)
- Frame metrics accumulator

**State Read (Immutable View)**:
- Tile cache (populated by medium loop)
- LOD policy (set by slow loop)
- Budget profile (set by slow loop)

**Crate Mapping**:
- `ruvector-vwm` :: `render_loop` module -- draw list assembly
- `ruvector-attention-wasm` -- view attention for block selection
- `ruvector-vwm-wasm` -- browser-side cache queries

**Invariant**: The fast loop never writes to the world model. It reads a snapshot and produces frames.

```
┌─────────────────────────────────────────────────────┐
│                  FAST LOOP (30-60 Hz)                │
│                                                      │
│  Pose ──► Block Select ──► Draw List ──► WebGPU     │
│    │          │                              │       │
│    │     [tile cache]                   [frame out]  │
│    │      (read-only)                        │       │
│    └──────────────────── metrics ─────────────┘      │
└─────────────────────────────────────────────────────┘
```

### 2.2 Medium Loop (2-10 Hz)

**Purpose**: Update the dynamic layer of the world model. Continuous learning within bounded compute.

**Operations**:
- Ingest new Gaussian deltas from sensor/optimization pipeline
- Update active Gaussian subset in tile cache
- Update object tracks (position, velocity, bounding volumes)
- Update semantic embeddings for changed entities
- Write deltas into RuVector entity graph
- Queue promotion candidates for coherence gate

**Latency Budget**: 100-500 ms per tick. Must complete before next tick; drops tick if overrun rather than accumulating debt.

**State Owned**:
- Pending delta buffer
- Active Gaussian subset (mutable)
- Object track state
- Embedding update queue

**State Written**:
- Tile cache (consumed by fast loop)
- Entity graph deltas (consumed by slow loop)
- Promotion candidate queue (consumed by coherence gate)

**State Read**:
- Sensor event stream (external input)
- Coherence gate verdicts (from slow loop)
- LOD and budget policies (from slow loop)

**Crate Mapping**:
- `ruvector-vwm` :: `update_loop` module -- delta ingestion, track updates
- `ruvector-temporal-tensor` -- time-bucketed delta compression
- `ruvector-delta-core` -- delta encoding and merging
- `ruvector-graph` -- entity graph writes
- `ruvector-attention` -- temporal attention for relevant time slices

**Invariant**: The medium loop only touches the dynamic layer. It does not modify structural identity (that belongs to the slow loop).

```
┌─────────────────────────────────────────────────────┐
│                MEDIUM LOOP (2-10 Hz)                 │
│                                                      │
│  Sensors ──► Delta Buffer ──► Tile Cache Update     │
│                  │                                   │
│           Track Updates ──► Entity Graph Deltas      │
│                  │                                   │
│         Embedding Updates ──► Promotion Queue        │
│                                     │                │
│                              [to coherence gate]     │
└─────────────────────────────────────────────────────┘
```

### 2.3 Slow Loop (0.1-1 Hz)

**Purpose**: Structural refinement, identity resolution, governance. Can run server-side.

**Operations**:
- GNN refinement for identity grouping and structure inference
- Coherence gate evaluation on queued promotions
- Consolidation: merge redundant Gaussians, prune dead primitives
- Keyframe publishing: snapshot canonical state at time anchors
- LOD policy adjustment based on budget pressure and render metrics
- Lineage event recording

**Latency Budget**: 1-10 seconds per tick. Allowed to use server GPU. Must not block medium or fast loops.

**State Owned**:
- GNN model weights and training state
- Coherence gate thresholds and history
- Consolidation buffers
- Keyframe index

**State Written**:
- Canonical entity graph (authoritative state)
- LOD and budget policies (consumed by fast and medium loops)
- Coherence verdicts (consumed by medium loop)
- Lineage events (append-only audit log)

**State Read**:
- Promotion candidate queue (from medium loop)
- Frame metrics (from fast loop)
- Entity graph deltas (from medium loop)

**Crate Mapping**:
- `ruvector-gnn` -- identity grouping, dynamics prediction, scene graph
- `ruvector-mincut` -- graph cut cost for coherence decisions
- `ruvector-vwm` :: `governance_loop` module -- coherence gate integration
- `ruvector-nervous-system` -- CoherenceGatedSystem for write promotion
- `cognitum-gate-kernel` -- tile-level coherence evaluation
- `sona` -- threshold self-tuning
- `ruvector-temporal-tensor` -- keyframe compression

**Invariant**: The slow loop is the only loop that modifies structural identity and publishes coherence verdicts.

```
┌─────────────────────────────────────────────────────┐
│                 SLOW LOOP (0.1-1 Hz)                 │
│                                                      │
│  Promotion Queue ──► Coherence Gate ──► Verdicts    │
│         │                                   │        │
│  Entity Deltas ──► GNN Refinement ──► Identity      │
│         │                                   │        │
│  Frame Metrics ──► LOD Policy ──► Budget Adjust     │
│         │                                   │        │
│  Consolidation ──► Pruning ──► Keyframe Publish     │
│                                       │              │
│                              [lineage events]        │
└─────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Between Loops

### 3.1 Flow Diagram

```
                    ┌──────────────┐
                    │  Slow Loop   │
                    │  0.1-1 Hz    │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         LOD Policy   Verdicts    Keyframes
              │            │            │
              ▼            ▼            ▼
                    ┌──────────────┐
                    │ Medium Loop  │
                    │  2-10 Hz     │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         Tile Cache   Promotions   Metrics
              │            │            │
              ▼            ▲            ▲
                    ┌──────────────┐
                    │  Fast Loop   │
                    │  30-60 Hz    │
                    └──────────────┘
```

### 3.2 Inter-Loop Communication

| From | To | Channel | Data | Semantics |
|------|----|---------|------|-----------|
| Medium | Fast | Shared tile cache | Updated Gaussian blocks | Copy-on-write swap; fast loop reads old until medium publishes new |
| Medium | Slow | Promotion queue | Candidate deltas with evidence | Bounded MPSC channel; drops oldest if full |
| Fast | Medium | Metrics buffer | Frame time, cache hits, overdraw | Lock-free ring buffer; medium samples latest |
| Slow | Medium | Verdict channel | Accept/Defer/Freeze/Rollback per entity | Broadcast; medium loop applies on next tick |
| Slow | Fast | Policy slot | LOD parameters, budget profile | Atomic swap; fast loop picks up on next frame |
| Slow | Audit | Lineage log | Append-only events | Write-ahead log; never read by fast or medium |

### 3.3 Latency Budgets Summary

| Loop | Target Rate | Max Tick Duration | Overrun Policy |
|------|------------|-------------------|----------------|
| Fast | 30-60 Hz | 12 ms | Drop frame, log warning, reduce LOD on next frame |
| Medium | 2-10 Hz | 500 ms | Skip tick, log, do not accumulate debt |
| Slow | 0.1-1 Hz | 10 s | Continue, do not block other loops |

---

## 4. Key Invariant

**The world model is the source of truth. The splats are a view of it.**

This means:
- The fast loop never modifies the world model. It renders a view.
- The medium loop proposes changes. It does not finalize them.
- The slow loop is the only authority that promotes changes to canonical state.
- Coherence verdicts flow downward (slow to medium). Never upward.
- If the slow loop is unavailable, the medium loop continues with last-known verdicts. The fast loop continues with last-known tile cache. Rendering degrades gracefully to stale data, never to corrupt data.

---

## 5. Consequences

### 5.1 Benefits

- **Bounded frame latency**: Fast loop is isolated from heavy compute. Frame drops are a LOD policy issue, not a structural issue.
- **Safe learning**: Medium loop updates cannot corrupt identity because the slow loop gates structural changes.
- **Server offload**: Slow loop can run on a remote GPU without affecting client render.
- **Deterministic replay**: Each loop has explicit inputs and outputs. Recording loop inputs enables replay.
- **Independent scaling**: Each loop can be profiled, tuned, and load-tested in isolation.

### 5.2 Costs

- **Three-loop coordination complexity**: Shared-state contracts between loops require careful design. Copy-on-write, bounded channels, and atomic swaps add implementation surface.
- **Latency between loops**: A change detected by the medium loop will not be visible in the fast loop until the next tile cache swap (up to 500 ms). Changes requiring slow loop approval may take 1-10 seconds to be committed.
- **Duplicate state**: The tile cache exists as both the medium loop's mutable copy and the fast loop's read-only snapshot. Memory cost is bounded by the active block set size.

### 5.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Inter-loop channel backpressure | Medium | Medium | Bounded channels with drop-oldest policy; metrics on queue depth |
| Slow loop unavailability stalls medium loop | Low | High | Medium loop operates on last-known verdicts; timeout promotes to "defer" |
| Tile cache swap causes frame glitch | Medium | Low | Double-buffer with atomic pointer swap at frame boundary |
| Medium loop overrun cascades | Low | Medium | Tick skip with no debt accumulation; reduce update rate adaptively |

---

## 6. Acceptance Tests

### Test A: Rate Isolation

Run all three loops for 5 minutes. Inject a 5-second GNN refinement into the slow loop. Verify fast loop frame time remains below 16.6 ms throughout. Verify medium loop tick rate does not drop below 2 Hz.

### Test B: Stale-But-Correct Rendering

Stop the slow loop. Continue medium and fast loops for 60 seconds. Verify rendering continues with stale but consistent state. No crashes, no undefined behavior, no identity drift in rendered output.

### Test C: Promotion Latency

Inject a known-good delta into the medium loop. Measure wall-clock time until it appears in the fast loop's rendered output. Target: under 1 second for medium-to-fast path. Under 12 seconds for medium-to-slow-to-fast path (including coherence gate evaluation).

### Test D: Overrun Recovery

Force the medium loop to exceed its tick budget for 3 consecutive ticks. Verify it skips ticks cleanly, does not accumulate backlog, and resumes normal cadence when load decreases.

---

## 7. References

- ADR-018: Visual World Model as a Bounded Nervous System
- ADR-014: Coherence Engine Architecture
- ADR-017: Temporal Tensor Compression
- 4DGS-1K: Active masks and computation trimming
- AirGS: Streaming optimization for Gaussian splatting
