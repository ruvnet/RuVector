# ADR-020: GNN-to-Coherence-Gate Feedback Pipeline

**Status**: Proposed
**Date**: 2026-02-08
**Parent**: ADR-018 Visual World Model, ADR-014 Coherence Engine, ADR-019 Three-Cadence Loop Architecture
**Author**: System Architecture Team
**SDK**: Claude-Flow

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1 | 2026-02-08 | Architecture Team | Initial GNN-to-coherence-gate feedback design |

---

## Abstract

Identity drift is the killer problem in video analytics. Dynamic Gaussians can "bleed" between objects across time, causing entity confusion, phantom tracks, and corrupted scene graphs. The GNN layer performs instance grouping, dynamics prediction, and scene graph generation. The coherence gate controls what updates are accepted into canonical state. This ADR defines the precise mechanism by which a GNN identity verdict translates into a coherence decision (Accept, Defer, Freeze, or Rollback), using graph cut cost as the bridging signal.

---

## 1. Context and Motivation

### 1.1 The Identity Drift Problem

When dynamic Gaussians deform over time, their spatial extents can overlap. Two people walking past each other produce Gaussian clouds that temporarily merge. Without structural constraints, the world model may:

- Assign Gaussians from person A to person B's track
- Create phantom entities from overlapping regions
- Lose track continuity when entities reappear after occlusion

Traditional confidence thresholds on per-entity scores are insufficient. A single entity can have high confidence while the structural relationships between entities are incoherent. The problem is not per-node but per-edge: identity is a graph property.

### 1.2 The Gap

The GNN layer (in `ruvector-gnn`) produces rich signals: instance groupings, dynamics predictions, scene graph edges with confidence. The coherence gate (in `ruvector-vwm` and `ruvector-nervous-system`) consumes binary decisions: Accept, Defer, Freeze, Rollback. The gap is the mapping function between these two representations. This ADR fills that gap.

### 1.3 The Insight: Mincut as Write Attention

Graph cut cost measures how cleanly the entity graph partitions into distinct objects. A low cut cost means entities are well-separated. A high cut cost means entity boundaries are ambiguous. This maps directly to coherence: a write that increases cut cost is making the world model less certain about identity boundaries. The mincut signal acts as "write attention" -- the fourth attention level (see ADR-021).

---

## 2. Decision

### 2.1 GNN Output Signals

The GNN produces two signals per update cycle:

**Per-Entity Identity Confidence** (`c_i`):
- Range: [0.0, 1.0]
- Semantics: probability that entity `i` is correctly identified and tracked
- Computed from: embedding stability over time, track continuity score, appearance consistency
- Source: `ruvector-gnn` :: `IdentityHead` output layer

**Per-Tile Structural Coherence** (`s_t`):
- Range: [0.0, 1.0]
- Semantics: fraction of edges within tile `t` that have residuals below threshold
- Computed from: edge residuals in the entity graph restricted to tile `t`
- Source: `ruvector-gnn` :: `StructureHead` output layer, using `ruvector-graph` adjacency

### 2.2 Graph Cut Cost Computation

For each proposed write (a batch of entity graph deltas from the medium loop), compute the graph cut cost:

```rust
use ruvector_mincut::{SubpolynomialMinCut, MinCutResult};

/// Compute the cost of cutting the entity graph along proposed write boundaries
fn compute_write_cut_cost(
    graph: &EntityGraph,
    proposed_deltas: &[EntityDelta],
) -> f64 {
    let mut mincut = SubpolynomialMinCut::new(Default::default());

    // Build subgraph around affected entities
    let affected_entities: HashSet<EntityId> = proposed_deltas
        .iter()
        .flat_map(|d| d.affected_entity_ids())
        .collect();

    // Insert edges with weights = GNN identity confidence on each endpoint
    for edge in graph.edges_touching(&affected_entities) {
        let weight = (edge.source_confidence + edge.target_confidence) / 2.0;
        mincut.insert_edge(
            edge.source.as_u64(),
            edge.target.as_u64(),
            weight as f64,
        ).ok();
    }

    // The min cut value tells us how "separable" the affected region is
    let result = mincut.min_cut();
    result.value
}
```

### 2.3 Signal-to-Decision Mapping

The mapping from GNN signals to coherence decisions uses three thresholds, calibrated from running statistics:

| Condition | Decision | Rationale |
|-----------|----------|-----------|
| `c_i >= T_accept` AND `delta_cut_cost <= T_cut_stable` | **Accept** | Entity is well-identified, write does not degrade graph separability |
| `c_i >= T_accept` AND `delta_cut_cost > T_cut_stable` | **Defer** | Entity seems identified, but write would blur boundaries. Wait for more evidence. |
| `c_i < T_accept` AND `c_i >= T_rollback` AND `s_t >= T_struct` | **Defer** | Identity uncertain but structure is intact. Hold pending, request refinement. |
| `c_i < T_accept` AND `c_i >= T_rollback` AND `s_t < T_struct` | **Freeze** | Identity uncertain and structure is degrading. Stop promotions, render from last coherent state. |
| `c_i < T_rollback` | **Rollback** | Identity confidence has dropped below recovery threshold. Revert to prior lineage pointer. |

Where:
- `delta_cut_cost = cut_cost_after_write - cut_cost_before_write`
- `T_accept`: identity confidence threshold for acceptance (calibrated)
- `T_rollback`: identity confidence threshold for rollback (calibrated)
- `T_cut_stable`: maximum allowed increase in graph cut cost
- `T_struct`: structural coherence threshold per tile

### 2.4 Decision Logic Implementation

```rust
/// Map GNN signals to coherence gate decision
pub fn map_gnn_to_coherence(
    entity_confidence: f32,       // c_i from GNN IdentityHead
    structural_coherence: f32,    // s_t from GNN StructureHead
    delta_cut_cost: f64,          // change in mincut cost from proposed write
    thresholds: &CalibratedThresholds,
) -> CoherenceDecision {
    if entity_confidence < thresholds.rollback {
        return CoherenceDecision::Rollback {
            reason: format!(
                "Identity confidence {:.3} below rollback threshold {:.3}",
                entity_confidence, thresholds.rollback
            ),
        };
    }

    if entity_confidence >= thresholds.accept {
        if delta_cut_cost <= thresholds.cut_stable {
            return CoherenceDecision::Accept;
        } else {
            return CoherenceDecision::Defer {
                reason: format!(
                    "Cut cost increase {:.4} exceeds stability threshold {:.4}",
                    delta_cut_cost, thresholds.cut_stable
                ),
            };
        }
    }

    // c_i is between rollback and accept thresholds
    if structural_coherence >= thresholds.structural {
        CoherenceDecision::Defer {
            reason: format!(
                "Identity confidence {:.3} below accept threshold {:.3}, awaiting evidence",
                entity_confidence, thresholds.accept
            ),
        }
    } else {
        CoherenceDecision::Freeze {
            reason: format!(
                "Identity {:.3} and structure {:.3} both below thresholds",
                entity_confidence, structural_coherence
            ),
        }
    }
}
```

### 2.5 Confidence Calibration Strategy

Thresholds are not fixed constants. They are calibrated from running statistics of the GNN output distribution.

**Calibration Method**:

1. Maintain a running histogram of `c_i` values (identity confidence) with 100 bins over [0.0, 1.0]. Update on every GNN output cycle (slow loop cadence).

2. Set thresholds at percentile boundaries:
   - `T_accept` = P75 of `c_i` distribution (top quartile is "confident")
   - `T_rollback` = P10 of `c_i` distribution (bottom decile is "lost")
   - `T_struct` = P50 of `s_t` distribution (median structural coherence)

3. `T_cut_stable` is calibrated differently: maintain an exponential moving average of `delta_cut_cost` over the last 100 write cycles. Set `T_cut_stable = EMA + 2 * std_dev` (two standard deviations above mean change).

4. Recalibrate every N slow loop ticks (default: N=10, approximately every 10-100 seconds). Use SONA (`sona` crate) for adaptive threshold tuning with EWC++ to prevent catastrophic forgetting when scene statistics change.

```rust
/// Running calibration state
pub struct ConfidenceCalibrator {
    /// Histogram of identity confidence scores, 100 bins
    confidence_histogram: [u64; 100],
    /// Histogram of structural coherence scores, 100 bins
    structure_histogram: [u64; 100],
    /// EMA of delta_cut_cost
    cut_cost_ema: f64,
    /// Variance estimate for cut cost
    cut_cost_var: f64,
    /// EMA smoothing factor
    alpha: f64,
    /// Total observations
    observation_count: u64,
    /// SONA tuner for adaptive thresholds
    sona_tuner: SonaThresholdTuner,
}

impl ConfidenceCalibrator {
    /// Observe a new GNN output and update histograms
    pub fn observe(&mut self, confidence: f32, structure: f32, delta_cut: f64) {
        let c_bin = (confidence * 99.0).min(99.0) as usize;
        let s_bin = (structure * 99.0).min(99.0) as usize;
        self.confidence_histogram[c_bin] += 1;
        self.structure_histogram[s_bin] += 1;

        // Update cut cost EMA
        self.cut_cost_ema = self.alpha * delta_cut + (1.0 - self.alpha) * self.cut_cost_ema;
        let diff = delta_cut - self.cut_cost_ema;
        self.cut_cost_var = self.alpha * diff * diff + (1.0 - self.alpha) * self.cut_cost_var;

        self.observation_count += 1;
    }

    /// Compute calibrated thresholds from current histograms
    pub fn calibrate(&self) -> CalibratedThresholds {
        CalibratedThresholds {
            accept: self.percentile(&self.confidence_histogram, 0.75),
            rollback: self.percentile(&self.confidence_histogram, 0.10),
            structural: self.percentile(&self.structure_histogram, 0.50),
            cut_stable: self.cut_cost_ema + 2.0 * self.cut_cost_var.sqrt(),
        }
    }

    fn percentile(&self, histogram: &[u64; 100], p: f64) -> f32 {
        let total: u64 = histogram.iter().sum();
        let target = (total as f64 * p) as u64;
        let mut cumulative = 0u64;
        for (i, count) in histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return (i as f32) / 99.0;
            }
        }
        1.0
    }
}
```

---

## 3. Integration Points

### 3.1 Crate Integration

| Crate | Role in Pipeline | Interface |
|-------|-----------------|-----------|
| `ruvector-gnn` | Produces `c_i` (identity confidence) and `s_t` (structural coherence) | `IdentityHead::forward()`, `StructureHead::forward()` |
| `ruvector-mincut` | Computes graph cut cost on entity subgraph | `SubpolynomialMinCut::min_cut()` |
| `ruvector-vwm` | Hosts the feedback pipeline in the slow loop governance module | `governance_loop::evaluate_promotions()` |
| `ruvector-graph` | Provides entity graph adjacency for mincut input | `EntityGraph::edges_touching()` |
| `ruvector-nervous-system` | Consumes coherence decisions for hysteresis and broadcast | `CoherenceGatedSystem::apply_decision()` |
| `cognitum-gate-kernel` | Per-tile coherence accumulation from structural coherence signal | `TileState::ingest_delta()` |
| `sona` | Adaptive threshold calibration via SONA tuner | `SonaThresholdTuner::learn_outcome()` |

### 3.2 Pipeline Flow

```
ruvector-gnn                    ruvector-mincut
     │                               │
     ├── c_i (identity confidence)    ├── delta_cut_cost
     ├── s_t (structural coherence)   │
     │                               │
     └──────────┬────────────────────┘
                │
                ▼
     ┌─────────────────────┐
     │  map_gnn_to_coherence│  (ruvector-vwm :: governance_loop)
     └──────────┬──────────┘
                │
                ├── Accept ──► promote deltas to canonical state
                ├── Defer  ──► keep pending, request more evidence
                ├── Freeze ──► halt promotions, render from last-known
                └── Rollback ─► revert to prior lineage pointer
                         │
                         ▼
                  [lineage_events]  (append-only audit log)
```

### 3.3 Loop Placement

This pipeline runs entirely within the **slow loop** (0.1-1 Hz), as defined in ADR-019. It consumes promotion candidates queued by the medium loop and produces verdicts consumed by both the medium loop (which writes apply) and the fast loop (which LOD policies change).

---

## 4. Consequences

### 4.1 Benefits

- **Structural identity protection**: Identity is not a per-entity scalar but a graph property. Mincut ensures writes that blur entity boundaries are caught.
- **Adaptive thresholds**: Calibration from running statistics means the system adjusts to different scene types (sparse outdoor, dense indoor, crowded) without manual tuning.
- **Auditable decisions**: Every coherence decision includes the GNN confidence, cut cost, and threshold values that produced it. Stored in lineage events.
- **Graceful degradation**: Defer and Freeze are intermediate states. The system does not jump from "accept everything" to "rollback everything."

### 4.2 Costs

- **Mincut computation per write batch**: Even with subpolynomial amortized cost (`ruvector-mincut`), this adds latency to the slow loop. Mitigated by running only on affected subgraph, not full entity graph.
- **Calibration warmup**: At cold start, histograms are empty. Default thresholds are used until sufficient observations accumulate (target: 100 GNN cycles).
- **Two-signal dependency**: Requires both GNN and mincut to be functional. If GNN is unavailable, fall back to mincut-only with conservative thresholds. If mincut is unavailable, fall back to GNN confidence-only with wider Defer band.

### 4.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Calibration drift in long-running sessions | Medium | Medium | Periodic recalibration with SONA; EWC++ prevents forgetting |
| GNN confidence poorly calibrated for new scene types | Medium | High | Cold-start uses conservative thresholds; SONA adapts within ~100 cycles |
| Mincut cost dominated by large static background | Low | Medium | Restrict mincut to dynamic entity subgraph only |
| Rollback cascade from transient sensor noise | Low | High | Require N consecutive low-confidence observations before rollback (hysteresis) |

---

## 5. Acceptance Tests

### Test A: Crossing Entities

Two tracked entities cross paths (Gaussians overlap for 2 seconds). Verify that during overlap, the pipeline produces Defer or Freeze decisions (not Accept). After separation, verify Accept resumes within 3 slow loop ticks.

### Test B: Rollback on Identity Loss

Simulate a tracked entity disappearing from sensor input for 10 seconds. Verify identity confidence drops below T_rollback and a Rollback is issued. Verify the entity's canonical state reverts to the last coherent lineage pointer.

### Test C: Calibration Convergence

Start with empty histograms and default thresholds. Run 200 GNN cycles with known scene statistics. Verify calibrated thresholds converge to within 5% of expected percentile values.

### Test D: Cut Cost Sensitivity

Propose a write that merges two distinct entity clusters. Verify delta_cut_cost exceeds T_cut_stable and produces a Defer or Freeze decision, even when per-entity confidence is high.

---

## 6. References

- ADR-018: Visual World Model as a Bounded Nervous System
- ADR-014: Coherence Engine Architecture (coherence gate semantics)
- ADR-019: Three-Cadence Loop Architecture (loop placement)
- ADR-021: Four-Level Attention Architecture (write attention)
- `ruvector-gnn`: Graph neural network for entity identity and structure
- `ruvector-mincut`: Subpolynomial dynamic graph cut
