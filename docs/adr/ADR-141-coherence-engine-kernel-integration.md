# ADR-141: Coherence Engine — Kernel Integration and Runtime Pipeline

**Status**: Accepted
**Date**: 2026-04-04
**Authors**: Claude Code (Opus 4.6)
**Related**: ADR-132 (DC-1, DC-2, DC-4, DC-6), ADR-133 (Split/Merge), ADR-136 (Memory Tiers)

---

## Context

ADR-132 specifies that the coherence engine is optional (DC-1) and provides
two scheduling signals (DC-4): deadline urgency and cut-pressure boost. The
engine was implemented in `rvm-coherence` as a standalone crate with graph,
mincut, scoring, and adaptive modules — but no runtime integration existed.
The kernel had no mechanism to:

1. Feed real-time communication patterns into the coherence graph
2. Propagate coherence scores to partition objects and the scheduler
3. Act on split/merge recommendations
4. Decay stale edges to prevent graph corruption
5. Enforce security gates on coherence-driven operations

## Decision

### 1. Unified CoherenceEngine with Pluggable Backends

The `CoherenceEngine<MCB, CB>` is generic over two backend traits:

- `MinCutBackend` — pluggable mincut algorithm (default: Stoer-Wagner)
- `CoherenceBackend` — pluggable coherence scoring (default: ratio-based)

When the `ruvector` feature is enabled, stub backends (`RuVectorMinCut`,
`SpectralCoherence`) become available. These will delegate to the ruvector
ecosystem crates once they gain `no_std` support.

Type aliases `DefaultCoherenceEngine` and `RuVectorCoherenceEngine` provide
ergonomic access.

### 2. IPC → Coherence Graph Auto-Feeding

Every `ipc_send()` call automatically increments the coherence graph edge
weight for the sender→receiver pair (weight += 1). This means the coherence
graph always reflects the actual communication topology without manual
instrumentation. The `IpcManager` also tracks cumulative weights per channel.

### 3. Edge Weight Decay

Edge weights decay by 5% per epoch (`decay_bp = 500`). Edges that reach
zero weight are automatically pruned. This prevents stale communication
patterns from dominating the graph and ensures the coherence engine tracks
the *current* topology, not historical patterns.

### 4. Score Propagation to Partition Objects

On every `tick()`, the kernel calls `sync_partition_scores()` which pushes
the coherence engine's per-partition `CoherenceScore` and `CutPressure`
values into the `Partition` struct fields. Downstream consumers (scheduler
priority computation, security gates, tier placement) always see fresh values.

### 5. Coherence-Driven Split/Merge Execution

The kernel provides:
- `execute_split(source)` — creates a child partition, registers in graph, emits `StructuralSplit` witness
- `execute_merge(absorber, absorbed)` — validates preconditions (coherence threshold, adjacency, resources), emits `StructuralMerge` witness
- `apply_decision(decision)` — dispatcher that takes a `CoherenceDecision` from `tick()` and executes the appropriate operation

### 6. Scheduler Integration

`enqueue_partition(cpu, id, deadline_urgency)` automatically injects the
partition's coherence-derived `CutPressure` into the scheduler's priority
computation: `priority = deadline_urgency + cut_pressure_boost`.

### 7. Security-Gated Operations

Capability-checked variants of kernel operations:
- `checked_create_partition(config, token)` — requires `Partition` type + `WRITE` rights
- `checked_ipc_send(edge, msg, token)` — requires `Partition` type + `WRITE` rights
- Denials emit `ProofRejected` witness records via the `SecurityGate` pipeline

### 8. Degraded Mode (DC-6)

`enter_degraded_mode()` / `exit_degraded_mode()` with witness records. In
degraded mode, the scheduler zeroes `CutPressure` for all enqueue operations,
falling back to deadline-only scheduling.

### 9. Memory Tier Integration

The `TierManager` is wired into `tick()`:
- Epoch advance + recency decay (200 bp per epoch)
- `update_region_cut_value()` bridges coherence scores → tier placement
- Residency rule: `cut_value + recency_score` drives Hot/Warm/Dormant/Cold decisions

## Runtime Pipeline

```
IPC message
    │
    ▼
ipc_send() ──→ coherence graph (edge weight += 1)
    │
    ▼
tick() ──→ decay_weights(5%) ──→ recompute scores ──→ sync_partition_scores()
    │                                                         │
    ▼                                                         ▼
EpochResult {                                      Partition.coherence
  summary: scheduler epoch                         Partition.cut_pressure
  decision: Split/Merge/NoAction                   TierManager.decay_recency()
}
    │
    ▼
apply_decision() ──→ execute_split() / execute_merge()
                         │
                         ▼
                    Witness record (StructuralSplit / StructuralMerge)
```

## Consequences

- The coherence engine is now a live, feedback-driven system rather than a static analysis tool
- Stale edges decay naturally, preventing graph corruption from historical traffic
- Security gates are enforced before privileged operations
- Degraded mode provides a clean fallback when coherence is unavailable
- 645 tests pass across the full workspace with 0 clippy warnings

## Test Coverage

| Component | Tests | Key Assertions |
|-----------|-------|---------------|
| CoherenceEngine | 14 | Creation, add/remove, tick, score, pressure, split/merge decisions |
| Bridge backends | 14 | Builtin + ruvector stubs, mincut, scoring, fallback identity |
| Edge decay | 4 | Reduce, prune at zero, 100% prune, zero is noop |
| Kernel integration | 62 | IPC→graph, score propagation, security gates, degraded mode, tier management |
| P3 deep proof | 4 | Root pass, derivation chain, nonexistent, revoked ancestor |
