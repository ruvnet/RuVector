# ADR-006: Signal-Class Partition via ruVector Dynamic MinCut

## Status

Accepted

## Context

The original premise warns against assuming the 26-second pulse has a *single*
cause. The record may contain several overlapping phenomena: an ocean-forced
carrier, gliding tremors, shelf resonance, storm-amplified events, instrument
artifacts. We must be able to **partition the event population into signal
classes without forcing one theory too early** (the clustering step in the
project vision), and to notice — in real time — when a *new* class enters the
record (a second mechanism appearing).

This is a graph-cut problem. Build a graph whose vertices are pulse events and
whose edges are embedding similarity (from the ruVector planetary memory,
ADR-002); a minimum cut separates weakly-connected sub-populations, and a
*dynamic* minimum cut maintains that separation as events stream in.

## Decision

Use **`@ruvector/mincut-wasm`** — ruVector's subpolynomial-time **dynamic
minimum cut** with φ-expander hierarchical decomposition (`crates/ruvector-mincut`,
arXiv:2512.13105) — as the partition engine, implemented in `src/partition.ts`.

- Vertices = events; weighted edges = cosine similarity above a threshold over a
  chosen embedding facet (default `waveform`; facets stay separate per ADR-002).
- `classes` = the hierarchy's **Level-2 cluster count** (`num_clusters`): 1 for a
  coherent population, rising as genuinely separate classes appear. The
  φ-expander count (`num_expanders`) is reported as a finer structural diagnostic,
  not as the class count.
- `streamPartition()` re-decomposes after each incoming batch; a jump in `classes`
  is a **regime change / anomaly** — the ADR-004 contradiction-logging hook for a
  second mechanism.

### Graceful degradation (ADR-150 pattern)

The currently published wasm build panics on its *timed* code paths under the
Node wasm runtime (`std::time` is unimplemented there), so `WasmMinCut.insertEdge`,
`WasmMinCutWrapper.query`, and `globalMinCut()` are avoided; we drive the untimed
`WasmThreeLevelHierarchy`. If the module fails to initialize at all,
`partitionEvents()` falls back to an exact connected-components partition
(union-find) and reports `backend: 'fallback-connected-components'`. The harness
never hard-fails on an optional native/wasm dependency.

## Consequences

### Positive

- Real ruVector dynamic-mincut decomposition over the actual event graph; proven
  on real GT.DBIC events (`__tests__/partition.test.ts`): the 26 s population
  forms **one** coherent class, and the mechanism separates two orthogonal
  classes and detects a new class entering a stream.
- Theory-agnostic: classes emerge from similarity structure, not from a labeled
  hypothesis — exactly the "don't force one cause" requirement.
- Dynamic: regime changes are first-class events, feeding anomaly detection.

### Negative

- The published wasm build's timed paths and `globalMinCut()` are unusable in
  this runtime, so we use cluster *count* (not a weighted cut value) as the
  signal and keep a fallback. A future build that compiles time for wasm (or a
  node-napi binding) would unlock the exact s/t partition and cut value.
- Similarity threshold is a tuning knob (an evolvable surface, ADR-001/ADR-005):
  too low merges classes, too high shatters a coherent one.

## Alternatives considered

- **Static global min cut (Stoer–Wagner) in TS** — exact and simple, but not
  dynamic and not the ruVector library the project standardizes on.
- **k-means / HNSW clustering** — needs a preset k and a flat metric; the whole
  point (ADR-002) is to avoid a single pre-mixed similarity. A cut respects the
  graph's natural weak boundaries instead.
