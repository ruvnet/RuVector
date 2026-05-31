# ADR-194: Semantic Drift Detection for Agent Memory and Vector Index Health

**Status**: Proposed  
**Date**: 2026-05-17  
**Authors**: nightly research agent  
**Crate**: `ruvector-drift`  
**Branch**: `research/nightly/2026-05-17-semantic-drift-detector`

---

## Context

RuVector is used as a long-term memory substrate for autonomous AI agents. As agents run for extended periods, the statistical distribution of vectors stored in the index changes — a phenomenon called *semantic drift*. Sources include:

- The agent's conversational or task context shifts over time.
- The embedding model is updated (model-induced drift).
- The document corpus is updated (corpus-induced drift).
- Memory compaction summarises and re-embeds older vectors.

Without a drift detection mechanism, RuVector cannot distinguish a healthy, stable index from a silently degraded one. Agents continue to query the index, retrieve stale neighbors, and degrade in quality without any observable signal.

Academic literature confirms this is a real problem: the SSGM framework (arXiv:2603.11768) formally proves that agent memory drift accumulates as O(T·ε) per iteration without governance mechanisms [^1]. DriftLens (arXiv:2406.17813) demonstrates that unsupervised embedding drift detection is both feasible and effective across 17 benchmarks [^2].

As of May 2026, no vector database (Qdrant, Milvus, Weaviate, Pinecone, LanceDB, FAISS, pgvector, Chroma, Vespa) includes native semantic drift detection. Existing Rust crates (`scouter-drift`, `irithyll`) target tabular and scalar data, not high-dimensional embedding vectors.

---

## Decision

Introduce `ruvector-drift`, a new standalone Rust crate implementing the `DriftDetector` trait with three complementary algorithms:

1. **`CentroidDriftDetector`** — O(d) per observation, O(d + window·d) space. Detects mean shift. Target use: high-throughput real-time monitoring embedded in the HNSW write path.

2. **`MmdDriftDetector`** — O(D·d) per observation, O(D·d + window·d) space. Uses random Fourier feature approximation of kernel MMD. Detects mean and variance shifts. Target use: default production drift detector, scheduled or per-batch.

3. **`GraphDriftDetector`** — O(n·k·d) per report, O((ref+cur)·d) space. Implements k-NN two-sample topology test. Detects structural/topological distributional changes. Target use: offline audit, scheduled at low frequency.

The public API is trait-based with two window primitives (`reset_current`, `promote_current`) and an alert/score output that is compatible with ruFlo event triggers and MCP tool surfaces.

---

## Consequences

### Positive

- RuVector gains the ability to self-diagnose memory health without external MLOps tooling.
- ruFlo can subscribe to `DriftScore` alerts and trigger memory compaction, re-indexing, or coherence audits.
- The centroid detector adds negligible overhead (<300 ns per HNSW insert at d=128) when embedded in the write path.
- The MCP tool surface gains a `vector_memory_health` tool backed by real measurements.
- The crate is independently buildable and testable with no external service dependencies.

### Negative / Risks

- Thresholds require per-deployment calibration; incorrect thresholds cause false positives (unnecessary reindexing) or false negatives (missed drift).
- MMD bandwidth σ = √d is a heuristic that degrades for L2-normalised embedding models (where ‖x‖₂ ≈ 1 always).
- Graph-kNN is O(n²) and unsuitable for real-time use at window sizes above ~500.
- Slow monotonic drift (gradual over thousands of observations) is not detected by per-observation thresholding — requires a CUSUM layer (future work).

---

## Alternatives Considered

### 1. External MLOps integration (Evidently AI, Arize AI)

These tools provide sophisticated drift dashboards but operate *outside* the vector database. They cannot access query-time retrieval semantics, cannot trigger ruFlo workflows directly, and require data egress that may violate edge/privacy constraints. Rejected: wrong architectural layer.

### 2. Fréchet Distance on PCA-compressed Gaussians (DriftLens approach)

More statistically rigorous than our MMD approximation. Requires eigendecomposition (O(d³)) per window update and a matrix square root — too expensive for streaming use at d≥128. Could be added as a `FrechetDriftDetector` variant for offline audit. Deferred.

### 3. Domain classifier (binary discriminator)

Trains a lightweight model to distinguish reference from current. Interpretable (AUC) and consistent with standard MLOps practice. Requires a training loop, not suitable for online streaming, and adds a training infrastructure dependency. Deferred.

### 4. HNSW-intrinsic drift signals (layer-crossing frequency, avg neighbor distance)

Zero additional memory overhead; uses the HNSW graph itself as a drift proxy. Requires modifying `ruvector-core`'s HNSW implementation and validating the correlation between HNSW structural metrics and true distributional drift. Promising but needs a separate research pass. Future work.

---

## Implementation Plan

### Phase 1 (this PR): Foundation

- [x] `DriftDetector` trait with `DriftScore` and `DriftReport` types
- [x] `CentroidDriftDetector` — O(d) streaming
- [x] `MmdDriftDetector` — RFF-based MMD approximation  
- [x] `GraphDriftDetector` — k-NN two-sample test
- [x] 9 unit tests, all green
- [x] Benchmark binary with acceptance test (PASS)
- [x] Workspace integration

### Phase 2: Integration

- [ ] Feature-flag `drift` in `ruvector-core`
- [ ] Inject `CentroidDriftDetector` into `HnswIndex::insert` write path
- [ ] Emit `DriftEvent` on the internal event bus
- [ ] ruFlo subscription for `DriftEvent` → memory compaction workflow

### Phase 3: Production hardening

- [ ] CUSUM layer over MMD time series for slow drift
- [ ] SIMD-accelerated `cos` approximation for MMD-RFF
- [ ] Online bandwidth estimation (reservoir sampling)
- [ ] Bootstrap threshold calibration
- [ ] `vector_memory_health` MCP tool
- [ ] `ruvector-verified` witness log anchor for drift bounds

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-drift --bin benchmark` on x86_64 Linux, Rust 1.94.1:

| Method | Dataset | Mean latency | p50 | p95 | Throughput | Memory | Drift score | Alert |
|---|---|---:|---:|---:|---:|---:|---:|---|
| centroid | null | 275 ns | 197 ns | 978 ns | 3.6M/s | 257 KB | 0.056 | no |
| mmd-rff | null | 15.7 µs | 19.6 µs | 20.8 µs | 64K/s | 323 KB | 0.044 | no |
| graph-knn | null | 1.98 ms | 1.80 ms | 4.38 ms | 506/s | 205 KB | 0.005 | no |
| centroid | +2σ shift | 205 ns | 169 ns | 269 ns | 4.9M/s | 257 KB | 2.000 | **yes** |
| mmd-rff | +2σ shift | 15.5 µs | 19.5 µs | 20.8 µs | 65K/s | 323 KB | 0.697 | **yes** |
| graph-knn | +2σ shift | 1.98 ms | 1.77 ms | 4.35 ms | 505/s | 205 KB | 1.000 | **yes** |
| centroid | GMM | 179 ns | 169 ns | 201 ns | 5.6M/s | 257 KB | 0.052 | no |
| mmd-rff | GMM | 15.5 µs | 19.5 µs | 20.8 µs | 65K/s | 323 KB | 0.658 | **yes** |
| graph-knn | GMM | 1.97 ms | 1.80 ms | 4.39 ms | 507/s | 205 KB | 1.000 | **yes** |

Critical finding: centroid fails to detect GMM structural drift (score 0.052 vs. null 0.056 — no separation). MMD-RFF and graph-kNN correctly detect it. This justifies providing multiple complementary algorithms.

---

## Failure Modes

1. **Threshold miscalibration**: False positives trigger unnecessary reindexing (compute waste). False negatives allow quality degradation to go undetected. Mitigation: provide calibration guidance; future bootstrap calibration.

2. **Adversarial drift suppression**: A malicious actor injecting vectors that mimic the reference distribution could suppress drift alerts. Mitigation: use multiple complementary detectors; anchor reports in witness log.

3. **Reference poisoning**: If `promote_current` is called when the current window is itself drifted, the new reference will be wrong. Mitigation: only promote after human or ruFlo confirmation.

4. **Cold-start instability**: Fewer than ~k+1 observations makes graph-kNN undefined; fewer than ~20 makes MMD-RFF noisy. Mitigation: require minimum window fill before alerting.

---

## Security Considerations

- Drift detectors operate on statistical summaries, not raw vectors. Centroid and MMD store only aggregate statistics. Graph stores raw vectors (bounded by window size) but does not expose them via public API.
- Drift score logs should be treated as operational metadata, not content-bearing data.
- The witness log anchor (Phase 3) enables verifiable audit without exposing raw embedding content.

---

## Migration Path

This crate is additive. Existing code is unchanged. Phase 2 integration adds a `drift` feature flag that defaults to disabled. Enabling it in `ruvector-core` requires only adding `drift-detector = Some(Box::new(CentroidDriftDetector::new(...)))` to index construction.

---

## Open Questions

1. What is the right default threshold for production agent memory in RuVector? Requires empirical calibration on real agent workloads.
2. Should drift detection be per-partition (per agent) or global? Per-partition is more accurate but requires one detector per agent session.
3. How frequently should the reference be refreshed? After every compaction? After every N vectors? After operator confirmation?
4. Is HNSW-intrinsic drift (using graph structural metrics directly) a viable zero-overhead alternative to the separate detector? Requires a separate research pass.

---

[^1]: "Governing Evolving Memory in LLM Agents." arXiv:2603.11768, 2026.
[^2]: DriftLens. arXiv:2406.17813, 2024.
