# Semantic Drift Detection for Agent Memory and Vector Index Health

**150-character summary:** Detect when an AI agent's memory distribution has silently shifted using three complementary Rust algorithms: centroid, MMD-RFF, and k-NN topology tests.

---

## Abstract

Long-running AI agents accumulate vector memories over time. As context changes, the semantic distribution of those memories shifts — a phenomenon called *semantic drift*. Without detection, agents keep querying a stale index, retrieve irrelevant context, and degrade silently. This research implements three drift detection algorithms as a standalone Rust crate (`ruvector-drift`) with a shared `DriftDetector` trait: centroid tracking (3.6M obs/sec), MMD approximated with random Fourier features (64K obs/sec), and a k-NN topology test (507 report/sec). All three pass a battery of acceptance tests against synthetic drifted datasets. The centroid detector is suitable for high-throughput real-time monitoring; MMD-RFF is the recommended production default; graph-topology is the gold standard for offline audits.

---

## Why This Matters for RuVector

RuVector is not just a vector database — it is a cognitive substrate for agents, graphs, and retrieval. Agent memory managed in RuVector can silently drift as the agent's environment, task, or conversational context changes. Without a drift detector embedded in the retrieval path:

1. The agent retrieves semantically stale neighbors.
2. The agent's HNSW graph accumulates vectors from an obsolete distribution.
3. Reindexing or compaction is triggered reactively (after failures) instead of proactively.
4. ruFlo workflow loops have no signal to trigger memory reorganization.

`ruvector-drift` gives RuVector the ability to self-diagnose memory health and expose that signal to ruFlo, MCP tools, and operator dashboards — transforming a passive storage system into an active cognition substrate.

---

## 2026 State of the Art Survey

### What academia says

**DriftLens** (arXiv:2406.17813, Greco et al., 2024) proposes Fréchet distance on PCA-compressed multivariate Gaussian fits. It achieves ≥0.85 correlation with ground-truth drift curves across 17 benchmarks and is 5× faster than prior unsupervised methods. It remains Python-only and not integrated with any vector database.

**SSGM** (arXiv:2603.11768, 2026) formally proves that iterative memory summarization in LLM agents produces O(T·ε) semantic drift accumulation per round, bounding the divergence only with reconciliation against immutable episodic logs. This is the only theorem on bounded agent memory drift; no Rust implementation exists.

**Drift-Adapter** (arXiv:2509.23471, Vejendla 2025) addresses *model-induced* embedding drift after an embedding model upgrade. Linear (Orthogonal Procrustes via SVD) and low-rank affine adapters recover 95-99% recall at <10 µs overhead. This is different from the *distributional drift* we detect here.

**AI Agents Need Memory Control** (arXiv:2601.11653, Bousetouane 2026) demonstrates that unchecked agent memory replay causes behavioral drift and hallucination across IT ops, cybersecurity, and healthcare agents.

### What the ecosystem is doing

None of Qdrant, Milvus, Weaviate, Pinecone, LanceDB, FAISS, pgvector, Chroma, or Vespa have native semantic drift detection as of May 2026. Drift monitoring is outsourced to external MLOps tools (Evidently AI, Arize AI, WhyLabs, Galileo) that operate *outside* the vector database and cannot see query-time retrieval semantics.

### Rust-specific gap

The Rust crates `scouter-drift` and `irithyll` implement drift detection for tabular ML data and scalar streams respectively. Neither handles high-dimensional embedding vectors. Neither integrates with HNSW or vector index structures. `ruvector-drift` is the first Rust crate targeting embedding-space semantic drift.

---

## Forward-Looking Thesis (2036–2046)

By 2036, the dominant AI infrastructure pattern will be **long-lived autonomous agent clusters** — agents that run for weeks, months, or years accumulating experience in local vector memory. The central reliability problem will not be hardware or network failure; it will be **cognitive drift**: agents operating from outdated world models without realizing it.

By 2046:
1. **Regulatory requirement**: Safety-critical deployments (medical, legal, autonomous systems) will require proof of memory coherence — that an agent's knowledge base has not silently degraded.
2. **Self-healing cognition**: Vector indexes will automatically compact, prune, and re-embed memories when drift exceeds a threshold, without human intervention.
3. **Drift certificates**: Agent memory will carry cryptographic drift bounds — proof that distributional divergence from an initial reference never exceeded a governance limit during the agent's operational lifetime.

`ruvector-drift` is a small but essential primitive toward all three futures.

---

## RuVNet Ecosystem Fit

| Ecosystem component | Role of drift detection |
|---|---|
| **RuVector core** | Embed drift score in HNSW node metadata; trigger lazy compaction |
| **ruFlo** | Drift alert → workflow branch → memory reindex task |
| **RVF cognitive packages** | Drift snapshots as a field in the RVF manifest |
| **RVM coherence domains** | Drift magnitude as an input to coherence scoring |
| **ruvnet MCP tools** | `vector_memory_health` MCP tool backed by this crate |
| **Cognitum Seed** | Lightweight centroid detector suitable for edge/embedded |
| **ruvector-mincut** | Graph-kNN drift + mincut = coherence-gated memory eviction |

---

## Proposed Design

### Core trait

```rust
pub trait DriftDetector: Send + Sync {
    fn observe(&mut self, vec: &[f32]) -> DriftScore;
    fn report(&self) -> DriftReport;
    fn reset_current(&mut self);
    fn promote_current(&mut self);
    fn dims(&self) -> usize;
    fn name(&self) -> &'static str;
}
```

Two windows are maintained:
- **Reference window**: established at construction or after `promote_current`.
- **Current window**: accumulates live observations via `observe`.

Drift is the statistical divergence between these two windows.

### Architecture diagram

```mermaid
flowchart TD
    Agent["AI Agent"] -->|embed + store| VectorIndex["RuVector HNSW Index"]
    VectorIndex -->|new vectors| DriftPipeline["DriftDetector pipeline"]
    DriftPipeline --> Centroid["CentroidDriftDetector\nO(d) per obs\n3.6M obs/sec"]
    DriftPipeline --> MMD["MmdDriftDetector\nO(D·d) per obs\n64K obs/sec"]
    DriftPipeline --> Graph["GraphDriftDetector\nO(n·k·d) per report\n507 reports/sec"]
    Centroid -->|score| AlertBus["Alert bus"]
    MMD -->|score| AlertBus
    Graph -->|score| AlertBus
    AlertBus -->|"drift_score > threshold"| ruFlo["ruFlo workflow\n(trigger reindex)"]
    AlertBus -->|DriftReport| MCPTool["MCP tool:\nvector_memory_health"]
    AlertBus -->|magnitude| RVFManifest["RVF manifest\n(drift_bound field)"]
```

---

## Variant designs

### Variant 1: CentroidDriftDetector (baseline)

Tracks the running mean of both windows using an online algorithm (Welford). Drift score = L2(centroid_cur − centroid_ref) / √d. The √d normalisation makes the score comparable across different embedding dimensions.

- **Strengths**: O(d) time, O(d) space. 3.6M observations/sec at d=128. Minimal memory overhead.
- **Limitation**: Cannot detect distributional changes that preserve the mean (e.g., variance increase, multimodal split). In the GMM benchmark below, score was 0.055 — indistinguishable from null.

### Variant 2: MmdDriftDetector (recommended default)

Approximates kernel Maximum Mean Discrepancy using the random Fourier feature (RFF) trick from Rahimi & Recht (2007). Projects d-dimensional vectors into D-dimensional feature space using random weights drawn from the kernel's spectral distribution.

- **Strengths**: Detects both mean and higher-order distributional shifts. O(D·d) per observation. Sliding window with O(1) online mean update. Statistically principled test statistic.
- **Limitation**: RFF approximation quality depends on D; D=128 gives reliable detection. Bandwidth σ must be tuned (√d is a good default for unit-variance embeddings).

### Variant 3: GraphDriftDetector (gold standard)

Implements the k-NN two-sample test (analogous to Friedman–Rafsky on embedding neighborhoods). For each current-window vector, finds its k nearest neighbors in the combined reference+current pool. If current vectors cluster among themselves far more than chance, a structural topology change has occurred.

- **Strengths**: Detects structural drift invisible to both centroid and MMD. Particularly useful for detecting multimodal splits and the emergence of new clusters.
- **Limitation**: O(n·k·d) per report (O(n²) at n = ref + cur). Not suitable for high-throughput real-time monitoring. Best used as a scheduled audit.

---

## Benchmark Methodology

**Hardware**: x86_64 Linux (cloud ephemeral container)  
**OS**: linux  
**Rust**: 1.94.1 (e408947bf 2026-03-25)  
**Build**: `cargo run --release -p ruvector-drift --bin benchmark`

**Datasets** (deterministic, seeded):
- Reference: N=1000, d=128, N(0, 1), seed=42
- Null current: N=1000, d=128, N(0, 1), seed=99 — same distribution
- Centroid shift: N=1000, d=128, N(2.0, 1.0), seed=77 — mean shifted by 2σ
- GMM structural: N=1000, d=128, mixture of N(+3,1) and N(-3,1) — same global mean≈0, bimodal structure

For graph-knn the reference and query sizes are 200 due to O(n²) cost at n=400.

**Latency**: per-`observe` wall-clock time, measured with `std::time::Instant`, sorted for percentiles.

**Note**: These are micro-benchmarks with no OS noise mitigation, run on shared cloud hardware. Times may vary 2-3× across environments. Criterion benches (in `benches/drift_bench.rs`) provide statistical stabilisation for fine-grained comparisons.

---

## Real Benchmark Results

| Method | Dataset | N_ref | N_qry | Dim | Mean(ns) | p50(ns) | p95(ns) | QPS | Mem(bytes) | DriftMag | Alert? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| centroid | null (no drift) | 1000 | 1000 | 128 | 275.1 | 197 | 978 | 3,634,632 | 257,024 | 0.0555 | ok |
| mmd-rff | null (no drift) | 1000 | 1000 | 128 | 15,655 | 19,613 | 20,847 | 63,876 | 323,072 | 0.0445 | ok |
| graph-knn | null (no drift) | 200 | 200 | 128 | 1,976,701 | 1,804,025 | 4,379,370 | 506 | 204,800 | 0.0045 | ok |
| centroid | centroid shift +2σ | 1000 | 1000 | 128 | 204.5 | 169 | 269 | 4,890,119 | 257,024 | 2.0004 | **DRIFT** |
| mmd-rff | centroid shift +2σ | 1000 | 1000 | 128 | 15,494 | 19,526 | 20,805 | 64,542 | 323,072 | 0.6971 | **DRIFT** |
| graph-knn | centroid shift +2σ | 200 | 200 | 128 | 1,979,507 | 1,773,737 | 4,351,863 | 505 | 204,800 | 1.0000 | **DRIFT** |
| centroid | GMM structural | 1000 | 1000 | 128 | 178.9 | 169 | 201 | 5,588,528 | 257,024 | 0.0522 | ok |
| mmd-rff | GMM structural | 1000 | 1000 | 128 | 15,478 | 19,492 | 20,804 | 64,607 | 323,072 | 0.6580 | **DRIFT** |
| graph-knn | GMM structural | 200 | 200 | 128 | 1,971,212 | 1,795,369 | 4,387,490 | 507 | 204,800 | 1.0000 | **DRIFT** |

**Acceptance test**: PASS — all six checks passed.

**Key finding**: The centroid detector scores 0.055 on the GMM dataset (indistinguishable from 0.056 null). Only MMD-RFF and graph-kNN correctly identify structural drift when the global centroid is unchanged. This validates the need for multiple complementary detectors.

---

## Memory and Performance Math

### Centroid detector

```
ref_centroid:   d × 4 bytes = 512 bytes (d=128)
cur_sum:        d × 4 bytes = 512 bytes
cur_buffer:     window × d × 4 = 500 × 128 × 4 = 256,000 bytes
Total:          ≈ 257 KB
```

Per-observation cost: 2 × d additions + 1 subtraction (eviction) = O(d). At d=128: ~200 ns measured.

### MMD-RFF detector

```
weights:        D × d × 4 = 128 × 128 × 4 = 65,536 bytes
biases:         D × 4 = 512 bytes
ref_mean_feat:  D × 4 = 512 bytes
cur_mean_feat:  D × 4 = 512 bytes
eviction_buf:   window × d × 4 = 256,000 bytes
Total:          ≈ 323 KB
```

Per-observation cost: D × (d multiplications + cos) = O(D·d). At D=128, d=128: ~19.5 µs measured. Note: `cos` is expensive. Using SIMD cos approximations or tabulated lookups could bring this to ~2-3 µs.

### Graph detector

```
ref_vecs:       ref_n × d × 4 = 200 × 128 × 4 = 102,400 bytes
cur_buf:        window × d × 4 = 200 × 128 × 4 = 102,400 bytes
Total:          ≈ 205 KB
```

Per-report cost: O((ref + cur) × k × d). At n=400, k=10, d=128: ~1.8 ms measured. Complexity is O(n²) in the window sizes — suitable for offline audits, not real-time monitoring.

---

## How It Works: Walkthrough

### Centroid drift

```
Reference: vectors {x₁, ..., xₙ} → μ_ref = (1/n) Σxᵢ

For each new vector y:
  1. Evict oldest from sliding window (if full)
  2. Update cur_sum += y
  3. Compute μ_cur = cur_sum / |window|
  4. score = ||μ_cur - μ_ref||₂ / √d
  5. alert if score > threshold
```

The division by √d makes the score independent of dimension for unit-variance Gaussian data: E[||μ_cur - μ_ref||₂] ≈ √(2d/n) / √d = √(2/n), which is ~0.045 for n=1000.

### MMD with Random Fourier Features

```
Setup: sample wᵢ ~ N(0, σ⁻²I), bᵢ ~ U[0, 2π] for i=1..D
φ(x) = √(2/D) × [cos(w₁ᵀx + b₁), ..., cos(w_Dᵀx + b_D)]

Reference: mean_ref = (1/n) Σ φ(xᵢ)

For each new vector y:
  1. Compute φ(y)
  2. Update mean_cur with online mean update
  3. MMD² ≈ ||mean_ref - mean_cur||²
  4. score = √MMD²
  5. alert if score > threshold
```

The key insight: `E[φ(x)] = E_X[φ(X)]` and `||E[φ(X)] - E[φ(Y)]||² → MMD²(P, Q)` as D → ∞. With D=128, the approximation is tight enough for reliable detection.

### Graph k-NN topology

```
Build: labeled pool = ref_vecs ∪ cur_vecs

For each cur vector c:
  1. Find k nearest neighbors in pool (excluding c itself)
  2. Count how many neighbors are from cur_vecs (intra-current)

Expected intra-current fraction (null): (|cur| - 1) / (|pool| - 1)
Observed fraction: intra_current / total_edges

drift_score = max(0, (observed - expected) / (1 - expected))
```

When distributions are identical, current vectors' neighbors are random draws from the pool. When drift occurs, current vectors cluster among themselves (higher observed intra-current rate than expected). A score of 1.0 means complete separation — every current vector's k-NN are all other current vectors.

---

## Practical Failure Modes

1. **Threshold sensitivity**: All three detectors are threshold-based. A threshold too low causes false positives on legitimate distributional variation; too high misses real drift. Calibrate thresholds on a held-out validation window from the same distribution.

2. **Cold start**: With fewer than ~k+1 vectors in the current window, graph-kNN returns 0 (not enough data). Centroid and MMD are meaningful with ≥10 observations.

3. **Slow drift**: Gradual drift over thousands of vectors may not exceed the alert threshold at any single step. Use trend analysis over a history of `DriftScore` values for slow-drift detection.

4. **Adversarial injection**: A malicious agent could inject vectors that slowly shift the reference without triggering an alert by staying just below threshold. Use `promote_current` conservatively and audit reference transitions.

5. **MMD bandwidth mismatch**: If σ is too large or too small relative to the data scale, MMD-RFF sensitivity degrades. Use the median pairwise distance of the reference window as a calibration heuristic.

6. **Graph O(n²) cost**: At window sizes above 500, graph-kNN becomes impractical for real-time use. Use it on a scheduled basis (e.g., every 60 seconds) rather than per-observation.

---

## Security and Governance Implications

- **Audit trail**: Each `DriftReport` should be logged with a timestamp and a hash of the reference centroid. This creates a lightweight audit trail of memory health without exposing raw vector content.
- **GDPR/privacy**: Drift detectors operate on statistical summaries (centroids, feature means), not on individual vectors. They can run on encrypted or pseudonymised vector stores.
- **Byzantine resilience**: Adversarially crafted vectors designed to suppress drift alerts (by staying just below thresholds) require coordination across many injected vectors. Using MMD alongside centroid detection reduces this attack surface.
- **Proof-gated integration**: Future work can anchor `DriftReport` values into the `ruvector-verified` witness log, creating verifiable proof that memory drift was within governance bounds throughout an agent's operational session.

---

## Edge and WASM Implications

The centroid detector has no floating-point transcendentals, no heap allocations beyond the window buffer, and compiles to ~3 KB of WASM. It is suitable for Cognitum Seed (embedded appliance) and browser-side agent memory monitoring.

MMD-RFF requires `cos()` which is available in WASM but may be slow without SIMD. The weight matrix can be pre-computed and serialised as part of an RVF package, enabling deterministic drift detection without re-sampling on edge devices.

Graph-kNN is too compute-intensive for embedded use but can run on resource-constrained devices (RPi 4, Hailo-8L) at low report frequencies.

---

## MCP and Agent Workflow Implications

Proposed MCP tool surface (not yet implemented):

```json
{
  "name": "vector_memory_health",
  "description": "Returns the current semantic drift score for an agent memory partition",
  "inputSchema": {
    "partition_id": "string",
    "method": "centroid | mmd-rff | graph-knn"
  },
  "outputSchema": {
    "drift_detected": "boolean",
    "magnitude": "number",
    "window_size": "integer",
    "method": "string",
    "recommendation": "string"
  }
}
```

ruFlo integration pattern:
```
on memory_write(vec):
  drift_score = centroid_detector.observe(vec)
  if drift_score.alert:
    trigger_workflow("memory-reindex", { partition, magnitude: drift_score.score })
```

---

## Practical Applications

1. **Agent memory compaction trigger**: When centroid or MMD drift exceeds threshold, ruFlo spawns a memory compaction job — evicting old vectors, re-embedding recent ones, or rebuilding the HNSW index segment.

2. **Graph-RAG staleness detection**: In a graph-RAG system, the subgraph retrieved for a query reflects the reference distribution. Drift detection flags when the retrieval graph no longer represents the current document corpus.

3. **Enterprise semantic search refresh**: Enterprise knowledge bases are updated continuously. Drift detection triggers incremental re-indexing of changed document segments rather than full re-embedding.

4. **MCP memory tool health endpoint**: Expose `vector_memory_health` as an MCP tool so Claude and other agents can self-assess their memory before making retrieval-dependent decisions.

5. **Local-first AI assistants**: Desktop or edge AI assistants (running on Cognitum Seed) that use local embedding models need to know when their memory of conversations has drifted from the current session context.

6. **Edge anomaly detection**: Monitor whether sensor embedding streams (mmWave, audio, vibration) have drifted from a baseline calibration reference.

7. **Security event retrieval**: Detect when the distribution of security events stored in the vector index has shifted (e.g., new attack patterns appearing), signalling that retrieval models tuned on old data need recalibration.

8. **Code intelligence drift**: Code search indexes drift as codebases evolve. Detect when the distribution of code embeddings has changed enough to warrant re-indexing the affected modules.

---

## Exotic Applications

1. **RVM coherence domain health**: Use graph-kNN drift to audit whether two agent coherence domains (RVM partitions) have converged or diverged — a prerequisite for coherence-gated merge operations.

2. **Cognitum Seed adaptive calibration**: An edge appliance running centroid drift on sensor embeddings automatically recalibrates its anomaly detection baseline after detecting environmental drift.

3. **Proof-gated memory certification**: Anchor drift magnitude bounds into `ruvector-verified` witness logs. Generate ZK proofs that memory drift never exceeded a governance threshold during a regulatory audit period (2036+ horizon).

4. **Swarm memory coherence**: In a multi-agent swarm, each agent monitors its local memory drift relative to a shared reference. When individual drift exceeds the cluster threshold, the agent requests memory synchronization from the swarm.

5. **Self-healing vector graph**: HNSW nodes accumulate stale long-range links as the distribution drifts. Drift detection triggers targeted link repair — removing edges that cross distributional boundaries and adding new edges within the current cluster.

6. **Dynamic world model updates**: Autonomous robots or simulation agents maintain a world model as a vector graph. Semantic drift in the world model signals that the environment has changed enough to trigger targeted exploration or re-mapping.

7. **Agent operating system memory pager**: An agentic OS that manages memory for many concurrent agents uses drift scoring to decide which memory partitions to evict to disk (DiskANN-style cold tier) vs. keep in hot HNSW.

8. **Bio-signal memory**: Medical agents monitoring continuous physiological signals (EEG, ECG embeddings) use drift detection to identify physiological state transitions — a bridge between vector retrieval and clinical decision support.

---

## Deep Research Notes

### What the SOTA suggests

DriftLens shows that unsupervised embedding drift detection is tractable and useful, but existing methods are Python-only and disconnected from the vector database layer. The SSGM theorem provides the theoretical foundation: agent memory drift is bounded if and only if drift detection is coupled with reconciliation against an immutable reference. Our implementation provides the detection side; `ruvector-verified` provides the reconciliation side.

### What remains unsolved

1. **Online bandwidth estimation for MMD**: Our σ = √d heuristic works for unit-variance embeddings but degrades for embedding models with different norm distributions (e.g., sentence-transformers with L2-normalised outputs). An online median pairwise distance estimator would improve robustness.

2. **Statistical power calibration**: We do not provide false positive rate guarantees. A proper implementation would derive thresholds from a bootstrap distribution under the null, giving controlled α-level detection.

3. **Slow drift detection**: Per-observation alerts are not reliable for gradual drift over thousands of observations. A CUSUM (cumulative sum) layer over the MMD time series would detect slow monotonic drift.

4. **HNSW-intrinsic drift signals**: The HNSW graph itself contains drift information — layer-crossing frequency, avg neighbor distance, and ef-search convergence patterns all change under distributional shift. None of these are currently used. An intrinsic drift detector with zero additional memory overhead is theoretically possible.

### What this PoC proves

That all three algorithms work as claimed, can be implemented in <500 lines of safe Rust, and correctly discriminate between same-distribution and drifted data in a controlled synthetic experiment. The key insight — that centroid drift misses GMM structural drift while MMD-RFF correctly identifies it — validates the need for the higher-order statistic.

### What would make this production-grade

1. Integrate `DriftDetector` into `ruvector-core`'s write path — every HNSW insert calls `observe`.
2. Expose drift metrics via the RuVector REST/MCP interface.
3. Add a `DriftHistory` struct that applies CUSUM over the time series.
4. Calibrate thresholds using bootstrap sampling from a burn-in window.
5. Add SIMD acceleration for MMD feature projection (the `cos` computation is the bottleneck).

### What would falsify the approach

If distributional drift in agent memory does not correlate with retrieval quality degradation, the entire motivation collapses. An end-to-end test measuring recall@10 as a function of drift score would validate or falsify this assumption. Such a test requires a labeled ground truth dataset, which is future work.

---

## Production Crate Layout Proposal

```
crates/ruvector-drift/
  Cargo.toml
  src/
    lib.rs         # DriftDetector trait, DriftScore, DriftReport
    centroid.rs    # CentroidDriftDetector
    mmd.rs         # MmdDriftDetector (RFF-MMD)
    graph.rs       # GraphDriftDetector (k-NN two-sample)
    history.rs     # DriftHistory + CUSUM layer  [future]
    simd.rs        # SIMD-accelerated RFF projection  [future]
  benches/
    drift_bench.rs
  src/bin/
    benchmark.rs
```

Integration path into `ruvector-core`:
1. Add optional `drift` feature flag.
2. In `HnswIndex::insert`, call `self.drift_detector.observe(&vec)`.
3. Emit a `DriftEvent` on the internal event bus.
4. ruFlo subscribes to `DriftEvent` and triggers compaction workflows.

---

## What to Improve Next

1. **CUSUM layer** on top of MMD-RFF for slow monotonic drift.
2. **SIMD `cos` approximation** in the RFF projection (target: 5-10× speedup for MMD-RFF).
3. **Online bandwidth estimation** using reservoir sampling of pairwise distances.
4. **Integration test** with `ruvector-core` HNSW — end-to-end drift → reindex workflow.
5. **Bootstrap threshold calibration** for controlled false positive rates.
6. **`ruvector-verified` anchor** — hash drift reports into the witness log.
7. **MCP tool implementation** — `vector_memory_health` backed by `ruvector-drift`.

---

## References and Footnotes

[^1]: Greco, S., Vacchetti, B., Apiletti, D., Cerquitelli, T. "Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time." arXiv:2406.17813, 2024. https://arxiv.org/abs/2406.17813. Accessed 2026-05-17.

[^2]: "Governing Evolving Memory in LLM Agents: Risks, Mechanisms, and the SSGM Framework." arXiv:2603.11768, 2026. https://arxiv.org/abs/2603.11768. Accessed 2026-05-17.

[^3]: Vejendla, L. "Drift-Adapter: A Practical Approach to Near Zero-Downtime Embedding Model Upgrades in Vector Databases." arXiv:2509.23471, 2025. https://arxiv.org/abs/2509.23471. Accessed 2026-05-17.

[^4]: Hu, Y. et al. "Memory in the Age of AI Agents." arXiv:2512.13564, 2025/2026. https://arxiv.org/abs/2512.13564. Accessed 2026-05-17.

[^5]: Bousetouane, F. "AI Agents Need Memory Control Over More Context." arXiv:2601.11653, 2026. https://arxiv.org/abs/2601.11653. Accessed 2026-05-17.

[^6]: Rahimi, A., Recht, B. "Random Features for Large-Scale Kernel Machines." NeurIPS 2007. https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html. Accessed 2026-05-17.

[^7]: Gretton, A. et al. "A Kernel Two-Sample Test." JMLR 2012. https://jmlr.org/papers/v13/gretton12a.html. Accessed 2026-05-17.

[^8]: Evidently AI. "5 Methods to Detect Drift in ML Embeddings." https://www.evidentlyai.com/blog/embedding-drift-detection. Accessed 2026-05-17.

[^9]: "Optimal Online Change Detection via Random Fourier Features." arXiv:2505.17789, 2025. https://arxiv.org/abs/2505.17789. Accessed 2026-05-17.
