# 05 - Analysis Layer: RuVector Primitives Applied to a Live Connectome

> Framing reminder: this is structural coherence analysis and auditable circuit-level intervention on a graph-native embodied connectome runtime. It is not a model of consciousness, upload, or sentience. See `./00-master-plan.md` §1 and `./07-positioning.md`.

## 1. Purpose

Show — concretely — how existing RuVector graph and vector primitives plug into a running connectome simulation to do novel work: discover motifs, detect subgraph boundary crossings, surface coherence-collapse events, compress spike-train trajectories, and run counterfactual circuit surgery. This is the layer that makes the RuVector substrate story more than "you could use any graph DB." Consumers: `./01-architecture.md` (which defines the seams), `./07-positioning.md` (which cites this as the differentiating layer), `./08-implementation-plan.md`.

The scientific ground is the 2024 Nature whole-fly-brain LIF paper: that behavior emerges from connectome-alone dynamics means the connectome graph is itself the load-bearing object. Analyses over that graph, in real time, are scientifically meaningful if they track or predict behavioral transitions.

## 2. What we already have

The following RuVector crates are directly reusable. I list the specific modules that apply.

- `ruvector-mincut` — dynamic min-cut with subpolynomial-time updates (`canonical/dynamic`), tree packing, sparsifier-backed (1+ε) approximation, certificates/witnesses, hierarchical decomposition (`jtree`, `cluster/hierarchy`). Relevant modules: `algorithm/`, `canonical/`, `cluster/`, `snn/` (small-network notions), `certificate/audit`, `witness/`, `jtree/`, `sparsify/`. Documented in its `lib.rs` with a subpolynomial-time guarantee for exact cuts and a `(1+ε)` approximate algorithm with SpectralAuditor-style drift detection.
- `ruvector-sparsifier` — dynamic spectral sparsification (ADKKP16) preserving Laplacian energy within `(1 ± ε)`. Modules: `backbone` (spanning forest), `importance` (effective-resistance estimates), `sampler`, `audit` (`SpectralAuditor` with max-error drift metric).
- `ruvector-coherence` (with `spectral` feature) — Fiedler estimation, spectral gap, effective-resistance sampling, degree regularity, largest eigenvalue; `SpectralTracker` and `HnswHealthMonitor` for live metrics. Metrics module: delta-behavior, contradiction rate, entailment consistency, quality checks.
- `ruvector-attention` — SDPA, multi-head, graph attention, sparse patterns, FlashAttention-3 tiling, MLA, state-space (Mamba). Used here as a trajectory encoder.
- `ruvector-solver` — Neumann series, CG, forward/backward push, BMSSP, true solver; CSR matrix type. Used for effective-resistance computation and personalized-PageRank-style diffusions.
- `ruvector-core` / AgentDB / DiskANN / HNSW — vector store, ONNX `all-MiniLM-L6-v2` 384-dim embedder, `ruvllm_hnsw_*` family, quantization (4/8-bit).
- `ruvector-cnn` — MobileNet-V3 Small/Large backbones, INT8, SIMD. Used here only if we move to mesh-based morphology embedding.
- `ruvector-nervous-system::eventbus` — lock-free ring-buffer + sharded bus quoted at 10K events/ms. Used as the analysis-side spike tap consumer.

Nothing on this list is new. The analysis-layer code we write is a thin orchestrator that mounts these primitives onto the live graph from `./02-connectome-layer.md` and the spike stream from `./03-neural-dynamics.md`.

## 3. Motif discovery

**Goal:** maintain an indexed library of recurrent subgraph motifs (winner-take-all triplets, feedforward inhibition triads, reciprocal pairs, recurrent loops) that are currently active under a behavior.

**Pipeline:**

1. `ruvector-sparsifier::AdaptiveGeoSpar::build(&G, cfg)` produces a Laplacian-preserving sparsifier `H` of the connectome with ~50-100× edge reduction.
2. `ruvector-mincut::jtree` builds a hierarchical decomposition of `H`. Its clusters (`cluster/hierarchy`) give us candidate motif neighborhoods.
3. For each small cluster (≤7 nodes) we enumerate isomorphism classes against a fixed motif vocabulary (3-node FFI, FF, cycle; 4-node diamond; etc.). This is fast because the clusters are small.
4. Found motifs are written back as `ruvector-graph::Hyperedge` with `MotifKind`.
5. AgentDB indexes a 384-dim embedding of each motif's cell-type signature via ONNX `all-MiniLM-L6-v2` so "motifs similar to this one" is a `k-NN` query on DiskANN.

**Why this is novel:** most motif analyses in connectomics are offline and on the full static graph. Running motif discovery on a live sparsifier means the motif library is maintained dynamically; as edge masks are applied (counterfactuals) or as incremental updates land (new FlyWire release), the set of active motifs is re-indexed automatically. The cost is bounded because the sparsifier is small.

**Costs:** sparsifier rebuild on delta is `O(|delta| · polylog(|G|))`; jtree update on the sparsifier is subpolynomial-time per edge update. Motif enumeration in clusters of size ≤7 is `O(7! · |clusters|) = O(5040 · |clusters|)`, trivially parallel.

## 4. Boundary-crossing events as behavioral-transition candidates

**Goal:** detect the moments when the active subgraph crosses a min-cut boundary of the connectome, and test whether those crossings correlate with behavioral state changes (walking → grooming, resting → feeding).

**Pipeline:**

1. Layer 2's `SpikeStream` feeds a time-windowed **activity graph** `A_t`: the subgraph of `G` induced by edges `(pre, post)` that fired in the last W ms (W ≈ 50 ms). Weighted by spike count.
2. `ruvector-mincut::canonical::dynamic` maintains the min-cut of `A_t` as spikes flow in and age out.
3. When the min-cut value changes across a significant threshold, or when the active subgraph crosses from one side of the whole-brain mincut to the other, emit a `BoundaryEvent{time, from_region, to_region, cut_value, witness}`.
4. Behavioral labels from the body simulator (walking, grooming, feeding) are timestamped independently by `./04-embodiment.md`. We correlate.

**Null control:** shuffle spike times within each neuron, rebuild the activity graph, rerun the detector. The boundary-event rate against the shuffled null gives us a valid p-value.

**Why this is novel:** the boundary-crossing signal is defined directly on the connectome's static structure; behaviors are read from the body. If crossings predict behavioral state changes with `p < 0.01` against the shuffled null, we have a connectome-native precursor signal for behavior. `ruvector-mincut` with `ruvector-mincut::certificate::audit` gives us an audit trail for each detected event.

## 5. Coherence-collapse as a neural-fragility signal

**Goal:** surface a real-time signal for "the system is about to transition / destabilize" before it does.

**Pipeline:**

1. Build the Laplacian `L_t` of the sparsifier `H_t` of the activity graph every `T_c = 100 ms`.
2. Use `ruvector-coherence::spectral::estimate_fiedler` and `estimate_spectral_gap` on `L_t`. Use `estimate_effective_resistance_sampled` on a set of anchor neuron pairs (fixed across the run).
3. Compose a scalar **coherence score** `C_t = normalize(gap) - penalize(|dR_eff|)` where `dR_eff` is the rate of change of effective resistance between anchors.
4. Detect "collapse": `C_t` drops below a threshold set from a baseline distribution.
5. Emit `CoherenceCollapseEvent{time, c_value, fiedler, gap, reff_flux}`.

**Hypothesis under test:** coherence collapses precede behavioral-state transitions. The shuffled-null is the same as §4. If the paired rate (`P(transition | collapse within 200 ms)`) beats chance, the signal is real.

**Why this is novel:** spectral graph health is an *a priori* candidate predictor of state changes that the RuVector stack computes cheaply on a dynamic Laplacian (see `ruvector-coherence::spectral::SpectralTracker` and `HnswHealthMonitor` for the infrastructure; we repurpose them from HNSW health checks to connectome health checks, which is an obvious fit).

## 6. Trajectory compression via DiskANN/HNSW + attention

**Goal:** represent each behavioral episode as a compact vector indexed by a vector store, enabling replay, search, and clustering.

**Pipeline:**

1. Segment the run into episodes bounded by `BoundaryEvent`s and behavioral labels.
2. For each episode, build a spike-window tensor `X ∈ R^{T × N_active}` (time × active neurons), e.g., T = 100 × 10 ms bins.
3. Encode with `ruvector-attention::attention::ScaledDotProductAttention` over time, cell-type-pooled across neurons, to produce a fixed 384-dim vector `v_ep`.
4. Store `v_ep` in AgentDB's DiskANN index with episode metadata (time range, behavioral labels, motif hits, coherence dip magnitude).
5. Query: "show me all past episodes similar to this walking bout" = `DiskANN::search(v_ep, k=50)`.

**Novelty:** RuVector's DiskANN Vamana stack (see ADR-144 / ADR-146 and `crates/mcp-brain/`) already operates at the 1.2M-edge pi-brain scale in production. Using the same stack to index behavioral episodes gives us Jupyter-speed semantic replay without custom infrastructure. The attention encoder is reusable (`ruvector-attention` already exposes SDPA + sparse + graph variants).

## 7. Counterfactual circuit surgery

**Goal:** answer causal questions such as "is the α/β lobe of the mushroom body load-bearing for this feeding response?" by cutting edges and replaying.

**Pipeline:**

1. Choose a candidate cut: either (a) a mincut boundary surfaced by §4, (b) a motif surfaced by §3, or (c) a user-specified region.
2. Build an `EdgeMask` containing the cut edges and `weight_scale = 0.0`.
3. Call `Engine::apply_edge_mask(&mask)` (from `./03-neural-dynamics.md` §10) and `Connectome::apply_mask(&mask)` (from `./01-architecture.md` §3), atomically.
4. Re-run the episode from its recorded sensory input trace (replay semantics, see `./04-embodiment.md` §7).
5. Compare the replay against the original: behavioral divergence (body pose distance, spike-raster divergence via KL on binned rates, coherence-score delta, mincut topology delta).
6. Attach a `ruvector-mincut::witness` receipt to the cut so the experiment is audit-grade.

**Novelty:** this is connectome-level causal intervention with an auditable trail, not a trained perturbation. It requires exactly what RuVector already provides: fast edge masking, recomputable mincut, recomputable sparsifier, deterministic replay.

**Guardrails:** the `ruvector-mincut::certificate` module already emits witnesses for dynamic cuts. We surface them to the operator. Cuts without witnesses are not publishable.

## 8. Anomaly and drift monitors

RuVector's `ruvector-sparsifier::audit::SpectralAuditor` and `ruvector-coherence::HnswHealthMonitor` are runtime health components. We reuse them to watch:

- Sparsifier drift (`audit.max_error > 2ε`) — rebuild `H`.
- Coherence-tracker divergence — recalibrate baseline.
- DiskANN recall drop on the trajectory index — rebuild index.

These do not surface to the scientist; they surface to the operator. But they are cheap insurance against silently corrupted results.

## 9. Data flow

```
 ┌───────────────┐      tap        ┌───────────────────┐
 │ LIF engine    │ ─── spikes ───▶ │ activity-graph    │
 │ (L2)          │                 │ builder (windowed)│
 └───────────────┘                 └─────────┬─────────┘
                                             ▼
                            ┌────────────────────────────┐
                            │ sparsifier H_t             │ (ruvector-sparsifier)
                            └─────────────┬──────────────┘
                                          ▼
     ┌───────────────┐   ┌────────────────────┐   ┌──────────────────────┐
     │ mincut tree   │   │ spectral tracker   │   │ motif clusterer      │
     │ (L boundary)  │   │ (L coherence)      │   │ (L motif)            │
     └──────┬────────┘   └────────┬───────────┘   └──────┬───────────────┘
            │                     │                      │
            ▼                     ▼                      ▼
      ┌──────────┐           ┌──────────┐           ┌──────────┐
      │ Boundary │           │ Collapse │           │ Motif    │
      │ events   │           │ events   │           │ hits     │
      └─────┬────┘           └────┬─────┘           └────┬─────┘
            │                     │                     │
            ▼                     ▼                     ▼
      ┌───────────────────────────────────────────────────────┐
      │ AgentDB + DiskANN trajectory / event index            │
      └───────────────────────────────────────────────────────┘
                                          │
                                          ▼
                            ┌────────────────────────┐
                            │ counterfactual harness │
                            └───────────┬────────────┘
                                        ▼
                    EdgeMask pushed to L1 (graph) + L2 (engine)
```

## 10. Cost analysis

| Analysis | Hot path cost | Frequency | Amortized load |
|---|---|---|---|
| Activity graph build | O(spikes/window) | per control tick | small |
| Sparsifier update | O(|Δ|·polylog(|G|)) | per tick | small after warm |
| Mincut dynamic update | subpolynomial per edge | per tick | small-to-medium |
| Spectral coherence (Fiedler, gap) | O(k·|H|) k small | every 100 ms | small |
| Motif enumeration in small clusters | O(7!·|clusters|) | every 500 ms | small |
| Trajectory encoder (SDPA over episode) | O(T·N_active·d) | per episode boundary | medium |
| DiskANN insert | O(log |index|) | per episode | small |
| Counterfactual replay | as full run | on demand | large, offline |

None of these are on the critical latency path of the sensorimotor loop. The analysis layer runs in a side thread subscribed to the taps.

## 11. What is genuinely new here

1. **Dynamic mincut boundaries as behavioral-state precursors, with audit trails.** No prior connectomics project has published this; RuVector has the only open-source subpolynomial dynamic mincut with certificates.
2. **Coherence-collapse as a real-time fragility score on a connectome Laplacian.** Existing spectral analyses are offline.
3. **DiskANN-indexed behavioral-episode trajectories.** Cross-episode retrieval in seconds on the production-proven pi-brain stack (ADR-150).
4. **Auditable counterfactual circuit surgery.** Mincut witnesses plus deterministic replay make each cut-and-run experiment publishable.

The 2024 Nature paper showed the connectome is the right object; this analysis layer shows that RuVector is the right substrate to study it live. That is the pitch, and it lives here.

## 12. Open questions

1. Is the windowed activity graph the right object, or should we use a `τ`-decayed weighted graph? Try both; pick after M4.
2. Can the `SpectralAuditor` be reused verbatim on the Laplacian of `H_t` that we are rebuilding, or do we need a streaming variant? Likely verbatim; confirm in M4.
3. Is the motif vocabulary static (3-node FF/FFI/cycle, 4-node diamond) or mined? Static v1; mined v2.
4. DiskANN embedding dimension: 384 from ONNX MiniLM is standard; 768 if we move to `mpnet-base`. 384 is the safer choice for SSD footprint.
5. Null-control statistics: shuffled spike times is the default. Do we also want a rewired-null connectome (Erdős–Rényi preserving degree)? Yes, as a secondary control for motif-level claims.

See `./03-neural-dynamics.md` for the tap contract, `./04-embodiment.md` for behavioral labels, `./06-prior-art.md` for what prior work these analyses extend or differ from, and `./08-implementation-plan.md` for sequencing.
