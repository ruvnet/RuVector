# 01 - Architecture: Four-Layer Embodied Connectome Runtime

> Framing reminder (binding): this document specifies a **graph-native embodied connectome runtime**. It is not a mind-upload, consciousness-upload, or sentience product. See `./00-master-plan.md` §1 and `./07-positioning.md`.

## 1. Purpose

Specify the four-layer system design required to run a connectome-constrained embodied simulation on RuVector. Define every inter-layer interface, the data-flow envelope, the failure modes, and the exact RuVector crate that plugs into each seam. This document is read by `03`, `04`, `05`, and `08`; anything they specify must fit within the contracts defined here.

## 2. Layer overview

The system is four concentric layers around a dynamic graph:

```
                 ┌──────────────────────────────────────────────────┐
                 │  Layer 4: Analysis & Adaptation                  │
                 │  mincut boundaries · coherence tracker ·         │
                 │  DiskANN/HNSW trajectory index · counterfactual  │
                 │  surgery harness                                 │
                 └───────────────▲───────────────┬──────────────────┘
                                 │ read graph    │ write masks / cuts
                                 │ read spikes   │
                 ┌───────────────┴───────────────▼──────────────────┐
                 │  Layer 2: Neural Dynamics (ruvector-lif, new)    │
                 │  event-driven LIF · synaptic delays ·            │
                 │  conductance model · spike queue                 │
                 └───────────────▲───────────────┬──────────────────┘
                                 │ graph ref     │ motor spikes
                                 │ sensory in    │
                 ┌───────────────┴───────────────▼──────────────────┐
                 │  Layer 1: Connectome / State Graph               │
                 │  ruvector-graph + rvf on-disk + AgentDB vectors  │
                 └───────────────▲───────────────┬──────────────────┘
                                 │ persist       │ motor torques
                                 │ vectors       │ sensor frames
                 ┌───────────────┴───────────────▼──────────────────┐
                 │  Layer 3: Embodiment (external sim + Rust bridge)│
                 │  NeuroMechFly / MuJoCo MJX · proprioception ·    │
                 │  contact · compound-eye vision                   │
                 └──────────────────────────────────────────────────┘
```

Layers 1 and 2 are the RuVector core. Layer 3 is external-sim-plus-bridge. Layer 4 is RuVector analysis riding on the live state of layers 1 and 2. The picture is deliberately not a stack: layer 2 and layer 3 form the closed sensorimotor loop, and layer 4 is a side-channel observer and interventionist.

## 3. Layer 1 — Connectome / state graph

**Substrate:** `ruvector-graph` for topology + ACID transactions, `ruvector-core` / AgentDB for embeddings and memory, rvf for on-disk format, DiskANN Vamana for vector search at scale.

**Data model (enforced by schema in `./02-connectome-layer.md`):**

- `Node` = neuron, with properties `{id, type, region, neurotransmitter, morphology_hash, source_dataset, flags}`.
- `Edge` = synapse (or compartment link), with properties `{pre_id, post_id, weight, sign, delay_ms, nt, count, confidence, source}`.
- Optional `Hyperedge` = a functional motif (e.g., winner-take-all triplet) discovered by layer 4 and written back.

**Public interface (what layers 2 and 4 see):**

```rust
pub trait Connectome: Send + Sync {
    fn neuron(&self, id: NeuronId) -> Option<NeuronView<'_>>;
    fn outgoing(&self, id: NeuronId) -> EdgeIter<'_>;
    fn incoming(&self, id: NeuronId) -> EdgeIter<'_>;
    fn by_region(&self, region: RegionId) -> NeuronIter<'_>;
    fn by_nt(&self, nt: Neurotransmitter) -> NeuronIter<'_>;
    /// Readable snapshot of the current adjacency for mincut/spectral analysis.
    fn snapshot(&self) -> ConnectomeSnapshot;
    /// Edge-mask API used by the counterfactual harness (layer 4).
    fn apply_mask(&mut self, mask: &EdgeMask) -> Result<MaskHandle, CError>;
    fn remove_mask(&mut self, handle: MaskHandle) -> Result<(), CError>;
}
```

**Persistence:** rvf-backed GraphDB, with a cold-path export to Parquet for external reproducibility. AgentDB keeps parallel per-neuron embeddings (384-dim, ONNX all-MiniLM-L6-v2) for semantic queries such as "find me neurons whose morphology is close to mushroom-body Kenyon cells." Vector search is handled by DiskANN Vamana (see `crates/mcp-brain/` prior art and ADR-144, ADR-146).

**Why RuVector here:** `ruvector-graph` already supports typed labels, properties, ACID transactions, indexes, and hybrid graph+vector queries (`hybrid::HybridIndex`, `GraphNeuralEngine`, `RagEngine`). That alignment is dense enough that the connectome project should subclass it rather than reinvent a neuron store.

## 4. Layer 2 — Neural dynamics

**Substrate:** new crate `ruvector-lif` (proposed in `./03-neural-dynamics.md`), optional reuse of `ruvector-nervous-system::dendrite` for dendritic coincidence detection where connectome resolution supports it.

**Execution model:** event-driven leaky integrate-and-fire. Each spike produces a future event scheduled at `now + edge.delay_ms` for each downstream neuron. Events are dispatched from a priority queue (binary heap or hierarchical time wheel — tradeoff analyzed in `./03`). The core loop is:

```rust
loop {
    let Some(event) = queue.pop_due(now) else {
        advance_simulation_clock(step);
        continue;
    };
    match event {
        Event::Spike { pre, post, weight, sign } => {
            let neuron = &mut neurons[post];
            neuron.integrate_ps(weight * sign, now);
            if neuron.crossed_threshold(now) {
                emit_spike(post, now);
                for edge in graph.outgoing(post) {
                    queue.push(Event::Spike {
                        pre: post, post: edge.target,
                        weight: edge.weight, sign: edge.sign,
                    }, now + edge.delay_ms);
                }
                neuron.reset();
            }
        }
        Event::SensoryInjection { neuron_id, current } => { /* ... */ }
    }
}
```

**Published interface:**

```rust
pub trait DynamicsEngine: Send + Sync {
    fn step(&mut self, dt_ms: f32) -> StepReport;
    fn inject_sensory(&mut self, id: NeuronId, current_pa: f32, at: Time);
    fn drain_motor_spikes(&mut self) -> Vec<MotorSpike>;
    fn subscribe_spikes(&mut self) -> SpikeStream;     // tap for layer 4
    fn subscribe_voltage(&mut self, ids: &[NeuronId]) -> VoltageStream;
    fn apply_edge_mask(&mut self, m: &EdgeMask) -> Result<MaskHandle, DError>;
}
```

`DynamicsEngine::step` advances up to `dt_ms` of simulated time. `SpikeStream` is a lock-free MPMC channel (AgentDB patterns from `crates/ruvector-nervous-system::eventbus` apply directly — they quote 10K events/ms throughput). The taps are read-only; layer 4 observes without stalling layer 2.

**Why a new crate, not `ruvector-nervous-system`:** the existing `ruvector-nervous-system::snn` and dendrite modules are biology-inspired primitives, not a connectome-scale event-driven integrator. The project-specific crate layout is justified in `./03-neural-dynamics.md` §4.

## 5. Layer 3 — Embodiment

**Substrate:** NeuroMechFly model running on MuJoCo MJX, wrapped in a `cxx`-backed Rust crate `ruvector-embodiment` (see `./04-embodiment.md`). Fallbacks: Brax (JAX) via process boundary; Isaac Gym excluded (Python/GPU-lock, license).

**Published interface:**

```rust
pub trait BodySim: Send + Sync {
    fn step(&mut self, torques: &[Torque]) -> BodyObservation;
    fn reset(&mut self);
    fn sensor_schema(&self) -> &SensorSchema;
    fn motor_schema(&self) -> &MotorSchema;
}

pub struct BodyObservation {
    pub time: Time,
    pub joint_positions: Vec<f32>,
    pub joint_velocities: Vec<f32>,
    pub contact_forces: Vec<ContactForce>,
    pub compound_eye: Option<VisionFrame>,
    pub antennae_chemistry: Option<ChemFrame>,
}
```

**Motor-neuron → torque contract:** spikes from the set of motor neurons flagged in layer 1 (FlyWire cell-type labels `Motor*`) are rate-coded over a 10 ms window, passed through a per-joint linear gain, and emitted as joint torques. Contract and alternatives (delta-coding, PID wrapper) are analyzed in `./04-embodiment.md` §Motor.

**Sensory → spike injection contract:** vision frames (compound eye, ommatidia count depending on the fly model) are encoded into spike currents injected into the photoreceptor neurons. Proprioception → chordotonal neurons. Contact → mechanosensory neurons. The encoder is connectome-neutral: it injects current, the LIF kernel decides what fires.

## 6. Layer 4 — Analysis and adaptation

**Substrate:** existing RuVector crates, no new code required for the first pass:

- `ruvector-mincut` — dynamic min-cut with subpolynomial-time updates (`crates/ruvector-mincut/src/algorithm/*`, `canonical/dynamic`). Boundary events become behavioral-state-transition candidates.
- `ruvector-sparsifier` — spectral sparsifier keeping Laplacian energy within `(1 ± ε)`. Lets mincut run on a 100×-smaller graph without destroying the signal (`crates/ruvector-sparsifier/src/sparsifier.rs`).
- `ruvector-coherence::spectral` — Fiedler / spectral-gap / effective-resistance estimators used as a live "neural fragility" score (`crates/ruvector-coherence/src/spectral.rs`).
- `ruvector-attention` (SDPA, sparse, graph) — encode spike-window trajectories into embeddings for motif indexing.
- AgentDB + DiskANN Vamana — long-term motif and trajectory store with 384-dim ONNX embeddings.
- `ruvector-solver` — iterative sparse solver for effective-resistance / PageRank-style diffusions on the dynamic graph (`crates/ruvector-solver/src/neumann.rs`, `cg.rs`).

**Feedback directions:**

- Observational: boundary events, coherence scores, motif hits are written to AgentDB and surfaced to the operator.
- Interventional: `EdgeMask` and `RegionMask` objects are pushed into layer 1 (graph mask) and layer 2 (engine mask) simultaneously, producing a counterfactual run. Witnesses from `ruvector-mincut::certificate` and `ruvector-mincut::witness` are attached so every cut is auditable.

## 7. Data flow

Steady-state frame (one simulation tick, ~4 ms wall clock target for 25 Hz control rate):

```
 (body)            (sensory          (LIF           (motor        (body)
 observation  →    encoder)     →    engine)   →    decoder) →    torques
  @Layer 3       @Layer 2         @Layer 2       @Layer 2        @Layer 3
      │                                │                              ▲
      │  spike tap                     │                              │
      └─────────────────────────► (Layer 4) ──── writeback motifs ───┘
                                       │                              │
                                       └─── optional masks ───────────┘
```

Three hazards matter: (a) the body tick is faster than the LIF tick — the embodiment bridge must buffer; (b) layer 4 must never block layer 2 — spike and voltage streams are bounded ring buffers with drop-on-full-with-warning semantics; (c) edge masks must be applied atomically to both layer 1 and layer 2, enforced by a single `apply_mask_everywhere` orchestrator that holds the graph transaction open until the engine confirms.

## 8. Failure modes

| Mode | Where | Symptom | Detection | Response |
|---|---|---|---|---|
| Event queue blowup | L2 | RAM spike, dispatch lag | queue depth metric > budget | Back off to time-stepped fallback for this region |
| Numerical divergence | L2 | voltage → ±∞ | finite-check on every integrate | Clamp + emit warning, do not silently NaN |
| Sim divergence | L3 | contact solver explodes | MuJoCo warning log | Reset episode, preserve last good checkpoint |
| Bridge latency | L3↔L2 | control rate < 10 Hz | rolling wall-clock | Drop vision resolution first, then proprioception |
| Mask desync | L1/L2 | masked edge still firing | mask-audit spot check | Rollback to pre-mask snapshot |
| Sparsifier drift | L4 | audit.max_error > 2ε | `SpectralAuditor` | Rebuild sparsifier from scratch |
| Coherence false alarm | L4 | many "collapse" events per second | shuffled-null control | Raise threshold or add debounce |
| Memory pressure | L1 | AgentDB eviction cascade | pi-brain patterns (ADR-149) | Quantize embeddings (4-bit), tier to SSD |

## 9. Crate mapping

| Layer | Role | Existing crate | New crate |
|---|---|---|---|
| 1 | Graph store | `ruvector-graph`, `ruvector-core`, rvf | `ruvector-connectome` (schema + importers) |
| 1 | Vector memory | AgentDB (`crates/mcp-brain`, `ruvector-core`) | — |
| 2 | LIF kernel | — | `ruvector-lif` |
| 2 | Dendrites (optional) | `ruvector-nervous-system::dendrite` | — |
| 2 | HDC / eventbus | `ruvector-nervous-system::eventbus` | — |
| 3 | Body sim bridge | — | `ruvector-embodiment` (cxx to MuJoCo MJX) |
| 4 | Boundary finder | `ruvector-mincut` | — |
| 4 | Sparsifier | `ruvector-sparsifier` | — |
| 4 | Coherence tracker | `ruvector-coherence` (spectral feature) | — |
| 4 | Trajectory index | AgentDB + DiskANN + `ruvector-attention` | `ruvector-connectome-traces` (thin) |
| 4 | Counterfactual harness | `ruvector-mincut::certificate` | `ruvector-connectome-cuts` (thin) |

Two first-party new crates (`ruvector-lif`, `ruvector-embodiment`) and three thin wrappers (`-connectome`, `-connectome-traces`, `-connectome-cuts`). Everything else is re-use. This crate footprint is the minimum that delivers the four-layer architecture without treading on existing RuVector scope.

## 10. Determinism and reproducibility

- **L1** is deterministic by construction (ACID transactions, content-addressed rvf file).
- **L2** is deterministic given a fixed seed for any stochastic component (e.g., membrane noise). The event queue is order-deterministic if ties are broken by `(time, pre_id, post_id)`.
- **L3** is deterministic only in MuJoCo's deterministic mode; we require it. Brax is deterministic per-seed.
- **L4** is deterministic once L1-L3 are. Sparsifier / DiskANN seeds are logged.

A full run is reproducible from `(dataset_hash, connectome_schema_ver, engine_config, body_config, seed_vector)`. These five values go into a manifest written alongside every replay bundle. Replay is `./examples/` territory, not a runtime feature.

## 11. Why this architecture fits RuVector

The 2024 Nature whole-fly-brain paper demonstrated that behavior can be reproduced from connectome-only LIF dynamics without trained parameters. That result pivots the scientific case: the substrate must be **graph-first**, not tensor-first. RuVector is already graph-first (`ruvector-mincut`, `ruvector-sparsifier`, `ruvector-graph`, `ruvector-solver` over CSR), and its vector stores (AgentDB, DiskANN, HNSW) are side-channels bolted to the graph rather than the central object. That inversion — graph as the primary data structure, vectors as a view — is what this architecture is designed around, and it is what makes the connectome runtime story natural rather than forced.

See `./02-connectome-layer.md` for the graph schema, `./03-neural-dynamics.md` for the LIF kernel, `./04-embodiment.md` for the body sim choice, `./05-analysis-layer.md` for the analysis hooks, `./06-prior-art.md` for differentiation, and `./08-implementation-plan.md` for the phased build.
