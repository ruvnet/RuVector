# 00 - Master Plan: Connectome-Driven Embodied Brain on RuVector

**Coordinator:** goal-planner (GOAP)
**Branch:** `research/connectome-ruvector`
**Status:** Research + Design (pre-ADR)
**Date:** 2026-04-21

## 1. Positioning (binding on every downstream doc)

This research program designs a **graph-native embodied connectome runtime with structural coherence analysis, counterfactual circuit testing, and auditable behavior generation**. It is explicitly **not** a mind-upload product, a consciousness-upload product, or a claim about subjective experience. Any downstream document or code that drifts toward such framing must be flagged and rewritten. See `07-positioning.md` for the full hype-avoidance rubric.

The scientific grounding is the 2024 Nature whole-fly-brain LIF paper (behavior emerging from connectome-only leaky integrate-and-fire dynamics) and the Eon / NeuroMechFly embodiment line of work. The substrate under study is RuVector: a Rust-first graph+vector runtime with ~123 crates covering graph primitives (`ruvector-mincut`, `ruvector-sparsifier`, `ruvector-cnn`, `ruvector-solver`, `ruvector-graph`), vector memory (AgentDB / DiskANN Vamana / HNSW / ONNX all-MiniLM-L6-v2 384-dim), neural infrastructure (`ruvector-attention` SDPA, SONA, ReasoningBank, 9 RL algorithms), and a production brain service at pi.ruv.io storing ~13K memories and 1.2M graph edges.

## 2. Primary goal

> **G0: Produce a credible, buildable specification for a fully-RuVector-native embodied connectome runtime that ingests FlyWire, simulates connectome-constrained LIF dynamics inside a physics-sim body, and uses RuVector's graph primitives to discover motifs, detect coherence-collapse events, and run counterfactual circuit cuts — without any claim of consciousness or upload.**

### Acceptance criteria for G0

- [A1] Full 4-layer architecture described in `01-architecture.md` with interfaces, data flow, failure modes, and crate mapping.
- [A2] A concrete plan to import FlyWire (~139K nodes, 50M+ edges) into a RuVector graph, with schema, storage sizing, and ingest throughput estimate.
- [A3] A Rust crate design for an event-driven LIF kernel with delays and conductance models, benchmarked conceptually against Brian2/GeNN/NEST.
- [A4] A selection between NeuroMechFly, MuJoCo MJX, Brax, and Isaac Gym for the embodiment layer, with a motor-neuron → joint-torque contract.
- [A5] Application of existing RuVector primitives (mincut, sparsifier, spectral CNN, DiskANN/HNSW) to live connectome analysis with concrete hooks (boundary events, coherence collapse, trajectory compression, counterfactuals).
- [A6] Prior-art map (2024 Nature, FlyWire, hemibrain, NeuroMechFly, Eon, OpenWorm, Blue Brain, FFN) with overlap/differentiation.
- [A7] Product-and-science framing (`07-positioning.md`) + phased build plan (`08-implementation-plan.md`) with go/no-go gates.

## 3. Goal tree (GOAP decomposition)

```
G0: Embodied connectome runtime spec
├── G1: Data substrate
│   ├── G1.1 FlyWire ingest pipeline (→ 02-connectome-layer.md)
│   │   Preconds: FlyWire public release accessible; ruvector-graph schema extensible
│   │   Effects:  139K nodes + 50M edges persisted in GraphDB with NT/region/morphology/weight
│   ├── G1.2 Typed node/edge schema for neurons/synapses (→ 02 §Schema)
│   │   Preconds: NodeBuilder / EdgeBuilder in ruvector-graph supports rich properties
│   │   Effects:  Queryable by neuron type, NT, region, edge weight, delay, sign
│   └── G1.3 Storage sizing + ingest throughput budget (→ 02 §Cost)
│       Preconds: rvf on-disk format; sql.js or native RocksDB backend selected
│       Effects:  Documented RAM/SSD budget, deterministic reproducibility
├── G2: Neural dynamics engine
│   ├── G2.1 Event-driven LIF kernel crate design (→ 03-neural-dynamics.md)
│   │   Preconds: G1 schema available; time-wheel or priority-queue data structure chosen
│   │   Effects:  Rust crate `ruvector-lif` spec with O(k log n) event dispatch
│   ├── G2.2 Synaptic delay + conductance model (→ 03 §Model)
│   │   Preconds: Per-edge delay/weight/sign available from FlyWire
│   │   Effects:  Conductance-based LIF with AMPA/GABA/NMDA channels or graded weights
│   └── G2.3 Comparison to Brian2/GeNN/NEST (→ 03 §Comparison)
│       Preconds: Published benchmarks available
│       Effects:  Positioning: Rust event-driven + graph-native, not replacement for GPU sims
├── G3: Embodiment
│   ├── G3.1 Simulator selection (→ 04-embodiment.md §Selection)
│   │   Preconds: Need articulated insect body, contact, vision, proprioception
│   │   Effects:  Primary choice (NeuroMechFly on MuJoCo), fallback (Brax), ruled-out (Isaac)
│   ├── G3.2 Motor-neuron → joint-torque contract (→ 04 §Motor)
│   │   Preconds: LIF engine emits spike trains on flagged motor neurons
│   │   Effects:  Rate-coded or delta-coded torque signal; body closes the loop
│   └── G3.3 Sensory pipeline (vision, proprioception, contact) (→ 04 §Sensory)
│       Preconds: Simulator exposes raw sensor frames at fixed rate
│       Effects:  Encoding → sensory-neuron spike injection back into LIF kernel
├── G4: Analysis and adaptation layer
│   ├── G4.1 Live motif discovery via mincut/sparsifier (→ 05-analysis-layer.md §Motif)
│   │   Preconds: Dynamic graph backing store; ruvector-mincut handles streaming updates
│   │   Effects:  Hierarchical boundary tree updated in real time; motif library indexed in AgentDB
│   ├── G4.2 Coherence-collapse detection (→ 05 §Coherence)
│   │   Preconds: ruvector-coherence spectral tracker over dynamic Laplacian
│   │   Effects:  Real-time "neural fragility" signal tied to behavioral state
│   ├── G4.3 Trajectory compression via DiskANN/HNSW + attention (→ 05 §Trajectory)
│   │   Preconds: Spike-train windows embeddable as fixed-dim vectors
│   │   Effects:  Motif-indexed replay + search over behavioral episodes
│   └── G4.4 Counterfactual circuit surgery (→ 05 §Counterfactual)
│       Preconds: mincut identifies candidate boundaries; LIF engine supports edge masking
│       Effects:  Cut-and-replay experiments: which subgraph is load-bearing for which behavior
├── G5: Prior art and differentiation (→ 06-prior-art.md)
│   Preconds: Literature review
│   Effects:  Clear map of what is published / open / ours-only
├── G6: Positioning and venue (→ 07-positioning.md)
│   Preconds: G0 + G5 done
│   Effects:  Hype-avoidance rubric, audience plan, publication/venue plan
└── G7: Phased build plan (→ 08-implementation-plan.md)
    Preconds: G1-G6 done
    Effects:  M1-M5 milestones, crate additions, effort estimate, go/no-go gates
```

## 4. Action catalog (GOAP operators)

Each operator has preconditions, expected effects, and cost (rough engineering-week estimate).

| Action | Preconditions | Effects | Cost |
|---|---|---|---|
| `ingest_flywire` | FlyWire export + `ruvector-graph` schema | Persisted connectome graph | 1.0 |
| `design_lif_crate` | Schema + delay model | `ruvector-lif` crate stub | 1.5 |
| `impl_event_queue` | `design_lif_crate` done | Priority queue spike dispatcher | 1.0 |
| `impl_conductance_lif` | Event queue + per-edge sign/delay | Biophysical-ish LIF step | 1.0 |
| `wrap_mujoco_mjx` | NeuroMechFly MJCF + Rust FFI | Rust-controlled body sim | 2.0 |
| `define_motor_contract` | LIF spikes + sim torques | ABI between spikes and torques | 0.5 |
| `define_sensory_contract` | Sim sensor frames + sensory-neuron list | Spike injection ABI | 0.5 |
| `hook_mincut_live` | `ruvector-mincut` on dynamic graph | Streaming boundary tree | 1.0 |
| `hook_coherence` | `ruvector-coherence::spectral` + Laplacian view | Live coherence metric | 1.0 |
| `hook_diskann_trajectories` | AgentDB + ONNX embedder on spike windows | Indexed behavioral episodes | 1.0 |
| `counterfactual_surgery` | LIF edge mask API + mincut boundaries | Cut-replay experimental harness | 1.0 |
| `writeup_prior_art` | Literature access | `06-prior-art.md` | 0.5 |
| `writeup_positioning` | G0-G5 drafts | `07-positioning.md` | 0.5 |
| `writeup_impl_plan` | All above | `08-implementation-plan.md` | 0.5 |

Critical path cost: approximately 11 engineering-weeks for v0 milestone including body. This is a rough forward estimate for gating — actuals belong in `08-implementation-plan.md`.

## 5. Dependency DAG across the 8 sub-documents

```
           ┌────────────────┐
           │ 00-master-plan │  (this file)
           └───────┬────────┘
                   │
   ┌───────────────┼────────────────┬─────────────────┬─────────────┐
   ▼               ▼                ▼                 ▼             ▼
┌──────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐ ┌───────────┐
│ 01 arch  │ │ 02 connectome│ │ 03 neural   │ │ 04 embodiment│ │ 06 prior  │
│          │ │ layer        │ │ dynamics    │ │              │ │ art       │
└──┬───────┘ └──────┬───────┘ └──────┬──────┘ └──────┬───────┘ └─────┬─────┘
   │                │                │               │               │
   │                └──────┬─────────┘               │               │
   │                       ▼                         │               │
   │               ┌────────────────┐                │               │
   └──────────────▶│ 05 analysis    │◀───────────────┘               │
                   │ layer          │                                │
                   └────────┬───────┘                                │
                            │                                       │
                            ▼                                       ▼
                   ┌────────────────────────────────────────────────────┐
                   │ 07 positioning (uses 01-06 as inputs)              │
                   └────────────────────────┬───────────────────────────┘
                                            ▼
                               ┌──────────────────────────┐
                               │ 08 implementation plan   │
                               └──────────────────────────┘
```

Parallelization rule: 01-06 are independent; they read the brief (`README.md`) and this master plan, nothing else. 07 depends on 01-06. 08 depends on 01-07. This is the structure any swarm dispatch must respect.

## 6. Milestone schedule (M1-M5)

The brief implies 5 phases. We map them to concrete go/no-go milestones here; detailed task lists live in `08-implementation-plan.md`.

### M1 — Substrate lock (Phase 1: Data + schema)

- **Exit criteria:** FlyWire fully imported into `ruvector-graph` backed by rvf on SSD; queryable by neuron type, NT, region; node/edge counts reconciled against FlyWire release notes.
- **Gate:** Can we answer "list all mushroom-body Kenyon cells and their downstream glutamatergic partners" in under 50 ms?
- **Risk-out:** Schema inadequate → redesign before M2.

### M2 — Dynamics lock (Phase 2: LIF kernel in isolation)

- **Exit criteria:** `ruvector-lif` runs a 10,000-neuron connectome-constrained LIF for 60 simulated seconds on CPU, single-threaded, deterministic.
- **Gate:** Can we reproduce a published feeding-circuit response qualitatively from connectome alone, with no synthetic training?
- **Risk-out:** Event queue throughput inadequate → time-stepped fallback for large scales.

### M3 — Body lock (Phase 3: Embodiment)

- **Exit criteria:** LIF kernel drives NeuroMechFly body (via MuJoCo MJX through a Rust FFI bridge); sensory frames feed back; closed-loop for 30 s without numerical instability.
- **Gate:** Does a grooming-like motor pattern emerge when the relevant descending neurons are activated?
- **Risk-out:** FFI latency crushes the loop → reduce sim step rate or drop visual resolution.

### M4 — Analysis lock (Phase 4: RuVector primitives applied)

- **Exit criteria:** `ruvector-mincut` streams boundary updates during a run; `ruvector-coherence::spectral` produces a per-second coherence score; DiskANN indexes spike-window embeddings; a cut-and-replay experiment shows a behaviorally meaningful change.
- **Gate:** Does coherence collapse precede a behavioral state transition in the replay dataset?
- **Risk-out:** Signal noisy → tune sparsifier/window sizes before claiming novelty.

### M5 — Publication-grade demo (Phase 5: External release)

- **Exit criteria:** End-to-end pipeline reproducible from a single `cargo run` + data download; paper-quality figures of motif discovery, coherence-collapse events, and counterfactual circuit cuts; clean README explaining this is **not** upload/consciousness work.
- **Gate:** Does an outside neuroscientist accept the framing and find the substrate story credible?
- **Risk-out:** Positioning drifts into hype → rewrite using the `07-positioning.md` rubric.

## 7. Risk register

| ID | Category | Risk | Mitigation | Owner |
|---|---|---|---|---|
| R1 | Technical | Event-driven LIF can't hit real-time for 139K neurons on CPU | Parallelize per region (`rayon`), profile hotspots, GPU fallback via `wgpu` if needed | `ruvector-lif` author |
| R2 | Technical | FlyWire edge weights/delays are not uniformly present | Default to uniform delay, sign from NT; mark edges as `weight_source: {explicit, nt_default}` in schema | connectome-layer |
| R3 | Technical | MuJoCo MJX has no first-class Rust binding | Build a thin `cxx` or `cbindgen`-backed bridge; consider Brax-on-JAX detour only if necessary | embodiment |
| R4 | Technical | Dynamic mincut on a graph of this density is expensive | Use sparsifier first, maintain mincut on H (the sparsifier), not G | analysis-layer |
| R5 | Scientific | LIF-from-connectome reproduction fails for our chosen behaviors | Anchor to behaviors the 2024 Nature paper already demonstrated before claiming new ones | lead scientist |
| R6 | Scientific | Coherence-collapse signal is a graph artifact, not a behavioral predictor | Require paired null analysis (shuffled connectome control) before publishing | analysis-layer |
| R7 | Scientific | Counterfactual cuts change too much to be interpretable | Constrain cuts to `ruvector-mincut` boundaries with witness audit trails | analysis-layer |
| R8 | Positioning | External readers interpret the work as "digital minds" | Apply `07-positioning.md` rubric to every README/paper/tweet | all |
| R9 | Positioning | Overclaim against Eon / NeuroMechFly | Use `06-prior-art.md` differentiation table verbatim in external comms | lead |
| R10 | Scope | Feature creep pulls in mammalian connectomes | Lock to Drosophila/FlyWire for v1; mammalian is v2 or never | PM |
| R11 | Scope | Build competes with `crates/ruvector-consciousness` / `ruvector-nervous-system` | Use those as prior art inside RuVector; this project stays bounded to the 4-layer architecture | architect |
| R12 | Operational | Project absorbed into ADR backlog before research is done | Keep branch `research/connectome-ruvector` isolated, no new ADRs in-branch | coordinator |
| R13 | Data | FlyWire license / attribution missed | Cite FlyWire explicitly, no derivative distribution of the dataset | data owner |

## 8. Success criteria per phase

- **Phase 1:** 100% of public FlyWire release imported, round-trip tested, queried under fixed budget.
- **Phase 2:** ≥10K neuron LIF reproduces published circuit response qualitatively.
- **Phase 3:** Closed-loop body sim runs 30 s at ≥25 Hz control rate without divergence.
- **Phase 4:** Coherence-collapse precedes ≥70% of behavioral state transitions in replay (with shuffled-null p<0.01).
- **Phase 5:** Reproducible one-command demo, neutral-language README, one submitted preprint.

## 9. Guarantees and guardrails binding on every sub-doc

1. **Rust only.** No Python, no JS for the kernel. Tooling scripts stay out of `crates/` and are not part of the runtime.
2. **Cite the brief and/or the 2024 Nature paper at least once per doc.**
3. **Zero consciousness/upload language.** Flag and rewrite any prose that implies subjective experience, mind transfer, or sentience.
4. **All outputs under `docs/research/connectome-ruvector/`.** No root files, no new ADRs, no cross-directory spillover.
5. **Cross-link via relative paths** (`./02-connectome-layer.md` etc.).
6. **No synthetic training of behavior.** Behavior emerges from connectome + dynamics + body, never from backprop.
7. **Public data only.** FlyWire public release. No proprietary connectomes.
8. **File-size discipline.** Docs stay under ~500 lines (project convention). Long appendices split into sections, not new files.

## 10. Handoff

`01`-`06` can be written in parallel. `07` depends on `01`-`06`. `08` depends on `07`. The coordinator (this doc's author) is responsible for writing `08` after the specialists return, for committing all nine files in a single commit, and for **not** pushing — the user reviews first.
