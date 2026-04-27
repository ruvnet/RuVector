# 08 - Implementation Plan: Phased Build, Effort, and Go/No-Go Gates

> Framing reminder: this is a graph-native embodied connectome runtime. Not consciousness, not upload. See `./00-master-plan.md` §1 and `./07-positioning.md`.

## 1. Purpose

Translate `./01-architecture.md` through `./07-positioning.md` into a concrete phased build plan with crate additions, sequencing, dependencies, effort estimates, and go/no-go gates tied to the M1-M5 milestones from `./00-master-plan.md` §6. This doc binds the engineering side; `./07-positioning.md` binds the communication side.

The scientific reference regime is still the 2024 Nature whole-fly-brain LIF paper: every phase's success criterion is measured against the behaviors that paper established.

## 2. Crate additions

Two first-party new crates and three thin project wrappers. Nothing else under `crates/` needs to be touched beyond optional feature-flag additions.

| Crate | Role | Lines (approx est.) | Depends on |
|---|---|---|---|
| `crates/ruvector-connectome/` | Schema, importers (flywire-loader, hemibrain-loader), indexes, graph view | 2.0-3.0 kLOC | `ruvector-graph`, `ruvector-core` |
| `crates/ruvector-lif/` | Event-driven LIF kernel, timing wheel, slow pools, taps | 3.0-4.0 kLOC | `ruvector-connectome`, `ruvector-nervous-system::eventbus` |
| `crates/ruvector-embodiment/` | MuJoCo-NeuroMechFly Rust bridge via `cxx`, vision ray-cast, motor / sensor schemas | 1.5-2.5 kLOC + vendored mujoco headers | `ruvector-connectome`, `cxx` |
| `crates/ruvector-connectome-traces/` | Trajectory encoder + AgentDB / DiskANN integration for behavioral episodes | 0.5-0.8 kLOC | `ruvector-attention`, `ruvector-core` / AgentDB |
| `crates/ruvector-connectome-cuts/` | Counterfactual surgery harness (EdgeMask, witness attach, replay driver) | 0.4-0.6 kLOC | `ruvector-mincut::certificate`, `ruvector-lif`, `ruvector-embodiment` |

Every crate obeys the 500-line-per-file project convention by subdividing into modules, not by compressing. `ruvector-lif` will be the largest and needs the most sub-module discipline — see `./03-neural-dynamics.md` §4 for its layout.

## 3. Phased plan (M1 through M5)

Phases execute sequentially; work inside a phase parallelizes. Engineering-week estimates assume a tight team of 1-2 senior Rust engineers plus part-time neuroscience review. They are rough forward estimates for gating, not commitments.

### Phase 1 — Substrate lock (M1 target)

**Scope.** Data ingestion + graph schema in production form.

**Tasks.**
1. `ruvector-connectome` crate scaffold; define `Neuron`, `Synapse`, `NeuronId`, `NT`, `EdgeFlags`, `NeuronFlags`, interned `RegionId`/`CellTypeId`/`ClassId`. (0.5 wk)
2. `flywire-loader` streaming CSV → `GraphDB`, batched 10K-edge transactions; NT→sign mapping table; delay heuristic. (1.0 wk)
3. CSR materializer for `outgoing` and `incoming` adjacency views. (0.25 wk)
4. AgentDB embedder: 384-dim ONNX `all-MiniLM-L6-v2` over neuron metadata; DiskANN index build. (0.5 wk)
5. `hemibrain-loader` (minimal variant for cross-val). (0.25 wk)
6. Integration tests: round-trip ingest → query envelope (`./02-connectome-layer.md` §5.4) hits latency targets. (0.25 wk)
7. OpenWorm *C. elegans* loader stub as end-to-end CI sanity. (0.25 wk)

**Gate criteria (M1).**
- Full FlyWire v783 imported, `node_count` exact, `edge_count` within 1% of published, schema versioned.
- `"list all mushroom-body Kenyon cells and their downstream glutamatergic partners"` under 50 ms.
- All integration tests green; CI runs the OpenWorm sanity in < 60 s.

**Risk-out.** Schema inadequate (e.g., delay field insufficient, morphology hash mis-used) → return to schema design before M2. See `./00` §7 R2, R13.

**Effort estimate.** ~3.0 engineering-weeks.

### Phase 2 — Dynamics lock (M2 target)

**Scope.** `ruvector-lif` in isolation at 10K-neuron scale.

**Tasks.**
1. Crate scaffold, `params.rs`, `state.rs`, `engine.rs` skeleton. (0.5 wk)
2. Hierarchical timing-wheel `EventQueue` + deterministic tie-break + binary-heap fallback. (1.0 wk)
3. Subthreshold exponential-Euler integration; conductance channels; refractory period. (0.5 wk)
4. Sensory injection API + motor drain API. (0.25 wk)
5. Tap (`SpikeStream`, `VoltageStream`) on top of `ruvector-nervous-system::eventbus`. (0.5 wk)
6. EdgeMask subsystem + mask-audit counter. (0.25 wk)
7. Slow pool (neuromodulator diffusion) per region. (0.5 wk)
8. Determinism test: 10 s run, two build hosts, bit-exact spike trace match. (0.25 wk)
9. Benchmarks: 10K, 50K, 100K LIF ticks; target from `./03` §5. (0.5 wk)
10. Reproduce one qualitative behavior from the 2024 Nature paper (e.g., feeding-circuit response) on a 10K sub-network of FlyWire. (1.0 wk — calibration-heavy)

**Gate criteria (M2).**
- 10K-neuron LIF runs 1 s simulated time in 2-4 s wall-clock, single-thread.
- Determinism test passes.
- Qualitative reproduction of one published connectome-LIF behavior on the same substrate.

**Risk-out.** Event queue blows up → time-stepped fallback per region; reproduction fails → verify parameters against paper's Lin et al. repo (`./06-prior-art.md` §2). See `./00` §7 R1, R5.

**Effort estimate.** ~5.0 engineering-weeks.

### Phase 3 — Body lock (M3 target)

**Scope.** NeuroMechFly body closed-loop with the LIF kernel at 25 Hz.

**Tasks.**
1. `ruvector-embodiment` crate scaffold; `cxx` build.rs linking `libmujoco`; vendor headers. (1.0 wk)
2. MJCF loader; `World` wrapping `mjModel` + `mjData`. (0.5 wk)
3. Observation pipeline: joint positions, velocities, contact forces extraction. (0.5 wk)
4. Compound-eye ray-cast (256 rays/eye v1). (1.0 wk)
5. Proprioception encoder: chordotonal-neuron current from joint state. (0.25 wk)
6. Motor-neuron → torque decoder (rate-coded, 10 ms window). (0.5 wk)
7. Motor schema loader (TOML) + sensor schema loader. (0.25 wk)
8. Closed-loop runner: `examples/walking_on_flat.rs` with a 30 s stable run. (1.0 wk)
9. Latency profiling per `./04-embodiment.md` §8 budget; per-region LIF parallelism if needed. (1.0 wk)
10. Qualitative demo: descending-command stimulation yields grooming-like pattern. (1.0 wk)

**Gate criteria (M3).**
- Closed-loop runs 30 s at >=25 Hz control rate, no NaN, no contact-solver explosions.
- Descending-command stimulation reproduces a recognized grooming-like pattern.
- Latency budget on a reference laptop (specified in README): 20-40 ms per control tick.

**Risk-out.** MuJoCo native FFI latency crushes the loop → drop vision resolution, then drop ommatidium count, then fall back to MJX sidecar (`./04` §3.5). See `./00` §7 R3.

**Effort estimate.** ~7.0 engineering-weeks.

### Phase 4 — Analysis lock (M4 target)

**Scope.** Mount `ruvector-mincut`, `ruvector-sparsifier`, `ruvector-coherence`, `ruvector-attention`, DiskANN onto the live simulation. Counterfactual harness.

**Tasks.**
1. `ruvector-connectome-traces` crate: activity-graph builder (windowed, decayed), episode segmenter, SDPA encoder, DiskANN insert/search. (1.0 wk)
2. Wire `ruvector-sparsifier::AdaptiveGeoSpar` to the activity graph; maintain `H_t`. (0.5 wk)
3. Wire `ruvector-mincut::canonical::dynamic` to `H_t`; emit `BoundaryEvent`. (0.75 wk)
4. Wire `ruvector-coherence::spectral::SpectralTracker` + effective-resistance sampling; emit `CoherenceCollapseEvent`. (0.75 wk)
5. Motif enumerator on `jtree` clusters of size ≤7; `ruvector-graph::Hyperedge` writeback; DiskANN motif index. (1.0 wk)
6. `ruvector-connectome-cuts` crate: `EdgeMask` assembly, atomic push to `Connectome::apply_mask` and `Engine::apply_edge_mask`, replay driver. (0.75 wk)
7. Null-control harness: shuffled-spike + rewired-connectome nulls; p-value tables. (0.5 wk)
8. Paired experiment: coherence-collapse → behavioral-transition correlation across 10+ episodes. (1.0 wk)
9. Counterfactual case study: cut α/β lobe of mushroom body, measure feeding-motor deflection in replay. (1.0 wk)

**Gate criteria (M4).**
- Live boundary / coherence / motif streams populate AgentDB during a run without stalling the LIF taps.
- Coherence collapse precedes >=70% of behavioral state transitions with shuffled-null p < 0.01.
- At least one counterfactual-cut case study produces a behaviorally meaningful delta with a `ruvector-mincut::certificate` witness attached.
- Sparsifier drift (`SpectralAuditor`) stays within budget throughout the runs.

**Risk-out.** Coherence signal dominated by graph artifact rather than behavioral precursor (see `./00` §7 R6) → adjust window, try rewired-null as primary control. Counterfactual cuts over-perturb (R7) → constrain to mincut-boundary cuts only.

**Effort estimate.** ~7.25 engineering-weeks.

### Phase 5 — Publication-grade demo (M5 target)

**Scope.** One-command end-to-end demo, paper figures, preprint, crate releases.

**Tasks.**
1. `examples/connectome_walking_grooming/` or similar: downloads FlyWire, ingests, runs LIF + body for 60 s, streams live analysis, produces figures. (1.0 wk)
2. Replay bundle format (manifest JSON + spike/voltage/observation Parquet + masks). (0.5 wk)
3. Figures: motif library time-course, coherence trace with behavioral labels, cut-and-replay panel. (1.0 wk)
4. Preprint draft (bioRxiv) — methods paper, 12-20 pages. Uses `./07-positioning.md` §5 long-form positioning verbatim. (2.0 wk, with review cycles)
5. Crate polishing, `README.md` per crate, CITATION.cff referencing 2024 Nature paper. (0.5 wk)
6. Apache-2.0 license alignment; no FlyWire data redistribution; citation integrity pass. (0.25 wk)
7. CI: full pipeline reproducible on a clean machine with `cargo run --example` + a data download script. (0.5 wk)
8. External review: invite one fly-circuits PI and one ML-safety reviewer to beta-read before posting. (0.5 wk)

**Gate criteria (M5).**
- Demo reproducible from a single `cargo run` + data download on two independent machines.
- Figures pass quality check against `./07-positioning.md` §6 hype-avoidance rubric.
- Preprint accepted by bioRxiv after admin review (standard).
- External reviewer acceptance on framing.

**Risk-out.** Positioning drifts into hype (R8, R9) → rewrite using `./07-positioning.md` §6 rubric before posting. Scope creep to mammalian / new dataset (R10) → explicit "future work" section, do not implement.

**Effort estimate.** ~6.25 engineering-weeks.

## 4. Total effort estimate

**Critical path.** ~29 engineering-weeks of focused Rust + science work (3.0 + 5.0 + 7.0 + 7.25 + 6.25). With a 1.5-person team plus part-time neuroscience review, that is ~8-10 calendar months. With a 2-person team, ~6-8 calendar months. These are planning figures; actuals track in a project management tool outside this repo.

## 5. Dependency gating across phases

```
  Phase 1 (schema + ingest) ───────────────────────────────┐
       │                                                   │
       ▼                                                   │
  Phase 2 (LIF kernel, isolated)                           │
       │                                                   │
       ▼                                                   │
  Phase 3 (body + closed loop)                             │
       │                                                   │
       ▼                                                   │
  Phase 4 (analysis hooks) ◀───── sparsifier, mincut, ────┘
       │                           coherence (already done)
       ▼
  Phase 5 (demo + preprint)
```

Phase 4 depends on Phase 1 (graph), Phase 2 (spike taps), Phase 3 (behavior labels). It does *not* depend on Phase 3 in isolation: you can mount the analysis layer on a non-embodied LIF run first to debug, but the publishable story requires the closed loop.

## 6. Go / no-go gates (decision matrix)

Each gate has three outcomes: **go**, **delay** (iterate inside the phase), **pivot** (re-enter earlier phase with specific fix).

| Gate | Condition | Go | Delay | Pivot |
|---|---|---|---|---|
| M1 | Data fidelity | All `node/edge_count` exact; query envelope met | Schema gaps → iterate | Major schema miss → redesign before M2 |
| M2 | Dynamics | Qualitative behavior reproduces; determinism holds | Params mis-calibrated → iterate | Event queue fundamentally inadequate → revisit wheel design |
| M3 | Closed loop | 30 s stable at 25 Hz; grooming-like pattern | Latency miss by ≤2× → optimize | FFI fundamentally inadequate → MJX sidecar |
| M4 | Analysis | Coherence-collapse beats null p<0.01 on 10 episodes | Signal noisy → tune window/sparsifier | No signal even with tuning → reframe analysis-layer claims in writeups |
| M5 | Release | Demo reproducible; positioning rubric passes | Figures weak → iterate | External reviewer rejects framing → rewrite per `./07-positioning.md` |

## 7. What we do **not** do in v1

These belong to v2 or later and are explicitly out of scope so scope creep doesn't derail M5:

- Mammalian connectomes (Blue Brain / MICrONS) — `./06-prior-art.md` §10.
- Full Hodgkin-Huxley biophysics — `./03-neural-dynamics.md` §12.
- Multi-compartment dendritic models for all neurons — `./03` §7 keeps this optional and region-scoped.
- Synthetic training of behavior — forbidden by project rules.
- GPU acceleration for the LIF kernel — stays CPU-first until embodiment latency forces it.
- Distributed runtime across hosts — single-machine v1.
- Graded-potential neurons in the optic lobe — `./03` §12 Q5.
- Gap-junction dynamics at full fidelity — `./02` §9 Q2.

## 8. Testing strategy

- **Unit tests** per crate, colocated in `src/**/tests.rs` or `tests/`.
- **Integration tests** under `tests/`: schema round-trip, LIF determinism, closed-loop sanity, analysis-layer invariants.
- **Benchmarks** under `benches/` using `criterion` for event queue, sparsifier, mincut dynamic updates, DiskANN.
- **Null-control tests** for every analytical claim (`./05-analysis-layer.md` §3-6) before any publication draft quotes a p-value.
- **Doc tests** for every public API per Rust convention.
- **CI rule** (per `./07-positioning.md` §11) scanning `docs/` and `src/` for red-flag phrases; matches fail the lint.

## 9. Operational runbook (v1)

- Every run produces a `run-{utc}/manifest.json` with: `dataset_version`, `schema_hash`, `engine_config`, `body_config`, `seed_vector`, `crate_versions`.
- Spike/voltage/observation traces compressed via `ruvector-temporal-tensor`-style on-disk format (ADR-017 applies here) for cheap replay.
- Replay runs bit-exact under the same manifest.
- Telemetry goes to the existing pi-brain node (ADR-150) as a namespace `connectome/*` with quantized embeddings (4-bit) to keep footprint under control.

## 10. What this plan assumes about RuVector's readiness

- `ruvector-mincut` dynamic API is stable (confirmed in `crates/ruvector-mincut/src/lib.rs` and ADR history).
- `ruvector-sparsifier::AdaptiveGeoSpar` supports insert/delete and audit (confirmed in `crates/ruvector-sparsifier/src/lib.rs`).
- `ruvector-coherence::spectral` is buildable with the `spectral` feature (confirmed in `crates/ruvector-coherence/src/lib.rs`).
- `ruvector-attention::attention::ScaledDotProductAttention` is the canonical SDPA implementation used by `neural-trader-strategies` and similar crates.
- AgentDB + DiskANN Vamana is production-grade at pi-brain scale (ADR-144, ADR-146, ADR-149, ADR-150).

No unreleased primitive is on the critical path. Every assumption above is checkable by cat-ing the relevant `lib.rs`.

## 11. Open decisions for the user before Phase 1

These are gating calls the coordinator cannot make alone.

1. **Team shape.** 1 vs 2 Rust engineers on the kernel + bridge; neuroscience review cadence.
2. **MuJoCo license version.** Pin 3.1.x vs latest.
3. **Compound-eye resolution target.** 256 (faster) vs 512 (richer) ommatidia for v1.
4. **Hemibrain inclusion in v1.** Yes (cross-val) vs no (save for v2).
5. **Publication venue intent.** bioRxiv then *Nature Methods*, or bioRxiv only with a methods-paper detour to *eLife*.
6. **External collaborator.** Invite a fly-circuits PI as co-author from the start, or keep internal until preprint.

## 12. Closing

Phases 1-3 establish the runtime. Phase 4 is where RuVector's substrate story either wins or doesn't — mincut / sparsifier / coherence / DiskANN applied live is the differentiator from everything in `./06-prior-art.md`. Phase 5 is about staying honest: the positioning rubric in `./07-positioning.md` §6 is non-negotiable and the preprint does not ship without passing it. The scientific anchor throughout is the 2024 Nature whole-fly-brain paper, and the success test at every phase is whether we stay inside the regime that paper established.
