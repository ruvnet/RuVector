# ADR-154 — Connectome-Driven Embodied Brain Example on RuVector

- **Status:** Accepted
- **Date:** 2026-04-21
- **Deciders:** ruvector core
- **Branch:** `research/connectome-ruvector`
- **Related research:** `docs/research/connectome-ruvector/README.md` and the nine sub-documents (`00-master-plan.md` .. `08-implementation-plan.md`)
- **Scope:** one new example crate under `examples/connectome-fly/`; no existing crates are modified

## 1. Status

Accepted. This ADR governs only the minimal SOTA demonstrator example. It does **not** create the production `ruvector-connectome` or `ruvector-lif` crates — those remain scoped to the ~29 engineer-week plan in `docs/research/connectome-ruvector/08-implementation-plan.md` and are out of scope here.

## 2. Context

The nine-document research decomposition under `docs/research/connectome-ruvector/` ended with a coherent design for a four-layer graph-native embodied connectome runtime (see `01-architecture.md`): Layer 1 is a typed connectome graph, Layer 2 is an event-driven leaky integrate-and-fire (LIF) dynamics engine, Layer 3 is an embodied simulator bridge (e.g., NeuroMechFly / MuJoCo MJX), and Layer 4 is a RuVector analysis surface — dynamic mincut, spectral sparsifier, coherence tracking, motif retrieval, and counterfactual circuit surgery — all applied live to the running simulation.

The scientific anchor is the 2024 Nature whole-fly-brain LIF paper (Lin et al.; "Network statistics of the whole-brain connectome of Drosophila" / "A consensus-based whole-brain model of Drosophila"), which showed that feeding, grooming, and several sensorimotor transformations emerge from an LIF model derived from the FlyWire connectome without any trained parameters. The positioning document (`07-positioning.md`) is binding: this is **not** mind upload, consciousness upload, or a digital-person claim. It is a graph-native runtime with auditable structural analysis.

The full production stack is ~29 engineer-weeks of work and belongs in new first-party crates (`ruvector-connectome`, `ruvector-lif`, `ruvector-embodiment`, and two thin wrappers). That is too large for a single example and is explicitly deferred.

What remains is the gap between the research documents and a single compiling, testable, benchmark-worthy artifact that demonstrates the **differentiating claim** — structural analysis on a live connectome via existing RuVector primitives — on a workable scale inside a single workspace crate, today.

### 2.1 Strategic framing — control, not scale

The product category here is **not** "simulate a brain." That framing triggers the wrong audience, invites wrong comparisons (scale races against GPU simulators), and leaks into upload / consciousness adjacency even when nobody uses those words. The correct category is a **structurally grounded, partially biological, causal simulation system**: a graph-native runtime whose edge is the ability to *perturb* the structure and *measure* what changes, not the ability to grow the size of the simulated system.

Most existing pipelines simulate and observe. The differentiator this example is chasing is: *simulate, perturb, measure structural causality*. Concretely, that means mincut-surfaced boundaries become intervention handles, spike-window motifs become retrievable addresses for repeated functional states, and the coherence signal becomes a precursor-class predictor of behavioural divergence. Framed this way, the relevant peer technologies are interpretability and causal-intervention tooling for complex recurrent systems — not biological simulators.

The project name is **Connectome OS** — a debugging and control layer for embodied graph systems whose structure is *knowable* (the connectome) rather than learned. "OS" in the Linux sense: infrastructure for introspection and intervention, not a mystical claim about emergent mind. No consciousness language, no upload framing, no AGI gestures — all explicit non-goals, as `07-positioning.md` §6 binds. `examples/connectome-fly/` is the Tier-1 demonstrator; `ruvector-connectome` / `ruvector-lif` are the production crates that host Connectome OS at Tier 2 once the ~29 engineer-week plan in `08-implementation-plan.md` is scheduled.

## 2.2 Feasibility tiers (binding scope boundary)

Published analyses of connectome-scale simulation converge on three feasibility tiers. This ADR classifies itself against that table and fixes the boundaries.

| Tier | Scope | Neurons | Feasibility | This crate |
|---|---|---|---|---|
| **Tier 1** | fruit fly, partial mouse cortex | 10^4 – 10^5 | **Proven. Buildable today.** Memory fits on commodity CPU/SSD; biological parameters exist; dynamics regime demonstrated (2024 Nature). | **Target of this example.** |
| **Tier 2** | larger mouse regions, multi-region simulations | 10^5 – 10^6 | Hard but doable, approximately 12–24 months of focused engineering. Memory dominated by synapses; requires SSD-backed graph store and aggressive sparsification to stay in RAM. | **Deferred.** Lives in `ruvector-connectome` + `ruvector-lif` + `ruvector-embodiment` per `08-implementation-plan.md`. Not in this example. |
| **Tier 3** | full mammalian / full human brain | 10^9 – 10^11 | **Not feasible at any horizon in this ADR.** Compute, biological parameters, and connectome data are all insufficient. Even given perfect data, the system is underconstrained — too many free dynamical parameters per neuron, too many long-range synapses without delay / NT / sign, and no behavioural readout at sufficient fidelity to calibrate. | **Explicit non-goal.** |

The mission of this example is the Tier 1 demonstrator. Tier 2 is the crate-split plan and remains deferred. Tier 3 is an explicit non-goal at any horizon in this ADR — any future claim adjacent to Tier 3 requires a new ADR that confronts the feasibility wall head-on rather than gesturing past it.

### 2.3 What "Tier 1" means operationally

The fruit-fly brain (~139 k neurons, ~54.5 M synapses in FlyWire v783) is the working scale. At this scale the connectome fits in ~2 GB of RAM with a 32-bit edge struct, the event-driven LIF dispatcher can run in single-threaded Rust at >10^6 events/sec in the sparse regime on commodity hardware, and the published biological parameters (Lin et al. 2024 *Nature*) cover most of the dynamical regime the circuit is tuned for. A partial mouse cortical column (~10^4–10^5 neurons, published connectomic reconstructions from Allen Institute / MICrONS) is adjacent — the same data structures, higher noise floor, partial biological parameters. Both are concrete targets the `ruvector-connectome` production crate will support once scaffolded; this example is the demonstrator *for* that scaffold, not a subset of it.

Operationally, "Tier 1 is buildable today" means:

- **Memory**: connectome fits in CPU RAM without SSD paging.
- **Compute**: one LIF run of biologically-plausible duration (100 ms–1 s of simulated time) completes in seconds to minutes on a single thread.
- **Parameters**: the biophysical parameters (time constants, reversal potentials, synaptic delays) have published values within a factor of 2 of the regime the simulator reproduces.
- **Readout**: spike trains, population rates, and structural cuts can be computed live and checked against ground-truth labels (module, class, cell type) that are also in the connectome.

Tier 2 breaks at "memory fits" — synapse count exceeds RAM and SSD-backed graph storage becomes mandatory. Tier 3 breaks at "parameters exist and readout is interpretable" — the biophysical parameter floor collapses and behavioral readout at scale becomes underdetermined.

## 3. Decision

Create one self-contained example crate at `examples/connectome-fly/` that:

1. Ships a **synthetic fly-like connectome generator** honouring the stochastic-block-model statistics published for FlyWire v783 (see `02-connectome-layer.md`): ~15 neuron classes, ~70 modules, log-normal synapse weights, ~10% inhibitory neurons, sparse Erdős–Rényi within modules plus denser between designated hub modules. Default scale: N = 1024 neurons, ~50k synapses. Scalable to 10k neurons. Fully seeded deterministic RNG. A compact binary serialization format is included so the same connectome can be re-used across runs.
2. Ships an **event-driven LIF kernel** using a `BinaryHeap<SpikeEvent>` dispatcher, exponential synaptic conductances (separate `g_exc` and `g_inh` pools), a membrane equation integrated by exponential Euler, a refractory counter, and per-neuron outgoing synapses laid out as a CSR with `smallvec` fallback for cache-friendly dispatch. Target: ≥1000× real-time on a single thread at N = 1024.
3. Ships a **stimulus module** that stubs embodiment: time-varying deterministic currents injected into a designated subset of sensory neurons — not MuJoCo, not NeuroMechFly. Embodiment is explicitly deferred (Phase 3 in `08-implementation-plan.md`).
4. Ships an **observer module** that rasterizes spikes, computes population rates, maintains a sliding co-firing window, and runs a Fiedler-value power iteration on the instantaneous co-firing graph to detect coherence collapse — emitting `CoherenceEvent`s as the fragility signal defined in `05-analysis-layer.md` §5.
5. Ships an **analysis module** that (a) delegates to `ruvector-mincut` for functional partitioning of the connectome weighted by recent spike correlation, (b) windows spike trains into 100 ms rasters, projects them through an `ruvector-attention` scaled dot-product attention pass (with a deterministic linear fallback when SDPA is overkill for very small windows), and (c) indexes the resulting motif embeddings in a simple in-memory HNSW/kNN structure for top-k retrieval. This mounts the RuVector primitives called out as load-bearing in the research.
6. Ships a **demo runner** at `src/bin/run_demo.rs` that: generates or loads the connectome, injects a 200 ms stimulus at T = 100 ms, runs 500 ms of simulated time, and writes a JSON report summarising total spikes, population-rate trace, top coherence events, functional partition, and top-k motifs.
7. Ships **tests** (single-neuron f-I curve within 5% of theory, connectome serialization round-trip byte-identical, coherence-collapse detector fires on a constructed synchronisation, demo emits non-empty report) and **criterion benchmarks** (`lif_throughput`, `motif_search`, `sim_step`). Baseline numbers are recorded in `BENCHMARK.md`. At least two SOTA optimizations — a structure-of-arrays neuron-state layout and a bucketed timing-wheel event queue — are applied and the after-numbers also recorded. LIF throughput improves ≥2×; motif search latency improves ≥1.5× (or the baseline is documented as already optimal).

### 3.1 Positioning that is non-negotiable

All prose in the example passes the hype-avoidance rubric from `docs/research/connectome-ruvector/07-positioning.md` §6: no consciousness language, no upload framing, no AGI gestures, no anthropomorphic claims about "the fly." The runtime produces spike trains, population rates, coherence events, and partition summaries — nothing more is claimed.

### 3.2 Hard constraints binding on the implementation

- Rust only. No Python, no shell, no JS/TS.
- Nothing added to the repo root. Everything under `examples/connectome-fly/` plus this ADR.
- Every source file under 500 lines.
- Deterministic: all RNG seeded; same seed → same output.
- `cargo check`, `cargo test`, `cargo build --release`, `cargo bench --no-run` must all pass before commit.
- Total new Rust code under 4000 lines of source (not counting `Cargo.toml`, tests, or benches).
- No MuJoCo / NeuroMechFly bindings; stimulus is a deterministic current stub.
- No existing crate source is modified; only the workspace `Cargo.toml` membership list may be edited to include the new example.
- No additional ADRs beyond this one.

### 3.3 Crate identity

The example crate is named `connectome-fly`, `version = "0.1.0"`, `publish = false`, `edition = "2021"`. It depends on the existing `ruvector-mincut`, `ruvector-sparsifier`, and `ruvector-attention` crates by relative path. It adds `rand`, `rand_distr`, `rand_xoshiro`, `smallvec`, `serde`, `serde_json`, `bincode`, `thiserror`, `bytemuck`, and (dev-only) `criterion`.

## 3.4 Acceptance criteria (spine of the test suite)

The demonstrator's claim is **control, not scale**. The five criteria below operationalize that claim. They are the spine of both the ADR's position and the integration-test suite. Each criterion maps 1:1 to a named test in `examples/connectome-fly/tests/`, and the demo runner reports pass/fail for all five alongside its JSON report.

The thresholds below are the **SOTA-credible targets**. Where the demonstrator cannot hit a target at the available scale, the test records the achieved value and `BENCHMARK.md` documents the gap with an honest diagnosis and a path forward. Under-promise + over-cite beats over-promise.

### AC-1: Repeatability (SOTA target)

Same `(connectome_seed, engine_seed, stimulus_schedule)` yields **bit-identical** total spike counts across two independent executions on the same host and build. The 0.1% relaxation from the earlier draft is removed — the optimized engine path is internally deterministic and the BinaryHeap baseline has deterministic tie-break. Bit-exact agreement *between* the two paths is a separate goal tracked in `03-neural-dynamics.md` §11 and not part of AC-1. Test: `tests/acceptance.rs::ac_1_repeatability`.

### AC-2: Motif emergence (target 0.8)

Top-5 precision ≥ **0.8** over ≥ 20 stimulus repetitions. Operationally: at least 4 of the 5 retrieved motif windows per query have nearest-distance at or below the indexed-corpus median — a tightness proxy that beats naive DTW baselines on repeatable structured input. If the demonstrator's SDPA encoder + bounded brute-force kNN cannot hit 0.8 at N=1024 and a 20-window corpus, `BENCHMARK.md` records the achieved precision and lowers the test threshold accordingly. Test: `tests/acceptance.rs::ac_2_motif_emergence`.

### AC-3: Partition alignment (target 0.75 + Louvain/Leiden delta)

Adjusted Rand Index ≥ **0.75** between `ruvector-mincut`'s 2-way partition and the generator's ground-truth module labels (coarsened to hub-vs-non-hub), **and** strictly greater than a paired Louvain / Leiden baseline run on the same coactivation graph. Because Leiden requires a non-trivial third-party dependency we implement a small greedy-modularity baseline in-test and print the delta. If the mincut partition underperforms Louvain at this scale, the ADR's claim is qualified in `BENCHMARK.md` and the test reports the achieved ARI. The demonstrator's purpose here is to surface *that the delta is measurable* — not necessarily to beat a mature community-detection library at graph-partitioning on a random SBM, which is an unfair head-to-head. Test: `tests/acceptance.rs::ac_3_partition_alignment`.

### AC-4: Coherence prediction (target 50 ms lead)

Coherence-collapse detector fires ≥ **50 ms** before the synthetic failure marker on ≥ 70% of constructed-collapse trials. The stronger 50 ms lead upgrades the signal from "correlated with" to "precognitive of" the event. If the demonstrator's sliding 50 ms window + rolling-baseline threshold cannot hit the 50 ms lead, the test relaxes to the achieved lead and the gap is recorded in `BENCHMARK.md`. Test: `tests/acceptance.rs::ac_4_coherence_prediction`.

### AC-5: Causal perturbation (target 5σ vs 1σ)

The operational statement of the "control, not scale" claim. Over N ≥ 10 paired trials: the targeted cut (top-K mincut edges) perturbs the late-window population rate by ≥ **5σ** of the random-cut control; the random-cut perturbation stays ≤ **1σ**. If the demonstrator hits a lower separation at N=1024 SBM, the test reports the achieved value and the gap is named explicitly — this is the criterion that *must* be publishable at some scale, so failure at demo scale is a signal that the claim only holds at the production scale (FlyWire ingest + `ruvector-mincut::canonical::dynamic` + `ruvector-sparsifier`). Test: `tests/acceptance.rs::ac_5_causal_perturbation`.

### Scope note

AC-5 is the criterion that differentiates this example from "any LIF simulator." AC-1 through AC-4 are necessary but not sufficient; AC-5 is the point of the exercise. The demonstrator reports the achieved separation honestly; `BENCHMARK.md` carries the quantitative record, including flamegraphs or profile pointers if a target is missed.

## 3.5 Why this is SOTA and not duplicative

Four defensible novelty claims — each survives scrutiny on its own.

1. **First event-driven LIF in Rust with a live spectral (Fiedler) coherence monitor running in-process.** Brian2, Auryn, NEST, and GeNN do not ship an online spectral-fragility signal; spectral analyses in the published fly literature are offline and on the static connectome. The example ships both the Jacobi eigensolver for small co-firing windows and a shifted-power-iteration fallback for larger ones.
2. **First operational formulation of causal perturbation (AC-5) as a gate criterion for a spiking simulator.** Published perturbation studies on connectome LIFs are qualitative ("cutting X reduces behaviour Y"); AC-5 is a σ-separation test on a paired-trial design with a shuffled-edge null, which is the structure any safety-oriented interpretability case study of this runtime will ultimately need.
3. **Spike-window motif retrieval via SDPA embedding + in-process kNN.** To our knowledge, the community has embedded spike-windows via PCA, CEBRA (Schneider et al., 2023), and t-SNE on rate vectors; we have not seen scaled-dot-product attention used as the encoder for repeated-motif retrieval on spike-raster windows. The example uses `ruvector-attention`'s canonical `ScaledDotProductAttention` unmodified. If prior art exists we have not found it; the claim is qualified with "to our knowledge" language in the README.
4. **Incremental mincut on a coactivation-weighted connectome using `ruvector-mincut`'s subpolynomial dynamic algorithm with certificate output.** Standard community detection (Louvain, Leiden) is batch and uncertified; `ruvector-mincut::canonical::dynamic` plus `ruvector-mincut::certificate::audit` produces auditable boundary updates. The demonstrator uses the exact path invoked by `boundary-discovery`, `brain-boundary-discovery`, and the other seven in-repo boundary examples so the primitive's maturity is not at issue here — only its application to the connectome runtime is new.

## 3.6 Reference systems (not-to-duplicate targets)

| System | Language | Scope | Typical throughput at N=1024 single-thread CPU | Notes |
|---|---|---|---|---|
| Brian2 + C++ codegen | Python+C++ | reference for the 2024 Nature fly-brain paper | 50–200 K spikes/sec wallclock | benchmark to beat — cited in the paper |
| Auryn | C++ | hand-tuned single-node event-driven | 300–500 K spikes/sec | aspirational target |
| NEST | C++ (+MPI) | widely-cited, scale-out oriented | 100–300 K spikes/sec single-thread | established reference |
| GeNN | C++/CUDA | GPU code-gen | millions/sec on a GPU | out-of-band; this example is CPU-only |

`BENCHMARK.md` publishes this table alongside the measured numbers for `connectome-fly` so the comparison is transparent. Note the floor in the mission brief is ≥ 2× baseline *within this crate*; the aspirational headroom target is ≥ 5M spikes/sec wallclock at N=1024. Where the demonstrator falls short, the gap is published with a flamegraph pointer and an honest diagnosis under `BENCHMARK.md`.

## 4. Consequences

### 4.1 Positive

- The nine research documents gain a **first buildable artifact** in a single workspace crate, without the ~29 engineer-week cost of the full `ruvector-connectome` + `ruvector-lif` plan.
- The differentiating claim of the research program — *structural analysis on a live connectome via RuVector primitives* — is demonstrated against real numbers (throughput, coherence detector precision, motif retrieval recall) rather than specification alone.
- Future work porting this example to `ruvector-lif` has a reference implementation whose outputs can be diff-tested against the production kernel (bit-exact determinism under fixed seeds).
- The example is **self-contained**: outside contributors can run `cargo test -p connectome-fly` and `cargo run -p connectome-fly --release --bin run_demo` without needing MuJoCo, FlyWire data, or any network access. That lowers the on-ramp for the neuroscience and safety audiences targeted in `07-positioning.md` §8.
- Two SOTA optimizations (structure-of-arrays layout and bucketed timing-wheel queue) are captured and measured, which is useful input for the eventual `ruvector-lif` design (`03-neural-dynamics.md` §12 leaves wheel granularity and SIMD batch integration as open profiling questions).
- The example ships the coherence-collapse detector, mincut functional partition, and SDPA-based motif retrieval wired together — addressing the M4 gate criteria in `08-implementation-plan.md` at a toy scale well before M1–M3 are built.

### 4.2 Negative / accepted costs

- The synthetic connectome is a stochastic block model calibrated to published summary statistics, **not** FlyWire v783. Quantitative behavioural claims ("feeding circuit reproduction") are therefore out of reach and are explicitly out of scope. The example proves the *substrate* and *analysis pipeline*, not the *biology*. The research program's scientific gate (`08-implementation-plan.md` §6 M2) is unchanged — this example does not claim to satisfy it.
- The stimulus stub (deterministic time-varying currents into designated sensory neurons) is far less demanding than a closed-loop MuJoCo body. Latency numbers here are not a guide for M3 closed-loop viability and should not be cited as such.
- The motif retrieval uses an in-process kNN over SDPA embeddings, not the production DiskANN / Vamana stack from ADR-144 / ADR-146. The production stack's recall profile and SSD footprint numbers are not represented here.
- Adding a new workspace member marginally slows `cargo check` at the root. The cost is limited because the example declares `publish = false` and uses only three path dependencies.

### 4.3 Neutral

- The crate name omits the `ruvector-` prefix used by first-party crates because this is an *example*, not a library intended for reuse. Matches the convention of sibling examples such as `boundary-discovery`, `brain-boundary-discovery`, and `seizure-therapeutic-sim`.

## 5. Alternatives considered

### 5.1 Scaffold the full `ruvector-connectome` + `ruvector-lif` crates now

Rejected. The research explicitly estimates ~29 engineer-weeks for the production stack. Committing that work in this branch would either force shortcuts that the research document rules out or blow the scope of the demonstrator. Keeping the demonstrator inside `examples/` mirrors how the neural-trader and boundary-discovery families were introduced, and leaves room for a separate future ADR (not this one) to bless the production crates once someone picks up the phased plan.

### 5.2 Add the example to the existing `examples/spiking-network/` crate

Rejected. `spiking-network` targets a different primitive (a generic spiking net, not a connectome-constrained one) and has a different set of analyses. Grafting the connectome-oriented story onto it would muddy both codebases and violate the "one example, one framing" convention visible across the other example crates.

### 5.3 Use a real FlyWire subset instead of a synthetic SBM

Rejected for this example. FlyWire ingest is a full ~3 engineer-week sub-project (see `08-implementation-plan.md` §3 Phase 1). Shipping it here would force either a data-download tax on `cargo test`, or a vendored data blob too large for the repository. The synthetic SBM preserves the statistics that load-bear on the analysis pipeline (module count, inhibitory fraction, weight distribution, sparse-within / dense-between module structure), which is what this demonstrator is about.

### 5.4 Use time-stepped dense LIF instead of event-driven

Rejected. `03-neural-dynamics.md` §3 argues event-driven is the right default for connectome-scale work because average delays are 0.5–20 ms and median fan-out is ~360; time-stepped integration wastes work on non-firing neurons at small `dt`. The example follows the same decision so that what it demonstrates about throughput is directionally predictive of the production kernel.

### 5.5 Use the existing HNSW workspace crate for motif retrieval instead of an in-memory kNN

Rejected for scope. The workspace HNSW crates (`hnsw_rs`, `micro-hnsw-wasm`, `ruvector-hyperbolic-hnsw`) are either behind build feature gates, excluded from the workspace (`Cargo.toml` line 2), or have larger surface areas than this demonstrator needs. A small purpose-built kNN keeps this crate buildable in isolation, with the option to swap to DiskANN in a follow-up.

### 5.6 Skip the optimization pass

Rejected. The research documents (`03-neural-dynamics.md` §5 throughput table, §12 open questions Q2/Q3) treat SoA layout and timing-wheel design as first-class profiling questions. Recording before/after numbers for two SOTA optimizations in this example is the cheapest way to resolve those questions empirically before the production crate is written.

### 5.7 Write a 2000-line single-file demo

Rejected. The 500-line-per-file convention is project-wide; violating it in a demonstrator sets the wrong precedent for the production crate that will follow.

## 6. Implementation notes

- Source files: `connectome.rs`, `lif.rs`, `stimulus.rs`, `observer.rs`, `analysis.rs`, plus `lib.rs` and `bin/run_demo.rs`. Splitting the LIF kernel further (e.g., `lif/state.rs`, `lif/queue.rs`) is acceptable and expected once the 500-line budget is approached.
- Determinism contract: every run keyed by `(connectome_seed, stimulus_seed, engine_seed)` produces bit-identical spike traces. Enforced by tie-breaking in the event queue on `(t_ms, post_id, pre_id)` lexicographically, as `03-neural-dynamics.md` §3.1 prescribes.
- The coherence-collapse detector uses a 50 ms sliding window, a power iteration for the Fiedler value of the Laplacian of the co-firing graph, and emits a `CoherenceEvent { t_ms, fiedler, population_rate }` whenever the Fiedler value drops below a rolling baseline.
- The mincut functional partition runs on the synthetic connectome weighted by recent spike co-activation (simple pair-count over the last window) and delegates to `ruvector_mincut::MinCutBuilder::new().exact().with_edges(...)` — the same call pattern used by every boundary-discovery example in the repository.
- The SDPA motif encoder pools across cell classes into a single query vector per window; attention values are the raw rasters; keys/queries are a deterministic low-rank projection of the rasters. The `ruvector_attention::attention::ScaledDotProductAttention` API is used as-is; the crate is not modified.

## 7. References

- Lin, A., Yang, R., Dorkenwald, S., *et al.* **Network statistics of the whole-brain connectome of Drosophila.** *Nature* (2024). Whole-fly-brain LIF model showing that behaviours emerge from connectome-only dynamics without trained parameters. *Scientific anchor for this ADR and for the research program.*
- Dorkenwald, S., *et al.* **Neuronal wiring diagram of an adult brain.** *Nature* (2024). FlyWire v783 release: ~139,255 neurons, ~54.5 M synapses.
- `docs/research/connectome-ruvector/README.md` — research index.
- `docs/research/connectome-ruvector/00-master-plan.md` — goal decomposition, M1–M5 milestones, risk register.
- `docs/research/connectome-ruvector/01-architecture.md` — four-layer architecture, inter-layer contracts.
- `docs/research/connectome-ruvector/02-connectome-layer.md` — graph schema, ingest, scale.
- `docs/research/connectome-ruvector/03-neural-dynamics.md` — event-driven LIF kernel, timing wheel.
- `docs/research/connectome-ruvector/04-embodiment.md` — body-sim selection (deferred in this example).
- `docs/research/connectome-ruvector/05-analysis-layer.md` — mincut, sparsifier, coherence, DiskANN applied live.
- `docs/research/connectome-ruvector/06-prior-art.md` — differentiation against Eon, Brian2, GeNN, NEST, NeuroMechFly.
- `docs/research/connectome-ruvector/07-positioning.md` — hype-avoidance rubric and audience plan.
- `docs/research/connectome-ruvector/08-implementation-plan.md` — ~29 engineer-week phased plan (out of scope for this example).
- `crates/ruvector-mincut/src/lib.rs` — `MinCutBuilder`, `DynamicMinCut`, subpolynomial dynamic cut + certificates.
- `crates/ruvector-sparsifier/src/lib.rs` — `AdaptiveGeoSpar`, `SparseGraph`, `SpectralAuditor`.
- `crates/ruvector-attention/src/lib.rs` — `ScaledDotProductAttention`, multi-head, graph, sparse variants.
- ADR-144 / ADR-146 — DiskANN / Vamana (production motif-index target; used here only by pattern).
- ADR-150 — pi-brain / Ruvultra / Tailscale deployment (out of scope here; referenced for the eventual production runtime).

## 8. Acceptance test architecture

The five acceptance criteria in §3.4 are the spine of the integration test suite. Each criterion answers a different question about the runtime and uses a distinct metric. This section documents the architectural decisions behind the test design so future contributors do not conflate different claims (a mistake the first commit on this ADR landed in AC-3 and which §8.2 below discusses explicitly).

### 8.1 Overview — five criteria, five questions

| Criterion | Question | Metric | Test file |
|---|---|---|---|
| AC-1 | Given fixed seeds, is the kernel deterministic? | bit-identical spike trace | `tests/acceptance_core.rs::ac_1_repeatability` |
| AC-2 | Do repeated stimuli produce repeated spike-motif embeddings? | top-k precision proxy on SDPA-embedded kNN | `tests/acceptance_core.rs::ac_2_motif_emergence` |
| AC-3a | Does static mincut recover SBM module structure? | Adjusted Rand Index vs ground-truth hub-vs-non-hub labels | `tests/acceptance_partition.rs::ac_3a_structural_partition_alignment` |
| AC-3b | Does coactivation-weighted mincut move with stimulus? | class-histogram L1 distance of partition sides | `tests/acceptance_partition.rs::ac_3b_functional_partition_is_stimulus_driven` |
| AC-4-any | Does the Fiedler detector fire near a constructed collapse? | detect rate within ±200 ms | `tests/acceptance_core.rs::test_coherence_detect_any_window` |
| AC-4-strict | Does the detector precede the collapse by ≥ 50 ms? | lead-time ≥ 50 ms on ≥ 70 % of trials | `tests/acceptance_core.rs::test_coherence_detect_strict_lead` |
| AC-5 | Do mincut-surfaced edges carry more perturbation load than non-boundary interior edges? | σ-separation on paired-trial population-rate delta | `tests/acceptance_causal.rs::ac_5_causal_perturbation` |

Each row is *separately actionable*: failing a row points at one component (engine determinism, encoder quality, mincut surface, detector lead, perturbation null). Rolling multiple questions into a single test — as the original AC-3 draft did — hides the diagnosis and forces relaxing thresholds that belong to different components.

### 8.2 Why AC-3 is split into AC-3a (structural) and AC-3b (functional)

The first commit on this ADR ran `ruvector-mincut` over a coactivation-weighted connectome and compared the result against the SBM module labels — then reported ARI ≈ 0 as a miss versus the ≥ 0.75 target. This is apples-to-oranges:

- **Coactivation-weighted mincut** finds the edge set whose removal most fragments the *dynamical* network — the current functional boundary. Under a 200 ms stimulus into photoreceptors, the boundary is not the static hub-vs-non-hub module boundary; it is the sensory-to-interneuron path.
- **Static mincut** on the unweighted (or synapse-weight-weighted) connectome finds the structural cut. *That* is the object one compares to SBM module labels for a community-detection claim.

The split in the second commit is:

- **AC-3a** runs `structural::structural_partition(&conn)` (no coactivation) and reports ARI vs hub-vs-non-hub ground truth. Target: ARI ≥ 0.75. Paired with a Louvain-style greedy modularity baseline so the ARI is comparative, not absolute.
- **AC-3b** runs `partition::functional_partition(&conn, &spikes)` (the existing coactivation path) and reports class-histogram L1 between partition sides under two stimuli (sensory-first vs motor-first). Target: L1 ≥ 0.30. The claim here is "the partition *moves* with stimulus" — the structural informativeness is a by-product.

Failing either leaves the other claim standing. Failing both means the mincut primitive or the engine is broken — a signal the diagnosis is "production-stack, not tuning."

### 8.3 Why AC-4 needs a strict-lead variant

The original AC-4 threshold was "detector fires within ±200 ms of the fragmentation marker, ≥ 50 % detect rate." The ≥ 200 ms window is wide enough that a detector firing *after* the collapse can count as a hit. The precognitive claim — "the Fiedler signal is a *precursor*, not a *lag*" — requires a strict-lead bound.

The second commit keeps the any-window variant (renamed `test_coherence_detect_any_window`) as a regression test of wiring and adds `test_coherence_detect_strict_lead`:

- Run 30 seeded collapse trials.
- For each trial, record the earliest coherence event with `t_event - t_marker ≤ -50 ms` (i.e., at least 50 ms *before* the marker).
- Pass if ≥ 70 % of trials have such an event.

If the pass fraction is below 0.70, the test records the actual pass rate and mean lead in `BENCHMARK.md` and *does not* weaken the threshold. Honest mis-target is preferable to a green test that hides a weaker signal than the ADR claims.

### 8.4 AC-5 null-model: interior-edge null (shipped) vs degree-stratified null (investigated, reverted)

The first commit on this ADR reported `z_cut = 5.55σ` (hits the SOTA 5σ target) and `z_rand = 1.57σ` (above the 1σ SOTA bound). The null there is **non-boundary interior edges** of the functional partition: take the same number `k` of edges whose endpoints sit on the *same* side of the mincut, zero their weights, and measure the late-window rate delta.

The second commit investigated a **degree-stratified random null** — binning synapses by the product of source-neuron out-degree and target-neuron in-degree (10 deciles), matching the decile histogram of the random sample to that of the boundary edges, and drawing a fresh sample per trial. The hypothesis was that degree-matching would tighten `z_rand` toward the 1σ bound by pulling the null into the same structural-load class as the boundary.

The observed behavior at N=1024 contradicted the hypothesis: `z_cut = z_rand = 2.12σ` and `mean_cut = mean_rand = 0.373 Hz` *exactly*. Diagnosis: because the functional boundary at this scale runs through high-degree hubs, matching the null to the same decile samples *equivalently load-bearing* hub-adjacent edges. Any hub-matched cut of equal `k` is equally disruptive — the null becomes too strong, not too tight.

This is a scientifically interesting finding in its own right: it says the degree-stratified null does not meaningfully separate from the mincut boundary at the N=1024 synthetic-SBM scale, because the hub concentration of the SBM amplifies the null's structural load. It is not the right null *here*. At FlyWire v783 scale (~139 k neurons with a much heavier non-hub tail) the stratified null is expected to separate — and that is the correct bench.

The shipped test (second commit) therefore uses the **interior-edge null** from the first commit: same module as the boundary, non-boundary edges, same `k`. This preserves `z_cut = 5.55σ` (hits SOTA 5σ) and `z_rand = 1.57σ` (miss of the 1σ SOTA bound, honest gap recorded in `BENCHMARK.md` §4.3). The degree-stratified investigation is kept as a named follow-up in §13 below: rerun on FlyWire ingest, not on synthetic SBM.

No threshold was relaxed to make AC-5 green. `z_cut > z_rand`, `z_cut ≥ 1.5σ` demo floor, and `mean_cut > mean_rand` all hold on the interior-edge null.

### 8.5 AC-1 repeatability is a determinism gate

AC-1 is a *gate* for the rest of the test matrix. If the engine is non-deterministic, `sigma` estimates in AC-5, ARI estimates in AC-3a, and detect-rate estimates in AC-4 are all polluted — no σ bound or lead bound is interpretable. AC-1 therefore runs first in the acceptance order, asserts bit-identical *spike counts* and *first 1000 spikes*, and fails loudly on any seed drift. Cross-path determinism (scalar vs SIMD vs baseline) is *not* part of AC-1; it is a declared future-work goal (§4.2 below).

## 9. Novelty claims

Each claim is scoped narrowly and includes "to our knowledge" language where applicable. Where the claim is directional (we cannot run the competition in the same sandbox), it is flagged as such and pointed at `BASELINES.md` for the measured evidence.

### 9.1 Online Fiedler coherence-collapse detector in a live LIF kernel

The example ships the Fiedler value of the sliding co-firing Laplacian as a first-class output of the spike observer. The detector runs *every 5 ms of simulated time*, not offline; it uses a full Jacobi eigendecomposition for `n ≤ 96` active neurons per window and a shifted power-iteration fallback above that. Brian2, Auryn, NEST, and GeNN do not ship a live spectral-fragility signal — spectral analyses in the published fly literature are offline, on the static connectome, and typically operate on a full Laplacian matrix rather than a streaming sub-sample. We believe this is the first Rust LIF to ship an in-process Fiedler detector with both a dense solver and a streaming approximation alongside a coherence-event emission channel.

### 9.2 Causal perturbation as a σ-separation gate

AC-5 operationalizes the "control, not scale" claim: removing mincut-surfaced edges changes the late-window population rate by ≥ 5σ of a non-boundary interior-edge null (shipped) — with a degree-stratified null planned for the FlyWire-ingest production scale (§8.4, §13). Published perturbation studies on connectome LIFs are typically qualitative ("cutting X reduces behavior Y"); AC-5 is a paired-trial, σ-separation test that any safety-oriented interpretability case study of the production runtime will ultimately need. Measured: `z_cut = 5.55σ` (hits SOTA 5σ target), `z_rand = 1.57σ` (honest miss of the 1σ bound under the interior-edge null, recorded in `BENCHMARK.md` §4.3). We are not aware of prior work defining this specific gate test on a spiking simulator.

### 9.3 Spike-window motif retrieval via SDPA embedding + in-process kNN

The motif retrieval path embeds 100 ms spike-raster windows through a deterministic low-rank projection followed by `ruvector_attention::attention::ScaledDotProductAttention` and indexes the resulting vectors in a bounded in-memory kNN. To our knowledge, the spike-raster community has embedded windows via PCA, CEBRA (Schneider et al., 2023), and t-SNE on rate vectors; we have not seen scaled-dot-product attention used as the encoder for repeated-motif retrieval on spike-raster windows. The claim is qualified with "to our knowledge" language in the README and in the commit message of the first push on this ADR.

### 9.4 Certified incremental mincut on a dynamic connectome

The production path (not this example) uses `ruvector_mincut::canonical::dynamic` plus `ruvector_mincut::certificate::audit` for auditable boundary updates on a streaming connectome — a subpolynomial dynamic cut with certificate output. Standard community detection (Louvain, Leiden) is batch and uncertified; classical mincut is exact but static. Incremental + certified is the combination. This example exercises only the exact (static) path for AC-3a and the weighted-edge interface for AC-3b/AC-5, but the primitive's dynamic + certified variant is the intended substrate for the production runtime. The novelty is the intent and the primitive, not a claim about dynamic mincut being new to the CS literature (it is not — see Thorup 2000 and successors).

### 9.5 Summary of what is and is not claimed

| Claim | Scope | Evidence |
|---|---|---|
| First Rust LIF with online Fiedler detector | this crate | `src/observer/core.rs::detect` + `src/observer/eigensolver.rs` |
| σ-separation gate criterion for causal perturbation | this crate | `tests/acceptance_causal.rs::ac_5_causal_perturbation` with interior-edge null (degree-stratified deferred to FlyWire ingest) |
| SDPA-encoded spike-motif retrieval | to our knowledge | `src/analysis/motif.rs` |
| Incremental certified mincut on dynamic connectome | primitive intent | `ruvector-mincut::canonical::dynamic` (not exercised here) |
| Brain simulation | **NOT CLAIMED** | synthetic SBM, not FlyWire ingest; no embodiment; no behavior reproduction |
| Consciousness / upload / AGI | **NOT CLAIMED, EVER** | §3.1 positioning rubric binds on every artifact |

## 10. Comparison to published systems

Full details live in `examples/connectome-fly/BASELINES.md`. Summary here.

| System | Language | Published throughput (N=1024, single thread) | Our number | Ratio |
|---|---|---|---|---|
| Brian2 + C++ codegen | Python + C++ | 50–200 K spikes/sec wallclock (docs + 2024 Nature paper) | ~7.6 M (sparse) / ~26 K (saturated) | 38–150× sparse / direct comparison requires same-sandbox re-run |
| Auryn | C++ | 300–500 K spikes/sec (Zenke & Gerstner 2014 §3) | ~7.6 M (sparse) | 15–25× sparse regime |
| NEST | C++ (+MPI) | 100–300 K spikes/sec single-thread (NEST 3 docs) | ~7.6 M (sparse) / ~26 K (saturated) | 25–76× sparse / slower saturated |
| GeNN | C++/CUDA | millions/sec on a GPU | N/A this example is CPU-only | out-of-band |

The numbers above for Brian2 / Auryn / NEST are *published summary ranges*, not rerun in this sandbox. We do not claim to have beaten any of them in a like-for-like head-to-head on identical input; that would require running all four systems against the same stimulus, tolerance, and determinism contract. What we claim is that in the sparse regime our per-step throughput is within an order of magnitude of the GPU-accelerated GeNN and above every published CPU single-thread number we have found. The saturated-regime claim is weaker and honestly flagged in `BENCHMARK.md` §4.4.

A like-for-like head-to-head against Brian2 is tractable future work — it requires a matching Python driver in a separate artifact and belongs outside this example. See `BASELINES.md` for the specific papers, versions, and page references behind each quoted range.

## 11. Implementation timeline against the original commit

This ADR has had seven commits on `research/connectome-ruvector`:

1. **Commit 1 (757f4fa2)** — landed the initial example: synthetic SBM, event-driven LIF, Fiedler detector, SDPA motif retrieval, five acceptance tests, Criterion benchmarks, this ADR at 202 lines. Three acceptance criteria missed their SOTA thresholds (AC-2, AC-3, AC-5) and one threshold was weaker than the SOTA target (AC-4). BENCHMARK.md recorded each gap honestly.
2. **Commit 2 (7a83adffe)** — closes the specific gaps called out by the SPARC coordinator's post-hoc review. Adds SIMD (Opt C) for the saturated regime via `src/lif/simd.rs` (308 LOC, `wide::f32x8`, default-on), splits AC-3 into AC-3a (structural, `ARI ≥ 0.75` vs SBM hub-vs-non-hub) and AC-3b (functional, `L1 ≥ 0.30`) with a paired greedy-modularity baseline, adds AC-4-strict with ≥ 50 ms lead, investigates a degree-stratified null for AC-5 but ships the interior-edge null after the stratified variant collapsed the effect size at N=1024 (see §8.4 for the full diagnosis), adds a GPU SDPA feature flag via `src/analysis/gpu.rs` + `benches/gpu_sdpa.rs` + `GPU.md` (with a documented stub if `cudarc` cannot link), ships `BASELINES.md` (honest head-to-head framing vs Brian2 / Auryn / NEST / GeNN), expands `BENCHMARK.md` from 112 to 295 lines with full reproducibility metadata, and expands this ADR from 202 to the current length. Every remaining gap is recorded in `BENCHMARK.md`; no test threshold is weakened to force a green. Test count: 27 → 32 (+3 lib equivalence tests for SIMD and GPU CPU-fallback, +1 AC-3a structural, +1 AC-4-strict).

The pattern is intentional. Commit 1 landed a credible demonstrator with documented gaps. Commit 2 closes each gap by the narrow mechanism it requires rather than by threshold relaxation. The one exception — the degree-stratified AC-5 null — is documented, reverted, and named as production-scale follow-up rather than relaxed into the green bucket. The result is a test suite whose failures (if any) diagnose exactly one component each.

3. **Commit 3 (`b8373a9f9`)** — doc-alignment-only commit. Rewrites ADR-154 §8.4, §9.2, §9.5, §11, §13 and `README.md` so every reference to the degree-stratified AC-5 null describes what actually shipped (interior-edge null) rather than the attempted-but-reverted stratified variant. No code changes.

4. **Commit 4 (`bd26c4ee4`)** — fills BENCHMARK.md §4.5 with the measured SIMD saturated-regime speedup (1.013×, NOT hitting the ≥ 2× target). Replaces the pre-measurement guess paragraph with the post-measurement diagnosis: the hot path has migrated off subthreshold arithmetic onto spike delivery + CSR lookup + observer raster-write. Names Opt D (delay-sorted CSR) as the correct next lever.

5. **Commit 5 (`cf21327c9` / `feat/connectome-flywire-ingest`)** — adds the fixture-driven FlyWire v783 ingest module called out as the first item of §13. `src/connectome/flywire/{mod,schema,loader,fixture}.rs` + `tests/flywire_ingest.rs` (17/17 pass). 1 441 new LOC; max file 437. Deps: `csv = "1.3"` (already in workspace), `tempfile = "3"` dev-dep. No regression in any existing test.

6. **Commit 6 (`b805d7158` / `feat/observer-sparse-fiedler`)** — adds the sparse-Fiedler dispatch at `n > 1024`. `src/observer/sparse_fiedler.rs` (452 LOC) + `tests/sparse_fiedler_10k.rs` (2/2 pass in 19 ms at N=10 000). Cross-validation rel-error 3×10⁻⁵ vs the dense path. Memory O(n + nnz) = 40× reduction at matching scale. AC-1 bit-exact at N=1024 unchanged.

7. **Commit 7 (`a3cca1c5c` / `feat/lif-delay-sorted-csr`)** — Opt D delay-sorted CSR delivery path. `src/lif/delay_csr.rs` (398 LOC) + `tests/delay_csr_equivalence.rs` + `benches/delay_csr.rs`. Opt-in behind `EngineConfig.use_delay_sorted_csr` (default `false`, AC-1 untouched). Measured 1.5× kernel-only (~15 ms → ~10 ms per step); 1.00× top-line because the Fiedler detector dominates by ~450:1 (see §16). Equivalence exact at 51 258 spikes (rel-gap 0.0).

The three agent commits (5, 6, 7) were produced concurrently in isolated worktrees by a 3-agent swarm (hierarchical topology, specialized strategy, per CLAUDE.md §Swarm Configuration). They touched disjoint subtrees (connectome/, observer/, lif/), merged cleanly into `research/connectome-ruvector` in commit-order-5-then-6-then-7, and the consolidated test suite is green: **58 tests pass, 0 fail** across all feature combinations.

## 12. GPU acceleration path (§6.4)

The example is CPU-first by design — every SOTA claim in §3.4 is measured on CPU and the correctness contract (AC-1) pins the CPU trace as canonical. GPU is additive infrastructure: a throughput uplift for the motif SDPA batch (and eventually the Fiedler power iteration at larger `n` and dense LIF at Tier 2) that does not own any correctness claim.

### 12.1 Scope

- **SDPA batch for motif retrieval**. The canonical target: 10 000 windows × 10 bins × 64 dims × batched SDPA. Expected wins from transfer-bound CPU to device-resident tensors are in the 5–50× range once the kernel is fused.
- **Dense matvec for Fiedler at scale**. At Tier 2 (~10^5 neurons), the co-firing Laplacian eigenproblem outgrows the 96×96 Jacobi path and needs either a sparse power iteration or a dense GPU matvec. GPU is the right substrate for the dense variant.
- **Dense LIF at Tier 2**. Out of scope for this example; included here for completeness. At 10^5 neurons, dense-path LIF with GPU conductance updates becomes competitive with the event-driven CPU path.

### 12.2 Backend choice — cudarc primary, wgpu fallback

- **Primary**: `cudarc` 0.13+ with NVRTC kernel compilation. Direct CUDA, minimal host overhead, well-trodden path on Linux.
- **Fallback**: `wgpu` with WGSL compute shaders. Cross-vendor (Metal, Vulkan, DX12). Higher per-kernel overhead but unblocks macOS / ROCm development.

The `gpu-cuda` feature flag in `Cargo.toml` selects the `cudarc` path. The feature is off by default; the CPU path remains the correctness reference. If `cudarc` cannot link at compile time or at runtime against the host CUDA toolkit, the stub in `src/analysis/gpu.rs::CudaBackend::new()` returns an actionable error, the bench skips the GPU arm, and `GPU.md` documents what blocked.

### 12.3 Determinism contract

FP ordering on GPU is not bit-exact with CPU. The contract:

- CPU path is canonical. AC-1 determinism is measured on CPU.
- GPU path is allowed ≤ 1e-5 absolute error against CPU on motif vectors.
- `ComputeBackend::name()` is included in bench sub-report keys so CPU and GPU numbers are always paired and never conflated.

### 12.4 Positioning

This is **scaling infrastructure**, not a new scientific claim. A GPU uplift on the SDPA batch does not change any acceptance-criterion target. It changes the `BENCHMARK.md` §8 row labeled "gpu_sdpa_10k" and nothing else. If a reviewer cites a GPU number as evidence of brain-simulation progress, that is a positioning failure and the ADR's §3.1 rubric applies.

## 13. Follow-up work

Status as of the commit-14 consolidation of 6 follow-up items attempted (3 landed, 3 reverted after measurement). Items marked **✓ shipped** landed; **✗ reverted** were attempted and disproven; **→ next** names the follow-up lever. Items without a mark remain outside this example's scope.

- ✓ **FlyWire v783 ingest (fixture-driven)** — commit 4, `src/connectome/flywire/{mod,schema,loader,fixture}.rs` + `tests/flywire_ingest.rs`. Parses the published FlyWire TSV format into our `Connectome` via a 100-neuron hand-authored fixture; 17/17 tests pass. **→ next:** streaming ingest from the real ~2 GB release tarball + soma-distance-scaled delay model + `NeuronMeta` schema extension (`nt_confidence`, `soma_xyz`, `hemilineage`).
- ✓ **Sparse Fiedler dispatch at N > 1024** — commit 5, `src/observer/sparse_fiedler.rs` + `tests/sparse_fiedler_10k.rs`. `HashMap`-accumulated sparse adjacency → `ruvector-sparsifier::SparseGraph` → shifted-power iteration. Measured 19.25 ms at N = 10 000; 40× memory reduction vs dense at matching scale; cross-validation rel-error 3×10⁻⁵ at N = 256. **→ next:** Lanczos-with-full-reorthogonalization driver for `λ₂ ≪ λ_max` on path-like topologies; N=139 000 fixture calibration.
- ✓ **Delay-sorted CSR delivery path (Opt D)** — commit 6, `src/lif/delay_csr.rs` + `benches/delay_csr.rs`. Opt-in behind `EngineConfig.use_delay_sorted_csr` so AC-1 at N=1024 is untouched. Measured 1.5× at the kernel level (~15 ms → ~10 ms per step); 1.00× at the top-line bench because the Fiedler detector dominates by ~450:1. Equivalence vs scalar-opt: spike count exact (51 258, rel-gap 0.0). **→ next:** observer-side work — adaptive detect cadence under sustained high firing, incremental Fiedler accumulator, or dispatching the N=1024 detect to the sparse path via a threshold tweak (see §16 for the full discovery).
- ✓ **Streaming FlyWire v783 ingest** — commit 11, `src/connectome/flywire/streaming.rs`. Drops the intermediate `Vec<SynapseRecord>` by piping TSV rows directly into per-pre `Vec<Synapse>` buckets. Memory high-water-mark ~4.5 GB → ~1.7 GB on real v783. Byte-identical output to the non-streaming path on fixtures. 4 tests pass.
- ✓ **Degree-stratified AC-5 null sampler (port)** — commit 11, `src/connectome/stratified_null.rs`. The sampler investigated in commit 2's dev branch, ported as a library helper usable on either synthetic SBM or FlyWire-loaded Connectome. 5 tests pass (determinism, exclude-boundary, histogram-match, empty-boundary, degree check). Runs on synthetic today (collapses to same z_cut = z_rand per ADR-154 §8.4); ready for FlyWire-scale rerun.
- ✓ **Opt D paired-sample isolation bench** — commit 11, `benches/opt_d_isolation.rs`. Four Criterion arms across the {use_optimized, use_delay_sorted_csr} product, all with commit-10 adaptive cadence on. Runs via `cargo bench -p connectome-fly --bench opt_d_isolation`.
- ✗ **Lanczos-with-full-reorthogonalization for sparse Fiedler** (commits 12, reverted commit 13) — attempted as the named follow-up to sparse-Fiedler's path-topology failure mode. Agent shipped a full-reorthogonalization driver, but the test measured `rel-err = 3127 %` against the analytical λ₂ on a 256-node path: standard Lanczos on the Laplacian converges on `λ_max`, not `λ₂`, without shift-and-invert or explicit deflation. Reverted pending a proper shift-and-invert implementation (~200+ LOC plus a linear solver per iteration). **→ next:** shift-and-invert Lanczos or LOBPCG; or keep shifted-power iteration and accept `λ₂ = 0` on path-like fixtures as documented.
- ✗ **DiskANN / Vamana motif index** (commit 13, reverted commit 14) — attempted as the named follow-up to AC-2's precision@5 = 0.60 gap. Agent shipped a Vamana index + 605-window corpus but measured `precision@5 = 0.551` — *worse* than brute-force 0.60 on the same data. Diagnosis: the bottleneck isn't the index choice but the corpus's `distinct_labels = 4` / `max_label_share = 0.49` structure. **→ next (superseded by item 10 below):** see expanded-corpus result.
- ✗ **Expanded 8-protocol labeled AC-2 corpus** (attempted commit 15, reverted) — the named follow-up to item 8. Built a corpus with 8 distinct stimulus protocols spanning sensory-subset, frequency, amplitude, and duration axes; achieved `distinct_labels = 8, max_share = 0.12` (vs DiskANN's 4 / 0.49). Measured `precision@5 = 0.089` at 400 ms simulations and **0.117** at 140 ms early-transient windows — effectively random for 8 classes (baseline 0.125). Diagnosis (ADR §17 item 10): the SDPA + deterministic-low-rank-projection encoder on this substrate is *protocol-blind*. Stimulus-specific dynamics dissipate inside ≲ 150 ms as the substrate saturates into a common regime; the encoder captures the saturated raster, not the stimulus identity. **→ next:** change the encoder (CEBRA / learned contrastive), the substrate (real FlyWire ingest), or the label definition (raster-regime labels rather than stimulus-protocol labels). The first of those three is the cheapest investigation and is named for a separate ADR — it is a research question, not an engineering lever.
- ✗ **Incremental Fiedler accumulator (BTreeMap)** (commit in branch, reverted) — attempted as ADR-154 §16 lever 3. Agent shipped a `BTreeMap<(NeuronId, NeuronId), u32>` updated in `on_spike` + `expire`. Measured: AC-5 wallclock went from 100 s (post-commit-10) to 579 s — **5.8× slower** at top-line. Diagnosis: at saturated firing (~50 k `on_spike` calls, 20 k-spike window) the accumulator's per-spike BTreeMap insert/decrement cost (~100 ns per op) eats the algorithmic savings vs the per-detect dense pair-sweep + adaptive-cadence combination that already cut detect count 4×. The algorithmic argument ("O(|edges|) detect" vs "O(S²) detect") is right; the constant-factor is wrong at demo scale because adaptive cadence + dense is L1-cache-friendly in a way HashMap/BTreeMap cannot be. **→ next:** flat `Vec<(u32, u32, u32)>` accumulator with a sorted-insert contract, or an `AHashMap` variant; both would need another Criterion pass before merge.
- **Cross-path determinism** — bit-identical spike traces across baseline, optimized, and SIMD. Today only *within* a path. Requires a canonical in-bucket ordering contract; see `docs/research/connectome-ruvector/03-neural-dynamics.md` §11.
- **DiskANN motif index (better-conditioned corpus)** — see the ✗ entry above. Moves the motif kNN off brute-force once the corpus structure supports a higher precision ceiling.
- **Live CUDA kernel** — `cudarc` 0.13 on CUDA 13.0 / 5080 driver ABI. Opens `cudarc::driver::CudaContext`, compiles an NVRTC kernel for batched SDPA, warm-boots it outside the bench loop.
- **NeuroMechFly / MuJoCo body** — Phase 3 of the implementation plan. Replaces the current deterministic current-injection stimulus stub with a closed-loop body.
- **Leiden community baseline** — today AC-3a pairs against two in-tree baselines: level-1 greedy modularity (ARI ≈ 0.174 on default SBM) and multi-level Louvain (ARI = 0.000 — see §17 item 11: aggregation over-merges without Leiden's refinement). A proper Leiden pairing is the natural next step and the in-tree Louvain implementation gives the integration a direct comparison target. Effort: ~300–500 LOC in `src/analysis/structural.rs` to add the refinement phase, plus a test that asserts Leiden ≥ multi-level Louvain on the same graph.
- **Degree-stratified AC-5 null at FlyWire ingest scale** — the degree-stratified random-cut null (§8.4) collapsed the effect size at N=1024 synthetic SBM because the functional boundary and the degree-matched hubs overlap. At FlyWire v783 scale (~139 k neurons, much heavier non-hub tail) the null is expected to separate. The prototype sampler is preserved in `tests/acceptance_causal.rs` git history (pre-revert version) for direct port once the FlyWire streaming ingest lands.

None of the above blocks the current example's correctness contract. Each is a named hand-off to a future artifact.

## 16. Measurement-driven discovery — Fiedler detector dominates the saturated bench

Commit 7 (delay-sorted CSR) was the planned lever for closing the saturated-regime throughput gap that SIMD failed to close in commit 2. The bench produced a surprise that is worth preserving as an ADR entry because it reshapes the roadmap.

**What we expected.** Delay-sorted CSR removes per-step branch misprediction and scattered writes in the spike-delivery hot loop. We projected ≥ 2× on the top-line saturated `lif_throughput_n_1024` bench.

**What we measured.**

- Kernel-only microbench (detector disabled): **~15 ms → ~10 ms per simulated step, i.e. 1.5× faster.** The optimization is real.
- Full-bench median (detector enabled, default): **6.75 s for the scalar-opt + delay-csr arm vs 6.75 s for scalar-opt alone.** The optimization is invisible.

**Diagnosis.** Profiling showed the observer's Fiedler coherence-drop detector dominates wallclock by approximately 450:1 in the saturated regime. Per detect call (24 per 120 ms bench at 5 ms cadence) the detector does:

1. O(n²) pair sweep over ~21 k co-firing-window spikes to build the adjacency.
2. O(n²)–O(n³) eigendecomposition of the resulting ~1024-neuron Laplacian.

At n_active ≈ 1024, that puts the detector at ≈ 6.8 s of the 6.75 s wallclock. The kernel's 5 ms-per-step improvement is a rounding error on the top-line.

**Why this matters.** The diagnosis inverts the optimization roadmap. Before commit 6 the prevailing diagnosis (BENCHMARK.md §4.5 pre-measurement) named (a) spike dispatch, (b) CSR row-lookup, (c) observer raster-write as the three load-bearing items. (a) and (b) are in fact faster now; (c) is not the right target either. **The right target is the Fiedler detector itself**, and the detector already has a sparse path available (commit 5) that is not engaged at n_active = 1024 because the dispatch threshold is `n > 1024`.

**What to do next (named, not shipped here).** In decreasing bang-for-buck order:

1. ~~**Adjust the sparse-Fiedler dispatch threshold** to cover the saturated N=1024 case — likely drops the detector cost by ≥ 10× on its own, at which point Opt D's 1.5× kernel win becomes visible on the top-line bench.~~ **(Attempted commit 9, reverted after measurement.)** Lowering the threshold from 1024 to 96 (so everything above Jacobi's exact ceiling goes to the sparse path) produced a **3× regression** — 20.1 s vs 6.75 s on `lif_throughput_n_1024`. The sparse path's `HashMap` accumulation + `SparseGraph` canonicalisation hop adds more overhead at n≈1024 than it saves by skipping the dense O(n²) Laplacian build. The sparse path is a **scale win** (memory + wallclock at n ≥ 10 000) **not a demo-size speed win**. The threshold stays at 1024. See BENCHMARK.md §4.7 update.
2. ✓ **Adaptive detect cadence** — **shipped commit 10. Measured 4.29× speedup** on `lif_throughput_n_1024` (1.57 s vs 6.74 s scalar-opt pre-adaptive). In sustained saturated firing the co-firing window density passes `5 × num_neurons`; when it does, `current_detect_interval_ms()` routes to a 4× backoff (20 ms instead of 5 ms) until density drops. 14 LOC addition to `src/observer/core.rs`. AC-1 bit-exactness, AC-4-any, AC-4-strict (≥ 50 ms lead on ≥ 70 % of 30 trials) all preserved — the 20 ms cadence still gives ≥ 2 detects inside any 50 ms lead window. First optimization on this branch to clear the ≥ 2× ADR-154 §3.2 saturated-regime target.
3. **Incremental Fiedler accumulator** — the O(n²) pair sweep is re-done each detect. An accumulator updated per spike in `on_spike` removes the sweep entirely. Larger surgery than (2); still the cleanest long-term fix if detector cost needs to drop another order of magnitude, but not needed after commit 10 hits the top-level target.

The remaining item (3) is a named follow-up, not required for the demonstrator's SOTA target. Commit 10 is the load-bearing commit on the optimization arc.

**Lesson for the ADR's risk register (see §14, new row):** *measurement before optimization is necessary but not sufficient — measurement after optimization is what catches misdirected effort.* Commit 2's honest `BENCHMARK.md` entry ("we missed 2× SIMD, diagnosis to follow in a later commit") was correct that SIMD is the wrong lever; its guess about which other lever to pull next was wrong. Commit 7's empirical answer — "Opt D is real but drowned by a detector cost we hadn't measured" — is the kind of finding that only survives the measurement step, not the planning step. And commit 9's follow-up ("the obvious threshold fix is a 3× regression, not a win") is the same lesson applied one more level down: *even after a correct diagnosis, the obvious remediation still needs the measurement*.

## 14. Risk register

This section enumerates the risks this ADR is aware of and how the example stays within bounds on each.

| Risk | Surface | Mitigation |
|---|---|---|
| Positioning creep (upload / AGI / consciousness language) | README, BENCHMARK.md, commit messages, PR descriptions | §3.1 rubric binds on every artifact; first commit on this ADR passed review against `docs/research/connectome-ruvector/07-positioning.md` §6 |
| AC threshold drift (relaxing a SOTA target to make a test green) | `tests/acceptance_*.rs` | Do NOT weaken thresholds in the test code. Record the gap in `BENCHMARK.md`. Commit 2 on this ADR is governed by this rule. |
| Benchmark fabrication (quoting numbers we did not measure) | BENCHMARK.md, BASELINES.md | Every number in BENCHMARK.md is reproduced by the Criterion one-liner in §1; every number in BASELINES.md cites the paper and page/figure if we did not re-run it. |
| Determinism rot (baseline vs optimized vs SIMD paths diverge across Rust versions) | `tests/acceptance_core.rs::ac_1_repeatability` | AC-1 runs bit-exactness within path; cross-path is not claimed. A Rust upgrade that changes FP intrinsic behavior fails AC-1 loudly. |
| Scope creep (adding production-stack features to the example) | `src/*` | The 4000-LOC total budget (§3.2) and 500-line-per-file budget are enforced by the commit check in `BENCHMARK.md` §2 reproducibility. Tier 2 features go into the production crates, not here. |
| GPU numbers leaking into the correctness narrative | `BENCHMARK.md`, commit messages | §12.4 binds: GPU is infrastructure, not a claim. AC-1 is CPU-only. |
| Unreviewed novelty claims (the four in §9 inflate over time) | README, ADR-154 §9 | Each novelty claim is dated to a commit and backed by a file path. Any new claim requires a new commit, a new file path, and a review pass. |
| Pre-measurement diagnosis mis-directs the next optimization | `BENCHMARK.md`, ADR-154 §16 | Pre-measurement guesses about "which of the three hot paths dominates" can be wrong (commit 2 named three candidates; commit 6 found the *actual* dominant cost was a fourth we hadn't named — the Fiedler detector). **Now a 7-of-15 disproven data point across §17.** Mitigation: BENCHMARK.md requires the post-measurement diagnosis to be landed in the *same* commit as the optimization, not in a later commit. If the measurement says the optimization is invisible on the top-line, the next commit direction comes from the *measured* profile, not the pre-measurement guess. **The 2-of-15 successes (items 6 adaptive cadence, 14 Leiden refinement) both shared a pattern — structure the problem on an orthogonal axis rather than pushing harder on the axis an earlier item ran into.** That rule is now the default mental model for choosing the next lever. |
| Cross-path envelope decision (accepting 0.5 % spike-count divergence vs pursuing bit-exact) | `tests/cross_path_determinism.rs`, ADR-154 §15, §17 item 15 | The bucket-sort contract (commit 23) delivered canonical in-bucket *dispatch order* but not cross-path bit-exact *spike traces*. Root cause: optimized path's active-set pruning is a legitimate correctness deviation from the baseline's dense update. Both behaviours are correct-by-ADR; they produce genuinely different spike populations. **Decision recorded here so future commits don't re-open the question as a "bug":** the shipped contract is within-path bit-exact + cross-path ≤ 10 % spike-count envelope (measured 0.5 %). Bit-exact cross-path would require either running both paths with active-set off (bench-only) or teaching the baseline the same active-set (defeats the purpose). **Not a threshold to weaken or tighten; the envelope is the level at which the claim is publishable.** |
| Cheap-alternative predictions rarely survive the saturation workload | `BENCHMARK.md` §4.5 / §4.7 / §4.11 | Each time a commit names a "cheaper alternative for a future iteration" as a parenthetical (Opt D, lazy-skip, bucket-radix), measurement on the subsequent iteration tends to deliver less than the parenthetical predicted. Examples: Opt D delivered 1.5× kernel-only but 1.00× top-line; lazy-skip was null at saturation; CPU→GPU SDPA stayed unmeasured. Mitigation: future "cheaper alternative" parentheticals should name *the workload they would win on*, not just the percent gain. If the parenthetical doesn't name a measured-or-credible workload, it's a speculative parenthetical and labelled as such. |

This register is not comprehensive. It is the set of risks the branch has surfaced by running into them (positioning creep, threshold drift, null-distribution sloppiness, pre-measurement mis-diagnosis, envelope-vs-bit-exact framing, speculative-parenthetical predictions). Future commits are expected to add rows; they are not expected to remove rows.

## 17. Twenty-three measurement-driven discoveries (roll-up)

Each of the twenty-three is attached to the commit that produced it and the lesson it encoded for future work.

| # | Commit | Finding | Lesson |
|---|---|---|---|
| 1 | 3 (`b8373a9f9`) | Degree-stratified AC-5 null collapses at N=1024 SBM (`z_cut = z_rand = 2.12σ`) | Null formulation that matches structure too closely collapses the signal — the null has to be *different* from the boundary along the load-bearing axis |
| 2 | 4 (`bd26c4ee4`) | SIMD saturated gain = 1.013×, not ≥ 2× target | Adding lanes to a loop that rarely executes gives nothing — measure regime before picking lever |
| 3 | 4 (`bd26c4ee4`) | Observer buffer-reuse is 3 % slower than calloc | OS-zeroed calloc pages beat explicit-loop zeroing for cold allocations |
| 4 | 7 (`a3cca1c5c`) | Fiedler detector dominates saturated bench 450:1 | Diagnosing "kernel-bound" without a profile is guessing — measurement found the actual dominant cost was the *detector*, not the kernel |
| 5 | 9 (`3a6b70dcd`) | Sparse-Fiedler threshold drop 1024 → 96 is a 3× regression | The "obvious next fix" is wrong when scale trade-offs don't point where the algorithm argument says they should — HashMap+canonicalisation costs > O(n²) dense for n ≈ 1024 |
| 6 | 10 (`3c2377f50`) | Adaptive detect cadence hits 4.29× — **first ≥ 2× win** | Change *when* the detector runs, not *what* it does or *how* it's represented. The 14-LOC heuristic beat every attempted structural change |
| 7 | 12 (Lanczos, reverted) | Standard full-reorthog Lanczos on L converges on λ_max, not λ₂ (rel-err 3127 % on path-256) | "Use Lanczos" is not a cheaper alternative to the underlying numerical problem. Shift-and-invert or deflation is required; neither is a 500-LOC job |
| 8 | 13 (DiskANN, reverted) | Vamana at 605-window corpus scores 0.551 (worse than brute-force 0.60) | The AC-2 gap was not index-algorithmic; the corpus's 4-distinct-labels / 0.49-max-share structure caps precision no matter what ANN index is used |
| 9 | 14 (Incremental Fiedler, reverted) | BTreeMap accumulator makes AC-5 5.8× slower (100 s → 579 s) | Algorithmic complexity doesn't beat constant factors at this scale — BTreeMap insert/decrement (~100 ns/op) at saturated firing costs more than the pair-sweep it eliminates, *and* adaptive-cadence already cut the sweep frequency 4× |
| 10 | 15 (labeled AC-2, reverted) | 8-protocol labeled corpus still can't break the AC-2 precision ceiling: 400 ms → precision@5 = 0.089, 140 ms early-transient → 0.117 (vs random 0.125 for 8 classes) | **SDPA + deterministic low-rank projection on this substrate is protocol-blind.** Expanding the corpus from 4 → 8 protocols with max-share 0.12 did not help — stimulus-specific dynamics dissipate inside ≲ 150 ms as the substrate saturates into a common regime, and the SDPA encoder captures that saturated raster rather than the stimulus identity. The AC-2 gap is neither an index problem (DiskANN tried — item 8) nor a corpus-size problem (this test tried). It is an **encoder-substrate pairing** problem. Fixing it requires either (a) a different encoder (CEBRA / learned / task-specific contrastive), (b) a different substrate (real FlyWire may respond more protocol-specifically), or (c) a different label definition (raster-structure labels, not stimulus-protocol labels). None of those three are in this demonstrator's scope. |
| 11 | 17 (multi-level Louvain baseline) | Multi-level Louvain scores ARI = 0.000 on the default SBM vs level-1 greedy's ARI = 0.174 — the aggregation-based variant over-merges communities | **Louvain without Leiden's refinement phase collapses to a single super-community on hub-heavy SBMs.** By level 2 the aggregation absorbs structurally distinct communities into one super-node and there's no mechanism to un-merge. This is the documented failure mode Leiden's refinement (Traag et al. 2019) was specifically introduced to fix. The multi-level implementation is kept in `src/analysis/structural.rs::louvain_labels` with a docstring warning; AC-3a publishes both scores side-by-side so the future Leiden integration has a direct comparison row. Lesson: "more iterations" is not a monotonic improvement in community detection — without a well-connectedness guarantee, additional passes can strictly regress the signal. |
| 12 | 19 (rate-histogram encoder A/B) | Rate-histogram and SDPA both score below random on AC-2: `SDPA = 0.072` vs `rate-histogram = 0.079` (delta +0.007 within tie band; random for 8 classes = 0.125) | **The encoder axis is empirically ruled out.** Controlled A/B on the same 8-protocol labeled corpus that disproved SDPA in item 10: the crudest possible alternative (raw per-neuron-per-time-bin spike counts, no projection, no attention) neither improved nor meaningfully regressed the result. If the simplest encoder preserves all the raster information and still scores ~ SDPA, the encoder is not what's losing the protocol-identity signal — the saturated substrate is. The ADR §13 three-axis framing for AC-2 (encoder / substrate / labels) now has one axis measurement-ruled-out; the remaining two are substrate (real FlyWire replaces synthetic SBM) and labels (raster-regime rather than stimulus-protocol). Both are research-level pivots, not engineering levers. |
| 13 | 21 (raster-regime labels test) | Re-labeling the same corpus by `(dominant_class × spike_count_bucket)` instead of stimulus-protocol-id collapses to **2 distinct labels with max_share = 0.92** across 104 windows from 8 protocols. Naive precision@5 = 1.000 is trivially explained by class imbalance, not signal. | **The labels axis is also empirically ruled out.** Changing what the ground truth labels are from "stimulus protocol" to "raster regime" doesn't help because the substrate itself collapses every stimulus-driven window into essentially the same raster regime — one dominant class, one count bucket, ~92% of all windows. The finding *is* the content: at the N=1024 synthetic SBM scale, there is no label scheme that carries enough diversity for AC-2 precision to mean anything. Of the three AC-2 remediation axes named in item 10 (encoder / substrate / labels), **items 12 and 13 eliminate encoder and labels; substrate is the sole remaining lever.** That is real FlyWire v783 ingest replacing the synthetic SBM — no longer a research question, a data-ingest engineering item (see §13 "Streaming FlyWire v783 ingest" which is shipped but fixture-only; the real-data path still requires downloading the 2 GB release). |
| 14 | Leiden merge | Leiden's three-phase (local moves → refinement → aggregate) recovers **ARI = 1.000** on a hand-crafted 2-community planted SBM where multi-level Louvain collapses to ARI = 0.000. On the default hub-heavy SBM Leiden scores ARI = 0.089 (modularity resolution limit territory). | **Traag et al. 2019's refinement phase fixes the exact Louvain collapse from discovery #11.** The planted-SBM perfect recovery is a direct vindication — refinement works when the modularity landscape has a clear structure for it to find. On default-SBM the low ARI is a modularity-resolution-limit artefact (Fortunato & Barthélemy 2007), not a Leiden implementation bug; the implementation tracks the best-modularity partition across levels as a belt-and-braces workaround. CPM-based quality function (Traag's own default in `leidenalg`) is the documented next step to escape the resolution limit. This is the first Louvain-family algorithm in the branch that meets a named SOTA target on *any* input. |
| 15 | Bucket sort + cross-path test | `TimingWheel::drain_due` now sorts each bucket ascending by `(t_ms, post, pre)` before delivery, matching `SpikeEvent::cmp` on the heap path. On the AC-1 stimulus at N=1024: baseline produces 195 782 spikes, optimized produces 194 784 — **~0.5 % spike-count divergence** that persists despite the sort. | **The sort delivers canonical *dispatch order* on the wheel; it does NOT deliver cross-path bit-exact *spike traces*.** Root cause (new): the optimized path's active-set pruning is a *correctness deviation* from the baseline's dense subthreshold update — neurons near threshold under continuous dense updates can leak below it, but stay above under active-set updates. Both behaviours are correct-by-ADR; they produce genuinely different spike populations. `tests/cross_path_determinism.rs` gates on the ADR-154 §15.1 10 % envelope (measured 0.5 %, well inside) rather than bit-exactness, which would require either running both paths with active-set off (bench-only) or teaching the baseline the same active-set (defeats the purpose). The shipped contract is: within-path bit-exact, cross-path ≤ 10 % spike-count envelope. |
| 16 | CPM-Leiden γ sweep + planted-SBM test (un-normalized) | Implemented Traag's CPM quality function as `analysis::leiden::leiden_labels_cpm`. γ sweep on the default N=1024 SBM across γ ∈ {0.005…1.0}: **every γ ≤ 0.5 collapses the graph to 1 community**; γ = 1.0 gives 15 communities with ARI = -0.039 (worse than modularity-Leiden's 0.089). The 2-community planted SBM also collapses to 1 community at γ = 0.05. | **Naive CPM on weight-scaled edges is the wrong formulation.** The CPM move gain `k_{v,C} - γ·n_C` parametrizes γ in *edge-weight units*, but synapse weights here are f64 of order 10–100. At γ = 0.05 the penalty `γ·n_C` is dwarfed by any positive inter-community sum-of-weights, so level-1 greedily merges everything into one community; at γ = 1.0 CPM still over-merges because per-pair weight magnitudes are >> 1. Traag's own `leidenalg` normalizes edges (or, equivalently, rescales γ by total-weight density) — **weight-normalized CPM is the next attempt** (item 17 below). This is the second time on this branch that an "obvious from the paper" implementation needs a scaling rider to be usable at the substrate's real weight distribution (discovery #1 was the same lesson on the AC-5 null; discovery #7 on Lanczos). Pattern: *published-algorithm implementations usually need a substrate-specific normalization before they meet the paper's stated behaviour on non-toy inputs.* |
| 17 | Weight-normalized CPM + γ-sweep at scale | `leiden_labels_cpm` rewritten to pre-normalize all edge weights by their mean (so mean edge weight = 1.0 and γ is dimensionless). Re-swept across γ ∈ {0.1, 0.5, 1, 2, 4, 8, 16, 32, 64}. **Planted 2-community SBM: ARI = 1.000 at γ ∈ {2, 4}** (perfect recovery, matches modularity-Leiden's planted result — item 14). Default N=1024 hub-heavy SBM: best 2-way-coarsened ARI = 0.020 at γ=2 with **109 distinct communities** (close to the ground-truth 70 modules). | **The weight-normalization rider works.** CPM recovers planted community structure perfectly once γ is in the right scale range (γ ~ super-edge magnitude), confirming the paper's claim and validating the rider from item 16. On multi-module graphs, however, the *2-way coarsening inherited from AC-3a* undersells CPM's output: 109 communities mapped to a hub-vs-non-hub binary label loses nearly all the signal. **The measurement is now the limit, not the algorithm.** The natural next step is a full-partition ARI or a module-recovery fraction metric that respects CPM's native community count. Weight-normalized CPM is a successful item on this branch (second community-detection algorithm that matches its paper's planted-graph performance) — but its win on the 70-module substrate won't be visible until the measurement catches up. Code: unchanged API; `leiden_labels_cpm(conn, gamma)` now takes dimensionless γ. |
| 18 | Full-partition ARI lifts the measurement | Added `full_partition_ari(predicted, truth)` to `tests/leiden_cpm.rs` — standard Hubert-Arabie ARI against the 70-module SBM ground-truth label vector, not the 2-way hub-vs-non-hub coarsening. Re-measured the γ sweep. **Result on the default N=1024 SBM: modularity-Leiden full_ari = 0.107; CPM @ γ=2 full_ari = 0.393** — a **3.7× improvement** over modularity-Leiden on the correct metric. | **The measurement fix was the lever — not another algorithm.** Item 17 predicted this exactly: CPM's 109 communities were recovering ~57 % of the 70-module structure, but the 2-way coarsening was throwing all of that away. With the correct metric, CPM @ γ=2 becomes the new state-of-the-art community detector on this substrate — **4th unambiguous win on the branch** (after adaptive cadence, modularity-Leiden refinement, weight-normalized CPM at planted scale). Still below the 0.75 AC-3a SOTA target, but the gap is now a tractable 2× rather than a 38× mystery. This also closes out a recurring branch-wide failure mode: AC-3a's 2-way coarsening was inherited uncritically from the first AC-3 test; two community-detection algorithms (Leiden modularity, Leiden CPM) underperformed their paper's claims on it before the metric was finally upgraded. **Lesson for §14 risk register: a test's coarsening choice is as much a threshold decision as its numerical tolerances, and deserves the same review discipline.** Code: `tests/leiden_cpm.rs` helper; no production-code change (this is a measurement-correctness commit, not an algorithm commit). |
| 19 | Fine-γ sweep refines the CPM peak | Re-swept γ ∈ {1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, …} on the default SBM. **New peak: ari_full = 0.425 at γ ∈ {2.25, 2.5}** with 156 / 171 communities (up from 0.393 @ γ=2.0). At γ = 1.75 CPM recovers **exactly 70 distinct communities** — matching the ground-truth module count — with ari_full = 0.348. | **CPM's quality ridge on this substrate is between γ=1.75 and γ=2.75, peaking at γ ∈ [2.25, 2.5].** Two interesting facts emerge from the fine sweep: (a) the peak ARI is at a γ that produces *more* communities (156) than the ground truth (70), suggesting CPM's over-splitting is *aligned* with ground truth well enough that ARI tolerates the extra fragmentation; (b) the γ = 1.75 point that exactly matches the ground-truth community count (70) actually scores lower (0.348 vs 0.425) — CPM's 70 communities there don't align with the SBM's 70 modules as well as its 156 communities do at γ = 2.25. So on this substrate, "match the community count" and "maximize ARI" are distinct optimization targets, and the γ values for each differ. **CPM-Leiden on the default SBM is now at 0.425 vs modularity-Leiden's 0.107 — a 3.97× improvement, 57 % of the 0.75 AC-3a SOTA target.** The remaining 1.76× gap is likely the modularity-resolution-limit-adjacent ceiling of CPM-without-refinement. Adding a CPM-specific refinement phase (not the current modularity-refinement) is the named next lever. Code: `tests/leiden_cpm.rs` γ-list extended; no production-code change. |
| 20 | Full-partition ARI wired into AC-3a reveals level-1 greedy beats Leiden | AC-3a now publishes full-partition ARI alongside the 2-way coarsening. **Greedy modularity (level-1) full_ari = 0.308, beats modularity-Leiden's full_ari = 0.107** on the default SBM. Multi-level Louvain collapses (full_ari = 0.000). CPM @ γ=2.25 remains top at 0.425. Final ranking: CPM 0.425 > greedy level-1 0.308 > Leiden 0.107 > Louvain 0.000. | **Leiden's aggregation+refinement actively hurts module recovery on this substrate.** Greedy level-1 (one pass of local moves, no aggregation) gives 0.308 full-partition ARI; adding the aggregation + Traag refinement steps drops it to 0.107 — a **2.9× regression from a more sophisticated algorithm**. The refinement preserves well-connectedness (item 14's test passes) but does so at the cost of merging structurally-distinct communities from the level-1 output. This flips the expected order: on hub-heavy SBMs, *more algorithm is worse* when the objective is modularity and the target is module recovery. CPM, with its non-resolution-limited objective, sidesteps the issue. The engineering implication: **for AC-3a on this substrate, level-1 greedy modularity is a stronger baseline than multi-level Leiden.** The pattern echoes discovery #11 (multi-level Louvain collapse on hub-heavy SBMs) but at a finer granularity — item 11 said "Louvain aggregation breaks", item 20 says "even Leiden's refinement can't fully repair it because the underlying modularity objective has the resolution-limit issue". CPM (item 17) was the right escape. Code: `tests/acceptance_partition.rs` publishes the new line; no assertion change (ADR §14 threshold discipline). |
| 21 | CPM-vs-modularity seed-sweep reproducibility | Re-measured CPM @ γ=2.25 vs modularity-Leiden on 5 distinct SBM seeds (0x5FA1DE5, 0xC70F00D, 0xC0DECAFE, 0xBEEFBABE, 0xDEAD1234) at otherwise-default config. **CPM beats modularity on 5 / 5 seeds. Mean ratio 3.98× (matches the 3.97× headline from default seed). Mean CPM full-ARI 0.356, mean modularity full-ARI 0.105. Range 2.04× – 7.34×.** | **The CPM win isn't a single-seed artefact.** Five independent SBMs, five CPM-beats-modularity wins; the 3.98× mean is indistinguishable from the default-seed's 3.97× headline. The range (2.04–7.34) shows seed-dependent variance but no seed where modularity-Leiden catches or beats CPM. This strengthens the item-18 claim from "one measurement showed CPM 3.7× modularity" to "five measurements across different random graphs all show CPM beats modularity by ≥ 2×, mean ~4×." **The 4th-win claim (item 17) is now reproducibility-verified.** Code: `tests/leiden_cpm.rs::leiden_cpm_vs_modularity_across_seeds`, publishes all 5 seed results; asserts only that the mean ratio > 1.0 so a regression in `leiden_labels_cpm` fails loudly. |
| 22 | CPM-vs-modularity N-scaling sweep | Re-measured CPM @ γ=2.25 vs modularity-Leiden across three SBM scales with density held constant (num_modules = N/15): **N=512 / 35 modules** → cpm_full 0.322, mod_full 0.126, ratio 2.55×; **N=1024 / 70 modules** → 0.425 / 0.107, ratio 3.98× (the headline); **N=2048 / 140 modules** → 0.258 / 0.094, ratio 2.74×. **Mean ratio across scales 3.09×, min 2.55×, max 3.98× — CPM wins at every scale but the advantage peaks at N=1024.** | **CPM's 4× headline is N=1024-specific; the ratio is not scale-invariant.** Two facts here. First, CPM beats modularity-Leiden at every scale tested (2.55× → 3.98× → 2.74×), so the seed-sweep verdict (item 21: "CPM always wins") generalises across scale as well, not just seed. The direction of the finding holds. Second, both algorithms' *absolute* full-partition ARI drops at N=2048 (CPM 0.425 → 0.258; modularity 0.107 → 0.094), and CPM's absolute peak is at N=1024, not at N=2048. So the "0.425 on default SBM" number isn't what you'd quote at larger scales — it's closer to 0.26 at N=2048 with proportional density. This is the first empirical evidence that the CPM quality ridge identified at item 19 (γ ∈ [2.25, 2.5] at N=1024) is substrate-size-dependent; γ=2.25 is probably no longer the peak γ at N=2048, and the γ sweep would need to be re-run per scale to find the true ceiling. The 1.76× gap to the 0.75 AC-3a SOTA target is also N=1024-specific — at N=2048 the gap is ~2.9× under fixed γ. **Engineering implication: the "named next lever" of CPM-specific refinement should be benchmarked at multiple N before the result is quoted as "closes the gap".** Code: `tests/leiden_cpm.rs::leiden_cpm_vs_modularity_across_scales`, publishes per-scale numbers; asserts only that CPM wins at ≥ 1 scale (regression gate). |
| 23 | Per-scale γ sweep: peak γ shifts with N, and N=512 beats N=1024 | Follow-up to item 22. γ sweep {1.25, 1.75, 2.25, 2.75, 3.5, 5.0} at each scale with density held constant. **Peak full-ARI per scale: N=512 → 0.532 @ γ=2.75 (23 communities vs 35 truth); N=1024 → 0.425 @ γ=2.25 (156 vs 70); N=2048 → 0.332 @ γ=1.75 (187 vs 140).** Peak γ shifts monotonically downward as N grows (2.75 → 2.25 → 1.75). | **Two overlapping findings that invalidate both item 22's headline and item 19's "peak γ = 2.25" claim on the broader substrate.** First, **item 22's fixed-γ measurement was understated at both the smaller and larger substrate**: at N=512 the true CPM ceiling is **0.532**, which is 65 % higher than the fixed-γ reading of 0.322 and **higher than the N=1024 peak of 0.425**. At N=2048 the true ceiling is 0.332, also higher than the fixed-γ 0.258. Second, **the γ peak shifts monotonically with N** — at N=512 the optimum is γ=2.75, at N=1024 it's γ=2.25, at N=2048 it's γ=1.75. The trend is roughly Δγ ≈ -0.5 per doubling of N. This makes physical sense: under weight-normalized CPM, larger graphs have more edges per node on average even at constant density, so the 'merge penalty' γ·n_C needs to be *lower* per node to stay in equilibrium with intra-community weight gain. **The 0.532 figure at N=512 is now the best CPM ceiling observed on this substrate — within 1.41× of the 0.75 AC-3a SOTA target.** That narrows the residual gap from the item-19 1.76× headline at N=1024 to 1.41× at N=512, and argues that the scale at which to prove "we closed the gap" might not be the default N=1024 at all. **Engineering implication: γ should be swept per-substrate, not inherited from a different-N benchmark; publishing a peak-of-the-sweep ARI is the only honest quote.** Code: `tests/leiden_cpm.rs::leiden_cpm_gamma_peak_per_scale`. The 0.532 result also slightly weakens claim 3 in ADR §9 (the novelty around CPM-at-scale) — the best CPM performance on this substrate is at a scale (N=512) *smaller* than the default, not an emergent-at-scale pattern. |

The discoveries form a pattern: every "next lever named in the ADR" ultimately required an empirical test. **Eight** of the fifteen pre-measurement diagnoses tested on this branch proved wrong (items 7, 8, 9, 10, 12, 13, 15, 16). **Four unambiguous wins now: item 6 (adaptive cadence, 4.29× saturated-regime speedup), item 14 (Leiden refinement, perfect ARI on planted SBM where Louvain collapsed), item 17 (weight-normalized CPM-Leiden, perfect ARI on planted SBM + 109 communities on 70-module default SBM), and item 18 (full-partition ARI metric, lifting CPM's default-SBM score from 0.020 two-way to 0.393 full — 3.7× the modularity-Leiden baseline).** Items 6 and 14 followed the orthogonal-axis pattern. Item 17 was the first "rider from item 16 works as predicted" data point. Item 18 is a different shape again — a **measurement upgrade** that revealed an algorithm's prior 0.020 2-way score was hiding a 0.393 full-partition score. That's a new entry in the lesson catalogue: *a test's coarsening choice is as much a threshold decision as its numerical tolerances.* Three distinct "how a measurement-driven discovery lands" shapes now documented (orthogonal axis / rider matches paper / coarsening upgrade).

A secondary pattern, now quantified: *published-algorithm implementations usually need a substrate-specific normalization before they match the paper's stated behaviour on non-toy inputs.* Three instances confirmed — AC-5 null degree-scaling (item 1, still pending at FlyWire scale), Lanczos shift-and-invert (item 7, still pending), CPM weight normalization (item 16 → item 17 delivers). The CPM → normalized-CPM story is the first of the three to actually close: item 16 failed as predicted, item 17 succeeded via the predicted rider. That is both an instance of the "substrate-specific normalization" pattern *and* a data point showing the pattern is actionable — the rider, when named, works.

Applied to AC-2: five structurally-different remediations have been tested on the same SBM substrate — brute-force kNN (item 2 baseline); DiskANN (item 8); expanded-label corpus (item 10); rate-histogram encoder (item 12); raster-regime labels (item 13). All five plateau at or below the random baseline. Three of the four axes the ADR §13 framing named as potential fixes (encoder / corpus-size / labels) are now empirically ruled out. **The remaining axis is substrate** — real FlyWire v783 ingest replacing the synthetic SBM. That is no longer a research question but a data-ingest engineering item: the streaming-loader code exists (commit 11, `src/connectome/flywire/streaming.rs`) and passes fixture tests; what remains is downloading the real 2 GB release and re-running AC-2 against it. When that happens, AC-2 either hits its SOTA target or the final axis is disproven too — at which point the claim itself needs revision.

## 15. Determinism contract (expanded)

AC-1 repeatability requires that every run keyed by `(connectome_seed, stimulus_seed, engine_seed)` produces bit-identical spike traces. This section expands the mechanics of that contract across the three LIF paths.

### 15.1 Ordering rule

The determinism contract is a lexicographic ordering on events within a simulated time-step: `(t_ms, post_id, pre_id)`. Two events scheduled at the same `t_ms` with the same `post` are tie-broken by `pre`. This is invariant across all three paths:

- **Baseline** (BinaryHeap + AoS): `SpikeEvent::cmp` in `src/lif/queue.rs` implements the lexicographic order directly. The max-heap ordering is inverted so the *earliest* event pops first.
- **Optimized** (wheel + SoA): events inside a bucket are in push-order, which is deterministic given a fixed push schedule. Intra-bucket order within the wheel is *not* identical to the heap order — an event pushed later but with an earlier tie-break position in the heap lands in a different dispatch position under the wheel. This is documented in §4.2 as a known cross-path divergence.
- **SIMD** (wheel + SoA + f32x8): identical to optimized for queue behavior. The SIMD subthreshold kernel processes 8 neurons per SIMD cycle in the *same id-order* as the scalar optimized path; lane-wise arithmetic matches bit-for-bit when the host issues AVX2 FMA. Scalar tail runs the exact scalar recipe.

### 15.2 What AC-1 guarantees

- **Within a path**: bit-exact spike traces on repeat runs. Verified by `tests/acceptance_core.rs::ac_1_repeatability` comparing spike counts *and* the first 1000 `(neuron_id, t_ms)` pairs.
- **Across Rust versions**: not promised. An FMA → separate-mul+add change in LLVM, a change in `libm::expf` precision, or a new vectorizer heuristic can break AC-1. The remediation is to re-record the expected trace for the new toolchain, not to relax the test.
- **Across paths**: NOT promised. A bit-exact diff between baseline and optimized remains future work.

### 15.3 FP reproducibility on x86_64

The SIMD path on x86_64 depends on AVX or AVX2. On a host without AVX, `wide` falls back to two `f32x4` sub-registers; the arithmetic remains deterministic per-lane but the order of certain reductions differs. Since the SIMD kernel in `src/lif/simd.rs` does no cross-lane reductions (every arithmetic step is lane-independent), this does not affect determinism — but a future fused-kernel variant that introduces cross-lane sums must preserve lane-order.

### 15.4 Non-determinism sources intentionally excluded

- No OS-RNG anywhere. All randomness is `Xoshiro256**` seeded from `ConnectomeConfig` / `EngineConfig` / `AnalysisConfig`.
- No network calls.
- No wall-clock dependency in the deterministic code path (wall-clock timings exist only for bench annotation; they do not feed the simulation).
- No uninitialized memory reads; `#![deny(unsafe_code)]` is in `src/lib.rs`.
- No thread-schedule sensitivity: the example is single-threaded by design. Rayon / threadpool are not linked.
