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

Most existing pipelines simulate and observe. The differentiator this example is chasing is: *simulate, perturb, measure structural causality*. Concretely, that means mincut-surfaced boundaries become intervention handles, spike-window motifs become retrievable addresses for repeated functional states, and the coherence signal becomes a precursor-class predictor of behavioural divergence. Framed this way, the relevant peer technologies are interpretability and causal-intervention tooling for complex recurrent systems — not biological simulators. The outward framing for this line of work is "operating system for intelligence" / "structural intelligence infrastructure": a debugging and control layer for embodied graph systems whose structure is *knowable* (the connectome) rather than learned. No consciousness language, no upload framing, no AGI gestures — those are all still explicit non-goals, as `07-positioning.md` §6 binds.

## 2.2 Feasibility tiers (binding scope boundary)

Published analyses of connectome-scale simulation converge on three feasibility tiers. This ADR classifies itself against that table and fixes the boundaries.

| Tier | Scope | Neurons | Feasibility | This crate |
|---|---|---|---|---|
| **Tier 1** | fruit fly, partial mouse cortex | 10^4 – 10^5 | **Proven. Buildable today.** Memory fits on commodity CPU/SSD; biological parameters exist; dynamics regime demonstrated (2024 Nature). | **Target of this example.** |
| **Tier 2** | larger mouse regions, multi-region simulations | 10^5 – 10^6 | Hard but doable, approximately 12–24 months of focused engineering. Memory dominated by synapses; requires SSD-backed graph store and aggressive sparsification to stay in RAM. | **Deferred.** Lives in `ruvector-connectome` + `ruvector-lif` + `ruvector-embodiment` per `08-implementation-plan.md`. Not in this example. |
| **Tier 3** | full mammalian / full human brain | 10^9 – 10^11 | **Not feasible at any horizon in this ADR.** Compute, biological parameters, and connectome data are all insufficient. Even given perfect data, the system is underconstrained — too many free dynamical parameters per neuron, too many long-range synapses without delay / NT / sign, and no behavioural readout at sufficient fidelity to calibrate. | **Explicit non-goal.** |

The mission of this example is the Tier 1 demonstrator. Tier 2 is the crate-split plan and remains deferred. Tier 3 is an explicit non-goal at any horizon in this ADR — any future claim adjacent to Tier 3 requires a new ADR that confronts the feasibility wall head-on rather than gesturing past it.

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
