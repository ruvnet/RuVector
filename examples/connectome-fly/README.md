# connectome-fly

**Status:** Example crate for ADR-154.
**Positioning:** *A graph-native embodied connectome runtime with structural coherence analysis, counterfactual circuit testing, and auditable behavior generation.* This is **not** a consciousness-upload, mind-upload, or digital-person artifact. See `docs/research/connectome-ruvector/07-positioning.md` for the full hype-avoidance rubric.

## What it is

`connectome-fly` is a self-contained demonstrator for the research program documented in `docs/research/connectome-ruvector/` (nine-document deep dive) and formalized in `docs/adr/ADR-154-connectome-embodied-brain-example.md`. It wires together, in a single workspace crate:

1. A **synthetic fly-like connectome** — a deterministic stochastic block model matching the published FlyWire v783 summary statistics (modules, classes, log-normal weights, inhibitory fraction, hub modules).
2. An **event-driven leaky integrate-and-fire (LIF) kernel** with exponential conductances, absolute refractory period, CSR-backed synaptic dispatch, and two interchangeable queue back-ends (binary heap + AoS baseline; bucketed timing wheel + SoA optimized).
3. A **deterministic stimulus stub** that injects current into designated sensory neurons in place of an embodied simulator. Embodiment (MuJoCo / NeuroMechFly) is explicitly out of scope for this example — it is a Phase 3 deliverable of the full research plan (`08-implementation-plan.md`).
4. A **spike observer** that rasterizes spikes, computes a population-rate trace, and runs a Fiedler-value detector on the sliding co-firing graph to emit coherence-collapse events (the "structural fragility" signal from `05-analysis-layer.md` §5).
5. An **analysis layer** that plugs `ruvector-mincut`, `ruvector-sparsifier`, and `ruvector-attention` into the live spike stream: mincut-based functional partitioning of the connectome weighted by recent coactivation, and SDPA-embedded motif retrieval with a bounded in-memory kNN.

The scientific anchor is the 2024 Nature whole-fly-brain LIF paper showing that behavior emerges from connectome-only LIF dynamics without trained parameters. This example does **not** claim to reproduce that biology — the default connectome is a calibrated SBM toy. What it claims is that RuVector's graph primitives can be mounted live on a connectome-scale LIF simulation and surface useful structural signals.

## What's new on this branch (commit-8 consolidation)

Three capabilities landed concurrently from isolated-worktree agents and merged onto `research/connectome-ruvector`:

1. **Real FlyWire v783 ingest** (`src/connectome/flywire/`) — parses the published FlyWire TSV format into `Connectome` via `load_flywire(path)`. Fixture-driven tests exercise the full parse path without a ~2 GB download; 17/17 tests pass. The synthetic SBM generator remains available and unchanged.
2. **Sparse-Fiedler dispatch for N > 1024** (`src/observer/sparse_fiedler.rs`) — `O(n + nnz)` memory path via `ruvector-sparsifier`, validated at N = 10 000 in 19 ms wallclock, cross-validated against the dense path at N = 256 within 3×10⁻⁵ relative error. Dense-path behavior at `n ≤ 1024` unchanged.
3. **Opt D — delay-sorted CSR delivery path** (`src/lif/delay_csr.rs`) — opt-in behind `EngineConfig.use_delay_sorted_csr` (default `false`, AC-1 untouched). Measured 1.5× at the kernel level but 1.00× at the top-line saturated bench because the Fiedler detector dominates by ~450:1. See `BENCHMARK.md` §4.7 and ADR-154 §16 for the measurement-driven discovery that reshaped the roadmap.

All 58 tests across 11 test binaries pass. No regression in any existing acceptance criterion. Positioning rubric (no consciousness / upload / AGI language) holds across all added artifacts.

## Directory layout

```
examples/connectome-fly/
├── Cargo.toml
├── README.md                 this file
├── BENCHMARK.md              baseline and post-optimization numbers
├── BASELINES.md              head-to-head framing vs Brian2/Auryn/NEST/GeNN
├── GPU.md                    status of the gpu-cuda feature + unblock plan
├── src/
│   ├── lib.rs
│   ├── connectome/           SBM generator + binary serialization + FlyWire ingest
│   │   ├── generator.rs      synthetic SBM calibrated to FlyWire v783 stats
│   │   ├── schema.rs         typed IDs, NeuronMeta, Synapse
│   │   ├── persist.rs        binary mmap for reuse across runs
│   │   └── flywire/          real FlyWire v783 TSV ingest (fixture-tested)
│   │       ├── mod.rs        public load_flywire()
│   │       ├── schema.rs     NeuronRecord, SynapseRecord, CellTypeRecord
│   │       ├── loader.rs     TSV → Connectome
│   │       └── fixture.rs    100-neuron hand-authored FlyWire fixture
│   ├── lif/                  event-driven LIF kernel (AoS+heap / SoA+wheel / SIMD / delay-csr)
│   │   ├── engine.rs         hot loop, scalar + SIMD-gated subthreshold + delay-csr dispatch
│   │   ├── queue.rs          BinaryHeap baseline + bucketed timing wheel + push_at_slot fast path
│   │   ├── simd.rs           f32x8 vectorized subthreshold (feature: simd)
│   │   ├── delay_csr.rs      Opt D delay-sorted CSR delivery (opt-in via EngineConfig)
│   │   └── types.rs          EngineConfig (incl. use_delay_sorted_csr flag)
│   ├── stimulus.rs           deterministic current-injection schedules
│   ├── observer/             raster + population rate + Fiedler detector
│   │   ├── core.rs           on_spike hot path + dispatch to dense/sparse Fiedler
│   │   ├── eigensolver.rs    Jacobi (n ≤ 96) + shifted power iteration
│   │   ├── sparse_fiedler.rs sparse-Laplacian Fiedler for n > 1024 (O(n+nnz) memory)
│   │   └── report.rs
│   ├── analysis/             mincut partition + SDPA motif retrieval
│   │   ├── motif.rs          SDPA encoder + bounded in-memory kNN
│   │   ├── partition.rs      coactivation-weighted mincut (AC-3b)
│   │   ├── structural.rs     static mincut + greedy-modularity (AC-3a)
│   │   ├── gpu.rs            ComputeBackend trait + CPU/CUDA backends
│   │   └── types.rs
│   └── bin/run_demo.rs       CLI demo runner
├── tests/
│   ├── lif_correctness.rs        monotonicity + refractory-limit invariants
│   ├── connectome_schema.rs      schema + serialization round-trip
│   ├── analysis_coherence.rs     coherence detector fires on fragmentation
│   ├── acceptance_core.rs        AC-1, AC-2, AC-4-any, AC-4-strict
│   ├── acceptance_partition.rs   AC-3a (structural), AC-3b (functional)
│   ├── acceptance_causal.rs      AC-5 causal perturbation (interior-edge null)
│   ├── flywire_ingest.rs         FlyWire v783 TSV parse + round-trip (17 tests)
│   ├── sparse_fiedler_10k.rs     sparse-Fiedler scale test at N=10 000
│   ├── delay_csr_equivalence.rs  delay-csr spike-count equivalence vs scalar-opt
│   └── integration.rs            end-to-end non-empty report
└── benches/
    ├── lif_throughput.rs     LIF events/sec at N ∈ {100, 1024, 10_000}
    ├── motif_search.rs       kNN retrieval latency for spike-window embeddings
    ├── sim_step.rs           per-simulated-ms wallclock
    ├── gpu_sdpa.rs           CPU/CUDA SDPA batch (feature: gpu-cuda)
    └── delay_csr.rs          Opt D ablation bench (3-way: baseline / scalar-opt / + delay-csr)
```

## Feature flags

- **`default = ["simd"]`** — ships with SIMD enabled on all hosts.
- **`simd`** — enables `wide::f32x8` vectorization of the subthreshold LIF loop (Opt C in ADR-154 §3.2). Required to hit the ≥ 2× speedup in the saturated-regime `lif_throughput_n_1024` bench. Falls back to lane-wise scalar on hosts without AVX.
- **`gpu-cuda`** — opt-in GPU SDPA path for motif retrieval via `cudarc`. Off by default. If CUDA is not installed or `cudarc` cannot link, the stub in `src/analysis/gpu.rs` returns an actionable error and bench + tests skip the GPU arm. See `GPU.md` for status.

To disable SIMD for comparison:

```bash
cargo test --release -p connectome-fly --no-default-features
```

## How to run

```bash
# From the repo root.
cargo build --release -p connectome-fly
cargo test  --release -p connectome-fly
cargo run   --release -p connectome-fly --bin run_demo
# → JSON report on stdout.

# Or write the report to a file:
cargo run --release -p connectome-fly --bin run_demo -- /tmp/connectome-fly-report.json
```

Benchmarks:

```bash
cargo bench -p connectome-fly --bench lif_throughput
cargo bench -p connectome-fly --bench motif_search
cargo bench -p connectome-fly --bench sim_step
```

All benches are Criterion-backed and emit baseline vs. optimized comparisons.

## How to interpret the demo report

The runner writes a single JSON object. Each top-level field carries concrete meaning:

- `config` — stimulus schedule and engine flags. `use_optimized_lif: true` selects the SoA + timing-wheel kernel path.
- `connectome` — synthetic SBM stats. The `seed` is stable across runs; the connectome is bit-identical for a given seed.
- `simulation.total_spikes` — number of observed spikes in the run window.
- `simulation.mean_population_rate_hz` — average firing rate across the full 500 ms, Hz per neuron. High values (>200 Hz) indicate the network is in an excited regime; this is expected under the demo's default stimulus amplitude and reflects the demonstrator's *dynamics* not any claim about biology.
- `simulation.first_10_rate_samples_hz` — the first 10 of the 5 ms-binned rate samples. Zeros at the start are expected because stimulus onset is T = 100 ms.
- `coherence.events_total` — how many times the Fiedler value of the instantaneous co-firing Laplacian dropped below `threshold_factor · baseline_std` during the run. A populated list is evidence that the detector is wired correctly; interpretation as a *behavioral-precursor* signal requires the closed-loop stack (deferred).
- `partition.cut_value` — the mincut value over the recent-spike-weighted connectome edges. `side_a_class_histogram` and `side_b_class_histogram` show the class composition of each half; a meaningful split groups sensory and motor classes on opposite sides when the stimulus is fresh.
- `motifs[]` — top-k repeated spike-window motifs ranked by retrieval tightness (small `nearest_distance` = more repeated). `dominant_class` identifies which class contributes most of the spikes in the representative window; `frequency` counts how many windows clustered under this representative under the greedy radius-dedup.
- `timings_ms` — wallclock breakdown for the generator, engine run, and analysis pass.

## Determinism

All RNG is `Xoshiro256StarStar`. Given `(connectome_seed, engine_seed)`, the *connectome itself* is bit-identical across runs and machines (`ConnectomeConfig::default()` fixes both). The three LIF paths (`use_optimized: false` baseline heap+AoS; `use_optimized: true` wheel+SoA; `simd` feature layering `wide::f32x8` on top of the optimized path) produce spike *counts* within ≈10% of each other but not bit-identical spike traces — the timing-wheel groups events within a tick differently from the binary heap, which is a realistic engineering tradeoff and is documented in ADR-154 §4.2 and §15. Bit-exact determinism *within* a path is guaranteed and verified by `ac_1_repeatability`. Bit-exact determinism *between* paths is a future-work goal captured in `docs/research/connectome-ruvector/03-neural-dynamics.md` §11 and ADR-154 §15.1.

## What this example is *not*

- **Not a FlyWire import.** The synthetic SBM matches summary statistics but no individual-edge fidelity.
- **Not embodied.** Stimulus is a deterministic current schedule; there is no MuJoCo or NeuroMechFly wiring.
- **Not calibrated for a specific behavior.** The demo shows spike dynamics, a partition, and motif retrieval — not grooming, feeding, or any named fly behavior. Reproducing named behaviors is the M2/M3 gate of the full production plan and is deferred.
- **Not a performance claim against GPU simulators.** `ruvector-lif` is a CPU-first event-driven kernel whose differentiator is determinism, graph integration, and analysis wiring — not raw throughput against GeNN or NEST.

## Pointers for extension

- **Port to `ruvector-lif`.** The production kernel planned in `docs/research/connectome-ruvector/03-neural-dynamics.md` will subsume this LIF with a hierarchical timing wheel, slow pools for neuromodulators, per-region parallelism, and explicit EdgeMask support. The data structures in `src/lif.rs` are a reference ABI for that port.
- **Swap kNN for DiskANN.** `MotifIndex` is a bounded brute-force kNN for demo scale. The production path (see ADR-144 / ADR-146) uses DiskANN Vamana with 4/8-bit quantization against the pi-brain (ADR-150) substrate.
- **Wire `ruvector-sparsifier`.** The current analysis calls `MinCutBuilder` directly on the full coactivation graph because it fits in memory at N = 1024. At 10k–139k neurons the pipeline should sparsify first, as `05-analysis-layer.md` §3 prescribes.
- **Embodiment.** See `docs/research/connectome-ruvector/04-embodiment.md` for the NeuroMechFly / MuJoCo MJX plan and the sensor/motor ABI.

## References

The binding references are the ADR and the nine research documents:

- `docs/adr/ADR-154-connectome-embodied-brain-example.md`
- `docs/research/connectome-ruvector/README.md`
- `docs/research/connectome-ruvector/00-master-plan.md` through `08-implementation-plan.md`

Scientific anchor: Lin *et al.*, 2024, *Nature* — whole-fly-brain LIF model derived from the connectome.
