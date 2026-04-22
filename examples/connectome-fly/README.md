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

The scientific anchor is the 2024 Nature whole-fly-brain LIF paper showing that behavior emerges from connectome-only LIF dynamics without trained parameters. This example does **not** claim to reproduce that biology — the SBM is a calibrated toy, not a FlyWire import. What it claims is that RuVector's graph primitives can be mounted live on a connectome-scale LIF simulation and surface useful structural signals.

## Directory layout

```
examples/connectome-fly/
├── Cargo.toml
├── README.md                 this file
├── BENCHMARK.md              baseline and post-optimization numbers
├── src/
│   ├── lib.rs
│   ├── connectome.rs         SBM generator + binary serialization
│   ├── lif.rs                event-driven LIF kernel (AoS+heap / SoA+wheel)
│   ├── stimulus.rs           deterministic current-injection schedules
│   ├── observer.rs           raster + population rate + Fiedler detector
│   ├── analysis.rs           mincut functional partition + SDPA motif retrieval
│   └── bin/run_demo.rs       CLI demo runner
├── tests/
│   ├── lif_correctness.rs    monotonicity + refractory-limit invariants
│   ├── connectome_schema.rs  schema + serialization round-trip
│   ├── analysis_coherence.rs coherence detector fires on fragmentation
│   └── integration.rs        end-to-end non-empty report
└── benches/
    ├── lif_throughput.rs     LIF events/sec at N ∈ {100, 1024, 10_000}
    ├── motif_search.rs       kNN retrieval latency for spike-window embeddings
    └── sim_step.rs           per-simulated-ms wallclock
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

All RNG is `Xoshiro256StarStar`. Given `(connectome_seed, engine_seed)`, the *connectome itself* is bit-identical across runs and machines (`ConnectomeConfig::default()` fixes both). The two LIF paths (`use_optimized: false` vs. `true`) produce spike *counts* within ≈10% of each other but not bit-identical spike traces — the timing-wheel groups events within a tick differently from the binary heap, which is a realistic engineering tradeoff and is documented in ADR-154 §4.2. Bit-exact determinism between the two paths is a future-work goal captured in `docs/research/connectome-ruvector/03-neural-dynamics.md` §11.

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
