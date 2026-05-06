# 03 - Neural Dynamics: Event-Driven LIF Kernel in Rust

> Framing reminder: this is a graph-native embodied connectome runtime. The LIF kernel is a simulation engine, not a model of subjective experience. See `./00-master-plan.md` §1.

## 1. Purpose

Design a new Rust crate `ruvector-lif` that runs event-driven leaky integrate-and-fire dynamics over the connectome schema defined in `./02-connectome-layer.md`, with synaptic delays, a conductance-or-current model, and taps for the analysis layer (`./05-analysis-layer.md`). The kernel must plug into the `DynamicsEngine` trait defined in `./01-architecture.md` §4.

The target behavioral regime is the one the 2024 Nature whole-fly-brain paper established: LIF + FlyWire connectome reproduces feeding, grooming, and several sensorimotor transformations without synthetic training. Our engine must be at least faithful enough to that regime to reproduce those behaviors on the same connectome.

## 2. Model specification

### 2.1 Neuron

Leaky integrate-and-fire with optional conductance channels:

```
τ_m · dV/dt = -(V - V_rest) - R·(g_E(t)·(V - E_E) + g_I(t)·(V - E_I)) + I_ext(t)
if V ≥ V_thresh:
    emit spike
    V ← V_reset
    refractory for τ_refrac
```

Per-neuron parameters (typed struct):

```rust
pub struct NeuronParams {
    pub tau_m: f32,          // membrane time const, ms (default 10.0)
    pub v_rest: f32,         // mV (default -65.0)
    pub v_reset: f32,        // mV (default -70.0)
    pub v_thresh: f32,       // mV (default -50.0)
    pub r_m: f32,            // MOhm (default 10.0)
    pub tau_refrac: f32,     // ms (default 2.0)
    pub e_excitatory: f32,   // mV (default 0.0)
    pub e_inhibitory: f32,   // mV (default -80.0)
    pub tau_syn_e: f32,      // ms (default 5.0)
    pub tau_syn_i: f32,      // ms (default 10.0)
    pub noise_sigma: f32,    // mV (default 0.0; gated by run config)
}
```

Defaults are the canonical values used in Drosophila LIF literature and consistent with the 2024 Nature paper's regime. They are overridable per cell type from a config TOML.

### 2.2 Synapse

Each spike arriving at a post-synaptic neuron updates the relevant conductance trace:

```
g_E(t+) = g_E(t) + w_e · δ   (pre is excitatory)
g_I(t+) = g_I(t) + w_i · δ   (pre is inhibitory)
g_E, g_I decay with τ_syn_e, τ_syn_i between events
```

For `sign == 0` (neuromodulatory), we do **not** update fast conductances. Instead we push a delta to a per-region slow pool (see §6) that modulates `g_E`/`g_I` gains over 100-1000 ms. This keeps the event loop fast and honest about what is and isn't fast synaptic.

`weight` from `Synapse` ends up as `w_e = weight * base_gain` or `w_i = weight * base_gain`, where `base_gain` is a single global calibration knob. Calibration is the subject of M2 (see `./00-master-plan.md` §6).

## 3. Event-driven core

### 3.1 Data structures

```rust
pub enum Event {
    Spike { post: NeuronId, w: f32, sign: i8 },
    SensoryInj { post: NeuronId, current_pa: f32 },
    Checkpoint { tag: u32 },
}

pub struct ScheduledEvent { pub t_ms: f32, pub ev: Event }

pub struct Engine {
    neurons: Vec<NeuronState>,
    params:  Vec<NeuronParams>,
    csr:     CsrOutgoing,        // from ./02-connectome-layer.md §3.4
    masks:   MaskTable,          // edge masks (counterfactual)
    queue:   TimeWheel,          // hierarchical timing wheel
    clock:   f32,                // simulated ms
    taps:    Taps,               // spike/voltage broadcast channels
    slow:    SlowPools,          // neuromodulator diffusion (per region)
    cfg:     EngineConfig,
}
```

`TimeWheel` is a hierarchical timing wheel (hashed wheels at 0.1 ms, 1 ms, 10 ms, 100 ms granularity). With typical synaptic delays of 0.5-20 ms and events-per-spike averaging ~360 (median degree in FlyWire), a binary heap is workable but touches `O(log N)` per event. The timing wheel is `O(1)` amortized for insert and pop of due events and is the default. Binary-heap `BinaryEventQueue` is a config alternative for reproducibility debugging.

Tie-breaking is deterministic: `(t_ms, post_id, pre_id)` lexicographic order. This makes replay bit-exact across machines.

### 3.2 Main loop

```rust
pub fn run_until(&mut self, t_end_ms: f32) -> StepReport {
    let mut report = StepReport::default();
    while self.clock < t_end_ms {
        while let Some(ev) = self.queue.pop_due(self.clock) {
            self.dispatch(ev, &mut report);
        }
        self.advance_subthreshold_dynamics();
        self.clock += self.cfg.dt_ms;        // e.g., 0.1 ms
    }
    report
}
```

Two simulation clocks: the **event clock** advances when events fire; the **integration clock** advances by `dt_ms` for the subthreshold dynamics of neurons that did not spike. The integration step uses exponential Euler with the closed-form decay `V(t+dt) = V_rest + (V(t) - V_rest) * exp(-dt/tau_m)` between events, which is stable and accurate at `dt = 0.1 ms`.

### 3.3 Dispatch

```rust
fn dispatch(&mut self, ev: ScheduledEvent, report: &mut StepReport) {
    match ev.ev {
        Event::Spike { post, w, sign } => {
            let n = &mut self.neurons[post.idx()];
            if n.in_refractory(ev.t_ms) { return; }
            n.apply_psp(ev.t_ms, w, sign, &self.params[post.idx()]);
            if n.voltage(ev.t_ms) >= self.params[post.idx()].v_thresh {
                self.emit_spike(post, ev.t_ms);
                report.spikes += 1;
            }
        }
        Event::SensoryInj { post, current_pa } => { /* ... */ }
        Event::Checkpoint { tag } => { report.checkpoints.push(tag); }
    }
}
```

Emission schedules downstream events over the CSR row for `post`, adding `delay_ms` to the current time. Masked edges are skipped at emission; this is the seam layer 4 uses for counterfactual cuts.

## 4. Crate layout

```
crates/ruvector-lif/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── params.rs        // NeuronParams, SynapseParams, EngineConfig
│   ├── state.rs         // NeuronState, voltage arith
│   ├── csr.rs           // borrowed view of outgoing CSR from ruvector-connectome
│   ├── queue/
│   │   ├── mod.rs       // trait EventQueue
│   │   ├── wheel.rs     // hierarchical timing wheel (default)
│   │   └── heap.rs      // binary heap (deterministic fallback)
│   ├── engine.rs        // Engine, main loop, dispatch
│   ├── slow_pool.rs     // neuromodulator diffusion
│   ├── taps.rs          // SpikeStream, VoltageStream
│   ├── mask.rs          // EdgeMask application
│   ├── config.rs        // TOML/JSON config parsing
│   └── simd.rs          // optional batch voltage update
├── benches/
│   ├── queue_bench.rs
│   ├── small_ring_bench.rs
│   └── flywire_subset_bench.rs
└── tests/
    ├── integration_small.rs
    └── determinism.rs
```

Dependencies: `ruvector-connectome` (new, see `./02-connectome-layer.md`), `ruvector-nervous-system::eventbus` for lock-free taps (already specs 10K events/ms), `crossbeam` for channels, `rayon` for per-region parallelization, `smallvec`, `serde`, `bitflags`. No Python, no JAX.

## 5. Memory and throughput

Per neuron runtime state:

```rust
struct NeuronState {
    voltage: f32,
    g_e: f32,
    g_i: f32,
    last_spike_ms: f32,
    refrac_until_ms: f32,
    last_update_ms: f32,
}   // ~24 B
```

139K neurons × 24 B = **3.3 MB** for state. `params` indexed by type code ≈ negligible. Event queue at peak: high-watermark ~10M pending events × 16 B = **160 MB**. CSR (borrowed from connectome): ~1.2 GB, not owned by the engine. Total working set: well under 2 GB without taps; taps add O(fan-out × window).

Throughput targets (single-thread, modern laptop CPU):

| Workload | Target | Justification |
|---|---|---|
| 10K neuron LIF, 1 s sim | ~2-4 s wall | M2 gate |
| 100K neuron LIF, 1 s sim | ~30-60 s wall | Pre-M3 budget |
| Full 139K neuron LIF, 1 s sim | ~60-120 s wall | Embodiment loop runs at smaller scale until optimized |
| Event dispatch | ≥2M events/s | Timing wheel + cache-friendly state |

With `rayon` per-region parallelization (fly brain has ~80 neuropils with reasonable isolation between some), we expect a 4-6× wall-time reduction on an 8-core laptop. GPU (`wgpu` or CUDA) is out of scope for v1 but a natural M5+ extension if needed for real-time embodiment.

## 6. Neuromodulation

Neuromodulators (DA, 5-HT, OA) are handled out-of-band via a `SlowPool` per region:

```rust
pub struct SlowPool {
    region: RegionId,
    conc: [f32; 3],       // [DA, 5HT, OA]
    decay_ms: [f32; 3],   // per-mod decay
}
```

When a neuromodulatory spike fires, it adds to the concentration of the target region's pool (cheap). Pools decay with per-modulator τ (100-1000 ms). At integration time, `g_E` and `g_I` gains are scaled by a per-region, per-modulator function. This keeps the fast loop untouched and gives layer 4 a handle on slow state.

## 7. Dendritic coincidence (optional)

`ruvector-nervous-system::dendrite` already implements reduced-compartment NMDA-like coincidence detection. For neurons where FlyWire indicates clear dendritic compartmentalization (e.g., mushroom-body output neurons), an optional wrapper swaps the simple PSP integration for a dendritic tree. The swap is per-neuron, keyed off a schema flag, and costs one extra allocation per affected neuron. Default is off for v1.

## 8. Comparison to existing simulators

| Simulator | Language | Model | Scale | Why we don't use it |
|---|---|---|---|---|
| Brian2 | Python + cython | Equation-driven, time-stepped | ~100K neurons | Python dependency; our rule is Rust-only for the runtime |
| GeNN | C++/CUDA | Time-stepped, GPU code-gen | Millions | GPU-heavy and Python front-end; overkill for CPU-first v1 |
| NEST | C++ | Mixed event/time-stepped | Millions | Massive, MPI-focused, poor graph integration; wrong fit for tightly coupled graph analysis |
| Nengo | Python | Population-coding + LIF | ~10K-100K | Not connectome-native; Python |
| PyNN | Python | Frontend to Brian/NEST | — | Python only |
| Eon (2026) | Python + JAX | Connectome-LIF + NeuroMechFly | 139K | The thing we are differentiating from — see `./06-prior-art.md` |

`ruvector-lif` positions as **event-driven, Rust-native, connectome-first**, with first-class taps for graph analysis and a deterministic replay contract. It is not trying to beat GeNN on raw throughput; it is trying to be the right engine when the surrounding system is a dynamic graph that mincut and sparsifier are also operating on.

## 9. Taps for the analysis layer

```rust
pub struct SpikeStream(Receiver<Spike>);
pub struct VoltageStream(Receiver<VoltageSample>);

impl Engine {
    pub fn subscribe_spikes(&mut self) -> SpikeStream { /* bounded ring, drop-with-warn on full */ }
    pub fn subscribe_voltage(&mut self, ids: &[NeuronId], hz: u32) -> VoltageStream { /* ... */ }
}
```

Taps are non-blocking. Back-pressure from layer 4 is an error, not a behavior. The bound is configurable; default is 1 s of spikes at an estimated 5 kHz × 139K fan-in ≈ 700M events/s total, far beyond what taps can carry, so consumers subscribe to filters (e.g., "motor neurons only", "mushroom body only").

## 10. Edge-mask protocol

```rust
pub struct EdgeMask { pub ids: Vec<EdgeId>, pub weight_scale: f32 }
pub struct MaskHandle(u64);

impl Engine {
    pub fn apply_edge_mask(&mut self, m: &EdgeMask) -> Result<MaskHandle, EngineError>;
    pub fn remove_mask(&mut self, h: MaskHandle) -> Result<(), EngineError>;
}
```

Mask application is O(|ids|) and reflected in the next event emission. For `weight_scale = 0.0` the edge is effectively cut. This is how `./05-analysis-layer.md` runs counterfactuals. Witnesses from `ruvector-mincut::certificate` can be attached to a mask so every cut is audit-grade.

## 11. Determinism guarantees

- Same `(config, seed, input trace)` ⇒ same `(spike trace, voltage trace)` on any platform.
- Enforced by: deterministic tie-break, no unbounded parallel reductions on the fast path (per-region parallelism merges at event boundaries), explicit `Xoshiro256` seed for any noise.
- Integration tests include 10 s runs compared byte-for-byte across two build hosts.

## 12. Open design questions

1. **Wheel granularity.** Four levels (0.1/1/10/100 ms) vs five (0.05 ms included). Decide after M2 profiling.
2. **SIMD batch integration** — when a region has many non-spiking neurons, vectorize their subthreshold decay. `wide` or `packed_simd_2`. Profile-driven.
3. **Per-region parallelism** — use fly neuropil partitioning as the unit (80-ish regions). Works if inter-region connectivity is the minority of events. Verify empirically.
4. **Membrane noise** — on by default in the 2024 Nature regime or off? Off by default, on when reproducing specific behavioral variability.
5. **Graded-potential neurons** — FlyWire has non-spiking cells in the optic lobe. v1: treat as spiking with high threshold; v2: graded-potential compartment model.

## 13. Interfaces for downstream docs

- `./04-embodiment.md` consumes `Engine::drain_motor_spikes()` and calls `Engine::inject_sensory()`.
- `./05-analysis-layer.md` consumes `Engine::subscribe_*()` and uses `apply_edge_mask`/`remove_mask` for counterfactuals.
- `./08-implementation-plan.md` owns the crate creation under `crates/ruvector-lif/` and sequences the benches.

The engine is the smallest-possible thing that can honestly reproduce the 2024 Nature regime on top of `./02-connectome-layer.md`'s schema, with taps that make the RuVector analysis story possible and without locking us into GPU or Python. Anything more exotic (Hodgkin-Huxley, multi-compartment, STDP) is out of v1 scope and flagged in `./08-implementation-plan.md`.
