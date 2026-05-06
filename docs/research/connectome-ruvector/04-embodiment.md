# 04 - Embodiment: Body Simulator and Sensorimotor Loop

> Framing reminder: this is a graph-native embodied connectome runtime. The body simulator is a tool for closing the sensorimotor loop, not a stand-in for a living organism. See `./00-master-plan.md` §1 and `./07-positioning.md`.

## 1. Purpose

Pick the body simulator, define the Rust bridge, specify the motor-neuron → joint-torque contract, and define the sensory encoder that pushes current into the LIF kernel. Consumers: `./03-neural-dynamics.md` (which emits motor spikes and consumes sensory current), `./05-analysis-layer.md` (which observes behavioral states), and `./08-implementation-plan.md`.

The scientific anchor is the Eon / NeuroMechFly line of work (see `./06-prior-art.md` §NeuroMechFly) which demonstrated that a connectome-derived LIF model, coupled to a realistic articulated insect body with visual, mechanosensory, and chemosensory inputs, produces recognizable feeding and grooming behaviors. The 2024 Nature whole-fly-brain paper is the upstream evidence that connectome-constrained dynamics alone are expressive; embodiment is what turns those dynamics into behavior.

## 2. Requirements

Hard requirements on the embodiment layer:

1. **Articulated insect body** — Drosophila-realistic kinematics: 6 legs × ~7 DoF, neck, head, proboscis, wings, halteres. NeuroMechFly v2 is the reference model.
2. **Contact dynamics** — feet-ground, proboscis-substrate, leg-leg during grooming.
3. **Proprioception** — joint angles and velocities at ~1 kHz.
4. **Compound-eye vision** — per-ommatidium luminance and motion, at a configurable resolution.
5. **Chemosensation** (optional v1) — antennal and tarsal chemoreceptor signals.
6. **Deterministic** — seeded simulation runs reproduce exactly.
7. **Rust-callable** — the bridge must speak Rust, even if the sim is C++/CUDA internally.
8. **Closed-loop latency budget** — sim step + bridge RTT + LIF step + decode ≤ 40 ms (25 Hz control rate). We will relax to 100 ms if necessary for v1.
9. **No Python in the runtime path.** Offline configuration (generating MJCF) is allowed outside `crates/`.

## 3. Candidate simulators

### 3.1 MuJoCo (native, now Apache-2.0)

C++ physics engine with first-class contact, articulated bodies, soft constraints. NeuroMechFly v2 ships MJCF for Drosophila directly. MuJoCo 3.x has good determinism and is fast on CPU. Python bindings exist but the core is C. Rust bindings exist via `mujoco-rs` crates, and `cxx` / `bindgen` wrap the C API cleanly.

Pros:
- NeuroMechFly-ready.
- Deterministic per seed.
- CPU-real-time feasible at our scale.
- Apache-2.0, clean Rust FFI story.

Cons:
- Vision is not native; must be bolted on (ray-cast a coarse compound eye, or render via EGL and sample).
- No GPU batch by default (MuJoCo 3 has CPU parallelism; MJX is the GPU variant).

### 3.2 MuJoCo MJX (JAX)

GPU-accelerated MuJoCo via JAX. Same MJCF, different executor. Very fast for batched environments; we only need one.

Pros:
- Same body model.
- GPU-ready if we need it later.

Cons:
- JAX is Python. Our runtime rule is no Python.
- Single-env speed is not necessarily better than native MuJoCo; the win is batch.

### 3.3 Brax

Google's pure-JAX rigid-body simulator. Very fast batched; good API; not NeuroMechFly-native.

Pros:
- Fully differentiable; interesting for long-term calibration.
- Fast batched.

Cons:
- JAX; Python runtime dependency.
- Would require re-authoring the NeuroMechFly body for Brax. That is a nontrivial science-quality port we should not do in v1.

### 3.4 Isaac Gym / Isaac Sim

NVIDIA RL-focused sim. High fidelity, GPU-lock.

Pros:
- GPU throughput.

Cons:
- Not friendly for single-agent science runs.
- Licensing friction.
- Python-first.
- No NeuroMechFly port.

Excluded for v1.

### 3.5 Decision

**Primary: native MuJoCo 3 with the NeuroMechFly v2 MJCF, wrapped by a new Rust crate `ruvector-embodiment` using `cxx`.** Rationale: NeuroMechFly-native, deterministic, CPU-real-time feasible, Apache-2.0, clean Rust FFI. Vision gets a custom coarse-compound-eye ray-cast module on top.

**Fallback: MuJoCo MJX, via a local sidecar process speaking `bincode` over UDS.** If GPU proves necessary. Breaks the Rust-only-runtime rule only by running a sidecar, not by importing Python into our crates.

**Excluded: Brax, Isaac.** For clear reasons above.

## 4. `ruvector-embodiment` crate

```
crates/ruvector-embodiment/
├── Cargo.toml
├── build.rs           # links libmujoco, regenerates cxx bindings
├── include/           # vendored mujoco headers (minimal subset)
├── src/
│   ├── lib.rs
│   ├── mjcf.rs        # MJCF loader
│   ├── world.rs       # World wraps mjModel + mjData
│   ├── contact.rs     # typed contact forces
│   ├── vision.rs      # coarse compound-eye ray-cast
│   ├── proprio.rs     # joint/velocity extraction
│   ├── chemo.rs       # antennal chemoreceptor stub
│   ├── motor.rs       # torque application
│   ├── sensor_schema.rs
│   └── motor_schema.rs
├── tests/
│   ├── neuromechfly_load.rs
│   └── closed_loop_sanity.rs
└── examples/
    └── walking_on_flat.rs
```

Crate depends on `ruvector-connectome` for the `NeuronId` type (so flagged motor/sensory lists resolve) and publishes the `BodySim` trait defined in `./01-architecture.md` §5.

## 5. Motor-neuron → joint-torque contract

Inputs: a `SpikeStream` of `(neuron_id, time_ms)` events from `./03-neural-dynamics.md`, filtered to neurons flagged `NeuronFlags::Motor`.

```rust
pub struct Torque { pub joint: JointId, pub value_nm: f32 }
pub struct MotorSchema {
    /// For each Drosophila joint we drive, the motor neurons
    /// (left/right pools) that contribute and their gain.
    pub map: HashMap<JointId, MotorGroup>,
}
pub struct MotorGroup {
    pub agonists:   Vec<(NeuronId, f32)>,
    pub antagonists: Vec<(NeuronId, f32)>,
    pub max_nm: f32,
}
```

v1 decoding rule (rate-coded):

```
for each JointId j in schema:
    f_ag = spike_count(agonists[j], window=10ms) / window
    f_an = spike_count(antagonists[j], window=10ms) / window
    raw  = sum_i gain_i · (f_ag_i - f_an_i)
    T_j  = clamp(raw, -max_nm, +max_nm)
emit Torque{j, T_j}
```

Windows of 10 ms at a 25 Hz control rate mean every control tick integrates the last 2-3 ticks of motor firing. Alternative (v2) is PID on target angle; alternative (v3) is learned linear decoder from spike windows to torque. v1 stays rate-coded because it is interpretable, cheap, and consistent with the "no synthetic training" rule.

The motor schema is a configuration file, not code. It lives in `configs/embodiment/neuromechfly_v2_motor.toml` and carries its own version hash that becomes part of the replay manifest.

## 6. Sensory → spike-injection contract

### 6.1 Vision

Compound-eye model: two eyes × N ommatidia per eye. A coarse v1 uses N ≈ 256-512 per eye (real fly is ~780). Each ommatidium ray-casts through MuJoCo's geometry, samples luminance + motion energy, and maps to a photoreceptor neuron ID pool in FlyWire (R1-R6 analogs). The encoder:

```rust
pub struct VisionFrame { pub left: Vec<f32>, pub right: Vec<f32> }

pub fn encode_vision(frame: &VisionFrame, schema: &VisionSchema, now: Time) -> Vec<SensoryInj> {
    let mut out = Vec::new();
    for (i, &l) in frame.left.iter().enumerate() {
        let nid = schema.left_pr[i];
        let current = linear_encode(l); // pA
        out.push(SensoryInj{ post: nid, current_pa: current, t_ms: now });
    }
    // same for right
    out
}
```

Linear encoding: `current_pa = offset + gain · luminance`. Calibration in M3. Motion-energy channel optional for v1.

### 6.2 Proprioception

Chordotonal neurons (`NeuronFlags::Chordotonal`) receive a current proportional to joint position and velocity. One neuron per degree of freedom per sign (flexor/extensor) is the minimum; FlyWire provides hundreds of relevant sensory neurons and we use the first-pass assignment from cell-type labels plus hemilineage.

### 6.3 Mechanosensation

Contact forces from the sim (feet, antennal mechanoreceptors) are rate-encoded onto mechanosensory neurons.

### 6.4 Chemosensation (optional v1)

Antennal and tarsal chemoreceptors receive a current proportional to scalar fields in MuJoCo's custom volume. Default: zero field, feature off. Flip on to drive feeding-behavior demos.

## 7. Closed-loop operation

```
let mut world = World::new("configs/embodiment/neuromechfly_v2.mjcf")?;
let mut engine = Engine::new(connectome, engine_cfg)?;

let motor_schema = MotorSchema::load(...)?;
let sensor_schema = SensorSchema::load(...)?;

loop {
    let obs = world.step_no_torque()?;                        // pure reading
    let injections = encode_sensory(&obs, &sensor_schema, engine.clock());
    for inj in injections { engine.inject_sensory_bulk(&injections); }
    let report = engine.run_until(engine.clock() + CONTROL_TICK_MS);
    let motor_spikes = engine.drain_motor_spikes();
    let torques = decode_motor(&motor_spikes, &motor_schema, engine.clock());
    world.step_with_torques(&torques)?;
}
```

`CONTROL_TICK_MS = 40.0` at 25 Hz. The sim may internally take several 1-2 ms physics substeps; we read observations at the control tick and apply torques held constant across substeps.

## 8. Latency budget

Target per control tick (40 ms):

| Stage | Target | Notes |
|---|---|---|
| MuJoCo substeps (3-5 @ 1 ms) | 5-10 ms | CPU-real-time feasible |
| Vision ray-cast (256 rays × 2) | 2-4 ms | Simple ray-in-MJX |
| Proprio + mechano encoding | <1 ms | trivial |
| Sensory injection into LIF | <1 ms | bulk insert into queue |
| LIF step (40 ms sim wall-clock equiv) | 10-20 ms | from `./03` §5 targets |
| Motor decode | <1 ms | rate-code with 10 ms window |
| Torque application | <1 ms | MuJoCo set |
| **Total** | **20-40 ms** | within 25 Hz budget |

If over, actions (in order of preference): drop vision resolution, drop chemosensation, bump `dt_ms` in LIF, enable per-region parallelism.

## 9. Failure modes

- **NaN in MuJoCo state** — usually contact instability or numerical blowup from torque spikes; we clamp motor output and reset on NaN with a logged episode boundary.
- **Motor-neuron list drift** — if FlyWire labels change and our motor schema references a neuron that no longer exists, we fail closed at schema-load time, not at run time.
- **Sensory silence** — if vision returns NaN/zeros (camera failure), we inject zero current and log the modality as dropped; the LIF loop continues.
- **Bridge hangs** — MuJoCo calls are bounded; we do not block the LIF taps on FFI.

## 10. Comparison table

| Sim | NeuroMechFly fit | Rust bridge effort | Runtime language | Deterministic | Vision | v1 pick? |
|---|---|---|---|---|---|---|
| MuJoCo 3 native | yes | low (`cxx`) | C++ | yes | ray-cast DIY | **yes** |
| MuJoCo MJX | yes | medium (sidecar) | Python/JAX | yes | native | fallback |
| Brax | no (port needed) | medium (sidecar) | Python/JAX | yes | DIY | no |
| Isaac Gym/Sim | no | high | Python | partial | native | no |
| Custom Rust sim | no | trivial but huge cost | Rust | yes | DIY | no (out of scope) |

## 11. What the brief says

The brief (`./README.md`) asks for "NeuroMechFly / MuJoCo / Brax body + sensory loop" with proprioception, contact, vision, and motor-neuron → joint-torque mapping. This doc fulfills that exactly: NeuroMechFly on native MuJoCo, ray-cast compound eye, proprioception via chordotonal sensory-neuron current, contact-driven mechanosensation, rate-coded motor decoding. The 2024 Nature paper's regime demonstrated the LIF side; Eon / NeuroMechFly work demonstrates the body side; this doc fuses them over a Rust bridge.

## 12. Open questions

1. **Resolution of the compound eye.** 256 / 512 / 780 per eye. Start at 256 for CPU; target 780 after M4.
2. **Whether to model halteres.** They matter for flight stabilization which is probably out of v1 scope.
3. **Chemosensation.** Off by default v1; gate behind a feature flag tied to feeding-behavior demos.
4. **MJX sidecar vs. native MuJoCo** — revisit only if native CPU cannot hit 25 Hz for the full model.
5. **Motor schema authoritativeness** — community-curated list vs. auto-generated from FlyWire labels. Start auto, allow curation overlay.

See `./03-neural-dynamics.md` for how spikes are produced, `./05-analysis-layer.md` for how the analysis layer reads the closed-loop behavior, and `./08-implementation-plan.md` for the phased build sequence including this crate.
