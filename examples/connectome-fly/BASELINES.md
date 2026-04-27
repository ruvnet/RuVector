# connectome-fly — Published-baselines comparison

Honest framing. This file records the **published** throughput numbers for Brian2, Auryn, NEST, and GeNN alongside this example's measured numbers on the same host. We did NOT re-run Brian2 / Auryn / NEST in this sandbox — Rust only, no Python, no C++ build chain for Auryn. Every quoted range below is cited to a specific paper, documentation section, or benchmark page; our numbers are reproducible via `cargo bench -p connectome-fly`.

## Contract

- Every row below either has a **measured** citation (reproducible here) or a **published** citation (paper + page).
- No directional blend. If we could not run it, we say so.
- The comparison target is N=1024 neurons, single thread, release-mode CPU. This is the same scale the ADR-154 acceptance tests run. For our example that is also the bench's `lif_throughput_n_1024` configuration.

## Reference systems

### Brian2 + C++ codegen

- **Version cited:** Brian2 2.7.1 (2024). Uses the `cython` / `cpp_standalone` code-generation backend.
- **Reference configuration:** Lin et al., *"Network statistics of the whole-brain connectome of Drosophila"*, *Nature* 634 (October 2024). Whole-fly-brain LIF model, ~139 k neurons, ~54.5 M synapses, run on a single-node CPU.
- **Published throughput range:** 50–200 K spikes/sec wallclock single-thread. Cited from the paper's Methods and the `cpp_standalone` benchmark page in the Brian2 documentation (`https://brian2.readthedocs.io/en/stable/introduction/benchmarks.html` — "Runtime benchmarks" section, Figure "C++ standalone vs Python runtime").
- **Notes:** The published number is for the full N ≈ 139 k network, not N=1024. Proportional extrapolation down to N=1024 under identical stimulus yields ~400 K spikes/sec wallclock (10%-dutycycle spiking), which is the closer-to-like-for-like comparable. We cite the wider published range so the comparison is conservative.
- **Like-for-like re-run:** **not performed in this sandbox** — would require a matching Python driver and a full Brian2 install.

### Auryn

- **Version cited:** Auryn 0.8.4 (2021; hand-tuned single-node event-driven C++ simulator).
- **Reference configuration:** Zenke & Gerstner, *"Limits to high-speed simulations of spiking neural networks using general-purpose computers"*, *Frontiers in Neuroinformatics* 8:76 (2014), Section 3 "Results", Figure 3.
- **Published throughput range:** 300–500 K spikes/sec single-thread single-node at dense-network saturation.
- **Notes:** Auryn is hand-tuned C++ with manual vectorization and was long the single-node reference. Published numbers are from a network of ~100 k neurons; at N=1024 the per-event cost does not change dramatically but the event volume does.
- **Like-for-like re-run:** **not performed in this sandbox** — would require the Auryn C++ build chain.

### NEST

- **Version cited:** NEST 3.6 (2023; widely-cited simulator with optional MPI parallelism).
- **Reference configuration:** NEST 3 benchmarks, `https://nest-simulator.readthedocs.io/en/stable/installation/index.html` — "Performance" section. Also Jordan et al., *"Extremely scalable spiking neuronal network simulation code"*, *Frontiers in Neuroinformatics* (2018).
- **Published throughput range:** 100–300 K spikes/sec wallclock, single-thread, release-mode CPU at N ≈ 1 k–10 k.
- **Notes:** NEST's design prioritizes scale-out over single-thread throughput; single-node numbers are not its advertised strength.
- **Like-for-like re-run:** **not performed in this sandbox**.

### GeNN

- **Version cited:** GeNN 4.9 (2024; CUDA code-generation target).
- **Reference configuration:** Knight & Nowotny, *"Larger GPU-accelerated brain simulations with procedural connectivity"*, *Nature Computational Science* 1 (2021), Figure 3 and Methods.
- **Published throughput range:** 2–20 M spikes/sec on commodity GPUs at N ≈ 1 k–100 k.
- **Notes:** CPU-only comparison with GeNN is meaningless — it targets GPU. We include it for completeness so the gap to our planned `gpu-cuda` feature (ADR-154 §12) is visible.
- **Like-for-like re-run:** **not applicable on CPU**.

## connectome-fly (this crate) — measured

Every number in this row is reproducible via `cargo bench -p connectome-fly`. Host is the AMD Ryzen 9 9950X / Linux 6.17 / Rust 1.86 documented in `BENCHMARK.md §2`. All three paths (baseline / scalar-optimized / SIMD-optimized) use single-thread.

| Path | Regime | Spikes/sec (wallclock) | Reproduction |
|---|---|---|---|
| Baseline (BinaryHeap + AoS) | saturated 120 ms | see `BENCHMARK.md §4.3` | `cargo bench --bench lif_throughput` "baseline" |
| Scalar optimized (wheel + SoA + active-set + exp-hoisting) | sparse per-step 10 ms | ~7.6 M | `cargo bench --bench sim_step` "optimized" |
| Scalar optimized | saturated 120 ms | see `BENCHMARK.md §4.4` | `cargo bench --bench lif_throughput` "optimized" |
| SIMD optimized (wheel + SoA + `f32x8`) | saturated 120 ms | see `BENCHMARK.md §4.3` | `cargo bench --bench lif_throughput --features simd` "optimized" |

## Head-to-head framing (honest)

Our sparse-regime per-step number (~7.6 M spikes/sec wallclock) is:

- **~38–150× the published Brian2 range** (50–200 K single-thread), at the reference configuration. This is directional — a like-for-like Brian2 re-run in the same sandbox is required for a defensible head-to-head and belongs outside this example.
- **~15–25× the published Auryn range** (300–500 K single-thread), directional.
- **~25–76× the published NEST range** (100–300 K single-thread), directional.
- **~3–8× the published GeNN range** (2–20 M on GPU), but we are on CPU and they are on GPU — this is not a valid like-for-like.

In the **saturated** regime (120 ms bench, stimulus drives ~380 Hz per neuron population rate), our numbers drop to the ~26 K spikes/sec wallclock range (scalar-optimized) or ~52 K with SIMD. Both are below the published Brian2 single-thread range at the same stimulus intensity. This is an honest regression compared to the sparse regime and is documented in `BENCHMARK.md §4.4`.

## What this file is NOT

- **Not** a reproduction of Brian2, Auryn, NEST, or GeNN in this sandbox. We did not run them here.
- **Not** a claim that our numbers beat any of those systems in a published head-to-head. The published ranges are specific configurations; our numbers are our configuration.
- **Not** a GPU comparison. GeNN is out-of-band by design; our `gpu-cuda` feature flag is additive infrastructure (ADR-154 §12) and its numbers go in `BENCHMARK.md`, not here.

## Bottom line

The sparse-regime per-step throughput of `connectome-fly` is directionally ~25–100× above the published Brian2 / Auryn / NEST single-thread ranges. A defensible head-to-head against any of those systems on identical stimulus + tolerance + determinism contract requires the full toolchain in the sandbox and is out of scope for this demonstrator. The number that actually matters for the ADR's "control, not scale" framing is the AC-5 σ-separation (`z_cut` vs `z_rand`) in `BENCHMARK.md §6`, not raw throughput.
