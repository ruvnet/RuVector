# PhotonLayer

**A deterministic optical AI front end.** A learned phase mask performs task-specific *analog*
preprocessing on incoming light, so the sensor records a small compressed measurement and a tiny
digital decoder reads the answer.

> **The camera no longer captures the image. It captures the answer shaped by physics.**
>
> Or, technically: *PhotonLayer learns what light should measure before silicon ever sees the data.*

[![Rust](https://img.shields.io/badge/Rust-pure%2C%20deterministic-orange?logo=rust)](https://www.rust-lang.org)
[![WASM](https://img.shields.io/badge/WebAssembly-ready-654ff0?logo=webassembly&logoColor=white)](#crates)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

## The wedge: auditable optical compression for task-useful sensing

This is **not** "an optical neural network." It is narrower and far more defensible:

1. **Task-first** — the mask is trained for the downstream objective, not generic image reconstruction.
2. **Compression-first** — the flagship result compresses a 16×16 input to a handful of sensor pixels.
3. **Privacy by physics** — classify/verify from an optical measurement that need not look like the scene.
4. **Deterministic receipts** — every run is reproducible and bound to BLAKE3 receipts (audit trails).
5. **Rust-native** — pure Rust across simulation, training, WASM, CLI, and (eventually) hardware control.

## Measured: compression sweep (more-data benchmark)

Synthetic 4-class task, 16×16 input (256 px), 40 samples/class, learned mask vs random mask vs a
digital tiny-sensor baseline. From `cargo test -p photonlayer-bench --release --test more_data_bench`:

| Sensor | pixels | reduction | **learned** | random | digital baseline |
|---|---:|---:|---:|---:|---:|
| 2×2 | 4 | **64×** | **0.988** | 0.738 | 0.688 |
| 3×3 | 9 | 28× | 1.000 | 0.938 | 1.000 |
| 4×4 | 16 | 16× | 1.000 | 0.950 | 1.000 |
| 1×1 | 1 | 256× | 0.250 | 0.250 | 0.250 (chance — too tight) |

At **64× pixel reduction** the learned optical mask reaches ~99% where a random mask gets ~74% — the
learned front end preserves task-useful information that random projection and naive sub-sampling lose.
(1 pixel is below the task's information floor; both collapse to chance — reported honestly.)

## Crates

| Crate | Role |
|---|---|
| `photonlayer-core` | Scalar diffraction sim — complex/FFT, field/phase-mask, Fresnel/Fraunhofer/angular-spectrum propagation, detector (shot/read noise, binning, quantization), metrics, BLAKE3 receipts |
| `photonlayer-bench` | Learned-vs-random-vs-direct baselines, decoder, privacy/verification, synthetic data |
| `photonlayer-cli` | Command-line demos |
| `photonlayer-ruvector` | ruvector integration — coherence, boundary, embeddings, experiment memory, receipts |
| `photonlayer-wasm` | Browser execution (the eventual GitHub Pages demo) |

## Quick start

```bash
cargo test -p photonlayer-core                                                  # 23 unit tests + doctest
cargo run  -p photonlayer-bench --bin photonlayer-bench --release -- all        # flagship benchmark
cargo test -p photonlayer-bench --release --test more_data_bench -- --nocapture # compression sweep
```

## Honest scope — what *not* to claim yet

Until hardware proves it, avoid: "ultra-low-power" product claims, "medical diagnostic", "autonomous-
vehicle ready", "face recognition", "unbreakable privacy", "all-optical neural network". Use instead:
**lower digital-compute potential**, **task-useful optical compression**, **privacy-reduced measurement**,
**research-grade imaging simulator**, **consented verification**. Position medical/AV uses as *research
infrastructure and preprocessing*, not decision automation. First commercial wedge: industrial &
scientific sensing.

See [`docs/research/photonlayer/ASSESSMENT.md`](../../docs/research/photonlayer/ASSESSMENT.md) for
positioning, use-case fit, the energy model + harder-dataset + reconstruction-attack roadmap, and
references. Design: ADR-260, ADR-261.

## License

MIT © Ruvector Team.
