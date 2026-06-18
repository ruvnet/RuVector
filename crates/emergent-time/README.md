# emergent-time

A dependency-free Rust crate for **relational/emergent time**: time defined as ordered internal change rather than an external coordinate. It contains two parts — implementations of four physics formalisms for time without an external clock, and an "Agentic Time" primitive that applies the same idea to AI agent traces.

```toml
[dependencies]
emergent-time = "2.2.4"
```

Zero runtime dependencies. 72 tests.

## Physics formalisms

Each is implemented on a self-contained numerical core (real-symmetric Jacobi eigensolver, complex spectral matrix exponentiation, von Neumann entropy) and verified by tests.

| Module | Computes | Verification |
|---|---|---|
| `wheeler_dewitt` | Bipartite constraint `Ĵ = H_C⊗I + I⊗H_R` and its kernel (timeless physical states). | Constructed kernel is a consistency check; a separate test confirms a generic clock Hamiltonian yields an empty kernel. |
| `page_wootters` | Schrödinger evolution recovered by conditioning a static entangled clock+system state on clock eigenstates. | Conditioned state matches an independently computed propagator to < 1e-8 across positive and negative times. |
| `entropic` | `τ_S = (S − S₀)/k`, internal-time rate as a function of entropy production over a β-swept Gibbs ensemble. | Clock rate checked against finite-difference `dS/dβ` of the measured Gibbs entropy. |
| `thermal` | Connes–Rovelli thermal time: modular Hamiltonian `K = −ln ρ` and modular flow `A(s) = e^{isK} A e^{-isK}`. | `K = βH + (ln Z)I` and modular-flow = rescaled physical evolution, each verified by independent recomputation. |

The numerical core uses the stable Jacobi rotation formula, a 2n×2n real-symmetric embedding for complex-Hermitian eigenvalues, and spectral (not series) matrix exponentials. `PageWootters` evolves in a cached eigenbasis (`ψ(t) = Σ_k e^{-iE_k t} c_k |E_k⟩`), which is ~53× faster than re-diagonalizing per timestep.

## Agentic Time

`agentic_time` measures internal time as arc length through a system's state manifold over six channels — belief, memory, retrieval, goal-graph, contradiction, plan. It provides:

- **Explainable ticks** — each tick carries a class, a reason string, and per-channel attribution.
- **Agentic Time Index (ATI)** — progress per unit of internal change.
- **A 7-state health classifier** — Healthy, Drifting, Stuck, NeedsReplan, Contradicting, Collapsing, NeedsHumanReview.
- **Change-point alarms** — a fixed-window `mean + kσ` detector and an adaptive Page–Hinkley detector (`adaptive` module).

## Benchmarks

`examples/emergent_time.rs` runs a multi-clock comparison (wall, step-count, token-count, agentic, and a fair rolling-window baseline) on a synthetic failing-agent trace.

`examples/real_trace_eval.rs` runs an early-warning evaluation on recorded agent traces with pre-registered thresholds, predicting a real error cascade defined independently of the agentic channels. Measured results:

| Detector | Agentic clock vs fair baseline (n=2 real traces) |
|---|---|
| Fixed-window `mean + 3σ` | 0 win / 1 tie / 1 loss |
| Adaptive Page–Hinkley | 0 win / 0 tie / 2 loss |

The agentic clock does not lead the fair baseline on these traces. Its demonstrated value is the diagnostic layer (per-channel attribution + health classifier), not early-warning lead. A larger pre-registered corpus would be required to establish a lead; the harness ships in the crate.

`examples/train_model.rs` learns a weighted channel composition on a controlled signal-plus-noise dataset (`weight_learning`), with held-out evaluation: AUC 0.759 (learned) vs 0.708 (equal-weight) vs 0.681 (best single channel), recovering near-zero weights on the pure-noise channels. The result is sealed with a reproducible FNV-1a provenance chain (`witness`) linking the committed model to the reported metrics. The hash is an integrity/provenance check, not a cryptographic commitment.

## Examples

```bash
cargo run --example emergent_time      # clocks + multi-clock comparison
cargo run --example real_trace_eval    # real-trace early-warning gate (skips with no data)
cargo run --example train_model        # learned channel weights + provenance seal
cargo bench                            # numerical-core and clock benchmarks
```

## References

Wheeler–DeWitt (DeWitt 1967); Page & Wootters, "Evolution without evolution" (1983); Giovannetti–Lloyd–Maccone (2015); Connes & Rovelli, thermal time (1994); Page–Hinkley (Page 1954, Hinkley 1970); ADWIN (Bifet–Gavaldà 2007). See `docs/adr/ADR-251-agentic-time.md` for the full design record and limitations.

## License

MIT OR Apache-2.0.
