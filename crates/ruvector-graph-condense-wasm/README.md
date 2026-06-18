# RuVector Graph Condense — WASM

[![Crates.io](https://img.shields.io/crates/v/ruvector-graph-condense-wasm.svg)](https://crates.io/crates/ruvector-graph-condense-wasm)
[![Documentation](https://docs.rs/ruvector-graph-condense-wasm/badge.svg)](https://docs.rs/ruvector-graph-condense-wasm)
[![License](https://img.shields.io/crates/l/ruvector-graph-condense-wasm.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-ruvnet%2Fruvector-blue?logo=github)](https://github.com/ruvnet/ruvector)
[![ruv.io](https://img.shields.io/badge/ruv.io-AI%20Infrastructure-orange)](https://ruv.io)

**WASM bindings for [`ruvector-graph-condense`](https://crates.io/crates/ruvector-graph-condense).**

*Structure-preserving + differentiable-min-cut graph condensation in the browser or on the edge.*

---

Thin `wasm-bindgen` wrapper over the core condenser, built without the Rayon
`parallel` feature (wasm32 has no threads) and with the JS `getrandom` backend
gated to `cfg(target_arch = "wasm32")` so native builds are unaffected.

For the algorithm, region methods, and limitations, see the core crate's
[README](https://crates.io/crates/ruvector-graph-condense) and **ADR-196 / ADR-197**.

## Build

```bash
wasm-pack build crates/ruvector-graph-condense-wasm --target web
```

## License

MIT © [ruv.io](https://ruv.io)
