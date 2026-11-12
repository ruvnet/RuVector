# ADR: npm demand driven Python capability portfolio

Status: Implemented, release validation required
Date: 2026-10-11

## Context

The ten highest monthly download RuVector npm packages include both a base SDK and alternate RVF runtimes. Creating ten arbitrary wrappers would misrepresent demand and duplicate functionality. npm download counts do not measure unique users. The ranking evidence records the full discovery population, excluded binary packages and exact API time window.

## Decision

Map ten npm distributions to ten Python tracks. Reuse the existing base SDK and graph package. Add core, GNN, attention, SONA, semantic router, ruvllm cache, native RVF and RVF WASM companions. Keep existing cluster and rerank packages supported. Ship eight new companion distributions, not ten redundant new names.

Extend the shared PyO3 runtime to version 0.1.3 with real Rust RVF and ruvllm KV cache bindings. Do not duplicate native extensions in each companion. Disable model inference/download features in ruvllm and enable its minimal feature set because upstream code requires Tokio. Scope ruvllm to cache primitives, SONA to inference and GNN to forward passes. Native RVF accepts trusted locally produced files only.

Run the verified upstream RVF WASM bytecode in Wasmtime without imports or WASI. Pin the host package and module checksum. Bound fuel, memory and input sizes. Keep native and WASM RVF APIs distinct because their persistence and algorithmic properties differ.

## Security and correctness

Reject nonfinite and excessively large vector coordinates before float32 native math. Bound GNN aggregate weight allocation to 64 MiB and normalize neighbor weights in float64. Bound native KV cache raw capacity to 64 MiB. RVF creation must not overwrite existing paths; close releases native file locks. Untrusted RVF parsing requires a separate hardened loader and is out of scope.

## Release sequence

First publish the native runtime after Rust tests, dependency audit, five platform wheels, source rebuild and metadata gates. Then publish the companion version after tests against that exact published runtime. Both use version changes on main, environment scoped credentials and immutable versions. Neither documentation edits nor pull requests publish. Failed gates block publication. Clean installs from PyPI across three operating systems are the final acceptance gate.

## Consequences

Users gain focused Python APIs with no Node dependency. Features absent in bindings remain explicit. The runtime stays shared but gains RVF and cache code; wheel size and dependency audit remain enforced. Two release stages avoid publishing companions whose pinned runtime is unavailable. Package Readmes and animated SVG diagrams describe tested capabilities rather than aspirational npm feature lists.
