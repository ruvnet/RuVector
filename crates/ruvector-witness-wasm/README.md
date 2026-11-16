# ruvector-witness-wasm

A minimal, self-contained wasm32-portability probe for
`ruvector-agent-memory`'s ADR-347 span-signed witness chain design
(`witness_signing.rs`): same SHA-256-chained, Ed25519-signed span scheme,
same `PerRecord` vs `BatchTail` strategies, reduced to a synthetic
deterministic record so it has no dependency beyond `rvf-types`.

See `docs/research/nightly/2026-11-16-witness-wasm-portability/README.md`
for the hypothesis, benchmark methodology, and results. Reproduce with:

```bash
cargo test -p ruvector-witness-wasm
cargo build --target wasm32-unknown-unknown --release -p ruvector-witness-wasm
cargo run --release -p ruvector-witness-wasm --example wasm_witness_bench
```
