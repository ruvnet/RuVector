# ruvector-matryoshka-rabitq

Matryoshka coarse-to-fine ANN search with a RaBitQ 1-bit brute-force coarse
stage (`RaBitQCascadeIndex`), benchmarked head-to-head against the HNSW-coarse
`TwoStageIndex` from `ruvector-matryoshka` and the single-stage `FullDimIndex`
baseline.

See `docs/research/nightly/2026-10-19_matryoshka-rabitq-cascade/README.md`
for the full nightly research write-up, SOTA survey, and measured results.

```bash
cargo run --release -p ruvector-matryoshka-rabitq --bin benchmark
```
