# ruvector-residual-quant

Research implementation for ADR-362: does sheaf-Laplacian residual energy
(the "contradiction energy" of ADR-CE-001/ADR-CE-011) make a useful signal
for deciding *which* agent-memory vectors deserve more quantization bits?

Three bit-allocation policies are raced against each other at an identical,
matched byte budget (same count of high-bit / low-bit vectors, differing
only in which vectors are chosen):

- `random` — baseline.
- `residual_gated` — candidate A, the novel signal under test.
- `density_gated` — candidate B, the local-density control that ADR-353's
  open questions flagged as untested.

Ground truth is a synthetic corpus with planted "boundary" vectors (midpoints
between two cluster centers — a stand-in for a contradictory/surprising
agent memory). The benchmark measures recall@10 overall and on the
boundary subset, plus residual/density AUC at separating boundary vectors,
and a hash-chained witness log over every compression decision.

```sh
cargo test -p ruvector-residual-quant
cargo run --release -p ruvector-residual-quant --bin benchmark
```

See `docs/adr/ADR-362-residual-adaptive-quantization.md` and
`docs/research/nightly/2026-11-25-residual-adaptive-quantization/` for the
pre-registered hypothesis, acceptance gates, prior-art summary (ADR-305,
ADR-353), and measured results.
