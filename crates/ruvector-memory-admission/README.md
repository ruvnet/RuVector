# ruvector-memory-admission

Global-min-cut gated write-time cluster admission for streaming agent memory.

`ruvector-namespace-merge` (ADR-299) uses S-T max-flow/min-cut to answer a
**read-time** question: given a query and a fixed namespace set, which
namespaces should be searched? This crate answers the dual **write-time**
question, which has no natural source/sink to fix the flow terminals on:
given a stream of incoming agent-memory vectors and a growing set of
clusters, should the next vector merge into an existing cluster, or does it
belong to a new one?

A fixed cosine-to-nearest-centroid threshold (the obvious baseline) only
looks at one edge. The **global** minimum cut (Stoer-Wagner, no source/sink)
of the (existing centroids + candidate point) similarity graph looks at the
whole structure: it finds the single weakest link in the graph and checks
whether the candidate sits on the weak side of it.

Three policies implement `AdmissionPolicy`:

1. `NearestCentroidThreshold` — baseline: merge into the nearest centroid if
   cosine similarity clears a fixed threshold, else spawn a new cluster.
2. `MincutGatedAdmission` — candidate A: global min-cut over centroids +
   candidate, gated on the cut's average crossing-edge weight against a
   fixed `tau`.
3. `AdaptiveMincutAdmission` — candidate B: same mechanism, but `tau` is set
   online from a running mean/std of previously observed cut weights
   (Welford's algorithm) instead of a hand-tuned constant.

Run the benchmark:

```bash
cargo run --release -p ruvector-memory-admission --bin benchmark
```

See `docs/research/nightly/2026-09-02-mincut-streaming-memory-admission/README.md`
for the full hypothesis, methodology, and measured results, and
`docs/adr/ADR-344-mincut-gated-streaming-memory-admission.md` for the
architecture decision record.
