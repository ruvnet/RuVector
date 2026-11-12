# ruvector-cluster-rag

Experimental hierarchical retrieval over a two-level tree of leaf vectors and
k-means cluster summaries. It includes exhaustive, IVF-style, and
coherence-weighted search variants.

```bash
cargo test -p ruvector-cluster-rag
cargo run --release -p ruvector-cluster-rag --bin benchmark
```

Clustered variants are approximate. Tune the cluster count, probe count, and
coherence weight against workload-specific recall and latency targets.
