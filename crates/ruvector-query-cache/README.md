# ruvector-query-cache

Experimental query-result caching for vector search. The crate compares an
uncached brute-force baseline, exact-query caching, and cosine-based semantic
caching with bounded capacity.

```bash
cargo test -p ruvector-query-cache
cargo run --release -p ruvector-query-cache --bin benchmark
```

Semantic hits deliberately trade exact result fidelity for latency. Select a
threshold using workload-specific recall measurements before production use.
