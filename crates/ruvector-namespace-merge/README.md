# ruvector-namespace-merge

Experimental routing for vector search across multiple namespaces. The crate
compares exhaustive search, centroid filtering, and an S-T min-cut router that
balances query relevance with inter-namespace cohesion.

```bash
cargo test -p ruvector-namespace-merge
cargo run --release -p ruvector-namespace-merge --bin benchmark
```

Routing can omit relevant namespaces. Evaluate recall against exhaustive search
on representative data before deploying a selective strategy.
