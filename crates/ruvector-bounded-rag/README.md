# ruvector-bounded-rag

Research implementations of context-budgeted retrieval for RuVector.

The crate compares cosine top-k retrieval, priority traversal over a dense
similarity graph, and two Edmonds–Karp min-cut partition variants (dense
all-pairs, and LSH-sparsified — see `sparse_knn`) followed by relevance
ranking and budget truncation. The graph-based variants rebuild their
similarity structures per query and are intended as auditable research
baselines rather than production-scale indexes.

```sh
cargo test -p ruvector-bounded-rag
cargo run --release -p ruvector-bounded-rag --bin benchmark
```

See `docs/adr/ADR-272-bounded-rag-mincut.md` for the original PoC and
`docs/adr/ADR-352-sparse-knn-mincut-bounded-rag.md` for the sparse k-NN
graph follow-up — including why sparsifying graph *construction* alone
does not fix `MinCutBounded`'s scalability cliff, and what does.
