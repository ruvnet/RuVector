# ruvector-streaming-qng

Experimental product quantization for streaming vector collections under
distribution drift. It provides full-precision, static-PQ, and reservoir-based
adaptive PQ variants behind a common interface.

```bash
cargo test -p ruvector-streaming-qng
cargo run --release -p ruvector-streaming-qng --bin benchmark
```

The adaptive implementation periodically retrains its codebook and re-encodes
stored vectors. This improves drift resilience at the cost of insertion time
and retaining full-precision vectors.
