# ruvector-e-means

CPU reference implementation of [e-means (Bessa et al., 2026)](https://arxiv.org/abs/2611.09501), shared by RuVector and ruv-FANN. MIT-licensed original Rust implementation; no upstream implementation code is copied.

```rust
use ruvector_e_means::{fit, Config};
let data = [1.0f32, 2.0, 3.0, 4.0]; // two 2-dimensional rows
let config = Config { clusters: 1, ..Config::default() };
let model = fit(&data, 2, &config)?;
assert_eq!(model.centroids, [2.0, 3.0]);
# Ok::<(), ruvector_e_means::Error>(())
```

The implementation includes seeded random initialization, EMA counts/sums initialized at 1e-10, cosine scheduling, automatic batch/step scaling, weighted donor splitting every five updates, and final-epoch mean or main EMA finalization. Prediction and returned MSE use exact squared-L2 distances in original input order. All rows, including a partial final batch, are visited each epoch. Scratch memory is O(n + kd). Accumulation and distance arithmetic use f64; codebooks use f32. Extreme revival perturbations saturate to finite f32. Dead slots outnumbering available live donors wait for the next revival pass.

Assignment prunes candidates once their partial squared distance cannot beat the current best, checking every eight coordinates above 16 dimensions. Small vectors retain the scalar path. A zero-distance winner stops the search. Ordered f64 addition and lowest-index ties are preserved. Training reuses revival scratch and omits last-epoch buffers in `Ema` mode. The npm bridge returns owned typed arrays and frees its temporary Rust model after materialization.

Scaling laws were fitted on large, high-dimensional embeddings; override `batch_size` and `peak_step` for other distributions. Last-epoch means average assignments observed during the last epoch, when centroids were moving; this is not a separate Lloyd pass. Empty final clusters retain their EMA centroid. A final exact pass recomputes assignments and MSE against the returned codebook. `mse` is the mean squared L2 distance per row, not per coordinate.

This version does not implement GPU/low-precision kernels, distributed or approximate IVF assignment, or slow-EMA finalization. It makes no claim to reproduce the paper's GPU speedups or universally outperform Lloyd. Existing consumer training defaults remain available.

Native Rust has no default features. `serde` adds config/model serialization; `wasm` adds the wasm-bindgen bridge used by `@ruvector/e-means`. Native core MSRV is Rust 1.81; WASM builds use the toolchain required by the pinned wasm-bindgen dependency. See [ADR-359](../../docs/adr/ADR-359-e-means-shared-clustering.md) for integration, validation, and staged release instructions.

Run the deterministic native benchmark with `cargo run -p ruvector-e-means --release --example benchmark -- --case all --repeats 5 --seed 42 --finalization both`. Dataset generation and warmup are excluded from median fit/predict times. The [local measurements](../../docs/research/e-means/performance-2026-11-11.json) include baseline comparisons, allocation counts and unchanged-output checks; gains depend on shape and distribution.

Reusable `quantization::ProductQuantizer` and `ivf::IvfPartition` modules add
checked byte-code encode/decode, subspace-major codebooks, and stable inverted
postings. Their private shape fields and serde restoration reject malformed
persisted state. `ruvector-pq-search` and the IVF/RAIRS/SEIL family expose explicit
e-means training using this dependency. See [ADR-360](../../docs/adr/ADR-360-e-means-ecosystem-integrations.md)
for the native/npm integration matrix, MetaHarness evaluation, and Cloudflare
static-WASM entrypoints. These additions preserve existing training defaults.
