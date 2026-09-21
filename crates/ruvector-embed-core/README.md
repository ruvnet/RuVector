# ruvector-embed-core

Sentence-embedding backends for `@ruvector/typesafe`: two implementations of
the [`Embedder`](../ruvector-typesafe-core/src/embedder.rs) trait, one per
target (ADR-002).

| Feature | Backend | Target | Engine |
|---|---|---|---|
| `native` | `OrtEmbedder` | native CPU | `ort` 2.0.0-rc.13 (onnxruntime) |
| `wasm` | `TractEmbedder` | wasm32 + native | `tract-onnx` 0.23 (pure Rust) |
| *(default)* | — | any | none — no engine, no download |

`cargo build --workspace` (no features) compiles **no** inference engine and
downloads nothing: only the manifest, pooling and error types are present.

## Behaviour

- Weights are pinned by a content-hash [model manifest](../../npm/packages/typesafe/models/manifest.json)
  (`{name, file, sha256, dims, license, source_url, added, review_by, pooling,
  tokenizer_file, tokenizer_sha256, max_tokens}`) and verified at load. A
  mismatch is a typed `EmbedError::HashMismatch` and **nothing is loaded** —
  fail closed (ADR-005). The optional `tokenizer_sha256` closes the
  "pinned model, unpinned tokenizer" gap; it is populated for all shipped models.
- **No network at runtime.** Native reads a local directory; wasm is handed the
  bytes. Model download is a dev-time script, never the library.
- Every embedding is mean- or CLS-pooled (per the manifest — bge uses CLS,
  MiniLM uses mean) and L2-normalised, so cosine similarity is a dot product.
- Inputs are truncated by the tokenizer at `max_tokens` (256), preserving
  `[CLS]`/`[SEP]`. Longer `state` is rejected upstream at 16 KB (ADR-005), so
  this is a backstop, not the primary limit.
- `Embedder: Send + Sync`. `ort`'s `Session::run` is `&mut self`, so the native
  session lives behind a `Mutex`.

## Usage

```rust
use ruvector_embed_core::{ManifestFile, OrtEmbedder, Embedder};

let mf = ManifestFile::from_json(&std::fs::read_to_string("models/manifest.json")?)?;
let m = mf.get("bge-small-en-v1.5").unwrap();
let emb = OrtEmbedder::from_manifest("models", m)?;   // verifies sha256, fails closed
let vecs = emb.embed(&["ticket: laptop won't boot", "invoice paid late"])?; // L2-normalised
assert_eq!(emb.dims(), 384);
assert!(emb.id().starts_with("bge-small-en-v1.5@")); // model@sha256-prefix, for receipts
```

WASM has no filesystem, so `TractEmbedder::from_bytes(model_bytes, tokenizer_bytes, &m)`
takes the bytes directly (hashes verified before parse).

## Models

Fetch + verify with the zero-dependency script (Node ≥ 20):

```bash
node npm/packages/typesafe/scripts/fetch-models.mjs        # download + verify
node npm/packages/typesafe/scripts/fetch-models.mjs --check # verify only
```

Weights live in `models/<name>/` and are **git-ignored**; the manifest is
committed. Shipped: `bge-small-en-v1.5` (MIT, CLS) and `all-MiniLM-L6-v2`
(Apache-2.0, mean), each FP32 and INT8, 384-d, from the Xenova ONNX exports.

## The spike (ADR-002 §3, §7)

Single-thread CPU, 100 fixture texts from `bench/fixtures/tickets-corpus.json`,
ms per embed (cache warmed first, so timing is inference not model build). Full
numbers in [`bench/embedder-spike-2026-09-21.json`](../../npm/packages/typesafe/bench/embedder-spike-2026-09-21.json).

ms per embed (lower is better), one thread, x86_64 (2026-09-21):

| Model | Prec | ort b1 | ort b32 | tract b1 | tract b32 |
|---|---|---|---|---|---|
| bge-small-en-v1.5 | FP32 | 6.49 | 5.81 | 6.06 | 7.76 |
| bge-small-en-v1.5 | INT8 | 2.42 | 2.59 | *load fails* | — |
| all-MiniLM-L6-v2 | FP32 | 14.00 | 14.20 | 14.83 | 15.10 |
| all-MiniLM-L6-v2 | INT8 | 6.36 | 7.45 | *load fails* | — |

ort INT8 is ~2.2–2.7× faster than FP32, matching the ADR-002 expectation. `tract`
runs rows one at a time, so its "b32" is not a throughput win; the FP32 numbers
track ort closely. Plan caches are warmed before timing, so the numbers are
inference, not model build.

**INT8 on `tract` 0.23: still fails — INT8 is native-only.** Both quantized
graphs fail at `into_optimized()` with
`Failed analyse for node "/Unsqueeze" AddDims` — the *same* failure ADR-194 §5
recorded on tract 0.21. The 0.23 bump does not fix it. WASM therefore ships
**FP32**; INT8 is available only through the native `ort` backend (ADR-002 §3).

**Parity (ADR-002 §7): pass.** Min cosine(ort, tract) over 50 FP32 texts is
0.999999 for bge-small and 0.999999 for MiniLM, both ≥ 0.9999.

Run it:

```bash
cargo run -p ruvector-embed-core --features native,wasm --example spike --release
```
