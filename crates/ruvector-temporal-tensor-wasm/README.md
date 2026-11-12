<p align="center"><img src="https://raw.githubusercontent.com/ruvnet/ruvector/main/assets/ruvector/crates/ruvector-temporal-tensor-wasm.svg" alt="ruvector-temporal-tensor-wasm: f32 frame, wasm FFI, tiered quantizer, segment blob" width="100%"></p>

# ruvector-temporal-tensor-wasm

WebAssembly build of [ruvector-temporal-tensor](https://crates.io/crates/ruvector-temporal-tensor): tiered tensor compression exposed as a C-ABI module (`ttc_create`, `ttc_push_frame`, `ttc_flush`, `ttc_decode_segment`, `tts_*`).

Environments: wasm32-unknown-unknown (`rustup target add wasm32-unknown-unknown`). The crate is a `cdylib`, so you build it rather than depend on it.

## Quickstart

```bash
V=2.3.1
curl -sL https://crates.io/api/v1/crates/ruvector-temporal-tensor-wasm/$V/download | tar xz
cd ruvector-temporal-tensor-wasm-$V
cargo build --release --target wasm32-unknown-unknown
ls target/wasm32-unknown-unknown/release/*.wasm
```

Expected output:

```text
target/wasm32-unknown-unknown/release/ruvector_temporal_tensor_wasm.wasm
```

## Next step

Use the native Rust API in [ruvector-temporal-tensor](https://crates.io/crates/ruvector-temporal-tensor).

## Works with / Used by

Depends on `ruvector-temporal-tensor` (feature `ffi`).

---

Built by [rUv](https://ruv.io) · [ruv.io](https://ruv.io) · [Source](https://github.com/ruvnet/ruvector/tree/main/crates/ruvector-temporal-tensor-wasm)
