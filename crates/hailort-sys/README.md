# hailort-sys

Rust FFI bindings to Hailo's HailoRT C library. Generated via
bindgen at build time; no hand-curated wrappers — `ruvector-hailo`
adds the safe Rust surface on top.

> **Status:** -sys crate, **0 tests** (intentional — bindgen output
> is verified by the consumer crate's tests at the safe-API layer).
> Tracks HailoRT 4.23 (the Pi 5 + AI HAT+ runtime).

## Build requirements

- HailoRT C library installed (`libhailort.so` + headers) at
  `/usr/lib/aarch64-linux-gnu/libhailort.so` and `/usr/include/hailo/`.
  Standard package on Hailo-supported boards; on x86 dev hosts
  `cargo check` works without it because `ruvector-hailo`'s default
  feature set excludes `hailo`.
- `bindgen 0.71+` build dependency (pulls libclang).

## What's exposed

The full HailoRT C API surface, including:

- `hailo_create_vdevice` / `hailo_release_vdevice` — virtual device
  lifecycle.
- `hailo_create_hef_file` / `hailo_release_hef` — HEF artifact load.
- `hailo_init_configure_params_by_vdevice` /
  `hailo_configure_vdevice` — bind a HEF to a vdevice.
- `hailo_make_input_vstream_params` /
  `hailo_make_output_vstream_params` —
  build vstream params with `HAILO_FORMAT_TYPE_FLOAT32` so HailoRT
  handles the quant/dequant for us.
- `hailo_create_input_vstreams` / `hailo_create_output_vstreams` —
  open the I/O ring buffers.
- `hailo_vstream_write_raw_buffer` / `hailo_vstream_read_raw_buffer` —
  blocking inference forward pass.
- `hailo_get_chip_temperature` — on-die thermal sensors (used by
  worker health probes).

## Why no safe wrapper here

The -sys crate is an intentional thin layer. `ruvector-hailo`
wraps these in `HailoDevice`, `HefPipeline`, `HefEmbedder`, and
`HefEmbedderPool` types that enforce:

- `Mutex<>` serialization on per-device + per-pipeline state
- `Drop` ordering (vstreams → network group → HEF) per HailoRT
  lifetime contracts
- Magic-byte + sha256-pin validation before handing bytes to
  the SDK
- Bounded payload caps before quantize

Splitting that out keeps the FFI binding cargo-publishable
independently if the rest of the cluster needs to evolve faster
than the C API.

## Cross-compile

Cross-compile to aarch64 from x86 dev hosts via:

```bash
cargo build --release --target aarch64-unknown-linux-gnu \
    -p hailort-sys
```

Requires `gcc-aarch64-linux-gnu` and the Hailo aarch64 sysroot
(libhailort.so + headers for the target arch). The CI workflow
`hailo-backend-audit.yml` runs this build on every push to keep
the binding regression-free.

## See also

- `crates/ruvector-hailo/README.md` — the safe Rust surface that
  consumes these bindings.
- HailoRT release notes:
  https://github.com/hailo-ai/hailort/releases
- ADR-167 §5 — original FFI design rationale.
