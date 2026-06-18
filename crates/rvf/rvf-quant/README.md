# rvf-quant

Temperature-tiered vector quantization for RuVector Format.

## Overview

`rvf-quant` provides quantization codecs that reduce vector storage size based on access temperature:

- **f32** -- full precision for hot vectors
- **f16** -- half precision for warm vectors
- **u8** -- scalar quantization for cool vectors
- **binary** -- 1-bit quantization for cold/archive vectors
- **RaBitQ** -- randomized 1-bit codes (32x code compression) with unbiased
  distance estimation, used by the runtime's opt-in two-stage query path
  (recall@10 0.972 after exact rescore at 100k x 64-dim, Windows x64
  criterion release)
- **Automatic tiering** -- promote/demote vectors based on access patterns

## Usage

```toml
[dependencies]
rvf-quant = "0.2"
```

## Breaking Changes in 0.2.0

- `decode_sketch_seg` now returns `Result` instead of panicking on
  malformed input (crafted-file DoS hardening).

## Features

- `std` (default) -- enable `std` support
- `simd` -- enable SIMD-accelerated quantization

## License

MIT OR Apache-2.0
