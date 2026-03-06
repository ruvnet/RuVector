# ADR-084: ruvllm-wasm — First Functional npm Publish

**Status**: Accepted
**Date**: 2026-03-06
**Authors**: RuVector Team
**Deciders**: ruv
**Related**: ADR-083 (Brain Training Loops), Issue #238 (placeholder deprecation)

## 1. Context

The `@ruvector/ruvllm-wasm` npm package (v0.1.0) was a placeholder — published without compiled WASM binaries. It was deprecated in PR #239. Meanwhile, the Rust crate `ruvllm-wasm` (v2.0.0) contains substantial working code:

| Subsystem | Status | Exports |
|-----------|--------|---------|
| KV Cache (two-tier FP32+u8) | Working | `KvCacheWasm`, `KvCacheConfigWasm` |
| Memory (arena + buffer pool) | Working | `InferenceArenaWasm`, `BufferPoolWasm` |
| Chat Templates (7 formats) | Working | `ChatTemplateWasm`, `ChatMessageWasm` |
| HNSW Semantic Router | Working | `HnswRouterWasm`, `PatternWasm`, `RouteResultWasm` |
| MicroLoRA (rank 1-4) | Working | `MicroLoraWasm`, `AdaptFeedbackWasm` |
| SONA Instant Learning | Working | `SonaInstantWasm`, `SonaConfigWasm` |
| Web Workers | Working | `ParallelInference`, feature detection |
| WebGPU (matmul shader) | Feature-gated | `WebGpuInference`, `WebGpuContext` |
| IntelligentLLM (combined) | Commented out | Pending API compatibility |

## 2. Decision

### 2.1 Fix WASM Build

The Rust 1.91 compiler has a codegen bug where release-profile optimizations produce invalid WASM (type mismatch: `expected i32, found f64` in wasm-bindgen post-processing). Debug builds validate fine.

**Workaround**: Build with `codegen-units=256` + `lto=off`. This prevents cross-function optimization passes that trigger the bug while still producing optimized output.

```bash
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256 \
CARGO_PROFILE_RELEASE_LTO=off \
wasm-pack build crates/ruvllm-wasm --target web --scope ruvector --release
```

Added `wasm-opt = false` to `[package.metadata.wasm-pack.profile.release]` since wasm-opt's validator also rejects the binary.

### 2.2 Gate WebGPU Features

WebGPU `web-sys` features (`gpu_map_mode`, `GpuSupportedLimits`, 28 GPU types) were compiled unconditionally, inflating binary size. Moved all GPU web-sys features behind the `webgpu` Cargo feature flag.

Removed unused `bytemuck` dependency and `gpu_map_mode` / `GpuSupportedLimits` (declared but never referenced in source).

### 2.3 Publish as v2.0.0

Published `@ruvector/ruvllm-wasm@2.0.0` to npm with:
- Compiled WASM binary (~435 KB, ~150 KB gzipped)
- TypeScript definitions (`.d.ts`)
- ES module JS glue code
- Accurate README with working API examples

### 2.4 README

Replaced placeholder README with accurate documentation covering all exported types, working code examples, and browser compatibility table.

## 3. Files Modified

| File | Changes |
|------|---------|
| `crates/ruvllm-wasm/Cargo.toml` | Gate WebGPU features, remove unused bytemuck/gpu_map_mode/GpuSupportedLimits, add wasm-opt=false |
| `crates/ruvllm-wasm/pkg/README.md` | Complete rewrite with accurate API docs |
| `crates/ruvllm-wasm/pkg/` | Generated: `.wasm`, `.js`, `.d.ts` files |

## 4. Build Artifact Details

| File | Size |
|------|------|
| `ruvllm_wasm_bg.wasm` | 435 KB |
| `ruvllm_wasm.js` | 128 KB |
| `ruvllm_wasm.d.ts` | 45 KB |

## 5. Known Limitations

| Area | Limitation | Resolution Path |
|------|-----------|-----------------|
| Rust 1.91 codegen bug | Requires `codegen-units=256` workaround | Fixed in future Rust compiler release |
| IntelligentLLMWasm | Commented out, references non-existent `HnswRouterConfigWasm` | Create config struct or pass params directly |
| WebGPU attention | CPU fallback only (matmul has GPU path) | Implement attention WGSL shader pipeline |
| Worker pool | Uses `setTimeout` polling instead of proper task completion signals | Implement message-based completion tracking |
| GGUF model loading | Not yet wired (no `load_model_from_url`) | Requires streaming fetch + parser integration |
