# ADR-008: WASM-First Cross-Platform Strategy

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-12 |
| **Authors** | RuVector Architecture Team |
| **Reviewers** | Architecture Review Board |
| **Supersedes** | - |
| **Related** | ADR-005 (WASM Runtime Integration), ADR-030 (Cognitive Container) |

## 1. Context

### 1.1 The Deployment Challenge

RuVector must run on diverse platforms:

| Platform | Constraints | Use Cases |
|----------|-------------|-----------|
| Cloud Servers | Full resources | Production workloads |
| Edge Devices | Limited RAM/CPU | IoT, embedded AI |
| Browsers | Sandboxed, no native | Web applications |
| Mobile | Battery, memory | Apps, offline AI |
| TEE Enclaves | Restricted, measured | Confidential compute |

### 1.2 Traditional Approaches

| Approach | Portability | Performance | Security | Maintenance |
|----------|-------------|-------------|----------|-------------|
| Native binaries per platform | Poor | Excellent | Platform-dependent | High |
| JVM/CLR | Good | Good | Sandbox | Medium |
| Interpreted (Python) | Excellent | Poor | Poor | Low |
| **WASM** | **Excellent** | **Good-Excellent** | **Excellent** | **Low** |

### 1.3 Why WASM?

WebAssembly provides:

1. **Universal bytecode**: Same binary runs everywhere
2. **Near-native performance**: 1.1-2x native speed
3. **Sandboxed execution**: Memory-safe by default
4. **No runtime dependencies**: Self-contained modules
5. **Verified execution**: Hash-pinned, reproducible

## 2. Decision

### 2.1 Adopt WASM-First Architecture

Design all portable components as WASM modules first, with native optimization as an enhancement:

```
                        RuVector Codebase
                              |
            +-----------------+-----------------+
            |                                   |
      Core Logic (Rust)                    Platform Bindings
            |                                   |
            v                                   v
    +---------------+                   +---------------+
    | wasm32-wasi   |                   | Native Targets|
    | wasm32-unknown|                   | x86_64, arm64 |
    +---------------+                   +---------------+
            |                                   |
            v                                   v
    +---------------+                   +---------------+
    | WASM Runtime  |                   | OS/Hardware   |
    | (Wasmtime,    |                   | Direct access |
    |  wasmer, V8)  |                   +---------------+
    +---------------+
```

### 2.2 Architecture Layers

```
+------------------------------------------------------------------------+
|                          APPLICATION LAYER                               |
|   Browser App | Node.js | CLI | Mobile App | Cloud Service | Edge       |
+------------------------------------------------------------------------+
                                    |
+------------------------------------------------------------------------+
|                          BINDING LAYER                                   |
|   wasm-bindgen | napi-rs | FFI | Swift/Kotlin bindings | C ABI          |
+------------------------------------------------------------------------+
                                    |
+------------------------------------------------------------------------+
|                          WASM RUNTIME LAYER                             |
|   Wasmtime (server) | wasmer | wasm3 (embedded) | V8 (browser)         |
+------------------------------------------------------------------------+
                                    |
+------------------------------------------------------------------------+
|                          CORE WASM MODULES                              |
|   ruvector-core.wasm | hnsw.wasm | distance.wasm | rvf-kernel.wasm     |
+------------------------------------------------------------------------+
```

### 2.3 Module Boundaries

| Module | Size Target | Exports | Purpose |
|--------|-------------|---------|---------|
| `ruvector-core.wasm` | <500 KB | VectorDB API | Full database |
| `hnsw.wasm` | <100 KB | Index ops | Standalone index |
| `distance.wasm` | <20 KB | Distance funcs | Compute kernels |
| `rvf-kernel.wasm` | <8 KB | Search API | RVF embedded kernel |
| `quantization.wasm` | <50 KB | Quant/dequant | Compression |

## 3. Implementation

### 3.1 Build Configuration

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib", "rlib"]

[target.wasm32-wasi]
rustflags = [
    "-C", "link-arg=-zstack-size=131072",  # 128KB stack
    "-C", "link-arg=--import-memory",
]

[target.wasm32-unknown-unknown]
rustflags = [
    "-C", "link-arg=-zstack-size=65536",   # 64KB stack (browser)
]

[features]
default = ["std"]
std = []                    # Standard library (native)
wasm = ["wee_alloc"]        # WASM-optimized allocator
simd = []                   # SIMD intrinsics (native only)
wasm-simd = []              # WASM SIMD128
```

### 3.2 Conditional Compilation

```rust
// Platform-specific distance computation
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "wasm32", feature = "wasm-simd"))]
    {
        unsafe { cosine_similarity_wasm_simd(a, b) }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        unsafe { cosine_similarity_neon(a, b) }
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { cosine_similarity_avx2(a, b) }
        } else {
            cosine_similarity_scalar(a, b)
        }
    }

    #[cfg(not(any(
        all(target_arch = "wasm32", feature = "wasm-simd"),
        all(target_arch = "aarch64", feature = "simd"),
        all(target_arch = "x86_64", feature = "simd"),
    )))]
    {
        cosine_similarity_scalar(a, b)
    }
}
```

### 3.3 WASM SIMD128 Implementation

```rust
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::*;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn cosine_similarity_wasm_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();

    let mut dot_sum = f32x4_splat(0.0);
    let mut norm_a_sum = f32x4_splat(0.0);
    let mut norm_b_sum = f32x4_splat(0.0);

    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let va = v128_load(a.as_ptr().add(idx) as *const v128);
        let vb = v128_load(b.as_ptr().add(idx) as *const v128);

        dot_sum = f32x4_add(dot_sum, f32x4_mul(va, vb));
        norm_a_sum = f32x4_add(norm_a_sum, f32x4_mul(va, va));
        norm_b_sum = f32x4_add(norm_b_sum, f32x4_mul(vb, vb));
    }

    // Horizontal sum
    let mut dot = f32x4_extract_lane::<0>(dot_sum)
        + f32x4_extract_lane::<1>(dot_sum)
        + f32x4_extract_lane::<2>(dot_sum)
        + f32x4_extract_lane::<3>(dot_sum);
    let mut norm_a = f32x4_extract_lane::<0>(norm_a_sum)
        + f32x4_extract_lane::<1>(norm_a_sum)
        + f32x4_extract_lane::<2>(norm_a_sum)
        + f32x4_extract_lane::<3>(norm_a_sum);
    let mut norm_b = f32x4_extract_lane::<0>(norm_b_sum)
        + f32x4_extract_lane::<1>(norm_b_sum)
        + f32x4_extract_lane::<2>(norm_b_sum)
        + f32x4_extract_lane::<3>(norm_b_sum);

    // Remainder
    for i in (chunks * 4)..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}
```

### 3.4 Memory Management

WASM has linear memory - we optimize with:

```rust
// Custom allocator for WASM
#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// Pre-allocate arena for batch operations
pub struct WasmArena {
    buffer: Vec<u8>,
    offset: usize,
}

impl WasmArena {
    pub fn new(size: usize) -> Self {
        Self {
            buffer: vec![0u8; size],
            offset: 0,
        }
    }

    pub fn alloc<T>(&mut self, count: usize) -> &mut [T] {
        let align = std::mem::align_of::<T>();
        let size = std::mem::size_of::<T>() * count;

        // Align offset
        self.offset = (self.offset + align - 1) & !(align - 1);

        let start = self.offset;
        self.offset += size;

        unsafe {
            std::slice::from_raw_parts_mut(
                self.buffer.as_mut_ptr().add(start) as *mut T,
                count,
            )
        }
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }
}
```

### 3.5 JavaScript Bindings (wasm-bindgen)

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct VectorDB {
    inner: ruvector_core::VectorDB,
}

#[wasm_bindgen]
impl VectorDB {
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Result<VectorDB, JsError> {
        let inner = ruvector_core::VectorDB::new(dimension, Default::default())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(VectorDB { inner })
    }

    #[wasm_bindgen]
    pub fn insert(&mut self, id: &str, vector: &[f32]) -> Result<(), JsError> {
        self.inner.insert(id.to_string(), vector.to_vec())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen]
    pub fn search(&self, query: &[f32], k: usize) -> Result<JsValue, JsError> {
        let results = self.inner.search(query, k)
            .map_err(|e| JsError::new(&e.to_string()))?;

        // Convert to JS array
        let js_results: Vec<JsValue> = results.into_iter()
            .map(|r| {
                let obj = js_sys::Object::new();
                js_sys::Reflect::set(&obj, &"id".into(), &r.id.into()).unwrap();
                js_sys::Reflect::set(&obj, &"score".into(), &r.score.into()).unwrap();
                obj.into()
            })
            .collect();

        Ok(js_sys::Array::from_iter(js_results).into())
    }
}
```

### 3.6 Node.js Bindings (napi-rs)

```rust
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub struct VectorDB {
    inner: ruvector_core::VectorDB,
}

#[napi]
impl VectorDB {
    #[napi(constructor)]
    pub fn new(dimension: u32) -> Result<Self> {
        let inner = ruvector_core::VectorDB::new(dimension as usize, Default::default())
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
        Ok(Self { inner })
    }

    #[napi]
    pub fn insert(&mut self, id: String, vector: Float32Array) -> Result<()> {
        self.inner.insert(id, vector.to_vec())
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
    }

    #[napi]
    pub fn search(&self, query: Float32Array, k: u32) -> Result<Vec<SearchResult>> {
        let results = self.inner.search(&query, k as usize)
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;

        Ok(results.into_iter()
            .map(|r| SearchResult { id: r.id, score: r.score })
            .collect())
    }
}

#[napi(object)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
}
```

## 4. Runtime Selection

### 4.1 Runtime Comparison

| Runtime | Use Case | Startup | Performance | Size |
|---------|----------|---------|-------------|------|
| Wasmtime | Servers, CLI | ~10ms | Excellent | 15 MB |
| Wasmer | General purpose | ~5ms | Excellent | 10 MB |
| wasm3 | Embedded, IoT | <1ms | Good | 64 KB |
| V8 (browser) | Web apps | Built-in | Excellent | - |
| Wazero | Go integration | ~5ms | Good | 5 MB |

### 4.2 Feature Detection

```rust
/// Detect available WASM features at runtime
pub fn detect_wasm_features() -> WasmFeatures {
    WasmFeatures {
        simd: cfg!(target_feature = "simd128"),
        threads: cfg!(target_feature = "atomics"),
        bulk_memory: cfg!(target_feature = "bulk-memory"),
        reference_types: cfg!(target_feature = "reference-types"),
        tail_call: cfg!(target_feature = "tail-call"),
        relaxed_simd: cfg!(target_feature = "relaxed-simd"),
    }
}
```

## 5. Performance Considerations

### 5.1 WASM vs Native Benchmarks

| Operation | Native (M4 Pro) | WASM (Wasmtime) | Ratio |
|-----------|-----------------|-----------------|-------|
| Cosine (384-dim) | 143ns | 185ns | 1.29x |
| HNSW search (10K) | 61us | 89us | 1.46x |
| Insert (1K batch) | 12ms | 18ms | 1.50x |
| Serialize | 45ms | 62ms | 1.38x |

### 5.2 Optimization Strategies

1. **Use SIMD128**: 2-3x speedup for vector math
2. **Batch operations**: Amortize call overhead
3. **Avoid allocations**: Use arenas, pre-allocate
4. **Minimize JS interop**: Batch results, use TypedArrays

## 6. Consequences

### 6.1 Benefits

1. **Universal deployment**: Same code runs everywhere
2. **Sandboxed security**: Memory-safe by default
3. **Small binaries**: <500 KB for full functionality
4. **Reproducible**: Hash-pinned execution
5. **Easy updates**: Replace WASM without recompilation

### 6.2 Costs

1. **Performance overhead**: 1.2-1.5x slower than native
2. **Limited SIMD**: Only 128-bit vs 256/512-bit native
3. **No threads (browser)**: SharedArrayBuffer restrictions
4. **Toolchain complexity**: Multiple build targets

### 6.3 Mitigation Strategies

| Cost | Mitigation |
|------|------------|
| Performance | Native binaries for server workloads |
| SIMD limits | Unroll loops, use WASM relaxed SIMD |
| No threads | Web Workers, async streaming |
| Build complexity | CI/CD with multi-target builds |

## 7. Related Decisions

- **ADR-005-wasm-runtime-integration**: Runtime selection details
- **ADR-030-cognitive-container**: WASM in RVF files
- **ADR-001-simd-first**: Native SIMD when available

## 8. References

1. WebAssembly Specification: https://webassembly.github.io/spec/
2. WASI: https://wasi.dev/
3. wasm-bindgen: https://rustwasm.github.io/wasm-bindgen/
4. Implementation: `/crates/ruvector-wasm/`

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-12 | Architecture Team | Initial decision record |
