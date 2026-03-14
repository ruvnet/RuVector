# ADR-005: Cross-Platform Bindings (WASM + NAPI-RS Strategy)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-13 |
| **Authors** | RuVector Architecture Team |
| **Reviewers** | Architecture Review Board |
| **Supersedes** | - |
| **Related** | ADR-001-core-simd-strategy, ADR-004-rvf-format |

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
| Node.js | Server-side JS | Backend services |

### 1.2 Binding Technology Comparison

| Technology | Platforms | Performance | Size | Maintenance |
|------------|-----------|-------------|------|-------------|
| Native binaries | Per-platform | Excellent | Large | High |
| FFI/C API | Most | Excellent | Small | Medium |
| **WASM** | Universal | Good-Excellent | Small | Low |
| **NAPI-RS** | Node.js | Near-native | Medium | Low |
| wasm-bindgen | Browser | Good | Small | Low |
| PyO3 | Python | Near-native | Medium | Medium |

### 1.3 Why Dual Strategy?

Single technology cannot optimally serve all use cases:

- **WASM**: Universal but slower for compute-heavy ops
- **NAPI-RS**: Native speed for Node.js but platform-specific

Our solution: **WASM-first design with native acceleration where it matters**.

## 2. Decision

### 2.1 Dual Binding Architecture

```
                        RuVector Core (Rust)
                              |
            +-----------------+-----------------+
            |                                   |
    +---------------+                   +---------------+
    | WASM Bindings |                   | Native Bindings|
    | (wasm-bindgen)|                   | (NAPI-RS)      |
    +---------------+                   +---------------+
            |                                   |
    +-------+-------+                   +-------+-------+
    |       |       |                   |               |
  Browser  WASI  Embedded            Node.js        Deno
  (JS)   (CLI)   (IoT)             (Backend)     (Runtime)
```

### 2.2 Platform Matrix

| Platform | Primary Binding | Fallback | Notes |
|----------|-----------------|----------|-------|
| Browser | WASM (wasm-bindgen) | - | SIMD128 for perf |
| Node.js | NAPI-RS native | WASM | Native preferred |
| Deno | NAPI-RS/FFI | WASM | Use FFI plugin |
| Cloudflare Workers | WASM | - | Only WASM allowed |
| AWS Lambda | NAPI-RS | WASM | Native for cold start |
| CLI | Native Rust | WASM | Direct execution |
| iOS | WASM (via Safari) | - | WKWebView integration |
| Android | WASM | - | WebView integration |
| ESP32/Embedded | WASM (wasm3) | - | 64KB interpreter |

### 2.3 Module Structure

```
ruvector/
  crates/
    ruvector-core/          # Pure Rust, no dependencies
    ruvector-wasm/          # WASM bindings (wasm-bindgen)
    ruvector-node/          # Node.js bindings (NAPI-RS)
    ruvector-wasm-simd/     # WASM with SIMD128 feature
  npm/
    packages/
      @ruvector/core/       # WASM package (universal)
      @ruvector/node/       # NAPI native package
      ruvector/             # Main package (auto-selects)
```

## 3. WASM Implementation

### 3.1 Build Configuration

```toml
# Cargo.toml for ruvector-wasm
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
ruvector-core = { path = "../ruvector-core", default-features = false }
wasm-bindgen = "0.2"
js-sys = "0.3"
web-sys = { version = "0.3", features = ["console"] }
wee_alloc = { version = "0.4", optional = true }

[features]
default = ["wasm"]
wasm = ["wee_alloc"]
simd = []  # Enables SIMD128

[target.wasm32-unknown-unknown]
rustflags = [
    "-C", "link-arg=-zstack-size=65536",
]

[target.wasm32-wasi]
rustflags = [
    "-C", "link-arg=-zstack-size=131072",
]
```

### 3.2 WASM Bindings (wasm-bindgen)

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

    #[wasm_bindgen]
    pub fn delete(&mut self, id: &str) -> Result<bool, JsError> {
        self.inner.delete(id)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(getter)]
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}
```

### 3.3 WASM SIMD128 Support

```rust
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use std::arch::wasm32::*;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub unsafe fn cosine_similarity_wasm_simd(a: &[f32], b: &[f32]) -> f32 {
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

## 4. NAPI-RS Implementation

### 4.1 Build Configuration

```toml
# Cargo.toml for ruvector-node
[lib]
crate-type = ["cdylib"]

[dependencies]
ruvector-core = { path = "../ruvector-core" }
napi = { version = "2", default-features = false, features = ["async", "napi8"] }
napi-derive = "2"

[build-dependencies]
napi-build = "2"

[profile.release]
lto = true
codegen-units = 1
```

### 4.2 NAPI-RS Bindings

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
            .map(|r| SearchResult { id: r.id, score: r.score as f64 })
            .collect())
    }

    #[napi]
    pub fn delete(&mut self, id: String) -> Result<bool> {
        self.inner.delete(&id)
            .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
    }

    #[napi(getter)]
    pub fn count(&self) -> u32 {
        self.inner.len() as u32
    }

    #[napi(getter)]
    pub fn dimension(&self) -> u32 {
        self.inner.dimension() as u32
    }

    #[napi]
    pub async fn batch_insert(
        &mut self,
        ids: Vec<String>,
        vectors: Vec<Float32Array>,
    ) -> Result<()> {
        // Async batch insertion for large datasets
        for (id, vec) in ids.into_iter().zip(vectors.into_iter()) {
            self.inner.insert(id, vec.to_vec())
                .map_err(|e| Error::new(Status::GenericFailure, e.to_string()))?;
        }
        Ok(())
    }
}

#[napi(object)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
}

#[napi(object)]
pub struct Config {
    pub m: Option<u32>,
    pub ef_construction: Option<u32>,
    pub ef_search: Option<u32>,
}
```

### 4.3 Platform-Specific Builds

```yaml
# .github/workflows/build-native.yml
jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
            node: linux-x64-gnu
          - os: ubuntu-latest
            target: aarch64-unknown-linux-gnu
            node: linux-arm64-gnu
          - os: macos-latest
            target: x86_64-apple-darwin
            node: darwin-x64
          - os: macos-latest
            target: aarch64-apple-darwin
            node: darwin-arm64
          - os: windows-latest
            target: x86_64-pc-windows-msvc
            node: win32-x64-msvc

    steps:
      - uses: actions/checkout@v4
      - uses: actions-rs/toolchain@v1
        with:
          target: ${{ matrix.target }}
      - run: npm run build -- --target ${{ matrix.target }}
      - uses: actions/upload-artifact@v4
        with:
          name: bindings-${{ matrix.node }}
          path: npm/packages/ruvector-node/*.node
```

## 5. NPM Package Structure

### 5.1 Main Package (Auto-Selection)

```javascript
// npm/packages/ruvector/index.js

let binding;

// Try native bindings first
try {
  binding = require('@ruvector/node');
  binding._implementation = 'native';
} catch (e) {
  // Fall back to WASM
  binding = require('@ruvector/core');
  binding._implementation = 'wasm';
}

module.exports = binding;
```

### 5.2 Package.json

```json
{
  "name": "ruvector",
  "version": "2.0.0",
  "description": "High-performance vector database with HNSW indexing",
  "main": "index.js",
  "types": "index.d.ts",
  "scripts": {
    "postinstall": "node scripts/select-binding.js"
  },
  "optionalDependencies": {
    "@ruvector/node-linux-x64-gnu": "2.0.0",
    "@ruvector/node-darwin-arm64": "2.0.0",
    "@ruvector/node-darwin-x64": "2.0.0",
    "@ruvector/node-win32-x64-msvc": "2.0.0"
  },
  "dependencies": {
    "@ruvector/core": "2.0.0"
  }
}
```

### 5.3 TypeScript Definitions

```typescript
// index.d.ts
export class VectorDB {
  constructor(dimension: number);

  insert(id: string, vector: Float32Array): void;
  search(query: Float32Array, k: number): SearchResult[];
  delete(id: string): boolean;

  readonly count: number;
  readonly dimension: number;

  batchInsert(ids: string[], vectors: Float32Array[]): Promise<void>;
}

export interface SearchResult {
  id: string;
  score: number;
}

export interface Config {
  m?: number;
  efConstruction?: number;
  efSearch?: number;
}

export const _implementation: 'native' | 'wasm';
```

## 6. Performance Comparison

### 6.1 Benchmark Results

| Operation | Native (NAPI) | WASM | WASM+SIMD | Ratio |
|-----------|---------------|------|-----------|-------|
| insert (1K) | 8.2ms | 12.5ms | 10.1ms | 1.5x/1.2x |
| search (10K) | 0.52ms | 0.89ms | 0.61ms | 1.7x/1.2x |
| cosine (384d) | 0.14us | 0.22us | 0.16us | 1.6x/1.1x |
| batch (100K) | 820ms | 1350ms | 980ms | 1.6x/1.2x |

### 6.2 Bundle Size

| Package | Size (gzipped) |
|---------|----------------|
| @ruvector/core (WASM) | 185 KB |
| @ruvector/node (native) | 1.2 MB |
| ruvector (main) | 3 KB + deps |

## 7. Consequences

### 7.1 Benefits

1. **Universal Coverage**: Runs everywhere from browser to edge
2. **Optimal Performance**: Native where possible, WASM as fallback
3. **Small Bundles**: WASM only 185 KB for full functionality
4. **Type Safety**: Full TypeScript support
5. **Zero Config**: Auto-selects best binding

### 7.2 Costs

1. **Build Complexity**: Multiple targets to maintain
2. **CI/CD Load**: 5+ platform builds per release
3. **Testing Matrix**: Test all platforms independently
4. **Version Sync**: Keep all packages in sync

### 7.3 Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| WASM-only | Too slow for server workloads |
| Native-only | No browser/edge support |
| Node-API (C++) | More complexity than NAPI-RS |
| FFI-NAPI | Manual memory management |
| Electron | Overkill for embedding |

## 8. Related Decisions

- **ADR-001-core-simd-strategy**: SIMD for native and WASM SIMD128
- **ADR-004-rvf-format**: WASM kernel embedded in RVF files
- **ADR-003-mcp-protocol**: MCP server available as WASM

## 9. References

1. wasm-bindgen Guide: https://rustwasm.github.io/wasm-bindgen/
2. NAPI-RS Documentation: https://napi.rs/
3. WASM SIMD Proposal: https://github.com/WebAssembly/simd
4. Implementation: `/crates/ruvector-wasm/`, `/crates/ruvector-node/`

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-13 | Architecture Team | Initial decision record |
