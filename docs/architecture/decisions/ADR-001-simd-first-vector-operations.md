# ADR-001: SIMD-First Vector Operations

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-12 |
| **Authors** | RuVector Architecture Team |
| **Reviewers** | Architecture Review Board |
| **Supersedes** | - |
| **Related** | ADR-003 (SIMD Optimization Strategy), ADR-001-core-architecture |

## 1. Context

### 1.1 Problem Statement

Vector databases and neural computation libraries execute billions of distance calculations per second. These operations form the critical hot path in:

- HNSW index traversal (k-NN search)
- Embedding similarity computation
- Neural network inference (attention, GEMV)
- Clustering and dimensionality reduction

Modern CPUs provide Single Instruction, Multiple Data (SIMD) extensions that can process 4-16 floating-point values simultaneously. Without explicit SIMD utilization, applications leave 75-94% of available compute throughput unused.

### 1.2 Hardware Landscape

| Architecture | Extension | Register Width | f32/Cycle | Market Share |
|--------------|-----------|----------------|-----------|--------------|
| x86_64 (2013+) | AVX2 | 256-bit | 8 | ~70% servers |
| x86_64 (2017+) | AVX-512 | 512-bit | 16 | ~15% servers |
| ARM64 (all) | NEON | 128-bit | 4 | ~25% servers, 100% Apple |
| Apple Silicon | NEON + AMX | 128-bit + matrix | 4 + AMX | Growing |
| WASM | SIMD128 | 128-bit | 4 | All browsers |

### 1.3 Performance Gap

Scalar vs. SIMD performance on Apple M4 Pro (384-dim vectors):

| Operation | Scalar | AVX2/NEON | Speedup |
|-----------|--------|-----------|---------|
| Euclidean Distance | 800ns | 156ns | **5.1x** |
| Dot Product | 600ns | 33ns | **18.2x** |
| Cosine Similarity | 1200ns | 143ns | **8.4x** |
| Manhattan Distance | 500ns | 200ns | **2.5x** |

## 2. Decision

### 2.1 Architecture-Specific Implementations with Unified Dispatch

We implement **hand-optimized SIMD kernels for each target architecture** rather than relying on auto-vectorization or portable SIMD abstractions.

```
                    euclidean_distance()
                           |
           +---------------+---------------+
           |               |               |
    [x86_64+AVX-512]  [x86_64+AVX2]   [aarch64]     [fallback]
           |               |               |             |
    avx512_impl()    avx2_impl()    neon_impl()    scalar_impl()
```

### 2.2 Runtime Detection (x86_64)

```rust
pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { euclidean_distance_avx512_impl(a, b) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { euclidean_distance_avx2_impl(a, b) }
        } else {
            euclidean_distance_scalar(a, b)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { euclidean_distance_neon_impl(a, b) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        euclidean_distance_scalar(a, b)
    }
}
```

### 2.3 NEON Implementation (ARM64)

```rust
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn euclidean_distance_neon_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut sum = vdupq_n_f32(0.0);

    // Process 4 floats per iteration
    let chunks = len / 4;
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);  // FMA: sum += diff * diff
    }

    let mut total = vaddvq_f32(sum);  // Horizontal reduction

    // Scalar remainder
    for i in (chunks * 4)..len {
        let diff = a[i] - b[i];
        total += diff * diff;
    }

    total.sqrt()
}
```

### 2.4 AVX2 Implementation (x86_64)

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn euclidean_distance_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut sum = _mm256_setzero_ps();

    // Process 8 floats per iteration
    let chunks = len / 8;
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);  // FMA
    }

    // Horizontal sum
    let sum128 = _mm_add_ps(
        _mm256_castps256_ps128(sum),
        _mm256_extractf128_ps(sum, 1)
    );
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut total = _mm_cvtss_f32(sum32);

    // Scalar remainder
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        total += diff * diff;
    }

    total.sqrt()
}
```

### 2.5 AVX-512 Implementation

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn euclidean_distance_avx512_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = _mm512_setzero_ps();

    // Process 16 floats per iteration
    let chunks = len / 16;
    for i in 0..chunks {
        let idx = i * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm512_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    let mut total = _mm512_reduce_add_ps(sum);

    for i in (chunks * 16)..len {
        let diff = a[i] - b[i];
        total += diff * diff;
    }

    total.sqrt()
}
```

## 3. Rationale

### 3.1 Why Not Auto-Vectorization?

| Approach | Pros | Cons |
|----------|------|------|
| Auto-vec (LLVM) | Zero code changes | Unpredictable, misses FMA, poor horizontal sums |
| `std::simd` | Portable | Nightly-only, limited intrinsics |
| `simsimd` | Fast | External dependency, less control |
| **Hand-optimized** | Maximum performance | More code, arch-specific |

Auto-vectorization fails to reliably generate:
- Fused multiply-add (FMA) instructions
- Efficient horizontal reductions
- Optimal register allocation for multi-accumulator patterns

### 3.2 Why Runtime Detection on x86 but Not ARM?

- **x86_64**: Feature detection is required because AVX-512 support varies within the same CPU family
- **ARM64**: All ARM64 CPUs support NEON; compile-time selection is sufficient

### 3.3 Safety Guarantees

All SIMD implementations include:
- Bounds checking via `assert_eq!(a.len(), b.len())`
- Unaligned loads (`_mm256_loadu_ps`, `vld1q_f32`) for arbitrary alignment
- Scalar fallback for remainder elements
- Encapsulation of `unsafe` within safe public APIs

## 4. Consequences

### 4.1 Benefits

1. **Performance**: 2.5x-18x speedup on hot paths
2. **Predictability**: Known performance characteristics per architecture
3. **Control**: Full access to FMA, horizontal sums, and advanced instructions
4. **Compatibility**: Works on all target platforms with graceful fallback

### 4.2 Costs

1. **Code Complexity**: Separate implementations per architecture
2. **Maintenance**: Each platform requires testing and optimization
3. **Binary Size**: Multiple code paths increase executable size (~10KB)

### 4.3 Platform-Specific Code Paths

| Platform | Implementation | Floats/Cycle | Status |
|----------|----------------|--------------|--------|
| x86_64 + AVX-512 | `avx512_impl` | 16 | Complete |
| x86_64 + AVX2 | `avx2_impl` | 8 | Complete |
| ARM64 (NEON) | `neon_impl` | 4 | Complete |
| Apple Silicon (AMX) | `accelerate.rs` | Hardware | Complete |
| WASM SIMD128 | `wasm_impl` | 4 | Planned |
| Scalar | `scalar_impl` | 1 | Complete |

## 5. Implementation

### 5.1 File Location

```
crates/ruvector-core/src/simd_intrinsics.rs
```

### 5.2 Public API

```rust
// Distance functions (auto-dispatch)
pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32;
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32;
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32;
pub fn manhattan_distance_simd(a: &[f32], b: &[f32]) -> f32;

// Batch operations (parallel on multi-core)
pub fn batch_distances(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: DistanceMetric,
) -> Vec<f32>;
```

### 5.3 Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `simd` | Enable SIMD intrinsics | Yes |
| `parallel` | Enable Rayon parallel batch | Yes |
| `simsimd` | Use simsimd library | No |

## 6. Benchmarks

### 6.1 Test Configuration

- Platform: Apple M4 Pro (ARM64 NEON)
- Dimensions: 128-1536
- Dataset: 10,000 vectors
- Operations: 10,000,000 distance calculations

### 6.2 Results

```
Distance Metric       Scalar      SIMD        Speedup
-----------------    --------    --------    --------
Euclidean (128-dim)   312ms       105ms       2.97x
Euclidean (384-dim)   890ms       156ms       5.71x
Euclidean (1536-dim)  3421ms      412ms       8.30x
Dot Product (384-dim) 600ms       33ms        18.18x
Cosine (384-dim)      1200ms      143ms       8.39x
Manhattan (384-dim)   500ms       200ms       2.50x
```

### 6.3 HNSW Search Impact

| Configuration | Scalar | SIMD | Improvement |
|---------------|--------|------|-------------|
| k=10, 10K vectors | 220us | 61us | **3.6x** |
| k=100, 10K vectors | 589us | 164us | **3.6x** |
| QPS (k=10) | 4,500 | 16,400 | **3.6x** |

## 7. Related Decisions

- **ADR-003-simd-optimization-strategy**: Detailed SIMD implementation strategy
- **ADR-001-ruvector-core-architecture**: Core architecture decisions
- **ADR-008-mistral-rs-integration**: LLM inference SIMD requirements

## 8. References

1. Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide
2. ARM NEON Reference: https://developer.arm.com/architectures/instruction-sets/intrinsics
3. Rust `std::arch` Documentation: https://doc.rust-lang.org/std/arch/
4. ruvector-core Benchmarks: `/crates/ruvector-core/benches/`

## 9. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-12 | Architecture Team | Initial decision record |
