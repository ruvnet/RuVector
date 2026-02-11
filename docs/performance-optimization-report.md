# RuVector Performance Optimization Report

**Date**: 2026-02-11
**Scope**: Workspace-wide performance analysis covering compilation profiles, SIMD vectorization, memory allocation patterns, and parallelism opportunities.

---

## 1. Compilation Profile Analysis

### Current Release Profile (`Cargo.toml` workspace root)

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = true
panic = "abort"
```

**Assessment: STRONG** -- This is a near-optimal release profile.

| Setting | Value | Impact | Status |
|---------|-------|--------|--------|
| `opt-level` | 3 | Maximum optimization | Optimal |
| `lto` | "fat" | Full cross-crate inlining | Optimal |
| `codegen-units` | 1 | Best single-threaded codegen | Optimal |
| `strip` | true | Smaller binaries | Good |
| `panic` | "abort" | No unwinding overhead | Good |

**Bench profile** inherits from release with `debug = true` -- correct for profiling with symbols.

### Missing Optimizations

#### 1.1 No Profile-Guided Optimization (PGO) Setup

**Impact: MEDIUM (5-15% throughput improvement)**

There is no PGO workflow configured. For a vector database workload that is heavily compute-bound (distance calculations, HNSW graph traversal), PGO can significantly improve branch prediction and code layout.

**Recommendation**: Add a PGO build script:
```bash
# Step 1: Build instrumented binary
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release
# Step 2: Run representative workload (benchmarks)
./target/release/ruvector-bench
# Step 3: Merge profiles
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data
# Step 4: Build optimized binary
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" cargo build --release
```

#### 1.2 No `target-cpu=native` in RUSTFLAGS

**Impact: LOW-MEDIUM (enables AVX-512 on supporting hardware)**

The workspace relies entirely on runtime feature detection (`is_x86_feature_detected!`) rather than compile-time targeting. While this maximizes portability, adding a CI pipeline that also produces `target-cpu=native` builds would eliminate the feature detection overhead in hot loops and enable the compiler to use wider instructions throughout.

**Recommendation**: For deployment builds, set:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

#### 1.3 No `overflow-checks = false` for Release

**Impact: LOW**

The default release profile has overflow checks disabled, which is correct. No issue here.

---

## 2. SIMD Coverage Assessment

### 2.1 `ruvector-core` Distance Module

**File**: `crates/ruvector-core/src/distance.rs`

The primary distance dispatch uses SimSIMD (a C library with hardware-optimized kernels) when the `simd` feature is enabled:

```rust
// Lines 28-33
#[cfg(all(feature = "simd", not(target_arch = "wasm32")))]
{
    (simsimd::SpatialSimilarity::sqeuclidean(a, b)
        .expect("SimSIMD euclidean failed")
        .sqrt()) as f32
}
```

**Coverage**:
- Euclidean: SimSIMD (auto-dispatched AVX2/AVX-512/NEON)
- Cosine: SimSIMD (auto-dispatched)
- Dot Product: SimSIMD (auto-dispatched)
- Manhattan: **NO SIMD** -- scalar only

**Finding OPT-SIMD-1**: Manhattan distance (`distance.rs:85`) has no SIMD path:
```rust
// Line 85 -- scalar only, no SIMD acceleration
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}
```
**Impact**: LOW (Manhattan is less commonly used for vector search, but is a supported metric)
**Recommendation**: Use `simsimd::SpatialSimilarity::l1` or implement a SIMD-based absolute-difference-sum kernel.

### 2.2 `ruvector-core` SIMD Intrinsics Module

**File**: `crates/ruvector-core/src/simd_intrinsics.rs`

This module provides hand-tuned intrinsics with a complete dispatch hierarchy:
- AVX-512 > AVX2+FMA > AVX2 > SSE4.2 (x86_64)
- NEON with 4x unrolling (aarch64)
- Scalar fallback

**Assessment: STRONG** -- Comprehensive coverage with loop unrolling, prefetch hints, and FMA usage.

**Finding OPT-SIMD-2**: However, `simd_intrinsics.rs` is not used by the main `distance.rs` module. The main distance calculations route through SimSIMD, making this module effectively dead code for the core search path.

**Impact**: MEDIUM -- The `simd_intrinsics.rs` module has potentially better-tuned implementations (e.g., prefetch hints, 4x unrolling) than SimSIMD's generic dispatch. Consider benchmarking against SimSIMD to determine which path is faster on each platform.

### 2.3 `prime-radiant` SIMD Module

**Files**: `crates/prime-radiant/src/simd/vectors.rs`, `matrix.rs`, `energy.rs`

Uses the `wide` crate (`f32x8`) for cross-platform SIMD:

- `dot_product_simd`: 4-accumulator ILP with 8-wide FMA -- **EXCELLENT**
- `norm_squared_simd`: 4-accumulator ILP -- **EXCELLENT**
- `subtract_simd`: 8-wide vectorized -- **GOOD**
- `squared_distance_simd`: Fused subtract+square without allocation -- **EXCELLENT**
- `scale_simd`: 8-wide vectorized -- **GOOD**
- `fma_simd`: Vectorized FMA -- **GOOD**

Feature gating:
- NEON (aarch64): Covered via `wide` crate abstraction
- AVX2 (x86_64): Covered via `wide` crate abstraction (emits AVX2 with `target-cpu=native`)
- AVX-512: **NOT explicitly covered** by `wide` crate; relies on auto-vectorization

**Finding OPT-SIMD-3**: `prime-radiant` vectors module small-vector threshold is 16 elements. For `wide::f32x8` this means only vectors with >= 16 floats use SIMD. Since coherence vectors are typically 256-1024 dimensions, this is fine. However, the unrolling pattern uses nested `if let` chains instead of proper loop unrolling:

```rust
// vectors.rs:75-97 -- Nested if-let chains for unrolling
while let (Some(ca0), Some(cb0)) = (chunks_a_iter.next(), chunks_b_iter.next()) {
    // ...
    if let (Some(ca1), Some(cb1)) = (chunks_a_iter.next(), chunks_b_iter.next()) {
        // ...nested 4 levels deep
```

**Impact**: LOW -- This achieves the goal but may confuse the optimizer. A flat loop with index-based access and `#[unroll]` attributes or compiler-hint-based approaches may produce slightly cleaner machine code.

### 2.4 `ruvector-core` Cache-Optimized Module (SoA Layout)

**File**: `crates/ruvector-core/src/cache_optimized.rs`

The `SoAVectorStorage` has architecture-specific SIMD for batch euclidean distances:

- **aarch64 NEON**: Lines 247-302 -- `batch_euclidean_distances_neon` using `vfmaq_f32` (FMA), `vsqrtq_f32`
- **x86_64 AVX2**: Lines 305-351 -- `batch_euclidean_distances_avx2`

**Finding OPT-SIMD-4**: The AVX2 batch distance path (line 347) falls back to scalar `sqrt()`:
```rust
// cache_optimized.rs:347-349
// Take square root (no SIMD sqrt in basic AVX2, use scalar)
for distance in output.iter_mut() {
    *distance = distance.sqrt();
}
```
This is incorrect -- AVX2 does have `_mm256_sqrt_ps` for 8-wide f32 sqrt. The NEON path correctly uses `vsqrtq_f32`.

**Impact**: MEDIUM -- For large batch operations, the final sqrt pass is pure scalar. Using `_mm256_sqrt_ps` would give ~8x speedup for this specific pass. Alternatively, if only relative ordering matters (e.g., for k-NN), the sqrt can be deferred entirely.

**Finding OPT-SIMD-5**: The AVX2 path does not use FMA (`_mm256_fmadd_ps`):
```rust
// cache_optimized.rs:333-335
let diff = _mm256_sub_ps(dim_vals, query_val);
let sq = _mm256_mul_ps(diff, diff);
let result = _mm256_add_ps(out_vals, sq);
```
Should be: `let result = _mm256_fmadd_ps(diff, diff, out_vals);` -- This fuses multiply+add into a single instruction, improving throughput and accuracy.

**Impact**: LOW-MEDIUM -- One fewer instruction per iteration in the inner loop.

### 2.5 Quantization SIMD

**File**: `crates/ruvector-core/src/quantization.rs`

- ScalarQuantized distance: NEON and AVX2 paths present
- BinaryQuantized hamming distance: NEON (`vcntq_u8`) and x86_64 (`_popcnt64`) paths present
- Int4Quantized distance: **NO SIMD** -- Pure scalar nibble extraction

**Finding OPT-SIMD-6**: `Int4Quantized::distance()` (lines 245-264) is entirely scalar:
```rust
// quantization.rs:254-262
for i in 0..self.dimensions {
    let byte_idx = i / 2;
    let shift = if i % 2 == 0 { 0 } else { 4 };
    let a = ((self.data[byte_idx] >> shift) & 0x0F) as i32;
    let b = ((other.data[byte_idx] >> shift) & 0x0F) as i32;
    let diff = a - b;
    sum_sq += diff * diff;
}
```

**Impact**: MEDIUM -- Int4 is the "cool data" tier (10-40% access frequency). A SIMD implementation using nibble extraction with `vpshufb` (AVX2) or NEON equivalents could provide 8-16x speedup.

---

## 3. Memory Allocation Hotspots

### 3.1 Vector Cloning in `insert_batch`

**File**: `crates/ruvector-core/src/vector_db.rs:155-159`

```rust
let index_entries: Vec<_> = ids
    .iter()
    .zip(entries.iter())
    .map(|(id, entry)| (id.clone(), entry.vector.clone()))
    .collect();
```

**Impact**: HIGH -- Every batch insert clones every vector (`Vec<f32>`) unnecessarily. For 128-dimensional vectors, each clone allocates 512 bytes. A batch of 10,000 vectors wastes ~5MB in transient allocations.

**Recommendation**: Pass `entries` by value (it's already owned) and destructure instead of cloning. Or restructure the index API to accept references.

### 3.2 HNSW `add_batch` Double Clone

**File**: `crates/ruvector-core/src/index/hnsw.rs:304-311`

```rust
let data_with_ids: Vec<_> = entries
    .iter()
    .enumerate()
    .map(|(i, (id, vector))| {
        let idx = inner.next_idx + i;
        (id.clone(), idx, vector.clone())  // Clone both id AND vector
    })
    .collect();
```

Then at lines 324-327, the id is cloned AGAIN:
```rust
for (id, idx, vector) in data_with_ids {
    inner.vectors.insert(id.clone(), vector);  // Third clone of id
    inner.id_to_idx.insert(id.clone(), idx);   // Fourth clone of id
    inner.idx_to_id.insert(idx, id);
}
```

**Impact**: HIGH -- Each vector and ID is cloned multiple times during batch insertion. The vector data is cloned at least twice (once in data_with_ids, once into DashMap). For a 128-dim batch of 10,000 vectors, this wastes ~10MB.

**Recommendation**: Consume the entries vector directly. Use `into_iter()` instead of `iter()` on `entries` and restructure to avoid redundant clones.

### 3.3 K-Means Clustering Allocation Storm

**File**: `crates/ruvector-core/src/quantization.rs:559-596`

```rust
for _ in 0..iterations {
    let mut assignments = vec![Vec::new(); k];  // Allocates k empty Vecs per iteration
    for vector in vectors {
        // ...
        assignments[nearest].push(vector.clone());  // Clones every vector every iteration
    }
    // ...
    *centroid = vec![0.0; dim];  // Allocates new centroid every iteration
}
```

**Impact**: HIGH -- For `iterations=20` with `k=256` codebooks on 1000 vectors of 128 dimensions:
- 20 iterations * 1000 vectors * 128 dims * 4 bytes = ~10MB of cloned vector data per iteration
- 20 * 256 centroid allocations = 5,120 allocations

**Recommendation**:
1. Pre-allocate `assignments` as `Vec<Vec<usize>>` (store indices, not clones)
2. Pre-allocate centroids outside the loop and zero-fill with `fill(0.0)` instead of re-allocating

### 3.4 Product Quantization Subspace Extraction

**File**: `crates/ruvector-core/src/quantization.rs:146-147`

```rust
let subspace_vectors: Vec<Vec<f32>> =
    vectors.iter().map(|v| v[start..end].to_vec()).collect();
```

**Impact**: MEDIUM -- Creates `n` new `Vec<f32>` allocations per subspace, when slices would suffice. For 8 subspaces on 1000 vectors, this creates 8,000 heap allocations.

**Recommendation**: Pass slices directly to `kmeans_clustering` instead of copying into owned Vecs.

### 3.5 Search Result Enrichment

**File**: `crates/ruvector-core/src/vector_db.rs:170-178`

```rust
for result in &mut results {
    if let Ok(Some(entry)) = self.storage.get(&result.id) {
        result.vector = Some(entry.vector);  // Moves full vector into result
        result.metadata = entry.metadata;
    }
}
```

**Impact**: LOW-MEDIUM -- Each search result that includes vectors loads the full vector data from storage. For large k values (e.g., k=100 with 768-dim vectors), this is ~300KB of vector data copied per query. If the caller does not need the full vectors (only scores), this is wasted work.

**Recommendation**: Add a `include_vectors: bool` field to `SearchQuery` to make this opt-in.

### 3.6 `AtomicVectorPool::acquire` Zeroing

**File**: `crates/ruvector-core/src/lockfree.rs:270-273`

```rust
let vec = if let Some(mut v) = self.pool.pop() {
    self.pool_hits.fetch_add(1, Ordering::Relaxed);
    v.fill(0.0);  // Zero-fills every acquired vector
    v
```

**Impact**: LOW -- If the caller is going to overwrite the vector immediately (via `copy_from`), the `fill(0.0)` is wasted work. Consider a `acquire_uninitialized()` variant.

---

## 4. Parallelism Opportunities

### 4.1 Current Rayon Usage

| Location | Pattern | Status |
|----------|---------|--------|
| `distance.rs:96-100` | `batch_distances` with `par_iter` | Parallel (feature-gated) |
| `index/flat.rs:42-49` | FlatIndex search with `par_bridge` | Parallel (feature-gated) |
| `index/hnsw.rs:301` | `use rayon::prelude::*` imported but **NOT USED** | **DEAD IMPORT** |

### 4.2 HNSW Batch Insert is Sequential

**File**: `crates/ruvector-core/src/index/hnsw.rs:316-321`

```rust
// Insert into HNSW sequentially
// Note: Using sequential insertion to avoid Send requirements with RwLock guard
for (_id, idx, vector) in &data_with_ids {
    inner.hnsw.insert_data(vector, *idx);
}
```

**Impact**: HIGH -- HNSW insertion is O(log n) per vector with significant graph traversal. For batch inserts of 10,000+ vectors, this serial bottleneck dominates insertion time. The `hnsw_rs` crate supports `parallel_insert` via its API.

**Recommendation**: Use `hnsw_rs::Hnsw::parallel_insert` for batch operations. This requires restructuring to release the write lock before parallel insertion, or using the crate's built-in parallel insertion API.

### 4.3 HNSW Deserialization is O(n^2)

**File**: `crates/ruvector-core/src/index/hnsw.rs:205-211`

```rust
for entry in idx_to_id.iter() {
    let idx = *entry.key();
    let id = entry.value();
    if let Some(vector) = state.vectors.iter().find(|(vid, _)| vid == id) {  // O(n) scan
        hnsw.insert_data(&vector.1, idx);
    }
}
```

**Impact**: HIGH -- The inner `find()` performs a linear scan of the vectors list for every index entry, making deserialization O(n^2). For 100,000 vectors, this means 10 billion comparisons.

**Recommendation**: Convert `state.vectors` to a `HashMap<String, Vec<f32>>` before the loop for O(n) deserialization.

### 4.4 FlatIndex Uses `par_bridge` Instead of `par_iter`

**File**: `crates/ruvector-core/src/index/flat.rs:42-43`

```rust
.iter()
.par_bridge()
```

**Impact**: MEDIUM -- `par_bridge()` converts a sequential iterator into a parallel one with limited work-stealing efficiency. Since `DashMap::iter()` internally iterates shard-by-shard, using `par_bridge` loses the natural shard parallelism.

**Recommendation**: Instead of `DashMap`, consider sharding vectors into `N` separate `Vec<(VectorId, Vec<f32>)>` shards that can be processed with true `par_iter()`. Alternatively, collect into a Vec first and then `par_iter()` on that, though this adds allocation overhead.

### 4.5 VectorDB Index Rebuild at Startup is Serial

**File**: `crates/ruvector-core/src/vector_db.rs:101-118`

```rust
let stored_ids = storage.all_ids()?;
// ...
let mut entries = Vec::with_capacity(stored_ids.len());
for id in stored_ids {
    if let Some(entry) = storage.get(&id)? {
        entries.push((id, entry.vector));
    }
}
index.add_batch(entries)?;
```

**Impact**: MEDIUM -- Storage reads are sequential. For databases with many vectors, reading all vectors from disk sequentially leaves I/O bandwidth on the table. The subsequent `add_batch` is also serial (as noted in 4.2).

**Recommendation**: Use `rayon::par_iter` on `stored_ids` for parallel storage reads (if the storage backend supports concurrent reads, which `redb` does).

### 4.6 K-Means Clustering is Not Parallelized

**File**: `crates/ruvector-core/src/quantization.rs:559-596`

The assignment step (find nearest centroid for each vector) is embarrassingly parallel but runs sequentially.

**Impact**: MEDIUM -- Product quantization training is typically done offline, but for large datasets it can take minutes. Parallelizing the assignment step with `rayon::par_iter` would give near-linear speedup.

---

## 5. Summary of Recommendations

### High Impact

| ID | Finding | File:Lines | Estimated Impact |
|----|---------|-----------|-----------------|
| OPT-MEM-1 | Double/triple vector cloning in `insert_batch` | `vector_db.rs:155-159`, `hnsw.rs:304-327` | 2-5x less allocation in batch insert |
| OPT-MEM-2 | K-means clones all vectors every iteration | `quantization.rs:559-596` | 10-20x less allocation during PQ training |
| OPT-PAR-1 | HNSW batch insert is sequential | `hnsw.rs:316-321` | 3-8x faster batch insertion (CPU-bound) |
| OPT-PAR-2 | HNSW deserialization is O(n^2) | `hnsw.rs:205-211` | O(n) instead of O(n^2) for index loading |

### Medium Impact

| ID | Finding | File:Lines | Estimated Impact |
|----|---------|-----------|-----------------|
| OPT-SIMD-4 | AVX2 batch distance uses scalar sqrt | `cache_optimized.rs:347-349` | ~8x faster final sqrt pass |
| OPT-SIMD-6 | Int4 quantized distance has no SIMD | `quantization.rs:254-262` | 8-16x faster Int4 distance |
| OPT-SIMD-2 | `simd_intrinsics.rs` not used by main distance path | `distance.rs` vs `simd_intrinsics.rs` | Potential improvement if hand-tuned kernels beat SimSIMD |
| OPT-MEM-3 | PQ subspace vectors copied instead of sliced | `quantization.rs:146-147` | 8,000 fewer heap allocations per PQ training |
| OPT-PAR-3 | FlatIndex uses `par_bridge` inefficiently | `flat.rs:42-43` | Better work distribution in brute-force search |
| OPT-PGO-1 | No PGO build pipeline | Workspace root | 5-15% overall throughput improvement |
| OPT-PAR-4 | K-means not parallelized | `quantization.rs:559-596` | Near-linear speedup for PQ training |

### Low Impact

| ID | Finding | File:Lines | Estimated Impact |
|----|---------|-----------|-----------------|
| OPT-SIMD-1 | Manhattan distance has no SIMD | `distance.rs:85` | 4-8x for Manhattan queries |
| OPT-SIMD-5 | AVX2 batch distance misses FMA | `cache_optimized.rs:333-335` | ~10% fewer cycles in inner loop |
| OPT-SIMD-3 | Nested if-let unrolling pattern | `prime-radiant/vectors.rs:75-97` | Marginal codegen improvement |
| OPT-MEM-4 | Pool zero-fills on acquire unconditionally | `lockfree.rs:270-273` | Avoids unnecessary memset for overwrite patterns |
| OPT-MEM-5 | Search enrichment always loads vectors | `vector_db.rs:170-178` | Saves ~300KB/query when vectors not needed |

---

## 6. Architecture-Level Observations

### Strengths

1. **Well-structured SIMD hierarchy**: Multiple SIMD implementations (SimSIMD, hand-tuned intrinsics, `wide` crate) provide good coverage across architectures.
2. **SoA layout for batch operations**: `SoAVectorStorage` with dimension-wise processing is cache-optimal for batch distance computations.
3. **Arena allocator**: `CacheAlignedVec` and `Arena` types with 64-byte alignment eliminate allocation overhead in hot paths.
4. **Lock-free data structures**: `AtomicVectorPool`, `LockFreeWorkQueue`, and `LockFreeBatchProcessor` minimize contention.
5. **Feature gating**: Clean separation of SIMD, parallel, storage, and HNSW features enables WASM compatibility.

### Weaknesses

1. **Redundant SIMD implementations**: `simd_intrinsics.rs`, `distance.rs` (SimSIMD), and `cache_optimized.rs` all implement distance calculations independently. Consolidation would reduce maintenance burden and ensure all paths benefit from optimizations.
2. **Excessive cloning in data paths**: The insert and batch-insert paths clone vectors 2-4 times. This is the single largest allocator overhead in the hot path.
3. **No deferred sqrt optimization**: Most k-NN workloads only need relative ordering. Computing `sqrt()` on every distance is unnecessary when the monotonicity of `sqrt` preserves ordering. Deferring sqrt to only the final k results would save significant compute.
4. **DashMap overhead in HNSW**: The HNSW index stores vectors in a `DashMap` alongside the graph. Since the graph already stores the data via `hnsw_rs`, this is redundant storage. Consider removing the `vectors` DashMap and reading directly from the graph when needed.
