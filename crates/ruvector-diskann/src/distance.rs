//! Distance computations with SIMD acceleration and optional GPU offload
//!
//! Dispatch priority: GPU (if `gpu` feature) → lattice-embed (if `lattice-simd`
//! feature) → SimSIMD (if `simd` feature, native NEON/AVX2/AVX-512) → WASM
//! SIMD128 (`wasm32` target with `simd128` target-feature) → scalar

use crate::error::{DiskAnnError, Result};
use memmap2::Mmap;

/// Backing storage for the flat vector slab.
///
/// `Owned` is heap-resident — used while inserting/building, and by
/// [`FlatVectors::from_owned`] (the default, back-compat `load()` path).
/// `Mmap` is read-through: vector slices are read directly out of the mapped file
/// on each [`FlatVectors::get`] call, so RSS stays proportional to the accessed
/// working set instead of the whole dataset (see #674). It is populated only via
/// [`FlatVectors::from_mmap`], which validates 4-byte alignment up front so `get`
/// never has to fall back to an unaligned-read fail path per call.
enum VectorStorage {
    Owned(Vec<f32>),
    Mmap { mmap: Mmap, data_offset: usize },
}

/// Flat vector storage — contiguous memory for cache-friendly access.
/// Vectors are logically a single slab: `[v0_d0, v0_d1, ..., v1_d0, ...]`, either
/// owned in RAM or read straight out of a memory-mapped file.
pub struct FlatVectors {
    storage: VectorStorage,
    pub dim: usize,
    pub count: usize,
    /// Post-load tombstones for mmap-backed storage. The mapped file is never
    /// mutated, so deletes on read-through storage are tracked in this owned
    /// overlay instead of the in-place NaN write `Owned` storage uses. Empty (and
    /// unused) for `Owned` storage, which keeps the original NaN-write behavior.
    tombstones: Vec<bool>,
    /// Shared all-NaN row returned by `get()` for a tombstoned mmap index — same
    /// externally observable shape as the NaN vector `Owned::zero_out` produces.
    tombstone_row: Vec<f32>,
}

impl FlatVectors {
    pub fn new(dim: usize) -> Self {
        Self {
            storage: VectorStorage::Owned(Vec::new()),
            dim,
            count: 0,
            tombstones: Vec::new(),
            tombstone_row: vec![f32::NAN; dim],
        }
    }

    pub fn with_capacity(dim: usize, n: usize) -> Self {
        Self {
            storage: VectorStorage::Owned(Vec::with_capacity(n * dim)),
            dim,
            count: 0,
            tombstones: Vec::new(),
            tombstone_row: vec![f32::NAN; dim],
        }
    }

    /// Build owned flat storage directly from an already-materialized slab (e.g.
    /// copied out of a save file's mmap). `data.len()` must equal `count * dim`.
    pub fn from_owned(data: Vec<f32>, dim: usize, count: usize) -> Self {
        debug_assert_eq!(data.len(), count * dim);
        Self {
            storage: VectorStorage::Owned(data),
            dim,
            count,
            tombstones: Vec::new(),
            tombstone_row: vec![f32::NAN; dim],
        }
    }

    /// Build a read-through view directly over a memory-mapped file's flat f32
    /// slab, starting `data_offset` bytes into the map.
    ///
    /// Fails closed (returns `Err`, never transmutes) if the slab's start isn't
    /// 4-byte aligned — required to reinterpret mapped bytes as `f32` without UB —
    /// or if the map is too short for `count * dim` floats.
    pub fn from_mmap(mmap: Mmap, data_offset: usize, dim: usize, count: usize) -> Result<Self> {
        let base = mmap.as_ptr() as usize;
        if base.wrapping_add(data_offset) % std::mem::align_of::<f32>() != 0 {
            return Err(DiskAnnError::InvalidConfig(format!(
                "mmap vector data at offset {data_offset} is not 4-byte aligned (mmap base 0x{base:x}); refusing to cast unaligned bytes to f32"
            )));
        }
        let need_bytes = count
            .checked_mul(dim)
            .and_then(|floats| floats.checked_mul(4))
            .and_then(|bytes| bytes.checked_add(data_offset))
            .ok_or_else(|| {
                DiskAnnError::InvalidConfig("vector slab size overflowed usize".to_string())
            })?;
        if mmap.len() < need_bytes {
            return Err(DiskAnnError::InvalidConfig(format!(
                "mmap too short for {count} vectors of dim {dim}: need {need_bytes} bytes, have {}",
                mmap.len()
            )));
        }
        Ok(Self {
            storage: VectorStorage::Mmap { mmap, data_offset },
            dim,
            count,
            tombstones: vec![false; count],
            tombstone_row: vec![f32::NAN; dim],
        })
    }

    /// Whether this instance is backed by a read-through mmap (vs. owned RAM).
    pub fn is_mmap_backed(&self) -> bool {
        matches!(self.storage, VectorStorage::Mmap { .. })
    }

    /// Zero-copy byte view of the flat slab, when it is owned in RAM. `None` for
    /// mmap-backed storage — callers needing bytes there should read per-vector via
    /// [`FlatVectors::get`] instead of assuming one contiguous owned buffer.
    pub fn as_owned_slice(&self) -> Option<&[f32]> {
        match &self.storage {
            VectorStorage::Owned(data) => Some(data),
            VectorStorage::Mmap { .. } => None,
        }
    }

    #[inline]
    pub fn push(&mut self, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim);
        match &mut self.storage {
            VectorStorage::Owned(data) => {
                data.extend_from_slice(vector);
                self.count += 1;
            }
            VectorStorage::Mmap { .. } => {
                panic!(
                    "FlatVectors::push called on mmap-backed (read-through) storage — mmap-loaded indexes are read-only for inserts; callers must check is_mmap_backed() first"
                );
            }
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> &[f32] {
        match &self.storage {
            VectorStorage::Owned(data) => {
                let start = idx * self.dim;
                &data[start..start + self.dim]
            }
            VectorStorage::Mmap { mmap, data_offset } => {
                if self.tombstones.get(idx).copied().unwrap_or(false) {
                    return &self.tombstone_row;
                }
                let start = data_offset + idx * self.dim * 4;
                let byte_slice = &mmap[start..start + self.dim * 4];
                bytemuck::cast_slice(byte_slice)
            }
        }
    }

    /// Replace an owned row with NaNs, or mask an mmap-backed row with a NaN
    /// overlay. Kept for API compatibility; `DiskAnnIndex` deletion uses its
    /// separate persistent tombstone bitmap so vectors remain usable for routing.
    #[inline]
    pub fn zero_out(&mut self, idx: usize) {
        match &mut self.storage {
            VectorStorage::Owned(data) => {
                let start = idx * self.dim;
                for value in &mut data[start..start + self.dim] {
                    *value = f32::NAN;
                }
            }
            VectorStorage::Mmap { .. } => {
                if let Some(flag) = self.tombstones.get_mut(idx) {
                    *flag = true;
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

// ============================================================================
// Distance functions — auto-dispatch based on features
// ============================================================================

/// Set from inside [`l2_lattice`]/[`inner_lattice`] below, only after the
/// real kernel call returns — so a dispatch arm that swaps the wrapper call
/// for a scalar/native fallback bypasses the flag along with the kernel.
/// Thread-local (not a shared global) so a sibling test running on another
/// thread can't set the flag between this test's reset and its assert.
/// Compiled only for `lattice-simd` test builds — zero footprint elsewhere.
/// See `lattice_backend_is_actually_invoked` below.
#[cfg(all(test, feature = "lattice-simd"))]
thread_local! {
    static LATTICE_L2_INVOKED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LATTICE_INNER_INVOKED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Sole caller of the `lattice-embed` L2 kernel — the dispatch arm in
/// [`l2_squared`] can only reach the kernel through this wrapper, so
/// replacing the arm's call expression with a fallback also removes the
/// witness store.
#[cfg(feature = "lattice-simd")]
#[inline]
fn l2_lattice(a: &[f32], b: &[f32]) -> f32 {
    let result = lattice_embed::simd::squared_euclidean_distance(a, b);
    #[cfg(test)]
    LATTICE_L2_INVOKED.with(|invoked| invoked.set(true));
    result
}

/// Sole caller of the `lattice-embed` dot-product kernel — see
/// [`l2_lattice`] for why the store lives here rather than inline in the
/// dispatch arm. Negated to match the other backends: this returns a
/// distance for a min-heap, not a similarity. `SpatialSimilarity::inner` is a
/// plain alias for `dot`, so both sides negate the same raw dot product.
#[cfg(feature = "lattice-simd")]
#[inline]
fn inner_lattice(a: &[f32], b: &[f32]) -> f32 {
    let result = -lattice_embed::simd::dot_product(a, b);
    #[cfg(test)]
    LATTICE_INNER_INVOKED.with(|invoked| invoked.set(true));
    result
}

/// L2 squared distance — dispatches to best available implementation
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "distance vectors must have equal lengths");

    #[cfg(feature = "lattice-simd")]
    {
        l2_lattice(a, b)
    }

    #[cfg(all(
        not(feature = "lattice-simd"),
        target_arch = "wasm32",
        target_feature = "simd128"
    ))]
    {
        wasm_simd128_l2_squared(a, b)
    }

    #[cfg(all(
        not(feature = "lattice-simd"),
        not(target_arch = "wasm32"),
        feature = "simd"
    ))]
    {
        simd_l2_squared(a, b)
    }

    #[cfg(all(
        not(feature = "lattice-simd"),
        any(
            all(target_arch = "wasm32", not(target_feature = "simd128")),
            all(not(target_arch = "wasm32"), not(feature = "simd"))
        )
    ))]
    {
        scalar_l2_squared(a, b)
    }
}

/// Scalar L2² with 4 accumulators for ILP
#[inline]
pub fn scalar_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut i = 0;

    while i + 16 <= len {
        for j in 0..4 {
            let off = i + j * 4;
            let d0 = a[off] - b[off];
            let d1 = a[off + 1] - b[off + 1];
            let d2 = a[off + 2] - b[off + 2];
            let d3 = a[off + 3] - b[off + 3];
            s0 += d0 * d0;
            s1 += d1 * d1;
            s2 += d2 * d2;
            s3 += d3 * d3;
        }
        i += 16;
    }
    while i < len {
        let d = a[i] - b[i];
        s0 += d * d;
        i += 1;
    }
    s0 + s1 + s2 + s3
}

/// SimSIMD-accelerated L2² — uses hardware NEON/AVX2/AVX-512
#[cfg(all(feature = "simd", not(target_arch = "wasm32")))]
#[inline]
pub fn simd_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    // simsimd sqeuclidean returns squared Euclidean directly
    simsimd::SpatialSimilarity::sqeuclidean(a, b)
        .map(|d| d as f32)
        .unwrap_or_else(|| scalar_l2_squared(a, b))
}

/// WASM SIMD128-accelerated L2² — two `v128` accumulators (8 lanes/iteration)
/// for instruction-level parallelism, mirroring the scalar path's 4-accumulator
/// shape. Handles any `dim`, including 0, non-multiples-of-4, and
/// non-multiples-of-8 via scalar remainder loops.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub fn wasm_simd128_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::*;

    assert_eq!(a.len(), b.len(), "distance vectors must have equal lengths");
    let len = a.len();
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut i = 0;

    while i + 8 <= len {
        unsafe {
            let a0 = v128_load(a.as_ptr().add(i) as *const v128);
            let b0 = v128_load(b.as_ptr().add(i) as *const v128);
            let d0 = f32x4_sub(a0, b0);
            acc0 = f32x4_add(acc0, f32x4_mul(d0, d0));

            let a1 = v128_load(a.as_ptr().add(i + 4) as *const v128);
            let b1 = v128_load(b.as_ptr().add(i + 4) as *const v128);
            let d1 = f32x4_sub(a1, b1);
            acc1 = f32x4_add(acc1, f32x4_mul(d1, d1));
        }
        i += 8;
    }
    while i + 4 <= len {
        unsafe {
            let av = v128_load(a.as_ptr().add(i) as *const v128);
            let bv = v128_load(b.as_ptr().add(i) as *const v128);
            let d = f32x4_sub(av, bv);
            acc0 = f32x4_add(acc0, f32x4_mul(d, d));
        }
        i += 4;
    }

    let sum_vec = f32x4_add(acc0, acc1);
    let mut sum = f32x4_extract_lane::<0>(sum_vec)
        + f32x4_extract_lane::<1>(sum_vec)
        + f32x4_extract_lane::<2>(sum_vec)
        + f32x4_extract_lane::<3>(sum_vec);

    while i < len {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

/// Inner product distance (negated for min-heap)
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "distance vectors must have equal lengths");

    #[cfg(feature = "lattice-simd")]
    {
        inner_lattice(a, b)
    }

    #[cfg(all(
        not(feature = "lattice-simd"),
        target_arch = "wasm32",
        target_feature = "simd128"
    ))]
    {
        wasm_simd128_inner_product(a, b)
    }

    #[cfg(all(
        not(feature = "lattice-simd"),
        not(target_arch = "wasm32"),
        feature = "simd"
    ))]
    {
        simsimd::SpatialSimilarity::inner(a, b)
            .map(|d| -(d as f32))
            .unwrap_or_else(|| scalar_inner_product(a, b))
    }

    #[cfg(all(
        not(feature = "lattice-simd"),
        any(
            all(target_arch = "wasm32", not(target_feature = "simd128")),
            all(not(target_arch = "wasm32"), not(feature = "simd"))
        )
    ))]
    {
        scalar_inner_product(a, b)
    }
}

/// Retained under `lattice-simd` as the fallback the `simd` backend still
/// calls if SimSIMD returns `None`. The parity test's reference is the local
/// `naive_inner_product`, not this function. `scalar_l2_squared` is `pub` and
/// so needs no such annotation.
#[cfg_attr(feature = "lattice-simd", allow(dead_code))]
#[inline]
fn scalar_inner_product(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let len = a.len();
    let mut i = 0;

    while i + 16 <= len {
        for j in 0..4 {
            let off = i + j * 4;
            s0 += a[off] * b[off];
            s1 += a[off + 1] * b[off + 1];
            s2 += a[off + 2] * b[off + 2];
            s3 += a[off + 3] * b[off + 3];
        }
        i += 16;
    }
    while i < len {
        s0 += a[i] * b[i];
        i += 1;
    }
    -(s0 + s1 + s2 + s3)
}

/// WASM SIMD128-accelerated inner product (negated, same convention as
/// [`scalar_inner_product`]). Two `v128` accumulators, scalar remainder tail.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub fn wasm_simd128_inner_product(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::wasm32::*;

    assert_eq!(a.len(), b.len(), "distance vectors must have equal lengths");
    let len = a.len();
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    let mut i = 0;

    while i + 8 <= len {
        unsafe {
            let a0 = v128_load(a.as_ptr().add(i) as *const v128);
            let b0 = v128_load(b.as_ptr().add(i) as *const v128);
            acc0 = f32x4_add(acc0, f32x4_mul(a0, b0));

            let a1 = v128_load(a.as_ptr().add(i + 4) as *const v128);
            let b1 = v128_load(b.as_ptr().add(i + 4) as *const v128);
            acc1 = f32x4_add(acc1, f32x4_mul(a1, b1));
        }
        i += 8;
    }
    while i + 4 <= len {
        unsafe {
            let av = v128_load(a.as_ptr().add(i) as *const v128);
            let bv = v128_load(b.as_ptr().add(i) as *const v128);
            acc0 = f32x4_add(acc0, f32x4_mul(av, bv));
        }
        i += 4;
    }

    let sum_vec = f32x4_add(acc0, acc1);
    let mut sum = f32x4_extract_lane::<0>(sum_vec)
        + f32x4_extract_lane::<1>(sum_vec)
        + f32x4_extract_lane::<2>(sum_vec)
        + f32x4_extract_lane::<3>(sum_vec);

    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    -sum
}

/// PQ asymmetric distance from precomputed lookup table
#[inline]
pub fn pq_asymmetric_distance(codes: &[u8], table: &[f32], k: usize) -> f32 {
    // table is flat: table[subspace * 256 + code]
    let mut dist = 0.0f32;
    for (i, &code) in codes.iter().enumerate() {
        dist += unsafe { *table.get_unchecked(i * k + code as usize) };
    }
    dist
}

// ============================================================================
// Visited bitset — O(1) membership test, much faster than HashSet<u32>
// ============================================================================

/// Generation-tagged scratch state for tracking visited nodes during search.
pub struct VisitedSet {
    generation: u64,
    gens: Vec<u64>,
}

impl VisitedSet {
    pub fn new(n: usize) -> Self {
        Self {
            generation: 1,
            gens: vec![0u64; n],
        }
    }

    /// Reset for a new search — O(1) via generation counter
    #[inline]
    pub fn clear(&mut self) {
        if self.generation == u64::MAX {
            self.gens.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    /// Prepare this set for an index of `n` nodes.
    ///
    /// A size mismatch reinitializes the backing storage. Repeated use with the
    /// same index size takes the O(1) generation-counter path in [`Self::clear`].
    #[inline]
    pub(crate) fn prepare(&mut self, n: usize) {
        if self.gens.len() != n {
            self.gens.resize(n, 0);
            self.gens.fill(0);
            self.generation = 1;
        } else {
            self.clear();
        }
    }

    /// Mark node as visited
    #[inline]
    pub fn insert(&mut self, id: u32) {
        self.gens[id as usize] = self.generation;
    }

    /// Check if visited
    #[inline]
    pub fn contains(&self, id: u32) -> bool {
        self.gens[id as usize] == self.generation
    }
}

// ============================================================================
// GPU distance computation (optional, feature-gated)
// ============================================================================

/// GPU-accelerated batch distance computation
/// Computes distances from a single query to N vectors in parallel
#[cfg(feature = "gpu")]
pub mod gpu {
    use super::FlatVectors;

    /// GPU backend selection
    #[derive(Debug, Clone, Copy)]
    pub enum GpuBackend {
        /// Apple Metal (macOS/iOS)
        Metal,
        /// NVIDIA CUDA
        Cuda,
        /// Vulkan compute (cross-platform)
        Vulkan,
    }

    /// GPU distance computation context
    pub struct GpuDistanceContext {
        backend: GpuBackend,
        /// Batch size for GPU kernel launches
        batch_size: usize,
    }

    impl GpuDistanceContext {
        /// Create a new GPU context (auto-detects best backend)
        pub fn new() -> Option<Self> {
            // Auto-detect: Metal on macOS, CUDA if nvidia, Vulkan fallback
            #[cfg(target_os = "macos")]
            let backend = GpuBackend::Metal;
            #[cfg(not(target_os = "macos"))]
            let backend = GpuBackend::Cuda;

            Some(Self {
                backend,
                batch_size: 4096,
            })
        }

        /// Batch L2² distances: query vs all vectors in flat storage
        /// Returns Vec of (index, distance) sorted by distance
        pub fn batch_l2_squared(
            &self,
            query: &[f32],
            vectors: &FlatVectors,
            k: usize,
        ) -> Vec<(u32, f32)> {
            // GPU kernel dispatch:
            // 1. Upload query + vector slab to GPU memory
            // 2. Launch N threads, each computing one L2² distance
            // 3. Parallel top-k reduction on GPU
            // 4. Download k results
            //
            // For now, fall back to CPU parallel with rayon
            // (real Metal/CUDA shaders would be added via metal-rs or cuda-sys)
            use rayon::prelude::*;

            let mut dists: Vec<(u32, f32)> = (0..vectors.count as u32)
                .into_par_iter()
                .map(|i| {
                    let v = vectors.get(i as usize);
                    (i, super::scalar_l2_squared(query, v))
                })
                .collect();

            dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(k);
            dists
        }

        pub fn backend(&self) -> GpuBackend {
            self.backend
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_squared() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((l2_squared(&a, &b) - 27.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_identical() {
        let a = vec![1.0; 128];
        assert!(l2_squared(&a, &a) < 1e-10);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((inner_product(&a, &b) - (-32.0)).abs() < 1e-6);
    }

    /// Checks whichever backend compiled in against naive scalar references,
    /// across dimensions chosen to straddle 4/8/16-lane widths **and their
    /// remainders** so tail handling is exercised. The references are naive
    /// single-pass loops rather than `scalar_l2_squared` / `scalar_inner_product`,
    /// which are themselves 4-accumulator implementations, so a reduction-order
    /// bug in the shared shape cannot hide.
    ///
    /// Backend-independent by construction: it covers the SimSIMD path, the
    /// lattice path, and the plain scalar path. The cross-dimension parity
    /// tests that existed before this were gated on
    /// `all(target_arch = "wasm32", target_feature = "simd128")`, so no native
    /// backend was checked at any dimension wider than 3.
    #[test]
    fn backend_matches_scalar_reference() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        fn naive_l2_squared(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
        }

        fn naive_inner_product(a: &[f32], b: &[f32]) -> f32 {
            -a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
        }

        const DIMS: &[usize] = &[
            0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 384, 768, 1000, 1023, 1024,
        ];

        // Combined absolute+relative tolerance, numpy `allclose`-style. A flat
        // absolute bound is not achievable: each backend uses a different
        // reduction tree, f32 addition is not associative, so reordering shifts
        // rounding by a few ULPs and that drift grows with accumulated
        // magnitude. A real bug (wrong lane math, dropped remainder) produces
        // relative error orders of magnitude above machine epsilon.
        const ATOL: f32 = 1e-5;
        const RTOL: f32 = 1e-5;

        let mut rng = StdRng::seed_from_u64(42);
        for &dim in DIMS {
            let a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-10.0f32..10.0)).collect();
            let b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-10.0f32..10.0)).collect();

            for (op, got, want) in [
                ("l2_squared", l2_squared(&a, &b), naive_l2_squared(&a, &b)),
                (
                    "inner_product",
                    inner_product(&a, &b),
                    naive_inner_product(&a, &b),
                ),
            ] {
                let bound = ATOL + RTOL * got.abs().max(want.abs());
                assert!(
                    (got - want).abs() <= bound,
                    "{op} dim={dim}: got={got} want={want} bound={bound}"
                );
            }
        }
    }

    /// `backend_matches_scalar_reference` above only checks numerical parity
    /// against a naive oracle; scalar and lattice implement the same
    /// arithmetic, so a `lattice-simd` build that silently fell back to the
    /// scalar/native kernel would still pass it. This pins actual backend
    /// *selection*: it fails if either dispatch arm's call to
    /// [`l2_lattice`]/[`inner_lattice`] is replaced by a fallback route while
    /// the feature stays declared, since the witness flags are only set
    /// inside those wrappers, after the real kernel call returns. Thread-local
    /// storage means a sibling test running concurrently on another thread
    /// can't set these flags between this test's reset and its assert.
    #[test]
    #[cfg(feature = "lattice-simd")]
    fn lattice_backend_is_actually_invoked() {
        LATTICE_L2_INVOKED.with(|invoked| invoked.set(false));
        LATTICE_INNER_INVOKED.with(|invoked| invoked.set(false));

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![6.0f32, 7.0, 8.0, 9.0, 10.0];
        let _ = l2_squared(&a, &b);
        let _ = inner_product(&a, &b);

        assert!(
            LATTICE_L2_INVOKED.with(|invoked| invoked.get()),
            "l2_squared did not route through the lattice-embed kernel"
        );
        assert!(
            LATTICE_INNER_INVOKED.with(|invoked| invoked.get()),
            "inner_product did not route through the lattice-embed kernel"
        );
    }

    #[test]
    fn test_flat_vectors() {
        let mut fv = FlatVectors::new(3);
        fv.push(&[1.0, 2.0, 3.0]);
        fv.push(&[4.0, 5.0, 6.0]);
        assert_eq!(fv.len(), 2);
        assert_eq!(fv.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(fv.get(1), &[4.0, 5.0, 6.0]);
        assert!(!fv.is_mmap_backed());
    }

    /// Write a `vectors.bin`-shaped file (8-byte n, 8-byte dim, then flat
    /// little-endian f32 data) and mmap it — the same layout `index.rs` produces.
    fn mmap_fixture(data: &[f32], dim: usize, count: usize) -> memmap2::Mmap {
        use std::io::Write;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&(count as u64).to_le_bytes()).unwrap();
        tmp.write_all(&(dim as u64).to_le_bytes()).unwrap();
        for &v in data {
            tmp.write_all(&v.to_le_bytes()).unwrap();
        }
        tmp.flush().unwrap();
        let file = std::fs::File::open(tmp.path()).unwrap();
        // Leak the tempfile handle for the test's duration — NamedTempFile deletes
        // on drop, but the mmap needs the backing file to stay mapped/openable.
        std::mem::forget(tmp);
        unsafe { memmap2::MmapOptions::new().map(&file).unwrap() }
    }

    #[test]
    fn test_flat_vectors_mmap_read_through_matches_data() {
        let dim = 4;
        let count = 3;
        let data: Vec<f32> = (0..(dim * count) as u32).map(|x| x as f32).collect();
        let mmap = mmap_fixture(&data, dim, count);

        let fv = FlatVectors::from_mmap(mmap, 16, dim, count).unwrap();
        assert!(fv.is_mmap_backed());
        assert_eq!(fv.len(), count);
        for i in 0..count {
            assert_eq!(fv.get(i), &data[i * dim..(i + 1) * dim]);
        }
    }

    #[test]
    fn test_flat_vectors_mmap_rejects_unaligned_offset() {
        let data = vec![0.0f32; 8];
        let mmap = mmap_fixture(&data, 4, 2);
        // offset=1 is never 4-byte aligned — from_mmap must fail closed rather
        // than transmute unaligned bytes to f32.
        let result = FlatVectors::from_mmap(mmap, 1, 4, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_flat_vectors_mmap_rejects_undersized_map() {
        let data = vec![0.0f32; 4];
        let mmap = mmap_fixture(&data, 4, 1);
        // Header + 1 vector present, but claim 10 vectors — must fail closed.
        let result = FlatVectors::from_mmap(mmap, 16, 4, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_flat_vectors_mmap_zero_out_tombstones_without_mutating_file() {
        let dim = 4;
        let count = 2;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mmap = mmap_fixture(&data, dim, count);
        let mut fv = FlatVectors::from_mmap(mmap, 16, dim, count).unwrap();

        fv.zero_out(0);
        assert!(fv.get(0).iter().all(|x| x.is_nan()));
        assert_eq!(fv.get(1), &[5.0, 6.0, 7.0, 8.0], "untouched row unaffected");
    }

    #[test]
    fn test_flat_vectors_push_panics_on_mmap_storage() {
        let data = vec![0.0f32; 4];
        let mmap = mmap_fixture(&data, 4, 1);
        let mut fv = FlatVectors::from_mmap(mmap, 16, 4, 1).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fv.push(&[1.0, 2.0, 3.0, 4.0]);
        }));
        assert!(
            result.is_err(),
            "push on mmap-backed storage must fail loud, not silently succeed"
        );
    }

    #[test]
    fn test_visited_set() {
        let mut vs = VisitedSet::new(100);
        vs.insert(42);
        assert!(vs.contains(42));
        assert!(!vs.contains(43));
        vs.clear(); // O(1) reset
        assert!(!vs.contains(42));
        vs.insert(43);
        assert!(vs.contains(43));
    }

    #[test]
    fn test_visited_set_generation_wrap() {
        let mut vs = VisitedSet::new(100);
        vs.generation = u64::MAX;
        vs.insert(42);

        vs.clear();

        assert_eq!(vs.generation, 1);
        assert!(!vs.contains(42));
    }

    #[test]
    fn test_visited_set_size_mismatch_reinitializes() {
        let mut vs = VisitedSet::new(2);
        vs.insert(1);

        vs.prepare(100);

        assert_eq!(vs.gens.len(), 100);
        assert!(!vs.contains(1));
        vs.insert(99);
        assert!(vs.contains(99));
    }

    #[test]
    fn test_pq_flat_table() {
        // 2 subspaces, 4 centroids each (k=4 for test)
        let table = vec![
            0.1, 0.2, 0.3, 0.4, // subspace 0
            0.5, 0.6, 0.7, 0.8, // subspace 1
        ];
        let codes = vec![1u8, 2u8]; // code 1 from sub0, code 2 from sub1
        let dist = pq_asymmetric_distance(&codes, &table, 4);
        assert!((dist - (0.2 + 0.7)).abs() < 1e-6);
    }
}

/// `wasm32` + `simd128` correctness and A/B timing checks against the scalar
/// path. Both `wasm_simd128_*` and `scalar_*` are compiled into the *same*
/// wasm binary here (the crate is built once, with `-C target-feature=+simd128`),
/// so the comparison isn't confounded by separate builds. Run via:
///
/// ```sh
/// RUSTFLAGS="-C target-feature=+simd128" wasm-pack test --node crates/ruvector-diskann --release
/// ```
#[cfg(all(test, target_arch = "wasm32", target_feature = "simd128"))]
mod wasm_simd128_tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_test::*;

    // No `wasm_bindgen_test_configure!` call: Node.js is the default test
    // runner for `wasm-pack test --node` / `wasm-bindgen-test-runner`; the
    // macro is only needed to opt into `run_in_browser` / `run_in_worker`.

    /// Dims exercising: empty, 1-3 elements (below one v128 lane), the
    /// 4/8-lane boundaries, non-multiples-of-4, and production embedding
    /// sizes (384/768/1024).
    const CORRECTNESS_DIMS: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 9, 384, 768, 1000, 1023, 1024];

    fn random_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.gen_range(-10.0f32..10.0)).collect()
    }

    /// Combined absolute+relative tolerance (numpy `allclose`-style:
    /// `|a - b| <= atol + rtol * max(|a|, |b|)`). A flat `< 1e-6` absolute
    /// bound is not achievable here: the scalar path sums with 4 interleaved
    /// f32 accumulators and the simd128 path with 2 `v128` (8-wide)
    /// accumulators plus a horizontal-sum tail, so the two reduction trees
    /// visit terms in different orders. f32 addition isn't associative, so
    /// reordering shifts rounding by a few ULPs — expected floating-point
    /// behavior, not a correctness bug, and it grows with the accumulated
    /// magnitude (up to dim=1024 terms here). Observed on this exact grid:
    /// max relative error ~1.1e-7 (essentially f32 machine epsilon). rtol
    /// here is ~90x that observed margin — tight enough to catch a real bug
    /// (wrong lane math, dropped remainder) which would produce relative
    /// error orders of magnitude larger, not machine-epsilon-scale drift.
    fn assert_close(op: &str, dim: usize, scalar: f32, simd: f32) {
        const ATOL: f32 = 1e-5;
        const RTOL: f32 = 1e-5;
        let diff = (scalar - simd).abs();
        let bound = ATOL + RTOL * scalar.abs().max(simd.abs());
        assert!(
            diff <= bound,
            "{op} dim={dim}: scalar={scalar} simd128={simd} diff={diff} bound={bound}"
        );
    }

    #[wasm_bindgen_test]
    fn l2_squared_matches_scalar_across_dims() {
        let mut rng = StdRng::seed_from_u64(42);
        for &dim in CORRECTNESS_DIMS {
            let a = random_vec(&mut rng, dim);
            let b = random_vec(&mut rng, dim);
            let scalar = scalar_l2_squared(&a, &b);
            let simd = wasm_simd128_l2_squared(&a, &b);
            assert_close("l2_squared", dim, scalar, simd);
        }
    }

    #[wasm_bindgen_test]
    fn inner_product_matches_scalar_across_dims() {
        let mut rng = StdRng::seed_from_u64(7);
        for &dim in CORRECTNESS_DIMS {
            let a = random_vec(&mut rng, dim);
            let b = random_vec(&mut rng, dim);
            let scalar = scalar_inner_product(&a, &b);
            let simd = wasm_simd128_inner_product(&a, &b);
            assert_close("inner_product", dim, scalar, simd);
        }
    }

    #[wasm_bindgen_test]
    fn identical_vectors_are_zero_distance() {
        let a = vec![1.0f32; 384];
        assert!(wasm_simd128_l2_squared(&a, &a) < 1e-10);
    }

    #[wasm_bindgen_test]
    #[should_panic(expected = "distance vectors must have equal lengths")]
    fn l2_rejects_mismatched_lengths_before_unsafe_load() {
        let _ = wasm_simd128_l2_squared(&[1.0; 8], &[1.0; 4]);
    }

    #[wasm_bindgen_test]
    #[should_panic(expected = "distance vectors must have equal lengths")]
    fn inner_product_rejects_mismatched_lengths_before_unsafe_load() {
        let _ = wasm_simd128_inner_product(&[1.0; 8], &[1.0; 4]);
    }

    /// A/B timing: geometric mean of scalar/simd128 wall time across
    /// {l2_squared, inner_product} x {384, 768, 1024} dims, printed for the
    /// PR's A/B table. Not a pass/fail assertion — the speedup gate is
    /// evaluated from this test's output, pre-registered in PR_BODY.md
    /// before this test was run. MUST be run with `--release` (`wasm-pack
    /// test --node --release`): the default debug/unopt profile leaves
    /// `unsafe` intrinsic calls uninlined and makes simd128 look *slower*
    /// than scalar — verified: debug gave geomean 0.12x, release 1.2x+ on
    /// the identical source.
    #[wasm_bindgen_test]
    fn ab_timing_report() {
        const DIMS: &[usize] = &[384, 768, 1024];
        const WARMUP_ITERS: usize = 2_000;
        const TIMED_ITERS: usize = 300_000;
        const ROUNDS: usize = 5;

        let mut rng = StdRng::seed_from_u64(1234);
        let mut log_ratio_sum = 0.0f64;
        let mut cell_count = 0u32;

        for &dim in DIMS {
            let vectors: Vec<(Vec<f32>, Vec<f32>)> = (0..64)
                .map(|_| (random_vec(&mut rng, dim), random_vec(&mut rng, dim)))
                .collect();

            let ops: [(&str, fn(&[f32], &[f32]) -> f32, fn(&[f32], &[f32]) -> f32); 2] = [
                (
                    "l2_squared",
                    scalar_l2_squared as fn(&[f32], &[f32]) -> f32,
                    wasm_simd128_l2_squared as fn(&[f32], &[f32]) -> f32,
                ),
                (
                    "inner_product",
                    scalar_inner_product as fn(&[f32], &[f32]) -> f32,
                    wasm_simd128_inner_product as fn(&[f32], &[f32]) -> f32,
                ),
            ];

            for (op_name, scalar_fn, simd_fn) in ops {
                // Warmup both paths (JIT/tiering, cache warm-up).
                let mut sink = 0.0f32;
                for i in 0..WARMUP_ITERS {
                    let (a, b) = &vectors[i % vectors.len()];
                    sink += scalar_fn(a, b) + simd_fn(a, b);
                }
                core::hint::black_box(sink);

                let mut scalar_rounds = [0.0f64; ROUNDS];
                let mut simd_rounds = [0.0f64; ROUNDS];

                for r in 0..ROUNDS {
                    let t0 = performance_now();
                    let mut sink = 0.0f32;
                    for i in 0..TIMED_ITERS {
                        let (a, b) = &vectors[i % vectors.len()];
                        sink += scalar_fn(a, b);
                    }
                    scalar_rounds[r] = performance_now() - t0;
                    core::hint::black_box(sink);

                    let t0 = performance_now();
                    let mut sink = 0.0f32;
                    for i in 0..TIMED_ITERS {
                        let (a, b) = &vectors[i % vectors.len()];
                        sink += simd_fn(a, b);
                    }
                    simd_rounds[r] = performance_now() - t0;
                    core::hint::black_box(sink);
                }

                let scalar_ms = median(&mut scalar_rounds);
                let simd_ms = median(&mut simd_rounds);

                let ratio = if simd_ms > 0.0 {
                    scalar_ms / simd_ms
                } else {
                    f64::NAN
                };
                log_ratio_sum += ratio.ln();
                cell_count += 1;

                web_sys_console_log(&format!(
                    "AB_675 op={op_name} dim={dim} iters={TIMED_ITERS} rounds={ROUNDS} scalar_ms={scalar_ms:.4} simd128_ms={simd_ms:.4} speedup={ratio:.4}"
                ));
            }
        }

        let geomean = (log_ratio_sum / cell_count as f64).exp();
        web_sys_console_log(&format!("AB_675 geomean_speedup={geomean:.4}"));
    }

    /// In-place median (sorts `xs`).
    fn median(xs: &mut [f64]) -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    }

    /// `performance.now()` — sub-millisecond monotonic clock, available as a
    /// global under both Node (16+) and browsers. `js_sys`/`web_sys` only
    /// expose it hung off `window`/`Performance` objects that don't exist
    /// under Node's `--node` test runner, so bind the global directly.
    fn performance_now() -> f64 {
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = performance, js_name = now)]
            fn now() -> f64;
        }
        now()
    }

    /// Minimal `console.log` shim so timing output shows up in
    /// `wasm-pack test --node` output without pulling in `web-sys`'s full
    /// `console` feature for this dev-only path.
    fn web_sys_console_log(msg: &str) {
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = console, js_name = log)]
            fn log(s: &str);
        }
        log(msg);
    }
}
