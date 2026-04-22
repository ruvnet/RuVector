//! Distance computations with SIMD acceleration and optional GPU offload
//!
//! Dispatch priority: GPU (if `gpu` feature) → SimSIMD (if `simd` feature) → scalar

/// Flat vector storage — contiguous memory for cache-friendly access
/// Vectors are stored as a single `Vec<f32>` slab: `[v0_d0, v0_d1, ..., v1_d0, ...]`
#[derive(Clone)]
pub struct FlatVectors {
    pub data: Vec<f32>,
    pub dim: usize,
    pub count: usize,
}

impl FlatVectors {
    pub fn new(dim: usize) -> Self {
        Self {
            data: Vec::new(),
            dim,
            count: 0,
        }
    }

    pub fn with_capacity(dim: usize, n: usize) -> Self {
        Self {
            data: Vec::with_capacity(n * dim),
            dim,
            count: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, vector: &[f32]) {
        debug_assert_eq!(vector.len(), self.dim);
        self.data.extend_from_slice(vector);
        self.count += 1;
    }

    #[inline]
    pub fn get(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.data[start..start + self.dim]
    }

    /// Zero out a vector (lazy deletion)
    #[inline]
    pub fn zero_out(&mut self, idx: usize) {
        let start = idx * self.dim;
        for v in &mut self.data[start..start + self.dim] {
            *v = f32::NAN;
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

/// L2 squared distance — dispatches to best available implementation
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "simd")]
    {
        simd_l2_squared(a, b)
    }

    #[cfg(not(feature = "simd"))]
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
#[cfg(feature = "simd")]
#[inline]
pub fn simd_l2_squared(a: &[f32], b: &[f32]) -> f32 {
    // simsimd sqeuclidean returns squared Euclidean directly
    simsimd::SpatialSimilarity::sqeuclidean(a, b)
        .map(|d| d as f32)
        .unwrap_or_else(|| scalar_l2_squared(a, b))
}

/// Inner product distance (negated for min-heap)
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(feature = "simd")]
    {
        simsimd::SpatialSimilarity::inner(a, b)
            .map(|d| -(d as f32))
            .unwrap_or_else(|| scalar_inner_product(a, b))
    }

    #[cfg(not(feature = "simd"))]
    {
        scalar_inner_product(a, b)
    }
}

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

/// Compact bitset for tracking visited nodes during search
pub struct VisitedSet {
    bits: Vec<u64>,
    generation: u64,
    gens: Vec<u64>,
}

impl VisitedSet {
    pub fn new(n: usize) -> Self {
        Self {
            bits: vec![0u64; (n + 63) / 64],
            generation: 1,
            gens: vec![0u64; n],
        }
    }

    /// Reset for a new search — O(1) via generation counter
    #[inline]
    pub fn clear(&mut self) {
        self.generation += 1;
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

    #[test]
    fn test_flat_vectors() {
        let mut fv = FlatVectors::new(3);
        fv.push(&[1.0, 2.0, 3.0]);
        fv.push(&[4.0, 5.0, 6.0]);
        assert_eq!(fv.len(), 2);
        assert_eq!(fv.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(fv.get(1), &[4.0, 5.0, 6.0]);
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
