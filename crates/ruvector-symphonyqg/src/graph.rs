/// Number of neighbors evaluated per XNOR-popcount pass.
/// 32 → for dim=128 (16 bytes/code) one batch reads 512 bytes (8 cache lines).
/// Aligning the graph degree to a multiple of BATCH_SIZE eliminates wasted
/// SIMD lanes — the core architectural innovation of SymphonyQG (SIGMOD 2025).
pub const BATCH_SIZE: usize = 32;

/// Round `m` up to the nearest multiple of BATCH_SIZE.
#[inline]
pub fn batch_pad(m: usize) -> usize {
    m.div_ceil(BATCH_SIZE) * BATCH_SIZE
}

/// Sentinel value stored in `neighbors` slots that are padding (not real
/// edges). Search must skip these — they exist only to keep the per-vertex
/// adjacency a multiple of `BATCH_SIZE` so the XNOR-popcount loop stays
/// on full SIMD lanes. We use `u32::MAX` because (a) it can never collide
/// with a real vertex id (`n` is bounded well below 2^32 in this crate's
/// target regime), and (b) a single `nb == PADDING_SENTINEL` check is
/// cheap and branch-predictor-friendly. The corresponding code bytes are
/// zero-filled so a stray Hamming-distance read yields a deterministic,
/// well-defined value rather than data from an unrelated vertex.
pub const PADDING_SENTINEL: u32 = u32::MAX;

/// Squared Euclidean distance (no sqrt — monotone for ranking).
#[inline]
pub fn dist_l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Cosine distance: 1 − cos(θ).
#[inline]
pub fn dist_cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 1.0;
    }
    (1.0 - dot / (na * nb)).max(0.0)
}

/// Encode a vector into a packed 1-bit code after random-sign rotation.
///
/// Rotation: y[i] = signs[i] * v[perm[i]]
/// Code bit i = 1 iff y[i] > 0
///
/// This is the randomised sign-flip + permutation approximation to the full
/// random orthogonal rotation used in RaBitQ. For the PoC it produces
/// well-conditioned codes without requiring QR decomposition.
pub fn encode(v: &[f32], signs: &[f32], perm: &[usize]) -> Vec<u8> {
    let dim = v.len();
    let code_bytes = dim / 8;
    let mut code = vec![0u8; code_bytes];
    for i in 0..dim {
        if signs[i] * v[perm[i]] > 0.0 {
            code[i >> 3] |= 1u8 << (i & 7);
        }
    }
    code
}

/// Batch XNOR-popcount estimated cosine distance.
///
/// For each of `n` neighbors whose packed codes are laid out contiguously in
/// `neighbor_codes` (stride = `code_bytes`), returns the estimated distance
/// to `query_code`:
///
///   dist_est ≈ 2 · |{bit positions where q ≠ d}| / (code_bytes * 8)
///
/// **Implementation note** (perf-critical hot path): we read codes as u64
/// chunks where possible (code_bytes ≥ 8) so the popcount runs on full
/// 64-bit words instead of bytes. This matters because the byte-loop form
/// the original implementation used:
///   - Doesn't auto-vectorise to AVX-512 VPOPCNTDQ in our measured asm
///     output (verified with `cargo asm --release` on x86_64-v4 host).
///   - Issues 8× as many `popcnt` instructions as the u64 form, even on
///     scalar targets that ship the SSE4.2 `popcnt` instruction.
///
/// On the common operating point (dim=128 → code_bytes=16) this collapses
/// what was 16 byte-popcounts per neighbour into 2 u64 popcounts. We use
/// raw pointer reads (`std::ptr::read_unaligned`) so the function works
/// regardless of caller alignment — the codes section in `SymphonyGraph`
/// happens to be 4-byte aligned (Vec<u32>-backed), but other callers may
/// pass arbitrary `&[u8]`. Pure safe Rust would force a per-byte loop
/// here through Miri-cleanliness; the unaligned read is the right
/// trade-off (no UB, observable perf win).
///
/// Tail bytes (`code_bytes % 8`) are handled by a final per-byte loop
/// so non-multiple-of-8 code widths (e.g. dim=72 → code_bytes=9) work.
pub fn batch_hamming_dist(
    query_code: &[u8],
    neighbor_codes: &[u8],
    n: usize,
    code_bytes: usize,
) -> Vec<f32> {
    let dim_f = (code_bytes * 8) as f32;
    let q_words = code_bytes / 8;
    let q_tail = code_bytes % 8;
    let q_tail_off = q_words * 8;

    (0..n)
        .map(|i| {
            let c = &neighbor_codes[i * code_bytes..(i + 1) * code_bytes];

            // u64-word path: count_ones on 64 bits at a time. `read_unaligned`
            // is sound for any pointer (no alignment requirement) and
            // bit-identical to a byte-wise reconstruction.
            let mut differing: u32 = 0;
            // SAFETY: q_words * 8 ≤ code_bytes ≤ q.len() and ≤ c.len() by
            // construction; both pointers point to readable byte slices.
            // read_unaligned has no alignment requirement.
            for w in 0..q_words {
                let off = w * 8;
                let q_word: u64 = unsafe {
                    core::ptr::read_unaligned(query_code.as_ptr().add(off) as *const u64)
                };
                let d_word: u64 =
                    unsafe { core::ptr::read_unaligned(c.as_ptr().add(off) as *const u64) };
                differing += (q_word ^ d_word).count_ones();
            }

            // Tail (code_bytes not a multiple of 8) — at most 7 bytes.
            for j in 0..q_tail {
                differing += (query_code[q_tail_off + j] ^ c[q_tail_off + j]).count_ones();
            }

            2.0 * differing as f32 / dim_f
        })
        .collect()
}

/// CSR-style graph with inline 1-bit codes.
///
/// Memory layout (all flat `Vec` for cache locality):
///
/// ```text
///   vectors   : [n × dim]              f32  — full precision for re-ranking
///   self_codes: [n × code_bytes]        u8  — vertex's own 1-bit code (ep seed)
///   blocks    : [n × block_stride_u32] u32 — packed per-vertex adjacency + codes
///   signs     : [dim]                  f32
///   perm      : [dim]                  usize
/// ```
///
/// **Per-vertex block layout** (the SymphonyQG inline-codes-with-adjacency
/// invariant — the central memory-layout claim of the SIGMOD 2025 paper):
///
/// ```text
/// block[v]:
///   [ id_0 id_1 ... id_{m-1} ]                       (m  × u32  =  4·m bytes)
///   [ code_0 || code_1 || ... || code_{m-1} ]        (m  × code_bytes  bytes,
///                                                     stored as u32 words for
///                                                     natural alignment)
/// ```
///
/// Both `neighbors_of(v)` and `nb_codes_of(v)` slice from the **same**
/// per-vertex contiguous region of `blocks`, so the first cache-line touch
/// on a vertex lookup brings in BOTH the IDs and the codes.
///
/// `m` is always a multiple of `BATCH_SIZE` (=32), and `code_bytes = dim/8`.
/// Since `m * code_bytes = 32 · code_bytes` is always a multiple of 4 for any
/// `dim ≥ 8`, the codes section is trivially u32-aligned without padding.
#[derive(Clone)]
pub struct SymphonyGraph {
    pub n: usize,
    pub dim: usize,
    pub code_bytes: usize,
    pub m: usize,

    /// Per-vertex stride in u32 words: `m + m * code_bytes / 4`.
    /// Cached so hot-path slice arithmetic is one mul + one add, not three.
    pub block_stride_u32: usize,

    pub vectors: Vec<f32>,
    /// Packed adjacency + inline codes — ONE allocation, perfect cache
    /// co-location of a vertex's neighbour IDs and their 1-bit codes.
    pub blocks: Vec<u32>,
    /// Self codes: self_codes[v*code_bytes .. (v+1)*code_bytes]
    pub self_codes: Vec<u8>,

    pub signs: Vec<f32>,
    pub perm: Vec<usize>,
    pub entry: usize,
}

impl SymphonyGraph {
    /// Neighbor IDs for vertex `v` — first `m` u32 words of its block.
    #[inline]
    pub fn neighbors_of(&self, v: usize) -> &[u32] {
        let off = v * self.block_stride_u32;
        &self.blocks[off..off + self.m]
    }

    /// Inline 1-bit codes for vertex `v`'s neighbors — the bytes immediately
    /// following the IDs in the same per-vertex block. Cast `&[u32] → &[u8]`
    /// is alignment-safe (u8 has weaker alignment than u32).
    #[inline]
    pub fn nb_codes_of(&self, v: usize) -> &[u8] {
        let off = v * self.block_stride_u32 + self.m;
        let codes_u32_len = (self.m * self.code_bytes) / 4;
        let codes_u32 = &self.blocks[off..off + codes_u32_len];
        // SAFETY: u8 has alignment 1, u32 has alignment 4 — every u32 ptr is
        // a valid u8 ptr. Length scales by 4. No mutation; lifetime tied to
        // self.blocks. This is the canonical pattern used by bytemuck::cast_slice
        // for non-overlapping primitive types of strictly weaker alignment.
        unsafe { core::slice::from_raw_parts(codes_u32.as_ptr() as *const u8, codes_u32.len() * 4) }
    }

    /// Full-precision vector for vertex `v`.
    #[inline]
    pub fn vector_of(&self, v: usize) -> &[f32] {
        &self.vectors[v * self.dim..(v + 1) * self.dim]
    }

    /// 1-bit self-code for vertex `v`.
    #[inline]
    pub fn self_code_of(&self, v: usize) -> &[u8] {
        &self.self_codes[v * self.code_bytes..(v + 1) * self.code_bytes]
    }

    /// Encode a query with this graph's rotation parameters.
    pub fn encode_query(&self, query: &[f32]) -> Vec<u8> {
        encode(query, &self.signs, &self.perm)
    }

    /// Total heap-allocated bytes.
    pub fn memory_bytes(&self) -> usize {
        self.vectors.len() * 4
            + self.blocks.len() * 4
            + self.self_codes.len()
            + self.signs.len() * 4
            + self.perm.len() * 8
    }

    /// Helper: per-vertex stride in u32 words. Inline so build.rs can use it.
    #[inline]
    pub fn block_stride_u32_for(m: usize, code_bytes: usize) -> usize {
        // Codes section is m * code_bytes bytes; we know it's a multiple of 4
        // (m is multiple of BATCH_SIZE=32, code_bytes ≥ 1).
        debug_assert_eq!(
            (m * code_bytes) % 4,
            0,
            "m * code_bytes must be a multiple of 4 for u32 packing"
        );
        m + (m * code_bytes) / 4
    }
}
