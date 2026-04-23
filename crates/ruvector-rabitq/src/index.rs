//! RaBitQ flat indexes — four search backends behind one trait:
//!
//!   - [`FlatF32Index`]         — exact f32 brute force baseline.
//!   - [`RabitqIndex`]          — symmetric 1-bit scan, Charikar-style estimator,
//!                                no reranking. Lowest memory.
//!   - [`RabitqPlusIndex`]      — symmetric scan + exact f32 rerank on the top
//!                                `rerank_factor·k` candidates.
//!   - [`RabitqAsymIndex`]      — asymmetric scan (query in f32, db 1-bit),
//!                                optional rerank. Tighter IP estimator at the
//!                                cost of O(D) arithmetic per candidate vs
//!                                O(D/64) popcount.
//!
//! All share the [`AnnIndex`] trait.
//!
//! ## Optimizations
//!
//! - **Top-k via bounded binary heap**: O(n log k) per query instead of
//!   O(n log n), matters at n ≥ 10 000.
//! - **NaN-safe scoring**: `f32::total_cmp` means a rogue NaN never panics the
//!   search — it sorts to the back.
//! - **Padding-safe popcount**: [`BinaryCode::masked_xnor_popcount`] correctly
//!   handles `D % 64 != 0`.
//! - **Query rotated once**: both symmetric and asymmetric paths compute
//!   `q_rot = P · q̂` once per search, amortised across n.
//! - **Honest memory accounting**: `RabitqIndex` does *not* keep the originals;
//!   only `RabitqPlusIndex` and `RabitqAsymIndex` do (and only when rerank > 0).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::{RabitqError, Result};
use crate::quantize::BinaryCode;
use crate::rotation::{normalize_inplace, RandomRotation};

/// A single search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    /// Estimated or exact squared-L2 distance.
    pub score: f32,
}

/// Common trait so benchmarks can swap backends transparently.
pub trait AnnIndex: Send + Sync {
    fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<()>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn dim(&self) -> usize;
    /// Actual bytes allocated by the index (all of: originals + codes +
    /// rotation matrix + bookkeeping). Honest — no hidden `Vec<f32>`.
    fn memory_bytes(&self) -> usize;
}

// ── score ordering helpers (NaN-safe) ───────────────────────────────────────

#[inline]
fn cmp_score_asc(a: f32, b: f32) -> Ordering {
    // total_cmp sorts NaN to the back of ascending order — safer than
    // partial_cmp().unwrap() which panics.
    a.total_cmp(&b)
}

/// Bounded max-heap that keeps the smallest `k` scores seen so far. Entries
/// carry both the external `id` and the internal `pos` (row-index in the SoA
/// storage) so rerank paths can fetch `originals[pos]` in O(1) after the
/// scan.
struct TopK {
    k: usize,
    // Max-heap by score so the worst of the top-k is at the top and can be
    // evicted when something smaller comes along.
    heap: BinaryHeap<HeapEntry>,
}

/// Heap entry — ordered by score ascending (so BinaryHeap's max acts as our
/// "worst candidate in the top-k" evictee).
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    id: usize,
    score: f32,
    pos: u32,
}
impl PartialEq for HeapEntry {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score && self.id == o.id
    }
}
impl Eq for HeapEntry {}
impl Ord for HeapEntry {
    fn cmp(&self, o: &Self) -> Ordering {
        self.score.total_cmp(&o.score).then(self.id.cmp(&o.id))
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            heap: BinaryHeap::with_capacity(k.max(1) + 1),
        }
    }
    #[inline]
    fn push(&mut self, id: usize, score: f32) {
        self.push_raw(id, score, 0);
    }
    /// Variant that carries the candidate's row-position through the heap.
    #[inline]
    fn push_raw(&mut self, id: usize, score: f32, pos: usize) {
        if self.heap.len() < self.k {
            self.heap.push(HeapEntry {
                id,
                score,
                pos: pos as u32,
            });
            return;
        }
        let worst = self.heap.peek().unwrap().score;
        if score.total_cmp(&worst) == Ordering::Less {
            self.heap.pop();
            self.heap.push(HeapEntry {
                id,
                score,
                pos: pos as u32,
            });
        }
    }
    fn into_sorted_asc(self) -> Vec<SearchResult> {
        let mut v: Vec<SearchResult> = self
            .heap
            .into_iter()
            .map(|e| SearchResult {
                id: e.id,
                score: e.score,
            })
            .collect();
        v.sort_unstable_by(|a, b| cmp_score_asc(a.score, b.score));
        v
    }
    /// Returns (pos, id, score) triples sorted ascending by score. Used by
    /// the rerank path to look up the original f32 vector in O(1).
    fn into_sorted_with_pos(self) -> Vec<(u32, u32, f32)> {
        let mut v: Vec<(u32, u32, f32)> = self
            .heap
            .into_iter()
            .map(|e| (e.pos, e.id as u32, e.score))
            .collect();
        v.sort_unstable_by(|a, b| cmp_score_asc(a.2, b.2));
        v
    }
}

// ── Variant A: naive f32 brute-force ─────────────────────────────────────────

pub struct FlatF32Index {
    dim: usize,
    vectors: Vec<(usize, Vec<f32>)>,
}

impl FlatF32Index {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vectors: Vec::new(),
        }
    }
}

#[inline]
fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum()
}

impl AnnIndex for FlatF32Index {
    fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        self.vectors.push((id, vector));
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if self.vectors.is_empty() {
            return Err(RabitqError::EmptyIndex);
        }
        if query.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let k_eff = k.min(self.vectors.len());
        let mut top = TopK::new(k_eff);
        for (id, v) in &self.vectors {
            top.push(*id, sq_l2(query, v));
        }
        Ok(top.into_sorted_asc())
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn memory_bytes(&self) -> usize {
        // each Vec<f32> is dim*4 bytes; the (usize, Vec<f32>) pair adds
        // 16 bytes (id + Vec header) per entry.
        self.vectors.len() * (self.dim * 4 + 16)
    }
}

// ── Variant B: RaBitQ symmetric scan (no originals, no rerank) ──────────────

/// Pure 1-bit index. Stores the rotation matrix + binary codes in
/// **struct-of-arrays** layout for cache-friendly scanning:
///
/// - `ids:   Vec<u32>`   — one u32 per vector (4 B)
/// - `norms: Vec<f32>`   — one f32 per vector (4 B)
/// - `packed: Vec<u64>`  — flat row-major binary codes (`n_words` per row)
///
/// This replaces the previous `Vec<(usize, BinaryCode)>` where each
/// `BinaryCode` owned its own heap-allocated `Vec<u64>`. The indirection
/// forced a pointer chase per candidate; the SoA version walks contiguous
/// memory and is ~2× faster on the symmetric scan at scale.
///
/// A precomputed cos-lookup table replaces the `.cos()` call in the
/// estimator: since `agreement ∈ [0, D]` takes at most D+1 distinct values,
/// a length-(D+1) f32 table gives the cos answer in one indexed load.
/// At D=128 that's a 516 B table — fits in L1.
///
/// Does NOT keep the original f32 vectors — `RabitqPlusIndex` /
/// `RabitqAsymIndex` own that storage when rerank is desired.
pub struct RabitqIndex {
    dim: usize,
    n_words: usize,
    rotation: RandomRotation,
    ids: Vec<u32>,
    norms: Vec<f32>,
    /// Flat row-major binary codes. Row i lives at `packed[i*n_words ..][..n_words]`.
    packed: Vec<u64>,
    /// Masked-popcount mask for the last word (zero padding bits).
    last_word_mask: u64,
    /// cos(π · (1 − B/D)) for B ∈ {0, 1, ..., D}.
    cos_lut: Vec<f32>,
}

fn build_last_word_mask(dim: usize) -> u64 {
    let n_words = (dim + 63) / 64;
    if n_words == 0 {
        return 0;
    }
    let valid_bits = dim - 64 * (n_words - 1);
    if valid_bits == 64 {
        !0u64
    } else {
        !0u64 << (64 - valid_bits)
    }
}

fn build_cos_lut(dim: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    let d = dim as f32;
    (0..=dim)
        .map(|b| (PI * (1.0 - b as f32 / d)).cos())
        .collect()
}

impl RabitqIndex {
    pub fn new(dim: usize, seed: u64) -> Self {
        let n_words = (dim + 63) / 64;
        Self {
            dim,
            n_words,
            rotation: RandomRotation::random(dim, seed),
            ids: Vec::new(),
            norms: Vec::new(),
            packed: Vec::new(),
            last_word_mask: build_last_word_mask(dim),
            cos_lut: build_cos_lut(dim),
        }
    }

    /// Encode a raw vector and return the resulting `BinaryCode` (useful for
    /// callers who want to hand codes to another index or serialise them
    /// separately). The index mirrors the bits into its SoA storage.
    pub fn encode_vector(&self, v: &[f32]) -> BinaryCode {
        let norm: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let mut unit = v.to_vec();
        normalize_inplace(&mut unit);
        let rotated = self.rotation.apply(&unit);
        BinaryCode::encode(&rotated, norm)
    }

    /// Encode a query — same math as `encode_vector`, returns just the
    /// packed words (the caller already has `q_norm` from
    /// [`Self::prepare_query_f32`] if they need it).
    pub fn encode_query_packed(&self, q: &[f32]) -> (Vec<u64>, f32) {
        let norm: f32 = q.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let mut unit = q.to_vec();
        normalize_inplace(&mut unit);
        let rotated = self.rotation.apply(&unit);
        // Pack MSB-first, same as BinaryCode::encode.
        let mut words = vec![0u64; self.n_words];
        for (i, &v) in rotated.iter().enumerate() {
            if v >= 0.0 {
                words[i / 64] |= 1u64 << (63 - (i % 64));
            }
        }
        (words, norm.max(1e-10))
    }

    /// Retained for the BinaryCode-shaped public surface. Thin wrapper
    /// around [`Self::encode_query_packed`] that boxes the result.
    pub fn encode_query(&self, q: &[f32]) -> BinaryCode {
        let (words, norm) = self.encode_query_packed(q);
        BinaryCode {
            words,
            norm,
            dim: self.dim,
        }
    }

    /// Prepare a query for the asymmetric estimator — returns the rotated
    /// *unit* query (f32) and the original norm. Done once per search.
    pub fn prepare_query_f32(&self, q: &[f32]) -> (Vec<f32>, f32) {
        let norm: f32 = q.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let mut unit = q.to_vec();
        normalize_inplace(&mut unit);
        (self.rotation.apply(&unit), norm.max(1e-10))
    }

    /// Bytes used by the binary codes alone (not counting the rotation matrix).
    pub fn codes_bytes(&self) -> usize {
        // SoA: n_entries * (u32 id + f32 norm + n_words u64 code) = 8 + n_words*8 per row.
        self.ids.len() * 8 + self.packed.len() * 8 + self.cos_lut.len() * 4
    }

    pub fn rotation(&self) -> &RandomRotation {
        &self.rotation
    }

    /// Reconstruct the old `(id, BinaryCode)` view — O(n) allocation, for
    /// back-compat with callers that still want boxed codes. Prefer the SoA
    /// accessors ([`Self::ids`], [`Self::norms`], [`Self::packed`]) at hot
    /// paths.
    pub fn codes_materialised(&self) -> Vec<(usize, BinaryCode)> {
        (0..self.ids.len())
            .map(|i| {
                let s = i * self.n_words;
                let words = self.packed[s..s + self.n_words].to_vec();
                (
                    self.ids[i] as usize,
                    BinaryCode {
                        words,
                        norm: self.norms[i],
                        dim: self.dim,
                    },
                )
            })
            .collect()
    }

    /// SoA accessors — stable contiguous slices for hot-loop consumers.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
    pub fn norms(&self) -> &[f32] {
        &self.norms
    }
    pub fn packed(&self) -> &[u64] {
        &self.packed
    }
    pub fn n_words(&self) -> usize {
        self.n_words
    }
    pub fn cos_lut(&self) -> &[f32] {
        &self.cos_lut
    }

    /// Core SoA symmetric scan. Walks all n codes against a pre-packed
    /// query, returns the heap-reduced top-k. Exposed so the other index
    /// variants (`RabitqPlusIndex`) can share the tuned loop.
    #[inline]
    pub(crate) fn symmetric_scan_topk(
        &self,
        q_packed: &[u64],
        q_norm: f32,
        k: usize,
    ) -> Vec<(u32, u32, f32)> {
        // Returns (pos, id, score) so rerank callers can map back to `originals[pos]`.
        let mut top = TopK::new(k.min(self.ids.len()));
        let n_words = self.n_words;
        let mask = self.last_word_mask;
        let d = self.dim as f32;
        let q_sq = q_norm * q_norm;
        let lut = &self.cos_lut;

        // Unrolled walk with manual prefetch-friendly stride. LLVM can already
        // do most of this; the important part is the flat `packed` slice — no
        // per-candidate indirection.
        let n = self.ids.len();
        let p = self.packed.as_ptr();
        let aligned = mask == !0u64; // dim % 64 == 0
        for i in 0..n {
            // SAFETY: p is valid for `n * n_words` u64 reads. Using ptr offsets
            // avoids the bounds-check in the inner loop.
            let base = unsafe { p.add(i * n_words) };
            let mut agree: u32 = 0;
            if aligned && n_words == 2 {
                // D=128 fast path: 2 popcounts, no last-word mask needed.
                unsafe {
                    agree = (!(*base ^ q_packed[0])).count_ones()
                        + (!(*base.add(1) ^ q_packed[1])).count_ones();
                }
            } else if aligned {
                // Aligned but more words — skip the mask AND on the last word.
                unsafe {
                    for w in 0..n_words {
                        agree += (!(*base.add(w) ^ q_packed[w])).count_ones();
                    }
                }
            } else {
                // Unaligned: mask the last word's padding bits off.
                unsafe {
                    for w in 0..n_words - 1 {
                        agree += (!(*base.add(w) ^ q_packed[w])).count_ones();
                    }
                    agree +=
                        (!(*base.add(n_words - 1) ^ q_packed[n_words - 1]) & mask).count_ones();
                }
            }
            // cos LUT replaces the `.cos()` call — one indexed load.
            let est_cos = unsafe { *lut.get_unchecked(agree as usize) };
            let x_norm = self.norms[i];
            let est_ip = q_norm * x_norm * est_cos;
            let score = q_sq + x_norm * x_norm - 2.0 * est_ip;
            // ignoring d here — already baked into the LUT indices.
            let _ = d;
            top.push_raw(self.ids[i] as usize, score, i);
        }
        top.into_sorted_with_pos()
    }
}

impl AnnIndex for RabitqIndex {
    fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        if vector.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        // Inline-encode directly into the SoA buffers — no intermediate BinaryCode.
        let norm: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let mut unit = vector;
        normalize_inplace(&mut unit);
        let rotated = self.rotation.apply(&unit);
        let start = self.packed.len();
        self.packed.resize(start + self.n_words, 0);
        let slot = &mut self.packed[start..start + self.n_words];
        for (i, &v) in rotated.iter().enumerate() {
            if v >= 0.0 {
                slot[i / 64] |= 1u64 << (63 - (i % 64));
            }
        }
        self.ids.push(id as u32);
        self.norms.push(norm);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if self.ids.is_empty() {
            return Err(RabitqError::EmptyIndex);
        }
        if query.len() != self.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let (q_packed, q_norm) = self.encode_query_packed(query);
        let results = self.symmetric_scan_topk(&q_packed, q_norm, k);
        Ok(results
            .into_iter()
            .map(|(_, id, score)| SearchResult {
                id: id as usize,
                score,
            })
            .collect())
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn memory_bytes(&self) -> usize {
        self.rotation.bytes() + self.codes_bytes()
    }
}

// ── Variant C: symmetric scan + exact f32 rerank ─────────────────────────────

/// Owns an inner [`RabitqIndex`] plus the original f32 vectors. Symmetric
/// 1-bit scan produces candidate IDs, exact f32 rerank picks the winners.
pub struct RabitqPlusIndex {
    inner: RabitqIndex,
    originals: Vec<Vec<f32>>, // parallel to inner.codes; indexed by ID-order position
    rerank_factor: usize,
}

impl RabitqPlusIndex {
    pub fn new(dim: usize, seed: u64, rerank_factor: usize) -> Self {
        Self {
            inner: RabitqIndex::new(dim, seed),
            originals: Vec::new(),
            rerank_factor: rerank_factor.max(1),
        }
    }

    pub fn rerank_factor(&self) -> usize {
        self.rerank_factor
    }
    pub fn set_rerank_factor(&mut self, f: usize) {
        self.rerank_factor = f.max(1);
    }

    /// Search with a per-call rerank factor override. Same body as
    /// [`AnnIndex::search`] but takes `rerank_factor` as a parameter
    /// instead of reading the field, so callers can tune recall/cost
    /// per query without mutating shared state.
    ///
    /// Added for `ruvector-rulake` (ADR-155): federated fan-out divides
    /// the global rerank factor across K shards (`per_shard = max(floor,
    /// global / K)`) so K-shard federation stops paying K× the rerank
    /// cost. The field `self.rerank_factor` remains the default used by
    /// plain `search`; nothing about the stored index changes.
    pub fn search_with_rerank(
        &self,
        query: &[f32],
        k: usize,
        rerank_factor: usize,
    ) -> Result<Vec<SearchResult>> {
        if self.inner.ids.is_empty() {
            return Err(RabitqError::EmptyIndex);
        }
        if query.len() != self.inner.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.inner.dim,
                actual: query.len(),
            });
        }
        let rf = rerank_factor.max(1);
        let n = self.inner.ids.len();
        let candidates = k.saturating_mul(rf).max(k).min(n);

        let (q_packed, q_norm) = self.inner.encode_query_packed(query);
        let cand = self
            .inner
            .symmetric_scan_topk(&q_packed, q_norm, candidates);

        let k_eff = k.min(cand.len());
        let mut top = TopK::new(k_eff);
        for (pos, id, _score) in &cand {
            let v = &self.originals[*pos as usize];
            top.push(*id as usize, sq_l2(query, v));
        }
        Ok(top.into_sorted_asc())
    }
}

impl AnnIndex for RabitqPlusIndex {
    fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        self.inner.add(id, vector.clone())?;
        self.originals.push(vector);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if self.inner.ids.is_empty() {
            return Err(RabitqError::EmptyIndex);
        }
        if query.len() != self.inner.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.inner.dim,
                actual: query.len(),
            });
        }
        let n = self.inner.ids.len();
        let candidates = k.saturating_mul(self.rerank_factor).max(k).min(n);

        // Binary-code scan via the tuned SoA loop.
        let (q_packed, q_norm) = self.inner.encode_query_packed(query);
        let cand = self
            .inner
            .symmetric_scan_topk(&q_packed, q_norm, candidates);

        // Exact rerank on the candidate set — `pos` is the row index.
        let k_eff = k.min(cand.len());
        let mut top = TopK::new(k_eff);
        for (pos, id, _score) in &cand {
            let v = &self.originals[*pos as usize];
            top.push(*id as usize, sq_l2(query, v));
        }
        Ok(top.into_sorted_asc())
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes() + self.originals.len() * (self.inner.dim * 4 + 24)
    }
}

// ── Variant D: asymmetric RaBitQ-2024 estimator ─────────────────────────────

/// Asymmetric scan: query stays in f32 (rotated once per search), database is
/// 1-bit. Uses [`BinaryCode::estimated_sq_distance_asymmetric`] — the RaBitQ
/// SIGMOD 2024-style IP estimator. Optional exact rerank on top-k·factor.
pub struct RabitqAsymIndex {
    inner: RabitqIndex,
    originals: Vec<Vec<f32>>, // needed only if rerank_factor > 1
    rerank_factor: usize,
    store_originals: bool,
}

impl RabitqAsymIndex {
    /// `rerank_factor = 1` disables rerank (purest 1-bit memory footprint).
    /// Any value > 1 stores the originals alongside the codes for exact rerank.
    pub fn new(dim: usize, seed: u64, rerank_factor: usize) -> Self {
        let rf = rerank_factor.max(1);
        Self {
            inner: RabitqIndex::new(dim, seed),
            originals: Vec::new(),
            rerank_factor: rf,
            store_originals: rf > 1,
        }
    }
}

impl AnnIndex for RabitqAsymIndex {
    fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        if self.store_originals {
            self.originals.push(vector.clone());
        }
        self.inner.add(id, vector)
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if self.inner.ids.is_empty() {
            return Err(RabitqError::EmptyIndex);
        }
        if query.len() != self.inner.dim {
            return Err(RabitqError::DimensionMismatch {
                expected: self.inner.dim,
                actual: query.len(),
            });
        }
        let n = self.inner.ids.len();
        let candidates = k.saturating_mul(self.rerank_factor).max(k).min(n);

        let (q_rot_unit, q_norm) = self.inner.prepare_query_f32(query);

        // Asymmetric scan — O(D) per candidate. Walks SoA directly so the
        // memory footprint is a flat `n × n_words` slab instead of n heap
        // allocations.
        let d = self.inner.dim;
        let n_words = self.inner.n_words;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();
        let q_sq = q_norm * q_norm;

        let mut top_cand = TopK::new(candidates);
        for i in 0..n {
            let base = i * n_words;
            let slot = &self.inner.packed[base..base + n_words];
            let mut ip = 0.0f32;
            for (idx, &q_i) in q_rot_unit.iter().enumerate() {
                let bit_set = (slot[idx / 64] >> (63 - (idx % 64))) & 1 == 1;
                ip += if bit_set { q_i } else { -q_i };
            }
            let unit_ip = ip * inv_sqrt_d;
            let x_norm = self.inner.norms[i];
            let est_ip = q_norm * x_norm * unit_ip;
            let score = q_sq + x_norm * x_norm - 2.0 * est_ip;
            top_cand.push_raw(self.inner.ids[i] as usize, score, i);
        }
        let cand = top_cand.into_sorted_with_pos();

        if self.rerank_factor <= 1 || !self.store_originals {
            let k_eff = k.min(cand.len());
            let mut out: Vec<SearchResult> = cand
                .into_iter()
                .take(k_eff)
                .map(|(_, id, score)| SearchResult {
                    id: id as usize,
                    score,
                })
                .collect();
            out.sort_unstable_by(|a, b| cmp_score_asc(a.score, b.score));
            return Ok(out);
        }

        let k_eff = k.min(cand.len());
        let mut top = TopK::new(k_eff);
        for (pos, id, _) in &cand {
            let v = &self.originals[*pos as usize];
            top.push(*id as usize, sq_l2(query, v));
        }
        Ok(top.into_sorted_asc())
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn memory_bytes(&self) -> usize {
        let mut b = self.inner.memory_bytes();
        if self.store_originals {
            b += self.originals.len() * (self.inner.dim * 4 + 24);
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uniform random data — only use for non-recall tests.
    fn make_dataset(n: usize, d: usize, seed: u64) -> Vec<(usize, Vec<f32>)> {
        use rand::{Rng as _, SeedableRng as _};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                (i, v)
            })
            .collect()
    }

    /// Gaussian-cluster data. See `main.rs` for the equivalent.
    fn make_clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::{Rng as _, SeedableRng as _};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let centroids: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| {
                (0..d)
                    .map(|_| rng.gen::<f32>() * 4.0 - 2.0)
                    .collect::<Vec<_>>()
            })
            .collect();
        (0..n)
            .map(|_| {
                let c = &centroids[rng.gen_range(0..n_clusters)];
                c.iter()
                    .map(|&x| x + (rng.gen::<f32>() - 0.5) * 0.3)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn flat_f32_returns_exact_nn() {
        let d = 64;
        let mut idx = FlatF32Index::new(d);
        let data = make_dataset(200, d, 1);
        for (id, v) in &data {
            idx.add(*id, v.clone()).unwrap();
        }
        let query = &data[7].1;
        let results = idx.search(query, 1).unwrap();
        assert_eq!(results[0].id, 7);
        assert!(results[0].score < 1e-6);
    }

    /// Regression test for the 20%-but-named-70% test. The no-rerank 1-bit
    /// scan on D=128 clustered data is expected to sit in the 20–45% range;
    /// we assert the weaker "better than random chance" bound.
    #[test]
    fn rabitq_recall_above_random() {
        let d = 128;
        let n = 1000;
        let nq = 100;
        let all_data = make_clustered(n + nq, d, 20, 42);
        let (db_vecs, query_vecs) = all_data.split_at(n);
        let data: Vec<(usize, Vec<f32>)> = db_vecs.iter().cloned().enumerate().collect();
        let queries: Vec<Vec<f32>> = query_vecs.to_vec();

        let mut exact = FlatF32Index::new(d);
        let mut idx = RabitqIndex::new(d, 42);
        for (id, v) in &data {
            exact.add(*id, v.clone()).unwrap();
            idx.add(*id, v.clone()).unwrap();
        }

        let k = 10;
        let mut hits = 0usize;
        for q in &queries {
            let e: std::collections::HashSet<usize> =
                exact.search(q, k).unwrap().iter().map(|r| r.id).collect();
            hits += idx
                .search(q, k)
                .unwrap()
                .iter()
                .filter(|r| e.contains(&r.id))
                .count();
        }
        let recall = hits as f64 / (nq * k) as f64;
        // k/n = 1% is random chance; we want the estimator to beat it by ≥ 10×.
        assert!(
            recall > 0.20,
            "recall@10={:.1}% — not above 20 % baseline",
            recall * 100.0
        );
    }

    #[test]
    fn rabitq_plus_recall_above_90pct() {
        let d = 128;
        let n = 1000;
        let nq = 100;
        let all_data = make_clustered(n + nq, d, 20, 55);
        let (db_vecs, query_vecs) = all_data.split_at(n);
        let data: Vec<(usize, Vec<f32>)> = db_vecs.iter().cloned().enumerate().collect();
        let queries: Vec<Vec<f32>> = query_vecs.to_vec();

        let mut exact = FlatF32Index::new(d);
        let mut idx = RabitqPlusIndex::new(d, 55, 5);
        for (id, v) in &data {
            exact.add(*id, v.clone()).unwrap();
            idx.add(*id, v.clone()).unwrap();
        }
        let k = 10;
        let mut hits = 0usize;
        for q in &queries {
            let e: std::collections::HashSet<usize> =
                exact.search(q, k).unwrap().iter().map(|r| r.id).collect();
            hits += idx
                .search(q, k)
                .unwrap()
                .iter()
                .filter(|r| e.contains(&r.id))
                .count();
        }
        let recall = hits as f64 / (nq * k) as f64;
        assert!(
            recall > 0.90,
            "rerank×5 recall@10={:.1}% < 90 %",
            recall * 100.0
        );
    }

    /// Asymmetric (f32 query × 1-bit db) should *equal or beat* symmetric on
    /// the same substrate because it uses the unrotated-query magnitude
    /// directly. We assert: asym_recall ≥ sym_recall − 2% (noise-margin).
    #[test]
    fn asymmetric_meets_or_beats_symmetric() {
        let d = 128;
        let n = 1000;
        let nq = 100;
        let all_data = make_clustered(n + nq, d, 20, 77);
        let (db_vecs, query_vecs) = all_data.split_at(n);
        let data: Vec<(usize, Vec<f32>)> = db_vecs.iter().cloned().enumerate().collect();
        let queries: Vec<Vec<f32>> = query_vecs.to_vec();

        let mut exact = FlatF32Index::new(d);
        let mut sym = RabitqIndex::new(d, 77);
        let mut asym = RabitqAsymIndex::new(d, 77, 1);
        for (id, v) in &data {
            exact.add(*id, v.clone()).unwrap();
            sym.add(*id, v.clone()).unwrap();
            asym.add(*id, v.clone()).unwrap();
        }
        let k = 10;
        let mut sh = 0usize;
        let mut ah = 0usize;
        for q in &queries {
            let e: std::collections::HashSet<usize> =
                exact.search(q, k).unwrap().iter().map(|r| r.id).collect();
            sh += sym
                .search(q, k)
                .unwrap()
                .iter()
                .filter(|r| e.contains(&r.id))
                .count();
            ah += asym
                .search(q, k)
                .unwrap()
                .iter()
                .filter(|r| e.contains(&r.id))
                .count();
        }
        let sr = sh as f64 / (nq * k) as f64;
        let ar = ah as f64 / (nq * k) as f64;
        eprintln!("sym={:.1}%  asym={:.1}%", sr * 100.0, ar * 100.0);
        // Asymmetric should not drop materially below symmetric.
        assert!(ar + 0.02 >= sr, "asymmetric regressed vs symmetric");
    }

    /// At D=100 the popcount padding bug would bias estimated distances toward
    /// zero (padding bits always match). Confirm the masked path keeps
    /// recall > 0.15 (above random=1%).
    #[test]
    fn recall_holds_at_non_aligned_dim() {
        let d = 100;
        let n = 500;
        let nq = 50;
        let all_data = make_clustered(n + nq, d, 15, 17);
        let (db_vecs, query_vecs) = all_data.split_at(n);
        let data: Vec<(usize, Vec<f32>)> = db_vecs.iter().cloned().enumerate().collect();
        let queries: Vec<Vec<f32>> = query_vecs.to_vec();

        let mut exact = FlatF32Index::new(d);
        let mut idx = RabitqPlusIndex::new(d, 17, 5);
        for (id, v) in &data {
            exact.add(*id, v.clone()).unwrap();
            idx.add(*id, v.clone()).unwrap();
        }
        let k = 10;
        let mut hits = 0usize;
        for q in &queries {
            let e: std::collections::HashSet<usize> =
                exact.search(q, k).unwrap().iter().map(|r| r.id).collect();
            hits += idx
                .search(q, k)
                .unwrap()
                .iter()
                .filter(|r| e.contains(&r.id))
                .count();
        }
        let r = hits as f64 / (nq * k) as f64;
        assert!(r > 0.80, "D=100 rerank×5 recall={:.1}% < 80 %", r * 100.0);
    }

    #[test]
    fn nan_query_does_not_panic() {
        let d = 64;
        let mut idx = RabitqIndex::new(d, 42);
        let data = make_dataset(100, d, 3);
        for (id, v) in &data {
            idx.add(*id, v.clone()).unwrap();
        }
        let mut q = data[0].1.clone();
        q[5] = f32::NAN;
        // Must not panic — NaN scores sort to the back via total_cmp.
        let _ = idx.search(&q, 5);
    }

    #[test]
    fn memory_accounting_is_honest() {
        let d = 256;
        let n = 1000;
        let data = make_dataset(n, d, 0);
        let mut flat = FlatF32Index::new(d);
        let mut rq = RabitqIndex::new(d, 0);
        let mut rq_plus = RabitqPlusIndex::new(d, 0, 5);
        for (id, v) in &data {
            flat.add(*id, v.clone()).unwrap();
            rq.add(*id, v.clone()).unwrap();
            rq_plus.add(*id, v.clone()).unwrap();
        }
        let f = flat.memory_bytes();
        let rqb = rq.memory_bytes();
        let rqpb = rq_plus.memory_bytes();
        // Pure 1-bit index MUST report less than flat — it holds no originals.
        assert!(rqb < f, "RabitqIndex {rqb} should be < Flat {f}");
        // Plus-index reports MORE than flat — it holds originals AND codes.
        assert!(
            rqpb > f,
            "RabitqPlusIndex {rqpb} should be > Flat {f} (rerank stores both)"
        );
    }

    #[test]
    fn heap_topk_is_sorted_ascending() {
        let d = 64;
        let mut idx = FlatF32Index::new(d);
        let data = make_dataset(50, d, 2);
        for (id, v) in &data {
            idx.add(*id, v.clone()).unwrap();
        }
        let r = idx.search(&data[0].1, 10).unwrap();
        assert_eq!(r.len(), 10);
        for w in r.windows(2) {
            assert!(w[0].score <= w[1].score, "top-k not ascending: {:?}", r);
        }
    }
}
