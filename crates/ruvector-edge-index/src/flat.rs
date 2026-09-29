//! M2a: quantized flat scan with exact f32 rerank.
//!
//! Codes live in one fixed-stride arena indexed by iid (`dim` bytes per
//! slot) with a presence bitmap; iid gaps from deleted rows are allowed and
//! skipped (the store's iids start at 1, so slot 0 simply stays empty).
//! Resident cost ≈ `dim + 1/8` bytes per slot; since the store never
//! reuses iids, run [`QuantFlatIndex::compact_ids`] when
//! [`QuantFlatIndex::dead_ratio`] grows.

use crate::bytes::{Reader, Sink};
use crate::error::{DecodeError, EmitError, EncodeError, IndexError, RerankError};
use crate::grow;
use crate::heap::{to_hits, Cand, Hit, TopK};
use crate::metric::validate_vector;
use crate::persist::{
    decode, decode_slice, encode, encode_all, EncodedIndex, IndexChunk, IndexDigest, IndexKind,
};

use crate::quant::QuantParams;
use crate::rerank::{rerank, RerankFetch};

/// Quantized flat index.
#[derive(Debug, Clone)]
pub struct QuantFlatIndex {
    quant: QuantParams,
    max_slots: u32,
    slots: u32,
    live: u32,
    codes: Vec<u8>,
    present: Vec<u64>,
}

fn bit(words: &[u64], i: u32) -> bool {
    words
        .get((i / 64) as usize)
        .is_some_and(|w| w & (1u64 << (i % 64)) != 0)
}

/// `max_slots × dim` must be a valid Vec size on this target (wasm32).
fn check_slots(max_slots: u32, dim: usize) -> Result<(), &'static str> {
    if max_slots == 0 || max_slots == u32::MAX {
        return Err("max_slots out of range");
    }
    match (max_slots as usize).checked_mul(dim) {
        Some(b) if b <= isize::MAX as usize => Ok(()),
        _ => Err("max_slots too large for this target"),
    }
}

impl QuantFlatIndex {
    /// Empty index; `max_slots` bounds `max_iid + 1` (and so memory).
    pub fn new(quant: QuantParams, max_slots: u32) -> Result<Self, IndexError> {
        Self::with_capacity(quant, max_slots, 0)
    }

    /// Empty index with exact room for `n` slots (a fixed reusable arena,
    /// ADR §6.1).
    pub fn with_capacity(quant: QuantParams, max_slots: u32, n: usize) -> Result<Self, IndexError> {
        check_slots(max_slots, quant.dim()).map_err(IndexError::InvalidParams)?;
        let mut s = Self {
            quant,
            max_slots,
            slots: 0,
            live: 0,
            codes: Vec::new(),
            present: Vec::new(),
        };
        s.reserve(n);
        Ok(s)
    }

    /// Quantizer parameters.
    pub fn quant(&self) -> &QuantParams {
        &self.quant
    }
    /// Live vectors.
    pub fn len(&self) -> usize {
        self.live as usize
    }
    /// No live vectors.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
    /// Slots (`max_iid + 1`): what memory is paid for.
    pub fn slots(&self) -> u32 {
        self.slots
    }
    /// Share of slots holding no vector; compact when it grows.
    pub fn dead_ratio(&self) -> f64 {
        if self.slots == 0 {
            0.0
        } else {
            f64::from(self.slots - self.live) / f64::from(self.slots)
        }
    }
    /// Whether `iid` holds a vector.
    pub fn contains(&self, iid: u32) -> bool {
        bit(&self.present, iid)
    }
    /// Code of `iid` (the `vectors.q8` BLOB).
    pub fn code(&self, iid: u32) -> Option<&[u8]> {
        if !self.contains(iid) {
            return None;
        }
        let d = self.quant.dim();
        let s = iid as usize * d;
        self.codes.get(s..s + d)
    }

    /// Reserve exact room for `additional` more slots.
    pub fn reserve(&mut self, additional: usize) {
        let n = (self.slots as usize)
            .saturating_add(additional)
            .min(self.max_slots as usize);
        grow::fit(&mut self.codes, n * self.quant.dim(), n * self.quant.dim());
        grow::fit(&mut self.present, n.div_ceil(64), n.div_ceil(64));
    }

    /// Bytes an upsert at `iid` would newly allocate (0 within capacity).
    pub fn growth_bytes(&self, iid: u32) -> usize {
        if iid < self.slots || iid >= self.max_slots {
            return 0;
        }
        let (n, t, d) = (
            iid as usize + 1,
            self.target(iid as usize + 1),
            self.quant.dim(),
        );
        grow::fit_bytes(&self.codes, n * d, t * d)
            + grow::fit_bytes(&self.present, n.div_ceil(64), t.div_ceil(64))
    }

    fn target(&self, n: usize) -> usize {
        let len = self.slots as usize;
        (len + grow::step(len, grow::MIN_SLOTS))
            .min(self.max_slots as usize)
            .max(n)
    }

    fn ensure_slot(&mut self, iid: u32) -> Result<usize, IndexError> {
        if iid >= self.max_slots {
            return Err(IndexError::CapacityExceeded {
                max_slots: self.max_slots,
            });
        }
        if iid >= self.slots {
            let (n, d) = (iid as usize + 1, self.quant.dim());
            let t = self.target(n);
            grow::fit(&mut self.codes, n * d, t * d);
            grow::fit(&mut self.present, n.div_ceil(64), t.div_ceil(64));
            self.slots = iid + 1;
            self.codes.resize(n * d, 0);
            self.present.resize(n.div_ceil(64), 0);
        }
        let w = &mut self.present[(iid / 64) as usize];
        let m = 1u64 << (iid % 64);
        if *w & m == 0 {
            *w |= m;
            self.live += 1;
        }
        Ok(iid as usize * self.quant.dim())
    }

    /// Insert or replace `iid` by encoding `v`.
    pub fn upsert(&mut self, iid: u32, v: &[f32]) -> Result<(), IndexError> {
        validate_vector(self.quant.metric(), self.quant.dim(), v)?;
        let s = self.ensure_slot(iid)?;
        let d = self.quant.dim();
        self.quant.encode(v, &mut self.codes[s..s + d]);
        Ok(())
    }

    /// Insert or replace `iid` from a stored code (`vectors.q8` with its
    /// `q8_epoch`), avoiding a re-encode on cold start.
    pub fn upsert_code(&mut self, iid: u32, code: &[u8], q8_epoch: u64) -> Result<(), IndexError> {
        if q8_epoch != self.quant.epoch() {
            return Err(IndexError::QuantEpochMismatch {
                index: self.quant.epoch(),
                got: q8_epoch,
            });
        }
        if code.len() != self.quant.dim() {
            return Err(IndexError::DimMismatch {
                expected: self.quant.dim(),
                got: code.len(),
            });
        }
        let s = self.ensure_slot(iid)?;
        self.codes[s..s + code.len()].copy_from_slice(code);
        Ok(())
    }

    /// Remove `iid`; returns whether it was present.
    pub fn remove(&mut self, iid: u32) -> bool {
        if !self.contains(iid) {
            return false;
        }
        self.present[(iid / 64) as usize] &= !(1u64 << (iid % 64));
        self.live -= 1;
        true
    }

    /// Renumber live vectors densely from `base` (the store's first iid) in
    /// place. Every live iid is `≥ base`, so each new iid is `≤` its old
    /// one and an ascending pass rewrites the arena without a second copy.
    /// Returns `map[old]` = new iid or `u32::MAX`; the store applies it to
    /// `vectors.iid` and sets `next_iid` past the last new iid. Errors if
    /// a live iid is below `base`.
    pub fn compact_ids(&mut self, base: u32) -> Result<Vec<u32>, IndexError> {
        if let Some(iid) = (0..base.min(self.slots)).find(|&i| self.contains(i)) {
            return Err(IndexError::IidBelowBase { iid, base });
        }
        let d = self.quant.dim();
        let mut map = vec![u32::MAX; self.slots as usize];
        let mut next = base;
        for old in 0..self.slots {
            if self.contains(old) {
                map[old as usize] = next;
                let (o, n) = (old as usize * d, next as usize * d);
                self.codes.copy_within(o..o + d, n);
                next += 1;
            }
        }
        self.slots = if self.live == 0 { 0 } else { next };
        let next = self.slots;
        self.codes.truncate(next as usize * d);
        self.present.clear();
        self.present.resize((next as usize).div_ceil(64), 0);
        for i in base..next {
            self.present[(i / 64) as usize] |= 1u64 << (i % 64);
        }
        Ok(map)
    }

    fn scan(&self, query: &[f32], k: usize) -> Result<Vec<Cand>, IndexError> {
        validate_vector(self.quant.metric(), self.quant.dim(), query)?;
        let pq = self.quant.prepare(query);
        let d = self.quant.dim();
        let mut top = TopK::new(k);
        for (wi, &w) in self.present.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let id = (wi as u32) * 64 + bits.trailing_zeros();
                bits &= bits - 1;
                let s = id as usize * d;
                let dist = pq.distance(&self.codes[s..s + d]);
                if top.bound().map_or(true, |b| dist <= b) {
                    top.push(Cand { d: dist, id });
                }
            }
        }
        Ok(top.into_sorted())
    }

    /// Approximate top-`k` by code distance (L2 is squared).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>, IndexError> {
        Ok(to_hits(self.scan(query, k)?))
    }

    /// Scan codes for `candidates` (≥ k; e.g. `4 * k`), then rerank them
    /// exactly through `fetch` and return the best `k`.
    pub fn search_rerank<F: RerankFetch>(
        &self,
        query: &[f32],
        k: usize,
        candidates: usize,
        fetch: &mut F,
    ) -> Result<Vec<Hit>, RerankError<F::Error>> {
        let cands = self
            .scan(query, candidates.max(k))
            .map_err(RerankError::Invalid)?;
        let ids: Vec<u32> = cands.iter().map(|c| c.id).collect();
        rerank(self.quant.metric(), query, &ids, k, fetch)
    }

    /// Resident bytes (allocated capacity plus the struct).
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.codes.capacity()
            + self.present.capacity() * 8
            + self.quant.dim() * 8
    }

    /// Release spare capacity.
    pub fn shrink_to_fit(&mut self) {
        self.codes.shrink_to_fit();
        self.present.shrink_to_fit();
    }

    fn write_payload(&self, w: &mut dyn Sink) {
        self.quant.write(w);
        w.put_u32(self.max_slots);
        w.put_u32(self.slots);
        w.put_u32(self.live);
        w.put_u64s(&self.present);
        w.put(&self.codes);
    }

    /// Stream into ≤ `max_chunk`-byte chunks stamped `epoch` (see
    /// [`crate::HnswIndex::encode_into`]).
    pub fn encode_into<E>(
        &self,
        epoch: u64,
        max_chunk: usize,
        emit: &mut dyn FnMut(IndexChunk) -> Result<(), E>,
    ) -> Result<IndexDigest, EmitError<E>> {
        encode(
            epoch,
            IndexKind::QuantFlat,
            max_chunk,
            &|w| self.write_payload(w),
            emit,
        )
    }

    /// [`Self::encode_into`] collected in memory.
    pub fn to_chunks(&self, epoch: u64, max_chunk: usize) -> Result<EncodedIndex, EncodeError> {
        encode_all(epoch, IndexKind::QuantFlat, max_chunk, &|w| {
            self.write_payload(w)
        })
    }

    /// Decode and validate chunks (see [`crate::HnswIndex::from_chunks`]).
    pub fn from_chunks(
        chunks: &[IndexChunk],
        expected_sha256: Option<&[u8; 32]>,
    ) -> Result<Self, DecodeError> {
        decode_slice(chunks, IndexKind::QuantFlat, expected_sha256, Self::parse)
    }

    /// Streaming decode (see [`crate::HnswIndex::from_chunk_iter`]).
    pub fn from_chunk_iter<I: IntoIterator<Item = IndexChunk>>(
        chunks: I,
        expected_sha256: Option<&[u8; 32]>,
        max_payload_bytes: u64,
    ) -> Result<Self, DecodeError> {
        let mut it = chunks.into_iter();
        decode(
            &mut it,
            IndexKind::QuantFlat,
            expected_sha256,
            max_payload_bytes,
            Self::parse,
        )
    }

    fn parse(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let quant = QuantParams::read(r)?;
        let max_slots = r.u32()?;
        let slots = r.u32()?;
        let live = r.u32()?;
        check_slots(max_slots, quant.dim()).map_err(DecodeError::Malformed)?;
        if slots > max_slots {
            return Err(DecodeError::Malformed("flat slots"));
        }
        let present = r.u64_vec((slots as usize).div_ceil(64))?;
        let codes = r.bytes_vec(slots as usize, quant.dim())?;
        let ones: u32 = present.iter().map(|w| w.count_ones()).sum();
        let tail_clear = slots % 64 == 0 || present.last().map_or(true, |w| w >> (slots % 64) == 0);
        if ones != live || !tail_clear {
            return Err(DecodeError::Malformed("flat presence bitmap"));
        }
        Ok(Self {
            quant,
            max_slots,
            slots,
            live,
            codes,
            present,
        })
    }
}
