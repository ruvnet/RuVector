//! M2b: HNSW over u8 codes (ADR §6.1 "DistanceOracle traversal").
//!
//! Layout, all indexed by dense slot (`slot = iid - iid_base`):
//! * `codes`: `slots × dim` bytes;
//! * `levels`: one byte per slot, [`ABSENT`] for a never-assigned iid;
//! * `links0`: layer-0 adjacency, fixed stride `m0` `u32`s, [`NONE`]-padded
//!   at the tail only;
//! * `upper`: rows of `m` `u32`s; a node of level `L` owns `L` consecutive
//!   rows (layers `1..=L`) starting at `upper_off[slot]`;
//! * `deleted`: tombstone bitmap (deleted nodes still route until a purge).
//!
//! Links hold slots. Every public method speaks store iids: the store
//! allocates from 1 (`next_iid: 1`), so it sets `iid_base = 1` and the
//! first flush is dense without any renumbering.
//!
//! Resident cost ≈ `dim + 4·m0 + 5 + 4·m/(m-1) + 1/8` bytes per slot
//! ([`crate::memory::hnsw_bytes_per_vector`]).

mod build;
mod codec;
mod compact;
mod search;

use crate::error::IndexError;
use crate::grow;
use crate::quant::QuantParams;

pub(crate) const NONE: u32 = u32::MAX;
pub(crate) const ABSENT: u8 = u8::MAX;

/// Graph parameters (persisted with the index; ADR `meta` m0/…).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswParams {
    /// Links per node on layers ≥ 1, and links selected per insert.
    pub m: u16,
    /// Layer-0 link capacity (≥ `m`).
    pub m0: u16,
    /// Beam width while inserting.
    pub ef_construction: u16,
    /// Highest level a node may draw.
    pub max_level: u8,
    /// Slot cap (`max_iid + 1 - iid_base`), bounding memory.
    pub max_slots: u32,
    /// Smallest iid (slot 0). The edge store allocates iids from 1.
    pub iid_base: u32,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 128,
            max_level: 16,
            max_slots: 1 << 20,
            iid_base: 0,
        }
    }
}

impl HnswParams {
    /// Validate against `dim`: every `max_slots × width` arena size must
    /// fit a Vec on this target (so nothing can wrap on wasm32).
    pub(crate) fn check(&self, dim: usize) -> Result<(), &'static str> {
        if self.m < 2 || self.m0 < self.m {
            return Err("need 2 <= m <= m0");
        }
        if self.ef_construction == 0 {
            return Err("ef_construction must be > 0");
        }
        if self.max_level > 30 {
            return Err("max_level must be <= 30");
        }
        if self.max_slots == 0 || self.max_slots == NONE {
            return Err("max_slots out of range");
        }
        if u64::from(self.iid_base) + u64::from(self.max_slots) > u64::from(NONE) {
            return Err("iid_base + max_slots exceeds the u32 iid space");
        }
        let slots = self.max_slots as usize;
        let fits = |w: usize| {
            slots
                .checked_mul(w)
                .is_some_and(|b| b <= isize::MAX as usize)
        };
        if !fits(dim.max(1)) || !fits(4 * self.m0 as usize) {
            return Err("max_slots too large for this target");
        }
        Ok(())
    }
}

/// HNSW index whose traversal reads only u8 codes.
#[derive(Debug, Clone)]
pub struct HnswIndex {
    pub(crate) params: HnswParams,
    pub(crate) quant: QuantParams,
    pub(crate) codes: Vec<u8>,
    pub(crate) levels: Vec<u8>,
    pub(crate) links0: Vec<u32>,
    pub(crate) upper_off: Vec<u32>,
    pub(crate) upper: Vec<u32>,
    pub(crate) deleted: Vec<u64>,
    pub(crate) entry: u32,
    pub(crate) top: u8,
    pub(crate) present: u32,
    pub(crate) live: u32,
}

impl HnswIndex {
    /// Empty index.
    pub fn new(params: HnswParams, quant: QuantParams) -> Result<Self, IndexError> {
        Self::with_capacity(params, quant, 0)
    }

    /// Empty index with exact room for `n` nodes (upper rows sized to the
    /// expected `n·m/(m-1)` links), so a bulk build does not over-allocate.
    pub fn with_capacity(
        params: HnswParams,
        quant: QuantParams,
        n: usize,
    ) -> Result<Self, IndexError> {
        params
            .check(quant.dim())
            .map_err(IndexError::InvalidParams)?;
        let mut s = Self {
            params,
            codes: Vec::new(),
            levels: Vec::new(),
            links0: Vec::new(),
            upper_off: Vec::new(),
            upper: Vec::new(),
            deleted: Vec::new(),
            entry: NONE,
            top: 0,
            present: 0,
            live: 0,
            quant,
        };
        s.reserve(n);
        Ok(s)
    }

    /// Graph parameters.
    pub fn params(&self) -> &HnswParams {
        &self.params
    }
    /// Quantizer parameters.
    pub fn quant(&self) -> &QuantParams {
        &self.quant
    }
    /// Live (present, not deleted) nodes.
    pub fn len(&self) -> usize {
        self.live as usize
    }
    /// No live nodes.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
    /// Slots, i.e. `max_iid + 1 - iid_base`. This — not [`Self::len`] — is
    /// what memory is paid for; pass it to the `memory` estimators.
    pub fn node_count(&self) -> u32 {
        self.levels.len() as u32
    }
    /// Slots that hold a node (deleted or not). `< node_count()` means the
    /// iid space is gapped and [`Self::to_chunks`] will refuse.
    pub fn present_count(&self) -> u32 {
        self.present
    }
    /// Tombstoned share of present nodes; schedule a purge
    /// ([`Self::compact`]) when it grows.
    pub fn dead_ratio(&self) -> f64 {
        if self.present == 0 {
            0.0
        } else {
            f64::from(self.present - self.live) / f64::from(self.present)
        }
    }
    /// Entry point iid and top level (ADR `meta.entry_point/max_layer`).
    pub fn entry_point(&self) -> Option<(u32, u8)> {
        (self.entry != NONE).then(|| (self.iid(self.entry), self.top))
    }
    /// Whether `iid` is a live node.
    pub fn contains(&self, iid: u32) -> bool {
        self.slot(iid)
            .is_ok_and(|s| self.is_present(s) && !self.is_deleted(s))
    }
    /// Level of the node at `iid` (deleted or not).
    pub fn level(&self, iid: u32) -> Option<u8> {
        let s = self.slot(iid).ok()?;
        self.is_present(s).then(|| self.levels[s as usize])
    }

    /// Tombstone `iid` (it keeps routing, never returned). Returns whether
    /// a live node was deleted.
    pub fn delete(&mut self, iid: u32) -> bool {
        if !self.contains(iid) {
            return false;
        }
        let s = iid - self.params.iid_base;
        self.deleted[(s / 64) as usize] |= 1u64 << (s % 64);
        self.live -= 1;
        true
    }

    /// Resident bytes (allocated capacity plus the struct).
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.codes.capacity()
            + self.levels.capacity()
            + 4 * (self.links0.capacity() + self.upper_off.capacity() + self.upper.capacity())
            + 8 * self.deleted.capacity()
            + 8 * self.quant.dim()
    }

    /// Release spare capacity (after a bulk build).
    pub fn shrink_to_fit(&mut self) {
        self.codes.shrink_to_fit();
        self.levels.shrink_to_fit();
        self.links0.shrink_to_fit();
        self.upper_off.shrink_to_fit();
        self.upper.shrink_to_fit();
        self.deleted.shrink_to_fit();
    }

    /// Reserve exact room for `additional` more slots (e.g. the ops about
    /// to be replayed), so inserts up to that point never reallocate.
    pub fn reserve(&mut self, additional: usize) {
        let len = self.levels.len();
        let n = len
            .saturating_add(additional)
            .min(self.params.max_slots as usize);
        self.fit_slots(n, n);
        let m = self.params.m as usize;
        // Expected upper-layer u32s: n·m/(m-1) (E[level] = 1/(m-1) rows of m).
        let elems = (n - len).saturating_mul(m).div_ceil(m - 1) + 4 * m;
        let need = self.upper.len() + elems;
        grow::fit(&mut self.upper, need, need);
    }

    /// Bytes an insert at `iid` would newly allocate (0 while reserved
    /// capacity lasts; an upper bound, counting one upper-row step), so
    /// the store can refuse *before* the allocation happens.
    pub fn growth_bytes(&self, iid: u32) -> usize {
        let Ok(s) = self.slot(iid) else { return 0 };
        let (len, n) = (self.levels.len(), s as usize + 1);
        let mut b = 0;
        if n > len {
            let t = self.slot_target(n);
            let (d, m0) = (self.quant.dim(), self.params.m0 as usize);
            b += grow::fit_bytes(&self.codes, n * d, t * d)
                + grow::fit_bytes(&self.levels, n, t)
                + grow::fit_bytes(&self.links0, n * m0, t * m0)
                + grow::fit_bytes(&self.upper_off, n, t)
                + grow::fit_bytes(&self.deleted, n.div_ceil(64), t.div_ceil(64));
        }
        let m = self.params.m as usize;
        let need = self.upper.len() + self.params.max_level as usize * m;
        b + grow::fit_bytes(&self.upper, need, self.upper_target(need))
    }

    pub(crate) fn slot_target(&self, n: usize) -> usize {
        let len = self.levels.len();
        (len + grow::step(len, grow::MIN_SLOTS))
            .min(self.params.max_slots as usize)
            .max(n)
    }

    pub(crate) fn upper_target(&self, need: usize) -> usize {
        let m = self.params.m as usize;
        let rows = self.upper.len() / m;
        need.max(self.upper.len() + m * grow::step(rows, grow::MIN_ROWS))
    }

    pub(crate) fn fit_slots(&mut self, need: usize, target: usize) {
        let (d, m0) = (self.quant.dim(), self.params.m0 as usize);
        grow::fit(&mut self.codes, need * d, target * d);
        grow::fit(&mut self.levels, need, target);
        grow::fit(&mut self.links0, need * m0, target * m0);
        grow::fit(&mut self.upper_off, need, target);
        grow::fit(&mut self.deleted, need.div_ceil(64), target.div_ceil(64));
    }

    /// Slot of a store iid (range-checked against base and cap).
    pub(crate) fn slot(&self, iid: u32) -> Result<u32, IndexError> {
        let base = self.params.iid_base;
        let s = iid
            .checked_sub(base)
            .ok_or(IndexError::IidBelowBase { iid, base })?;
        if s >= self.params.max_slots {
            return Err(IndexError::CapacityExceeded {
                max_slots: self.params.max_slots,
            });
        }
        Ok(s)
    }

    /// Store iid of a slot.
    #[inline]
    pub(crate) fn iid(&self, slot: u32) -> u32 {
        slot + self.params.iid_base
    }

    pub(crate) fn is_present(&self, i: u32) -> bool {
        self.levels.get(i as usize).is_some_and(|&l| l != ABSENT)
    }

    pub(crate) fn is_deleted(&self, i: u32) -> bool {
        self.deleted
            .get((i / 64) as usize)
            .is_some_and(|w| w & (1u64 << (i % 64)) != 0)
    }

    #[inline]
    pub(crate) fn code(&self, i: u32) -> &[u8] {
        let d = self.quant.dim();
        &self.codes[i as usize * d..(i as usize + 1) * d]
    }

    fn row(&self, i: u32, layer: usize) -> (usize, usize) {
        if layer == 0 {
            let s = self.params.m0 as usize;
            (i as usize * s, s)
        } else {
            let s = self.params.m as usize;
            ((self.upper_off[i as usize] as usize + layer - 1) * s, s)
        }
    }

    /// Full stride of `i`'s links at `layer` (NONE-padded tail). The node
    /// must have `level >= layer` (decode enforces it for every link).
    #[inline]
    pub(crate) fn links(&self, i: u32, layer: usize) -> &[u32] {
        let (s, n) = self.row(i, layer);
        if layer == 0 {
            &self.links0[s..s + n]
        } else {
            &self.upper[s..s + n]
        }
    }

    pub(crate) fn set_links(&mut self, i: u32, layer: usize, ids: &[u32]) {
        let (s, n) = self.row(i, layer);
        let row = if layer == 0 {
            &mut self.links0[s..s + n]
        } else {
            &mut self.upper[s..s + n]
        };
        for (k, slot) in row.iter_mut().enumerate() {
            *slot = ids.get(k).copied().unwrap_or(NONE);
        }
    }
}

/// Dense visited bitmap, cleared between uses by zeroing only the words
/// that were touched.
pub(crate) struct Visited {
    bits: Vec<u64>,
    touched: Vec<u32>,
}

impl Visited {
    pub fn new(n: usize) -> Self {
        Self {
            bits: vec![0; n.div_ceil(64)],
            touched: Vec::new(),
        }
    }

    /// Mark `i`; returns whether it was already marked.
    #[inline]
    pub fn test_set(&mut self, i: u32) -> bool {
        let w = (i / 64) as usize;
        let m = 1u64 << (i % 64);
        let word = &mut self.bits[w];
        if *word & m != 0 {
            return true;
        }
        if *word == 0 {
            self.touched.push(w as u32);
        }
        *word |= m;
        false
    }

    pub fn clear(&mut self) {
        for &w in &self.touched {
            self.bits[w as usize] = 0;
        }
        self.touched.clear();
    }
}
