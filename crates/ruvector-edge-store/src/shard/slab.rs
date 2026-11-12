//! Resident row bookkeeping and its memory accounting (ADR-351 §6.1).
//!
//! At M2 the slab holds **no vector data**: f32 rows stay in SQLite (read
//! back for the rerank), codes and links live in the index (`ann`). Per row
//! it keeps the id, the iid, the compact filter form and the metadata
//! length (for stored-bytes usage); metadata text is read from SQLite only
//! for returned rows. Every row is charged its real resident footprint:
//! the id twice (the `ids` vector and the `index` key), iid, lengths, the
//! compact filter form, the iid→slot entry and a fixed per-row overhead.

use crate::filter::{compact_bytes, Compact};
use std::collections::BTreeMap;

/// Fixed per-row bookkeeping: `String`/`Box<[_]>` headers in the column
/// vectors, the `index` entry and its share of B-tree nodes.
pub const ROW_OVERHEAD_BYTES: u64 = 128;

/// No row at this iid.
const NO_SLOT: u32 = u32::MAX;

/// Resident bytes of one row's bookkeeping (index codes are separate).
pub fn row_resident(id_len: usize, filt: &[(u8, crate::filter::Scalar)]) -> u64 {
    (8 + 4 + 4 + 2 * id_len + compact_bytes(filt)) as u64 + ROW_OVERHEAD_BYTES
}

/// Deletes swap-remove, so slot order is not stable; every observable
/// ordering is by id (or by `(distance, id)`).
#[derive(Debug, Clone, Default)]
pub(crate) struct Slab {
    pub ids: Vec<String>,
    pub iids: Vec<i64>,
    pub filt: Vec<Compact>,
    pub meta_len: Vec<u32>,
    pub index: BTreeMap<String, usize>,
    /// `slot_of_iid[iid]` = slot, [`NO_SLOT`] when absent.
    slot_of_iid: Vec<u32>,
    /// Sum of [`row_resident`] over live rows.
    pub resident: u64,
    /// Sum of stored bytes over live rows (tenant `bytes` usage).
    pub stored: u64,
}

impl Slab {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Stored bytes of a row (tenant `bytes` usage: id + f32 + metadata).
    pub fn row_bytes(&self, slot: usize, dim: usize) -> u64 {
        (self.ids[slot].len() + dim * 4 + self.meta_len[slot] as usize) as u64
    }

    /// Resident bytes of a live row.
    pub fn slot_resident(&self, slot: usize) -> u64 {
        row_resident(self.ids[slot].len(), &self.filt[slot])
    }

    /// Bytes of the iid→slot map (capacity).
    pub fn map_bytes(&self) -> u64 {
        self.slot_of_iid.capacity() as u64 * 4
    }

    /// Slot holding `iid`.
    pub fn slot_of(&self, iid: i64) -> Option<usize> {
        let i = usize::try_from(iid).ok()?;
        match self.slot_of_iid.get(i) {
            Some(&s) if s != NO_SLOT => Some(s as usize),
            _ => None,
        }
    }

    fn set_slot(&mut self, iid: i64, slot: u32) {
        let Ok(i) = usize::try_from(iid) else { return };
        if i >= self.slot_of_iid.len() {
            let want = i + 1;
            if want > self.slot_of_iid.capacity() {
                // Proportional steps, like the index arenas: no doubling.
                let step = (want / 16).max(256);
                self.slot_of_iid
                    .reserve_exact(want + step - self.slot_of_iid.len());
            }
            self.slot_of_iid.resize(want, NO_SLOT);
        }
        self.slot_of_iid[i] = slot;
    }

    /// Stored bytes a row of this shape would have.
    pub fn stored_of(id_len: usize, dim: usize, meta_len: usize) -> u64 {
        (id_len + dim * 4 + meta_len) as u64
    }

    pub fn put(&mut self, id: String, iid: i64, dim: usize, meta_len: usize, filt: Compact) {
        let add = row_resident(id.len(), &filt);
        let add_stored = Self::stored_of(id.len(), dim, meta_len);
        let meta_len = u32::try_from(meta_len).unwrap_or(u32::MAX);
        if let Some(&slot) = self.index.get(&id) {
            self.stored = self.stored.saturating_sub(self.row_bytes(slot, dim));
            self.resident = self.resident.saturating_sub(self.slot_resident(slot));
            let old = self.iids[slot];
            if old != iid {
                self.set_slot(old, NO_SLOT);
                self.set_slot(iid, slot as u32);
            }
            self.iids[slot] = iid;
            self.filt[slot] = filt;
            self.meta_len[slot] = meta_len;
        } else {
            let slot = self.ids.len();
            self.index.insert(id.clone(), slot);
            self.ids.push(id);
            self.iids.push(iid);
            self.filt.push(filt);
            self.meta_len.push(meta_len);
            self.set_slot(iid, slot as u32);
        }
        self.resident = self.resident.saturating_add(add);
        self.stored = self.stored.saturating_add(add_stored);
    }

    /// Remove `id`; returns its iid.
    pub fn remove(&mut self, id: &str, dim: usize) -> Option<i64> {
        let slot = *self.index.get(id)?;
        self.stored = self.stored.saturating_sub(self.row_bytes(slot, dim));
        self.resident = self.resident.saturating_sub(self.slot_resident(slot));
        self.index.remove(id);
        let iid = self.iids[slot];
        self.set_slot(iid, NO_SLOT);
        let last = self.ids.len() - 1;
        if slot != last {
            self.index.insert(self.ids[last].clone(), slot);
            let moved = self.iids[last];
            self.set_slot(moved, slot as u32);
        }
        self.ids.swap_remove(slot);
        self.iids.swap_remove(slot);
        self.filt.swap_remove(slot);
        self.meta_len.swap_remove(slot);
        Some(iid)
    }

    /// Apply a dense renumbering `old iid → new iid` (compaction).
    pub fn renumber(&mut self, map: impl Fn(i64) -> Option<i64>) {
        self.slot_of_iid.clear();
        for slot in 0..self.iids.len() {
            let new = map(self.iids[slot]).unwrap_or(self.iids[slot]);
            self.iids[slot] = new;
            self.set_slot(new, slot as u32);
        }
        self.slot_of_iid.shrink_to_fit();
    }

    /// Highest live iid (0 when empty).
    pub fn max_iid(&self) -> i64 {
        self.iids.iter().copied().max().unwrap_or(0)
    }
}
