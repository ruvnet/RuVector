//! Resident f32 slab and its memory accounting (ADR-351 §6.1).
//!
//! Every row is charged its real resident footprint, not just its floats:
//! the values, the norm and iid, the id twice (the `ids` vector and the
//! `index` key), the metadata text, the compact filter form, and a fixed
//! per-row overhead for the vector headers and B-tree nodes. The running
//! total is what the per-shard cap and the isolate registry see.

use crate::filter::{compact_bytes, Compact};
use std::collections::BTreeMap;

/// Fixed per-row bookkeeping: `String`/`Option<String>`/`Box<[_]>` headers
/// in the column vectors, the `index` entry and its share of B-tree nodes.
pub const ROW_OVERHEAD_BYTES: u64 = 128;

/// Resident bytes of one row.
pub fn row_resident(
    dim: usize,
    id_len: usize,
    meta_len: usize,
    filt: &[(u8, crate::filter::Scalar)],
) -> u64 {
    (dim * 4 + 8 + 8 + 2 * id_len + meta_len + compact_bytes(filt)) as u64 + ROW_OVERHEAD_BYTES
}

/// Deletes swap-remove, so slot order is not stable; every observable
/// ordering is by id.
#[derive(Debug, Clone, Default)]
pub(crate) struct Slab {
    pub ids: Vec<String>,
    pub iids: Vec<i64>,
    pub values: Vec<f32>,
    pub norms: Vec<f64>,
    pub filt: Vec<Compact>,
    pub meta_text: Vec<Option<String>>,
    pub index: BTreeMap<String, usize>,
    /// Sum of [`row_resident`] over live rows.
    pub resident: u64,
    /// Sum of [`Slab::row_bytes`] over live rows (tenant `bytes` usage).
    pub stored: u64,
}

impl Slab {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn row(&self, slot: usize, dim: usize) -> &[f32] {
        &self.values[slot * dim..(slot + 1) * dim]
    }

    /// Stored bytes of a row (tenant `bytes` usage: id + f32 + metadata).
    pub fn row_bytes(&self, slot: usize, dim: usize) -> u64 {
        (self.ids[slot].len() + dim * 4 + self.meta_text[slot].as_ref().map_or(0, String::len))
            as u64
    }

    /// Resident bytes of a live row.
    pub fn slot_resident(&self, slot: usize, dim: usize) -> u64 {
        row_resident(
            dim,
            self.ids[slot].len(),
            self.meta_text[slot].as_ref().map_or(0, String::len),
            &self.filt[slot],
        )
    }

    /// Metadata of a row, parsed on demand (only for returned rows).
    pub fn metadata(&self, slot: usize) -> Option<serde_json::Value> {
        self.meta_text[slot]
            .as_deref()
            .and_then(|t| serde_json::from_str(t).ok())
    }

    pub fn put(
        &mut self,
        id: String,
        iid: i64,
        vals: &[f32],
        meta_text: Option<String>,
        filt: Compact,
    ) {
        let norm = crate::distance::norm(vals);
        let dim = vals.len();
        let add = row_resident(
            dim,
            id.len(),
            meta_text.as_ref().map_or(0, String::len),
            &filt,
        );
        let add_stored = (id.len() + dim * 4 + meta_text.as_ref().map_or(0, String::len)) as u64;
        if let Some(&slot) = self.index.get(&id) {
            self.stored = self.stored.saturating_sub(self.row_bytes(slot, dim));
            self.resident = self.resident.saturating_sub(self.slot_resident(slot, dim));
            self.values[slot * dim..(slot + 1) * dim].copy_from_slice(vals);
            self.iids[slot] = iid;
            self.norms[slot] = norm;
            self.filt[slot] = filt;
            self.meta_text[slot] = meta_text;
        } else {
            self.index.insert(id.clone(), self.ids.len());
            self.ids.push(id);
            self.iids.push(iid);
            self.values.extend_from_slice(vals);
            self.norms.push(norm);
            self.filt.push(filt);
            self.meta_text.push(meta_text);
        }
        self.resident = self.resident.saturating_add(add);
        self.stored = self.stored.saturating_add(add_stored);
    }

    pub fn remove(&mut self, id: &str, dim: usize) -> bool {
        let Some(&slot) = self.index.get(id) else {
            return false;
        };
        self.stored = self.stored.saturating_sub(self.row_bytes(slot, dim));
        self.resident = self.resident.saturating_sub(self.slot_resident(slot, dim));
        self.index.remove(id);
        let last = self.ids.len() - 1;
        if slot != last {
            let (head, tail) = self.values.split_at_mut(last * dim);
            head[slot * dim..(slot + 1) * dim].copy_from_slice(&tail[..dim]);
            self.index.insert(self.ids[last].clone(), slot);
        }
        self.ids.swap_remove(slot);
        self.iids.swap_remove(slot);
        self.norms.swap_remove(slot);
        self.filt.swap_remove(slot);
        self.meta_text.swap_remove(slot);
        self.values.truncate(last * dim);
        true
    }
}
