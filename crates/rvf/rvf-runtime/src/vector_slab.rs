//! Contiguous slab storage for the store's vectors.
//!
//! Replaces the former `HashMap<u64, Vec<f32>>` layout (one heap allocation
//! per vector, pointer-chasing on every distance call) with the layout used
//! by SOTA engines (usearch, faiss): one flat row-major `Vec<f32>` with a
//! fixed dimension per store, an id -> ordinal map, and an ordinal -> id
//! table.
//!
//! Properties:
//! - `get()` returns a `&[f32]` row without copying.
//! - New IDs always append; overwriting an existing ID rewrites its row in
//!   place (same ordinal).
//! - `remove()` tombstones the ordinal; slots are reclaimed only by
//!   [`VectorData::compact_in_place`] (called from store compaction), so
//!   live rows never move underneath readers between mutations.
//! - Iteration (`ids()` / `iter()`) walks live rows in ordinal order, which
//!   is deterministic across process restarts (unlike HashMap key order).
//!   All query consumers already apply `(distance, id)` tie-breaking, so
//!   result semantics are unchanged.

use std::collections::HashMap;

/// In-memory vector storage: contiguous row-major slab + id mapping.
pub(crate) struct VectorData {
    /// Fixed dimension of every row.
    pub dimension: u16,
    /// Flat row-major storage; ordinal `o` occupies `data[o*dim..(o+1)*dim]`.
    data: Vec<f32>,
    /// Live id -> ordinal. Removed ids are absent.
    id_to_ord: HashMap<u64, u32>,
    /// Ordinal -> id (including tombstoned ordinals, see `live`).
    ord_to_id: Vec<u64>,
    /// Parallel to `ord_to_id`: false for tombstoned ordinals.
    live: Vec<bool>,
    /// Number of tombstoned ordinals (reclaimed by `compact_in_place`).
    tombstones: usize,
}

impl VectorData {
    pub(crate) fn new(dimension: u16) -> Self {
        Self {
            dimension,
            data: Vec::new(),
            id_to_ord: HashMap::new(),
            ord_to_id: Vec::new(),
            live: Vec::new(),
            tombstones: 0,
        }
    }

    /// Create a slab with capacity pre-reserved for `n` vectors.
    pub(crate) fn with_capacity(dimension: u16, n: usize) -> Self {
        let mut slab = Self::new(dimension);
        slab.reserve(n);
        slab
    }

    /// Reserve capacity for `additional` more vectors.
    pub(crate) fn reserve(&mut self, additional: usize) {
        let dim = self.dimension as usize;
        self.data.reserve(additional.saturating_mul(dim));
        self.id_to_ord.reserve(additional);
        self.ord_to_id.reserve(additional);
        self.live.reserve(additional);
    }

    /// Number of live vectors.
    pub(crate) fn len(&self) -> usize {
        self.id_to_ord.len()
    }

    /// Borrow the row for `id` as a slice (no copy).
    pub(crate) fn get(&self, id: u64) -> Option<&[f32]> {
        let dim = self.dimension as usize;
        self.id_to_ord.get(&id).map(|&ord| {
            let start = ord as usize * dim;
            &self.data[start..start + dim]
        })
    }

    /// Insert or overwrite the vector for `id`.
    ///
    /// Rows whose length does not match the slab dimension are rejected
    /// (they would corrupt the row-major layout). The store validates
    /// dimensions at ingest, so this only triggers on corrupt input.
    pub(crate) fn insert_slice(&mut self, id: u64, data: &[f32]) {
        let dim = self.dimension as usize;
        if data.len() != dim {
            debug_assert_eq!(data.len(), dim, "row length must match slab dimension");
            return;
        }
        match self.id_to_ord.get(&id) {
            Some(&ord) => {
                let start = ord as usize * dim;
                self.data[start..start + dim].copy_from_slice(data);
            }
            None => {
                let ord = self.alloc_ordinal(id);
                debug_assert_eq!(ord as usize * dim, self.data.len());
                self.data.extend_from_slice(data);
            }
        }
    }

    /// Insert or overwrite by owned vector (compatibility shim for the old
    /// `HashMap::insert` call sites; the buffer is copied into the slab).
    pub(crate) fn insert(&mut self, id: u64, data: Vec<f32>) {
        self.insert_slice(id, &data);
    }

    /// Tombstone the ordinal for `id`. The row's slot is reclaimed only by
    /// [`Self::compact_in_place`].
    pub(crate) fn remove(&mut self, id: u64) {
        if let Some(ord) = self.id_to_ord.remove(&id) {
            let ord = ord as usize;
            if self.live[ord] {
                self.live[ord] = false;
                self.tombstones += 1;
            }
        }
    }

    /// Iterate the live IDs in ordinal (insertion) order.
    pub(crate) fn ids(&self) -> impl Iterator<Item = &u64> {
        self.ord_to_id
            .iter()
            .zip(self.live.iter())
            .filter_map(|(id, &live)| live.then_some(id))
    }

    /// Iterate `(id, row)` pairs over live vectors in ordinal order.
    /// Rows are borrowed straight from the slab (no copies).
    pub(crate) fn iter(&self) -> impl Iterator<Item = (u64, &[f32])> {
        let dim = self.dimension as usize;
        self.ord_to_id
            .iter()
            .enumerate()
            .filter_map(move |(ord, &id)| {
                self.live[ord].then(|| {
                    let start = ord * dim;
                    (id, &self.data[start..start + dim])
                })
            })
    }

    /// Reclaim tombstoned slots: rebuild the slab densely, preserving the
    /// relative ordinal order of live rows. No-op when nothing is deleted.
    pub(crate) fn compact_in_place(&mut self) {
        if self.tombstones == 0 {
            return;
        }
        let dim = self.dimension as usize;
        let live_count = self.id_to_ord.len();
        let mut new_data = Vec::with_capacity(live_count * dim);
        let mut new_ord_to_id = Vec::with_capacity(live_count);
        for (ord, &id) in self.ord_to_id.iter().enumerate() {
            if self.live[ord] {
                let start = ord * dim;
                new_data.extend_from_slice(&self.data[start..start + dim]);
                new_ord_to_id.push(id);
            }
        }
        self.id_to_ord = new_ord_to_id
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i as u32))
            .collect();
        self.live = vec![true; new_ord_to_id.len()];
        self.ord_to_id = new_ord_to_id;
        self.data = new_data;
        self.tombstones = 0;
    }

    /// Bulk-load a VEC_SEG payload directly into the slab.
    ///
    /// Payload layout (written by `write_path::write_vec_seg`):
    /// `dimension: u16 | vector_count: u32 | [id: u64 | f32 * dimension]*`.
    ///
    /// Avoids the per-vector `Vec<f32>` allocation of the legacy
    /// `read_vec_seg_payload` path: ids and rows are copied straight from
    /// the payload buffer into the contiguous slab (a single byte copy per
    /// row on little-endian targets). Duplicate ids overwrite in place,
    /// matching the legacy load semantics.
    ///
    /// Returns `None` (load nothing) when the payload is truncated, the
    /// arithmetic overflows, or the segment dimension does not match the
    /// slab dimension; the caller may fall back to the legacy parser.
    pub(crate) fn load_from_vec_seg(&mut self, payload: &[u8]) -> Option<usize> {
        if payload.len() < 6 {
            return None;
        }
        let dim = u16::from_le_bytes([payload[0], payload[1]]) as usize;
        let count = u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]) as usize;
        if dim != self.dimension as usize {
            return None;
        }
        let bytes_per_vec = dim.checked_mul(4)?;
        let stride = bytes_per_vec.checked_add(8)?;
        let expected = count.checked_mul(stride)?.checked_add(6)?;
        if payload.len() < expected {
            return None;
        }

        self.reserve(count);
        let mut offset = 6;
        for _ in 0..count {
            let id = u64::from_le_bytes(payload[offset..offset + 8].try_into().ok()?);
            offset += 8;
            self.insert_row_le(id, &payload[offset..offset + bytes_per_vec]);
            offset += bytes_per_vec;
        }
        Some(count)
    }

    /// Insert or overwrite a row from little-endian f32 wire bytes.
    fn insert_row_le(&mut self, id: u64, row_bytes: &[u8]) {
        let dim = self.dimension as usize;
        debug_assert_eq!(row_bytes.len(), dim * 4);
        let start = match self.id_to_ord.get(&id) {
            Some(&ord) => ord as usize * dim,
            None => {
                let ord = self.alloc_ordinal(id);
                let start = ord as usize * dim;
                debug_assert_eq!(start, self.data.len());
                self.data.resize(start + dim, 0.0);
                start
            }
        };
        let dst = &mut self.data[start..start + dim];
        #[cfg(target_endian = "little")]
        {
            // SAFETY: f32 has no padding and every 4-byte pattern is a
            // valid f32 bit pattern. `row_bytes` holds exactly `dim`
            // little-endian f32 values and `dst` covers exactly `dim`
            // f32 slots, so the regions match in size and do not overlap
            // (source is the segment payload, destination the slab).
            // Mirrors the write path's LE fast path in `write_vec_seg`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row_bytes.as_ptr(),
                    dst.as_mut_ptr() as *mut u8,
                    dim * 4,
                );
            }
        }
        #[cfg(target_endian = "big")]
        for (i, chunk) in row_bytes.chunks_exact(4).enumerate() {
            dst[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    }

    /// Allocate the next ordinal for a new id.
    fn alloc_ordinal(&mut self, id: u64) -> u32 {
        let ord = self.ord_to_id.len();
        assert!(ord <= u32::MAX as usize, "vector slab ordinal overflow");
        let ord = ord as u32;
        self.ord_to_id.push(id);
        self.live.push(true);
        self.id_to_ord.insert(id, ord);
        ord
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(seed: f32, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| seed + i as f32).collect()
    }

    #[test]
    fn insert_get_overwrite() {
        let mut slab = VectorData::new(4);
        slab.insert_slice(10, &row(1.0, 4));
        slab.insert_slice(20, &row(2.0, 4));
        assert_eq!(slab.len(), 2);
        assert_eq!(slab.get(10).unwrap(), row(1.0, 4).as_slice());

        // Overwrite keeps the ordinal (no growth) and replaces the row.
        slab.insert_slice(10, &row(9.0, 4));
        assert_eq!(slab.len(), 2);
        assert_eq!(slab.get(10).unwrap(), row(9.0, 4).as_slice());
        assert_eq!(slab.data.len(), 8);
    }

    #[test]
    fn mismatched_row_length_is_rejected() {
        let mut slab = VectorData::new(4);
        // debug_assert fires in debug builds; verify release semantics via
        // the public surface only when not debug-asserting.
        if !cfg!(debug_assertions) {
            slab.insert_slice(1, &[1.0, 2.0]);
            assert_eq!(slab.len(), 0);
            assert!(slab.get(1).is_none());
        }
    }

    #[test]
    fn remove_tombstones_until_compact() {
        let mut slab = VectorData::new(2);
        for id in 0..5u64 {
            slab.insert_slice(id, &row(id as f32, 2));
        }
        slab.remove(1);
        slab.remove(3);
        assert_eq!(slab.len(), 3);
        assert!(slab.get(1).is_none());
        // Slots are not reused before compaction.
        assert_eq!(slab.data.len(), 10);
        let ids: Vec<u64> = slab.ids().copied().collect();
        assert_eq!(ids, vec![0, 2, 4]);

        slab.compact_in_place();
        assert_eq!(slab.data.len(), 6);
        assert_eq!(slab.len(), 3);
        for id in [0u64, 2, 4] {
            assert_eq!(slab.get(id).unwrap(), row(id as f32, 2).as_slice());
        }

        // Re-inserting a removed id appends again.
        slab.insert_slice(1, &row(7.0, 2));
        assert_eq!(slab.get(1).unwrap(), row(7.0, 2).as_slice());
        assert_eq!(slab.len(), 4);
    }

    #[test]
    fn iter_yields_live_rows_in_ordinal_order() {
        let mut slab = VectorData::new(2);
        for id in [30u64, 10, 20] {
            slab.insert_slice(id, &row(id as f32, 2));
        }
        slab.remove(10);
        let pairs: Vec<(u64, Vec<f32>)> = slab.iter().map(|(id, v)| (id, v.to_vec())).collect();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, 30);
        assert_eq!(pairs[1].0, 20);
        assert_eq!(pairs[1].1, row(20.0, 2));
    }

    #[test]
    fn load_from_vec_seg_round_trip() {
        // Same payload as read_path::tests::vec_seg_round_trip.
        let dim: u16 = 2;
        let count: u32 = 2;
        let mut payload = Vec::new();
        payload.extend_from_slice(&dim.to_le_bytes());
        payload.extend_from_slice(&count.to_le_bytes());
        payload.extend_from_slice(&10u64.to_le_bytes());
        payload.extend_from_slice(&1.0f32.to_le_bytes());
        payload.extend_from_slice(&2.0f32.to_le_bytes());
        payload.extend_from_slice(&20u64.to_le_bytes());
        payload.extend_from_slice(&3.0f32.to_le_bytes());
        payload.extend_from_slice(&4.0f32.to_le_bytes());

        let mut slab = VectorData::new(2);
        assert_eq!(slab.load_from_vec_seg(&payload), Some(2));
        assert_eq!(slab.get(10).unwrap(), &[1.0, 2.0]);
        assert_eq!(slab.get(20).unwrap(), &[3.0, 4.0]);

        // A later segment overwriting id 10 replaces its row in place.
        let mut payload2 = Vec::new();
        payload2.extend_from_slice(&dim.to_le_bytes());
        payload2.extend_from_slice(&1u32.to_le_bytes());
        payload2.extend_from_slice(&10u64.to_le_bytes());
        payload2.extend_from_slice(&8.0f32.to_le_bytes());
        payload2.extend_from_slice(&9.0f32.to_le_bytes());
        assert_eq!(slab.load_from_vec_seg(&payload2), Some(1));
        assert_eq!(slab.get(10).unwrap(), &[8.0, 9.0]);
        assert_eq!(slab.len(), 2);
    }

    #[test]
    fn load_from_vec_seg_rejects_bad_payloads() {
        let mut slab = VectorData::new(2);
        // Too short.
        assert_eq!(slab.load_from_vec_seg(&[0u8; 4]), None);
        // Dimension mismatch.
        let mut payload = Vec::new();
        payload.extend_from_slice(&3u16.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(slab.load_from_vec_seg(&payload), None);
        // Truncated body: claims 1 vector but has no bytes for it.
        let mut payload = Vec::new();
        payload.extend_from_slice(&2u16.to_le_bytes());
        payload.extend_from_slice(&1u32.to_le_bytes());
        assert_eq!(slab.load_from_vec_seg(&payload), None);
        // Implausible count must not overflow.
        let mut payload = Vec::new();
        payload.extend_from_slice(&2u16.to_le_bytes());
        payload.extend_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(slab.load_from_vec_seg(&payload), None);
        assert_eq!(slab.len(), 0);
    }
}
