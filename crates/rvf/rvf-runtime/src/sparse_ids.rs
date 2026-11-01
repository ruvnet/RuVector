//! Chunked set of `u64` identifiers for sparse membership filters.
//!
//! Memory is proportional to the number of members, not to the largest
//! identifier, so a store whose IDs are hashed or start at a high ordinal does
//! not need a bitmap sized by the maximum ID.
//!
//! The layout is "roaring-style": an ID splits into a 48-bit chunk key
//! (`id >> 16`) and a 16-bit low part. A chunk holds its low parts either as a
//! sorted array (up to [`ARRAY_MAX`] entries) or as a 65,536-bit bitmap, and
//! switches between the two at that cardinality. Lookup is one ordered-map
//! probe plus a binary search or a bit test.
//!
//! Wire format (`FilterType::RoaringBitmap`), all integers little-endian:
//!
//! ```text
//! u32  chunk_count
//! per chunk, ascending by key:
//!   u64  key              (id >> 16, below 2^48)
//!   u8   kind             (0 = array, 1 = bitmap)
//!   array:  u16 n (1..=4096), then n x u16 strictly ascending
//!   bitmap: 1024 x u64    (cardinality above 4096)
//! ```
//!
//! The encoding is canonical (a chunk is an array exactly when its
//! cardinality is at most [`ARRAY_MAX`]), so equal sets hash equally and the
//! decoder can reject anything that is not the canonical form.

use std::collections::BTreeMap;

use rvf_types::{ErrorCode, RvfError};

/// Largest cardinality stored as a sorted array; above this a chunk is a bitmap.
pub(crate) const ARRAY_MAX: usize = 4096;
const BITMAP_WORDS: usize = 1024;
const KEY_LIMIT: u64 = 1 << 48;

const KIND_ARRAY: u8 = 0;
const KIND_BITMAP: u8 = 1;

#[derive(Clone)]
enum Chunk {
    Array(Vec<u16>),
    Bitmap {
        words: Box<[u64; BITMAP_WORDS]>,
        card: u32,
    },
}

impl Chunk {
    fn contains(&self, low: u16) -> bool {
        match self {
            Chunk::Array(values) => values.binary_search(&low).is_ok(),
            Chunk::Bitmap { words, .. } => words[(low >> 6) as usize] & (1u64 << (low & 63)) != 0,
        }
    }

    fn insert(&mut self, low: u16) -> bool {
        match self {
            Chunk::Array(values) => match values.binary_search(&low) {
                Ok(_) => false,
                Err(pos) => {
                    values.insert(pos, low);
                    if values.len() > ARRAY_MAX {
                        let mut words = Box::new([0u64; BITMAP_WORDS]);
                        for &v in values.iter() {
                            words[(v >> 6) as usize] |= 1u64 << (v & 63);
                        }
                        let card = values.len() as u32;
                        *self = Chunk::Bitmap { words, card };
                    }
                    true
                }
            },
            Chunk::Bitmap { words, card } => {
                let word = &mut words[(low >> 6) as usize];
                let mask = 1u64 << (low & 63);
                if *word & mask != 0 {
                    return false;
                }
                *word |= mask;
                *card += 1;
                true
            }
        }
    }

    /// Returns whether the value was present, and whether the chunk is now empty.
    fn remove(&mut self, low: u16) -> (bool, bool) {
        match self {
            Chunk::Array(values) => match values.binary_search(&low) {
                Ok(pos) => {
                    values.remove(pos);
                    (true, values.is_empty())
                }
                Err(_) => (false, false),
            },
            Chunk::Bitmap { words, card } => {
                let word = &mut words[(low >> 6) as usize];
                let mask = 1u64 << (low & 63);
                if *word & mask == 0 {
                    return (false, false);
                }
                *word &= !mask;
                *card -= 1;
                if *card as usize <= ARRAY_MAX {
                    let mut values = Vec::with_capacity(*card as usize);
                    for (i, &w) in words.iter().enumerate() {
                        let mut w = w;
                        while w != 0 {
                            values.push((i * 64) as u16 + w.trailing_zeros() as u16);
                            w &= w - 1;
                        }
                    }
                    *self = Chunk::Array(values);
                }
                (true, false)
            }
        }
    }
}

/// A set of `u64` identifiers whose size tracks its member count.
#[derive(Clone, Default)]
pub(crate) struct SparseIdSet {
    chunks: BTreeMap<u64, Chunk>,
    len: u64,
}

impl SparseIdSet {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    #[inline]
    pub(crate) fn contains(&self, id: u64) -> bool {
        self.chunks
            .get(&(id >> 16))
            .is_some_and(|chunk| chunk.contains(id as u16))
    }

    pub(crate) fn insert(&mut self, id: u64) -> bool {
        let chunk = self
            .chunks
            .entry(id >> 16)
            .or_insert_with(|| Chunk::Array(Vec::new()));
        let added = chunk.insert(id as u16);
        if added {
            self.len += 1;
        }
        added
    }

    pub(crate) fn remove(&mut self, id: u64) -> bool {
        let key = id >> 16;
        let Some(chunk) = self.chunks.get_mut(&key) else {
            return false;
        };
        let (removed, now_empty) = chunk.remove(id as u16);
        if now_empty {
            self.chunks.remove(&key);
        }
        if removed {
            self.len -= 1;
        }
        removed
    }

    /// Largest member, if any.
    pub(crate) fn max(&self) -> Option<u64> {
        let (&key, chunk) = self.chunks.iter().next_back()?;
        let low = match chunk {
            Chunk::Array(values) => *values.last()?,
            Chunk::Bitmap { words, .. } => {
                let i = words.iter().rposition(|&w| w != 0)?;
                (i * 64) as u16 + (63 - words[i].leading_zeros() as u16)
            }
        };
        Some((key << 16) | u64::from(low))
    }

    /// Bytes `serialize` will produce.
    pub(crate) fn serialized_len(&self) -> usize {
        4 + self
            .chunks
            .values()
            .map(|chunk| {
                9 + match chunk {
                    Chunk::Array(values) => 2 + values.len() * 2,
                    Chunk::Bitmap { .. } => BITMAP_WORDS * 8,
                }
            })
            .sum::<usize>()
    }

    pub(crate) fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.serialized_len());
        buf.extend_from_slice(&(self.chunks.len() as u32).to_le_bytes());
        for (&key, chunk) in &self.chunks {
            buf.extend_from_slice(&key.to_le_bytes());
            match chunk {
                Chunk::Array(values) => {
                    buf.push(KIND_ARRAY);
                    buf.extend_from_slice(&(values.len() as u16).to_le_bytes());
                    for v in values {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                Chunk::Bitmap { words, .. } => {
                    buf.push(KIND_BITMAP);
                    for w in words.iter() {
                        buf.extend_from_slice(&w.to_le_bytes());
                    }
                }
            }
        }
        buf
    }

    /// Decode the canonical form; any deviation is `MembershipInvalid`.
    pub(crate) fn deserialize(data: &[u8]) -> Result<Self, RvfError> {
        let bad = || RvfError::Code(ErrorCode::MembershipInvalid);
        let mut pos = 0usize;
        let mut take = |n: usize| -> Result<&[u8], RvfError> {
            let end = pos.checked_add(n).ok_or_else(bad)?;
            let slice = data.get(pos..end).ok_or_else(bad)?;
            pos = end;
            Ok(slice)
        };

        let chunk_count = u32::from_le_bytes(take(4)?.try_into().expect("4 bytes")) as usize;
        // Smallest chunk is 9 + 2 + 2 bytes; refuse a count the input cannot hold
        // before allocating anything.
        if chunk_count > data.len() / 13 {
            return Err(bad());
        }
        let mut set = Self::new();
        let mut prev_key: Option<u64> = None;
        for _ in 0..chunk_count {
            let key = u64::from_le_bytes(take(8)?.try_into().expect("8 bytes"));
            if key >= KEY_LIMIT || prev_key.is_some_and(|p| key <= p) {
                return Err(bad());
            }
            prev_key = Some(key);
            let kind = take(1)?[0];
            let (chunk, card) = match kind {
                KIND_ARRAY => {
                    let n = u16::from_le_bytes(take(2)?.try_into().expect("2 bytes")) as usize;
                    if n == 0 || n > ARRAY_MAX {
                        return Err(bad());
                    }
                    let raw = take(n * 2)?;
                    let values: Vec<u16> = raw
                        .chunks_exact(2)
                        .map(|b| u16::from_le_bytes([b[0], b[1]]))
                        .collect();
                    if values.windows(2).any(|w| w[0] >= w[1]) {
                        return Err(bad());
                    }
                    (Chunk::Array(values), n as u64)
                }
                KIND_BITMAP => {
                    let raw = take(BITMAP_WORDS * 8)?;
                    let mut words = Box::new([0u64; BITMAP_WORDS]);
                    for (w, b) in words.iter_mut().zip(raw.chunks_exact(8)) {
                        *w = u64::from_le_bytes(b.try_into().expect("8 bytes"));
                    }
                    let card: u32 = words.iter().map(|w| w.count_ones()).sum();
                    if card as usize <= ARRAY_MAX {
                        return Err(bad());
                    }
                    (Chunk::Bitmap { words, card }, u64::from(card))
                }
                _ => return Err(bad()),
            };
            set.chunks.insert(key, chunk);
            set.len += card;
        }
        if pos != data.len() {
            return Err(bad());
        }
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_remove_contains_across_chunks() {
        let mut s = SparseIdSet::new();
        let ids = [
            0u64,
            1,
            65_535,
            65_536,
            1 << 40,
            (1 << 40) + 7,
            u64::MAX >> 16,
        ];
        for &id in &ids {
            assert!(s.insert(id));
            assert!(!s.insert(id));
        }
        assert_eq!(s.len(), ids.len() as u64);
        for &id in &ids {
            assert!(s.contains(id));
        }
        assert!(!s.contains(2));
        assert_eq!(s.max(), ids.iter().copied().max());
        assert!(s.remove(65_536));
        assert!(!s.remove(65_536));
        assert!(!s.contains(65_536));
        assert_eq!(s.len(), ids.len() as u64 - 1);
    }

    #[test]
    fn array_promotes_to_bitmap_and_demotes_back() {
        let mut s = SparseIdSet::new();
        for id in 0..=(ARRAY_MAX as u64) {
            s.insert(id);
        }
        assert!(matches!(s.chunks[&0], Chunk::Bitmap { .. }));
        assert!(s.remove(0));
        assert!(matches!(s.chunks[&0], Chunk::Array(_)));
        assert_eq!(s.len(), ARRAY_MAX as u64);
        for id in 1..=(ARRAY_MAX as u64) {
            assert!(s.remove(id));
        }
        assert!(s.chunks.is_empty());
        assert_eq!(s.max(), None);
    }

    #[test]
    fn serialize_round_trips_in_canonical_form() {
        let mut s = SparseIdSet::new();
        for id in (0..20_000u64).step_by(3) {
            s.insert(id);
        }
        s.insert(1 << 40);
        let bytes = s.serialize();
        assert_eq!(bytes.len(), s.serialized_len());
        let back = SparseIdSet::deserialize(&bytes).unwrap();
        assert_eq!(back.len(), s.len());
        assert_eq!(back.serialize(), bytes);
        for id in 0..20_000u64 {
            assert_eq!(back.contains(id), id % 3 == 0, "id {id}");
        }
        assert!(back.contains(1 << 40));
    }

    #[test]
    fn deserialize_rejects_malformed_input() {
        let mut s = SparseIdSet::new();
        s.insert(5);
        s.insert(6);
        let good = s.serialize();
        assert!(SparseIdSet::deserialize(&good).is_ok());
        // truncated, trailing byte, unsorted array, bad kind, impossible count
        assert!(SparseIdSet::deserialize(&good[..good.len() - 1]).is_err());
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(SparseIdSet::deserialize(&trailing).is_err());
        let mut unsorted = good.clone();
        let n = unsorted.len();
        unsorted.swap(n - 2, n - 4);
        unsorted.swap(n - 1, n - 3);
        assert!(SparseIdSet::deserialize(&unsorted).is_err());
        let mut kind = good.clone();
        kind[12] = 9;
        assert!(SparseIdSet::deserialize(&kind).is_err());
        let mut count = good;
        count[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(SparseIdSet::deserialize(&count).is_err());
        assert!(SparseIdSet::deserialize(&[]).is_err());
    }

    #[test]
    fn non_canonical_small_bitmap_is_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.push(KIND_BITMAP);
        let mut words = [0u64; BITMAP_WORDS];
        words[0] = 1;
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        assert!(SparseIdSet::deserialize(&bytes).is_err());
    }
}
