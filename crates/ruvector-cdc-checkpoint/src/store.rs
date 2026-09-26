//! Content-addressed chunk store.
//!
//! Models the durable side of an incremental checkpoint: chunks are keyed
//! by SHA-256 of their bytes, so a chunk that already exists (because an
//! earlier checkpoint wrote identical content) is never written again. The
//! store reports, per `put`, how many *new* bytes it actually persisted —
//! that count is the honest "bytes written this round" metric the
//! benchmark compares across variants.

use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub type ChunkHash = [u8; 32];

pub fn hash_bytes(data: &[u8]) -> ChunkHash {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().into()
}

#[derive(Default)]
pub struct ChunkStore {
    chunks: HashMap<ChunkHash, Vec<u8>>,
}

impl ChunkStore {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    /// Insert a chunk if not already present. Returns `(hash, new_bytes)`
    /// where `new_bytes` is `data.len()` if this chunk was not already in
    /// the store, or `0` if it was a dedup hit.
    pub fn put(&mut self, data: &[u8]) -> (ChunkHash, usize) {
        let hash = hash_bytes(data);
        match self.chunks.entry(hash) {
            std::collections::hash_map::Entry::Occupied(_) => (hash, 0),
            std::collections::hash_map::Entry::Vacant(e) => {
                let len = data.len();
                e.insert(data.to_vec());
                (hash, len)
            }
        }
    }

    pub fn get(&self, hash: &ChunkHash) -> Option<&[u8]> {
        self.chunks.get(hash).map(Vec::as_slice)
    }

    /// Total bytes actually resident in the store (post-dedup), used to
    /// report cumulative repository size after N checkpoints.
    pub fn resident_bytes(&self) -> usize {
        self.chunks.values().map(Vec::len).sum()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

/// Split `data` at the given `(start, end)` ranges and store each piece,
/// returning the ordered list of chunk hashes and the total number of
/// bytes newly persisted (i.e. excluding dedup hits).
pub fn store_ranges(
    store: &mut ChunkStore,
    data: &[u8],
    ranges: &[(usize, usize)],
) -> (Vec<ChunkHash>, usize) {
    let mut hashes = Vec::with_capacity(ranges.len());
    let mut new_bytes = 0usize;
    for &(s, e) in ranges {
        let (h, n) = store.put(&data[s..e]);
        hashes.push(h);
        new_bytes += n;
    }
    (hashes, new_bytes)
}

/// Reconstruct the original byte stream from an ordered chunk-hash list.
/// Returns `None` if any referenced chunk is missing from the store —
/// e.g. it was evicted, or the manifest references a chunk that was never
/// written, both being reconstruction failures a real checkpoint system
/// must be able to detect.
pub fn reconstruct(store: &ChunkStore, hashes: &[ChunkHash]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    for h in hashes {
        out.extend_from_slice(store.get(h)?);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::{cdc_boundaries, CdcParams};

    #[test]
    fn dedup_hit_reports_zero_new_bytes() {
        let mut store = ChunkStore::new();
        let (_, n1) = store.put(b"hello world");
        let (_, n2) = store.put(b"hello world");
        assert_eq!(n1, 11);
        assert_eq!(n2, 0);
        assert_eq!(store.chunk_count(), 1);
    }

    #[test]
    fn store_and_reconstruct_round_trips_exactly() {
        let data: Vec<u8> = (0..10_000u32).flat_map(u32::to_le_bytes).collect();
        let params = CdcParams::new(256, 1024, 4096);
        let ranges = cdc_boundaries(&data, &params);
        let mut store = ChunkStore::new();
        let (hashes, _) = store_ranges(&mut store, &data, &ranges);
        let round_tripped = reconstruct(&store, &hashes).expect("all chunks present");
        assert_eq!(round_tripped, data);
    }

    #[test]
    fn reconstruct_fails_cleanly_on_missing_chunk() {
        let store = ChunkStore::new();
        let bogus = hash_bytes(b"never stored");
        assert!(reconstruct(&store, &[bogus]).is_none());
    }
}
