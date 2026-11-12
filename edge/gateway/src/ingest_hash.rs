//! Resumable sha256 pass over a staged import upload (ADR-351 §3
//! rv-ingest). A multipart R2 object carries no sha256, so the declared
//! one is checked by hashing the whole upload (up to 1 GiB) — far more than
//! one Worker invocation may spend under the default CPU limit. The pass is
//! therefore split across queue deliveries: each delivery range-reads at
//! most [`HASH_BYTES_PER_DELIVERY`] from the persisted offset, runs the
//! SHA-256 compression function over the whole 64-byte blocks
//! (`sha2::compress256`) and persists the eight chaining words plus the
//! offset in the job record ([`HashPass`]). The delivery that reaches the
//! end pads the final block(s) (FIPS 180-4 §5.1.1) and compares the digest.
//!
//! The offset only ever advances by whole blocks, so the state is exactly
//! `(H0..H7, bytes absorbed)`; a delivery that dies before its save simply
//! re-reads the same range (the pass is idempotent).

use serde::{Deserialize, Serialize};

/// Bytes hashed per delivery (a multiple of 64): one 8 MiB range read, the
/// size of the upload parts and of the import tail buffer already in
/// flight (the range is buffered whole, so this is a memory bound). At the
/// ≈ 100 MB/s wasm SHA-256 estimate that is ≈ 84 ms of the Workers Paid
/// 30 s `cpu_ms`; 128 deliveries per 1 GiB upload. A larger slice would
/// need bounded sub-reads inside one delivery. (Free: 512 KiB.)
pub const HASH_BYTES_PER_DELIVERY: u64 = 8 << 20;

/// SHA-256 initial hash value (FIPS 180-4 §5.3.3).
const IV: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// Persisted state of an unfinished pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashPass {
    /// Chaining words after `off` bytes.
    pub h: [u32; 8],
    /// Bytes absorbed (always a multiple of 64).
    pub off: u64,
}

impl Default for HashPass {
    fn default() -> Self {
        HashPass { h: IV, off: 0 }
    }
}

/// What one delivery of the pass concluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashStep {
    /// More to hash: persist this state and continue.
    More(HashPass),
    /// The whole object was hashed: its digest.
    Done([u8; 32]),
}

impl HashPass {
    /// The byte range `(offset, len)` the next delivery reads from an
    /// object of `size` bytes.
    pub fn next_range(&self, size: u64) -> (u64, u64) {
        let off = self.off.min(size);
        (off, (size - off).min(HASH_BYTES_PER_DELIVERY))
    }

    fn absorb(&mut self, blocks: &[u8]) {
        debug_assert_eq!(blocks.len() % 64, 0);
        for b in blocks.chunks_exact(64) {
            let mut block = [0u8; 64];
            block.copy_from_slice(b);
            sha2::compress256(&mut self.h, &[block.into()]);
        }
        self.off += blocks.len() as u64;
    }

    /// Absorb `piece` (the bytes at [`Self::next_range`] of an object of
    /// `size` bytes). `None` when `piece` is not exactly that range.
    pub fn step(mut self, piece: &[u8], size: u64) -> Option<HashStep> {
        let (off, len) = self.next_range(size);
        if off != self.off || piece.len() as u64 != len {
            return None;
        }
        if off + len < size {
            // Mid-object: `HASH_BYTES_PER_DELIVERY` keeps this whole blocks.
            if piece.len() % 64 != 0 {
                return None;
            }
            self.absorb(piece);
            return Some(HashStep::More(self));
        }
        let whole = piece.len() / 64 * 64;
        self.absorb(&piece[..whole]);
        let rest = &piece[whole..];
        // Padding: 0x80, zeros, the 64-bit big-endian bit length.
        let mut last = [0u8; 128];
        last[..rest.len()].copy_from_slice(rest);
        last[rest.len()] = 0x80;
        let n = if rest.len() < 56 { 64 } else { 128 };
        last[n - 8..n].copy_from_slice(&size.wrapping_mul(8).to_be_bytes());
        self.absorb(&last[..n]);
        let mut out = [0u8; 32];
        for (o, w) in out.chunks_exact_mut(4).zip(self.h) {
            o.copy_from_slice(&w.to_be_bytes());
        }
        Some(HashStep::Done(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn chunked(data: &[u8]) -> ([u8; 32], usize) {
        let size = data.len() as u64;
        let mut st = HashPass::default();
        let mut deliveries = 0;
        loop {
            deliveries += 1;
            let (off, len) = st.next_range(size);
            let piece = &data[off as usize..(off + len) as usize];
            // Round-trip the persisted state like the job record does.
            let json = serde_json::to_string(&st).unwrap();
            st = serde_json::from_str(&json).unwrap();
            match st.step(piece, size).unwrap() {
                HashStep::More(next) => st = next,
                HashStep::Done(d) => return (d, deliveries),
            }
        }
    }

    #[test]
    fn chunked_pass_matches_one_shot_sha256_at_every_padding_edge() {
        let big = 2 * HASH_BYTES_PER_DELIVERY as usize + 7;
        for n in [0usize, 1, 55, 56, 63, 64, 65, 119, 120, 128, 4096, big] {
            let data: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
            let (d, deliveries) = chunked(&data);
            assert_eq!(d, <[u8; 32]>::from(Sha256::digest(&data)), "len {n}");
            let want = (n as u64).div_ceil(HASH_BYTES_PER_DELIVERY).max(1) as usize;
            assert_eq!(deliveries, want, "len {n}");
        }
    }

    #[test]
    fn a_short_or_misplaced_piece_is_refused() {
        let st = HashPass::default();
        let size = HASH_BYTES_PER_DELIVERY * 2;
        assert!(st.step(&[0; 10], size).is_none());
        let moved = HashPass { off: 64, ..st };
        assert_eq!(moved.next_range(size), (64, HASH_BYTES_PER_DELIVERY));
        assert!(moved.step(&[0; 100], size).is_none());
    }
}
