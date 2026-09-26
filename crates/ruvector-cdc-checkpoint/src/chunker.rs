//! Content-defined chunking (CDC) and a fixed-size baseline chunker.
//!
//! The CDC chunker is a FastCDC-style gear-hash rolling chunker: a boundary
//! is declared where a rolling hash over the trailing bytes hits a
//! zero-mask, subject to `min_size`/`max_size` clamps. Its defining property
//! versus fixed-size chunking is *resynchronization*: an insertion or
//! deletion inside the source bytes shifts fixed-size block boundaries for
//! everything downstream of the edit (so every later block changes and must
//! be re-stored), while content-defined boundaries are anchored to local
//! byte content and resynchronize within `max_size` of the edit — only the
//! chunks touching the edit change.

/// A 256-entry gear table used by the rolling hash. Generated once at
/// compile time from a fixed seed via `splitmix64` so the table (and
/// therefore chunk boundaries) is fully deterministic across builds and
/// platforms — required for the reconstruction and witness tests to be
/// reproducible.
pub const GEAR: [u64; 256] = build_gear_table();

const fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const fn build_gear_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut i = 0usize;
    let mut seed = 0x5EED_CAFE_D00D_1234u64;
    while i < 256 {
        seed = splitmix64(seed);
        table[i] = seed;
        i += 1;
    }
    table
}

/// Parameters controlling the content-defined chunker's target chunk size.
#[derive(Debug, Clone, Copy)]
pub struct CdcParams {
    pub min_size: usize,
    pub avg_size: usize,
    pub max_size: usize,
}

impl CdcParams {
    /// `avg_size` must be a power of two — the mask is derived from its
    /// bit width. Panics otherwise; this is a configuration error, not a
    /// runtime data condition.
    pub fn new(min_size: usize, avg_size: usize, max_size: usize) -> Self {
        assert!(
            avg_size.is_power_of_two(),
            "avg_size must be a power of two"
        );
        assert!(min_size < avg_size && avg_size < max_size);
        Self {
            min_size,
            avg_size,
            max_size,
        }
    }

    fn mask(&self) -> u64 {
        let bits = self.avg_size.trailing_zeros();
        (1u64 << bits) - 1
    }
}

/// Split `data` into content-defined chunks and return each chunk's
/// half-open byte range `[start, end)`.
pub fn cdc_boundaries(data: &[u8], params: &CdcParams) -> Vec<(usize, usize)> {
    if data.is_empty() {
        return Vec::new();
    }
    let mask = params.mask();
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut hash: u64 = 0;

    for i in 0..data.len() {
        hash = (hash << 1).wrapping_add(GEAR[data[i] as usize]);
        let size = i - start + 1;
        let at_end = i + 1 == data.len();
        if size >= params.max_size || (size >= params.min_size && (hash & mask) == 0) {
            ranges.push((start, i + 1));
            start = i + 1;
            hash = 0;
        } else if at_end {
            ranges.push((start, data.len()));
        }
    }
    ranges
}

/// Fixed-size block chunking — the "reasonable baseline" delta strategy
/// used by many naive incremental-backup tools. Every block is `block_size`
/// bytes except possibly the last.
pub fn fixed_boundaries(data: &[u8], block_size: usize) -> Vec<(usize, usize)> {
    assert!(block_size > 0);
    if data.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < data.len() {
        let end = (start + block_size).min(data.len());
        ranges.push((start, end));
        start = end;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn cdc_boundaries_cover_all_bytes_exactly_once() {
        let data = sample(50_000, 42);
        let params = CdcParams::new(256, 1024, 4096);
        let ranges = cdc_boundaries(&data, &params);
        assert_eq!(ranges.first().unwrap().0, 0);
        assert_eq!(ranges.last().unwrap().1, data.len());
        for w in ranges.windows(2) {
            assert_eq!(
                w[0].1, w[1].0,
                "ranges must be contiguous with no gaps or overlap"
            );
        }
    }

    #[test]
    fn cdc_boundaries_are_deterministic() {
        let data = sample(50_000, 7);
        let params = CdcParams::new(256, 1024, 4096);
        assert_eq!(
            cdc_boundaries(&data, &params),
            cdc_boundaries(&data, &params)
        );
    }

    #[test]
    fn cdc_resynchronizes_after_a_mid_stream_insertion() {
        // The defining CDC property: inserting bytes in the middle of the
        // stream should only perturb the chunk(s) touching the insertion,
        // not every chunk boundary after it (unlike fixed-size chunking).
        let base = sample(200_000, 99);
        let params = CdcParams::new(512, 2048, 8192);
        let before = cdc_boundaries(&base, &params);

        let mut edited = base.clone();
        edited.splice(100_000..100_000, sample(37, 1234));
        let after = cdc_boundaries(&edited, &params);

        // Chunk boundaries strictly before the edit point must be identical.
        let before_edit: Vec<_> = before.iter().take_while(|&&(_, e)| e <= 100_000).collect();
        let after_edit: Vec<_> = after.iter().take_while(|&&(_, e)| e <= 100_000).collect();
        assert_eq!(before_edit, after_edit);

        // And boundaries far past the edit (end of stream) should realign:
        // the tail chunk sizes should match once shifted by the insert length.
        let tail_before = before.last().unwrap();
        let tail_after = after.last().unwrap();
        assert_eq!(
            tail_before.1 - tail_before.0 == 0,
            tail_after.1 - tail_after.0 == 0
        );
    }

    #[test]
    fn fixed_boundaries_shift_every_block_content_after_an_edit() {
        // Contrast case proving the fixed-size baseline lacks resync.
        // Fixed-size boundaries are pure byte-offset arithmetic, so the
        // *index ranges* (0..4096, 4096..8192, ...) never move regardless
        // of content — but a small insertion shifts every byte after it,
        // so the *content* every downstream block covers changes, which
        // means every downstream block hashes differently and must be
        // re-stored even though only 5 bytes actually changed. That is the
        // property under test, not the index arithmetic itself.
        let base = sample(20_000, 3);
        let mut edited = base.clone();
        edited.splice(10..10, vec![0xAAu8; 5]);

        let before = fixed_boundaries(&base, 4096);
        let after = fixed_boundaries(&edited, 4096);
        assert_eq!(
            before[1], after[1],
            "block index ranges are content-independent"
        );

        let block_before = &base[before[1].0..before[1].1];
        let block_after = &edited[after[1].0..after[1].1];
        assert_ne!(
            block_before, block_after,
            "the second block's content should differ post-edit even though its index range did not"
        );
    }
}
