//! Membership filter for shared HNSW index traversal.
//!
//! Include mode (default): vector visible iff `filter.contains(id)`.
//! Exclude mode: vector visible iff `!filter.contains(id)`.
//!
//! Empty filter in include mode = empty view (fail-safe).
//!
//! HNSW traversal integration:
//! - Excluded nodes MAY be pushed onto exploration heap (routing waypoints)
//! - Excluded nodes MUST NOT be pushed onto result heap
//! - Excluded nodes DO NOT decrement `ef_remaining`

use rvf_types::membership::{FilterMode, FilterType, MembershipHeader, MEMBERSHIP_MAGIC};
use rvf_types::{ErrorCode, RvfError};

use crate::sparse_ids::SparseIdSet;

/// Capacities up to this many 64-bit words (64 KiB of bitmap) are always
/// stored dense: lookup is a single load and the bitmap is cheap.
const DENSE_ALWAYS_WORDS: u64 = 8192;
/// Past that, a filter is dense only while the bitmap costs at most this many
/// words per member, so a few members spread over a huge ID range do not pay
/// for the range.
const DENSE_WORDS_PER_MEMBER: u64 = 4;

/// Storage behind a [`MembershipFilter`].
#[derive(Clone)]
enum Bits {
    /// One bit per ID below `vector_count` (`FilterType::Bitmap`).
    Dense(Vec<u64>),
    /// Chunked set whose size follows the member count
    /// (`FilterType::RoaringBitmap`).
    Sparse(SparseIdSet),
}

/// Membership filter over vector IDs.
///
/// Small or dense ID ranges use a flat bitmap; large, sparse ranges use a
/// chunked set, so memory follows the number of IDs rather than the largest
/// one. Both behave identically through this API.
#[derive(Clone)]
pub struct MembershipFilter {
    /// Include or exclude mode.
    mode: FilterMode,
    bits: Bits,
    /// Total vector count (capacity of the filter).
    vector_count: u64,
    /// Number of members in the filter.
    member_count: u64,
    /// Generation counter for optimistic concurrency.
    generation_id: u32,
}

impl MembershipFilter {
    fn empty_as(mode: FilterMode, vector_count: u64, dense: bool) -> Self {
        let bits = if dense {
            Bits::Dense(vec![0u64; vector_count.div_ceil(64) as usize])
        } else {
            Bits::Sparse(SparseIdSet::new())
        };
        Self {
            mode,
            bits,
            vector_count,
            member_count: 0,
            generation_id: 0,
        }
    }

    fn empty(mode: FilterMode, vector_count: u64) -> Self {
        let dense = vector_count.div_ceil(64) <= DENSE_ALWAYS_WORDS;
        Self::empty_as(mode, vector_count, dense)
    }

    /// Bytes the membership data occupies, for tests that bound memory.
    #[cfg(test)]
    pub(crate) fn approx_heap_bytes(&self) -> usize {
        match &self.bits {
            Bits::Dense(bitmap) => bitmap.len() * 8,
            Bits::Sparse(set) => set.serialized_len(),
        }
    }

    /// Create a new include-mode filter with given capacity. All bits start clear.
    ///
    /// Capacities too large for a flat bitmap to be sensible start out sparse,
    /// so this never allocates in proportion to `vector_count` past 64 KiB.
    pub fn new_include(vector_count: u64) -> Self {
        Self::empty(FilterMode::Include, vector_count)
    }

    /// Create a new exclude-mode filter with given capacity. All bits start clear.
    pub fn new_exclude(vector_count: u64) -> Self {
        Self::empty(FilterMode::Exclude, vector_count)
    }

    /// Build an include-mode filter holding `ids`, choosing the dense or the
    /// sparse form from how thinly they cover `0..vector_count`.
    ///
    /// IDs at or above `vector_count` are ignored, as with [`add`](Self::add).
    pub fn include_from_ids(vector_count: u64, ids: &[u64]) -> Self {
        let words = vector_count.div_ceil(64);
        let dense = words <= DENSE_ALWAYS_WORDS
            || words <= (ids.len() as u64).saturating_mul(DENSE_WORDS_PER_MEMBER);
        let mut filter = Self::empty_as(FilterMode::Include, vector_count, dense);
        for &id in ids {
            filter.add(id);
        }
        filter
    }

    /// Add a vector ID to the filter.
    pub fn add(&mut self, vector_id: u64) {
        if vector_id >= self.vector_count {
            return;
        }
        let added = match &mut self.bits {
            Bits::Dense(bitmap) => {
                let word = (vector_id / 64) as usize;
                let mask = 1u64 << (vector_id % 64);
                match bitmap.get_mut(word) {
                    Some(w) if *w & mask == 0 => {
                        *w |= mask;
                        true
                    }
                    _ => false,
                }
            }
            Bits::Sparse(set) => set.insert(vector_id),
        };
        if added {
            self.member_count += 1;
        }
    }

    /// Remove a vector ID from the filter.
    pub fn remove(&mut self, vector_id: u64) {
        if vector_id >= self.vector_count {
            return;
        }
        let removed = match &mut self.bits {
            Bits::Dense(bitmap) => {
                let word = (vector_id / 64) as usize;
                let mask = 1u64 << (vector_id % 64);
                match bitmap.get_mut(word) {
                    Some(w) if *w & mask != 0 => {
                        *w &= !mask;
                        true
                    }
                    _ => false,
                }
            }
            Bits::Sparse(set) => set.remove(vector_id),
        };
        if removed {
            self.member_count -= 1;
        }
    }

    /// Check if a vector ID is in the filter bitmap.
    #[inline]
    fn bitmap_contains(&self, vector_id: u64) -> bool {
        if vector_id >= self.vector_count {
            return false;
        }
        match &self.bits {
            Bits::Dense(bitmap) => bitmap
                .get((vector_id / 64) as usize)
                .is_some_and(|w| w & (1u64 << (vector_id % 64)) != 0),
            Bits::Sparse(set) => set.contains(vector_id),
        }
    }

    /// Check if a vector ID is visible through this filter.
    ///
    /// In Include mode: visible iff the bit is set.
    /// In Exclude mode: visible iff the bit is NOT set.
    #[inline]
    pub fn contains(&self, vector_id: u64) -> bool {
        match self.mode {
            FilterMode::Include => self.bitmap_contains(vector_id),
            FilterMode::Exclude => !self.bitmap_contains(vector_id),
        }
    }

    /// Number of set bits (members in the bitmap).
    pub fn member_count(&self) -> u64 {
        self.member_count
    }

    /// Total vector capacity.
    pub fn vector_count(&self) -> u64 {
        self.vector_count
    }

    /// Filter mode.
    pub fn mode(&self) -> FilterMode {
        self.mode
    }

    /// Generation ID.
    pub fn generation_id(&self) -> u32 {
        self.generation_id
    }

    /// Increment generation ID.
    pub fn bump_generation(&mut self) {
        self.generation_id += 1;
    }

    /// The `FilterType` tag this filter serializes as.
    fn filter_type(&self) -> FilterType {
        match self.bits {
            Bits::Dense(_) => FilterType::Bitmap,
            Bits::Sparse(_) => FilterType::RoaringBitmap,
        }
    }

    /// Serialize the filter body: raw bitmap words for a dense filter, the
    /// chunked encoding (see `sparse_ids`) for a sparse one.
    pub fn serialize(&self) -> Vec<u8> {
        match &self.bits {
            Bits::Dense(bitmap) => {
                let mut buf = Vec::with_capacity(bitmap.len() * 8);
                for &word in bitmap {
                    buf.extend_from_slice(&word.to_le_bytes());
                }
                buf
            }
            Bits::Sparse(set) => set.serialize(),
        }
    }

    /// Deserialize a MembershipFilter from bitmap bytes and a header.
    ///
    /// Reads both forms, picked by `header.filter_type`; files written before
    /// the sparse form existed are all `Bitmap` and load as before.
    pub fn deserialize(data: &[u8], header: &MembershipHeader) -> Result<Self, RvfError> {
        let bad = || RvfError::Code(ErrorCode::MembershipInvalid);
        let mode = FilterMode::try_from(header.filter_mode).map_err(|_| bad())?;
        let filter_type = FilterType::try_from(header.filter_type).map_err(|_| bad())?;

        match filter_type {
            FilterType::Bitmap => {
                let word_count = header.vector_count.div_ceil(64) as usize;
                let expected_bytes = word_count.checked_mul(8).ok_or_else(bad)?;
                if data.len() < expected_bytes {
                    return Err(bad());
                }

                let mut bitmap = Vec::with_capacity(word_count);
                for chunk in data[..expected_bytes].chunks_exact(8) {
                    bitmap.push(u64::from_le_bytes(chunk.try_into().expect("8 bytes")));
                }

                // Recount set bits
                let member_count: u64 = bitmap.iter().map(|w| w.count_ones() as u64).sum();

                Ok(Self {
                    mode,
                    bits: Bits::Dense(bitmap),
                    vector_count: header.vector_count,
                    member_count,
                    generation_id: header.generation_id,
                })
            }
            FilterType::RoaringBitmap => {
                let set = SparseIdSet::deserialize(data)?;
                // Every member must be addressable, or `add` could never have
                // produced this set.
                if set.max().is_some_and(|max| max >= header.vector_count) {
                    return Err(bad());
                }
                Ok(Self {
                    mode,
                    member_count: set.len(),
                    bits: Bits::Sparse(set),
                    vector_count: header.vector_count,
                    generation_id: header.generation_id,
                })
            }
        }
    }

    /// Build a MembershipHeader for this filter.
    pub fn to_header(&self) -> MembershipHeader {
        let bitmap_bytes = self.serialize();
        let filter_hash = crate::store::simple_shake256_256(&bitmap_bytes);

        MembershipHeader {
            magic: MEMBERSHIP_MAGIC,
            version: 1,
            filter_type: self.filter_type() as u8,
            filter_mode: self.mode as u8,
            vector_count: self.vector_count,
            member_count: self.member_count,
            filter_offset: 96, // right after header
            filter_size: bitmap_bytes.len() as u32,
            generation_id: self.generation_id,
            filter_hash,
            bloom_offset: 0,
            bloom_size: 0,
            _reserved: 0,
            _reserved2: [0u8; 8],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Deterministic ID stream (splitmix64).
    fn ids(seed: u64, n: usize, range: u64) -> Vec<u64> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = x;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                (z ^ (z >> 31)) % range
            })
            .collect()
    }

    fn header_for(filter: &MembershipFilter) -> MembershipHeader {
        filter.to_header()
    }

    #[test]
    fn dense_and_sparse_agree_with_a_reference_set() {
        for (mode, seed, capacity, n) in [
            (FilterMode::Include, 1, 300_000u64, 5_000usize),
            (FilterMode::Exclude, 2, 300_000, 5_000),
            (FilterMode::Include, 3, 70_000, 40_000),
            (FilterMode::Exclude, 4, 1000, 200),
        ] {
            let mut dense = MembershipFilter::empty_as(mode, capacity, true);
            let mut sparse = MembershipFilter::empty_as(mode, capacity, false);
            let mut reference = BTreeSet::new();
            let adds = ids(seed, n, capacity + 50); // a few out of range
            for (step, &id) in adds.iter().enumerate() {
                dense.add(id);
                sparse.add(id);
                if id < capacity {
                    reference.insert(id);
                }
                // Interleave removals so both forms see deletes.
                if step % 3 == 0 {
                    let victim = adds[step / 2];
                    dense.remove(victim);
                    sparse.remove(victim);
                    reference.remove(&victim);
                }
            }
            assert_eq!(dense.member_count(), reference.len() as u64);
            assert_eq!(sparse.member_count(), reference.len() as u64);
            for id in (0..capacity + 100).step_by(7).chain(adds.iter().copied()) {
                let in_set = reference.contains(&id);
                let expect = match mode {
                    FilterMode::Include => in_set,
                    FilterMode::Exclude => !in_set,
                };
                assert_eq!(dense.contains(id), expect, "dense id {id}");
                assert_eq!(sparse.contains(id), expect, "sparse id {id}");
            }
        }
    }

    #[test]
    fn both_forms_round_trip_through_a_header() {
        for dense in [true, false] {
            let mut f = MembershipFilter::empty_as(FilterMode::Include, 100_000, dense);
            for id in ids(9, 3_000, 100_000) {
                f.add(id);
            }
            f.bump_generation();
            let header = header_for(&f);
            assert_eq!(
                header.filter_type,
                if dense {
                    FilterType::Bitmap as u8
                } else {
                    FilterType::RoaringBitmap as u8
                }
            );
            let back = MembershipFilter::deserialize(&f.serialize(), &header).unwrap();
            assert_eq!(back.member_count(), f.member_count());
            assert_eq!(back.generation_id(), 1);
            assert_eq!(back.serialize(), f.serialize());
            for id in 0..100_000u64 {
                assert_eq!(back.contains(id), f.contains(id));
            }
        }
    }

    #[test]
    fn dense_wire_format_is_unchanged() {
        let mut f = MembershipFilter::new_include(130);
        for id in [0u64, 1, 65, 129] {
            f.add(id);
        }
        let mut expected = Vec::new();
        for word in [0b11u64, 0b10, 0b10] {
            expected.extend_from_slice(&word.to_le_bytes());
        }
        assert_eq!(f.serialize(), expected);
        assert_eq!(f.to_header().filter_type, FilterType::Bitmap as u8);
    }

    /// `cargo test --release -p rvf-runtime --lib bench_contains -- --ignored --nocapture`
    #[test]
    #[ignore = "timing, run by hand in release mode"]
    fn bench_contains() {
        use std::hint::black_box;
        use std::time::Instant;

        fn ns_per_lookup(filter: &MembershipFilter, probes: &[u64]) -> f64 {
            let rounds = 200;
            let start = Instant::now();
            let mut hits = 0u64;
            for _ in 0..rounds {
                for &p in probes {
                    hits += u64::from(filter.contains(black_box(p)));
                }
            }
            black_box(hits);
            start.elapsed().as_nanos() as f64 / (rounds * probes.len()) as f64
        }

        // Same 100k members in a 4M-ID range, once per form.
        let capacity = 4_000_000u64;
        let members = ids(11, 100_000, capacity);
        let probes = ids(12, 50_000, capacity);
        for dense in [true, false] {
            let mut f = MembershipFilter::empty_as(FilterMode::Include, capacity, dense);
            for &id in &members {
                f.add(id);
            }
            println!(
                "{:>6} 100k/4M   {:>6.1} ns/contains  {:>8} bytes",
                if dense { "dense" } else { "sparse" },
                ns_per_lookup(&f, &probes),
                f.approx_heap_bytes()
            );
        }
        // 100k members spread over 2^40: only the sparse form is possible.
        let capacity = 1u64 << 40;
        let members = ids(13, 100_000, capacity);
        let mut probes = ids(14, 25_000, capacity);
        probes.extend(members.iter().take(25_000));
        let mut f = MembershipFilter::empty_as(FilterMode::Include, capacity, false);
        for &id in &members {
            f.add(id);
        }
        println!(
            "{:>6} 100k/2^40 {:>6.1} ns/contains  {:>8} bytes",
            "sparse",
            ns_per_lookup(&f, &probes),
            f.approx_heap_bytes()
        );
    }

    #[test]
    fn huge_capacity_does_not_allocate_by_capacity() {
        let mut f = MembershipFilter::new_include(1 << 40);
        for id in [3u64, 1 << 20, (1 << 40) - 1] {
            f.add(id);
        }
        f.add(1 << 40); // out of range: ignored
        assert_eq!(f.member_count(), 3);
        assert!(f.contains((1 << 40) - 1) && !f.contains(1 << 40));
        assert!(f.approx_heap_bytes() < 1024);
    }

    #[test]
    fn from_ids_picks_the_form_by_density() {
        let thin = MembershipFilter::include_from_ids(1 << 40, &[1, 2, 3]);
        assert!(matches!(thin.bits, Bits::Sparse(_)));
        let small = MembershipFilter::include_from_ids(1000, &[1, 2, 3]);
        assert!(matches!(small.bits, Bits::Dense(_)));
        let all: Vec<u64> = (0..1_000_000).collect();
        let packed = MembershipFilter::include_from_ids(1_000_000, &all);
        assert!(matches!(packed.bits, Bits::Dense(_)));
        assert_eq!(packed.member_count(), 1_000_000);
    }

    #[test]
    fn sparse_deserialize_rejects_ids_beyond_capacity_and_bad_type() {
        let mut f = MembershipFilter::empty_as(FilterMode::Include, 1 << 40, false);
        f.add(1 << 39);
        let mut header = header_for(&f);
        header.vector_count = 1 << 30;
        assert!(MembershipFilter::deserialize(&f.serialize(), &header).is_err());
        let mut header = header_for(&f);
        header.filter_type = 7;
        assert!(MembershipFilter::deserialize(&f.serialize(), &header).is_err());
    }

    #[test]
    fn include_mode_empty_is_empty_view() {
        let filter = MembershipFilter::new_include(100);
        for i in 0..100 {
            assert!(!filter.contains(i));
        }
    }

    #[test]
    fn include_mode_add_and_check() {
        let mut filter = MembershipFilter::new_include(100);
        filter.add(10);
        filter.add(50);
        filter.add(99);

        assert!(filter.contains(10));
        assert!(filter.contains(50));
        assert!(filter.contains(99));
        assert!(!filter.contains(0));
        assert!(!filter.contains(11));
        assert_eq!(filter.member_count(), 3);
    }

    #[test]
    fn exclude_mode() {
        let mut filter = MembershipFilter::new_exclude(100);
        // In exclude mode, all are visible when bitmap is empty
        assert!(filter.contains(0));
        assert!(filter.contains(50));

        // Add to bitmap means "exclude this vector"
        filter.add(50);
        assert!(!filter.contains(50));
        assert!(filter.contains(0));
        assert!(filter.contains(99));
    }

    #[test]
    fn add_remove() {
        let mut filter = MembershipFilter::new_include(64);
        filter.add(10);
        assert_eq!(filter.member_count(), 1);
        assert!(filter.contains(10));

        filter.remove(10);
        assert_eq!(filter.member_count(), 0);
        assert!(!filter.contains(10));
    }

    #[test]
    fn add_out_of_bounds_ignored() {
        let mut filter = MembershipFilter::new_include(10);
        filter.add(100); // beyond vector_count
        assert_eq!(filter.member_count(), 0);
    }

    #[test]
    fn double_add_no_double_count() {
        let mut filter = MembershipFilter::new_include(64);
        filter.add(5);
        filter.add(5);
        assert_eq!(filter.member_count(), 1);
    }

    #[test]
    fn serialize_deserialize_round_trip() {
        let mut filter = MembershipFilter::new_include(200);
        filter.add(0);
        filter.add(63);
        filter.add(64);
        filter.add(127);
        filter.add(199);

        let header = filter.to_header();
        let bitmap_data = filter.serialize();

        let filter2 = MembershipFilter::deserialize(&bitmap_data, &header).unwrap();
        assert_eq!(filter2.vector_count(), 200);
        assert_eq!(filter2.member_count(), 5);
        assert!(filter2.contains(0));
        assert!(filter2.contains(63));
        assert!(filter2.contains(64));
        assert!(filter2.contains(127));
        assert!(filter2.contains(199));
        assert!(!filter2.contains(1));
        assert!(!filter2.contains(100));
    }

    #[test]
    fn generation_bump() {
        let mut filter = MembershipFilter::new_include(10);
        assert_eq!(filter.generation_id(), 0);
        filter.bump_generation();
        assert_eq!(filter.generation_id(), 1);
    }

    #[test]
    fn bitmap_word_boundary() {
        // Test vectors near 64-bit word boundaries
        let mut filter = MembershipFilter::new_include(130);
        filter.add(63);
        filter.add(64);
        filter.add(128);

        assert!(filter.contains(63));
        assert!(filter.contains(64));
        assert!(filter.contains(128));
        assert!(!filter.contains(62));
        assert!(!filter.contains(65));
        assert!(!filter.contains(129));
    }
}
