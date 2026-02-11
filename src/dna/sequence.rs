//! Core 2-bit packed DNA sequence types.
//!
//! DNA bases are encoded as 2-bit values packed into `u64` words,
//! achieving 4x memory reduction compared to ASCII representation.
//! Each `u64` stores up to 32 bases.
//!
//! Encoding: A=0b00, C=0b01, G=0b10, T=0b11

#[allow(dead_code)]

/// Errors that can occur during sequence operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceError {
    /// An invalid (non-ACGT) base was encountered at the given position.
    InvalidBase(char, usize),
    /// Index out of bounds: `pos` was accessed in a sequence of length `len`.
    OutOfBounds { pos: usize, len: usize },
    /// The input sequence was empty.
    EmptySequence,
}

impl std::fmt::Display for SequenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SequenceError::InvalidBase(c, pos) => {
                write!(f, "invalid base '{}' at position {}", c, pos)
            }
            SequenceError::OutOfBounds { pos, len } => {
                write!(f, "index {} out of bounds (length {})", pos, len)
            }
            SequenceError::EmptySequence => write!(f, "empty sequence"),
        }
    }
}

impl std::error::Error for SequenceError {}

/// A single DNA base with 2-bit encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Base {
    A = 0b00,
    C = 0b01,
    G = 0b10,
    T = 0b11,
}

impl Base {
    /// Complement using XOR trick: A<->T (00<->11), C<->G (01<->10).
    #[inline]
    pub fn complement(self) -> Base {
        match (self as u8) ^ 0b11 {
            0b00 => Base::A,
            0b01 => Base::C,
            0b10 => Base::G,
            0b11 => Base::T,
            _ => unreachable!(),
        }
    }

    /// Convert from ASCII byte. Returns `None` for non-ACGT characters.
    #[inline]
    pub fn from_ascii(byte: u8) -> Option<Base> {
        match byte {
            b'A' | b'a' => Some(Base::A),
            b'C' | b'c' => Some(Base::C),
            b'G' | b'g' => Some(Base::G),
            b'T' | b't' => Some(Base::T),
            _ => None,
        }
    }

    /// Convert to ASCII byte.
    #[inline]
    pub fn to_ascii(self) -> u8 {
        match self {
            Base::A => b'A',
            Base::C => b'C',
            Base::G => b'G',
            Base::T => b'T',
        }
    }

    /// Decode a 2-bit value to a Base.
    #[inline]
    pub fn from_bits(bits: u8) -> Base {
        match bits & 0b11 {
            0b00 => Base::A,
            0b01 => Base::C,
            0b10 => Base::G,
            0b11 => Base::T,
            _ => unreachable!(),
        }
    }
}

/// A 2-bit packed DNA sequence. Stores 32 bases per `u64` word.
///
/// Memory layout: bases are stored from the MSB side of each word.
/// Base at index `i` within word `w` occupies bits `(62 - 2*(i%32))..=(63 - 2*(i%32))`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedSequence {
    /// Packed 2-bit encoded bases, 32 per u64 word.
    data: Vec<u64>,
    /// Number of bases in the sequence.
    len: usize,
}

/// PHRED+33 encoded quality scores for sequencing reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualityScores {
    /// Raw PHRED+33 quality bytes.
    scores: Vec<u8>,
}

/// A sequencing read with sequence and quality data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceRead {
    /// Read identifier.
    pub id: String,
    /// The DNA sequence.
    pub sequence: PackedSequence,
    /// Quality scores (PHRED+33).
    pub quality: QualityScores,
}

// ---------------------------------------------------------------------------
// PackedSequence implementation
// ---------------------------------------------------------------------------

const BASES_PER_WORD: usize = 32;

impl PackedSequence {
    /// Create an empty packed sequence.
    pub fn new() -> Self {
        PackedSequence {
            data: Vec::new(),
            len: 0,
        }
    }

    /// Create a packed sequence from an ASCII DNA string.
    ///
    /// Returns `Ok` with the packed sequence. Non-ACGT characters are
    /// silently filtered out. Each valid base is packed as 2 bits into
    /// `u64` words (32 bases per word).
    pub fn from_ascii(ascii: &[u8]) -> Result<Self, SequenceError> {
        let bases: Vec<Base> = ascii.iter().filter_map(|&b| Base::from_ascii(b)).collect();
        let len = bases.len();
        let num_words = if len == 0 {
            0
        } else {
            (len + BASES_PER_WORD - 1) / BASES_PER_WORD
        };
        let mut data = vec![0u64; num_words];

        for (i, &base) in bases.iter().enumerate() {
            let word_idx = i / BASES_PER_WORD;
            let bit_offset = 62 - 2 * (i % BASES_PER_WORD);
            data[word_idx] |= (base as u64) << bit_offset;
        }

        Ok(PackedSequence { data, len })
    }

    /// Number of bases in the sequence.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the sequence is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the base at a given index. Returns `None` if out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<Base> {
        if index >= self.len {
            return None;
        }
        let word_idx = index / BASES_PER_WORD;
        let bit_offset = 62 - 2 * (index % BASES_PER_WORD);
        let bits = ((self.data[word_idx] >> bit_offset) & 0b11) as u8;
        Some(Base::from_bits(bits))
    }

    /// Get the base at a given index, panicking if out of bounds.
    #[inline]
    pub fn get_unchecked(&self, index: usize) -> Base {
        self.get(index).expect("index out of bounds")
    }

    /// Compute the reverse complement of this sequence.
    ///
    /// Uses the XOR trick: complement of a 2-bit base is `bits ^ 0b11`.
    /// The entire sequence is then reversed.
    pub fn reverse_complement(&self) -> PackedSequence {
        let len = self.len;
        if len == 0 {
            return PackedSequence::new();
        }

        let num_words = (len + BASES_PER_WORD - 1) / BASES_PER_WORD;
        let mut result = vec![0u64; num_words];

        for i in 0..len {
            let src_base = self.get_unchecked(i);
            let comp = src_base.complement();
            let dest_idx = len - 1 - i;
            let word_idx = dest_idx / BASES_PER_WORD;
            let bit_offset = 62 - 2 * (dest_idx % BASES_PER_WORD);
            result[word_idx] |= (comp as u64) << bit_offset;
        }

        PackedSequence { data: result, len }
    }

    /// Compute the GC content as a fraction in [0.0, 1.0].
    ///
    /// GC content is the proportion of bases that are G or C.
    pub fn gc_content(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        let gc_count = (0..self.len)
            .filter(|&i| matches!(self.get_unchecked(i), Base::G | Base::C))
            .count();
        gc_count as f64 / self.len as f64
    }

    /// Extract a subsequence from `start` (inclusive) to `end` (exclusive).
    ///
    /// # Panics
    /// Panics if `start > end` or `end > len`.
    pub fn subsequence(&self, start: usize, end: usize) -> PackedSequence {
        assert!(start <= end, "start ({}) must be <= end ({})", start, end);
        assert!(end <= self.len, "end ({}) out of bounds (len={})", end, self.len);

        let sub_len = end - start;
        if sub_len == 0 {
            return PackedSequence::new();
        }

        let num_words = (sub_len + BASES_PER_WORD - 1) / BASES_PER_WORD;
        let mut data = vec![0u64; num_words];

        for i in 0..sub_len {
            let base = self.get_unchecked(start + i);
            let word_idx = i / BASES_PER_WORD;
            let bit_offset = 62 - 2 * (i % BASES_PER_WORD);
            data[word_idx] |= (base as u64) << bit_offset;
        }

        PackedSequence { data, len: sub_len }
    }

    /// Iterator over bases in the sequence.
    pub fn iter(&self) -> PackedSequenceIter<'_> {
        PackedSequenceIter { seq: self, pos: 0 }
    }

    /// Convert the packed sequence back to an ASCII byte vector.
    pub fn to_ascii(&self) -> Vec<u8> {
        self.iter().map(|b| b.to_ascii()).collect()
    }

    /// Memory usage in bytes for the packed data (excluding Vec overhead).
    pub fn packed_bytes(&self) -> usize {
        self.data.len() * 8
    }

    /// Memory that would be needed for ASCII storage.
    pub fn ascii_bytes(&self) -> usize {
        self.len
    }
}

impl Default for PackedSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over bases in a `PackedSequence`.
pub struct PackedSequenceIter<'a> {
    seq: &'a PackedSequence,
    pos: usize,
}

impl<'a> Iterator for PackedSequenceIter<'a> {
    type Item = Base;

    #[inline]
    fn next(&mut self) -> Option<Base> {
        if self.pos < self.seq.len() {
            let base = self.seq.get_unchecked(self.pos);
            self.pos += 1;
            Some(base)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.seq.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for PackedSequenceIter<'a> {}

// ---------------------------------------------------------------------------
// QualityScores implementation
// ---------------------------------------------------------------------------

impl QualityScores {
    /// Create quality scores from raw PHRED+33 encoded bytes.
    pub fn from_phred33(raw: &[u8]) -> Self {
        QualityScores {
            scores: raw.to_vec(),
        }
    }

    /// Create quality scores where all bases have the given PHRED score.
    pub fn uniform(len: usize, phred: u8) -> Self {
        QualityScores {
            scores: vec![phred + 33; len],
        }
    }

    /// Get the PHRED quality score (0-based) at a given position.
    #[inline]
    pub fn phred_at(&self, index: usize) -> u8 {
        self.scores[index].saturating_sub(33)
    }

    /// Number of quality scores.
    #[inline]
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Whether the quality scores are empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Average PHRED quality score.
    pub fn mean_quality(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.scores.iter().map(|&q| (q.saturating_sub(33)) as u64).sum();
        sum as f64 / self.scores.len() as f64
    }

    /// Raw PHRED+33 bytes.
    pub fn raw(&self) -> &[u8] {
        &self.scores
    }
}

impl Default for QualityScores {
    fn default() -> Self {
        QualityScores { scores: Vec::new() }
    }
}

// ---------------------------------------------------------------------------
// SequenceRead implementation
// ---------------------------------------------------------------------------

impl SequenceRead {
    /// Create a new sequence read.
    pub fn new(
        id: String,
        sequence: PackedSequence,
        quality: QualityScores,
    ) -> Self {
        SequenceRead {
            id,
            sequence,
            quality,
        }
    }

    /// Number of bases in this read.
    #[inline]
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Whether the read is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// PhiX174 partial sequence (GenBank J02482.1).
    const PHIX174: &[u8] =
        b"GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTTGATAAAGCAGGAATTACTACTGCTTGTTTA";

    #[test]
    fn base_complement() {
        assert_eq!(Base::A.complement(), Base::T);
        assert_eq!(Base::T.complement(), Base::A);
        assert_eq!(Base::C.complement(), Base::G);
        assert_eq!(Base::G.complement(), Base::C);
    }

    #[test]
    fn base_from_ascii() {
        assert_eq!(Base::from_ascii(b'A'), Some(Base::A));
        assert_eq!(Base::from_ascii(b'c'), Some(Base::C));
        assert_eq!(Base::from_ascii(b'G'), Some(Base::G));
        assert_eq!(Base::from_ascii(b't'), Some(Base::T));
        assert_eq!(Base::from_ascii(b'N'), None);
        assert_eq!(Base::from_ascii(b'X'), None);
    }

    #[test]
    fn packed_from_ascii_roundtrip() {
        let seq = PackedSequence::from_ascii(PHIX174).unwrap();
        assert_eq!(seq.len(), PHIX174.len());
        assert_eq!(seq.to_ascii(), PHIX174.to_vec());
    }

    #[test]
    fn packed_memory_efficiency() {
        let seq = PackedSequence::from_ascii(PHIX174).unwrap();
        assert!(seq.packed_bytes() < seq.ascii_bytes());
        // Verify significant compression ratio
        assert!(seq.ascii_bytes() as f64 / seq.packed_bytes() as f64 >= 3.0);
    }

    #[test]
    fn packed_individual_bases() {
        let seq = PackedSequence::from_ascii(b"ACGT").unwrap();
        assert_eq!(seq.get(0), Some(Base::A));
        assert_eq!(seq.get(1), Some(Base::C));
        assert_eq!(seq.get(2), Some(Base::G));
        assert_eq!(seq.get(3), Some(Base::T));
        assert_eq!(seq.get(4), None);
    }

    #[test]
    fn reverse_complement_simple() {
        // ACGT -> reverse = TGCA -> complement of each = ACGT
        let seq = PackedSequence::from_ascii(b"ACGT").unwrap();
        let rc = seq.reverse_complement();
        assert_eq!(rc.to_ascii(), b"ACGT".to_vec());
    }

    #[test]
    fn reverse_complement_asymmetric() {
        // AACG -> reverse = GCAA -> complement = CGTT
        let seq = PackedSequence::from_ascii(b"AACG").unwrap();
        let rc = seq.reverse_complement();
        assert_eq!(rc.to_ascii(), b"CGTT".to_vec());
    }

    #[test]
    fn reverse_complement_phix174() {
        let seq = PackedSequence::from_ascii(PHIX174).unwrap();
        let rc = seq.reverse_complement();
        // Double reverse complement should return the original
        let rc2 = rc.reverse_complement();
        assert_eq!(rc2.to_ascii(), PHIX174.to_vec());
        // RC should have same length
        assert_eq!(rc.len(), PHIX174.len());
    }

    #[test]
    fn gc_content_all_gc() {
        let seq = PackedSequence::from_ascii(b"GCGCGC").unwrap();
        assert!((seq.gc_content() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gc_content_no_gc() {
        let seq = PackedSequence::from_ascii(b"ATATAT").unwrap();
        assert!((seq.gc_content() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn gc_content_phix174() {
        let seq = PackedSequence::from_ascii(PHIX174).unwrap();
        let gc = seq.gc_content();
        // PhiX174 has ~44% GC content genome-wide; first 100bp should be in range
        assert!(gc > 0.3 && gc < 0.6, "GC content {} out of expected range", gc);
    }

    #[test]
    fn subsequence_extraction() {
        let seq = PackedSequence::from_ascii(PHIX174).unwrap();
        let sub = seq.subsequence(0, 10);
        assert_eq!(sub.len(), 10);
        assert_eq!(sub.to_ascii(), b"GAGTTTTATC".to_vec());

        let end = seq.len();
        let start = end.saturating_sub(10);
        let sub2 = seq.subsequence(start, end);
        assert_eq!(sub2.len(), end - start);
    }

    #[test]
    fn subsequence_full() {
        let seq = PackedSequence::from_ascii(b"ACGT").unwrap();
        let sub = seq.subsequence(0, 4);
        assert_eq!(sub.to_ascii(), b"ACGT".to_vec());
    }

    #[test]
    fn subsequence_empty() {
        let seq = PackedSequence::from_ascii(b"ACGT").unwrap();
        let sub = seq.subsequence(2, 2);
        assert!(sub.is_empty());
    }

    #[test]
    fn iterator_count() {
        let seq = PackedSequence::from_ascii(PHIX174).unwrap();
        assert_eq!(seq.iter().count(), PHIX174.len());
    }

    #[test]
    fn iterator_collect() {
        let seq = PackedSequence::from_ascii(b"ACGT").unwrap();
        let bases: Vec<Base> = seq.iter().collect();
        assert_eq!(bases, vec![Base::A, Base::C, Base::G, Base::T]);
    }

    #[test]
    fn empty_sequence() {
        let seq = PackedSequence::new();
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
        assert_eq!(seq.gc_content(), 0.0);
        let rc = seq.reverse_complement();
        assert!(rc.is_empty());
    }

    #[test]
    fn long_sequence_across_word_boundary() {
        // 66 bases = 3 u64 words (32+32+2)
        let long = b"ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC";
        assert_eq!(long.len(), 66);
        let seq = PackedSequence::from_ascii(long).unwrap();
        assert_eq!(seq.len(), 66);
        assert_eq!(seq.to_ascii(), long.to_vec());
    }

    #[test]
    fn non_acgt_filtered() {
        let seq = PackedSequence::from_ascii(b"ACNGTX").unwrap();
        assert_eq!(seq.len(), 4); // N and X filtered out
        assert_eq!(seq.to_ascii(), b"ACGT".to_vec());
    }

    #[test]
    fn quality_scores_phred() {
        // Illumina-style quality string
        let raw = b"IIIIIIIIII"; // PHRED 40
        let qual = QualityScores::from_phred33(raw);
        assert_eq!(qual.len(), 10);
        assert_eq!(qual.phred_at(0), 40);
        assert!((qual.mean_quality() - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn quality_scores_uniform() {
        let qual = QualityScores::uniform(50, 30);
        assert_eq!(qual.len(), 50);
        assert_eq!(qual.phred_at(0), 30);
        assert!((qual.mean_quality() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sequence_read_construction() {
        let read = SequenceRead::new(
            "read1".to_string(),
            PackedSequence::from_ascii(b"ACGT").unwrap(),
            QualityScores::uniform(4, 30),
        );
        assert_eq!(read.len(), 4);
        assert_eq!(read.id, "read1");
    }

    #[test]
    fn sequence_error_display() {
        let err = SequenceError::InvalidBase('N', 5);
        assert!(format!("{}", err).contains("N"));
        assert!(format!("{}", err).contains("5"));

        let err = SequenceError::OutOfBounds { pos: 10, len: 5 };
        assert!(format!("{}", err).contains("10"));

        let err = SequenceError::EmptySequence;
        assert!(format!("{}", err).contains("empty"));
    }
}
