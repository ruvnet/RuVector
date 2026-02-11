//! K-mer extraction and canonical form computation.
//!
//! K-mers are contiguous subsequences of length k from a DNA sequence.
//! The canonical form is the lexicographically smaller of a k-mer and its
//! reverse complement, enabling strand-agnostic matching.

use super::sequence::{Base, PackedSequence, SequenceError};

/// A k-mer represented as a packed integer (up to k=32)
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Kmer {
    /// Packed 2-bit encoding, least significant bits = first base
    bits: u64,
    /// Length of the k-mer
    k: u8,
}

impl Kmer {
    /// Create a k-mer from a slice of bases
    pub fn from_bases(bases: &[Base]) -> Result<Self, SequenceError> {
        if bases.is_empty() {
            return Err(SequenceError::EmptySequence);
        }
        if bases.len() > 32 {
            return Err(SequenceError::OutOfBounds {
                pos: bases.len(),
                len: 32,
            });
        }
        let mut bits = 0u64;
        for (i, &base) in bases.iter().enumerate() {
            bits |= (base as u64) << (i * 2);
        }
        Ok(Kmer {
            bits,
            k: bases.len() as u8,
        })
    }

    /// Create from ASCII string
    pub fn from_ascii(s: &[u8]) -> Result<Self, SequenceError> {
        let bases: Result<Vec<Base>, _> = s
            .iter()
            .enumerate()
            .map(|(i, &c)| Base::from_ascii(c).ok_or(SequenceError::InvalidBase(c as char, i)))
            .collect();
        Self::from_bases(&bases?)
    }

    /// Reverse complement of this k-mer
    pub fn reverse_complement(&self) -> Self {
        let mut bits = 0u64;
        for i in 0..self.k as usize {
            let base_bits = (self.bits >> (i * 2)) & 0b11;
            let comp = base_bits ^ 0b11; // A(00)<->T(11), C(01)<->G(10)
            bits |= comp << ((self.k as usize - 1 - i) * 2);
        }
        Kmer { bits, k: self.k }
    }

    /// Canonical form: min(kmer, reverse_complement)
    pub fn canonical(&self) -> Self {
        let rc = self.reverse_complement();
        if self.bits <= rc.bits {
            *self
        } else {
            rc
        }
    }

    /// Convert to integer index (for frequency vectors)
    pub fn to_index(&self) -> usize {
        self.bits as usize
    }

    /// Get the raw packed bits.
    pub fn packed(&self) -> u64 {
        self.bits
    }

    /// Get k value
    pub fn k(&self) -> usize {
        self.k as usize
    }

    /// Convert to ASCII string
    pub fn to_ascii_string(&self) -> String {
        let mut s = String::with_capacity(self.k as usize);
        for i in 0..self.k as usize {
            let base_bits = (self.bits >> (i * 2)) & 0b11;
            s.push(match base_bits {
                0 => 'A',
                1 => 'C',
                2 => 'G',
                3 => 'T',
                _ => unreachable!(),
            });
        }
        s
    }
}

impl std::fmt::Debug for Kmer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kmer({})", self.to_ascii_string())
    }
}

/// Iterator that yields k-mers from a packed sequence using a sliding window
pub struct KmerIterator<'a> {
    seq: &'a PackedSequence,
    k: usize,
    pos: usize,
    current_bits: u64,
    mask: u64,
    initialized: bool,
}

impl<'a> KmerIterator<'a> {
    /// Create a new k-mer iterator over a packed sequence
    pub fn new(seq: &'a PackedSequence, k: usize) -> Self {
        let mask = if k >= 32 {
            u64::MAX
        } else {
            (1u64 << (k * 2)) - 1
        };
        KmerIterator {
            seq,
            k,
            pos: 0,
            current_bits: 0,
            mask,
            initialized: false,
        }
    }
}

impl<'a> Iterator for KmerIterator<'a> {
    type Item = Kmer;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + self.k > self.seq.len() {
            return None;
        }

        if !self.initialized {
            self.current_bits = 0;
            for i in 0..self.k {
                let base = self.seq.get(i)? as u64;
                self.current_bits |= base << (i * 2);
            }
            self.initialized = true;
        } else {
            self.current_bits >>= 2;
            let new_base = self.seq.get(self.pos + self.k - 1)? as u64;
            self.current_bits |= new_base << ((self.k - 1) * 2);
            self.current_bits &= self.mask;
        }

        let kmer = Kmer {
            bits: self.current_bits,
            k: self.k as u8,
        };
        self.pos += 1;
        Some(kmer)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.seq.len() < self.k {
            return (0, Some(0));
        }
        let remaining = self.seq.len() - self.k + 1 - self.pos;
        (remaining, Some(remaining))
    }
}

/// Extract canonical k-mers with their positions
pub fn extract_canonical_kmers(seq: &PackedSequence, k: usize) -> Vec<(Kmer, usize)> {
    KmerIterator::new(seq, k)
        .enumerate()
        .map(|(pos, kmer)| (kmer.canonical(), pos))
        .collect()
}

/// Compute k-mer frequency vector (bag-of-words embedding)
/// Returns a vector of length 4^k with L1-normalized frequencies
pub fn kmer_frequency_vector(seq: &PackedSequence, k: usize) -> Vec<f32> {
    let vocab_size = 4usize.pow(k as u32);
    let mut freqs = vec![0.0f32; vocab_size];
    let mut count = 0usize;

    for kmer in KmerIterator::new(seq, k) {
        let canonical = kmer.canonical();
        let idx = canonical.to_index();
        if idx < vocab_size {
            freqs[idx] += 1.0;
            count += 1;
        }
    }

    if count > 0 {
        let total = count as f32;
        for f in &mut freqs {
            *f /= total;
        }
    }

    freqs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmer_canonical() {
        let kmer = Kmer::from_ascii(b"ACG").unwrap();
        let rc = kmer.reverse_complement();
        assert_eq!(rc.to_ascii_string(), "CGT");
        let canonical = kmer.canonical();
        assert_eq!(canonical.to_ascii_string(), "ACG");
    }

    #[test]
    fn test_kmer_iterator_count() {
        let seq = PackedSequence::from_ascii(b"ACGTACGTACGT").unwrap();
        let count = KmerIterator::new(&seq, 6).count();
        assert_eq!(count, 12 - 6 + 1);
    }

    #[test]
    fn test_frequency_vector() {
        let seq = PackedSequence::from_ascii(b"ACGTACGTACGT").unwrap();
        let freqs = kmer_frequency_vector(&seq, 2);
        assert_eq!(freqs.len(), 16);
        let sum: f32 = freqs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Should be L1 normalized: sum={}", sum);
    }

    #[test]
    fn test_sliding_window_consistency() {
        let seq = PackedSequence::from_ascii(b"ACGTACGT").unwrap();
        let kmers: Vec<Kmer> = KmerIterator::new(&seq, 4).collect();
        assert_eq!(kmers.len(), 5);
        assert_eq!(kmers[0].to_ascii_string(), "ACGT");
        assert_eq!(kmers[1].to_ascii_string(), "CGTA");
        assert_eq!(kmers[2].to_ascii_string(), "GTAC");
        assert_eq!(kmers[3].to_ascii_string(), "TACG");
        assert_eq!(kmers[4].to_ascii_string(), "ACGT");
    }

    #[test]
    fn test_real_ecoli_kmers() {
        let ecoli = b"AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATT";
        let seq = PackedSequence::from_ascii(ecoli).unwrap();
        let kmers: Vec<_> = KmerIterator::new(&seq, 11).collect();
        assert_eq!(kmers.len(), ecoli.len() - 11 + 1);
    }

    #[test]
    fn test_kmer_from_bases_error() {
        let result = Kmer::from_bases(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_canonical_strand_agnostic() {
        // A k-mer and its reverse complement should have the same canonical form
        let fwd = Kmer::from_ascii(b"AACG").unwrap();
        let rc = fwd.reverse_complement();
        assert_eq!(fwd.canonical(), rc.canonical());
    }

    #[test]
    fn test_extract_canonical_positions() {
        let seq = PackedSequence::from_ascii(b"ACGTACGT").unwrap();
        let positioned = extract_canonical_kmers(&seq, 3);
        assert_eq!(positioned.len(), 6);
        // Positions should be sequential
        for (i, &(_, pos)) in positioned.iter().enumerate() {
            assert_eq!(pos, i);
        }
    }

    #[test]
    fn test_kmer_packed() {
        let kmer = Kmer::from_ascii(b"ACG").unwrap();
        assert_eq!(kmer.packed(), kmer.to_index() as u64);
    }
}
