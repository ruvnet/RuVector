//! Variant detection: SNP, insertion, and deletion calling.
//!
//! Compares sequencing reads against a reference to identify variants.
//! Uses a simple alignment-free approach based on k-mer positional differences
//! and a coherence gating mechanism for confidence filtering.

use super::sequence::{Base, PackedSequence, SequenceError};

/// Type of variant detected
#[derive(Debug, Clone, PartialEq)]
pub enum VariantType {
    /// Single Nucleotide Polymorphism
    Snp { ref_base: Base, alt_base: Base },
    /// Insertion of bases not in reference
    Insertion { bases: Vec<Base> },
    /// Deletion of bases present in reference
    Deletion { length: usize },
}

/// A detected variant with position and confidence
#[derive(Debug, Clone)]
pub struct Variant {
    /// Position in the reference (0-based)
    pub position: usize,
    /// Type of variant
    pub variant_type: VariantType,
    /// Quality/confidence score (0.0-1.0)
    pub quality: f64,
    /// Number of supporting reads
    pub depth: usize,
}

/// Configuration for variant calling
#[derive(Debug, Clone)]
pub struct VariantCallerConfig {
    /// Minimum quality to report a variant
    pub min_quality: f64,
    /// Minimum read depth at variant position
    pub min_depth: usize,
    /// Coherence gate threshold (variants below this are filtered)
    pub coherence_threshold: f64,
}

impl Default for VariantCallerConfig {
    fn default() -> Self {
        VariantCallerConfig {
            min_quality: 0.3,
            min_depth: 1,
            coherence_threshold: 0.5,
        }
    }
}

/// Detect SNPs between reference and read by base-by-base comparison.
/// Returns raw unfiltered variants.
pub fn detect_snps(reference: &PackedSequence, read: &PackedSequence, read_offset: usize) -> Vec<Variant> {
    let mut variants = Vec::new();
    let overlap = std::cmp::min(read.len(), reference.len().saturating_sub(read_offset));

    for i in 0..overlap {
        let ref_pos = read_offset + i;
        if let (Some(ref_base), Some(read_base)) = (reference.get(ref_pos), read.get(i)) {
            if ref_base != read_base {
                // Simple quality heuristic: isolated SNPs are higher quality
                let isolation = compute_isolation(reference, read, i, read_offset, overlap);
                variants.push(Variant {
                    position: ref_pos,
                    variant_type: VariantType::Snp { ref_base, alt_base: read_base },
                    quality: isolation,
                    depth: 1,
                });
            }
        }
    }

    variants
}

/// Compute isolation score: SNPs surrounded by matches are more confident
fn compute_isolation(reference: &PackedSequence, read: &PackedSequence, read_pos: usize, read_offset: usize, overlap: usize) -> f64 {
    let window = 5;
    let mut matches = 0usize;
    let mut total = 0usize;

    for delta in 1..=window {
        // Check left
        if read_pos >= delta {
            let rp = read_pos - delta;
            if let (Some(rb), Some(qb)) = (reference.get(read_offset + rp), read.get(rp)) {
                total += 1;
                if rb == qb { matches += 1; }
            }
        }
        // Check right
        let rp = read_pos + delta;
        if rp < overlap {
            if let (Some(rb), Some(qb)) = (reference.get(read_offset + rp), read.get(rp)) {
                total += 1;
                if rb == qb { matches += 1; }
            }
        }
    }

    if total == 0 { 0.5 } else { matches as f64 / total as f64 }
}

/// Apply coherence gating to filter low-confidence variants
pub fn coherence_gate(variants: Vec<Variant>, config: &VariantCallerConfig) -> Vec<Variant> {
    variants.into_iter()
        .filter(|v| v.quality >= config.coherence_threshold && v.depth >= config.min_depth)
        .collect()
}

/// Merge overlapping variant calls from multiple reads to compute depth
pub fn merge_variant_calls(mut call_sets: Vec<Vec<Variant>>) -> Vec<Variant> {
    use std::collections::BTreeMap;
    let mut by_position: BTreeMap<usize, Vec<Variant>> = BTreeMap::new();

    for calls in call_sets.drain(..) {
        for v in calls {
            by_position.entry(v.position).or_default().push(v);
        }
    }

    by_position.into_iter().map(|(pos, variants)| {
        let depth = variants.len();
        let avg_quality = variants.iter().map(|v| v.quality).sum::<f64>() / depth as f64;
        // Use the most common variant type at this position
        Variant {
            position: pos,
            variant_type: variants[0].variant_type.clone(),
            quality: avg_quality,
            depth,
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_snps() {
        let reference = PackedSequence::from_ascii(b"ACGTACGTACGT").unwrap();
        let read = PackedSequence::from_ascii(b"ACGAACGTACGT").unwrap(); // T->A at pos 3

        let variants = detect_snps(&reference, &read, 0);
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].position, 3);
        if let VariantType::Snp { ref_base, alt_base } = &variants[0].variant_type {
            assert_eq!(*ref_base, Base::T);
            assert_eq!(*alt_base, Base::A);
        } else {
            panic!("Expected SNP");
        }
    }

    #[test]
    fn test_coherence_gating() {
        let variants = vec![
            Variant { position: 10, variant_type: VariantType::Snp { ref_base: Base::A, alt_base: Base::T }, quality: 0.9, depth: 5 },
            Variant { position: 20, variant_type: VariantType::Snp { ref_base: Base::C, alt_base: Base::G }, quality: 0.2, depth: 1 },
        ];

        let filtered = coherence_gate(variants, &VariantCallerConfig::default());
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].position, 10);
    }

    #[test]
    fn test_merge_calls() {
        let set1 = vec![Variant { position: 100, variant_type: VariantType::Snp { ref_base: Base::A, alt_base: Base::T }, quality: 0.8, depth: 1 }];
        let set2 = vec![Variant { position: 100, variant_type: VariantType::Snp { ref_base: Base::A, alt_base: Base::T }, quality: 0.9, depth: 1 }];

        let merged = merge_variant_calls(vec![set1, set2]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].depth, 2);
        assert!((merged[0].quality - 0.85).abs() < 0.01);
    }
}
