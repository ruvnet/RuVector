//! Brute-force search over a quantized (bit-packed) index.
//!
//! Distance is asymmetric: the raw f32 query is compared against the
//! dequantized corpus vector, matching how a scalar-quantized ADC-style
//! index scans in production (the query itself is never quantized).

use crate::dataset::l2sq;
use crate::quantize::{dequantize, stored_bytes, QuantizedVector};

#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub id: usize,
    pub dist: f32,
}

pub struct QuantizedIndex {
    pub vectors: Vec<QuantizedVector>,
}

impl QuantizedIndex {
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut scored: Vec<Hit> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, q)| {
                let dec = dequantize(q);
                Hit {
                    id: i,
                    dist: l2sq(query, &dec),
                }
            })
            .collect();
        scored.sort_unstable_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored
    }

    pub fn total_bytes(&self) -> usize {
        self.vectors.iter().map(stored_bytes).sum()
    }

    pub fn mean_bits(&self) -> f32 {
        let total: u32 = self.vectors.iter().map(|v| v.bits.bits() as u32).sum();
        total as f32 / self.vectors.len().max(1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::random_unit_vectors;
    use crate::quantize::{quantize, Bits};

    #[test]
    fn eight_bit_search_finds_self_as_nearest() {
        let corpus = random_unit_vectors(50, 8, 5);
        let vectors: Vec<QuantizedVector> =
            corpus.iter().map(|v| quantize(v, Bits::Eight)).collect();
        let index = QuantizedIndex { vectors };
        let hits = index.search(&corpus[3], 1);
        assert_eq!(hits[0].id, 3);
    }

    #[test]
    fn mean_bits_reports_mix() {
        let corpus = random_unit_vectors(4, 4, 1);
        let vectors: Vec<QuantizedVector> = vec![
            quantize(&corpus[0], Bits::Four),
            quantize(&corpus[1], Bits::Four),
            quantize(&corpus[2], Bits::Eight),
            quantize(&corpus[3], Bits::Eight),
        ];
        let index = QuantizedIndex { vectors };
        assert!((index.mean_bits() - 6.0).abs() < 1e-6);
    }
}
