//! Per-dimension scalar quantizer with offline calibration.
//!
//! The quantizer maps each float32 dimension to an int8 value using a linear
//! mapping derived from per-dimension min/max statistics computed over a
//! calibration corpus.  The mapping is:
//!
//! ```text
//! q_i = clamp(round((x_i - min_i) / scale_i * 127), -128, 127)
//! ```
//!
//! where `scale_i = (max_i - min_i)`.  Decoding is the inverse.
//!
//! ## Integer-domain distance
//!
//! Squared L2 distance in the int8 domain:
//!
//! ```text
//! dist_sq_int = Σ (a_i - b_i)²   (accumulate in i64 to avoid overflow)
//! ```
//!
//! The int8 domain distance is proportional to (not identical to) the f32
//! domain distance when both vectors are encoded with the same calibration.
//! For HNSW traversal we only need the *ordering* to be consistent, which
//! holds for the int8 domain as long as the quantization error is small.

/// Per-dimension scalar quantizer.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    /// Per-dimension lower bound.
    pub dim_min: Vec<f32>,
    /// Per-dimension scale = (max - min); used to map to [-127, 127].
    pub dim_scale: Vec<f32>,
    /// Number of dimensions.
    pub dims: usize,
}

impl ScalarQuantizer {
    /// Build a calibrated quantizer from a representative set of vectors.
    ///
    /// `calibration_set` may be a subset of the corpus; using ~10 % of the
    /// full dataset is sufficient in practice.
    pub fn calibrate(calibration_set: &[Vec<f32>]) -> Self {
        assert!(
            !calibration_set.is_empty(),
            "calibration set must be non-empty"
        );
        let dims = calibration_set[0].len();

        let mut dim_min = vec![f32::INFINITY; dims];
        let mut dim_max = vec![f32::NEG_INFINITY; dims];

        for v in calibration_set {
            assert_eq!(
                v.len(),
                dims,
                "all calibration vectors must have the same length"
            );
            for (d, &x) in v.iter().enumerate() {
                if x < dim_min[d] {
                    dim_min[d] = x;
                }
                if x > dim_max[d] {
                    dim_max[d] = x;
                }
            }
        }

        let dim_scale: Vec<f32> = dim_min
            .iter()
            .zip(dim_max.iter())
            .map(|(&lo, &hi)| {
                let s = hi - lo;
                if s.abs() < 1e-12 {
                    1.0
                } else {
                    s
                }
            })
            .collect();

        Self {
            dim_min,
            dim_scale,
            dims,
        }
    }

    /// Encode a float32 vector to int8.  Values outside calibration range are clamped.
    pub fn encode(&self, v: &[f32]) -> Vec<i8> {
        assert_eq!(v.len(), self.dims);
        v.iter()
            .enumerate()
            .map(|(d, &x)| {
                let norm = (x - self.dim_min[d]) / self.dim_scale[d]; // [0, 1]
                let q = (norm * 254.0 - 127.0).round().clamp(-128.0, 127.0);
                q as i8
            })
            .collect()
    }

    /// Decode an int8 code back to approximate float32.
    pub fn decode(&self, code: &[i8]) -> Vec<f32> {
        assert_eq!(code.len(), self.dims);
        code.iter()
            .enumerate()
            .map(|(d, &q)| {
                let norm = (q as f32 + 127.0) / 254.0;
                self.dim_min[d] + norm * self.dim_scale[d]
            })
            .collect()
    }

    /// Compute squared L2 distance between two int8-encoded vectors.
    ///
    /// Uses i64 accumulator to prevent overflow (each term is at most 255²).
    #[inline]
    pub fn sq8_l2_sq(a: &[i8], b: &[i8]) -> i64 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc: i64 = 0;
        for (&ai, &bi) in a.iter().zip(b.iter()) {
            let d = (ai as i64) - (bi as i64);
            acc += d * d;
        }
        acc
    }

    /// Asymmetric distance: float32 query against int8 code.
    ///
    /// Decodes the code and computes f32 L2; more accurate than
    /// encoding the query to int8 first.  Used for reranking.
    pub fn asymmetric_l2_sq(&self, query: &[f32], code: &[i8]) -> f32 {
        let decoded = self.decode(code);
        query
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    /// Encode a query vector and compute int8 L2² against a stored code.
    ///
    /// Slightly faster than asymmetric (no decode allocation) but introduces
    /// double-quantization error.
    pub fn sq8_query_l2_sq(&self, query: &[f32], code: &[i8]) -> i64 {
        let q_code = self.encode(query);
        Self::sq8_l2_sq(&q_code, code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_accuracy() {
        let corpus: Vec<Vec<f32>> = (0..200)
            .map(|i| vec![i as f32 * 0.01, 1.0 - i as f32 * 0.005])
            .collect();
        let quant = ScalarQuantizer::calibrate(&corpus);
        let v = vec![0.5_f32, 0.5_f32];
        let code = quant.encode(&v);
        let decoded = quant.decode(&code);
        // Expect reconstruction error < 1% of range
        for (&orig, &dec) in v.iter().zip(decoded.iter()) {
            assert!(
                (orig - dec).abs() < 0.01,
                "roundtrip error {}",
                (orig - dec).abs()
            );
        }
    }

    #[test]
    fn sq8_ordering_preserves_rank() {
        let corpus: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 / 100.0, 0.5]).collect();
        let quant = ScalarQuantizer::calibrate(&corpus);
        let q = vec![0.5_f32, 0.5_f32];
        let q_code = quant.encode(&q);

        // Closest vector in f32 should also be closest in sq8
        let f32_closest = corpus
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da: f32 = a.iter().zip(q.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
                let db: f32 = b.iter().zip(q.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
                da.partial_cmp(&db).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();

        let sq8_closest = corpus
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| ScalarQuantizer::sq8_l2_sq(&quant.encode(v), &q_code))
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            f32_closest, sq8_closest,
            "nearest neighbour mismatch under sq8"
        );
    }
}
