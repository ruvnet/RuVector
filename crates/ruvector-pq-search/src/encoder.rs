//! PQ encode/decode utilities.

use crate::codebook::PqCodebook;

/// Encode a full-precision vector into M bytes using the trained codebook.
pub fn encode_vector(cb: &PqCodebook, vector: &[f32]) -> Vec<u8> {
    assert_eq!(vector.len(), cb.dim, "vector dimension mismatch");
    (0..cb.config.m)
        .map(|sub| {
            let sv = &vector[sub * cb.sub_dim..(sub + 1) * cb.sub_dim];
            cb.nearest_centroid(sub, sv)
        })
        .collect()
}

/// Decode a PQ code back to an approximate full-precision vector.
/// The reconstructed vector is the concatenation of the nearest centroids.
pub fn decode_vector(cb: &PqCodebook, code: &[u8]) -> Vec<f32> {
    assert_eq!(code.len(), cb.config.m, "code length must equal m");
    let mut out = Vec::with_capacity(cb.dim);
    for (sub, &c) in code.iter().enumerate() {
        out.extend_from_slice(cb.centroid(sub, c as usize));
    }
    out
}

/// Compute the quantization error (L2²) between original and reconstructed vector.
pub fn quantization_error(cb: &PqCodebook, vector: &[f32]) -> f32 {
    let code = encode_vector(cb, vector);
    let reconstructed = decode_vector(cb, &code);
    crate::l2_sq(vector, &reconstructed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::{PqCodebook, PqConfig};

    fn make_codebook() -> (PqCodebook, Vec<f32>, usize) {
        let dim = 32usize;
        let n = 100usize;
        let config = PqConfig::new(4, 16);
        let vecs: Vec<f32> = (0..n * dim).map(|i| (i as f32).sin()).collect();
        let cb = PqCodebook::train(config, &vecs, dim);
        (cb, vecs, dim)
    }

    #[test]
    fn encode_returns_m_bytes() {
        let (cb, vecs, dim) = make_codebook();
        let v = &vecs[0..dim];
        let code = encode_vector(&cb, v);
        assert_eq!(code.len(), 4);
        assert!(code.iter().all(|&c| (c as usize) < 16));
    }

    #[test]
    fn decode_returns_correct_dimension() {
        let (cb, vecs, dim) = make_codebook();
        let code = encode_vector(&cb, &vecs[0..dim]);
        let reconstructed = decode_vector(&cb, &code);
        assert_eq!(reconstructed.len(), dim);
    }

    #[test]
    fn quantization_error_is_non_negative() {
        let (cb, vecs, dim) = make_codebook();
        let err = quantization_error(&cb, &vecs[0..dim]);
        assert!(err >= 0.0);
    }
}
