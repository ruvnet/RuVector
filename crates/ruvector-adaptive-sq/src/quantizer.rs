//! Scalar quantization encode/decode for 8-bit and 16-bit precision tiers.
//!
//! Both tiers use uniform linear quantization with per-dimension bounds.
//! The 8-bit tier maps each f32 to a u8 in [0, 255]; the 16-bit tier maps
//! to a u16 in [0, 65535].  Quantization error for a dimension with value
//! range R is:
//!
//!   8-bit  RMS error ≈ R / (255 × √12)
//!   16-bit RMS error ≈ R / (65535 × √12)
//!
//! For a 32-dim vector with global range ~6 (data from N(0,1)):
//!   8-bit  total L2 error ≈ √32 × (6/(255×√12)) ≈ 0.038
//!   16-bit total L2 error ≈ √32 × (6/(65535×√12)) ≈ 0.00015
//!
//! The 8-bit error is large relative to intra-cluster distances when
//! clusters are tight (σ < 0.05 at dim=32), causing recall degradation
//! that the 16-bit tier avoids.

/// Encode one f32 value to 8-bit given dimension bounds.
#[inline]
pub fn encode_u8(val: f32, min: f32, range: f32) -> u8 {
    if range < 1e-12 {
        return 0;
    }
    ((val - min) / range * 255.0).clamp(0.0, 255.0) as u8
}

/// Decode one 8-bit code back to f32.
#[inline]
pub fn decode_u8(code: u8, min: f32, range: f32) -> f32 {
    min + code as f32 * (range / 255.0)
}

/// Encode one f32 value to 16-bit given dimension bounds.
#[inline]
pub fn encode_u16(val: f32, min: f32, range: f32) -> u16 {
    if range < 1e-12 {
        return 0;
    }
    ((val - min) / range * 65535.0).clamp(0.0, 65535.0) as u16
}

/// Decode one 16-bit code back to f32.
#[inline]
pub fn decode_u16(code: u16, min: f32, range: f32) -> f32 {
    min + code as f32 * (range / 65535.0)
}

/// Compute per-dimension min and value-range over a dataset.
///
/// Returns (mins[dim], ranges[dim]) where range = max - min.
/// Dimensions with zero range get range=0 so encode always returns 0.
pub fn compute_global_bounds(vectors: &[Vec<f32>], dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mins = vec![f32::MAX; dim];
    let mut maxs = vec![f32::NEG_INFINITY; dim];
    for v in vectors {
        for (d, &x) in v.iter().enumerate().take(dim) {
            if x < mins[d] {
                mins[d] = x;
            }
            if x > maxs[d] {
                maxs[d] = x;
            }
        }
    }
    let ranges: Vec<f32> = mins
        .iter()
        .zip(maxs.iter())
        .map(|(&mn, &mx)| (mx - mn).max(0.0))
        .collect();
    (mins, ranges)
}

/// Squared L2 distance between a decoded (reconstructed) vector and a query.
///
/// Avoids materialising the full reconstructed vector.
pub fn l2_sq_u8(codes: &[u8], query: &[f32], mins: &[f32], ranges: &[f32]) -> f32 {
    codes
        .iter()
        .zip(query.iter())
        .zip(mins.iter())
        .zip(ranges.iter())
        .map(|(((c, q), mn), rng)| {
            let decoded = decode_u8(*c, *mn, *rng);
            (decoded - q).powi(2)
        })
        .sum()
}

/// Squared L2 distance between a 16-bit decoded vector and a query.
pub fn l2_sq_u16(codes: &[u16], query: &[f32], mins: &[f32], ranges: &[f32]) -> f32 {
    codes
        .iter()
        .zip(query.iter())
        .zip(mins.iter())
        .zip(ranges.iter())
        .map(|(((c, q), mn), rng)| {
            let decoded = decode_u16(*c, *mn, *rng);
            (decoded - q).powi(2)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_u8_within_tolerance() {
        let min = -3.0f32;
        let range = 6.0f32;
        let vals: Vec<f32> = vec![-3.0, -1.5, 0.0, 1.5, 2.99];
        for v in vals {
            let code = encode_u8(v, min, range);
            let rec = decode_u8(code, min, range);
            let err = (rec - v).abs();
            // Max quantization step = 6/255 ≈ 0.0235
            assert!(
                err <= 6.0 / 255.0 + 1e-5,
                "u8 roundtrip error {err} at v={v}"
            );
        }
    }

    #[test]
    fn roundtrip_u16_within_tolerance() {
        let min = -3.0f32;
        let range = 6.0f32;
        let vals: Vec<f32> = vec![-3.0, -1.5, 0.0, 1.5, 2.99];
        for v in vals {
            let code = encode_u16(v, min, range);
            let rec = decode_u16(code, min, range);
            let err = (rec - v).abs();
            // Max quantization step = 6/65535 ≈ 0.0000916
            assert!(
                err <= 6.0 / 65535.0 + 1e-6,
                "u16 roundtrip error {err} at v={v}"
            );
        }
    }

    #[test]
    fn u16_more_precise_than_u8() {
        let min = 0.0f32;
        let range = 1.0f32;
        let v = 0.5012345f32;
        let err8 = (decode_u8(encode_u8(v, min, range), min, range) - v).abs();
        let err16 = (decode_u16(encode_u16(v, min, range), min, range) - v).abs();
        assert!(err16 < err8, "16-bit must be more precise than 8-bit");
    }

    #[test]
    fn compute_bounds_correctness() {
        let vecs = vec![vec![1.0f32, -2.0, 3.0], vec![-1.0, 4.0, 0.5]];
        let (mins, ranges) = compute_global_bounds(&vecs, 3);
        assert!((mins[0] - (-1.0)).abs() < 1e-6);
        assert!((mins[1] - (-2.0)).abs() < 1e-6);
        assert!((ranges[0] - 2.0).abs() < 1e-6);
        assert!((ranges[1] - 6.0).abs() < 1e-6);
    }
}
