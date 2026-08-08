//! In-core implementations of the [`VectorCodec`] plane: Fp32, Fp16, Int8,
//! and the Turbo4 adapter over `ruvector-turboquant` (ADR-297 §1).
//!
//! Fp32/Fp16/Int8 score by decoding — they are the verification / hot tiers,
//! where fidelity matters more than per-candidate cost. Turbo4 scores
//! directly on packed codes (never reconstructs), as required by ADR-296.

use super::{metric_distance, CodecKind, EncodedQuery, VectorCodec};
use crate::error::{Result, RuvectorError};
use crate::types::DistanceMetric;
use ruvector_turboquant as tq;

fn check_dim(expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(RuvectorError::DimensionMismatch { expected, actual });
    }
    Ok(())
}

fn check_blob(expected: usize, actual: usize) -> Result<()> {
    if expected != actual {
        return Err(RuvectorError::InvalidInput(format!(
            "encoded blob length {actual} != expected {expected}"
        )));
    }
    Ok(())
}

/// Query that scores by decoding the candidate blob to f32 first.
struct DecodingQuery {
    q: Vec<f32>,
    metric: DistanceMetric,
    decode: fn(&[u8]) -> Vec<f32>,
}

impl EncodedQuery for DecodingQuery {
    fn distance_to(&self, blob: &[u8]) -> f32 {
        let v = (self.decode)(blob);
        metric_distance(self.metric, &self.q, &v)
    }
}

// ---------------------------------------------------------------------------
// Fp32
// ---------------------------------------------------------------------------

/// Identity codec: 4 bytes/dim, exact. The verification anchor of the plane.
pub struct Fp32Codec {
    dim: usize,
    metric: DistanceMetric,
}

impl Fp32Codec {
    pub fn new(dim: usize, metric: DistanceMetric) -> Self {
        Self { dim, metric }
    }
}

fn fp32_decode(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

impl VectorCodec for Fp32Codec {
    fn kind(&self) -> CodecKind {
        CodecKind::Fp32
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn encoded_len(&self) -> usize {
        self.dim * 4
    }
    fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        check_dim(self.dim, v.len())?;
        Ok(v.iter().flat_map(|x| x.to_le_bytes()).collect())
    }
    fn decode(&self, blob: &[u8]) -> Result<Vec<f32>> {
        check_blob(self.encoded_len(), blob.len())?;
        Ok(fp32_decode(blob))
    }
    fn distance(&self, a: &[u8], b: &[u8]) -> Result<f32> {
        check_blob(self.encoded_len(), a.len())?;
        check_blob(self.encoded_len(), b.len())?;
        Ok(metric_distance(
            self.metric,
            &fp32_decode(a),
            &fp32_decode(b),
        ))
    }
    fn make_query(&self, q: &[f32]) -> Result<Box<dyn EncodedQuery>> {
        check_dim(self.dim, q.len())?;
        Ok(Box::new(DecodingQuery {
            q: q.to_vec(),
            metric: self.metric,
            decode: fp32_decode,
        }))
    }
}

// ---------------------------------------------------------------------------
// Fp16
// ---------------------------------------------------------------------------

/// IEEE 754 binary16: 2 bytes/dim. Hot-tier codec (~1e-3 relative error).
/// Conversion is implemented in-crate (no `half` dependency) with
/// round-to-nearest-even, matching hardware semantics.
pub struct Fp16Codec {
    dim: usize,
    metric: DistanceMetric,
}

impl Fp16Codec {
    pub fn new(dim: usize, metric: DistanceMetric) -> Self {
        Self { dim, metric }
    }
}

/// f32 → binary16 bits, round-to-nearest-even; overflow → ±inf.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp == 255 {
        // Inf / NaN (preserve NaN-ness with a set mantissa bit).
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7C00; // overflow → inf
    }
    if unbiased >= -14 {
        // Normal half.
        let m16 = (mant >> 13) as u16;
        let h = sign | (((unbiased + 15) as u16) << 10) | m16;
        let round = mant & 0x1FFF;
        // RNE; a carry out of the mantissa correctly increments the exponent.
        if round > 0x1000 || (round == 0x1000 && (m16 & 1) == 1) {
            h + 1
        } else {
            h
        }
    } else if unbiased >= -24 {
        // Subnormal half.
        let shift = (13 + (-14 - unbiased)) as u32;
        let full = mant | 0x0080_0000;
        let m16 = (full >> shift) as u16;
        let rem = full & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let h = sign | m16;
        if rem > half || (rem == half && (m16 & 1) == 1) {
            h + 1
        } else {
            h
        }
    } else {
        sign // underflow → ±0
    }
}

/// binary16 bits → f32 (exact).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: renormalize.
            let mut e = 113u32; // 127 - 15 + 1
            let mut m = mant;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x03FF) << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn fp16_decode(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(2)
        .map(|c| f16_bits_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
        .collect()
}

impl VectorCodec for Fp16Codec {
    fn kind(&self) -> CodecKind {
        CodecKind::Fp16
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn encoded_len(&self) -> usize {
        self.dim * 2
    }
    fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        check_dim(self.dim, v.len())?;
        Ok(v.iter()
            .flat_map(|x| f32_to_f16_bits(*x).to_le_bytes())
            .collect())
    }
    fn decode(&self, blob: &[u8]) -> Result<Vec<f32>> {
        check_blob(self.encoded_len(), blob.len())?;
        Ok(fp16_decode(blob))
    }
    fn distance(&self, a: &[u8], b: &[u8]) -> Result<f32> {
        check_blob(self.encoded_len(), a.len())?;
        check_blob(self.encoded_len(), b.len())?;
        Ok(metric_distance(
            self.metric,
            &fp16_decode(a),
            &fp16_decode(b),
        ))
    }
    fn make_query(&self, q: &[f32]) -> Result<Box<dyn EncodedQuery>> {
        check_dim(self.dim, q.len())?;
        Ok(Box::new(DecodingQuery {
            q: q.to_vec(),
            metric: self.metric,
            decode: fp16_decode,
        }))
    }
}

// ---------------------------------------------------------------------------
// Int8
// ---------------------------------------------------------------------------

/// Per-vector min/scale uniform int8: `[min f32 | scale f32 | D bytes]`.
/// Same scheme as the legacy `ScalarQuantized`, but blob-oriented so the
/// plane can move it around like any other representation.
pub struct Int8Codec {
    dim: usize,
    metric: DistanceMetric,
}

impl Int8Codec {
    pub fn new(dim: usize, metric: DistanceMetric) -> Self {
        Self { dim, metric }
    }
}

fn int8_decode(blob: &[u8]) -> Vec<f32> {
    let min = f32::from_le_bytes(blob[0..4].try_into().unwrap());
    let scale = f32::from_le_bytes(blob[4..8].try_into().unwrap());
    blob[8..].iter().map(|&u| min + u as f32 * scale).collect()
}

impl VectorCodec for Int8Codec {
    fn kind(&self) -> CodecKind {
        CodecKind::Int8
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn encoded_len(&self) -> usize {
        self.dim + 8
    }
    fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        check_dim(self.dim, v.len())?;
        let min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if (max - min).abs() < f32::EPSILON {
            1.0
        } else {
            (max - min) / 255.0
        };
        let mut blob = Vec::with_capacity(self.encoded_len());
        blob.extend_from_slice(&min.to_le_bytes());
        blob.extend_from_slice(&scale.to_le_bytes());
        blob.extend(
            v.iter()
                .map(|&x| ((x - min) / scale).round().clamp(0.0, 255.0) as u8),
        );
        Ok(blob)
    }
    fn decode(&self, blob: &[u8]) -> Result<Vec<f32>> {
        check_blob(self.encoded_len(), blob.len())?;
        Ok(int8_decode(blob))
    }
    fn distance(&self, a: &[u8], b: &[u8]) -> Result<f32> {
        check_blob(self.encoded_len(), a.len())?;
        check_blob(self.encoded_len(), b.len())?;
        Ok(metric_distance(
            self.metric,
            &int8_decode(a),
            &int8_decode(b),
        ))
    }
    fn make_query(&self, q: &[f32]) -> Result<Box<dyn EncodedQuery>> {
        check_dim(self.dim, q.len())?;
        Ok(Box::new(DecodingQuery {
            q: q.to_vec(),
            metric: self.metric,
            decode: int8_decode,
        }))
    }
}

// ---------------------------------------------------------------------------
// Turbo4 adapter
// ---------------------------------------------------------------------------

fn to_turbo_metric(metric: DistanceMetric) -> Result<tq::Metric> {
    match metric {
        DistanceMetric::Euclidean => Ok(tq::Metric::Euclidean),
        DistanceMetric::Cosine => Ok(tq::Metric::Cosine),
        DistanceMetric::DotProduct => Ok(tq::Metric::DotProduct),
        DistanceMetric::Manhattan => Err(RuvectorError::InvalidParameter(
            "Turbo4 does not support the Manhattan metric".into(),
        )),
    }
}

/// [`VectorCodec`] adapter over `ruvector_turboquant::Turbo4Codec` — scores
/// on packed codes, never reconstructs (ADR-296).
pub struct Turbo4PlaneCodec {
    inner: tq::Turbo4Codec,
    metric: tq::Metric,
}

impl Turbo4PlaneCodec {
    pub fn new(dim: usize, metric: DistanceMetric, rotation_seed: u64) -> Result<Self> {
        let metric = to_turbo_metric(metric)?;
        let inner = tq::Turbo4Codec::new(dim, rotation_seed)
            .map_err(|e| RuvectorError::InvalidParameter(e.to_string()))?;
        Ok(Self { inner, metric })
    }
}

struct Turbo4PlaneQuery {
    query: tq::Turbo4Query,
    metric: tq::Metric,
    dim: usize,
}

impl EncodedQuery for Turbo4PlaneQuery {
    fn distance_to(&self, blob: &[u8]) -> f32 {
        tq::rescore(self.metric, &self.query, blob, self.dim)
    }
}

impl VectorCodec for Turbo4PlaneCodec {
    fn kind(&self) -> CodecKind {
        CodecKind::Turbo4
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn encoded_len(&self) -> usize {
        self.inner.code_len()
    }
    fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        self.inner
            .encode(v)
            .map_err(|e| RuvectorError::InvalidInput(e.to_string()))
    }
    fn decode(&self, blob: &[u8]) -> Result<Vec<f32>> {
        check_blob(self.encoded_len(), blob.len())?;
        Ok(self.inner.decode(blob))
    }
    fn distance(&self, a: &[u8], b: &[u8]) -> Result<f32> {
        check_blob(self.encoded_len(), a.len())?;
        check_blob(self.encoded_len(), b.len())?;
        Ok(tq::symmetric_distance(self.metric, a, b, self.inner.dim()))
    }
    fn make_query(&self, q: &[f32]) -> Result<Box<dyn EncodedQuery>> {
        let query = self
            .inner
            .encode_query(q)
            .map_err(|e| RuvectorError::InvalidInput(e.to_string()))?;
        Ok(Box::new(Turbo4PlaneQuery {
            query,
            metric: self.metric,
            dim: self.inner.dim(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{codec_for, CodecKind};
    use super::*;

    /// Decorrelated deterministic vectors (SplitMix64 per (seed, i)).
    fn test_vec(dim: usize, seed: u64) -> Vec<f32> {
        (0..dim as u64)
            .map(|i| {
                let mut z = (seed << 32 | i).wrapping_add(0x9E3779B97F4A7C15);
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn f16_conversion_known_values() {
        for (x, bits) in [
            (0.0f32, 0x0000u16),
            (1.0, 0x3C00),
            (-2.0, 0xC000),
            (65504.0, 0x7BFF), // max finite half
            (0.5, 0x3800),
        ] {
            assert_eq!(f32_to_f16_bits(x), bits, "encode {x}");
            assert_eq!(f16_bits_to_f32(bits), x, "decode {x}");
        }
        // Overflow → inf; NaN stays NaN; subnormals roundtrip.
        assert_eq!(f32_to_f16_bits(1e30), 0x7C00);
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
        let sub = 3.0e-6f32;
        let rt = f16_bits_to_f32(f32_to_f16_bits(sub));
        assert!((rt - sub).abs() / sub < 0.05, "subnormal {sub} -> {rt}");
    }

    #[test]
    fn f16_roundtrip_relative_error() {
        for &x in &[1.0f32, -1.5, 0.123, 999.25, -0.0004883, 3.1415926] {
            let rt = f16_bits_to_f32(f32_to_f16_bits(x));
            assert!(
                (rt - x).abs() / x.abs() < 1e-3,
                "roundtrip {x} -> {rt} exceeds half precision"
            );
        }
    }

    #[test]
    fn all_codecs_roundtrip_and_score() {
        let dim = 64;
        let a = test_vec(dim, 1);
        let b = test_vec(dim, 7);
        let exact = metric_distance(DistanceMetric::Euclidean, &a, &b);

        for kind in [
            CodecKind::Fp32,
            CodecKind::Fp16,
            CodecKind::Int8,
            CodecKind::Turbo4,
        ] {
            let codec = codec_for(kind, dim, DistanceMetric::Euclidean, 42).unwrap();
            let ca = codec.encode(&a).unwrap();
            let cb = codec.encode(&b).unwrap();
            assert_eq!(ca.len(), codec.encoded_len(), "{kind:?} blob length");

            // decode error bounded per codec class
            let da = codec.decode(&ca).unwrap();
            let err: f32 = a.iter().zip(&da).map(|(x, y)| (x - y).abs()).sum::<f32>() / dim as f32;
            let tol = match kind {
                CodecKind::Fp32 => 1e-9,
                CodecKind::Fp16 => 1e-3,
                CodecKind::Int8 => 0.01,
                _ => 0.2,
            };
            assert!(err <= tol, "{kind:?} mean decode error {err} > {tol}");

            // symmetric + asymmetric agree with exact within codec tolerance
            let sym = codec.distance(&ca, &cb).unwrap();
            let asym = codec.make_query(&a).unwrap().distance_to(&cb);
            let dtol = match kind {
                CodecKind::Fp32 => 1e-4,
                CodecKind::Fp16 => 1e-2,
                CodecKind::Int8 => 0.05,
                _ => 0.30,
            };
            assert!(
                (sym - exact).abs() <= dtol * exact.max(1.0),
                "{kind:?} symmetric {sym} vs exact {exact}"
            );
            assert!(
                (asym - exact).abs() <= dtol * exact.max(1.0),
                "{kind:?} asymmetric {asym} vs exact {exact}"
            );
        }
    }

    #[test]
    fn turbo4_plane_matches_direct_codec() {
        let dim = 128;
        let v = test_vec(dim, 3);
        let plane = codec_for(CodecKind::Turbo4, dim, DistanceMetric::Cosine, 42).unwrap();
        let direct = tq::Turbo4Codec::new(dim, 42).unwrap();
        assert_eq!(plane.encode(&v).unwrap(), direct.encode(&v).unwrap());
    }

    #[test]
    fn manhattan_rejected_only_for_turbo4() {
        assert!(codec_for(CodecKind::Turbo4, 64, DistanceMetric::Manhattan, 42).is_err());
        assert!(codec_for(CodecKind::Fp16, 64, DistanceMetric::Manhattan, 42).is_ok());
    }
}
