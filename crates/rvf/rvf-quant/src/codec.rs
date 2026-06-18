//! QUANT_SEG and SKETCH_SEG wire format codec.
//!
//! Serializes / deserializes quantizer parameters and Count-Min Sketch
//! data to the binary layout defined in the RVF wire spec.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::binary;
use crate::product::ProductQuantizer;
use crate::rabitq::RabitqQuantizer;
use crate::scalar::ScalarQuantizer;
use crate::sketch::CountMinSketch;
use crate::traits::Quantizer;

// ---------------------------------------------------------------------------
// QUANT_SEG codec
// ---------------------------------------------------------------------------

/// Quantization type tags matching the QUANT_SEG wire spec.
/// (Tag 3 is reserved for residual PQ in `rvf_types::QuantType`.)
const QUANT_TYPE_SCALAR: u8 = 0;
const QUANT_TYPE_PRODUCT: u8 = 1;
const QUANT_TYPE_BINARY: u8 = 2;
const QUANT_TYPE_RABITQ: u8 = 4;

/// Current RaBitQ QUANT_SEG layout version. Bump on incompatible changes;
/// decoders reject unknown versions instead of misreading bytes.
const RABITQ_VERSION: u8 = 1;

/// Errors that can occur while decoding QUANT_SEG payloads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// Input data is shorter than expected.
    TooShort,
    /// Unknown quantization type tag.
    UnknownQuantType(u8),
    /// Known quantization type, but an unsupported layout version.
    UnsupportedVersion(u8),
    /// A header field is internally inconsistent (e.g. bad padded_dim).
    InvalidField,
}

impl core::fmt::Display for CodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooShort => write!(f, "input data too short"),
            Self::UnknownQuantType(t) => write!(f, "unknown quant_type: {}", t),
            Self::UnsupportedVersion(v) => write!(f, "unsupported quant_seg version: {}", v),
            Self::InvalidField => write!(f, "invalid quant_seg header field"),
        }
    }
}

/// Encode a quantizer into the QUANT_SEG binary payload.
///
/// Layout:
/// ```text
/// [quant_type: u8] [tier: u8] [dim: u16 LE] [padding: 60 bytes to 64B]
/// [type-specific data ...]
/// ```
pub fn encode_quant_seg(quantizer: &dyn Quantizer) -> Vec<u8> {
    // Downcast (via the `Any` supertrait) to serialize the concrete
    // quantizer's parameters.
    let any: &dyn core::any::Any = quantizer;
    if let Some(sq) = any.downcast_ref::<ScalarQuantizer>() {
        encode_scalar_quantizer(sq)
    } else if let Some(pq) = any.downcast_ref::<ProductQuantizer>() {
        encode_product_quantizer(pq)
    } else if let Some(rq) = any.downcast_ref::<RabitqQuantizer>() {
        encode_rabitq_quantizer(rq)
    } else if quantizer.tier() as u8 == 2 {
        // Binary quantization is parameter-free beyond the dimension.
        encode_binary_quant_seg(quantizer.dim() as u16)
    } else {
        panic!("unknown quantizer type")
    }
}

/// Decode a QUANT_SEG binary payload into a boxed Quantizer.
pub fn decode_quant_seg(data: &[u8]) -> Result<Box<dyn Quantizer>, CodecError> {
    if data.len() < 64 {
        return Err(CodecError::TooShort);
    }

    let quant_type = data[0];
    let _tier = data[1];
    let dim = u16::from_le_bytes([data[2], data[3]]) as usize;
    let body = &data[64..];

    match quant_type {
        QUANT_TYPE_SCALAR => Ok(Box::new(decode_scalar(body, dim)?)),
        QUANT_TYPE_PRODUCT => Ok(Box::new(decode_product(body, dim)?)),
        QUANT_TYPE_BINARY => Ok(Box::new(BinaryQuantizerWrapper { dim })),
        QUANT_TYPE_RABITQ => Ok(Box::new(decode_rabitq(data, body, dim)?)),
        _ => Err(CodecError::UnknownQuantType(quant_type)),
    }
}

// ---------------------------------------------------------------------------
// Scalar
// ---------------------------------------------------------------------------

/// Encode a ScalarQuantizer directly (preferred over trait-based encoding).
pub fn encode_scalar_quantizer(sq: &ScalarQuantizer) -> Vec<u8> {
    let dim = sq.dim as u16;
    let mut buf = vec![0u8; 64];
    buf[0] = QUANT_TYPE_SCALAR;
    buf[1] = 0; // Hot tier
    buf[2..4].copy_from_slice(&dim.to_le_bytes());

    // min[dim], max[dim]
    for &v in &sq.min_vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &sq.max_vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn decode_scalar(body: &[u8], dim: usize) -> Result<ScalarQuantizer, CodecError> {
    let float_bytes = dim * 4;
    if body.len() < float_bytes * 2 {
        return Err(CodecError::TooShort);
    }

    let mut min_vals = Vec::with_capacity(dim);
    let mut max_vals = Vec::with_capacity(dim);

    for d in 0..dim {
        let offset = d * 4;
        let v = f32::from_le_bytes([
            body[offset],
            body[offset + 1],
            body[offset + 2],
            body[offset + 3],
        ]);
        min_vals.push(v);
    }
    for d in 0..dim {
        let offset = (dim + d) * 4;
        let v = f32::from_le_bytes([
            body[offset],
            body[offset + 1],
            body[offset + 2],
            body[offset + 3],
        ]);
        max_vals.push(v);
    }

    Ok(ScalarQuantizer {
        min_vals,
        max_vals,
        dim,
    })
}

// ---------------------------------------------------------------------------
// Product
// ---------------------------------------------------------------------------

/// Encode a ProductQuantizer directly.
pub fn encode_product_quantizer(pq: &ProductQuantizer) -> Vec<u8> {
    let dim = (pq.m * pq.sub_dim) as u16;
    let mut buf = vec![0u8; 64];
    buf[0] = QUANT_TYPE_PRODUCT;
    buf[1] = 1; // Warm tier
    buf[2..4].copy_from_slice(&dim.to_le_bytes());

    // PQ header: M, K, sub_dim (each as u16 LE)
    // Written after the 64-byte aligned header.
    buf.extend_from_slice(&(pq.m as u16).to_le_bytes());
    buf.extend_from_slice(&(pq.k as u16).to_le_bytes());
    buf.extend_from_slice(&(pq.sub_dim as u16).to_le_bytes());

    // Codebook: M * K * sub_dim floats
    for sub_book in &pq.codebooks {
        for centroid in sub_book {
            for &val in centroid {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }
    }

    buf
}

fn decode_product(body: &[u8], _dim: usize) -> Result<ProductQuantizer, CodecError> {
    if body.len() < 6 {
        return Err(CodecError::TooShort);
    }

    let m = u16::from_le_bytes([body[0], body[1]]) as usize;
    let k = u16::from_le_bytes([body[2], body[3]]) as usize;
    let sub_dim = u16::from_le_bytes([body[4], body[5]]) as usize;

    // Compute the codebook size in u64 with checked arithmetic: on 32-bit
    // targets (wasm32) `m * k * sub_dim * 4` can wrap usize, slip past the
    // length check below, and then index out of bounds in the decode loop.
    let codebook_bytes = (m as u64)
        .checked_mul(k as u64)
        .and_then(|v| v.checked_mul(sub_dim as u64))
        .and_then(|v| v.checked_mul(4))
        .ok_or(CodecError::InvalidField)?;
    let expected = codebook_bytes
        .checked_add(6)
        .ok_or(CodecError::InvalidField)?;
    if (body.len() as u64) < expected {
        return Err(CodecError::TooShort);
    }

    let mut codebooks = Vec::with_capacity(m);
    let mut offset = 6;
    for _ in 0..m {
        let mut sub_book = Vec::with_capacity(k);
        for _ in 0..k {
            let mut centroid = Vec::with_capacity(sub_dim);
            for _ in 0..sub_dim {
                let v = f32::from_le_bytes([
                    body[offset],
                    body[offset + 1],
                    body[offset + 2],
                    body[offset + 3],
                ]);
                centroid.push(v);
                offset += 4;
            }
            sub_book.push(centroid);
        }
        codebooks.push(sub_book);
    }

    Ok(ProductQuantizer {
        m,
        k,
        sub_dim,
        codebooks,
    })
}

// ---------------------------------------------------------------------------
// Binary
// ---------------------------------------------------------------------------

fn encode_binary_quant_seg(dim: u16) -> Vec<u8> {
    let mut buf = vec![0u8; 64];
    buf[0] = QUANT_TYPE_BINARY;
    buf[1] = 2; // Cold tier
    buf[2..4].copy_from_slice(&dim.to_le_bytes());
    // Binary quantization has no additional parameters (sign-based).
    buf
}

/// Wrapper to implement `Quantizer` for binary quantization.
struct BinaryQuantizerWrapper {
    dim: usize,
}

impl Quantizer for BinaryQuantizerWrapper {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        binary::encode_binary(vector)
    }

    fn decode(&self, codes: &[u8]) -> Vec<f32> {
        binary::decode_binary(codes, self.dim)
    }

    fn tier(&self) -> crate::tier::TemperatureTier {
        crate::tier::TemperatureTier::Cold
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// RaBitQ
// ---------------------------------------------------------------------------

/// Encode a RaBitQ quantizer into a QUANT_SEG payload.
///
/// Header layout (within the shared 64-byte aligned header; bytes 4..20
/// were zero padding in pre-RaBitQ payloads, so old types are unaffected):
/// ```text
/// [quant_type=4: u8] [tier: u8] [dim: u16 LE]
/// [version: u8] [rounds: u8] [reserved: u16]
/// [seed: u64 LE] [padded_dim: u32 LE] [padding to 64B]
/// [centroid: dim * f32 LE]
/// ```
pub fn encode_rabitq_quantizer(rq: &RabitqQuantizer) -> Vec<u8> {
    let mut buf = vec![0u8; 64];
    buf[0] = QUANT_TYPE_RABITQ;
    buf[1] = 2; // Cold tier
    buf[2..4].copy_from_slice(&(rq.dim as u16).to_le_bytes());
    buf[4] = RABITQ_VERSION;
    buf[5] = rq.rounds;
    // buf[6..8] reserved (zero)
    buf[8..16].copy_from_slice(&rq.seed.to_le_bytes());
    buf[16..20].copy_from_slice(&(rq.padded_dim as u32).to_le_bytes());

    for &v in &rq.centroid {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Decode a RaBitQ QUANT_SEG payload (versioned; bounds-checked).
///
/// `data` is the full payload (for header fields beyond the shared
/// prefix), `body` is the slice after the 64-byte header.
fn decode_rabitq(data: &[u8], body: &[u8], dim: usize) -> Result<RabitqQuantizer, CodecError> {
    // Caller guarantees data.len() >= 64.
    let version = data[4];
    if version != RABITQ_VERSION {
        return Err(CodecError::UnsupportedVersion(version));
    }
    let rounds = data[5];
    let seed = u64::from_le_bytes(data[8..16].try_into().expect("len checked"));
    let padded_dim = u32::from_le_bytes(data[16..20].try_into().expect("len checked")) as usize;

    if dim == 0 || rounds == 0 {
        return Err(CodecError::InvalidField);
    }
    // padded_dim must be the canonical power-of-two padding of dim; this
    // also bounds it (dim is u16, so padded_dim <= 65536).
    if padded_dim != dim.max(1).next_power_of_two() {
        return Err(CodecError::InvalidField);
    }

    let centroid_bytes = dim.checked_mul(4).ok_or(CodecError::InvalidField)?;
    if body.len() < centroid_bytes {
        return Err(CodecError::TooShort);
    }
    let mut centroid = Vec::with_capacity(dim);
    for d in 0..dim {
        let offset = d * 4;
        centroid.push(f32::from_le_bytes(
            body[offset..offset + 4].try_into().expect("len checked"),
        ));
    }

    Ok(RabitqQuantizer::with_centroid(dim, centroid, seed, rounds))
}

// ---------------------------------------------------------------------------
// SKETCH_SEG codec
// ---------------------------------------------------------------------------

/// Encode a CountMinSketch into the SKETCH_SEG binary payload.
///
/// Layout:
/// ```text
/// [width: u32 LE] [depth: u32 LE] [total_accesses: u64 LE] [padding: 48 bytes to 64B]
/// [counters: depth * width bytes]
/// ```
pub fn encode_sketch_seg(sketch: &CountMinSketch) -> Vec<u8> {
    let mut buf = vec![0u8; 64]; // 64-byte aligned header

    buf[0..4].copy_from_slice(&(sketch.width as u32).to_le_bytes());
    buf[4..8].copy_from_slice(&(sketch.depth as u32).to_le_bytes());
    buf[8..16].copy_from_slice(&sketch.total_accesses.to_le_bytes());

    // Counter data: row-major
    for row in &sketch.counters {
        buf.extend_from_slice(row);
    }

    buf
}

/// Decode a SKETCH_SEG binary payload into a CountMinSketch.
///
/// Returns an error (never panics) on malformed input: short headers,
/// counter data shorter than `width * depth`, a zero `width` paired with a
/// non-zero `depth` (which would bypass the length check while driving an
/// unbounded row allocation), or `width * depth` overflow.
pub fn decode_sketch_seg(data: &[u8]) -> Result<CountMinSketch, CodecError> {
    if data.len() < 64 {
        return Err(CodecError::TooShort);
    }

    let width = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let depth = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let total_accesses = u64::from_le_bytes([
        data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
    ]);

    let body = &data[64..];

    // Every row must consume at least one byte; otherwise a crafted
    // depth (up to u32::MAX) passes the `expected == 0` length check and
    // OOMs in `Vec::with_capacity` below.
    if width == 0 && depth != 0 {
        return Err(CodecError::InvalidField);
    }
    // Checked u64 arithmetic: `width * depth` can wrap usize on 32-bit
    // targets (wasm32) and slip past the length check.
    let expected = (width as u64)
        .checked_mul(depth as u64)
        .ok_or(CodecError::InvalidField)?;
    if (body.len() as u64) < expected {
        return Err(CodecError::TooShort);
    }

    // Safe: width >= 1 here, so depth <= expected <= body.len().
    let mut counters = Vec::with_capacity(depth);
    for row in 0..depth {
        let start = row * width;
        counters.push(body[start..start + width].to_vec());
    }

    Ok(CountMinSketch {
        counters,
        width,
        depth,
        total_accesses,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_quant_seg_round_trip() {
        let sq = ScalarQuantizer {
            min_vals: vec![-1.0, -2.0, -0.5, 0.0],
            max_vals: vec![1.0, 2.0, 0.5, 1.0],
            dim: 4,
        };

        let encoded = encode_scalar_quantizer(&sq);
        let decoded = decode_quant_seg(&encoded).unwrap();

        assert_eq!(decoded.dim(), 4);
        assert_eq!(decoded.tier(), crate::tier::TemperatureTier::Hot);

        // Verify round-trip: encode a test vector, check similar output
        let test_vec = vec![0.5, 1.0, 0.0, 0.5];
        let codes_orig = sq.encode_vec(&test_vec);
        let codes_decoded = decoded.encode(&test_vec);
        assert_eq!(codes_orig, codes_decoded);
    }

    #[test]
    fn product_quant_seg_round_trip() {
        // Build a small PQ manually
        let pq = ProductQuantizer {
            m: 2,
            k: 4,
            sub_dim: 2,
            codebooks: vec![
                vec![
                    vec![0.0, 0.1],
                    vec![0.2, 0.3],
                    vec![0.4, 0.5],
                    vec![0.6, 0.7],
                ],
                vec![
                    vec![0.8, 0.9],
                    vec![1.0, 1.1],
                    vec![1.2, 1.3],
                    vec![1.4, 1.5],
                ],
            ],
        };

        let encoded = encode_product_quantizer(&pq);
        let decoded = decode_quant_seg(&encoded).unwrap();

        assert_eq!(decoded.dim(), 4);
        assert_eq!(decoded.tier(), crate::tier::TemperatureTier::Warm);

        let test_vec = vec![0.1, 0.2, 0.9, 1.0];
        let codes_orig = pq.encode_vec(&test_vec);
        let codes_decoded = decoded.encode(&test_vec);
        assert_eq!(codes_orig, codes_decoded);
    }

    #[test]
    fn binary_quant_seg_round_trip() {
        let dim: u16 = 16;
        let encoded = encode_binary_quant_seg(dim);
        let decoded = decode_quant_seg(&encoded).unwrap();

        assert_eq!(decoded.dim(), 16);
        assert_eq!(decoded.tier(), crate::tier::TemperatureTier::Cold);

        let test_vec: Vec<f32> = (0..16)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let codes = decoded.encode(&test_vec);
        let recon = decoded.decode(&codes);
        assert_eq!(recon.len(), 16);
    }

    #[test]
    fn encode_quant_seg_scalar_round_trip() {
        let sq = ScalarQuantizer {
            min_vals: vec![-1.0, -2.0, -0.5, 0.0],
            max_vals: vec![1.0, 2.0, 0.5, 1.0],
            dim: 4,
        };

        let encoded = encode_quant_seg(&sq);
        let decoded = decode_quant_seg(&encoded).unwrap();

        let any: &dyn core::any::Any = decoded.as_ref();
        let dec_sq = any
            .downcast_ref::<ScalarQuantizer>()
            .expect("expected ScalarQuantizer");
        assert_eq!(dec_sq.min_vals, sq.min_vals);
        assert_eq!(dec_sq.max_vals, sq.max_vals);
        assert_eq!(dec_sq.dim, sq.dim);
    }

    #[test]
    fn encode_quant_seg_product_round_trip() {
        let pq = ProductQuantizer {
            m: 2,
            k: 2,
            sub_dim: 2,
            codebooks: vec![
                vec![vec![0.0, 0.1], vec![0.2, 0.3]],
                vec![vec![0.8, 0.9], vec![1.0, 1.1]],
            ],
        };

        let encoded = encode_quant_seg(&pq);
        let decoded = decode_quant_seg(&encoded).unwrap();

        let any: &dyn core::any::Any = decoded.as_ref();
        let dec_pq = any
            .downcast_ref::<ProductQuantizer>()
            .expect("expected ProductQuantizer");
        assert_eq!(dec_pq.m, pq.m);
        assert_eq!(dec_pq.k, pq.k);
        assert_eq!(dec_pq.sub_dim, pq.sub_dim);
        assert_eq!(dec_pq.codebooks, pq.codebooks);
    }

    #[test]
    fn encode_quant_seg_binary_round_trip() {
        let bq = BinaryQuantizerWrapper { dim: 16 };
        let encoded = encode_quant_seg(&bq);
        let decoded = decode_quant_seg(&encoded).unwrap();

        assert_eq!(decoded.dim(), 16);
        assert_eq!(decoded.tier(), crate::tier::TemperatureTier::Cold);
    }

    #[test]
    fn decode_quant_seg_malformed_inputs() {
        // Header too short.
        assert!(matches!(
            decode_quant_seg(&[0u8; 8]),
            Err(CodecError::TooShort)
        ));

        // Unknown quant_type tag.
        let mut bad_type = vec![0u8; 64];
        bad_type[0] = 9;
        assert!(matches!(
            decode_quant_seg(&bad_type),
            Err(CodecError::UnknownQuantType(9))
        ));

        // Scalar header claims dim 4 but carries no min/max body.
        let mut truncated = vec![0u8; 64];
        truncated[0] = 0; // scalar
        truncated[2..4].copy_from_slice(&4u16.to_le_bytes());
        assert!(matches!(
            decode_quant_seg(&truncated),
            Err(CodecError::TooShort)
        ));

        // Product header present but codebook data missing.
        let mut pq_truncated = vec![0u8; 64];
        pq_truncated[0] = 1; // product
        pq_truncated[2..4].copy_from_slice(&4u16.to_le_bytes());
        pq_truncated.extend_from_slice(&2u16.to_le_bytes()); // m
        pq_truncated.extend_from_slice(&4u16.to_le_bytes()); // k
        pq_truncated.extend_from_slice(&2u16.to_le_bytes()); // sub_dim
        assert!(matches!(
            decode_quant_seg(&pq_truncated),
            Err(CodecError::TooShort)
        ));
    }

    #[test]
    fn rabitq_quant_seg_round_trip() {
        let centroid: Vec<f32> = (0..20).map(|i| i as f32 * 0.1 - 1.0).collect();
        let rq = RabitqQuantizer::with_centroid(20, centroid.clone(), 0x1234_5678_9ABC_DEF0, 3);

        let encoded = encode_rabitq_quantizer(&rq);
        let decoded = decode_quant_seg(&encoded).unwrap();
        assert_eq!(decoded.dim(), 20);
        assert_eq!(decoded.tier(), crate::tier::TemperatureTier::Cold);

        let any: &dyn core::any::Any = decoded.as_ref();
        let dec = any
            .downcast_ref::<RabitqQuantizer>()
            .expect("expected RabitqQuantizer");
        assert_eq!(dec.dim, rq.dim);
        assert_eq!(dec.padded_dim, 32);
        assert_eq!(dec.seed, rq.seed);
        assert_eq!(dec.rounds, rq.rounds);
        assert_eq!(dec.centroid, centroid);

        // The decoded quantizer must produce byte-identical codes.
        let v: Vec<f32> = (0..20).map(|i| (i as f32 * 0.7).sin()).collect();
        assert_eq!(dec.encode(&v), rq.encode(&v));

        // Trait-based encode dispatches to the RaBitQ layout too.
        assert_eq!(encode_quant_seg(&rq), encoded);
    }

    #[test]
    fn rabitq_quant_seg_rejects_bad_versions_and_fields() {
        let rq = RabitqQuantizer::with_centroid(8, vec![0.0; 8], 7, 3);
        let good = encode_rabitq_quantizer(&rq);

        // Future layout version: reject instead of misreading.
        let mut future = good.clone();
        future[4] = RABITQ_VERSION + 1;
        assert!(matches!(
            decode_quant_seg(&future),
            Err(CodecError::UnsupportedVersion(v)) if v == RABITQ_VERSION + 1
        ));

        // Inconsistent padded_dim.
        let mut bad_pad = good.clone();
        bad_pad[16..20].copy_from_slice(&7u32.to_le_bytes());
        assert!(matches!(
            decode_quant_seg(&bad_pad),
            Err(CodecError::InvalidField)
        ));

        // Truncated centroid body.
        assert!(matches!(
            decode_quant_seg(&good[..good.len() - 4]),
            Err(CodecError::TooShort)
        ));

        // Zero rounds.
        let mut zero_rounds = good.clone();
        zero_rounds[5] = 0;
        assert!(matches!(
            decode_quant_seg(&zero_rounds),
            Err(CodecError::InvalidField)
        ));
    }

    #[test]
    fn pre_rabitq_payloads_still_decode() {
        // A byte-frozen legacy binary-quantizer payload (type 2, header
        // bytes 4..64 all zero, no body) must keep decoding after the
        // RaBitQ extension claimed header bytes 4..20 for type 4.
        let mut legacy = vec![0u8; 64];
        legacy[0] = 2; // QUANT_TYPE_BINARY
        legacy[1] = 2; // Cold tier
        legacy[2..4].copy_from_slice(&24u16.to_le_bytes());
        let decoded = decode_quant_seg(&legacy).unwrap();
        assert_eq!(decoded.dim(), 24);
        assert_eq!(decoded.tier(), crate::tier::TemperatureTier::Cold);

        // Same for a legacy scalar payload.
        let sq = ScalarQuantizer {
            min_vals: vec![-1.0, 0.0],
            max_vals: vec![1.0, 2.0],
            dim: 2,
        };
        let legacy_scalar = encode_scalar_quantizer(&sq);
        assert!(decode_quant_seg(&legacy_scalar).is_ok());
    }

    #[test]
    fn decode_product_rejects_huge_codebook_dimensions() {
        // m = k = sub_dim = u16::MAX -> codebook of ~1.1e15 bytes. The
        // u64 checked size computation must reject this against the
        // actual body length instead of wrapping usize on 32-bit targets
        // (wasm32) and reading out of bounds.
        let mut pq = vec![0u8; 64];
        pq[0] = QUANT_TYPE_PRODUCT;
        pq[2..4].copy_from_slice(&4u16.to_le_bytes());
        pq.extend_from_slice(&u16::MAX.to_le_bytes()); // m
        pq.extend_from_slice(&u16::MAX.to_le_bytes()); // k
        pq.extend_from_slice(&u16::MAX.to_le_bytes()); // sub_dim
        assert!(matches!(decode_quant_seg(&pq), Err(CodecError::TooShort)));
    }

    #[test]
    fn decode_sketch_seg_rejects_malformed_inputs() {
        // Header too short: error, not panic.
        assert!(matches!(decode_sketch_seg(&[]), Err(CodecError::TooShort)));
        assert!(matches!(
            decode_sketch_seg(&[0u8; 16]),
            Err(CodecError::TooShort)
        ));

        // width = 0 + depth = u32::MAX: expected counter bytes are 0, so
        // the length check alone passes; the zero-width guard must reject
        // it before the depth-sized allocation OOMs.
        let mut zero_width = vec![0u8; 64];
        zero_width[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode_sketch_seg(&zero_width),
            Err(CodecError::InvalidField)
        ));

        // width = depth = u32::MAX: product (~1.8e19) wraps a 32-bit
        // usize; the checked u64 arithmetic must reject it against the
        // body length.
        let mut huge = vec![0u8; 64];
        huge[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
        huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            decode_sketch_seg(&huge),
            Err(CodecError::TooShort)
        ));

        // Counter data shorter than width * depth.
        let mut truncated = vec![0u8; 64 + 10];
        truncated[0..4].copy_from_slice(&8u32.to_le_bytes()); // width
        truncated[4..8].copy_from_slice(&4u32.to_le_bytes()); // depth -> needs 32
        assert!(matches!(
            decode_sketch_seg(&truncated),
            Err(CodecError::TooShort)
        ));

        // Degenerate-but-consistent empty sketch (width = depth = 0)
        // still decodes.
        let empty = decode_sketch_seg(&[0u8; 64]).expect("empty sketch decodes");
        assert_eq!(empty.width, 0);
        assert_eq!(empty.depth, 0);
        assert!(empty.counters.is_empty());
    }

    #[test]
    fn sketch_seg_round_trip() {
        let mut sketch = CountMinSketch::new(64, 4);
        for block_id in 0..20u64 {
            for _ in 0..(block_id + 1) {
                sketch.increment(block_id);
            }
        }

        let encoded = encode_sketch_seg(&sketch);
        let decoded = decode_sketch_seg(&encoded).expect("well-formed sketch should decode");

        assert_eq!(decoded.width, sketch.width);
        assert_eq!(decoded.depth, sketch.depth);
        assert_eq!(decoded.total_accesses, sketch.total_accesses);

        // Verify estimates match
        for block_id in 0..20u64 {
            assert_eq!(decoded.estimate(block_id), sketch.estimate(block_id));
        }
    }
}
