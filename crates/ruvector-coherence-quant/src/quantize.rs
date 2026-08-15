//! Per-vector min-max scalar quantization with real bit-packed storage.
//!
//! Codes are packed to their actual bit width (4-bit codes share a byte,
//! two codes per byte) so the memory numbers reported by the benchmark are
//! the real serialized size, not an approximation.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bits {
    Four,
    Eight,
}

impl Bits {
    fn levels(self) -> u32 {
        match self {
            Bits::Four => 15,   // 2^4 - 1
            Bits::Eight => 255, // 2^8 - 1
        }
    }

    pub fn bits(self) -> u8 {
        match self {
            Bits::Four => 4,
            Bits::Eight => 8,
        }
    }
}

pub struct QuantizedVector {
    pub bits: Bits,
    pub min: f32,
    pub scale: f32,
    pub dim: usize,
    /// Packed codes. `Eight` -> 1 byte/code. `Four` -> 2 codes/byte (low
    /// nibble = even index, high nibble = odd index).
    pub packed: Vec<u8>,
}

pub fn quantize(v: &[f32], bits: Bits) -> QuantizedVector {
    let dim = v.len();
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-9);
    let levels = bits.levels() as f32;
    let scale = range / levels;

    let codes: Vec<u8> = v
        .iter()
        .map(|&x| (((x - min) / scale).round().clamp(0.0, levels)) as u8)
        .collect();

    let packed = match bits {
        Bits::Eight => codes,
        Bits::Four => codes
            .chunks(2)
            .map(|c| {
                let lo = c[0] & 0x0F;
                let hi = if c.len() > 1 { c[1] & 0x0F } else { 0 };
                lo | (hi << 4)
            })
            .collect(),
    };

    QuantizedVector {
        bits,
        min,
        scale,
        dim,
        packed,
    }
}

pub fn dequantize(q: &QuantizedVector) -> Vec<f32> {
    match q.bits {
        Bits::Eight => q
            .packed
            .iter()
            .map(|&c| q.min + c as f32 * q.scale)
            .collect(),
        Bits::Four => {
            let mut out = Vec::with_capacity(q.dim);
            for &byte in &q.packed {
                let lo = byte & 0x0F;
                let hi = (byte >> 4) & 0x0F;
                out.push(q.min + lo as f32 * q.scale);
                if out.len() < q.dim {
                    out.push(q.min + hi as f32 * q.scale);
                }
            }
            out.truncate(q.dim);
            out
        }
    }
}

/// Real serialized size in bytes: `min` (4B) + `scale` (4B) + packed codes.
pub fn stored_bytes(q: &QuantizedVector) -> usize {
    8 + q.packed.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_eight_bit_is_low_error() {
        let v = vec![0.1_f32, -0.5, 0.9, -0.9, 0.0, 0.33];
        let q = quantize(&v, Bits::Eight);
        let d = dequantize(&q);
        for (a, b) in v.iter().zip(d.iter()) {
            assert!((a - b).abs() < 0.01, "a={a} b={b}");
        }
    }

    #[test]
    fn roundtrip_four_bit_is_bounded_error() {
        let v = vec![0.1_f32, -0.5, 0.9, -0.9, 0.0, 0.33, 0.7, -0.2, 0.15];
        let q = quantize(&v, Bits::Four);
        let d = dequantize(&q);
        assert_eq!(d.len(), v.len());
        // 4-bit range/15 step; error should never exceed one full step.
        let range = 0.9 - (-0.9);
        let step = range / 15.0;
        for (a, b) in v.iter().zip(d.iter()) {
            assert!((a - b).abs() <= step + 1e-4, "a={a} b={b} step={step}");
        }
    }

    #[test]
    fn four_bit_packing_is_half_size_of_eight_bit() {
        let v: Vec<f32> = (0..32).map(|i| i as f32 / 32.0).collect();
        let q8 = quantize(&v, Bits::Eight);
        let q4 = quantize(&v, Bits::Four);
        assert_eq!(stored_bytes(&q8), 8 + 32);
        assert_eq!(stored_bytes(&q4), 8 + 16);
    }

    #[test]
    fn odd_dimension_roundtrips_correct_length() {
        let v = vec![0.2_f32, -0.1, 0.4, 0.05, -0.3];
        let q4 = quantize(&v, Bits::Four);
        let d = dequantize(&q4);
        assert_eq!(d.len(), 5);
    }
}
