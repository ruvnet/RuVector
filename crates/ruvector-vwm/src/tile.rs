//! Spacetime tile system for organizing Gaussians into quantized blocks.
//!
//! The world is partitioned into a regular 3D spatial grid with temporal bucketing.
//! Each [`Tile`] holds a [`PrimitiveBlock`] containing encoded Gaussian data at a
//! particular [`QuantTier`]. Tiles are addressed by [`TileCoord`] which includes
//! spatial coordinates, a time bucket, and a level-of-detail index.

use crate::gaussian::Gaussian4D;

/// Tile coordinate in spacetime grid.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct TileCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    pub time_bucket: u32,
    pub lod: u8,
}

/// A spacetime tile containing a block of Gaussians.
#[derive(Clone, Debug)]
pub struct Tile {
    pub coord: TileCoord,
    pub primitive_block: PrimitiveBlock,
    /// Entity IDs contained in this tile.
    pub entity_refs: Vec<u64>,
    /// Coherence score from the last evaluation (0.0 = incoherent, 1.0 = fully coherent).
    pub coherence_score: f32,
    /// Epoch of the last update.
    pub last_update_epoch: u64,
}

/// Packed, quantized block of Gaussian primitives.
///
/// The `encode()` method applies Hot8 (8-bit uniform) quantization, storing each
/// float field as a `u8` with per-field min/max headers. The `encode_raw()` method
/// stores raw `f32` bytes for lossless roundtrip. The [`QuantTier`] enum tags the
/// intended compression level; tiers other than Hot8 currently fall back to Hot8.
#[derive(Clone, Debug)]
pub struct PrimitiveBlock {
    /// Encoded data (quantized or raw depending on `decode_descriptor.quantized`).
    pub data: Vec<u8>,
    /// Number of Gaussians in this block.
    pub count: u32,
    /// Quantization tier tag.
    pub quant_tier: QuantTier,
    /// Checksum over `data`.
    pub checksum: u32,
    /// Descriptor for decoding fields from the packed data.
    pub decode_descriptor: DecodeDescriptor,
}

/// Quantization tier controlling compression ratio.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantTier {
    /// 8-bit quantization, ~4x compression.
    Hot8,
    /// 7-bit quantization, ~4.57x compression.
    Warm7,
    /// 5-bit quantization, ~6.4x compression.
    Warm5,
    /// 3-bit quantization, ~10.67x compression.
    Cold3,
}

/// Descriptor that tells the decoder how to interpret a [`PrimitiveBlock`].
#[derive(Clone, Debug)]
pub struct DecodeDescriptor {
    /// Total bytes per Gaussian in the packed format (only meaningful when `quantized` is false).
    pub bytes_per_gaussian: u16,
    /// Byte offsets of each field within a Gaussian record (only meaningful when `quantized` is false).
    pub field_offsets: FieldOffsets,
    /// Per-field rescaling factors applied after dequantization.
    pub scale_factors: [f32; 4],
    /// Whether the block uses quantized encoding.
    pub quantized: bool,
}

/// Byte offsets of each field within a packed Gaussian record.
#[derive(Clone, Debug)]
pub struct FieldOffsets {
    pub center: u16,
    pub covariance: u16,
    pub color: u16,
    pub opacity: u16,
    pub scale: u16,
    pub rotation: u16,
    pub temporal: u16,
}

// Size of a single Gaussian when stored as raw f32 bytes:
//   center(3) + covariance(6) + sh_coeffs(3) + opacity(1) + scale(3) + rotation(4)
//   + time_range(2) + velocity(3) + id(1 as u32→f32 reinterpret, stored separately)
// We store the id as 4 bytes (u32) at the end.
// Total floats: 3+6+3+1+3+4+2+3 = 25 floats = 100 bytes + 4 bytes id = 104 bytes
const RAW_GAUSSIAN_BYTES: u16 = 104;

/// Number of float fields per Gaussian (excluding the u32 id).
const FLOAT_FIELD_COUNT: usize = 25;

/// Extract the 25 float fields from a Gaussian4D in canonical order.
///
/// Field layout: center(3), covariance(6), sh_coeffs(3), opacity(1),
/// scale(3), rotation(4), time_range(2), velocity(3).
fn gaussian_to_floats(g: &Gaussian4D) -> [f32; FLOAT_FIELD_COUNT] {
    let mut floats = [0.0f32; FLOAT_FIELD_COUNT];
    floats[0..3].copy_from_slice(&g.center);
    floats[3..9].copy_from_slice(&g.covariance);
    floats[9..12].copy_from_slice(&g.sh_coeffs);
    floats[12] = g.opacity;
    floats[13..16].copy_from_slice(&g.scale);
    floats[16..20].copy_from_slice(&g.rotation);
    floats[20..22].copy_from_slice(&g.time_range);
    floats[22..25].copy_from_slice(&g.velocity);
    floats
}

/// Reconstruct a Gaussian4D from 25 float fields and a u32 id.
fn floats_to_gaussian(floats: &[f32; FLOAT_FIELD_COUNT], id: u32) -> Gaussian4D {
    Gaussian4D {
        center: [floats[0], floats[1], floats[2]],
        covariance: [floats[3], floats[4], floats[5], floats[6], floats[7], floats[8]],
        sh_coeffs: [floats[9], floats[10], floats[11]],
        opacity: floats[12],
        scale: [floats[13], floats[14], floats[15]],
        rotation: [floats[16], floats[17], floats[18], floats[19]],
        time_range: [floats[20], floats[21]],
        velocity: [floats[22], floats[23], floats[24]],
        id,
    }
}

/// Quantize a slice of f32 values to u8 using min/max uniform quantization.
///
/// Returns `(min, max, quantized_bytes)`. When the range is zero or non-finite,
/// all values quantize to 0 and dequantize back to `min`.
fn quantize_field(values: &[f32]) -> (f32, f32, Vec<u8>) {
    if values.is_empty() {
        return (0.0, 0.0, vec![]);
    }

    // Find min/max among finite values only.
    let finite_min = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let finite_max = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    // If at least one finite value exists, use those bounds.
    // Otherwise all values are non-finite; use the first value for both bounds.
    let (min, max) = if finite_min <= finite_max {
        (finite_min, finite_max)
    } else {
        (values[0], values[0])
    };

    let range = max - min;

    let quantized = if range == 0.0 || !range.is_finite() {
        // All values are effectively the same (or range is degenerate).
        vec![0u8; values.len()]
    } else {
        values
            .iter()
            .map(|&v| {
                // Map non-finite values to the nearest bound.
                let v = if v.is_nan() {
                    min
                } else if !v.is_finite() {
                    if v > 0.0 { max } else { min }
                } else {
                    v
                };
                let q = ((v - min) / range * 255.0).round();
                q.clamp(0.0, 255.0) as u8
            })
            .collect()
    };

    (min, max, quantized)
}

/// Dequantize u8 values back to f32 using stored min/max.
fn dequantize_field(min: f32, max: f32, quantized: &[u8]) -> Vec<f32> {
    let range = max - min;
    if range == 0.0 || !range.is_finite() {
        return vec![min; quantized.len()];
    }
    quantized
        .iter()
        .map(|&q| min + (q as f32 / 255.0) * range)
        .collect()
}

impl PrimitiveBlock {
    /// Encode a slice of Gaussians into a quantized primitive block.
    ///
    /// Applies Hot8 (8-bit uniform) quantization. All tiers currently fall back
    /// to Hot8; the tier tag is stored for future sub-byte implementations.
    ///
    /// **Quantized data layout:**
    /// ```text
    /// [field_count: u16]
    /// for each field:
    ///   [min: f32] [max: f32] [quantized_values: u8 * gaussian_count]
    /// for each gaussian:
    ///   [id: u32]
    /// ```
    pub fn encode(gaussians: &[Gaussian4D], tier: QuantTier) -> Self {
        let count = gaussians.len() as u32;

        if count == 0 {
            return Self {
                data: Vec::new(),
                count: 0,
                quant_tier: tier,
                checksum: compute_checksum(&[]),
                decode_descriptor: DecodeDescriptor {
                    bytes_per_gaussian: 0,
                    field_offsets: FieldOffsets {
                        center: 0,
                        covariance: 0,
                        color: 0,
                        opacity: 0,
                        scale: 0,
                        rotation: 0,
                        temporal: 0,
                    },
                    scale_factors: [1.0; 4],
                    quantized: true,
                },
            };
        }

        let n = gaussians.len();

        // Collect float fields in column-major order (fields[field_idx][gaussian_idx]).
        let mut fields: Vec<Vec<f32>> = vec![Vec::with_capacity(n); FLOAT_FIELD_COUNT];
        let mut ids: Vec<u32> = Vec::with_capacity(n);

        for g in gaussians {
            let floats = gaussian_to_floats(g);
            for (i, &v) in floats.iter().enumerate() {
                fields[i].push(v);
            }
            ids.push(g.id);
        }

        // Pre-allocate: header(2) + per-field(8 + n) * 25 + ids(n * 4)
        let data_size = 2 + FLOAT_FIELD_COUNT * (8 + n) + n * 4;
        let mut data = Vec::with_capacity(data_size);

        // Header: field count
        data.extend_from_slice(&(FLOAT_FIELD_COUNT as u16).to_le_bytes());

        // Per-field: min, max, quantized values
        for field_values in &fields {
            let (min, max, quantized) = quantize_field(field_values);
            data.extend_from_slice(&min.to_le_bytes());
            data.extend_from_slice(&max.to_le_bytes());
            data.extend_from_slice(&quantized);
        }

        // IDs (unquantized u32)
        for &id in &ids {
            data.extend_from_slice(&id.to_le_bytes());
        }

        let checksum = compute_checksum(&data);

        Self {
            data,
            count,
            quant_tier: tier,
            checksum,
            decode_descriptor: DecodeDescriptor {
                bytes_per_gaussian: 0,
                field_offsets: FieldOffsets {
                    center: 0,
                    covariance: 0,
                    color: 0,
                    opacity: 0,
                    scale: 0,
                    rotation: 0,
                    temporal: 0,
                },
                scale_factors: [1.0; 4],
                quantized: true,
            },
        }
    }

    /// Encode a slice of Gaussians as raw f32 bytes (no quantization).
    ///
    /// This preserves exact values and is suitable when lossless roundtrip is required.
    pub fn encode_raw(gaussians: &[Gaussian4D], tier: QuantTier) -> Self {
        let count = gaussians.len() as u32;
        let mut data = Vec::with_capacity(gaussians.len() * RAW_GAUSSIAN_BYTES as usize);

        for g in gaussians {
            // center: 3 floats (offset 0)
            for &v in &g.center {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // covariance: 6 floats (offset 12)
            for &v in &g.covariance {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // sh_coeffs: 3 floats (offset 36)
            for &v in &g.sh_coeffs {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // opacity: 1 float (offset 48)
            data.extend_from_slice(&g.opacity.to_le_bytes());
            // scale: 3 floats (offset 52)
            for &v in &g.scale {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // rotation: 4 floats (offset 64)
            for &v in &g.rotation {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // time_range: 2 floats (offset 80)
            for &v in &g.time_range {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // velocity: 3 floats (offset 88)
            for &v in &g.velocity {
                data.extend_from_slice(&v.to_le_bytes());
            }
            // id: u32 (offset 100)
            data.extend_from_slice(&g.id.to_le_bytes());
        }

        let checksum = compute_checksum(&data);

        let decode_descriptor = DecodeDescriptor {
            bytes_per_gaussian: RAW_GAUSSIAN_BYTES,
            field_offsets: FieldOffsets {
                center: 0,
                covariance: 12,
                color: 36,
                opacity: 48,
                scale: 52,
                rotation: 64,
                temporal: 80,
            },
            scale_factors: [1.0, 1.0, 1.0, 1.0],
            quantized: false,
        };

        Self {
            data,
            count,
            quant_tier: tier,
            checksum,
            decode_descriptor,
        }
    }

    /// Decode the primitive block back into Gaussians.
    ///
    /// Dispatches to quantized or raw decoding based on `decode_descriptor.quantized`.
    pub fn decode(&self) -> Vec<Gaussian4D> {
        if self.count == 0 {
            return Vec::new();
        }

        if self.decode_descriptor.quantized {
            self.decode_quantized()
        } else {
            self.decode_raw()
        }
    }

    /// Decode Hot8-quantized data back into Gaussians.
    fn decode_quantized(&self) -> Vec<Gaussian4D> {
        let n = self.count as usize;
        let mut offset = 0;

        // Read field count from header.
        let field_count =
            u16::from_le_bytes([self.data[offset], self.data[offset + 1]]) as usize;
        offset += 2;

        // Read and dequantize each field (column-major).
        let mut fields: Vec<Vec<f32>> = Vec::with_capacity(field_count);
        for _ in 0..field_count {
            let min = f32::from_le_bytes([
                self.data[offset],
                self.data[offset + 1],
                self.data[offset + 2],
                self.data[offset + 3],
            ]);
            offset += 4;
            let max = f32::from_le_bytes([
                self.data[offset],
                self.data[offset + 1],
                self.data[offset + 2],
                self.data[offset + 3],
            ]);
            offset += 4;

            let quantized = &self.data[offset..offset + n];
            offset += n;

            fields.push(dequantize_field(min, max, quantized));
        }

        // Read IDs.
        let mut ids: Vec<u32> = Vec::with_capacity(n);
        for _ in 0..n {
            let id = u32::from_le_bytes([
                self.data[offset],
                self.data[offset + 1],
                self.data[offset + 2],
                self.data[offset + 3],
            ]);
            offset += 4;
            ids.push(id);
        }

        // Reconstruct Gaussians from column-major fields.
        let mut gaussians = Vec::with_capacity(n);
        for i in 0..n {
            let mut floats = [0.0f32; FLOAT_FIELD_COUNT];
            for (f, field) in fields.iter().enumerate() {
                floats[f] = field[i];
            }
            gaussians.push(floats_to_gaussian(&floats, ids[i]));
        }

        gaussians
    }

    /// Decode raw f32 data back into Gaussians (original lossless path).
    fn decode_raw(&self) -> Vec<Gaussian4D> {
        let stride = self.decode_descriptor.bytes_per_gaussian as usize;
        let mut gaussians = Vec::with_capacity(self.count as usize);

        for i in 0..self.count as usize {
            let base = i * stride;
            if base + stride > self.data.len() {
                break;
            }

            let read_f32 = |offset: usize| -> f32 {
                let o = base + offset;
                f32::from_le_bytes([
                    self.data[o],
                    self.data[o + 1],
                    self.data[o + 2],
                    self.data[o + 3],
                ])
            };

            let read_f32_3 = |offset: usize| -> [f32; 3] {
                [read_f32(offset), read_f32(offset + 4), read_f32(offset + 8)]
            };

            let read_f32_4 = |offset: usize| -> [f32; 4] {
                [
                    read_f32(offset),
                    read_f32(offset + 4),
                    read_f32(offset + 8),
                    read_f32(offset + 12),
                ]
            };

            let read_f32_6 = |offset: usize| -> [f32; 6] {
                [
                    read_f32(offset),
                    read_f32(offset + 4),
                    read_f32(offset + 8),
                    read_f32(offset + 12),
                    read_f32(offset + 16),
                    read_f32(offset + 20),
                ]
            };

            let center = read_f32_3(0);
            let covariance = read_f32_6(12);
            let sh_coeffs = read_f32_3(36);
            let opacity = read_f32(48);
            let scale = read_f32_3(52);
            let rotation = read_f32_4(64);
            let time_range = [read_f32(80), read_f32(84)];
            let velocity = read_f32_3(88);

            let id_offset = base + 100;
            let id = u32::from_le_bytes([
                self.data[id_offset],
                self.data[id_offset + 1],
                self.data[id_offset + 2],
                self.data[id_offset + 3],
            ]);

            gaussians.push(Gaussian4D {
                center,
                covariance,
                sh_coeffs,
                opacity,
                scale,
                rotation,
                time_range,
                velocity,
                id,
            });
        }

        gaussians
    }

    /// Recompute and return the checksum over the data.
    pub fn compute_checksum(&self) -> u32 {
        compute_checksum(&self.data)
    }

    /// Verify that the stored checksum matches the data.
    pub fn verify_checksum(&self) -> bool {
        self.checksum == compute_checksum(&self.data)
    }
}

/// Simple additive hash checksum (not cryptographic).
///
/// Processes data in 4-byte chunks, treating each as a little-endian u32
/// and summing with wrapping arithmetic. Remaining bytes are incorporated
/// by shifting into a final u32.
fn compute_checksum(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5; // FNV offset basis
    for &byte in data {
        hash = hash.wrapping_mul(0x0100_0193); // FNV prime
        hash ^= byte as u32;
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::Gaussian4D;

    #[test]
    fn test_encode_decode_roundtrip() {
        let gaussians = vec![
            Gaussian4D::new([1.0, 2.0, 3.0], 10),
            Gaussian4D::new([4.0, 5.0, 6.0], 20),
        ];
        let block = PrimitiveBlock::encode(&gaussians, QuantTier::Hot8);
        assert_eq!(block.count, 2);
        assert!(block.verify_checksum());
        assert!(block.decode_descriptor.quantized);

        let decoded = block.decode();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].center, [1.0, 2.0, 3.0]);
        assert_eq!(decoded[0].id, 10);
        assert_eq!(decoded[1].center, [4.0, 5.0, 6.0]);
        assert_eq!(decoded[1].id, 20);
    }

    #[test]
    fn test_encode_decode_preserves_all_fields() {
        let mut g = Gaussian4D::new([1.0, 2.0, 3.0], 99);
        g.covariance = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        g.sh_coeffs = [0.7, 0.8, 0.9];
        g.opacity = 0.75;
        g.scale = [1.5, 2.5, 3.5];
        g.rotation = [0.5, 0.5, 0.5, 0.5];
        g.time_range = [0.0, 10.0];
        g.velocity = [0.1, 0.2, 0.3];

        // Warm5 falls back to Hot8. With a single Gaussian, min==max for every
        // field so dequantization is exact.
        let block = PrimitiveBlock::encode(&[g.clone()], QuantTier::Warm5);
        let decoded = block.decode();
        assert_eq!(decoded.len(), 1);
        let d = &decoded[0];
        assert_eq!(d.center, g.center);
        assert_eq!(d.covariance, g.covariance);
        assert_eq!(d.sh_coeffs, g.sh_coeffs);
        assert_eq!(d.opacity, g.opacity);
        assert_eq!(d.scale, g.scale);
        assert_eq!(d.rotation, g.rotation);
        assert_eq!(d.time_range, g.time_range);
        assert_eq!(d.velocity, g.velocity);
        assert_eq!(d.id, g.id);
    }

    #[test]
    fn test_empty_encode() {
        let block = PrimitiveBlock::encode(&[], QuantTier::Cold3);
        assert_eq!(block.count, 0);
        assert!(block.data.is_empty());
        let decoded = block.decode();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_checksum_changes_with_data() {
        let g1 = Gaussian4D::new([1.0, 0.0, 0.0], 1);
        let g2 = Gaussian4D::new([2.0, 0.0, 0.0], 2);
        let block1 = PrimitiveBlock::encode(&[g1], QuantTier::Hot8);
        let block2 = PrimitiveBlock::encode(&[g2], QuantTier::Hot8);
        assert_ne!(block1.checksum, block2.checksum);
    }

    #[test]
    fn test_tile_coord_equality() {
        let c1 = TileCoord {
            x: 1,
            y: 2,
            z: 3,
            time_bucket: 0,
            lod: 0,
        };
        let c2 = TileCoord {
            x: 1,
            y: 2,
            z: 3,
            time_bucket: 0,
            lod: 0,
        };
        assert_eq!(c1, c2);
    }

    // -----------------------------------------------------------------------
    // Quantization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hot8_smaller_than_raw() {
        // With 10 Gaussians, Hot8 should be significantly smaller than raw.
        // Raw: 10 * 104 = 1040 bytes
        // Hot8: 2 + 25*(8+10) + 10*4 = 2 + 450 + 40 = 492 bytes
        let gaussians: Vec<Gaussian4D> = (0..10)
            .map(|i| {
                let f = i as f32;
                Gaussian4D::new([f, f * 2.0, f * 3.0], i)
            })
            .collect();

        let quantized = PrimitiveBlock::encode(&gaussians, QuantTier::Hot8);
        let raw = PrimitiveBlock::encode_raw(&gaussians, QuantTier::Hot8);

        assert!(
            quantized.data.len() < raw.data.len(),
            "Hot8 ({} bytes) should be smaller than raw ({} bytes)",
            quantized.data.len(),
            raw.data.len(),
        );
        assert!(quantized.decode_descriptor.quantized);
        assert!(!raw.decode_descriptor.quantized);
    }

    #[test]
    fn test_hot8_roundtrip_within_tolerance() {
        // Create Gaussians with diverse values and verify roundtrip error is bounded.
        let mut gaussians = Vec::new();
        for i in 0..20u32 {
            let f = i as f32;
            let mut g = Gaussian4D::new([f * 0.5, f * -0.3, f * 1.7], i);
            g.covariance = [
                f * 0.01,
                f * 0.02,
                f * 0.03,
                f * 0.04,
                f * 0.05,
                f * 0.06,
            ];
            g.sh_coeffs = [f * 0.1, f * 0.2, f * 0.3];
            g.opacity = (i as f32) / 19.0;
            g.scale = [1.0 + f * 0.1, 2.0 + f * 0.2, 3.0 + f * 0.3];
            g.rotation = [0.5, 0.5, 0.5, 0.5];
            g.time_range = [f, f + 10.0];
            g.velocity = [f * 0.01, f * -0.01, f * 0.005];
            gaussians.push(g);
        }

        let block = PrimitiveBlock::encode(&gaussians, QuantTier::Hot8);
        assert!(block.verify_checksum());

        let decoded = block.decode();
        assert_eq!(decoded.len(), gaussians.len());

        for (orig, dec) in gaussians.iter().zip(decoded.iter()) {
            assert_eq!(orig.id, dec.id, "IDs must match exactly");

            let orig_floats = gaussian_to_floats(orig);
            let dec_floats = gaussian_to_floats(dec);

            for (field_idx, (&o, &d)) in
                orig_floats.iter().zip(dec_floats.iter()).enumerate()
            {
                if !o.is_finite() {
                    continue; // skip non-finite values in tolerance check
                }
                // Collect the range for this field across all Gaussians.
                let field_values: Vec<f32> = gaussians
                    .iter()
                    .map(|g| gaussian_to_floats(g)[field_idx])
                    .filter(|v| v.is_finite())
                    .collect();
                let fmin = field_values
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, f32::min);
                let fmax = field_values
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                let range = fmax - fmin;

                // Maximum quantization error is range / 255 (half-step rounding).
                let tolerance = if range == 0.0 {
                    0.0
                } else {
                    range / 255.0 + 1e-6
                };

                assert!(
                    (o - d).abs() <= tolerance,
                    "Field {} of gaussian {}: orig={}, decoded={}, tolerance={}",
                    field_idx,
                    orig.id,
                    o,
                    d,
                    tolerance,
                );
            }
        }
    }

    #[test]
    fn test_quantize_all_same_values() {
        // When all Gaussians have identical field values, quantization should be exact.
        let gaussians: Vec<Gaussian4D> =
            (0..5).map(|i| Gaussian4D::new([7.0, 7.0, 7.0], i)).collect();

        let block = PrimitiveBlock::encode(&gaussians, QuantTier::Hot8);
        let decoded = block.decode();

        for (i, d) in decoded.iter().enumerate() {
            assert_eq!(
                d.center,
                [7.0, 7.0, 7.0],
                "Gaussian {} center should be exact",
                i
            );
            assert_eq!(d.id, i as u32);
        }
    }

    #[test]
    fn test_quantize_extreme_ranges() {
        // Extreme range: values from 0 to 1_000_000.
        let g1 = Gaussian4D::new([0.0, 0.0, 0.0], 1);
        let g2 = Gaussian4D::new([1_000_000.0, 1_000_000.0, 1_000_000.0], 2);

        let block = PrimitiveBlock::encode(&[g1, g2], QuantTier::Hot8);
        let decoded = block.decode();

        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].id, 1);
        assert_eq!(decoded[1].id, 2);

        // With range 1e6 and 8-bit quantization, max error ~ 1e6 / 255 ~ 3922.
        let tolerance = 1_000_000.0 / 255.0 + 1.0;
        for field_idx in 0..3 {
            assert!(
                (decoded[0].center[field_idx] - 0.0).abs() <= tolerance,
                "g1 center[{}] = {}, expected ~0.0",
                field_idx,
                decoded[0].center[field_idx]
            );
            assert!(
                (decoded[1].center[field_idx] - 1_000_000.0).abs() <= tolerance,
                "g2 center[{}] = {}, expected ~1000000.0",
                field_idx,
                decoded[1].center[field_idx]
            );
        }
    }

    #[test]
    fn test_encode_raw_exact_roundtrip() {
        // Raw encoding must preserve all values exactly, including non-finite.
        let mut g = Gaussian4D::new([1.0, 2.0, 3.0], 42);
        g.covariance = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        g.sh_coeffs = [0.7, 0.8, 0.9];
        g.opacity = 0.75;
        g.scale = [1.5, 2.5, 3.5];
        g.rotation = [0.5, 0.5, 0.5, 0.5];
        g.time_range = [f32::NEG_INFINITY, f32::INFINITY];
        g.velocity = [0.1, 0.2, 0.3];

        let block = PrimitiveBlock::encode_raw(&[g.clone()], QuantTier::Hot8);
        assert!(!block.decode_descriptor.quantized);
        assert!(block.verify_checksum());

        let decoded = block.decode();
        assert_eq!(decoded.len(), 1);
        let d = &decoded[0];
        assert_eq!(d.center, g.center);
        assert_eq!(d.covariance, g.covariance);
        assert_eq!(d.sh_coeffs, g.sh_coeffs);
        assert_eq!(d.opacity, g.opacity);
        assert_eq!(d.scale, g.scale);
        assert_eq!(d.rotation, g.rotation);
        assert_eq!(d.time_range[0], f32::NEG_INFINITY);
        assert_eq!(d.time_range[1], f32::INFINITY);
        assert_eq!(d.velocity, g.velocity);
        assert_eq!(d.id, g.id);
    }

    #[test]
    fn test_checksum_differs_raw_vs_quantized() {
        let gaussians: Vec<Gaussian4D> = (0..5)
            .map(|i| Gaussian4D::new([i as f32, 0.0, 0.0], i))
            .collect();

        let quantized = PrimitiveBlock::encode(&gaussians, QuantTier::Hot8);
        let raw = PrimitiveBlock::encode_raw(&gaussians, QuantTier::Hot8);

        // Different encodings of the same data must produce different checksums.
        assert_ne!(
            quantized.checksum, raw.checksum,
            "Quantized and raw checksums should differ"
        );
    }

    #[test]
    fn test_decode_descriptor_quantized_flag() {
        let g = Gaussian4D::new([1.0, 2.0, 3.0], 1);

        let q_block = PrimitiveBlock::encode(&[g.clone()], QuantTier::Hot8);
        assert!(q_block.decode_descriptor.quantized);

        let r_block = PrimitiveBlock::encode_raw(&[g], QuantTier::Hot8);
        assert!(!r_block.decode_descriptor.quantized);
    }
}
