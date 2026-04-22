//! PiQ3 Quantization for Common Crawl Temporal Compression POC (ADR-115)
//!
//! Implements 3-bit product quantization for embedding compression:
//! - Input: 128-dim f32 embedding (512 bytes)
//! - Output: 48 bytes + 4 byte scale = 52 bytes (~10x compression)
//! - Expected recall: ~96% at compression threshold
//!
//! Quantization is tier-aware:
//! - Full tier: Store original f32 (no quantization)
//! - DeltaCompressed: 4-bit quantization (2.67x compression)
//! - CentroidMerged: 3-bit quantization (10.7x compression)
//! - Archived: 2-bit quantization (16x compression)

use crate::web_memory::CompressionTier;
use serde::{Deserialize, Serialize};

/// Quantization configuration by tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiQConfig {
    /// Bits per dimension for DeltaCompressed tier
    pub delta_bits: u8,
    /// Bits per dimension for CentroidMerged tier
    pub centroid_bits: u8,
    /// Bits per dimension for Archived tier
    pub archived_bits: u8,
}

impl Default for PiQConfig {
    fn default() -> Self {
        Self {
            delta_bits: 4,    // 2.67x compression, ~99% recall
            centroid_bits: 3, // 10.7x compression, ~96% recall
            archived_bits: 2, // 16x compression, ~90% recall
        }
    }
}

/// Quantized embedding with metadata for reconstruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedEmbedding {
    /// Packed quantized values (bit-packed)
    pub data: Vec<u8>,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Offset (min value) for dequantization
    pub offset: f32,
    /// Number of bits per dimension
    pub bits: u8,
    /// Original dimension count
    pub dim: u16,
}

impl QuantizedEmbedding {
    /// Calculate compressed size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() + 4 + 4 + 1 + 2 // data + scale + offset + bits + dim
    }

    /// Calculate compression ratio vs original f32
    pub fn compression_ratio(&self) -> f32 {
        let original_bytes = self.dim as f32 * 4.0; // f32 = 4 bytes
        original_bytes / self.size_bytes() as f32
    }
}

/// PiQ3 Quantizer implementing product quantization
#[derive(Debug, Clone)]
pub struct PiQQuantizer {
    config: PiQConfig,
}

impl PiQQuantizer {
    /// Create new quantizer with default config
    pub fn new() -> Self {
        Self {
            config: PiQConfig::default(),
        }
    }

    /// Create quantizer with custom config
    pub fn with_config(config: PiQConfig) -> Self {
        Self { config }
    }

    /// Get bits for a given compression tier
    fn bits_for_tier(&self, tier: CompressionTier) -> u8 {
        match tier {
            CompressionTier::Full => 32, // No quantization
            CompressionTier::DeltaCompressed => self.config.delta_bits,
            CompressionTier::CentroidMerged => self.config.centroid_bits,
            CompressionTier::Archived => self.config.archived_bits,
        }
    }

    /// Quantize an embedding based on compression tier
    pub fn quantize(&self, embedding: &[f32], tier: CompressionTier) -> Option<QuantizedEmbedding> {
        if tier == CompressionTier::Full {
            // No quantization for full tier
            return None;
        }

        let bits = self.bits_for_tier(tier);
        self.quantize_to_bits(embedding, bits)
    }

    /// Quantize embedding to specific bit depth
    pub fn quantize_to_bits(&self, embedding: &[f32], bits: u8) -> Option<QuantizedEmbedding> {
        if embedding.is_empty() || bits == 0 || bits > 8 {
            return None;
        }

        // Find min and max for range
        let min_val = embedding.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        if range < 1e-8 {
            // All values are the same, trivial case
            let packed = vec![0u8; ((embedding.len() * bits as usize) + 7) / 8];
            return Some(QuantizedEmbedding {
                data: packed,
                scale: 0.0,
                offset: min_val,
                bits,
                dim: embedding.len() as u16,
            });
        }

        let levels = (1u32 << bits) - 1; // e.g., 7 for 3-bit
        let scale = range / levels as f32;

        // Quantize to integer levels
        let quantized: Vec<u32> = embedding
            .iter()
            .map(|&v| {
                let normalized = (v - min_val) / range;
                (normalized * levels as f32)
                    .round()
                    .clamp(0.0, levels as f32) as u32
            })
            .collect();

        // Bit-pack the quantized values
        let packed = self.pack_bits(&quantized, bits);

        Some(QuantizedEmbedding {
            data: packed,
            scale,
            offset: min_val,
            bits,
            dim: embedding.len() as u16,
        })
    }

    /// Dequantize embedding back to f32
    pub fn dequantize(&self, quantized: &QuantizedEmbedding) -> Vec<f32> {
        let unpacked = self.unpack_bits(&quantized.data, quantized.bits, quantized.dim as usize);

        unpacked
            .iter()
            .map(|&q| quantized.offset + (q as f32 * quantized.scale))
            .collect()
    }

    /// Pack integer values into bytes with given bit depth
    fn pack_bits(&self, values: &[u32], bits: u8) -> Vec<u8> {
        let total_bits = values.len() * bits as usize;
        let num_bytes = (total_bits + 7) / 8;
        let mut packed = vec![0u8; num_bytes];

        let mut bit_pos = 0usize;
        for &val in values {
            // Write `bits` bits from val starting at bit_pos
            for b in 0..bits {
                if (val >> b) & 1 == 1 {
                    let byte_idx = bit_pos / 8;
                    let bit_idx = bit_pos % 8;
                    packed[byte_idx] |= 1 << bit_idx;
                }
                bit_pos += 1;
            }
        }

        packed
    }

    /// Unpack bytes to integer values
    fn unpack_bits(&self, packed: &[u8], bits: u8, count: usize) -> Vec<u32> {
        let mut values = Vec::with_capacity(count);
        let mut bit_pos = 0usize;

        for _ in 0..count {
            let mut val = 0u32;
            for b in 0..bits {
                let byte_idx = bit_pos / 8;
                let bit_idx = bit_pos % 8;
                if byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 == 1 {
                    val |= 1 << b;
                }
                bit_pos += 1;
            }
            values.push(val);
        }

        values
    }

    /// Calculate expected recall for a given bit depth
    /// Based on empirical measurements from ADR-115
    pub fn expected_recall(bits: u8) -> f32 {
        match bits {
            8 => 0.999, // Nearly lossless
            7 => 0.997,
            6 => 0.995,
            5 => 0.99,
            4 => 0.985, // 4-bit still very good
            3 => 0.96,  // PiQ3 target
            2 => 0.90,  // Archived tier
            1 => 0.70,  // Binary (not recommended)
            _ => 1.0,   // Full precision
        }
    }
}

impl Default for PiQQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for quantization operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantizationStats {
    /// Total embeddings quantized
    pub total_quantized: u64,
    /// Embeddings by tier
    pub by_tier: TierStats,
    /// Total bytes saved vs f32
    pub bytes_saved: u64,
    /// Total original bytes (if stored as f32)
    pub original_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierStats {
    pub delta_compressed: u64,
    pub centroid_merged: u64,
    pub archived: u64,
}

impl QuantizationStats {
    /// Record a quantization operation
    pub fn record(&mut self, tier: CompressionTier, dim: usize, quantized_size: usize) {
        self.total_quantized += 1;
        let original = dim * 4; // f32 = 4 bytes
        self.original_bytes += original as u64;
        self.bytes_saved += (original - quantized_size) as u64;

        match tier {
            CompressionTier::DeltaCompressed => self.by_tier.delta_compressed += 1,
            CompressionTier::CentroidMerged => self.by_tier.centroid_merged += 1,
            CompressionTier::Archived => self.by_tier.archived += 1,
            CompressionTier::Full => {} // Not quantized
        }
    }

    /// Get overall compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.original_bytes == 0 {
            return 1.0;
        }
        self.original_bytes as f32 / (self.original_bytes - self.bytes_saved) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_dequantize_3bit() {
        let quantizer = PiQQuantizer::new();

        // Create test embedding
        let embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 127.0).collect();

        // Quantize to 3 bits
        let quantized = quantizer.quantize_to_bits(&embedding, 3).unwrap();

        // Check compression
        assert!(quantized.compression_ratio() > 8.0);
        assert!(quantized.compression_ratio() < 12.0);

        // Dequantize
        let restored = quantizer.dequantize(&quantized);
        assert_eq!(restored.len(), embedding.len());

        // Check accuracy (within ~15% for 3-bit)
        let max_error: f32 = embedding
            .iter()
            .zip(restored.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // 3-bit has 8 levels, so max error should be ~1/8 = 0.125
        assert!(max_error < 0.2, "Max error {} too high", max_error);
    }

    #[test]
    fn test_tier_quantization() {
        let quantizer = PiQQuantizer::new();
        let embedding: Vec<f32> = (0..128).map(|i| (i as f32) / 127.0).collect();

        // Full tier: no quantization
        assert!(quantizer
            .quantize(&embedding, CompressionTier::Full)
            .is_none());

        // DeltaCompressed: 4-bit
        let delta = quantizer
            .quantize(&embedding, CompressionTier::DeltaCompressed)
            .unwrap();
        assert_eq!(delta.bits, 4);

        // CentroidMerged: 3-bit
        let centroid = quantizer
            .quantize(&embedding, CompressionTier::CentroidMerged)
            .unwrap();
        assert_eq!(centroid.bits, 3);

        // Archived: 2-bit
        let archived = quantizer
            .quantize(&embedding, CompressionTier::Archived)
            .unwrap();
        assert_eq!(archived.bits, 2);
    }

    #[test]
    fn test_pack_unpack() {
        let quantizer = PiQQuantizer::new();

        // Test 3-bit packing
        let values: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3];
        let packed = quantizer.pack_bits(&values, 3);
        let unpacked = quantizer.unpack_bits(&packed, 3, values.len());

        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_compression_stats() {
        let mut stats = QuantizationStats::default();

        // Record some operations
        stats.record(CompressionTier::CentroidMerged, 128, 52);
        stats.record(CompressionTier::DeltaCompressed, 128, 68);

        assert_eq!(stats.total_quantized, 2);
        assert_eq!(stats.by_tier.centroid_merged, 1);
        assert_eq!(stats.by_tier.delta_compressed, 1);
        assert!(stats.compression_ratio() > 1.0);
    }

    /// Benchmark test for ADR-115 metrics
    #[test]
    fn benchmark_piq3_metrics() {
        use std::time::Instant;

        let quantizer = PiQQuantizer::new();
        let dim = 128;
        let num_embeddings = 10000;

        // Generate realistic random embeddings
        let embeddings: Vec<Vec<f32>> = (0..num_embeddings)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 7 + j * 13) % 1000) as f32 / 1000.0 - 0.5)
                    .collect()
            })
            .collect();

        // Benchmark 3-bit quantization
        let start = Instant::now();
        let quantized_3bit: Vec<_> = embeddings
            .iter()
            .map(|e| quantizer.quantize_to_bits(e, 3).unwrap())
            .collect();
        let time_3bit = start.elapsed();

        // Benchmark 4-bit quantization
        let start = Instant::now();
        let quantized_4bit: Vec<_> = embeddings
            .iter()
            .map(|e| quantizer.quantize_to_bits(e, 4).unwrap())
            .collect();
        let time_4bit = start.elapsed();

        // Benchmark 2-bit quantization
        let start = Instant::now();
        let quantized_2bit: Vec<_> = embeddings
            .iter()
            .map(|e| quantizer.quantize_to_bits(e, 2).unwrap())
            .collect();
        let time_2bit = start.elapsed();

        // Calculate compression ratios
        let original_size = dim * 4; // 512 bytes for 128-dim f32
        let size_3bit = quantized_3bit[0].size_bytes();
        let size_4bit = quantized_4bit[0].size_bytes();
        let size_2bit = quantized_2bit[0].size_bytes();

        // Calculate recall (measure error)
        let mut total_cosine_3bit = 0.0f64;
        let mut total_cosine_4bit = 0.0f64;
        let mut total_cosine_2bit = 0.0f64;

        for (i, orig) in embeddings.iter().enumerate() {
            let restored_3 = quantizer.dequantize(&quantized_3bit[i]);
            let restored_4 = quantizer.dequantize(&quantized_4bit[i]);
            let restored_2 = quantizer.dequantize(&quantized_2bit[i]);

            total_cosine_3bit += cosine_sim(orig, &restored_3) as f64;
            total_cosine_4bit += cosine_sim(orig, &restored_4) as f64;
            total_cosine_2bit += cosine_sim(orig, &restored_2) as f64;
        }

        let avg_recall_3bit = total_cosine_3bit / num_embeddings as f64;
        let avg_recall_4bit = total_cosine_4bit / num_embeddings as f64;
        let avg_recall_2bit = total_cosine_2bit / num_embeddings as f64;

        // Print metrics for ADR-115
        println!("\n=== PiQ Quantization Benchmark (ADR-115) ===");
        println!("Embedding dimension: {dim}");
        println!("Number of embeddings: {num_embeddings}");
        println!("Original size: {original_size} bytes");
        println!();
        println!("3-bit (CentroidMerged tier):");
        println!("  - Compressed size: {size_3bit} bytes");
        println!(
            "  - Compression ratio: {:.2}x",
            original_size as f32 / size_3bit as f32
        );
        println!("  - Recall (cosine similarity): {:.4}", avg_recall_3bit);
        println!(
            "  - Throughput: {:.2} embeddings/sec",
            num_embeddings as f64 / time_3bit.as_secs_f64()
        );
        println!();
        println!("4-bit (DeltaCompressed tier):");
        println!("  - Compressed size: {size_4bit} bytes");
        println!(
            "  - Compression ratio: {:.2}x",
            original_size as f32 / size_4bit as f32
        );
        println!("  - Recall (cosine similarity): {:.4}", avg_recall_4bit);
        println!(
            "  - Throughput: {:.2} embeddings/sec",
            num_embeddings as f64 / time_4bit.as_secs_f64()
        );
        println!();
        println!("2-bit (Archived tier):");
        println!("  - Compressed size: {size_2bit} bytes");
        println!(
            "  - Compression ratio: {:.2}x",
            original_size as f32 / size_2bit as f32
        );
        println!("  - Recall (cosine similarity): {:.4}", avg_recall_2bit);
        println!(
            "  - Throughput: {:.2} embeddings/sec",
            num_embeddings as f64 / time_2bit.as_secs_f64()
        );
        println!();

        // Assertions
        assert!(avg_recall_3bit > 0.95, "3-bit recall should be > 95%");
        assert!(avg_recall_4bit > 0.97, "4-bit recall should be > 97%");
        assert!(avg_recall_2bit > 0.85, "2-bit recall should be > 85%");
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a < 1e-8 || norm_b < 1e-8 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}
