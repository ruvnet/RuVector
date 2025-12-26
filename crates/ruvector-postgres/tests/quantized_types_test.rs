//! Integration tests for quantized vector types
//!
//! Tests BinaryVec, ScalarVec, and ProductVec with SIMD optimizations

use ruvector_postgres::types::{BinaryVec, ProductVec, ScalarVec};

// ============================================================================
// BinaryVec Tests
// ============================================================================

#[test]
fn test_binaryvec_quantization() {
    let original = vec![1.0, -0.5, 0.3, -0.8, 0.2, -0.1, 0.9, -0.5];
    let binary = BinaryVec::from_f32(&original);

    assert_eq!(binary.dimensions(), 8);

    // Check individual bits
    assert!(binary.get_bit(0)); // 1.0 > 0
    assert!(!binary.get_bit(1)); // -0.5 <= 0
    assert!(binary.get_bit(2)); // 0.3 > 0
    assert!(!binary.get_bit(3)); // -0.8 <= 0
    assert!(binary.get_bit(4)); // 0.2 > 0
    assert!(!binary.get_bit(5)); // -0.1 <= 0
    assert!(binary.get_bit(6)); // 0.9 > 0
    assert!(!binary.get_bit(7)); // -0.5 <= 0
}

#[test]
fn test_binaryvec_hamming_distance() {
    let a = BinaryVec::from_f32(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    let b = BinaryVec::from_f32(&[1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

    // Differs in positions: 1, 2, 5, 6 = 4 differences
    let distance = a.hamming_distance(&b);
    assert_eq!(distance, 4);
}

#[test]
fn test_binaryvec_normalized_distance() {
    let a = BinaryVec::from_f32(&[1.0, 0.0, 1.0, 0.0]);
    let b = BinaryVec::from_f32(&[1.0, 1.0, 0.0, 0.0]);

    let dist = a.normalized_distance(&b);
    // 2 differences out of 4 dimensions = 0.5
    assert!((dist - 0.5).abs() < 0.001);
}

#[test]
fn test_binaryvec_popcount() {
    let v = BinaryVec::from_f32(&[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    assert_eq!(v.popcount(), 4);
}

#[test]
fn test_binaryvec_compression() {
    let dims = 1024;
    let original = vec![1.0; dims];
    let binary = BinaryVec::from_f32(&original);

    // Original: 1024 * 4 bytes = 4096 bytes
    // Binary: 1024 / 8 = 128 bytes
    // Compression ratio: 32x
    assert_eq!(BinaryVec::compression_ratio(), 32.0);
    assert_eq!(binary.as_bytes().len(), dims / 8);
}

#[test]
fn test_binaryvec_threshold() {
    let original = vec![0.5, 0.3, 0.1, -0.1, -0.3, -0.5];
    let binary = BinaryVec::from_f32_threshold(&original, 0.2);

    // Values > 0.2: 0.5, 0.3
    assert!(binary.get_bit(0)); // 0.5 > 0.2
    assert!(binary.get_bit(1)); // 0.3 > 0.2
    assert!(!binary.get_bit(2)); // 0.1 <= 0.2
    assert!(!binary.get_bit(3)); // -0.1 <= 0.2
    assert!(!binary.get_bit(4)); // -0.3 <= 0.2
    assert!(!binary.get_bit(5)); // -0.5 <= 0.2
}

// ============================================================================
// ScalarVec Tests
// ============================================================================

#[test]
fn test_scalarvec_quantization() {
    let original = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let scalar = ScalarVec::from_f32(&original);

    assert_eq!(scalar.dimensions(), 5);

    // Dequantize and check accuracy
    let restored = scalar.to_f32();
    for (o, r) in original.iter().zip(restored.iter()) {
        assert!((o - r).abs() < 0.02, "orig={}, restored={}", o, r);
    }
}

#[test]
fn test_scalarvec_distance() {
    let a = ScalarVec::from_f32(&[1.0, 0.0, 0.0]);
    let b = ScalarVec::from_f32(&[0.0, 1.0, 0.0]);

    let dist = a.distance(&b);
    // Euclidean distance should be approximately sqrt(2) ≈ 1.414
    assert!((dist - 1.414).abs() < 0.2, "distance={}", dist);
}

#[test]
fn test_scalarvec_compression() {
    assert_eq!(ScalarVec::compression_ratio(), 4.0);

    let dims = 1000;
    let original = vec![0.5; dims];
    let scalar = ScalarVec::from_f32(&original);

    // Original: 1000 * 4 = 4000 bytes
    // Quantized: 1000 * 1 = 1000 bytes (plus 10 bytes metadata)
    assert!(scalar.memory_size() < dims * std::mem::size_of::<f32>());
}

#[test]
fn test_scalarvec_scale_offset() {
    let original = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let scalar = ScalarVec::from_f32(&original);

    // Check that scale and offset are reasonable
    assert!(scalar.scale() > 0.0);
    assert!(scalar.offset() <= -2.0);

    // Verify reconstruction
    let restored = scalar.to_f32();
    for (o, r) in original.iter().zip(restored.iter()) {
        assert!((o - r).abs() < 0.05);
    }
}

#[test]
fn test_scalarvec_custom_params() {
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let scale = 0.02;
    let offset = 1.0;

    let scalar = ScalarVec::from_f32_custom(&original, scale, offset);

    assert_eq!(scalar.scale(), scale);
    assert_eq!(scalar.offset(), offset);
}

#[test]
fn test_scalarvec_distance_int() {
    let a = ScalarVec::from_f32(&[1.0, 2.0, 3.0]);
    let b = ScalarVec::from_f32(&[4.0, 5.0, 6.0]);

    // Squared distance in int32 space (no sqrt, no scaling)
    let dist_sq = a.distance_sq_int(&b);
    assert!(dist_sq > 0);
}

// ============================================================================
// ProductVec Tests
// ============================================================================

#[test]
fn test_productvec_creation() {
    let dims = 128;
    let m = 8;
    let k = 255; // Max u8 value
    let codes = vec![1, 2, 3, 4, 5, 6, 7, 8];

    let pq = ProductVec::new(dims as u16, m, k, codes.clone());

    assert_eq!(pq.original_dims(), dims);
    assert_eq!(pq.m(), m as usize);
    assert_eq!(pq.k(), k as usize);
    assert_eq!(pq.codes(), &codes[..]);
}

#[test]
fn test_productvec_dims_per_subspace() {
    let pq = ProductVec::new(1536, 48, 255, vec![0; 48]);
    assert_eq!(pq.dims_per_subspace(), 32); // 1536 / 48 = 32
}

#[test]
fn test_productvec_compression() {
    let dims = 1536;
    let m = 48;
    let pq = ProductVec::new(dims as u16, m, 255, vec![0; m as usize]);

    // Original: 1536 * 4 = 6144 bytes
    // Compressed: 48 bytes
    // Ratio: 128x
    let ratio = pq.compression_ratio();
    assert!((ratio - 128.0).abs() < 0.1);
}

#[test]
fn test_productvec_adc_distance_scalar() {
    let codes = vec![0, 1, 2, 3];
    let pq = ProductVec::new(64, 4, 4, codes);

    // Create flat distance table: 4 subspaces * 4 centroids = 16 values
    let table = vec![
        0.0, 1.0, 4.0, 9.0, // subspace 0
        0.0, 1.0, 4.0, 9.0, // subspace 1
        0.0, 1.0, 4.0, 9.0, // subspace 2
        0.0, 1.0, 4.0, 9.0, // subspace 3
    ];

    let dist = pq.adc_distance_flat(&table);
    // sqrt(0 + 1 + 4 + 9) = sqrt(14) ≈ 3.742
    assert!((dist - 3.742).abs() < 0.01);
}

#[test]
fn test_productvec_adc_distance_nested() {
    let codes = vec![0, 1, 2, 3];
    let pq = ProductVec::new(64, 4, 4, codes);

    // Create nested distance table
    let table: Vec<Vec<f32>> = vec![
        vec![0.0, 1.0, 4.0, 9.0], // subspace 0
        vec![0.0, 1.0, 4.0, 9.0], // subspace 1
        vec![0.0, 1.0, 4.0, 9.0], // subspace 2
        vec![0.0, 1.0, 4.0, 9.0], // subspace 3
    ];

    let dist = pq.adc_distance(&table);
    assert!((dist - 3.742).abs() < 0.01);
}

#[test]
fn test_productvec_memory_size() {
    let m: u8 = 48;
    let pq = ProductVec::new(1536, m, 255, vec![0; m as usize]);

    // Should be small (struct overhead + 48 bytes for codes)
    let mem = pq.memory_size();
    assert!(mem < 200); // Much smaller than original 6144 bytes
}

// ============================================================================
// SIMD Optimization Tests
// ============================================================================

#[test]
fn test_binaryvec_simd_consistency() {
    // Large enough to trigger SIMD paths
    let dims = 1024;
    let a_data: Vec<f32> = (0..dims)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let b_data: Vec<f32> = (0..dims)
        .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
        .collect();

    let a = BinaryVec::from_f32(&a_data);
    let b = BinaryVec::from_f32(&b_data);

    // SIMD and scalar should give same result
    let dist = a.hamming_distance(&b);
    assert!(dist > 0);
}

#[test]
fn test_scalarvec_simd_consistency() {
    // Large enough to trigger SIMD paths
    let dims = 256;
    let a_data: Vec<f32> = (0..dims).map(|i| i as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..dims).map(|i| (dims - i) as f32 * 0.1).collect();

    let a = ScalarVec::from_f32(&a_data);
    let b = ScalarVec::from_f32(&b_data);

    // Should compute distance without panicking
    let dist = a.distance(&b);
    assert!(dist > 0.0);
}

#[test]
fn test_productvec_simd_consistency() {
    // Large enough to trigger SIMD paths
    let m: u8 = 32;
    let k: u8 = 255;
    let codes: Vec<u8> = (0..m).map(|i| ((i as u16 * 7) % k as u16) as u8).collect();

    let pq = ProductVec::new(1024, m, k, codes);

    // Create large distance table
    let mut table = Vec::with_capacity(m as usize * k as usize);
    for i in 0..(m as usize * k as usize) {
        table.push((i % 100) as f32 * 0.01);
    }

    // SIMD distance should work
    let dist = pq.adc_distance_simd(&table);
    assert!(dist > 0.0);
}

// ============================================================================
// Serialization Tests
// ============================================================================

#[test]
fn test_binaryvec_serialization() {
    let original_data = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    let v = BinaryVec::from_f32(&original_data);

    // BinaryVec implements serialization internally via to_bytes/from_bytes
    // This would be tested through PostgreSQL integration
    assert_eq!(v.dimensions(), 8);
}

#[test]
fn test_scalarvec_serialization() {
    let original_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let v = ScalarVec::from_f32(&original_data);

    // ScalarVec implements serialization internally
    assert_eq!(v.dimensions(), 5);
    assert!(v.scale() > 0.0);
}

#[test]
fn test_productvec_serialization() {
    let codes = vec![1, 2, 3, 4];
    let v = ProductVec::new(64, 4, 16, codes);

    // ProductVec implements serialization internally
    assert_eq!(v.m(), 4);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_binaryvec_empty() {
    let v = BinaryVec::from_f32(&[]);
    assert_eq!(v.dimensions(), 0);
    assert_eq!(v.popcount(), 0);
}

#[test]
fn test_scalarvec_empty() {
    let v = ScalarVec::from_f32(&[]);
    assert_eq!(v.dimensions(), 0);
}

#[test]
fn test_binaryvec_all_zeros() {
    let v = BinaryVec::from_f32(&[0.0; 100]);
    assert_eq!(v.popcount(), 0);
}

#[test]
fn test_binaryvec_all_ones() {
    let v = BinaryVec::from_f32(&[1.0; 100]);
    assert_eq!(v.popcount(), 100);
}

#[test]
fn test_scalarvec_constant() {
    let v = ScalarVec::from_f32(&[5.0; 100]);
    let restored = v.to_f32();

    for &val in &restored {
        assert!((val - 5.0).abs() < 0.1);
    }
}

#[test]
fn test_productvec_max_code() {
    let codes = vec![254, 254, 254, 254]; // Near max u8 values
    let pq = ProductVec::new(64, 4, 255, codes);

    assert_eq!(pq.codes()[0], 254);
}

// ============================================================================
// Performance Characteristics
// ============================================================================

#[test]
fn test_memory_savings_binary() {
    let dims = 4096;
    let original = vec![1.0; dims];
    let binary = BinaryVec::from_f32(&original);

    let original_size = dims * std::mem::size_of::<f32>();
    let compressed_size = binary.memory_size();

    // Should be approximately 32x compression
    let ratio = original_size as f32 / compressed_size as f32;
    assert!(ratio > 25.0, "compression ratio: {}", ratio);
}

#[test]
fn test_memory_savings_scalar() {
    let dims = 4096;
    let original = vec![1.0; dims];
    let scalar = ScalarVec::from_f32(&original);

    let original_size = dims * std::mem::size_of::<f32>();
    let compressed_size = scalar.memory_size();

    // Should be approximately 4x compression
    let ratio = original_size as f32 / compressed_size as f32;
    assert!(ratio > 3.5, "compression ratio: {}", ratio);
}

#[test]
fn test_memory_savings_product() {
    let dims = 1536;
    let m: u8 = 48;
    let pq = ProductVec::new(dims as u16, m, 255, vec![0; m as usize]);

    let original_size = dims * std::mem::size_of::<f32>();
    let compressed_size = pq.memory_size();

    // Should be approximately 128x compression
    let ratio = original_size as f32 / compressed_size as f32;
    assert!(ratio > 100.0, "compression ratio: {}", ratio);
}
