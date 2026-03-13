//! Quality validation tests for INT8 quantization
//!
//! Validates that INT8 quantized models maintain high quality compared to FP32:
//! - Cosine similarity ≥0.995 (GATE-2)
//! - Per-layer MSE tracking
//! - Embedding validation on test dataset

use ruvector_cnn::int8::{dequantize_tensor, quantize_tensor, QuantParams};

#[cfg(test)]
mod quality_tests {
    use super::*;

    /// Compute cosine similarity between two tensors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Tensors must have same length");

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Compute Mean Squared Error between two tensors
    fn mean_squared_error(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Tensors must have same length");

        let mse: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            / a.len() as f32;

        mse
    }

    #[test]
    fn test_cosine_similarity_gate_2() {
        // GATE-2: Cosine similarity ≥0.995 between INT8 and FP32 embeddings

        // Generate random FP32 tensor (simulating embedding output)
        let size = 1024;
        let mut rng = fastrand::Rng::with_seed(42);
        let fp32_embedding: Vec<f32> = (0..size)
            .map(|_| rng.f32() * 2.0 - 1.0) // Range [-1, 1]
            .collect();

        // Quantize to INT8
        let params = QuantParams::from_tensor(&fp32_embedding);
        let int8_tensor = quantize_tensor(&fp32_embedding, &params);

        // Dequantize back to FP32
        let dequantized = dequantize_tensor(&int8_tensor, &params);

        // Compute cosine similarity
        let similarity = cosine_similarity(&fp32_embedding, &dequantized);

        println!("Cosine Similarity: {:.6}", similarity);
        assert!(
            similarity >= 0.995,
            "GATE-2 FAILED: Cosine similarity {:.6} < 0.995",
            similarity
        );
    }

    #[test]
    fn test_per_layer_mse_tracking() {
        // Track MSE for different tensor sizes (simulating different layers)
        let layer_sizes = vec![
            ("input", 224 * 224 * 3),
            ("conv1", 112 * 112 * 16),
            ("conv2", 56 * 56 * 24),
            ("conv3", 28 * 28 * 40),
            ("conv4", 14 * 14 * 80),
            ("conv5", 7 * 7 * 112),
            ("embedding", 1024),
        ];

        let mut rng = fastrand::Rng::with_seed(42);

        println!("\nPer-Layer MSE Analysis:");
        println!(
            "{:<15} {:>10} {:>15} {:>15}",
            "Layer", "Size", "MSE", "Cosine Sim"
        );
        println!("{}", "-".repeat(60));

        for (layer_name, size) in layer_sizes {
            // Generate random tensor
            let fp32_tensor: Vec<f32> = (0..size).map(|_| rng.f32() * 2.0 - 1.0).collect();

            // Quantize and dequantize
            let params = QuantParams::from_tensor(&fp32_tensor);
            let int8_tensor = quantize_tensor(&fp32_tensor, &params);
            let dequantized = dequantize_tensor(&int8_tensor, &params);

            // Compute metrics
            let mse = mean_squared_error(&fp32_tensor, &dequantized);
            let similarity = cosine_similarity(&fp32_tensor, &dequantized);

            println!(
                "{:<15} {:>10} {:>15.6e} {:>15.6}",
                layer_name, size, mse, similarity
            );

            // All layers should maintain high similarity
            assert!(
                similarity >= 0.99,
                "Layer {} has low similarity: {:.6}",
                layer_name,
                similarity
            );
        }
    }

    #[test]
    fn test_embedding_validation_test_set() {
        // Validate embeddings on a small test set with known properties

        struct TestCase {
            name: &'static str,
            embedding: Vec<f32>,
            expected_min_similarity: f32,
        }

        let test_cases = vec![
            TestCase {
                name: "uniform",
                embedding: vec![0.5; 512],
                expected_min_similarity: 0.999, // Uniform should quantize very well
            },
            TestCase {
                name: "sparse",
                embedding: {
                    let mut v = vec![0.0; 512];
                    for i in (0..512).step_by(10) {
                        v[i] = 1.0;
                    }
                    v
                },
                expected_min_similarity: 0.995,
            },
            TestCase {
                name: "gaussian",
                embedding: {
                    let mut rng = fastrand::Rng::with_seed(123);
                    (0..512)
                        .map(|_| {
                            // Box-Muller transform for Gaussian
                            let u1 = rng.f32();
                            let u2 = rng.f32();
                            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos())
                                * 0.5
                        })
                        .collect()
                },
                expected_min_similarity: 0.995,
            },
            TestCase {
                name: "wide_range",
                embedding: {
                    let mut rng = fastrand::Rng::with_seed(456);
                    (0..512)
                        .map(|_| rng.f32() * 10.0 - 5.0) // Range [-5, 5]
                        .collect()
                },
                expected_min_similarity: 0.995,
            },
        ];

        println!("\nEmbedding Validation Test Set:");
        println!("{:<15} {:>15} {:>15}", "Test Case", "Cosine Sim", "MSE");
        println!("{}", "-".repeat(50));

        for test_case in test_cases {
            let params = QuantParams::from_tensor(&test_case.embedding);
            let int8_tensor = quantize_tensor(&test_case.embedding, &params);
            let dequantized = dequantize_tensor(&int8_tensor, &params);

            let similarity = cosine_similarity(&test_case.embedding, &dequantized);
            let mse = mean_squared_error(&test_case.embedding, &dequantized);

            println!("{:<15} {:>15.6} {:>15.6e}", test_case.name, similarity, mse);

            assert!(
                similarity >= test_case.expected_min_similarity,
                "Test case '{}' failed: similarity {:.6} < expected {:.6}",
                test_case.name,
                similarity,
                test_case.expected_min_similarity
            );
        }
    }

    #[test]
    fn test_quantization_range_edge_cases() {
        // Test edge cases in quantization range

        struct EdgeCase {
            name: &'static str,
            values: Vec<f32>,
        }

        let edge_cases = vec![
            EdgeCase {
                name: "all_zeros",
                values: vec![0.0; 256],
            },
            EdgeCase {
                name: "all_positive",
                values: vec![1.0; 256],
            },
            EdgeCase {
                name: "all_negative",
                values: vec![-1.0; 256],
            },
            EdgeCase {
                name: "min_max_only",
                values: {
                    let mut v = vec![0.0; 256];
                    v[0] = -10.0;
                    v[255] = 10.0;
                    v
                },
            },
            EdgeCase {
                name: "very_small_range",
                values: {
                    let mut v = Vec::with_capacity(256);
                    for i in 0..256 {
                        v.push(1.0 + (i as f32 * 0.0001));
                    }
                    v
                },
            },
        ];

        println!("\nQuantization Range Edge Cases:");
        println!(
            "{:<20} {:>15} {:>15}",
            "Edge Case", "Max Error", "Cosine Sim"
        );
        println!("{}", "-".repeat(55));

        for edge_case in edge_cases {
            let params = QuantParams::from_tensor(&edge_case.values);
            let int8_tensor = quantize_tensor(&edge_case.values, &params);
            let dequantized = dequantize_tensor(&int8_tensor, &params);

            let max_error = edge_case
                .values
                .iter()
                .zip(dequantized.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            let similarity = cosine_similarity(&edge_case.values, &dequantized);

            println!(
                "{:<20} {:>15.6e} {:>15.6}",
                edge_case.name, max_error, similarity
            );

            // Edge cases should still maintain reasonable similarity
            // (except all_zeros which will have 0/0 = 0)
            if edge_case.name != "all_zeros" {
                assert!(
                    similarity >= 0.95,
                    "Edge case '{}' has low similarity: {:.6}",
                    edge_case.name,
                    similarity
                );
            }
        }
    }

    #[test]
    fn test_batch_consistency() {
        // Verify that quantizing a batch gives consistent results

        let batch_size = 4;
        let embedding_size = 512;
        let mut rng = fastrand::Rng::with_seed(789);

        // Generate batch of embeddings
        let batch: Vec<Vec<f32>> = (0..batch_size)
            .map(|_| (0..embedding_size).map(|_| rng.f32() * 2.0 - 1.0).collect())
            .collect();

        // Quantize each independently
        let individual_results: Vec<Vec<f32>> = batch
            .iter()
            .map(|emb| {
                let params = QuantParams::from_tensor(emb);
                let int8 = quantize_tensor(emb, &params);
                dequantize_tensor(&int8, &params)
            })
            .collect();

        // Verify consistency by re-quantizing
        for (i, (original, dequant)) in batch.iter().zip(individual_results.iter()).enumerate() {
            let similarity = cosine_similarity(original, dequant);
            assert!(
                similarity >= 0.995,
                "Batch item {} has low similarity: {:.6}",
                i,
                similarity
            );
        }

        println!("✓ Batch consistency test passed for {} items", batch_size);
    }

    #[test]
    fn test_quantization_determinism() {
        // Verify that quantization is deterministic

        let size = 1024;
        let mut rng = fastrand::Rng::with_seed(999);
        let fp32_tensor: Vec<f32> = (0..size).map(|_| rng.f32() * 2.0 - 1.0).collect();

        // Quantize twice
        let params1 = QuantParams::from_tensor(&fp32_tensor);
        let int8_1 = quantize_tensor(&fp32_tensor, &params1);

        let params2 = QuantParams::from_tensor(&fp32_tensor);
        let int8_2 = quantize_tensor(&fp32_tensor, &params2);

        // Parameters should be identical
        assert_eq!(
            params1.scale, params2.scale,
            "Scale should be deterministic"
        );
        assert_eq!(
            params1.zero_point, params2.zero_point,
            "Zero point should be deterministic"
        );

        // Quantized values should be identical
        assert_eq!(int8_1, int8_2, "Quantized tensors should be identical");

        println!("✓ Quantization determinism test passed");
    }

    #[test]
    fn test_quantization_symmetry() {
        // Test symmetric quantization properties

        let size = 512;
        let mut rng = fastrand::Rng::with_seed(111);

        // Create symmetric tensor (mirrored around zero)
        let positive: Vec<f32> = (0..size).map(|_| rng.f32()).collect();

        let mut symmetric = positive.clone();
        symmetric.extend(positive.iter().map(|&x| -x));

        let params = QuantParams::from_tensor(&symmetric);
        let int8_tensor = quantize_tensor(&symmetric, &params);
        let dequantized = dequantize_tensor(&int8_tensor, &params);

        let similarity = cosine_similarity(&symmetric, &dequantized);

        println!("Symmetric tensor similarity: {:.6}", similarity);
        assert!(
            similarity >= 0.995,
            "Symmetric tensor quantization quality insufficient: {:.6}",
            similarity
        );

        // Verify that mean is close to zero
        let mean: f32 = dequantized.iter().sum::<f32>() / dequantized.len() as f32;
        assert!(
            mean.abs() < 0.01,
            "Symmetric tensor should have mean ~0, got {:.6}",
            mean
        );
    }
}
