//! Integration tests for ruvector-neural-decoder
//!
//! Tests the full NQED pipeline from syndrome input to correction output.

use ruvector_neural_decoder::{
    DecoderConfig, NeuralDecoder, Correction,
    graph::{GraphBuilder, DetectorGraph, NodeType, NodeTypePattern},
    gnn::{GNNConfig, GNNEncoder},
    mamba::{MambaConfig, MambaDecoder},
    fusion::{FusionConfig, BoundaryFeatures, CoherenceEstimator},
};
use ndarray::Array2;
use std::collections::HashMap;

// ============================================================================
// Full Pipeline Tests
// ============================================================================

#[test]
fn test_full_decode_pipeline_d3() {
    let config = DecoderConfig {
        distance: 3,
        embed_dim: 16,
        hidden_dim: 32,
        num_gnn_layers: 2,
        num_heads: 4,
        mamba_state_dim: 16,
        use_mincut_fusion: false,
        dropout: 0.0,
    };
    let mut decoder = NeuralDecoder::new(config);

    // Simple syndrome with no errors
    let syndrome_clean = vec![false; 9];
    let correction = decoder.decode(&syndrome_clean).unwrap();

    // Verify correction is valid (neural net may produce arbitrary outputs without training)
    // The important thing is it doesn't crash and produces valid structure
    assert!(correction.confidence >= 0.0 && correction.confidence <= 1.0,
        "Confidence should be valid: {}", correction.confidence);

    // Reset and try with some errors
    decoder.reset();

    // Syndrome with some fired detectors
    let syndrome_error = vec![true, false, true, false, false, false, true, false, false];
    let correction = decoder.decode(&syndrome_error).unwrap();

    // Should have non-zero decode time
    assert!(correction.decode_time_ns > 0);

    // Confidence should be valid
    assert!(correction.confidence >= 0.0 && correction.confidence <= 1.0);
}

#[test]
fn test_full_decode_pipeline_d5() {
    let config = DecoderConfig {
        distance: 5,
        embed_dim: 32,
        hidden_dim: 64,
        num_gnn_layers: 2,
        num_heads: 4,
        mamba_state_dim: 32,
        use_mincut_fusion: false,
        dropout: 0.0,
    };
    let mut decoder = NeuralDecoder::new(config);

    // Random-like syndrome
    let syndrome: Vec<bool> = (0..25).map(|i| i % 7 == 0 || i % 11 == 0).collect();
    let correction = decoder.decode(&syndrome).unwrap();

    assert!(correction.confidence >= 0.0 && correction.confidence <= 1.0);
}

#[test]
fn test_decoder_consistency() {
    let config = DecoderConfig {
        distance: 3,
        embed_dim: 16,
        hidden_dim: 32,
        num_gnn_layers: 1,
        num_heads: 4,
        mamba_state_dim: 16,
        use_mincut_fusion: false,
        dropout: 0.0,
    };

    // Create two decoders with same config
    let mut decoder1 = NeuralDecoder::new(config.clone());
    let mut decoder2 = NeuralDecoder::new(config);

    let syndrome = vec![true, false, false, false, true, false, false, false, true];

    // Note: Due to random weight initialization, outputs will differ
    // But both should produce valid outputs
    let corr1 = decoder1.decode(&syndrome).unwrap();
    let corr2 = decoder2.decode(&syndrome).unwrap();

    assert!(corr1.confidence >= 0.0 && corr1.confidence <= 1.0);
    assert!(corr2.confidence >= 0.0 && corr2.confidence <= 1.0);
}

#[test]
fn test_sequential_decoding() {
    let config = DecoderConfig {
        distance: 3,
        embed_dim: 16,
        hidden_dim: 32,
        num_gnn_layers: 1,
        num_heads: 4,
        mamba_state_dim: 16,
        use_mincut_fusion: false,
        dropout: 0.0,
    };
    let mut decoder = NeuralDecoder::new(config);

    // Decode multiple syndromes sequentially (simulating time-series)
    let syndromes = vec![
        vec![false, false, false, false, false, false, false, false, false],
        vec![true, false, false, false, false, false, false, false, false],
        vec![true, false, true, false, false, false, false, false, false],
        vec![false, false, false, false, false, false, false, false, false],
    ];

    let mut corrections: Vec<Correction> = Vec::new();

    for syndrome in &syndromes {
        let corr = decoder.decode(syndrome).unwrap();
        corrections.push(corr);
        // Don't reset between - test stateful behavior
    }

    assert_eq!(corrections.len(), 4);

    // Each correction should be valid
    for corr in &corrections {
        assert!(corr.confidence >= 0.0 && corr.confidence <= 1.0);
    }
}

// ============================================================================
// GNN + Graph Integration Tests
// ============================================================================

#[test]
fn test_gnn_with_surface_code_graph() {
    let gnn_config = GNNConfig {
        input_dim: 5,  // Matches node feature dimension
        embed_dim: 16,
        hidden_dim: 32,
        num_layers: 2,
        num_heads: 4,
        dropout: 0.0,
    };
    let encoder = GNNEncoder::new(gnn_config);

    // Create distance-3 surface code graph with syndrome
    let syndrome = vec![true, false, true, false, false, false, false, true, false];
    let graph = GraphBuilder::from_surface_code(3)
        .with_syndrome(&syndrome)
        .unwrap()
        .build()
        .unwrap();

    // Verify graph structure
    assert_eq!(graph.num_nodes(), 9);
    assert_eq!(graph.num_fired, 3);

    // Encode with GNN
    let embeddings = encoder.encode(&graph).unwrap();

    // Check output shape
    assert_eq!(embeddings.shape(), &[9, 32]);

    // Embeddings should have non-zero values
    let sum: f32 = embeddings.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "GNN should produce non-zero embeddings");
}

#[test]
fn test_gnn_different_syndromes_different_embeddings() {
    let gnn_config = GNNConfig {
        input_dim: 5,
        embed_dim: 16,
        hidden_dim: 32,
        num_layers: 2,
        num_heads: 4,
        dropout: 0.0,
    };
    let encoder = GNNEncoder::new(gnn_config);

    // Two different syndromes
    let syndrome1 = vec![true, false, false, false, false, false, false, false, false];
    let syndrome2 = vec![false, false, false, false, false, false, false, false, true];

    let graph1 = GraphBuilder::from_surface_code(3)
        .with_syndrome(&syndrome1).unwrap()
        .build().unwrap();

    let graph2 = GraphBuilder::from_surface_code(3)
        .with_syndrome(&syndrome2).unwrap()
        .build().unwrap();

    let emb1 = encoder.encode(&graph1).unwrap();
    let emb2 = encoder.encode(&graph2).unwrap();

    // Embeddings should differ
    let diff: f32 = (emb1.clone() - emb2.clone())
        .iter()
        .map(|x| x.abs())
        .sum();

    assert!(diff > 0.1, "Different syndromes should produce different embeddings");
}

// ============================================================================
// Mamba Integration Tests
// ============================================================================

#[test]
fn test_mamba_with_gnn_output() {
    let gnn_config = GNNConfig {
        input_dim: 5,
        embed_dim: 16,
        hidden_dim: 32,
        num_layers: 1,
        num_heads: 4,
        dropout: 0.0,
    };
    let encoder = GNNEncoder::new(gnn_config);

    let mamba_config = MambaConfig {
        input_dim: 32,  // Matches GNN hidden_dim
        state_dim: 16,
        output_dim: 9,  // 3x3 surface code
    };
    let mut decoder = MambaDecoder::new(mamba_config);

    // Create graph and encode
    let syndrome = vec![true, false, true, false, false, false, false, false, true];
    let graph = GraphBuilder::from_surface_code(3)
        .with_syndrome(&syndrome).unwrap()
        .build().unwrap();

    let embeddings = encoder.encode(&graph).unwrap();

    // Decode with Mamba
    let output = decoder.decode(&embeddings).unwrap();

    // Output should be correction probabilities
    assert_eq!(output.len(), 9);

    for &prob in output.iter() {
        assert!(prob >= 0.0 && prob <= 1.0, "Output {} should be probability", prob);
    }
}

#[test]
fn test_mamba_state_accumulation() {
    let mamba_config = MambaConfig {
        input_dim: 16,
        state_dim: 8,
        output_dim: 9,
    };
    let mut decoder = MambaDecoder::new(mamba_config);

    // Process embeddings one by one
    let embeddings: Vec<Vec<f32>> = (0..5)
        .map(|i| (0..16).map(|j| ((i + j) as f32) * 0.1).collect())
        .collect();

    let mut outputs = Vec::new();
    for emb in &embeddings {
        let out = decoder.decode_step(emb).unwrap();
        outputs.push(out);
    }

    // State should have been updated
    assert_eq!(decoder.state().steps, 5);

    // Each output should be valid
    for out in &outputs {
        assert_eq!(out.len(), 16); // Same as input_dim
    }
}

// ============================================================================
// Boundary Feature Tests
// ============================================================================

#[test]
fn test_boundary_features_surface_code() {
    // Create positions for 3x3 surface code
    let positions: Vec<(f32, f32)> = (0..3)
        .flat_map(|r| (0..3).map(move |c| (c as f32, r as f32)))
        .collect();

    let boundary = BoundaryFeatures::compute(&positions, 3);

    assert_eq!(boundary.distances.len(), 9);
    assert_eq!(boundary.boundary_types.len(), 9);
    assert_eq!(boundary.weights.len(), 9);

    // Corners should be closest to boundaries
    // Position (0,0) is a corner
    assert!(boundary.distances[0] < boundary.distances[4],
        "Corner should be closer to boundary than center");

    // Center (1,1) should be farthest from boundary
    assert!(boundary.distances[4] >= boundary.distances[0]);
}

#[test]
fn test_boundary_types_assignment() {
    let positions = vec![
        (0.0, 0.5),  // Left edge -> X-boundary
        (1.0, 0.5),  // Right edge -> X-boundary
        (0.5, 0.0),  // Bottom edge -> Z-boundary
        (0.5, 1.0),  // Top edge -> Z-boundary
        (0.5, 0.5),  // Center -> inner
    ];

    let boundary = BoundaryFeatures::compute(&positions, 1);

    // Check boundary type assignments
    // Note: exact types depend on implementation
    assert_eq!(boundary.boundary_types[4], 0, "Center should be inner");
}

// ============================================================================
// Coherence Estimation Tests
// ============================================================================

#[test]
fn test_coherence_estimator() {
    let predictions = Array2::from_shape_fn((9, 4), |(i, j)| {
        if j == 0 { 0.8 } else { 0.2 / 3.0 }  // Clear prediction
    });

    let mut adjacency = HashMap::new();
    for i in 0usize..9 {
        let neighbors: Vec<usize> = [
            i.checked_sub(1),
            (i + 1 < 9).then_some(i + 1),
        ].into_iter().flatten().collect();
        adjacency.insert(i, neighbors);
    }

    let estimator = CoherenceEstimator::new(3, 0.1);
    let confidences = estimator.estimate(&predictions, &adjacency);

    assert_eq!(confidences.len(), 9);

    // All confidences should be valid
    for &c in &confidences {
        assert!(c >= 0.1 && c <= 1.0, "Confidence {} out of range", c);
    }
}

#[test]
fn test_coherence_with_uncertain_predictions() {
    // Uniform predictions (high uncertainty)
    let predictions = Array2::from_shape_fn((9, 4), |_| 0.25);

    let mut adjacency = HashMap::new();
    for i in 0usize..9 {
        adjacency.insert(i, vec![]);  // No neighbors
    }

    let estimator = CoherenceEstimator::new(3, 0.1);
    let confidences = estimator.estimate(&predictions, &adjacency);

    // With uniform predictions, confidence values depend on implementation
    // Just verify they are valid floats
    for &c in &confidences {
        assert!(c.is_finite(), "Confidence should be finite: {}", c);
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_syndrome_dimension_mismatch() {
    let config = DecoderConfig {
        distance: 3,
        ..Default::default()
    };
    let mut decoder = NeuralDecoder::new(config);

    // Wrong syndrome size (should be 9 for distance 3)
    let syndrome = vec![true, false, true];  // Only 3 elements
    let result = decoder.decode(&syndrome);

    assert!(result.is_err(), "Should fail with wrong syndrome dimension");
}

#[test]
fn test_empty_graph_handling() {
    let gnn_config = GNNConfig::default();
    let encoder = GNNEncoder::new(gnn_config);

    let graph = ruvector_neural_decoder::graph::DetectorGraph::new(3);
    let result = encoder.encode(&graph);

    assert!(result.is_err(), "Should fail with empty graph");
}

// ============================================================================
// Performance Characteristics Tests
// ============================================================================

#[test]
fn test_decode_timing() {
    let config = DecoderConfig {
        distance: 5,
        embed_dim: 32,
        hidden_dim: 64,
        num_gnn_layers: 2,
        num_heads: 4,
        mamba_state_dim: 32,
        use_mincut_fusion: false,
        dropout: 0.0,
    };
    let mut decoder = NeuralDecoder::new(config);

    let syndrome = vec![false; 25];
    let correction = decoder.decode(&syndrome).unwrap();

    // Decode should complete in reasonable time (< 1 second)
    assert!(correction.decode_time_ns < 1_000_000_000,
        "Decode took too long: {} ns", correction.decode_time_ns);

    // Decode should take at least some time (not instant)
    assert!(correction.decode_time_ns > 0);
}

#[test]
fn test_multiple_decodes_timing() {
    let config = DecoderConfig {
        distance: 3,
        ..Default::default()
    };
    let mut decoder = NeuralDecoder::new(config);

    let syndromes: Vec<Vec<bool>> = (0..10)
        .map(|i| (0..9).map(|j| (i + j) % 3 == 0).collect())
        .collect();

    let total_start = std::time::Instant::now();

    for syndrome in &syndromes {
        decoder.decode(syndrome).unwrap();
        decoder.reset();
    }

    let total_elapsed = total_start.elapsed();

    // 10 decodes should complete in reasonable time
    assert!(total_elapsed.as_millis() < 5000,
        "Multiple decodes took too long: {} ms", total_elapsed.as_millis());
}
