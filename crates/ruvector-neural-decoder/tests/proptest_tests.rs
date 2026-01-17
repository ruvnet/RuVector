//! Property-based tests for ruvector-neural-decoder using proptest
//!
//! Tests fundamental mathematical properties that must hold regardless
//! of specific input values.

use proptest::prelude::*;
use ruvector_neural_decoder::{
    graph::{GraphBuilder, DetectorGraph, Node, NodeType},
    gnn::{GNNConfig, GNNEncoder, AttentionLayer},
    mamba::{MambaConfig, MambaDecoder, MambaState},
    DecoderConfig, NeuralDecoder,
};
use ndarray::Array2;

// ============================================================================
// Graph Properties
// ============================================================================

proptest! {
    /// Property: Graph construction should be deterministic for the same syndrome
    #[test]
    fn graph_construction_deterministic(
        distance in 3usize..8,
        syndrome_bits in prop::collection::vec(any::<bool>(), 9..64)
    ) {
        // Ensure syndrome matches distance squared
        let expected_len = distance * distance;
        if syndrome_bits.len() < expected_len {
            return Ok(());
        }

        let syndrome: Vec<bool> = syndrome_bits.into_iter().take(expected_len).collect();

        let graph1 = GraphBuilder::from_surface_code(distance)
            .with_syndrome(&syndrome)
            .unwrap()
            .build()
            .unwrap();

        let graph2 = GraphBuilder::from_surface_code(distance)
            .with_syndrome(&syndrome)
            .unwrap()
            .build()
            .unwrap();

        // Graphs should have identical structure
        prop_assert_eq!(graph1.num_nodes(), graph2.num_nodes());
        prop_assert_eq!(graph1.num_edges(), graph2.num_edges());
        prop_assert_eq!(graph1.num_fired, graph2.num_fired);
    }

    /// Property: Number of fired detectors equals count of true syndrome bits
    #[test]
    fn fired_count_matches_syndrome(
        distance in 3usize..6,
        syndrome_bits in prop::collection::vec(any::<bool>(), 9..36)
    ) {
        let expected_len = distance * distance;
        if syndrome_bits.len() < expected_len {
            return Ok(());
        }

        let syndrome: Vec<bool> = syndrome_bits.into_iter().take(expected_len).collect();
        let expected_fired = syndrome.iter().filter(|&&b| b).count();

        let graph = GraphBuilder::from_surface_code(distance)
            .with_syndrome(&syndrome)
            .unwrap()
            .build()
            .unwrap();

        prop_assert_eq!(graph.num_fired, expected_fired);
    }

    /// Property: Adjacency matrix should be symmetric
    #[test]
    fn adjacency_matrix_symmetric(distance in 3usize..6) {
        let graph = GraphBuilder::from_surface_code(distance)
            .build()
            .unwrap();

        let adj = graph.adjacency_matrix();
        let (rows, cols) = (adj.shape()[0], adj.shape()[1]);

        prop_assert_eq!(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                prop_assert!(
                    (adj[[i, j]] - adj[[j, i]]).abs() < 1e-10,
                    "Adjacency matrix not symmetric at ({}, {}): {} vs {}",
                    i, j, adj[[i, j]], adj[[j, i]]
                );
            }
        }
    }

    /// Property: Grid graph should have predictable edge count
    #[test]
    fn grid_edge_count(distance in 3usize..10) {
        let graph = GraphBuilder::from_surface_code(distance)
            .build()
            .unwrap();

        // For a d x d grid:
        // Horizontal edges: (d-1) * d
        // Vertical edges: d * (d-1)
        // Total: 2 * d * (d-1)
        let expected_edges = 2 * distance * (distance - 1);
        prop_assert_eq!(graph.num_edges(), expected_edges);
    }

    /// Property: All edge weights should be positive
    #[test]
    fn edge_weights_positive(distance in 3usize..6) {
        let graph = GraphBuilder::from_surface_code(distance)
            .with_error_rate(0.01)
            .build()
            .unwrap();

        let weights = graph.edge_weights();
        for &w in weights.iter() {
            prop_assert!(w > 0.0, "Edge weight {} should be positive", w);
        }
    }
}

// ============================================================================
// Attention Layer Properties
// ============================================================================

proptest! {
    /// Property: Attention scores should sum to 1 (softmax normalization)
    #[test]
    fn attention_scores_sum_to_one(
        embed_dim in (8usize..64).prop_filter("divisible by 4", |d| d % 4 == 0),
        num_keys in 1usize..10
    ) {
        let layer = AttentionLayer::new(embed_dim, 4).unwrap();

        let query: Vec<f32> = (0..embed_dim).map(|i| (i as f32) * 0.1).collect();
        let keys: Vec<Vec<f32>> = (0..num_keys)
            .map(|k| (0..embed_dim).map(|i| ((i + k) as f32) * 0.05).collect())
            .collect();

        let scores = layer.attention_scores(&query, &keys);

        if !scores.is_empty() {
            let sum: f32 = scores.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "Attention scores sum to {} instead of 1.0",
                sum
            );
        }
    }

    /// Property: Attention output has same dimension as input
    #[test]
    fn attention_preserves_dimension(
        embed_dim in (8usize..32).prop_filter("divisible by 4", |d| d % 4 == 0),
        num_neighbors in 0usize..5
    ) {
        let layer = AttentionLayer::new(embed_dim, 4).unwrap();

        let query: Vec<f32> = (0..embed_dim).map(|i| (i as f32) * 0.1).collect();
        let neighbors: Vec<Vec<f32>> = (0..num_neighbors)
            .map(|n| (0..embed_dim).map(|i| ((i + n) as f32) * 0.05).collect())
            .collect();

        let output = layer.forward(&query, &neighbors, &neighbors);

        prop_assert_eq!(
            output.len(),
            embed_dim,
            "Output dimension {} should match input dimension {}",
            output.len(),
            embed_dim
        );
    }
}

// ============================================================================
// GNN Properties
// ============================================================================

proptest! {
    /// Property: GNN encoding produces correct output shape
    #[test]
    fn gnn_output_shape(distance in 3usize..5) {
        let config = GNNConfig {
            input_dim: 5,
            embed_dim: 16,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            dropout: 0.0,
        };
        let encoder = GNNEncoder::new(config);

        let graph = GraphBuilder::from_surface_code(distance)
            .build()
            .unwrap();

        let embeddings = encoder.encode(&graph).unwrap();

        let expected_nodes = distance * distance;
        prop_assert_eq!(embeddings.shape()[0], expected_nodes);
        prop_assert_eq!(embeddings.shape()[1], 32); // hidden_dim
    }

    /// Property: Different syndromes should produce different embeddings
    #[test]
    fn gnn_syndrome_sensitivity(
        fired_idx in 0usize..9
    ) {
        let config = GNNConfig {
            input_dim: 5,
            embed_dim: 16,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            dropout: 0.0,
        };
        let encoder = GNNEncoder::new(config);

        // All zeros syndrome
        let syndrome_zero = vec![false; 9];
        let graph_zero = GraphBuilder::from_surface_code(3)
            .with_syndrome(&syndrome_zero)
            .unwrap()
            .build()
            .unwrap();

        // Single fired detector
        let mut syndrome_one = vec![false; 9];
        syndrome_one[fired_idx] = true;
        let graph_one = GraphBuilder::from_surface_code(3)
            .with_syndrome(&syndrome_one)
            .unwrap()
            .build()
            .unwrap();

        let emb_zero = encoder.encode(&graph_zero).unwrap();
        let emb_one = encoder.encode(&graph_one).unwrap();

        // Embeddings should differ
        let diff: f32 = (emb_zero.clone() - emb_one.clone())
            .iter()
            .map(|x| x.abs())
            .sum();

        prop_assert!(
            diff > 0.0,
            "Embeddings should differ when syndrome changes"
        );
    }
}

// ============================================================================
// Mamba State Properties
// ============================================================================

proptest! {
    /// Property: Mamba state updates should be consistent
    #[test]
    fn mamba_state_updates(
        state_dim in 4usize..16,
        num_steps in 1usize..10
    ) {
        let mut state = MambaState::new(state_dim);

        prop_assert_eq!(state.dim, state_dim);
        prop_assert_eq!(state.steps, 0);

        for step in 0..num_steps {
            let new_values: Vec<f32> = (0..state_dim)
                .map(|i| ((i + step) as f32) * 0.1)
                .collect();
            state.update(new_values);
        }

        prop_assert_eq!(state.steps, num_steps);
        prop_assert_eq!(state.get().len(), state_dim);
    }

    /// Property: Mamba reset clears all state
    #[test]
    fn mamba_state_reset(state_dim in 4usize..16) {
        let mut state = MambaState::new(state_dim);

        // Update with some values
        state.update((0..state_dim).map(|i| i as f32).collect());
        state.update((0..state_dim).map(|i| (i * 2) as f32).collect());

        prop_assert_eq!(state.steps, 2);

        state.reset();

        prop_assert_eq!(state.steps, 0);
        for &val in state.get() {
            prop_assert_eq!(val, 0.0);
        }
    }

    /// Property: Mamba decoder output is bounded (sigmoid)
    #[test]
    fn mamba_output_bounded(num_nodes in 1usize..10) {
        let config = MambaConfig {
            input_dim: 16,
            state_dim: 8,
            output_dim: 9,
        };
        let mut decoder = MambaDecoder::new(config);

        let embeddings = Array2::from_shape_fn(
            (num_nodes, 16),
            |(i, j)| ((i + j) as f32) * 0.1 - 0.5
        );

        let output = decoder.decode(&embeddings).unwrap();

        // Output should be probabilities in [0, 1]
        for &val in output.iter() {
            prop_assert!(
                val >= 0.0 && val <= 1.0,
                "Output {} should be in [0, 1]",
                val
            );
        }
    }
}

// ============================================================================
// Decoder Integration Properties
// ============================================================================

proptest! {
    /// Property: Decoder produces valid corrections
    #[test]
    fn decoder_valid_corrections(distance in 3usize..5) {
        let config = DecoderConfig {
            distance,
            embed_dim: 16,
            hidden_dim: 32,
            num_gnn_layers: 1,
            num_heads: 4,
            mamba_state_dim: 16,
            use_mincut_fusion: false,
            dropout: 0.0,
        };
        let mut decoder = NeuralDecoder::new(config);

        let syndrome = vec![false; distance * distance];
        let correction = decoder.decode(&syndrome).unwrap();

        // Confidence should be in [0, 1]
        prop_assert!(
            correction.confidence >= 0.0 && correction.confidence <= 1.0,
            "Confidence {} should be in [0, 1]",
            correction.confidence
        );
    }

    /// Property: Decoder reset clears state
    #[test]
    fn decoder_reset(distance in 3usize..4) {
        let config = DecoderConfig {
            distance,
            embed_dim: 16,
            hidden_dim: 32,
            num_gnn_layers: 1,
            num_heads: 4,
            mamba_state_dim: 16,
            use_mincut_fusion: false,
            dropout: 0.0,
        };
        let mut decoder = NeuralDecoder::new(config);

        // Decode something
        let syndrome = vec![true, false, true, false, false, false, false, false, true];
        decoder.decode(&syndrome).ok();

        // Reset and decode again - should work without error
        decoder.reset();
        let result = decoder.decode(&syndrome);

        prop_assert!(result.is_ok());
    }
}
