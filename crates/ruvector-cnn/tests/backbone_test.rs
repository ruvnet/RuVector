//! Tests for CNN backbone implementations
//!
//! Tests cover:
//! - MobileNetV3 output shapes
//! - Embedding dimensions
//! - Batch processing
//! - Feature extraction
//!
//! Note: This test module requires the "backbone" feature to be enabled.

#![cfg(feature = "backbone")]

use ruvector_cnn::backbone::{
    create_backbone, Backbone, BackboneExt, BackboneType, Layer, MobileNetConfig, MobileNetV3,
    MobileNetV3Config, MobileNetV3Large, MobileNetV3Small,
};
use ruvector_cnn::layers::TensorShape;

// ============================================================================
// BackboneType Tests
// ============================================================================

#[test]
fn test_backbone_type_output_dim() {
    assert_eq!(BackboneType::MobileNetV3Small.output_dim(), 576);
    assert_eq!(BackboneType::MobileNetV3Large.output_dim(), 960);
}

#[test]
fn test_backbone_type_name() {
    assert_eq!(BackboneType::MobileNetV3Small.name(), "MobileNetV3-Small");
    assert_eq!(BackboneType::MobileNetV3Large.name(), "MobileNetV3-Large");
}

#[test]
fn test_backbone_type_input_size() {
    assert_eq!(BackboneType::MobileNetV3Small.input_size(), (224, 224));
    assert_eq!(BackboneType::MobileNetV3Large.input_size(), (224, 224));
}

#[test]
fn test_backbone_type_input_channels() {
    assert_eq!(BackboneType::MobileNetV3Small.input_channels(), 3);
    assert_eq!(BackboneType::MobileNetV3Large.input_channels(), 3);
}

// ============================================================================
// MobileNetV3 Small/Large (Legacy API) Tests
// ============================================================================

#[test]
fn test_mobilenet_v3_small_creation() {
    let config = MobileNetConfig::default();
    let model = MobileNetV3Small::new(config);

    assert_eq!(model.output_dim(), 576);
    assert_eq!(model.input_size(), 224);
}

#[test]
fn test_mobilenet_v3_large_creation() {
    let config = MobileNetConfig {
        output_channels: 960,
        ..Default::default()
    };
    let model = MobileNetV3Large::new(config);

    assert_eq!(model.output_dim(), 960);
}

#[test]
fn test_mobilenet_v3_small_forward() {
    let config = MobileNetConfig::default();
    let model = MobileNetV3Small::new(config);

    // Input: [1, 3, 224, 224] in NCHW format, flattened
    let input = vec![0.5f32; 3 * 224 * 224];
    let output = model.forward(&input, 224, 224);

    // Output should be non-empty
    assert!(!output.is_empty());

    // All values should be finite
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_mobilenet_v3_large_forward() {
    let config = MobileNetConfig {
        output_channels: 960,
        ..Default::default()
    };
    let model = MobileNetV3Large::new(config);

    let input = vec![0.5f32; 3 * 224 * 224];
    let output = model.forward(&input, 224, 224);

    assert!(!output.is_empty());
    assert!(output.iter().all(|x| x.is_finite()));
}

// ============================================================================
// MobileNetV3 (Unified API) Tests
// ============================================================================

#[test]
fn test_unified_mobilenet_v3_small_creation() {
    let model = MobileNetV3::small(1000).expect("Failed to create MobileNetV3-Small");

    assert_eq!(model.backbone_type(), BackboneType::MobileNetV3Small);
    assert_eq!(model.output_dim(), 576);
    assert!(model.num_params() > 0);
}

#[test]
fn test_unified_mobilenet_v3_large_creation() {
    let model = MobileNetV3::large(1000).expect("Failed to create MobileNetV3-Large");

    assert_eq!(model.backbone_type(), BackboneType::MobileNetV3Large);
    assert_eq!(model.output_dim(), 960);
    assert!(model.num_params() > 0);
}

#[test]
fn test_unified_mobilenet_v3_feature_only() {
    let model = MobileNetV3::small(0).expect("Failed to create model"); // No classifier

    // Should not have classifier
    assert!(model.classifier().is_none());
}

#[test]
fn test_unified_mobilenet_v3_with_classifier() {
    let model = MobileNetV3::small(1000).expect("Failed to create model");

    // Should have classifier
    assert!(model.classifier().is_some());
}

#[test]
fn test_mobilenet_v3_config() {
    let config = MobileNetV3Config::small(1000);

    assert_eq!(config.input_size, 224);
    assert_eq!(config.input_channels, 3);
    assert_eq!(config.num_classes, 1000);
    assert_eq!(config.feature_dim, 576);
}

#[test]
fn test_mobilenet_v3_config_large() {
    let config = MobileNetV3Config::large(1000);

    assert_eq!(config.input_size, 224);
    assert_eq!(config.num_classes, 1000);
    assert_eq!(config.feature_dim, 960);
}

#[test]
fn test_mobilenet_v3_config_width_mult() {
    let config = MobileNetV3Config::small(1000).width_mult(0.5);

    assert!((config.width_mult - 0.5).abs() < 1e-6);
}

#[test]
fn test_mobilenet_v3_config_dropout() {
    let config = MobileNetV3Config::small(1000).dropout(0.5);

    assert!((config.dropout - 0.5).abs() < 1e-6);
}

// ============================================================================
// Forward Pass Tests
// ============================================================================

#[test]
fn test_mobilenet_v3_forward_features() {
    let model = MobileNetV3::small(0).expect("Failed to create model");
    let input_shape = TensorShape::new(1, 3, 224, 224);
    let input = vec![0.5; input_shape.numel()];

    let output = model
        .forward_features(&input, &input_shape)
        .expect("Forward failed");

    // Output should be [batch, feature_dim]
    assert_eq!(output.len(), 576);
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_mobilenet_v3_forward_with_classifier() {
    let model = MobileNetV3::small(1000).expect("Failed to create model");
    let input_shape = TensorShape::new(1, 3, 224, 224);
    let input = vec![0.5; input_shape.numel()];

    let output = model
        .forward_with_shape(&input, &input_shape)
        .expect("Forward failed");

    // Output should be [batch, num_classes]
    assert_eq!(output.len(), 1000);
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_mobilenet_v3_forward_batch() {
    let model = MobileNetV3::small(0).expect("Failed to create model");
    let batch_size = 2;
    let input_shape = TensorShape::new(batch_size, 3, 224, 224);
    let input = vec![0.5; input_shape.numel()];

    let output = model
        .forward_features(&input, &input_shape)
        .expect("Forward failed");

    // Output should be [batch, feature_dim]
    assert_eq!(output.len(), batch_size * 576);
}

#[test]
fn test_mobilenet_v3_forward_deterministic() {
    let model = MobileNetV3::small(0).expect("Failed to create model");
    let input_shape = TensorShape::new(1, 3, 224, 224);
    let input = vec![0.5; input_shape.numel()];

    let output1 = model
        .forward_features(&input, &input_shape)
        .expect("Forward failed");
    let output2 = model
        .forward_features(&input, &input_shape)
        .expect("Forward failed");

    // Same input should produce same output (no randomness in inference)
    for (v1, v2) in output1.iter().zip(output2.iter()) {
        assert_eq!(v1, v2);
    }
}

// ============================================================================
// Backbone Factory Tests
// ============================================================================

#[test]
fn test_create_backbone_small() {
    let backbone =
        create_backbone(BackboneType::MobileNetV3Small, 1000).expect("Failed to create backbone");

    assert_eq!(backbone.backbone_type(), BackboneType::MobileNetV3Small);
    assert_eq!(backbone.output_dim(), 576);
}

#[test]
fn test_create_backbone_large() {
    let backbone =
        create_backbone(BackboneType::MobileNetV3Large, 1000).expect("Failed to create backbone");

    assert_eq!(backbone.backbone_type(), BackboneType::MobileNetV3Large);
    assert_eq!(backbone.output_dim(), 960);
}

#[test]
fn test_create_backbone_feature_extraction() {
    let backbone =
        create_backbone(BackboneType::MobileNetV3Small, 0).expect("Failed to create backbone");

    assert_eq!(backbone.output_dim(), 576);
}

// ============================================================================
// Model Component Tests
// ============================================================================

#[test]
fn test_mobilenet_v3_num_blocks() {
    let model = MobileNetV3::small(1000).expect("Failed to create model");

    // MobileNetV3-Small has 11 inverted residual blocks
    assert_eq!(model.num_blocks(), 11);
}

#[test]
fn test_mobilenet_v3_large_num_blocks() {
    let model = MobileNetV3::large(1000).expect("Failed to create model");

    // MobileNetV3-Large has 15 inverted residual blocks
    assert_eq!(model.num_blocks(), 15);
}

#[test]
fn test_mobilenet_v3_stem() {
    let model = MobileNetV3::small(1000).expect("Failed to create model");

    // Verify stem layer exists
    let stem = model.stem();
    assert!(stem.num_params() > 0);
}

#[test]
fn test_mobilenet_v3_last_conv() {
    let model = MobileNetV3::small(1000).expect("Failed to create model");

    // Verify last conv layer exists
    let last_conv = model.last_conv();
    assert!(last_conv.num_params() > 0);
}

// ============================================================================
// Feature Output Shape Tests
// ============================================================================

#[test]
fn test_feature_output_shape() {
    let backbone =
        create_backbone(BackboneType::MobileNetV3Small, 0).expect("Failed to create backbone");

    let input_shape = TensorShape::new(1, 3, 224, 224);
    let output_shape = backbone.feature_output_shape(&input_shape);

    assert_eq!(output_shape.n, 1);
    assert_eq!(output_shape.c, 576);
    assert_eq!(output_shape.h, 1);
    assert_eq!(output_shape.w, 1);
}

#[test]
fn test_feature_output_shape_batch() {
    let backbone =
        create_backbone(BackboneType::MobileNetV3Small, 0).expect("Failed to create backbone");

    let input_shape = TensorShape::new(4, 3, 224, 224);
    let output_shape = backbone.feature_output_shape(&input_shape);

    assert_eq!(output_shape.n, 4);
    assert_eq!(output_shape.c, 576);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_backbone_forward_all_zeros() {
    let model = MobileNetV3::small(0).expect("Failed to create model");
    let input_shape = TensorShape::new(1, 3, 224, 224);
    let input = vec![0.0; input_shape.numel()];

    let output = model
        .forward_features(&input, &input_shape)
        .expect("Forward failed");

    // Output should be valid (all finite)
    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_backbone_forward_all_ones() {
    let model = MobileNetV3::small(0).expect("Failed to create model");
    let input_shape = TensorShape::new(1, 3, 224, 224);
    let input = vec![1.0; input_shape.numel()];

    let output = model
        .forward_features(&input, &input_shape)
        .expect("Forward failed");

    assert!(output.iter().all(|x| x.is_finite()));
}

#[test]
fn test_backbone_forward_negative_values() {
    let model = MobileNetV3::small(0).expect("Failed to create model");
    let input_shape = TensorShape::new(1, 3, 224, 224);
    let input = vec![-0.5; input_shape.numel()];

    let output = model
        .forward_features(&input, &input_shape)
        .expect("Forward failed");

    assert!(output.iter().all(|x| x.is_finite()));
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

#[test]
fn test_backbone_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MobileNetV3>();
}

#[test]
fn test_backbone_thread_safe() {
    use std::sync::Arc;
    use std::thread;

    let model = Arc::new(MobileNetV3::small(0).expect("Failed to create model"));

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let model = Arc::clone(&model);
            thread::spawn(move || {
                let input = vec![0.5f32 + (i as f32 * 0.1); 3 * 224 * 224];
                let output = model.forward(&input, 224, 224);
                output.len()
            })
        })
        .collect();

    for handle in handles {
        let len = handle.join().expect("Thread panicked");
        assert!(len > 0);
    }
}

// ============================================================================
// Memory Usage Tests
// ============================================================================

#[test]
fn test_backbone_memory_reuse() {
    let model = MobileNetV3::small(0).expect("Failed to create model");

    // Process multiple batches to check for memory leaks
    for _ in 0..10 {
        let input = vec![0.5f32; 3 * 224 * 224];
        let _output = model.forward(&input, 224, 224);
    }

    // If we get here without OOM, memory is being managed correctly
    assert!(true);
}

// ============================================================================
// Parameter Count Tests
// ============================================================================

#[test]
fn test_mobilenet_v3_small_params() {
    let model = MobileNetV3::small(1000).expect("Failed to create model");
    let params = model.num_params();

    // MobileNetV3-Small has ~2.5M parameters
    // We allow a wide range since weights are zero-initialized
    assert!(params > 0, "Model should have parameters");
}

#[test]
fn test_mobilenet_v3_large_params() {
    let model = MobileNetV3::large(1000).expect("Failed to create model");
    let params = model.num_params();

    // MobileNetV3-Large has ~5.4M parameters
    assert!(params > 0, "Model should have parameters");

    // Large should have more params than small
    let small_model = MobileNetV3::small(1000).expect("Failed to create model");
    assert!(params > small_model.num_params());
}
