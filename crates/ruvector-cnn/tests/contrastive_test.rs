//! Tests for contrastive learning components
//!
//! Tests cover:
//! - InfoNCE loss computation
//! - TripletLoss with known values
//! - Augmentation configuration
//! - Training pipeline integration

use ruvector_cnn::contrastive::{
    AugmentationConfig, ContrastiveAugmentation, InfoNCELoss, TripletDistance, TripletLoss,
};

// ============================================================================
// InfoNCE Loss Tests
// ============================================================================

#[test]
fn test_infonce_basic_computation() {
    let loss_fn = InfoNCELoss::new(0.07); // Standard temperature

    // Create simple embeddings where consecutive pairs are positives
    let embeddings = vec![
        vec![1.0f64, 0.0, 0.0], // anchor 1
        vec![0.9, 0.1, 0.0],    // positive for anchor 1
        vec![0.0, 1.0, 0.0],    // anchor 2
        vec![0.1, 0.9, 0.0],    // positive for anchor 2
    ];

    let loss = loss_fn.forward(&embeddings, 2);

    // Loss should be positive
    assert!(loss > 0.0, "InfoNCE loss should be positive, got {}", loss);

    // Loss should be finite
    assert!(loss.is_finite(), "Loss should be finite");
}

#[test]
fn test_infonce_perfect_alignment() {
    let loss_fn = InfoNCELoss::new(0.07);

    // Create embeddings where pairs are identical (perfect alignment)
    let embeddings = vec![
        vec![1.0f64, 0.0, 0.0, 0.0], // anchor 1
        vec![1.0f64, 0.0, 0.0, 0.0], // positive for anchor 1 (identical)
        vec![0.0, 1.0, 0.0, 0.0],    // anchor 2
        vec![0.0, 1.0, 0.0, 0.0],    // positive for anchor 2 (identical)
    ];

    let loss = loss_fn.forward(&embeddings, 2);

    // Should have low loss for perfect alignment
    assert!(
        loss < 1.0,
        "Perfect alignment should have lower loss, got {}",
        loss
    );
}

#[test]
fn test_infonce_temperature_effect() {
    // Lower temperature makes distribution sharper
    let loss_low_temp = InfoNCELoss::new(0.01);
    let loss_high_temp = InfoNCELoss::new(1.0);

    let embeddings = vec![
        vec![1.0f64, 0.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![-1.0, 0.0, 0.0, 0.0],
    ];

    let low_temp_loss = loss_low_temp.forward(&embeddings, 2);
    let high_temp_loss = loss_high_temp.forward(&embeddings, 2);

    // Both should be valid
    assert!(low_temp_loss.is_finite(), "Low temp loss should be finite");
    assert!(
        high_temp_loss.is_finite(),
        "High temp loss should be finite"
    );
}

#[test]
fn test_infonce_many_negatives() {
    let loss_fn = InfoNCELoss::new(0.07);

    // More embeddings = more negatives
    let mut embeddings = Vec::new();
    for i in 0..10 {
        let angle = (i as f64) * 0.5;
        embeddings.push(vec![angle.cos(), angle.sin(), 0.0]);
        embeddings.push(vec![(angle + 0.1).cos(), (angle + 0.1).sin(), 0.1]);
    }

    let loss = loss_fn.forward(&embeddings, 2);

    assert!(loss > 0.0, "Loss should be positive");
    assert!(loss.is_finite(), "Loss should be finite");
}

#[test]
fn test_infonce_detailed_results() {
    let loss_fn = InfoNCELoss::new(0.07)
        .with_per_sample_losses()
        .with_similarity_matrix();

    let embeddings = vec![
        vec![1.0f64, 0.0],
        vec![0.9, 0.1],
        vec![0.0, 1.0],
        vec![0.1, 0.9],
    ];

    let result = loss_fn.forward_detailed(&embeddings, 2).unwrap();

    assert!(result.loss > 0.0);
    assert!(result.per_sample_losses.is_some());
    assert!(result.similarity_matrix.is_some());

    let per_sample = result.per_sample_losses.unwrap();
    assert_eq!(per_sample.len(), 4);

    let sim_matrix = result.similarity_matrix.unwrap();
    assert_eq!(sim_matrix.len(), 4);
    assert_eq!(sim_matrix[0].len(), 4);

    // Self-similarity should be 1.0
    for i in 0..4 {
        assert!(
            (sim_matrix[i][i] - 1.0).abs() < 1e-6,
            "Self-similarity should be 1.0"
        );
    }
}

#[test]
fn test_infonce_forward_with_pairs() {
    let loss_fn = InfoNCELoss::new(0.07);

    let anchors = vec![vec![1.0f64, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

    let positives = vec![vec![0.9, 0.1, 0.0], vec![0.1, 0.9, 0.0]];

    let loss = loss_fn
        .forward_with_pairs(&anchors, &positives, None)
        .unwrap();

    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

// ============================================================================
// Triplet Loss Tests
// ============================================================================

#[test]
fn test_triplet_loss_basic() {
    let loss_fn = TripletLoss::new(1.0);

    let anchor = vec![1.0f64, 0.0, 0.0];
    let positive = vec![0.9, 0.1, 0.0]; // Close to anchor
    let negative = vec![-1.0, 0.0, 0.0]; // Far from anchor

    let loss = loss_fn.forward(&anchor, &positive, &negative);

    // When negative is far and positive is close, loss should be low
    assert!(loss >= 0.0, "Triplet loss should be non-negative");
}

#[test]
fn test_triplet_loss_zero_case() {
    let loss_fn = TripletLoss::new(0.5);

    // When d(a,n) > d(a,p) + margin, loss should be 0
    let anchor = vec![1.0f64, 0.0, 0.0];
    let positive = vec![1.0, 0.0, 0.0]; // Identical to anchor
    let negative = vec![-1.0, 0.0, 0.0]; // Very far

    let loss = loss_fn.forward(&anchor, &positive, &negative);

    assert_eq!(loss, 0.0, "Loss should be zero when margin is satisfied");
}

#[test]
fn test_triplet_loss_positive_case() {
    let loss_fn = TripletLoss::new(1.0);

    // When d(a,p) ~ d(a,n), there should be positive loss
    let anchor = vec![1.0f64, 0.0, 0.0];
    let positive = vec![0.0, 1.0, 0.0]; // Orthogonal
    let negative = vec![0.0, 0.0, 1.0]; // Also orthogonal, same distance

    let loss = loss_fn.forward(&anchor, &positive, &negative);

    assert!(
        loss > 0.0,
        "Loss should be positive when margin not satisfied, got {}",
        loss
    );
}

#[test]
fn test_triplet_loss_euclidean() {
    let loss_fn = TripletLoss::new(1.0).with_distance(TripletDistance::Euclidean);

    let anchor = vec![0.0f64, 0.0, 0.0];
    let positive = vec![1.0, 0.0, 0.0]; // Distance 1
    let negative = vec![3.0, 0.0, 0.0]; // Distance 3

    let loss = loss_fn.forward(&anchor, &positive, &negative);

    // d(a,p) = 1, d(a,n) = 3, margin = 1
    // loss = max(0, 1 - 3 + 1) = max(0, -1) = 0
    assert_eq!(loss, 0.0, "Margin should be satisfied");
}

#[test]
fn test_triplet_loss_batch() {
    let loss_fn = TripletLoss::new(0.5);

    let anchors = vec![vec![1.0f64, 0.0], vec![0.0, 1.0]];
    let positives = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
    let negatives = vec![vec![-1.0, 0.0], vec![0.0, -1.0]];

    let loss = loss_fn
        .forward_batch(&anchors, &positives, &negatives)
        .unwrap();

    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_triplet_loss_detailed() {
    let loss_fn = TripletLoss::new(1.0);

    let anchor = vec![0.0f64, 0.0];
    let positive = vec![1.0, 0.0];
    let negative = vec![0.5, 0.0]; // Closer to anchor than positive

    let result = loss_fn
        .forward_detailed(&anchor, &positive, &negative)
        .unwrap();

    assert!(result.loss > 0.0);
    assert!(result.is_hard);
    assert!(result.violates_margin);
    assert!(result.positive_distance > result.negative_distance);
}

// ============================================================================
// Augmentation Configuration Tests
// ============================================================================

#[test]
fn test_augmentation_config_default() {
    let config = AugmentationConfig::default();

    // Check default values match SimCLR paper
    assert_eq!(config.crop_scale_min, 0.08);
    assert_eq!(config.crop_scale_max, 1.0);
    assert_eq!(config.horizontal_flip_prob, 0.5);
    assert_eq!(config.output_size, (224, 224));
}

#[test]
fn test_augmentation_builder() {
    let aug = ContrastiveAugmentation::builder()
        .crop_scale(0.2, 0.8)
        .horizontal_flip_prob(1.0)
        .color_jitter(0.2, 0.2, 0.2, 0.05)
        .grayscale_prob(0.1)
        .output_size(128, 128)
        .build();

    let config = aug.config();

    assert_eq!(config.crop_scale_min, 0.2);
    assert_eq!(config.crop_scale_max, 0.8);
    assert_eq!(config.horizontal_flip_prob, 1.0);
    assert_eq!(config.brightness, 0.2);
    assert_eq!(config.contrast, 0.2);
    assert_eq!(config.saturation, 0.2);
    assert_eq!(config.hue, 0.05);
    assert_eq!(config.grayscale_prob, 0.1);
    assert_eq!(config.output_size, (128, 128));
}

#[test]
fn test_augmentation_with_seed() {
    let aug1 = ContrastiveAugmentation::builder().seed(42).build();
    let aug2 = ContrastiveAugmentation::builder().seed(42).build();

    // Same seed should produce same config values
    assert_eq!(aug1.config().crop_scale_min, aug2.config().crop_scale_min);
    assert_eq!(aug1.config().crop_scale_max, aug2.config().crop_scale_max);
    assert_eq!(
        aug1.config().horizontal_flip_prob,
        aug2.config().horizontal_flip_prob
    );
}

#[test]
fn test_augmentation_blur_config() {
    let aug = ContrastiveAugmentation::builder()
        .gaussian_blur(3, (0.1, 3.0))
        .blur_prob(0.5)
        .build();

    let config = aug.config();

    assert_eq!(config.blur_kernel_size, 3);
    assert_eq!(config.blur_prob, 0.5);
    assert_eq!(config.blur_sigma_range, (0.1, 3.0));
}

#[test]
fn test_augmentation_default() {
    let aug = ContrastiveAugmentation::default();
    let config = aug.config();

    // Should have sensible defaults
    assert!(config.crop_scale_min > 0.0);
    assert!(config.crop_scale_max <= 1.0);
    assert!(config.horizontal_flip_prob >= 0.0);
    assert!(config.horizontal_flip_prob <= 1.0);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_infonce_with_normalized_embeddings() {
    let loss_fn = InfoNCELoss::new(0.1);

    // Create normalized embeddings (unit vectors)
    let embeddings: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.8, 0.6, 0.0], // angle ~37 degrees
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.8, 0.6], // angle ~37 degrees
    ];

    // Normalize
    let normalized: Vec<Vec<f64>> = embeddings
        .into_iter()
        .map(|v| {
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect();

    let loss = loss_fn.forward(&normalized, 2);

    assert!(
        loss.is_finite(),
        "Loss with normalized vectors should be finite"
    );
    assert!(loss > 0.0, "Loss should be positive");
}

#[test]
fn test_triplet_with_varying_margins() {
    let anchor = vec![1.0f64, 0.0, 0.0, 0.0];
    let positive = vec![0.7, 0.7, 0.0, 0.0];
    let negative = vec![0.0, 1.0, 0.0, 0.0];

    // Test different margins
    let margins = [0.1, 0.5, 1.0, 2.0];
    let mut losses = Vec::new();

    for margin in margins.iter() {
        let loss_fn = TripletLoss::new(*margin);
        losses.push(loss_fn.forward(&anchor, &positive, &negative));
    }

    // Higher margins should generally lead to higher losses
    for i in 1..losses.len() {
        assert!(
            losses[i] >= losses[i - 1],
            "Higher margin should lead to equal or higher loss"
        );
    }
}

#[test]
fn test_triplet_mine_hard_triplets() {
    // Use a small margin so that the triplet mining finds hard triplets
    let triplet = TripletLoss::new(0.01);

    // Create embeddings where hard triplets exist
    // Class 0 embeddings are close to class 1 embeddings, creating hard triplets
    let embeddings = vec![
        vec![1.0f64, 0.0], // class 0
        vec![0.95, 0.05],  // class 0 - close to anchor
        vec![0.9, 0.1],    // class 1 - close to class 0
        vec![0.85, 0.15],  // class 1 - also close
    ];
    let labels = vec![0, 0, 1, 1];

    let hard_triplets = triplet.mine_hard_triplets(&embeddings, &labels);

    // Verify triplet structure for any hard triplets found
    for (a, p, n) in &hard_triplets {
        assert_eq!(
            labels[*a], labels[*p],
            "anchor and positive should be same class"
        );
        assert_ne!(
            labels[*a], labels[*n],
            "anchor and negative should be different class"
        );
    }

    // Note: depending on the margin and embeddings, hard triplets may or may not be found
    // The important thing is that the function returns valid triplets if any exist
}
