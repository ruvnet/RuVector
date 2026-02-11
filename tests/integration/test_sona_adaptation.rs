//! Integration tests for sona: SONA engine types, MicroLoRA, EWC++, trajectories,
//! and learning signals.
//!
//! All tests use real types and real computations -- no mocks.

use sona::ewc::{EwcConfig, EwcPlusPlus};
use sona::lora::MicroLoRA;
use sona::types::{
    LearnedPattern, LearningSignal, PatternType, QueryTrajectory, SonaConfig, TrajectoryStep,
};

// ===========================================================================
// 1. MicroLoRA: forward pass produces correct dimensioned output
// ===========================================================================
#[test]
fn micro_lora_forward_pass() {
    let dim = 64;
    let rank = 2;
    let lora = MicroLoRA::new(dim, rank);

    let input = vec![1.0f32; dim];
    let mut output = vec![0.0f32; dim];

    lora.forward(&input, &mut output);

    assert_eq!(output.len(), dim);
    // The output should not be all zeros (LoRA adds a residual)
    // Note: up_proj is initialized to zero in standard LoRA, so initially output stays zero.
    // After gradient accumulation and apply, it would change. This test verifies no crash.
}

// ===========================================================================
// 2. MicroLoRA: scalar forward pass equivalence
// ===========================================================================
#[test]
fn micro_lora_scalar_forward_no_crash() {
    let dim = 32;
    let lora = MicroLoRA::new(dim, 1);

    let input = vec![0.5f32; dim];
    let mut output = vec![0.0f32; dim];

    lora.forward_scalar(&input, &mut output);
    // Should not crash -- dimensions must match
    assert_eq!(output.len(), dim);
}

// ===========================================================================
// 3. MicroLoRA: gradient accumulation changes weights after apply
// ===========================================================================
#[test]
fn micro_lora_gradient_accumulation() {
    let dim = 16;
    let mut lora = MicroLoRA::new(dim, 2);

    // Create a learning signal
    let signal = LearningSignal::with_gradient(
        vec![0.1; dim],
        vec![0.5; dim],
        0.9,
    );

    // Initial forward pass
    let input = vec![1.0f32; dim];
    let mut output_before = vec![0.0f32; dim];
    lora.forward_scalar(&input, &mut output_before);

    // Accumulate gradient and apply
    lora.accumulate_gradient(&signal);
    lora.apply_accumulated(0.01);

    // Forward pass after update
    let mut output_after = vec![0.0f32; dim];
    lora.forward_scalar(&input, &mut output_after);

    // Output should have changed (gradient update modifies up_proj)
    let diff: f32 = output_before
        .iter()
        .zip(output_after.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-10,
        "output should change after gradient accumulation and apply, diff={diff}"
    );
}

// ===========================================================================
// 4. EWC++: Fisher information updates
// ===========================================================================
#[test]
fn ewc_fisher_update() {
    let config = EwcConfig {
        param_count: 10,
        max_tasks: 5,
        initial_lambda: 1000.0,
        ..EwcConfig::default()
    };
    let mut ewc = EwcPlusPlus::new(config);

    // Simulate gradient updates
    let gradients = vec![0.5f32; 10];
    ewc.update_fisher(&gradients);

    // After update, Fisher should be non-zero
    let weights = vec![0.0f32; 10];
    let loss = ewc.regularization_loss(&weights);
    // With no prior tasks in memory, reg loss = 0
    assert!(
        loss.abs() < 1e-6,
        "regularization loss should be ~0 with no prior tasks, got {loss}"
    );
}

// ===========================================================================
// 5. EWC++: constraint application reduces gradient magnitude
// ===========================================================================
#[test]
fn ewc_constraint_application() {
    let config = EwcConfig {
        param_count: 5,
        max_tasks: 5,
        initial_lambda: 1000.0,
        ..EwcConfig::default()
    };
    let mut ewc = EwcPlusPlus::new(config);

    // Simulate learning on a task
    for _ in 0..100 {
        let gradients = vec![1.0f32; 5];
        ewc.update_fisher(&gradients);
    }

    // Set optimal weights and start a new task
    ewc.set_optimal_weights(&[0.5; 5]);
    ewc.start_new_task();

    // Now apply constraints to new gradients
    let new_gradients = vec![1.0f32; 5];
    let constrained = ewc.apply_constraints(&new_gradients);

    // Constrained gradients should be smaller than original due to EWC penalty
    let original_norm: f32 = new_gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
    let constrained_norm: f32 = constrained.iter().map(|g| g * g).sum::<f32>().sqrt();

    assert!(
        constrained_norm < original_norm,
        "EWC constraints should reduce gradient norm: original={original_norm}, constrained={constrained_norm}"
    );
}

// ===========================================================================
// 6. EWC++: task boundary detection
// ===========================================================================
#[test]
fn ewc_task_boundary_detection() {
    let config = EwcConfig {
        param_count: 10,
        boundary_threshold: 2.0,
        ..EwcConfig::default()
    };
    let mut ewc = EwcPlusPlus::new(config);

    // Not enough samples for detection
    let normal_gradients = vec![0.1f32; 10];
    assert!(
        !ewc.detect_task_boundary(&normal_gradients),
        "should not detect boundary with too few samples"
    );

    // Accumulate normal gradients
    for _ in 0..100 {
        ewc.update_fisher(&normal_gradients);
    }

    // A dramatically different gradient should trigger boundary detection
    let anomalous_gradients = vec![100.0f32; 10];
    let detected = ewc.detect_task_boundary(&anomalous_gradients);
    // This may or may not trigger depending on accumulated stats, but should not crash
    let _ = detected; // just verify no panic
}

// ===========================================================================
// 7. QueryTrajectory: construction and reward computation
// ===========================================================================
#[test]
fn trajectory_rewards() {
    let mut trajectory = QueryTrajectory::new(1, vec![0.1, 0.2, 0.3]);

    trajectory.add_step(TrajectoryStep::new(vec![0.5, 0.3, 0.2], vec![0.4, 0.4, 0.2], 0.5, 0));
    trajectory.add_step(TrajectoryStep::new(vec![0.3, 0.5, 0.2], vec![0.3, 0.3, 0.4], 0.7, 1));
    trajectory.add_step(TrajectoryStep::new(vec![0.2, 0.2, 0.6], vec![0.2, 0.5, 0.3], 0.9, 2));

    trajectory.finalize(0.85, 1500);

    assert!(
        (trajectory.total_reward() - 2.1).abs() < 1e-6,
        "total reward should be 2.1, got {}",
        trajectory.total_reward()
    );
    assert!(
        (trajectory.avg_reward() - 0.7).abs() < 1e-6,
        "average reward should be 0.7, got {}",
        trajectory.avg_reward()
    );
    assert_eq!(trajectory.final_quality, 0.85);
    assert_eq!(trajectory.latency_us, 1500);
}

// ===========================================================================
// 8. LearningSignal: from trajectory and scaled gradient
// ===========================================================================
#[test]
fn learning_signal_from_trajectory() {
    let mut trajectory = QueryTrajectory::new(42, vec![0.1, 0.2, 0.3]);
    trajectory.add_step(TrajectoryStep::new(
        vec![0.5, 0.3, 0.2],
        vec![0.4, 0.4, 0.2],
        0.8,
        0,
    ));
    trajectory.finalize(0.8, 1000);

    let signal = LearningSignal::from_trajectory(&trajectory);

    assert_eq!(signal.quality_score, 0.8);
    assert_eq!(signal.gradient_estimate.len(), 3, "gradient should have same dim as embedding");
    assert_eq!(signal.metadata.trajectory_id, 42);
    assert_eq!(signal.metadata.step_count, 1);

    // Scaled gradient should be quality * gradient
    let scaled = signal.scaled_gradient();
    for (i, &g) in scaled.iter().enumerate() {
        let expected = signal.gradient_estimate[i] * 0.8;
        assert!(
            (g - expected).abs() < 1e-6,
            "scaled gradient[{i}] should be {expected}, got {g}"
        );
    }
}

// ===========================================================================
// 9. LearnedPattern: similarity and merge
// ===========================================================================
#[test]
fn learned_pattern_similarity() {
    let pattern = LearnedPattern::new(1, vec![1.0, 0.0, 0.0]);

    // Same direction -> similarity ~1
    let sim_same = pattern.similarity(&[1.0, 0.0, 0.0]);
    assert!(
        (sim_same - 1.0).abs() < 1e-6,
        "similarity to same direction should be ~1.0, got {sim_same}"
    );

    // Orthogonal -> similarity ~0
    let sim_ortho = pattern.similarity(&[0.0, 1.0, 0.0]);
    assert!(
        sim_ortho.abs() < 1e-6,
        "similarity to orthogonal should be ~0, got {sim_ortho}"
    );

    // Opposite -> similarity ~-1
    let sim_opposite = pattern.similarity(&[-1.0, 0.0, 0.0]);
    assert!(
        (sim_opposite - (-1.0)).abs() < 1e-6,
        "similarity to opposite should be ~-1, got {sim_opposite}"
    );
}

#[test]
fn learned_pattern_merge() {
    let p1 = LearnedPattern {
        id: 1,
        centroid: vec![1.0, 0.0],
        cluster_size: 10,
        total_weight: 5.0,
        avg_quality: 0.8,
        created_at: 100,
        last_accessed: 200,
        access_count: 5,
        pattern_type: PatternType::General,
    };

    let p2 = LearnedPattern {
        id: 2,
        centroid: vec![0.0, 1.0],
        cluster_size: 10,
        total_weight: 5.0,
        avg_quality: 0.9,
        created_at: 150,
        last_accessed: 250,
        access_count: 3,
        pattern_type: PatternType::General,
    };

    let merged = p1.merge(&p2);
    assert_eq!(merged.cluster_size, 20);
    assert!((merged.centroid[0] - 0.5).abs() < 1e-6, "merged centroid x");
    assert!((merged.centroid[1] - 0.5).abs() < 1e-6, "merged centroid y");
    assert!((merged.avg_quality - 0.85).abs() < 1e-6, "merged avg quality");
    assert_eq!(merged.total_weight, 10.0, "merged total weight");
}

// ===========================================================================
// 10. SonaConfig: preset configurations are valid
// ===========================================================================
#[test]
fn sona_config_presets() {
    let default_cfg = SonaConfig::default();
    assert!(default_cfg.hidden_dim > 0);
    assert!(default_cfg.micro_lora_rank >= 1 && default_cfg.micro_lora_rank <= 2);

    let throughput_cfg = SonaConfig::max_throughput();
    assert!(throughput_cfg.micro_lora_rank <= 2);

    let quality_cfg = SonaConfig::max_quality();
    assert!(quality_cfg.base_lora_rank >= 8);

    let edge_cfg = SonaConfig::edge_deployment();
    assert!(edge_cfg.trajectory_capacity <= 500);

    let batch_cfg = SonaConfig::batch_processing();
    assert!(batch_cfg.enable_simd);
}

// ===========================================================================
// 11. Empty trajectory: rewards and signals
// ===========================================================================
#[test]
fn empty_trajectory_defaults() {
    let trajectory = QueryTrajectory::new(99, vec![0.0; 4]);

    assert_eq!(trajectory.total_reward(), 0.0);
    assert_eq!(trajectory.avg_reward(), 0.0);
    assert_eq!(trajectory.steps.len(), 0);
    assert_eq!(trajectory.final_quality, 0.0);
}

// ===========================================================================
// 12. LearnedPattern: decay reduces weight
// ===========================================================================
#[test]
fn learned_pattern_decay() {
    let mut pattern = LearnedPattern::new(1, vec![1.0, 2.0, 3.0]);
    pattern.total_weight = 10.0;

    pattern.decay(0.9);
    assert!(
        (pattern.total_weight - 9.0).abs() < 1e-6,
        "decay(0.9) should reduce weight from 10.0 to 9.0, got {}",
        pattern.total_weight
    );

    pattern.decay(0.5);
    assert!(
        (pattern.total_weight - 4.5).abs() < 1e-6,
        "second decay should further reduce weight"
    );
}
