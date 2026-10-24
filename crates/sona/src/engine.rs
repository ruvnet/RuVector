//! SONA Engine - Main interface for self-optimizing neural architecture

use crate::loops::coordinator::{CoordinatorStats, LoopCoordinator};
use crate::trajectory::TrajectoryBuilder;
use crate::types::{QueryTrajectory, SonaConfig};

/// Main SONA engine integrating all components
pub struct SonaEngine {
    /// Loop coordinator
    coordinator: LoopCoordinator,
    /// Configuration
    config: SonaConfig,
    /// Whether engine is enabled
    enabled: bool,
}

impl std::fmt::Debug for SonaEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SonaEngine")
            .field("config", &self.config)
            .field("enabled", &self.enabled)
            .finish_non_exhaustive()
    }
}

impl SonaEngine {
    /// Create new SONA engine with default config
    pub fn new(hidden_dim: usize) -> Self {
        Self::with_config(SonaConfig {
            hidden_dim,
            embedding_dim: hidden_dim,
            ..Default::default()
        })
    }

    /// Create with custom config
    pub fn with_config(config: SonaConfig) -> Self {
        Self {
            coordinator: LoopCoordinator::with_config(config.clone()),
            config,
            enabled: true,
        }
    }

    /// Start trajectory recording for a query
    pub fn begin_trajectory(&self, query_embedding: Vec<f32>) -> TrajectoryBuilder {
        let id = self.coordinator.next_trajectory_id();
        TrajectoryBuilder::new(id, query_embedding)
    }

    /// Complete trajectory and submit for learning
    pub fn end_trajectory(&self, builder: TrajectoryBuilder, quality: f32) {
        if !self.enabled {
            return;
        }

        let trajectory = builder.build(quality);
        self.coordinator.on_inference(trajectory);
    }

    /// Submit pre-built trajectory
    pub fn submit_trajectory(&self, trajectory: QueryTrajectory) {
        if self.enabled {
            self.coordinator.on_inference(trajectory);
        }
    }

    /// Apply micro-LoRA to hidden states
    pub fn apply_micro_lora(&self, input: &[f32], output: &mut [f32]) {
        if !self.enabled {
            return;
        }

        if let Some(lora) = self.coordinator.micro_lora().try_read() {
            lora.forward(input, output);
        }
    }

    /// Apply base-LoRA to layer output
    pub fn apply_base_lora(&self, layer_idx: usize, input: &[f32], output: &mut [f32]) {
        if !self.enabled {
            return;
        }

        if let Some(lora) = self.coordinator.base_lora().try_read() {
            lora.forward_layer(layer_idx, input, output);
        }
    }

    /// Run background learning cycle if due
    pub fn tick(&self) -> Option<String> {
        if !self.enabled {
            return None;
        }

        if let Some(result) = self.coordinator.maybe_run_background() {
            Some(format!(
                "Background cycle: {} trajectories -> {} patterns in {:?}",
                result.trajectories_processed, result.patterns_extracted, result.elapsed
            ))
        } else {
            None
        }
    }

    /// Force background learning cycle
    pub fn force_learn(&self) -> String {
        let result = self.coordinator.force_background();
        format!(
            "Forced learning: {} trajectories -> {} patterns, status: {}",
            result.trajectories_processed, result.patterns_extracted, result.status
        )
    }

    /// Flush instant loop updates
    pub fn flush(&self) {
        self.coordinator.flush_instant();
    }

    /// Mark a task boundary for EWC++ (for example, a change of season or
    /// workload) so that later learning does not overwrite the current task.
    pub fn start_new_task(&self) {
        self.coordinator.start_new_task();
    }

    /// Find similar patterns to query
    pub fn find_patterns(&self, query_embedding: &[f32], k: usize) -> Vec<crate::LearnedPattern> {
        self.coordinator
            .reasoning_bank()
            .read()
            .find_similar(query_embedding, k)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Get engine statistics
    pub fn stats(&self) -> CoordinatorStats {
        self.coordinator.stats()
    }

    /// Get coordinator reference for state serialization (fixes #274)
    pub fn coordinator(&self) -> &LoopCoordinator {
        &self.coordinator
    }

    /// Enable/disable engine
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get config
    pub fn config(&self) -> &SonaConfig {
        &self.config
    }

    /// Get all learned patterns from the reasoning bank
    #[cfg(feature = "serde-support")]
    pub fn get_all_patterns(&self) -> Vec<crate::LearnedPattern> {
        self.coordinator.reasoning_bank().read().get_all_patterns()
    }

    /// Export LoRA state for serialization
    #[cfg(feature = "serde-support")]
    pub fn export_lora_state(&self) -> crate::export::safetensors::LoRAState {
        use crate::export::safetensors::{LoRALayerState, LoRAState};

        let mut state = LoRAState::default();

        // Export MicroLoRA (single layer)
        if let Some(lora) = self.coordinator.micro_lora().try_read() {
            let (down, up) = lora.get_weights();
            state.micro_lora_layers.push(LoRALayerState {
                lora_a: down.clone(),
                lora_b: up.clone(),
                rank: self.config.micro_lora_rank,
                input_dim: self.config.hidden_dim,
                output_dim: self.config.hidden_dim,
            });
        }

        // Export BaseLoRA (multi-layer)
        if let Some(lora) = self.coordinator.base_lora().try_read() {
            for idx in 0..lora.num_layers() {
                if let Some((down, up)) = lora.get_layer_weights(idx) {
                    state.base_lora_layers.push(LoRALayerState {
                        lora_a: down.clone(),
                        lora_b: up.clone(),
                        rank: lora.rank,
                        input_dim: lora.hidden_dim,
                        output_dim: lora.hidden_dim,
                    });
                }
            }
        }

        state
    }

    /// Get quality trajectories for preference learning export
    #[cfg(feature = "serde-support")]
    pub fn get_quality_trajectories(&self) -> Vec<crate::export::dataset::QualityTrajectory> {
        use crate::export::dataset::QualityTrajectory;

        // Get buffered trajectories from the instant loop via coordinator
        let trajectories = self.coordinator.reasoning_bank().read().get_all_patterns();

        trajectories
            .iter()
            .map(|p| {
                QualityTrajectory {
                    query_embedding: p.centroid.clone(),
                    response_embedding: p.centroid.clone(), // Use centroid as proxy
                    route: p.pattern_type.to_string(),
                    quality: p.avg_quality,
                    context_ids: vec![],
                }
            })
            .collect()
    }

    /// Get routing decisions for distillation export
    #[cfg(feature = "serde-support")]
    pub fn get_routing_decisions(&self) -> Vec<crate::export::dataset::RoutingDecision> {
        use crate::export::dataset::RoutingDecision;

        let patterns = self.coordinator.reasoning_bank().read().get_all_patterns();

        patterns
            .iter()
            .map(|p| {
                RoutingDecision {
                    query_embedding: p.centroid.clone(),
                    routing_logits: vec![p.avg_quality], // Simplified
                    selected_route: p.pattern_type.to_string(),
                    confidence: p.avg_quality,
                    quality: p.avg_quality,
                }
            })
            .collect()
    }
}

/// Builder for SonaEngine
pub struct SonaEngineBuilder {
    config: SonaConfig,
}

impl SonaEngineBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            config: SonaConfig::default(),
        }
    }

    /// Set hidden dimension
    pub fn hidden_dim(mut self, dim: usize) -> Self {
        self.config.hidden_dim = dim;
        self.config.embedding_dim = dim;
        self
    }

    /// Set micro-LoRA rank
    pub fn micro_lora_rank(mut self, rank: usize) -> Self {
        self.config.micro_lora_rank = rank.clamp(1, 2);
        self
    }

    /// Set base-LoRA rank
    pub fn base_lora_rank(mut self, rank: usize) -> Self {
        self.config.base_lora_rank = rank;
        self
    }

    /// Set micro-LoRA learning rate
    pub fn micro_lr(mut self, lr: f32) -> Self {
        self.config.micro_lora_lr = lr;
        self
    }

    /// Set base-LoRA learning rate
    pub fn base_lr(mut self, lr: f32) -> Self {
        self.config.base_lora_lr = lr;
        self
    }

    /// Set EWC lambda
    pub fn ewc_lambda(mut self, lambda: f32) -> Self {
        self.config.ewc_lambda = lambda;
        self
    }

    /// Set pattern clusters
    pub fn pattern_clusters(mut self, k: usize) -> Self {
        self.config.pattern_clusters = k;
        self
    }

    /// Set trajectory buffer capacity
    pub fn buffer_capacity(mut self, capacity: usize) -> Self {
        self.config.trajectory_capacity = capacity;
        self
    }

    /// Set quality threshold
    pub fn quality_threshold(mut self, threshold: f32) -> Self {
        self.config.quality_threshold = threshold;
        self
    }

    /// Build the engine
    pub fn build(self) -> SonaEngine {
        SonaEngine::with_config(self.config)
    }
}

impl Default for SonaEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TrajectoryStep;

    /// Deterministic pseudo-random unit vector.
    fn unit(dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let mut v: Vec<f32> = (0..dim)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        normalize(&mut v);
        v
    }

    fn normalize(v: &mut [f32]) {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
    }

    /// `centre` plus `noise_weight` of a random direction, normalised.
    fn near(centre: &[f32], seed: u64, noise_weight: f32) -> Vec<f32> {
        let noise = unit(centre.len(), seed);
        let mut v: Vec<f32> = centre
            .iter()
            .zip(&noise)
            .map(|(c, n)| c + noise_weight * n)
            .collect();
        normalize(&mut v);
        v
    }

    /// Mean cosine between the adapted inputs and `target`.
    fn alignment(engine: &SonaEngine, inputs: &[Vec<f32>], target: &[f32]) -> f32 {
        let total: f32 = inputs
            .iter()
            .map(|x| {
                let mut out = x.clone();
                engine.apply_micro_lora(x, &mut out);
                let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
                out.iter().zip(target).map(|(a, b)| a * b).sum::<f32>() / norm
            })
            .sum();
        total / inputs.len() as f32
    }

    /// Train task A, mark a task boundary, train task B on overlapping
    /// inputs with a different target, and return task A's held-out
    /// alignment after A and after B.
    fn task_a_then_b(ewc_lambda: f32) -> (f32, f32) {
        let dim = 64;
        let engine = SonaEngine::with_config(SonaConfig {
            hidden_dim: dim,
            embedding_dim: dim,
            micro_lora_lr: 0.5,
            ewc_lambda,
            ..Default::default()
        });
        let centre_a = unit(dim, 31);
        let centre_b = near(&centre_a, 32, 1.2); // cosine about 0.64 with task A
        let (target_a, target_b) = (unit(dim, 33), unit(dim, 34));
        let train = |centre: &[f32], target: &[f32], seed: u64| {
            for k in 0..300 {
                let x = near(centre, seed + k % 100, 0.4);
                let mut builder = engine.begin_trajectory(x);
                builder.add_step(target.to_vec(), vec![], 1.0);
                engine.end_trajectory(builder, 1.0);
                if k % 5 == 4 {
                    engine.flush();
                }
            }
        };
        let held_out_a: Vec<Vec<f32>> = (0..30).map(|k| near(&centre_a, 5_000 + k, 0.4)).collect();

        train(&centre_a, &target_a, 1_000);
        let after_a = alignment(&engine, &held_out_a, &target_a);
        engine.start_new_task();
        train(&centre_b, &target_b, 2_000);
        let after_b = alignment(&engine, &held_out_a, &target_a);
        (after_a, after_b)
    }

    #[test]
    fn test_micro_lora_learns_from_step_activations() {
        let (after_a, _) = task_a_then_b(0.0);
        // Unadapted inputs are close to orthogonal to the target.
        assert!(after_a > 0.5, "task A alignment {}", after_a);
    }

    #[test]
    fn test_ewc_reduces_forgetting() {
        let (a_plain, b_plain) = task_a_then_b(0.0);
        let (a_ewc, b_ewc) = task_a_then_b(2000.0);
        let forgot_plain = a_plain - b_plain;
        let forgot_ewc = a_ewc - b_ewc;
        assert!(
            forgot_plain > 0.4,
            "expected forgetting without EWC++, got {}",
            forgot_plain
        );
        assert!(
            forgot_ewc < forgot_plain * 0.5,
            "EWC++ forgot {} vs {} without",
            forgot_ewc,
            forgot_plain
        );
    }

    #[test]
    fn test_engine_creation() {
        let engine = SonaEngine::new(256);
        assert!(engine.is_enabled());
    }

    #[test]
    fn test_builder() {
        let engine = SonaEngineBuilder::new()
            .hidden_dim(512)
            .micro_lora_rank(2)
            .base_lora_rank(16)
            .micro_lr(0.002)
            .ewc_lambda(500.0)
            .build();

        assert_eq!(engine.config().hidden_dim, 512);
        assert_eq!(engine.config().micro_lora_rank, 2);
    }

    #[test]
    fn test_trajectory_workflow() {
        let engine = SonaEngine::new(64);

        // Begin trajectory
        let mut builder = engine.begin_trajectory(vec![0.1; 64]);
        builder.add_step(vec![0.5; 64], vec![], 0.8);
        builder.add_step(vec![0.6; 64], vec![], 0.9);

        // End trajectory
        engine.end_trajectory(builder, 0.85);

        let stats = engine.stats();
        assert_eq!(stats.trajectories_buffered, 1);
    }

    #[test]
    fn test_micro_lora_application() {
        let engine = SonaEngine::new(64);

        // Train a bit first
        for i in 0..10 {
            let mut builder = engine.begin_trajectory(vec![0.1; 64]);
            builder.add_step(vec![0.5; 64], vec![], 0.8);
            engine.end_trajectory(builder, 0.8);
        }
        engine.flush();

        // Apply LoRA
        let input = vec![1.0; 64];
        let mut output = vec![0.0; 64];
        engine.apply_micro_lora(&input, &mut output);

        // Output may or may not be modified depending on accumulated gradients
    }

    #[test]
    fn test_force_learn() {
        let engine = SonaEngine::new(256);

        for _i in 0..150 {
            let mut builder = engine.begin_trajectory(vec![0.1; 256]);
            builder.add_step(vec![0.5; 256], vec![], 0.8);
            engine.end_trajectory(builder, 0.8);
        }

        let result = engine.force_learn();
        assert!(result.contains("150 trajectories"));
    }

    #[test]
    fn test_force_learn_with_few_trajectories() {
        // Test that forceLearn works even with fewer than min_trajectories (100)
        let engine = SonaEngine::new(64);

        // Only record 10 trajectories (below the 100 minimum)
        for _i in 0..10 {
            let mut builder = engine.begin_trajectory(vec![0.1; 64]);
            builder.add_step(vec![0.5; 64], vec![], 0.8);
            engine.end_trajectory(builder, 0.8);
        }

        let result = engine.force_learn();
        // Should process 10 trajectories (not "insufficient trajectories")
        assert!(
            result.contains("10 trajectories"),
            "Expected '10 trajectories' but got: {}",
            result
        );
    }

    #[test]
    fn test_single_feedback_changes_lora_output() {
        // Regression test for #519: a series of single-step, constant-reward
        // feedback trajectories (what wasm learn_from_feedback synthesizes)
        // must produce an actual micro-LoRA weight update, i.e. apply_micro_lora
        // output must change.
        let engine = SonaEngine::new(64);

        let input = vec![1.0f32; 64];
        let mut before = vec![0.0f32; 64];
        engine.apply_micro_lora(&input, &mut before);

        // Mirror WasmSonaEngine::learn_from_feedback
        for _ in 0..5 {
            let embedding = vec![1.0 / (64f32).sqrt(); 64];
            let mut builder = engine.begin_trajectory(embedding.clone());
            builder.add_step(embedding, vec![], 0.9);
            let trajectory = builder.build_with_latency(0.9, 50_000);
            engine.submit_trajectory(trajectory);
            engine.flush();
        }

        let mut after = vec![0.0f32; 64];
        engine.apply_micro_lora(&input, &mut after);

        let delta: f32 = before
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            delta > 0.0,
            "apply_micro_lora output unchanged after feedback (delta={})",
            delta
        );
    }

    #[test]
    fn test_disabled_engine() {
        let mut engine = SonaEngine::new(64);
        engine.set_enabled(false);

        let builder = engine.begin_trajectory(vec![0.1; 64]);
        engine.end_trajectory(builder, 0.8);

        // Should not record when disabled
        let stats = engine.stats();
        assert_eq!(stats.trajectories_buffered, 0);
    }
}
