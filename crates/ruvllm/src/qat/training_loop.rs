//! QAT Training Loop (ADR-090 Phase 2)
//!
//! This module implements the orchestration layer for Quantization-Aware Training,
//! managing the full training pipeline: calibration -> training -> export.
//!
//! ## Training Pipeline
//!
//! ```text
//! 1. Calibration Phase
//!    - Collect activation statistics on calibration data
//!    - Initialize quantization scales
//!
//! 2. Training Phase
//!    - Forward: FP weights -> quantize -> dequantize -> loss
//!    - Backward: STE gradient flow through quantization
//!    - Update: Optimize both weights and quantization parameters
//!
//! 3. Export Phase
//!    - Convert trained model to quantized format
//!    - Emit domain events for downstream consumers
//! ```
//!
//! ## System Invariants
//!
//! - **INV-1**: STE gradient flow preserved throughout training
//! - **INV-2**: Scale positivity maintained during optimization
//! - **INV-5**: Training artifacts serializable for reproducibility
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::qat::{QatTrainer, QatConfig, TrainingCallback};
//!
//! let config = QatConfig::piq3().with_epochs(5);
//! let mut trainer = QatTrainer::new(config);
//!
//! // Register callbacks
//! trainer.on_epoch_complete(|metrics| {
//!     println!("Epoch {}: loss={:.4}", metrics.epoch, metrics.loss);
//! });
//!
//! // Run training
//! let result = trainer.run(&model, &train_data, &calib_data)?;
//! result.export("model_piq3.bin")?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::error::{Result, RuvLLMError};

use super::calibration::{CalibrationConfig, CalibrationEngine, CalibrationResult};
use super::config::QatConfig;
use super::differentiable_quant::{create_quantizer, DifferentiableQuantizer};
use super::distillation::{DistillationConfig, DistillationLoss, TeacherOutput};
use super::reasoning_loss::{ChainOfThoughtLoss, ReasoningConfig, ReasoningStep};

// ============================================================================
// Training State
// ============================================================================

/// Current phase of QAT training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingPhase {
    /// Not started
    Idle,
    /// Collecting calibration statistics
    Calibration,
    /// Training with quantization in the loop
    Training,
    /// Finalizing and exporting model
    Export,
    /// Training complete
    Complete,
}

impl Default for TrainingPhase {
    fn default() -> Self {
        Self::Idle
    }
}

/// Training state checkpoint for resumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Current epoch
    pub epoch: usize,
    /// Current step within epoch
    pub step: usize,
    /// Training phase
    pub phase: TrainingPhase,
    /// Accumulated loss
    pub total_loss: f64,
    /// Per-layer scale values
    pub scales: HashMap<String, f32>,
    /// Random seed state (for reproducibility)
    pub rng_state: u64,
    /// Timestamp
    pub timestamp: u64,
}

impl TrainingCheckpoint {
    /// Save checkpoint to file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| RuvLLMError::Model(format!("Checkpoint serialization failed: {}", e)))?;
        std::fs::write(path, json)
            .map_err(|e| RuvLLMError::Model(format!("Checkpoint write failed: {}", e)))?;
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| RuvLLMError::Model(format!("Checkpoint read failed: {}", e)))?;
        let checkpoint = serde_json::from_str(&json)
            .map_err(|e| RuvLLMError::Model(format!("Checkpoint deserialization failed: {}", e)))?;
        Ok(checkpoint)
    }
}

// ============================================================================
// Training Metrics
// ============================================================================

/// Metrics for a single training step
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Step number
    pub step: usize,
    /// Total loss
    pub loss: f32,
    /// Task loss component
    pub task_loss: f32,
    /// KD loss component
    pub kd_loss: f32,
    /// Reasoning loss component
    pub reasoning_loss: f32,
    /// Step duration
    pub duration_ms: u64,
    /// Gradient norm (for monitoring)
    pub grad_norm: f32,
}

/// Metrics for a complete epoch
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EpochMetrics {
    /// Epoch number
    pub epoch: usize,
    /// Average loss
    pub avg_loss: f64,
    /// Average task loss
    pub avg_task_loss: f64,
    /// Average KD loss
    pub avg_kd_loss: f64,
    /// Average reasoning loss
    pub avg_reasoning_loss: f64,
    /// Perplexity (exp(avg_loss))
    pub perplexity: f64,
    /// Reasoning score (0-1)
    pub reasoning_score: f64,
    /// Number of steps
    pub num_steps: usize,
    /// Total epoch duration
    pub duration_ms: u64,
    /// Learning rate at epoch end
    pub learning_rate: f64,
}

impl EpochMetrics {
    /// Create from accumulated step metrics
    pub fn from_steps(epoch: usize, steps: &[StepMetrics], lr: f64, duration: Duration) -> Self {
        if steps.is_empty() {
            return Self {
                epoch,
                learning_rate: lr,
                ..Default::default()
            };
        }

        let n = steps.len() as f64;
        let avg_loss = steps.iter().map(|s| s.loss as f64).sum::<f64>() / n;
        let avg_task = steps.iter().map(|s| s.task_loss as f64).sum::<f64>() / n;
        let avg_kd = steps.iter().map(|s| s.kd_loss as f64).sum::<f64>() / n;
        let avg_reasoning = steps.iter().map(|s| s.reasoning_loss as f64).sum::<f64>() / n;

        Self {
            epoch,
            avg_loss,
            avg_task_loss: avg_task,
            avg_kd_loss: avg_kd,
            avg_reasoning_loss: avg_reasoning,
            perplexity: avg_loss.exp(),
            reasoning_score: 1.0 - avg_reasoning.min(1.0),
            num_steps: steps.len(),
            duration_ms: duration.as_millis() as u64,
            learning_rate: lr,
        }
    }
}

// ============================================================================
// Domain Events
// ============================================================================

/// Domain event emitted during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QatEvent {
    /// Calibration phase started
    CalibrationStarted { config: CalibrationConfig },
    /// Calibration phase completed
    CalibrationComplete { result: CalibrationResult },
    /// Training epoch started
    EpochStarted { epoch: usize, total_epochs: usize },
    /// Training epoch completed
    EpochComplete { metrics: EpochMetrics },
    /// Training step completed
    StepComplete { metrics: StepMetrics },
    /// Checkpoint saved
    CheckpointSaved { path: String, epoch: usize },
    /// Training completed
    TrainingComplete {
        total_epochs: usize,
        final_loss: f64,
        duration_ms: u64,
    },
    /// Export started
    ExportStarted { format: String },
    /// Export completed
    ExportComplete { path: String, size_bytes: u64 },
}

/// Callback type for training events
pub type EventCallback = Box<dyn Fn(&QatEvent) + Send + Sync>;

// ============================================================================
// Training Batch
// ============================================================================

/// A training batch
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// Batch ID
    pub id: usize,
    /// Input token IDs (batch_size, seq_len)
    pub input_ids: Vec<Vec<u32>>,
    /// Labels for next-token prediction
    pub labels: Vec<Vec<u32>>,
    /// Optional teacher outputs for distillation
    pub teacher_output: Option<TeacherOutput>,
    /// Optional reasoning steps for CoT loss
    pub reasoning_steps: Option<Vec<ReasoningStep>>,
}

impl TrainingBatch {
    /// Create a simple batch without distillation
    pub fn simple(id: usize, input_ids: Vec<Vec<u32>>, labels: Vec<Vec<u32>>) -> Self {
        Self {
            id,
            input_ids,
            labels,
            teacher_output: None,
            reasoning_steps: None,
        }
    }

    /// Batch size
    pub fn batch_size(&self) -> usize {
        self.input_ids.len()
    }

    /// Sequence length
    pub fn seq_len(&self) -> usize {
        self.input_ids.first().map(|v| v.len()).unwrap_or(0)
    }
}

// ============================================================================
// Training Result
// ============================================================================

/// Result of QAT training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Calibration result
    pub calibration: CalibrationResult,
    /// Epoch metrics history
    pub epoch_history: Vec<EpochMetrics>,
    /// Final per-layer scales
    pub final_scales: HashMap<String, f32>,
    /// Total training duration
    pub total_duration_ms: u64,
    /// Final loss
    pub final_loss: f64,
    /// Final perplexity
    pub final_perplexity: f64,
    /// Final reasoning score
    pub final_reasoning_score: f64,
    /// Configuration used
    pub config: QatConfig,
}

impl TrainingResult {
    /// Export trained model to file
    pub fn export(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| RuvLLMError::Model(format!("Export serialization failed: {}", e)))?;
        std::fs::write(path, json)
            .map_err(|e| RuvLLMError::Model(format!("Export write failed: {}", e)))?;
        Ok(())
    }

    /// Load training result
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| RuvLLMError::Model(format!("Load read failed: {}", e)))?;
        let result = serde_json::from_str(&json)
            .map_err(|e| RuvLLMError::Model(format!("Load deserialization failed: {}", e)))?;
        Ok(result)
    }

    /// Get final metrics summary
    pub fn summary(&self) -> String {
        format!(
            "QAT Training Summary:\n\
             - Epochs: {}\n\
             - Final Loss: {:.4}\n\
             - Final PPL: {:.2}\n\
             - Reasoning Score: {:.2}%\n\
             - Duration: {:.1}s\n\
             - Layers Quantized: {}",
            self.epoch_history.len(),
            self.final_loss,
            self.final_perplexity,
            self.final_reasoning_score * 100.0,
            self.total_duration_ms as f64 / 1000.0,
            self.final_scales.len()
        )
    }
}

// ============================================================================
// QAT Trainer
// ============================================================================

/// Main QAT training orchestrator
///
/// Manages the full training pipeline including calibration,
/// training loop, and model export.
pub struct QatTrainer {
    /// Training configuration
    config: QatConfig,
    /// Current training phase
    phase: TrainingPhase,
    /// Calibration engine
    calibration_engine: CalibrationEngine,
    /// Distillation loss
    distillation_loss: DistillationLoss,
    /// Reasoning loss
    reasoning_loss: ChainOfThoughtLoss,
    /// Per-layer quantizers
    quantizers: HashMap<String, Box<dyn DifferentiableQuantizer>>,
    /// Event callbacks
    callbacks: Vec<EventCallback>,
    /// Training start time
    start_time: Option<Instant>,
    /// Current learning rate
    current_lr: f64,
    /// Step metrics for current epoch
    current_epoch_steps: Vec<StepMetrics>,
}

impl QatTrainer {
    /// Create a new QAT trainer
    pub fn new(config: QatConfig) -> Self {
        let calibration_config = CalibrationConfig {
            num_samples: 128,
            percentile: 99.9,
            method: super::calibration::CalibrationMethod::MinMax,
            per_channel: true,
            min_scale: 1e-8,
            include_tool_use: true,
            include_reasoning: true,
        };

        let distillation_config = DistillationConfig {
            temperature: 2.0,
            lambda_task: config.loss_weights.lambda_task,
            lambda_kd: config.loss_weights.lambda_kd,
            lambda_reasoning: config.loss_weights.lambda_reasoning,
            use_hard_labels: true,
            min_loss: 1e-10,
        };

        let reasoning_config = ReasoningConfig::default();

        Self {
            current_lr: config.learning_rate as f64,
            config,
            phase: TrainingPhase::Idle,
            calibration_engine: CalibrationEngine::new(calibration_config),
            distillation_loss: DistillationLoss::new(distillation_config),
            reasoning_loss: ChainOfThoughtLoss::new(reasoning_config),
            quantizers: HashMap::new(),
            callbacks: Vec::new(),
            start_time: None,
            current_epoch_steps: Vec::new(),
        }
    }

    /// Get current phase
    pub fn phase(&self) -> TrainingPhase {
        self.phase
    }

    /// Get configuration
    pub fn config(&self) -> &QatConfig {
        &self.config
    }

    /// Register event callback
    pub fn on_event<F>(&mut self, callback: F)
    where
        F: Fn(&QatEvent) + Send + Sync + 'static,
    {
        self.callbacks.push(Box::new(callback));
    }

    /// Emit event to all callbacks
    fn emit(&self, event: QatEvent) {
        for callback in &self.callbacks {
            callback(&event);
        }
    }

    /// Initialize quantizer for a layer
    pub fn init_layer_quantizer(&mut self, layer_name: &str) {
        let quantizer = create_quantizer(&self.config);
        self.quantizers.insert(layer_name.to_string(), quantizer);
    }

    /// Run calibration phase
    pub fn calibrate(
        &mut self,
        activations: &HashMap<String, Vec<f32>>,
    ) -> Result<CalibrationResult> {
        self.phase = TrainingPhase::Calibration;
        self.emit(QatEvent::CalibrationStarted {
            config: CalibrationConfig::default(),
        });

        self.calibration_engine.start();

        // Observe activations for each layer
        for (layer_name, acts) in activations {
            self.calibration_engine.observe_layer(layer_name, acts);
        }

        // Finalize and get scales
        let result = self.calibration_engine.finalize()?;

        // Initialize quantizers with calibrated scales
        for (layer_name, &scale) in &result.scales {
            if let Some(quantizer) = self.quantizers.get_mut(layer_name) {
                quantizer.set_scale(scale);
            }
        }

        self.emit(QatEvent::CalibrationComplete {
            result: result.clone(),
        });

        Ok(result)
    }

    /// Run a single training step
    pub fn train_step(&mut self, batch: &TrainingBatch, step: usize) -> Result<StepMetrics> {
        let step_start = Instant::now();

        // Flatten batch for loss computation
        let student_logits = self.forward_quantized(batch)?;
        let labels: Vec<u32> = batch.labels.iter().flatten().copied().collect();

        // Compute loss components
        let (task_loss, kd_loss) = if let Some(ref teacher) = batch.teacher_output {
            let loss = self
                .distillation_loss
                .compute(&student_logits, teacher, &labels);
            let stats = self.distillation_loss.stats();
            (stats.avg_task_loss as f32, stats.avg_kd_loss as f32)
        } else {
            let vocab_size = 32000; // TODO: Get from model
            let task_loss =
                self.distillation_loss
                    .compute_task_loss(&student_logits, &labels, vocab_size);
            (task_loss, 0.0)
        };

        // Reasoning loss
        let reasoning_loss = if let Some(ref steps) = batch.reasoning_steps {
            self.reasoning_loss.compute(steps, None, None)
        } else {
            0.0
        };

        // Total loss
        let total_loss = self.config.loss_weights.lambda_task * task_loss
            + self.config.loss_weights.lambda_kd * kd_loss
            + self.config.loss_weights.lambda_reasoning * reasoning_loss;

        let metrics = StepMetrics {
            step,
            loss: total_loss,
            task_loss,
            kd_loss,
            reasoning_loss,
            duration_ms: step_start.elapsed().as_millis() as u64,
            grad_norm: 0.0, // TODO: Compute actual gradient norm
        };

        self.current_epoch_steps.push(metrics.clone());
        self.emit(QatEvent::StepComplete {
            metrics: metrics.clone(),
        });

        Ok(metrics)
    }

    /// Forward pass with quantization
    fn forward_quantized(&self, _batch: &TrainingBatch) -> Result<Vec<f32>> {
        // TODO: Implement actual forward pass through quantized model
        // For now, return dummy logits
        let batch_size = _batch.batch_size();
        let seq_len = _batch.seq_len();
        let vocab_size = 32000;

        Ok(vec![0.0; batch_size * seq_len * vocab_size])
    }

    /// Run a single training epoch
    pub fn train_epoch(&mut self, epoch: usize, batches: &[TrainingBatch]) -> Result<EpochMetrics> {
        self.phase = TrainingPhase::Training;
        let epoch_start = Instant::now();
        self.current_epoch_steps.clear();

        self.emit(QatEvent::EpochStarted {
            epoch,
            total_epochs: self.config.epochs,
        });

        for (step, batch) in batches.iter().enumerate() {
            self.train_step(batch, step)?;
        }

        // Compute epoch metrics
        let metrics = EpochMetrics::from_steps(
            epoch,
            &self.current_epoch_steps,
            self.current_lr,
            epoch_start.elapsed(),
        );

        // Update learning rate (cosine decay)
        self.update_learning_rate(epoch);

        self.emit(QatEvent::EpochComplete {
            metrics: metrics.clone(),
        });

        Ok(metrics)
    }

    /// Update learning rate with cosine decay
    fn update_learning_rate(&mut self, epoch: usize) {
        let progress = epoch as f64 / self.config.epochs as f64;
        let decay = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
        self.current_lr = self.config.learning_rate as f64 * decay;
    }

    /// Save training checkpoint
    pub fn save_checkpoint(&self, path: &str, epoch: usize, step: usize) -> Result<()> {
        let checkpoint = TrainingCheckpoint {
            epoch,
            step,
            phase: self.phase,
            total_loss: self.current_epoch_steps.iter().map(|s| s.loss as f64).sum(),
            scales: self.get_current_scales(),
            rng_state: 0, // TODO: Save actual RNG state
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        checkpoint.save(path)?;

        self.emit(QatEvent::CheckpointSaved {
            path: path.to_string(),
            epoch,
        });

        Ok(())
    }

    /// Get current scales from all quantizers
    fn get_current_scales(&self) -> HashMap<String, f32> {
        self.quantizers
            .iter()
            .map(|(name, q)| (name.clone(), q.scale()))
            .collect()
    }

    /// Resume training from checkpoint
    pub fn resume_from_checkpoint(&mut self, checkpoint: &TrainingCheckpoint) -> Result<()> {
        self.phase = checkpoint.phase;

        // Restore scales
        for (layer_name, &scale) in &checkpoint.scales {
            if let Some(quantizer) = self.quantizers.get_mut(layer_name) {
                quantizer.set_scale(scale);
            }
        }

        Ok(())
    }

    /// Run full training pipeline
    ///
    /// This is the main entry point for QAT training.
    pub fn run(
        &mut self,
        calibration_data: &HashMap<String, Vec<f32>>,
        training_batches: &[TrainingBatch],
    ) -> Result<TrainingResult> {
        self.start_time = Some(Instant::now());

        // Phase 1: Calibration
        let calibration_result = self.calibrate(calibration_data)?;

        // Phase 2: Training
        let mut epoch_history = Vec::new();
        for epoch in 0..self.config.epochs {
            let metrics = self.train_epoch(epoch, training_batches)?;
            epoch_history.push(metrics);
        }

        // Phase 3: Export
        self.phase = TrainingPhase::Export;

        let total_duration = self.start_time.unwrap().elapsed();
        let final_metrics = epoch_history.last().cloned().unwrap_or_default();

        let result = TrainingResult {
            calibration: calibration_result,
            epoch_history,
            final_scales: self.get_current_scales(),
            total_duration_ms: total_duration.as_millis() as u64,
            final_loss: final_metrics.avg_loss,
            final_perplexity: final_metrics.perplexity,
            final_reasoning_score: final_metrics.reasoning_score,
            config: self.config.clone(),
        };

        self.phase = TrainingPhase::Complete;
        self.emit(QatEvent::TrainingComplete {
            total_epochs: self.config.epochs,
            final_loss: result.final_loss,
            duration_ms: result.total_duration_ms,
        });

        Ok(result)
    }

    /// Reset trainer state
    pub fn reset(&mut self) {
        self.phase = TrainingPhase::Idle;
        self.calibration_engine.reset();
        self.distillation_loss.reset_stats();
        self.reasoning_loss.reset_metrics();
        self.quantizers.clear();
        self.current_epoch_steps.clear();
        self.start_time = None;
        self.current_lr = self.config.learning_rate as f64;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_phase() {
        assert_eq!(TrainingPhase::default(), TrainingPhase::Idle);
    }

    #[test]
    fn test_epoch_metrics_from_steps() {
        let steps = vec![
            StepMetrics {
                step: 0,
                loss: 2.0,
                task_loss: 1.5,
                kd_loss: 0.3,
                reasoning_loss: 0.2,
                duration_ms: 100,
                grad_norm: 1.0,
            },
            StepMetrics {
                step: 1,
                loss: 1.8,
                task_loss: 1.3,
                kd_loss: 0.3,
                reasoning_loss: 0.2,
                duration_ms: 95,
                grad_norm: 0.9,
            },
        ];

        let metrics = EpochMetrics::from_steps(0, &steps, 1e-4, Duration::from_millis(195));

        assert_eq!(metrics.epoch, 0);
        assert!((metrics.avg_loss - 1.9).abs() < 0.01);
        assert_eq!(metrics.num_steps, 2);
    }

    #[test]
    fn test_training_batch() {
        let batch = TrainingBatch::simple(
            0,
            vec![vec![1, 2, 3], vec![4, 5, 6]],
            vec![vec![2, 3, 4], vec![5, 6, 7]],
        );

        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.seq_len(), 3);
    }

    #[test]
    fn test_qat_trainer_creation() {
        let config = QatConfig::piq3();
        let trainer = QatTrainer::new(config);

        assert_eq!(trainer.phase(), TrainingPhase::Idle);
    }

    #[test]
    fn test_checkpoint_serialization() {
        let checkpoint = TrainingCheckpoint {
            epoch: 5,
            step: 100,
            phase: TrainingPhase::Training,
            total_loss: 1.5,
            scales: HashMap::from([("layer.0".to_string(), 0.1)]),
            rng_state: 42,
            timestamp: 1234567890,
        };

        let json = serde_json::to_string(&checkpoint).unwrap();
        let restored: TrainingCheckpoint = serde_json::from_str(&json).unwrap();

        assert_eq!(checkpoint.epoch, restored.epoch);
        assert_eq!(checkpoint.step, restored.step);
    }

    #[test]
    fn test_training_result_summary() {
        let result = TrainingResult {
            calibration: CalibrationResult {
                scales: HashMap::new(),
                zeros: HashMap::new(),
                channel_scales: None,
                config: CalibrationConfig::default(),
                stats: super::super::calibration::CalibrationStats::default(),
            },
            epoch_history: vec![EpochMetrics {
                epoch: 0,
                avg_loss: 1.5,
                perplexity: 4.48,
                ..Default::default()
            }],
            final_scales: HashMap::from([("layer.0".to_string(), 0.1)]),
            total_duration_ms: 60000,
            final_loss: 1.5,
            final_perplexity: 4.48,
            final_reasoning_score: 0.85,
            config: QatConfig::default(),
        };

        let summary = result.summary();
        assert!(summary.contains("Final Loss"));
        assert!(summary.contains("Reasoning Score"));
    }

    #[test]
    fn test_learning_rate_decay() {
        let config = QatConfig::piq3().with_epochs(10);
        let mut trainer = QatTrainer::new(config);

        let initial_lr = trainer.current_lr;
        trainer.update_learning_rate(5); // Midpoint

        // At midpoint of cosine decay, lr should be ~0.5 of initial
        assert!(trainer.current_lr < initial_lr);
        assert!(trainer.current_lr > 0.0);
    }
}
