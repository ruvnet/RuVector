//! FastGRNN training pipeline with knowledge distillation
//!
//! This module provides a complete training infrastructure for the FastGRNN model:
//! - Adam optimizer implementation
//! - Binary Cross-Entropy loss with gradient computation
//! - Backpropagation Through Time (BPTT)
//! - Mini-batch training with validation split
//! - Early stopping and learning rate scheduling
//! - Knowledge distillation from teacher models
//! - Progress reporting and metrics tracking

use crate::error::{Result, TinyDancerError};
use crate::model::{FastGRNN, FastGRNNConfig, FastGRNNGradients};
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// One DRACO routing observation: a query embedding and the quality each model
/// achieved on it. This is the neutral `{ embedding, scores }` shape that
/// `@metaharness/router` (`fromExamples` / `trainRouter`) also consumes, so a
/// single dataset seeds the k-NN/KRR router *and* trains the native FastGRNN.
#[derive(Debug, Clone)]
pub struct DracoRow {
    /// Query embedding (used directly as the model's input features).
    pub embedding: Vec<f32>,
    /// model id → quality achieved on this query (0..1).
    pub scores: HashMap<String, f32>,
}

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Validation split ratio (0.0 to 1.0)
    pub validation_split: f32,
    /// Early stopping patience (epochs)
    pub early_stopping_patience: Option<usize>,
    /// Learning rate decay factor
    pub lr_decay: f32,
    /// Learning rate decay step (epochs)
    pub lr_decay_step: usize,
    /// Gradient clipping threshold
    pub grad_clip: f32,
    /// Adam beta1 parameter
    pub adam_beta1: f32,
    /// Adam beta2 parameter
    pub adam_beta2: f32,
    /// Adam epsilon for numerical stability
    pub adam_epsilon: f32,
    /// L2 regularization strength
    pub l2_reg: f32,
    /// Enable knowledge distillation
    pub enable_distillation: bool,
    /// Temperature for distillation
    pub distillation_temperature: f32,
    /// Alpha for balancing hard and soft targets (0.0 = only hard, 1.0 = only soft)
    pub distillation_alpha: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            validation_split: 0.2,
            early_stopping_patience: Some(10),
            lr_decay: 0.5,
            lr_decay_step: 20,
            grad_clip: 5.0,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            adam_epsilon: 1e-8,
            l2_reg: 1e-5,
            enable_distillation: false,
            distillation_temperature: 3.0,
            distillation_alpha: 0.5,
        }
    }
}

/// Training dataset with features and labels
#[derive(Debug, Clone)]
pub struct TrainingDataset {
    /// Input features (N x input_dim)
    pub features: Vec<Vec<f32>>,
    /// Target labels (N)
    pub labels: Vec<f32>,
    /// Optional teacher soft targets for distillation (N)
    pub soft_targets: Option<Vec<f32>>,
}

impl TrainingDataset {
    /// Create a new training dataset
    pub fn new(features: Vec<Vec<f32>>, labels: Vec<f32>) -> Result<Self> {
        if features.len() != labels.len() {
            return Err(TinyDancerError::InvalidInput(
                "Features and labels must have the same length".to_string(),
            ));
        }
        if features.is_empty() {
            return Err(TinyDancerError::InvalidInput(
                "Dataset cannot be empty".to_string(),
            ));
        }

        Ok(Self {
            features,
            labels,
            soft_targets: None,
        })
    }

    /// Build a training dataset from DRACO routing rows + a per-model price table.
    ///
    /// For each row, the binary label answers "is a cheap model good enough here?"
    /// — `1.0` if the cheapest priced model's quality is within `tolerance` of the
    /// best model's quality, else `0.0` (route to a stronger model). The soft
    /// target is the cheapest model's actual quality, so distillation can regress
    /// toward the real achieved quality. Features are the query embeddings; the
    /// `FastGRNNConfig.input_dim` must match the embedding length.
    pub fn from_draco(
        rows: &[DracoRow],
        prices: &HashMap<String, f32>,
        tolerance: f32,
    ) -> Result<Self> {
        if rows.is_empty() {
            return Err(TinyDancerError::InvalidInput(
                "DRACO dataset cannot be empty".to_string(),
            ));
        }

        let mut features = Vec::with_capacity(rows.len());
        let mut labels = Vec::with_capacity(rows.len());
        let mut soft = Vec::with_capacity(rows.len());

        for row in rows {
            if row.scores.is_empty() {
                return Err(TinyDancerError::InvalidInput(
                    "DRACO row has no model scores".to_string(),
                ));
            }
            // Cheapest model that appears in both the scores and the price table.
            let cheapest = row
                .scores
                .keys()
                .filter(|id| prices.contains_key(*id))
                .min_by(|a, b| {
                    prices[*a]
                        .partial_cmp(&prices[*b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .ok_or_else(|| {
                    TinyDancerError::InvalidInput(
                        "DRACO row has no model present in the price table".to_string(),
                    )
                })?;

            let best_q = row.scores.values().copied().fold(f32::MIN, f32::max);
            let cheap_q = row.scores[cheapest];
            let label = if cheap_q >= best_q - tolerance {
                1.0
            } else {
                0.0
            };

            features.push(row.embedding.clone());
            labels.push(label);
            soft.push(cheap_q.clamp(0.0, 1.0));
        }

        Self::new(features, labels)?.with_soft_targets(soft)
    }

    /// Add soft targets from teacher model for knowledge distillation
    pub fn with_soft_targets(mut self, soft_targets: Vec<f32>) -> Result<Self> {
        if soft_targets.len() != self.labels.len() {
            return Err(TinyDancerError::InvalidInput(
                "Soft targets must match dataset size".to_string(),
            ));
        }
        self.soft_targets = Some(soft_targets);
        Ok(self)
    }

    /// Split dataset into train and validation sets
    pub fn split(&self, val_ratio: f32) -> Result<(Self, Self)> {
        if !(0.0..=1.0).contains(&val_ratio) {
            return Err(TinyDancerError::InvalidInput(
                "Validation ratio must be between 0.0 and 1.0".to_string(),
            ));
        }

        let n_samples = self.features.len();
        let n_val = (n_samples as f32 * val_ratio) as usize;
        let n_train = n_samples - n_val;

        // Create shuffled indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        let train_indices = &indices[..n_train];
        let val_indices = &indices[n_train..];

        let train_features: Vec<Vec<f32>> = train_indices
            .iter()
            .map(|&i| self.features[i].clone())
            .collect();
        let train_labels: Vec<f32> = train_indices.iter().map(|&i| self.labels[i]).collect();

        let val_features: Vec<Vec<f32>> = val_indices
            .iter()
            .map(|&i| self.features[i].clone())
            .collect();
        let val_labels: Vec<f32> = val_indices.iter().map(|&i| self.labels[i]).collect();

        let mut train_dataset = Self::new(train_features, train_labels)?;
        let mut val_dataset = Self::new(val_features, val_labels)?;

        // Split soft targets if present
        if let Some(soft_targets) = &self.soft_targets {
            let train_soft: Vec<f32> = train_indices.iter().map(|&i| soft_targets[i]).collect();
            let val_soft: Vec<f32> = val_indices.iter().map(|&i| soft_targets[i]).collect();
            train_dataset.soft_targets = Some(train_soft);
            val_dataset.soft_targets = Some(val_soft);
        }

        Ok((train_dataset, val_dataset))
    }

    /// Normalize features using z-score normalization
    pub fn normalize(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        if self.features.is_empty() {
            return Err(TinyDancerError::InvalidInput(
                "Cannot normalize empty dataset".to_string(),
            ));
        }

        let n_features = self.features[0].len();
        let mut means = vec![0.0; n_features];
        let mut stds = vec![0.0; n_features];

        // Compute means
        for feature in &self.features {
            for (i, &val) in feature.iter().enumerate() {
                means[i] += val;
            }
        }
        for mean in &mut means {
            *mean /= self.features.len() as f32;
        }

        // Compute standard deviations
        for feature in &self.features {
            for (i, &val) in feature.iter().enumerate() {
                stds[i] += (val - means[i]).powi(2);
            }
        }
        for std in &mut stds {
            *std = (*std / self.features.len() as f32).sqrt();
            if *std < 1e-8 {
                *std = 1.0; // Avoid division by zero
            }
        }

        // Normalize features
        for feature in &mut self.features {
            for (i, val) in feature.iter_mut().enumerate() {
                *val = (*val - means[i]) / stds[i];
            }
        }

        Ok((means, stds))
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// Batch iterator for training
pub struct BatchIterator<'a> {
    dataset: &'a TrainingDataset,
    batch_size: usize,
    indices: Vec<usize>,
    current_idx: usize,
}

impl<'a> BatchIterator<'a> {
    /// Create a new batch iterator
    pub fn new(dataset: &'a TrainingDataset, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        Self {
            dataset,
            batch_size,
            indices,
            current_idx: 0,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Vec<Vec<f32>>, Vec<f32>, Option<Vec<f32>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];

        let features: Vec<Vec<f32>> = batch_indices
            .iter()
            .map(|&i| self.dataset.features[i].clone())
            .collect();

        let labels: Vec<f32> = batch_indices
            .iter()
            .map(|&i| self.dataset.labels[i])
            .collect();

        let soft_targets = self
            .dataset
            .soft_targets
            .as_ref()
            .map(|targets| batch_indices.iter().map(|&i| targets[i]).collect());

        self.current_idx = end_idx;

        Some((features, labels, soft_targets))
    }
}

/// Adam optimizer state
#[derive(Debug)]
struct AdamOptimizer {
    /// First moment estimates
    m_weights: Vec<Array2<f32>>,
    m_biases: Vec<Array1<f32>>,
    /// Second moment estimates
    v_weights: Vec<Array2<f32>>,
    v_biases: Vec<Array1<f32>>,
    /// Time step
    t: usize,
    /// Configuration
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl AdamOptimizer {
    fn new(model_config: &FastGRNNConfig, training_config: &TrainingConfig) -> Self {
        let hidden_dim = model_config.hidden_dim;
        let input_dim = model_config.input_dim;
        let output_dim = model_config.output_dim;

        Self {
            m_weights: vec![
                Array2::zeros((hidden_dim, input_dim)),  // w_reset
                Array2::zeros((hidden_dim, input_dim)),  // w_update
                Array2::zeros((hidden_dim, input_dim)),  // w_candidate
                Array2::zeros((hidden_dim, hidden_dim)), // w_recurrent
                Array2::zeros((output_dim, hidden_dim)), // w_output
            ],
            m_biases: vec![
                Array1::zeros(hidden_dim), // b_reset
                Array1::zeros(hidden_dim), // b_update
                Array1::zeros(hidden_dim), // b_candidate
                Array1::zeros(output_dim), // b_output
            ],
            v_weights: vec![
                Array2::zeros((hidden_dim, input_dim)),
                Array2::zeros((hidden_dim, input_dim)),
                Array2::zeros((hidden_dim, input_dim)),
                Array2::zeros((hidden_dim, hidden_dim)),
                Array2::zeros((output_dim, hidden_dim)),
            ],
            v_biases: vec![
                Array1::zeros(hidden_dim),
                Array1::zeros(hidden_dim),
                Array1::zeros(hidden_dim),
                Array1::zeros(output_dim),
            ],
            t: 0,
            beta1: training_config.adam_beta1,
            beta2: training_config.adam_beta2,
            epsilon: training_config.adam_epsilon,
        }
    }
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Epoch number
    pub epoch: usize,
    /// Training loss
    pub train_loss: f32,
    /// Validation loss
    pub val_loss: f32,
    /// Training accuracy
    pub train_accuracy: f32,
    /// Validation accuracy
    pub val_accuracy: f32,
    /// Learning rate
    pub learning_rate: f32,
}

/// FastGRNN trainer
pub struct Trainer {
    config: TrainingConfig,
    optimizer: AdamOptimizer,
    best_val_loss: f32,
    patience_counter: usize,
    metrics_history: Vec<TrainingMetrics>,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(model_config: &FastGRNNConfig, config: TrainingConfig) -> Self {
        let optimizer = AdamOptimizer::new(model_config, &config);

        Self {
            config,
            optimizer,
            best_val_loss: f32::INFINITY,
            patience_counter: 0,
            metrics_history: Vec::new(),
        }
    }

    /// Train the model
    pub fn train(
        &mut self,
        model: &mut FastGRNN,
        dataset: &TrainingDataset,
    ) -> Result<Vec<TrainingMetrics>> {
        // Split dataset
        let (train_dataset, val_dataset) = dataset.split(self.config.validation_split)?;

        println!("Training FastGRNN model");
        println!(
            "Train samples: {}, Val samples: {}",
            train_dataset.len(),
            val_dataset.len()
        );
        println!("Hyperparameters: {:?}", self.config);

        let mut current_lr = self.config.learning_rate;

        for epoch in 0..self.config.epochs {
            // Learning rate scheduling
            if epoch > 0 && epoch % self.config.lr_decay_step == 0 {
                current_lr *= self.config.lr_decay;
                println!("Decaying learning rate to {:.6}", current_lr);
            }

            // Training phase
            let train_loss = self.train_epoch(model, &train_dataset, current_lr)?;

            // Validation phase
            let (val_loss, val_accuracy) = self.evaluate(model, &val_dataset)?;
            let (_, train_accuracy) = self.evaluate(model, &train_dataset)?;

            // Record metrics
            let metrics = TrainingMetrics {
                epoch,
                train_loss,
                val_loss,
                train_accuracy,
                val_accuracy,
                learning_rate: current_lr,
            };
            self.metrics_history.push(metrics.clone());

            // Print progress
            println!(
                "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}, train_acc={:.4}, val_acc={:.4}",
                epoch + 1,
                self.config.epochs,
                train_loss,
                val_loss,
                train_accuracy,
                val_accuracy
            );

            // Early stopping
            if let Some(patience) = self.config.early_stopping_patience {
                if val_loss < self.best_val_loss {
                    self.best_val_loss = val_loss;
                    self.patience_counter = 0;
                    println!("New best validation loss: {:.4}", val_loss);
                } else {
                    self.patience_counter += 1;
                    if self.patience_counter >= patience {
                        println!("Early stopping triggered at epoch {}", epoch + 1);
                        break;
                    }
                }
            }
        }

        Ok(self.metrics_history.clone())
    }

    /// Train for one epoch
    fn train_epoch(
        &mut self,
        model: &mut FastGRNN,
        dataset: &TrainingDataset,
        learning_rate: f32,
    ) -> Result<f32> {
        let mut total_loss = 0.0;
        let mut n_batches = 0;

        let batch_iter = BatchIterator::new(dataset, self.config.batch_size, true);

        for (features, labels, soft_targets) in batch_iter {
            let batch_loss = self.train_batch(
                model,
                &features,
                &labels,
                soft_targets.as_ref(),
                learning_rate,
            )?;
            total_loss += batch_loss;
            n_batches += 1;
        }

        Ok(total_loss / n_batches as f32)
    }

    /// Train on a single batch: forward-cached → analytic backward per sample,
    /// mean-accumulate gradients, then one Adam step.
    fn train_batch(
        &mut self,
        model: &mut FastGRNN,
        features: &[Vec<f32>],
        labels: &[f32],
        soft_targets: Option<&Vec<f32>>,
        learning_rate: f32,
    ) -> Result<f32> {
        let batch_size = features.len();
        if batch_size == 0 {
            return Ok(0.0);
        }
        let mut total_loss = 0.0;
        let mut grad_accum = FastGRNNGradients::zeros(model.config());

        for (i, feature) in features.iter().enumerate() {
            let (prediction, cache) = model.forward_cached(feature, None)?;
            let hard = labels[i];

            // Loss and the effective target used for the gradient. For
            // BCE-with-sigmoid, dL/d(logit) = prediction - target; distillation
            // blends the hard label with the teacher's soft target.
            let (loss, target_for_grad) = if self.config.enable_distillation {
                if let Some(soft) = soft_targets {
                    let s = soft[i];
                    let hard_loss = binary_cross_entropy(prediction, hard);
                    let soft_loss = binary_cross_entropy(prediction, s);
                    let a = self.config.distillation_alpha;
                    (
                        a * soft_loss + (1.0 - a) * hard_loss,
                        a * s + (1.0 - a) * hard,
                    )
                } else {
                    (binary_cross_entropy(prediction, hard), hard)
                }
            } else {
                (binary_cross_entropy(prediction, hard), hard)
            };

            total_loss += loss;

            let d_logit = prediction - target_for_grad;
            let grads = model.backward(&cache, d_logit);
            grad_accum.add_scaled(&grads, 1.0);
        }

        // Mean gradient over the batch.
        grad_accum.scale(1.0 / batch_size as f32);

        // Adam step (L2 + global-norm clip + bias correction + update).
        self.apply_gradients(model, &grad_accum, learning_rate)?;

        Ok(total_loss / batch_size as f32)
    }

    /// Apply accumulated gradients with the Adam optimizer: L2 regularization,
    /// global-norm gradient clipping, bias-corrected moments, parameter update.
    fn apply_gradients(
        &mut self,
        model: &mut FastGRNN,
        grads: &FastGRNNGradients,
        learning_rate: f32,
    ) -> Result<()> {
        self.optimizer.t += 1;
        let t = self.optimizer.t as i32;

        // Working copy so we can add L2 and clip without mutating the caller's grads.
        let mut g = FastGRNNGradients::zeros(model.config());
        g.add_scaled(grads, 1.0);

        // L2 regularization on weight matrices (not biases): grad += l2 * w.
        let l2 = self.config.l2_reg;
        if l2 > 0.0 {
            let weights = model.weights_mut();
            for k in 0..5 {
                g.w[k] = &g.w[k] + &(&*weights[k] * l2);
            }
        }

        // Global-norm gradient clipping.
        if self.config.grad_clip > 0.0 {
            let norm = g.global_norm();
            if norm > self.config.grad_clip {
                g.scale(self.config.grad_clip / norm);
            }
        }

        let (b1, b2, eps) = (
            self.optimizer.beta1,
            self.optimizer.beta2,
            self.optimizer.epsilon,
        );
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);

        // Weight matrices.
        {
            let mut weights = model.weights_mut();
            for (k, w) in weights.iter_mut().enumerate() {
                let m = &mut self.optimizer.m_weights[k];
                let v = &mut self.optimizer.v_weights[k];
                *m = &*m * b1 + &(&g.w[k] * (1.0 - b1));
                *v = &*v * b2 + &(g.w[k].mapv(|x| x * x) * (1.0 - b2));
                let m_hat = &*m / bc1;
                let v_hat = &*v / bc2;
                let update = &m_hat * learning_rate / &(v_hat.mapv(|x| x.sqrt()) + eps);
                **w = &**w - &update;
            }
        }

        // Bias vectors.
        {
            let mut biases = model.biases_mut();
            for (k, b) in biases.iter_mut().enumerate() {
                let m = &mut self.optimizer.m_biases[k];
                let v = &mut self.optimizer.v_biases[k];
                *m = &*m * b1 + &(&g.b[k] * (1.0 - b1));
                *v = &*v * b2 + &(g.b[k].mapv(|x| x * x) * (1.0 - b2));
                let m_hat = &*m / bc1;
                let v_hat = &*v / bc2;
                let update = &m_hat * learning_rate / &(v_hat.mapv(|x| x.sqrt()) + eps);
                **b = &**b - &update;
            }
        }

        Ok(())
    }

    /// Evaluate model on dataset
    fn evaluate(&self, model: &FastGRNN, dataset: &TrainingDataset) -> Result<(f32, f32)> {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for (i, feature) in dataset.features.iter().enumerate() {
            let prediction = model.forward(feature, None)?;
            let target = dataset.labels[i];

            // Compute loss
            let loss = binary_cross_entropy(prediction, target);
            total_loss += loss;

            // Compute accuracy (threshold at 0.5)
            let predicted_class = if prediction >= 0.5 { 1.0_f32 } else { 0.0_f32 };
            let target_class = if target >= 0.5 { 1.0_f32 } else { 0.0_f32 };
            if (predicted_class - target_class).abs() < 0.01_f32 {
                correct += 1;
            }
        }

        let avg_loss = total_loss / dataset.len() as f32;
        let accuracy = correct as f32 / dataset.len() as f32;

        Ok((avg_loss, accuracy))
    }

    /// Get training metrics history
    pub fn metrics_history(&self) -> &[TrainingMetrics] {
        &self.metrics_history
    }

    /// Save metrics to file
    pub fn save_metrics<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.metrics_history)
            .map_err(|e| TinyDancerError::SerializationError(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// Binary cross-entropy loss
fn binary_cross_entropy(prediction: f32, target: f32) -> f32 {
    let eps = 1e-7;
    let pred = prediction.clamp(eps, 1.0 - eps);
    -target * pred.ln() - (1.0 - target) * (1.0 - pred).ln()
}

/// Temperature-scaled softmax for knowledge distillation with numerical stability
pub fn temperature_softmax(logit: f32, temperature: f32) -> f32 {
    // For binary classification, we can use temperature-scaled sigmoid
    let scaled = logit / temperature;
    if scaled > 0.0 {
        1.0 / (1.0 + (-scaled).exp())
    } else {
        let ex = scaled.exp();
        ex / (1.0 + ex)
    }
}

/// Generate teacher predictions for knowledge distillation
pub fn generate_teacher_predictions(
    teacher: &FastGRNN,
    features: &[Vec<f32>],
    temperature: f32,
) -> Result<Vec<f32>> {
    features
        .iter()
        .map(|feature| {
            let logit = teacher.forward(feature, None)?;
            // Apply temperature scaling
            Ok(temperature_softmax(logit, temperature))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let labels = vec![0.0, 1.0, 0.0];
        let dataset = TrainingDataset::new(features, labels).unwrap();
        assert_eq!(dataset.len(), 3);
    }

    #[test]
    fn test_dataset_split() {
        let features = vec![vec![1.0; 5]; 100];
        let labels = vec![0.0; 100];
        let dataset = TrainingDataset::new(features, labels).unwrap();
        let (train, val) = dataset.split(0.2).unwrap();
        assert_eq!(train.len(), 80);
        assert_eq!(val.len(), 20);
    }

    #[test]
    fn test_batch_iterator() {
        let features = vec![vec![1.0; 5]; 10];
        let labels = vec![0.0; 10];
        let dataset = TrainingDataset::new(features, labels).unwrap();
        let mut iter = BatchIterator::new(&dataset, 3, false);

        let batch1 = iter.next().unwrap();
        assert_eq!(batch1.0.len(), 3);

        let batch2 = iter.next().unwrap();
        assert_eq!(batch2.0.len(), 3);

        let batch3 = iter.next().unwrap();
        assert_eq!(batch3.0.len(), 3);

        let batch4 = iter.next().unwrap();
        assert_eq!(batch4.0.len(), 1); // Last batch

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_normalization() {
        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let labels = vec![0.0, 1.0, 0.0];
        let mut dataset = TrainingDataset::new(features, labels).unwrap();
        let (means, stds) = dataset.normalize().unwrap();

        assert_eq!(means.len(), 3);
        assert_eq!(stds.len(), 3);

        // Check that normalized features have mean ~0 and std ~1
        let sum: f32 = dataset.features.iter().map(|f| f[0]).sum();
        let mean = sum / dataset.len() as f32;
        assert!((mean.abs()) < 1e-5);
    }

    #[test]
    fn test_bce_loss() {
        let loss1 = binary_cross_entropy(0.9, 1.0);
        let loss2 = binary_cross_entropy(0.1, 1.0);
        assert!(loss1 < loss2); // Prediction closer to target has lower loss
    }

    #[test]
    fn test_temperature_softmax() {
        let logit = 2.0;
        let soft1 = temperature_softmax(logit, 1.0);
        let soft2 = temperature_softmax(logit, 2.0);

        // Higher temperature should make output closer to 0.5
        assert!((soft1 - 0.5).abs() > (soft2 - 0.5).abs());
    }

    /// End-to-end: training on a linearly-separable problem must reduce loss and
    /// reach high accuracy — proves the gradient/Adam step actually learns.
    #[test]
    fn test_training_converges() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Label = 1 iff sum(features) > 0, with a margin. 4 features.
        let n = 200;
        let mut features = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            let f: Vec<f32> = (0..4).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let s: f32 = f.iter().sum();
            if s.abs() < 0.15 {
                continue; // drop ambiguous points near the boundary
            }
            labels.push(if s > 0.0 { 1.0 } else { 0.0 });
            features.push(f);
        }
        let dataset = TrainingDataset::new(features, labels).unwrap();

        let model_config = FastGRNNConfig {
            input_dim: 4,
            hidden_dim: 8,
            output_dim: 1,
            ..Default::default()
        };
        let train_config = TrainingConfig {
            learning_rate: 0.05,
            batch_size: 16,
            epochs: 60,
            validation_split: 0.2,
            early_stopping_patience: None,
            l2_reg: 0.0,
            ..Default::default()
        };
        let mut model = FastGRNN::new(model_config.clone()).unwrap();
        let mut trainer = Trainer::new(&model_config, train_config);
        let metrics = trainer.train(&mut model, &dataset).unwrap();

        let first = &metrics[0];
        let last = &metrics[metrics.len() - 1];
        assert!(
            last.train_loss < first.train_loss,
            "loss did not decrease: {} -> {}",
            first.train_loss,
            last.train_loss
        );
        assert!(
            last.train_accuracy > 0.9,
            "final train accuracy too low: {}",
            last.train_accuracy
        );
    }

    #[test]
    fn test_from_draco() {
        let prices: HashMap<String, f32> = [("haiku".to_string(), 1.0), ("opus".to_string(), 15.0)]
            .into_iter()
            .collect();

        // Row 1: cheap model (haiku) is as good as opus → label 1 (route light).
        // Row 2: cheap model much worse than opus → label 0 (route heavy).
        let rows = vec![
            DracoRow {
                embedding: vec![0.1, 0.2, 0.3],
                scores: [("haiku".to_string(), 0.90), ("opus".to_string(), 0.92)]
                    .into_iter()
                    .collect(),
            },
            DracoRow {
                embedding: vec![0.4, 0.5, 0.6],
                scores: [("haiku".to_string(), 0.40), ("opus".to_string(), 0.95)]
                    .into_iter()
                    .collect(),
            },
        ];

        let ds = TrainingDataset::from_draco(&rows, &prices, 0.05).unwrap();
        assert_eq!(ds.labels, vec![1.0, 0.0]);
        let soft = ds.soft_targets.as_ref().unwrap();
        assert!((soft[0] - 0.90).abs() < 1e-6);
        assert!((soft[1] - 0.40).abs() < 1e-6);
        assert_eq!(ds.features[0].len(), 3);
    }
}
