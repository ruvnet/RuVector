//! LoRA (Low-Rank Adaptation) implementations for SONA
//!
//! Two-tier LoRA system:
//! - MicroLoRA: Rank 1-2, per-request adaptation (<100μs)
//! - BaseLoRA: Rank 4-16, background adaptation (hourly)

use crate::types::LearningSignal;
use serde::{Deserialize, Serialize};

/// Optimal batch size for processing (benchmark-validated)
pub const OPTIMAL_BATCH_SIZE: usize = 32;

/// Largest gradient norm a single MicroLoRA update may apply. Larger
/// gradients are scaled down to this norm before the learning rate.
pub const MICRO_LORA_MAX_GRAD_NORM: f32 = 1.0;

/// Upper bound on the MicroLoRA gain: `|forward(x)| <= MICRO_LORA_MAX_GAIN * |x|`.
/// Keeps repeated feedback, or a high learning rate, from growing the
/// adapter without limit.
pub const MICRO_LORA_MAX_GAIN: f32 = 1.0;

/// Micro-LoRA for per-request adaptation
///
/// Uses rank 1-2 for ultra-low latency updates.
/// Forward pass: output += scale * (input @ down) @ up
///
/// **Performance notes (from benchmarks):**
/// - Rank-2 is ~5% faster than Rank-1 due to better SIMD vectorization
/// - Batch size 32 optimal: 0.447ms per-vector, 2,236 ops/sec throughput
/// - SIMD-enabled: +10% speedup over scalar
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MicroLoRA {
    /// Down projection (hidden_dim -> rank)
    down_proj: Vec<f32>,
    /// Up projection (rank -> hidden_dim)
    up_proj: Vec<f32>,
    /// Rank (1-2 for micro updates)
    rank: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Accumulated gradients for down
    #[serde(skip)]
    grad_down: Vec<f32>,
    /// Accumulated gradients for up
    #[serde(skip)]
    grad_up: Vec<f32>,
    /// Update count for averaging
    #[serde(skip)]
    update_count: usize,
    /// Scaling factor
    scale: f32,
}

impl MicroLoRA {
    /// Create new Micro-LoRA adapter
    ///
    /// # Arguments
    /// * `hidden_dim` - Model hidden dimension
    /// * `rank` - LoRA rank (must be 1-2)
    ///
    /// # Panics
    /// Panics if rank > 2
    pub fn new(hidden_dim: usize, rank: usize) -> Self {
        assert!(
            (1..=2).contains(&rank),
            "MicroLoRA rank must be 1-2, got {}",
            rank
        );

        // Initialize down with deterministic random-like values in
        // [-1/sqrt(hidden_dim), 1/sqrt(hidden_dim)], the usual LoRA (Kaiming
        // uniform) bound. Updates to `up` scale with (down . x)^2, so a much
        // smaller bound makes learning stall.
        let bound = 1.0 / (hidden_dim.max(1) as f32).sqrt();
        let down_proj: Vec<f32> = (0..hidden_dim * rank)
            .map(|i| {
                let x = (i as f32 * 0.618_034) % 1.0;
                (x - 0.5) * 2.0 * bound
            })
            .collect();

        // Initialize up to zero (standard LoRA init)
        let up_proj = vec![0.0f32; rank * hidden_dim];

        Self {
            down_proj,
            up_proj,
            rank,
            hidden_dim,
            grad_down: vec![0.0; hidden_dim * rank],
            grad_up: vec![0.0; rank * hidden_dim],
            update_count: 0,
            scale: 1.0 / (rank as f32).sqrt(),
        }
    }

    /// Scalar forward pass (fallback)
    pub fn forward_scalar(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.hidden_dim);
        assert_eq!(output.len(), self.hidden_dim);

        // Down projection: hidden_dim -> rank
        let mut intermediate = vec![0.0f32; self.rank];
        for (r, inter) in intermediate.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            let offset = r * self.hidden_dim;
            for (i, &inp) in input.iter().enumerate() {
                sum += inp * self.down_proj[offset + i];
            }
            *inter = sum;
        }

        // Up projection: rank -> hidden_dim
        for (i, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (r, &inter) in intermediate.iter().enumerate() {
                sum += inter * self.up_proj[r * self.hidden_dim + i];
            }
            *out += sum * self.scale;
        }
    }

    /// SIMD-optimized forward pass (AVX2)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    pub fn forward_simd(&self, input: &[f32], output: &mut [f32]) {
        use std::arch::x86_64::*;

        assert_eq!(input.len(), self.hidden_dim);
        assert_eq!(output.len(), self.hidden_dim);

        unsafe {
            // Down projection: hidden_dim -> rank
            let mut intermediate = vec![0.0f32; self.rank];

            for r in 0..self.rank {
                let mut sum = _mm256_setzero_ps();
                let offset = r * self.hidden_dim;

                let mut i = 0;
                while i + 8 <= self.hidden_dim {
                    let inp = _mm256_loadu_ps(input[i..].as_ptr());
                    let weight = _mm256_loadu_ps(self.down_proj[offset + i..].as_ptr());
                    sum = _mm256_fmadd_ps(inp, weight, sum);
                    i += 8;
                }

                // Horizontal sum
                let mut result = [0.0f32; 8];
                _mm256_storeu_ps(result.as_mut_ptr(), sum);
                intermediate[r] = result.iter().sum();

                // Handle remaining elements
                for j in i..self.hidden_dim {
                    intermediate[r] += input[j] * self.down_proj[offset + j];
                }
            }

            // Up projection: rank -> hidden_dim
            let scale_vec = _mm256_set1_ps(self.scale);

            let mut i = 0;
            while i + 8 <= self.hidden_dim {
                let mut sum = _mm256_setzero_ps();

                for r in 0..self.rank {
                    let up_offset = r * self.hidden_dim;
                    let weight = _mm256_loadu_ps(self.up_proj[up_offset + i..].as_ptr());
                    let inter = _mm256_set1_ps(intermediate[r]);
                    sum = _mm256_fmadd_ps(inter, weight, sum);
                }

                // Scale and add to output
                sum = _mm256_mul_ps(sum, scale_vec);
                let existing = _mm256_loadu_ps(output[i..].as_ptr());
                let result = _mm256_add_ps(existing, sum);
                _mm256_storeu_ps(output[i..].as_mut_ptr(), result);

                i += 8;
            }

            // Handle remaining elements
            for j in i..self.hidden_dim {
                let mut val = 0.0;
                for r in 0..self.rank {
                    val += intermediate[r] * self.up_proj[r * self.hidden_dim + j];
                }
                output[j] += val * self.scale;
            }
        }
    }

    /// Forward pass with automatic SIMD detection
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            self.forward_simd(input, output);
            return;
        }

        #[allow(unreachable_code)]
        self.forward_scalar(input, output);
    }

    /// Accumulate gradient from learning signal
    ///
    /// `gradient_estimate` is the output-space direction that raises the
    /// reward for `query_embedding`. It is back-propagated through
    /// `delta = scale * up^T (down x)`:
    ///
    /// - `d/d up[r][i]   = scale * (down[r] . x) * g[i]`
    /// - `d/d down[r][j] = scale * (up[r] . g) * x[j]`
    ///
    /// so the update depends on the input and always moves `forward(x)`
    /// toward `g` for that input.
    pub fn accumulate_gradient(&mut self, signal: &LearningSignal) {
        let dim = self.hidden_dim;
        if signal.gradient_estimate.len() != dim || signal.query_embedding.len() != dim {
            return;
        }

        let quality = signal.quality_score;
        let x = &signal.query_embedding;
        let g = &signal.gradient_estimate;

        for r in 0..self.rank {
            let offset = r * dim;
            let down = &self.down_proj[offset..offset + dim];
            let up = &self.up_proj[offset..offset + dim];
            let h: f32 = down.iter().zip(x).map(|(d, x)| d * x).sum();
            let up_dot_g: f32 = up.iter().zip(g).map(|(u, g)| u * g).sum();

            let up_coeff = quality * self.scale * h;
            let down_coeff = quality * self.scale * up_dot_g;
            for i in 0..dim {
                self.grad_up[offset + i] += up_coeff * g[i];
                self.grad_down[offset + i] += down_coeff * x[i];
            }
        }

        self.update_count += 1;
    }

    /// Apply accumulated gradients with learning rate
    pub fn apply_accumulated(&mut self, learning_rate: f32) {
        if let Some(gradient) = self.take_accumulated() {
            self.apply_gradient(&gradient, learning_rate);
        }
    }

    /// Take the averaged pending gradient as one flat vector laid out as
    /// `[down.., up..]` (see [`MicroLoRA::flat_weights`]) and reset the
    /// accumulators. Returns `None` when no update is pending.
    pub fn take_accumulated(&mut self) -> Option<Vec<f32>> {
        if self.update_count == 0 {
            return None;
        }

        let n = self.update_count as f32;
        let mut gradient = Vec::with_capacity(self.grad_down.len() + self.grad_up.len());
        gradient.extend(self.grad_down.iter().map(|g| g / n));
        gradient.extend(self.grad_up.iter().map(|g| g / n));

        self.grad_up.fill(0.0);
        self.grad_down.fill(0.0);
        self.update_count = 0;
        Some(gradient)
    }

    /// Apply a flat `[down.., up..]` gradient step.
    ///
    /// The gradient is clipped to [`MICRO_LORA_MAX_GRAD_NORM`] and the
    /// adapter gain is then capped at [`MICRO_LORA_MAX_GAIN`], so the output
    /// stays bounded for any learning rate or amount of feedback.
    pub fn apply_gradient(&mut self, gradient: &[f32], learning_rate: f32) {
        let split = self.down_proj.len();
        if gradient.len() != split + self.up_proj.len() {
            return;
        }

        let norm = gradient.iter().map(|g| g * g).sum::<f32>().sqrt();
        if !norm.is_finite() {
            return;
        }
        let clip = if norm > MICRO_LORA_MAX_GRAD_NORM {
            MICRO_LORA_MAX_GRAD_NORM / norm
        } else {
            1.0
        };
        let step = learning_rate * clip;

        for (w, g) in self.down_proj.iter_mut().zip(&gradient[..split]) {
            *w += g * step;
        }
        for (w, g) in self.up_proj.iter_mut().zip(&gradient[split..]) {
            *w += g * step;
        }

        self.limit_gain();
    }

    /// Cap the adapter gain. `|forward(x)| <= scale * sum_r |down_r| |up_r| |x|`,
    /// so each rank's `|down_r| |up_r|` is limited and both factors are
    /// rescaled equally when it is exceeded.
    fn limit_gain(&mut self) {
        let denom = self.scale.abs() * self.rank as f32;
        if denom <= 0.0 {
            return;
        }
        let cap = MICRO_LORA_MAX_GAIN / denom;
        let dim = self.hidden_dim;

        for r in 0..self.rank {
            let range = r * dim..(r + 1) * dim;
            let down_norm = self.down_proj[range.clone()]
                .iter()
                .map(|w| w * w)
                .sum::<f32>()
                .sqrt();
            let up_norm = self.up_proj[range.clone()]
                .iter()
                .map(|w| w * w)
                .sum::<f32>()
                .sqrt();
            let gain = down_norm * up_norm;
            if gain > cap {
                let factor = (cap / gain).sqrt();
                self.down_proj[range.clone()]
                    .iter_mut()
                    .for_each(|w| *w *= factor);
                self.up_proj[range].iter_mut().for_each(|w| *w *= factor);
            }
        }
    }

    /// All weights as one flat vector laid out as `[down.., up..]`, the same
    /// layout used by [`MicroLoRA::take_accumulated`].
    pub fn flat_weights(&self) -> Vec<f32> {
        let mut weights = Vec::with_capacity(self.down_proj.len() + self.up_proj.len());
        weights.extend_from_slice(&self.down_proj);
        weights.extend_from_slice(&self.up_proj);
        weights
    }

    /// Replace all weights from a flat `[down.., up..]` vector (see
    /// [`MicroLoRA::flat_weights`]). The gain cap is applied afterwards.
    /// Vectors of the wrong length are ignored.
    pub fn set_flat_weights(&mut self, weights: &[f32]) {
        let split = self.down_proj.len();
        if weights.len() != split + self.up_proj.len() {
            return;
        }
        self.down_proj.copy_from_slice(&weights[..split]);
        self.up_proj.copy_from_slice(&weights[split..]);
        self.limit_gain();
    }

    /// Reset adapter to initial state
    pub fn reset(&mut self) {
        self.up_proj.fill(0.0);
        self.grad_up.fill(0.0);
        self.grad_down.fill(0.0);
        self.update_count = 0;
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        self.down_proj.len() + self.up_proj.len()
    }

    /// Get scale factor
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Set scale factor
    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    /// Get pending update count
    pub fn pending_updates(&self) -> usize {
        self.update_count
    }

    /// Get LoRA weights for export (lora_a, lora_b)
    pub fn get_weights(&self) -> (&Vec<f32>, &Vec<f32>) {
        (&self.down_proj, &self.up_proj)
    }

    /// Set LoRA weights from external source (disk load, other system)
    ///
    /// # Arguments
    /// * `down_proj` - Down projection weights (hidden_dim * rank)
    /// * `up_proj` - Up projection weights (rank * hidden_dim)
    ///
    /// # Errors
    /// Returns Err if dimensions don't match current rank/hidden_dim
    pub fn set_weights(&mut self, down_proj: Vec<f32>, up_proj: Vec<f32>) -> Result<(), String> {
        let expected_down = self.hidden_dim * self.rank;
        if down_proj.len() != expected_down {
            return Err(format!(
                "down_proj dimension mismatch: expected {}, got {}",
                expected_down,
                down_proj.len()
            ));
        }

        let expected_up = self.rank * self.hidden_dim;
        if up_proj.len() != expected_up {
            return Err(format!(
                "up_proj dimension mismatch: expected {}, got {}",
                expected_up,
                up_proj.len()
            ));
        }

        self.down_proj = down_proj;
        self.up_proj = up_proj;
        Ok(())
    }
}

/// Base LoRA for background adaptation
///
/// Higher rank (4-16) for more expressive adaptation.
/// Applied hourly during background learning cycles.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BaseLoRA {
    /// LoRA layers
    pub layers: Vec<LoRALayer>,
    /// Rank
    pub rank: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Alpha scaling factor
    pub alpha: f32,
}

/// Single LoRA layer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoRALayer {
    /// Down projection weights
    pub down_proj: Vec<f32>,
    /// Up projection weights
    pub up_proj: Vec<f32>,
    /// Layer index
    pub layer_idx: usize,
}

impl BaseLoRA {
    /// Create new Base LoRA
    pub fn new(hidden_dim: usize, rank: usize, num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|idx| LoRALayer {
                down_proj: vec![0.0; hidden_dim * rank],
                up_proj: vec![0.0; rank * hidden_dim],
                layer_idx: idx,
            })
            .collect();

        Self {
            layers,
            rank,
            hidden_dim,
            alpha: rank as f32,
        }
    }

    /// Forward pass for single layer
    pub fn forward_layer(&self, layer_idx: usize, input: &[f32], output: &mut [f32]) {
        if layer_idx >= self.layers.len() {
            return;
        }

        let layer = &self.layers[layer_idx];
        let scale = self.alpha / self.rank as f32;

        // Down projection
        let mut intermediate = vec![0.0f32; self.rank];
        for (r, inter) in intermediate.iter_mut().enumerate() {
            let offset = r * self.hidden_dim;
            *inter = input
                .iter()
                .zip(&layer.down_proj[offset..offset + self.hidden_dim])
                .map(|(a, b)| a * b)
                .sum();
        }

        // Up projection
        for (i, out) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (r, &inter) in intermediate.iter().enumerate() {
                sum += inter * layer.up_proj[r * self.hidden_dim + i];
            }
            *out += sum * scale;
        }
    }

    /// Merge LoRA weights into model weights (for inference optimization)
    pub fn merge_into(&self, model_weights: &mut [f32], layer_idx: usize) {
        if layer_idx >= self.layers.len() {
            return;
        }

        let layer = &self.layers[layer_idx];
        let scale = self.alpha / self.rank as f32;

        // W' = W + scale * (down @ up)
        // Assumes model_weights is [hidden_dim x hidden_dim]
        for i in 0..self.hidden_dim {
            for j in 0..self.hidden_dim {
                let mut delta = 0.0f32;
                for r in 0..self.rank {
                    delta +=
                        layer.down_proj[i * self.rank + r] * layer.up_proj[r * self.hidden_dim + j];
                }
                model_weights[i * self.hidden_dim + j] += delta * scale;
            }
        }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get total parameter count
    pub fn param_count(&self) -> usize {
        self.layers.len() * (self.hidden_dim * self.rank + self.rank * self.hidden_dim)
    }

    /// Get weights for a specific layer for export (lora_a, lora_b)
    pub fn get_layer_weights(&self, layer_idx: usize) -> Option<(&Vec<f32>, &Vec<f32>)> {
        self.layers
            .get(layer_idx)
            .map(|layer| (&layer.down_proj, &layer.up_proj))
    }
}

/// Combined LoRA engine managing both tiers
#[derive(Clone, Debug)]
pub struct LoRAEngine {
    /// Micro-LoRA for instant adaptation
    pub micro: MicroLoRA,
    /// Base LoRA for background adaptation
    pub base: BaseLoRA,
    /// Whether micro-LoRA is enabled
    pub micro_enabled: bool,
    /// Whether base LoRA is enabled
    pub base_enabled: bool,
}

impl LoRAEngine {
    /// Create new LoRA engine
    pub fn new(hidden_dim: usize, micro_rank: usize, base_rank: usize, num_layers: usize) -> Self {
        Self {
            micro: MicroLoRA::new(hidden_dim, micro_rank.clamp(1, 2)),
            base: BaseLoRA::new(hidden_dim, base_rank, num_layers),
            micro_enabled: true,
            base_enabled: true,
        }
    }

    /// Apply both LoRA tiers
    pub fn forward(&self, layer_idx: usize, input: &[f32], output: &mut [f32]) {
        if self.micro_enabled {
            self.micro.forward(input, output);
        }
        if self.base_enabled && layer_idx < self.base.num_layers() {
            self.base.forward_layer(layer_idx, input, output);
        }
    }

    /// Accumulate micro-LoRA gradient
    pub fn accumulate_micro(&mut self, signal: &LearningSignal) {
        if self.micro_enabled {
            self.micro.accumulate_gradient(signal);
        }
    }

    /// Apply micro-LoRA updates
    pub fn apply_micro(&mut self, learning_rate: f32) {
        if self.micro_enabled {
            self.micro.apply_accumulated(learning_rate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= n);
        v
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn lora_delta(lora: &MicroLoRA, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; x.len()];
        lora.forward(x, &mut out);
        out
    }

    #[test]
    fn test_micro_lora_update_moves_output_toward_feedback() {
        // Every single update must move forward(x) toward the feedback
        // direction for that input. The previous rule ignored the input, so
        // the sign depended on the fixed down projection (17 of 40 here).
        for k in 0..40 {
            let x = unit(64, 100 + k);
            let g = unit(64, 900 + k);
            let mut lora = MicroLoRA::new(64, 2);
            lora.accumulate_gradient(&LearningSignal::with_gradient(x.clone(), g.clone(), 1.0));
            lora.apply_accumulated(0.002);
            let delta = lora_delta(&lora, &x);
            assert!(
                dot(&delta, &g) > 0.0,
                "input {} moved away from feedback",
                k
            );
        }
    }

    #[test]
    fn test_micro_lora_update_depends_on_input() {
        // Two inputs trained toward two different targets must each end up
        // closer to their own target. The previous rule gave both inputs the
        // same update, so input 2 moved away from its own target.
        let (x1, x2, g1, g2) = (unit(64, 1), unit(64, 2), unit(64, 3), unit(64, 4));
        let mut lora = MicroLoRA::new(64, 2);
        for _ in 0..200 {
            lora.accumulate_gradient(&LearningSignal::with_gradient(x1.clone(), g1.clone(), 1.0));
            lora.accumulate_gradient(&LearningSignal::with_gradient(x2.clone(), g2.clone(), 1.0));
            lora.apply_accumulated(0.05);
        }
        let d1 = lora_delta(&lora, &x1);
        let d2 = lora_delta(&lora, &x2);
        assert!(dot(&d1, &g1) > dot(&d1, &g2) + 0.5);
        assert!(dot(&d2, &g2) > dot(&d2, &g1) + 0.5);
    }

    #[test]
    fn test_micro_lora_output_stays_bounded() {
        // Repeated feedback at a high learning rate used to grow `up` without
        // limit (|up| 707 after 5,000 steps at lr 0.1).
        let x = unit(64, 7);
        let g = unit(64, 8);
        let mut lora = MicroLoRA::new(64, 2);
        for _ in 0..5000 {
            lora.accumulate_gradient(&LearningSignal::with_gradient(x.clone(), g.clone(), 1.0));
            lora.apply_accumulated(0.1);
        }
        for k in 0..20 {
            let probe = if k == 0 { x.clone() } else { unit(64, 50 + k) };
            let d = lora_delta(&lora, &probe);
            let gain = dot(&d, &d).sqrt();
            assert!(
                gain.is_finite() && gain <= MICRO_LORA_MAX_GAIN + 1e-4,
                "gain {} above cap",
                gain
            );
        }
    }

    #[test]
    fn test_micro_lora_flat_weights_round_trip() {
        let mut lora = MicroLoRA::new(16, 2);
        lora.accumulate_gradient(&LearningSignal::with_gradient(
            unit(16, 1),
            unit(16, 2),
            1.0,
        ));
        let gradient = lora.take_accumulated().expect("pending gradient");
        assert_eq!(gradient.len(), lora.param_count());
        assert_eq!(lora.pending_updates(), 0);

        lora.apply_gradient(&gradient, 0.1);
        let weights = lora.flat_weights();
        let mut copy = MicroLoRA::new(16, 2);
        copy.set_flat_weights(&weights);
        assert_eq!(copy.flat_weights(), weights);
    }

    #[test]
    fn test_micro_lora_creation() {
        let lora = MicroLoRA::new(256, 1);
        assert_eq!(lora.rank(), 1);
        assert_eq!(lora.hidden_dim(), 256);
        assert_eq!(lora.param_count(), 256 + 256);
    }

    #[test]
    fn test_micro_lora_forward() {
        let lora = MicroLoRA::new(64, 1);
        let input = vec![1.0f32; 64];
        let mut output = vec![0.0f32; 64];

        lora.forward(&input, &mut output);

        // Output should be modified (even if small due to init)
        // With zero-init up_proj, output should still be zero
        let sum: f32 = output.iter().sum();
        assert!(
            sum.abs() < 1e-6,
            "Expected ~0 with zero up_proj, got {}",
            sum
        );
    }

    #[test]
    fn test_micro_lora_learning() {
        let mut lora = MicroLoRA::new(64, 1);

        let signal = LearningSignal::with_gradient(vec![0.1; 64], vec![0.5; 64], 0.8);

        lora.accumulate_gradient(&signal);
        assert_eq!(lora.pending_updates(), 1);

        lora.apply_accumulated(0.01);
        assert_eq!(lora.pending_updates(), 0);

        // Now forward should produce non-zero output
        let input = vec![1.0f32; 64];
        let mut output = vec![0.0f32; 64];
        lora.forward(&input, &mut output);

        let sum: f32 = output.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Expected non-zero output after learning");
    }

    #[test]
    fn test_base_lora() {
        let lora = BaseLoRA::new(64, 4, 12);
        assert_eq!(lora.num_layers(), 12);
        assert_eq!(lora.rank, 4);
    }

    #[test]
    fn test_lora_engine() {
        let mut engine = LoRAEngine::new(64, 1, 4, 12);

        let signal = LearningSignal::with_gradient(vec![0.1; 64], vec![0.5; 64], 0.9);

        engine.accumulate_micro(&signal);
        engine.apply_micro(0.01);

        let input = vec![1.0f32; 64];
        let mut output = vec![0.0f32; 64];
        engine.forward(0, &input, &mut output);
    }

    #[test]
    #[should_panic(expected = "MicroLoRA rank must be 1-2")]
    fn test_invalid_rank() {
        MicroLoRA::new(64, 5);
    }

    #[test]
    fn test_set_weights_valid() {
        let mut lora = MicroLoRA::new(64, 2);
        let down = vec![1.0f32; 64 * 2];
        let up = vec![0.5f32; 2 * 64];

        let result = lora.set_weights(down.clone(), up.clone());
        assert!(result.is_ok());

        let (got_down, got_up) = lora.get_weights();
        assert_eq!(got_down, &down);
        assert_eq!(got_up, &up);
    }

    #[test]
    fn test_set_weights_wrong_down_dim() {
        let mut lora = MicroLoRA::new(64, 2);
        let wrong_down = vec![1.0f32; 64 * 3];
        let up = vec![0.5f32; 2 * 64];

        let result = lora.set_weights(wrong_down, up);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("down_proj dimension mismatch"));
    }

    #[test]
    fn test_set_weights_wrong_up_dim() {
        let mut lora = MicroLoRA::new(64, 2);
        let down = vec![1.0f32; 64 * 2];
        let wrong_up = vec![0.5f32; 3 * 64];

        let result = lora.set_weights(down, wrong_up);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("up_proj dimension mismatch"));
    }
}
