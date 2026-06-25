//! Logits sampling utilities shared by the recurrent-depth models and the
//! backend. Operates on a host-side `Vec<f32>` of vocab logits so it is
//! independent of the tensor backend.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Sampling configuration. `temperature == 0` selects greedy (argmax).
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    /// Keep only the top-`k` logits (`0` disables).
    pub top_k: usize,
    /// Nucleus sampling cumulative-probability cutoff in `(0, 1]` (`1.0` disables).
    pub top_p: f32,
    /// Penalty (>1 discourages) applied to recently generated tokens.
    pub repetition_penalty: f32,
    /// Number of recent tokens the repetition penalty considers.
    pub repetition_window: usize,
    pub seed: u64,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            repetition_window: 64,
            seed: 42,
        }
    }
}

impl SamplingConfig {
    /// Deterministic greedy decoding.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 0,
            seed: 0,
        }
    }
}

/// Stateful sampler (owns the RNG so successive draws differ).
pub struct Sampler {
    cfg: SamplingConfig,
    rng: StdRng,
}

impl Sampler {
    pub fn new(cfg: SamplingConfig) -> Self {
        let rng = StdRng::seed_from_u64(cfg.seed);
        Self { cfg, rng }
    }

    /// Sample from pre-sorted top-k candidates supplied by the caller.
    ///
    /// `sorted_values[i]` is the logit for `sorted_indices[i]`, sorted
    /// **descending**. This avoids the O(vocab) `logits.to_vec()` copy: only
    /// `2 * top_k * 4` bytes need to be transferred from GPU vs `vocab * 4`.
    ///
    /// Repetition penalty is applied in-place to any candidate whose token id
    /// appears in `recent`; tokens outside the top-k cannot be sampled so
    /// excluding them from penalty is correct.
    pub fn sample_topk(
        &mut self,
        sorted_values: &[f32],
        sorted_indices: &[u32],
        recent: &[u32],
    ) -> u32 {
        if sorted_indices.is_empty() {
            return 0;
        }

        // Fast greedy with no rep penalty: first element is argmax.
        if self.cfg.temperature <= 0.0
            && ((self.cfg.repetition_penalty - 1.0).abs() <= f32::EPSILON
                || self.cfg.repetition_window == 0)
        {
            return sorted_indices[0];
        }

        // Apply rep penalty to candidates that appear in the recent window.
        let window_start = recent.len().saturating_sub(self.cfg.repetition_window);
        let recent_w = &recent[window_start..];
        let apply_pen = (self.cfg.repetition_penalty - 1.0).abs() > f32::EPSILON
            && self.cfg.repetition_window > 0;

        let mut cand: Vec<(u32, f32)> = sorted_values
            .iter()
            .zip(sorted_indices.iter())
            .map(|(&v, &id)| {
                let logit = if apply_pen && recent_w.contains(&id) {
                    if v > 0.0 {
                        v / self.cfg.repetition_penalty
                    } else {
                        v * self.cfg.repetition_penalty
                    }
                } else {
                    v
                };
                (id, logit)
            })
            .collect();

        // Greedy after rep penalty.
        if self.cfg.temperature <= 0.0 {
            return cand
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|&(id, _)| id)
                .unwrap_or(0);
        }

        // Temperature scaling + re-sort (pen may have changed order).
        let inv_t = 1.0 / self.cfg.temperature;
        for (_, l) in cand.iter_mut() {
            *l *= inv_t;
        }
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Softmax over candidates (already top-k truncated by caller).
        let max_l = cand.first().map(|&(_, l)| l).unwrap_or(0.0);
        let mut probs: Vec<f32> = cand.iter().map(|&(_, l)| (l - max_l).exp()).collect();
        let sum: f32 = probs.iter().sum::<f32>().max(1e-9);
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top-p (nucleus).
        if self.cfg.top_p < 1.0 {
            let mut cum = 0.0;
            let mut cutoff = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum >= self.cfg.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            cand.truncate(cutoff);
            probs.truncate(cutoff);
            let s: f32 = probs.iter().sum::<f32>().max(1e-9);
            for p in probs.iter_mut() {
                *p /= s;
            }
        }

        // Multinomial draw.
        let r: f32 = self.rng.gen::<f32>();
        let mut acc = 0.0;
        for (&(id, _), p) in cand.iter().zip(probs.iter()) {
            acc += *p;
            if r <= acc {
                return id;
            }
        }
        cand.last().map(|&(id, _)| id).unwrap_or(0)
    }

    /// Sample a token id from `logits`, applying repetition penalty over
    /// `recent` tokens, temperature, top-k and top-p filtering.
    pub fn sample(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
        // Zero-alloc fast path: greedy with no repetition penalty needs no copy.
        if self.cfg.temperature <= 0.0
            && ((self.cfg.repetition_penalty - 1.0).abs() <= f32::EPSILON
                || self.cfg.repetition_window == 0)
        {
            return argmax(logits);
        }

        let mut work = logits.to_vec();

        // Repetition penalty.
        if (self.cfg.repetition_penalty - 1.0).abs() > f32::EPSILON
            && self.cfg.repetition_window > 0
        {
            let start = recent.len().saturating_sub(self.cfg.repetition_window);
            for &tok in &recent[start..] {
                if let Some(l) = work.get_mut(tok as usize) {
                    *l = if *l > 0.0 {
                        *l / self.cfg.repetition_penalty
                    } else {
                        *l * self.cfg.repetition_penalty
                    };
                }
            }
        }

        // Greedy shortcut.
        if self.cfg.temperature <= 0.0 {
            return argmax(&work);
        }

        // Temperature scaling.
        let inv_t = 1.0 / self.cfg.temperature;
        for l in work.iter_mut() {
            *l *= inv_t;
        }

        // Candidate set: (index, logit). Top-k filter first.
        let mut cand: Vec<(usize, f32)> = work.iter().copied().enumerate().collect();
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        if self.cfg.top_k > 0 && self.cfg.top_k < cand.len() {
            cand.truncate(self.cfg.top_k);
        }

        // Softmax over candidates.
        let max_l = cand.first().map(|&(_, l)| l).unwrap_or(0.0);
        let mut probs: Vec<f32> = cand.iter().map(|&(_, l)| (l - max_l).exp()).collect();
        let sum: f32 = probs.iter().sum::<f32>().max(1e-9);
        for p in probs.iter_mut() {
            *p /= sum;
        }

        // Top-p (nucleus): keep the smallest prefix whose mass >= top_p.
        if self.cfg.top_p < 1.0 {
            let mut cum = 0.0;
            let mut cutoff = probs.len();
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum >= self.cfg.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            cand.truncate(cutoff);
            probs.truncate(cutoff);
            let s: f32 = probs.iter().sum::<f32>().max(1e-9);
            for p in probs.iter_mut() {
                *p /= s;
            }
        }

        // Multinomial draw.
        let r: f32 = self.rng.gen::<f32>();
        let mut acc = 0.0;
        for (idx, p) in cand.iter().zip(probs.iter()) {
            acc += *p;
            if r <= acc {
                return idx.0 as u32;
            }
        }
        cand.last().map(|&(i, _)| i as u32).unwrap_or(0)
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(SamplingConfig::greedy());
        let logits = vec![0.1, 0.2, 5.0, 0.3];
        assert_eq!(s.sample(&logits, &[]), 2);
    }

    #[test]
    fn temperature_sampling_is_seeded_deterministic() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            seed: 7,
            ..SamplingConfig::default()
        };
        let logits = vec![1.0, 2.0, 3.0, 0.5, 0.2];
        let a = Sampler::new(cfg.clone()).sample(&logits, &[]);
        let b = Sampler::new(cfg).sample(&logits, &[]);
        assert_eq!(a, b, "same seed must yield the same draw");
    }

    #[test]
    fn top_k_one_is_argmax() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 0,
            seed: 1,
        };
        let logits = vec![0.1, 9.0, 0.2];
        assert_eq!(Sampler::new(cfg).sample(&logits, &[]), 1);
    }

    #[test]
    fn repetition_penalty_demotes_recent() {
        let cfg = SamplingConfig {
            temperature: 0.0, // greedy after penalty
            repetition_penalty: 10.0,
            repetition_window: 8,
            ..SamplingConfig::default()
        };
        // Token 0 is the natural argmax; penalizing it should flip to token 1.
        let logits = vec![2.0, 1.9, 0.0];
        let pick = Sampler::new(cfg).sample(&logits, &[0]);
        assert_eq!(pick, 1);
    }
}
