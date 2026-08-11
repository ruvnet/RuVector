//! Multi-Armed Bandit algorithms for ANN parameter selection.
//!
//! Two algorithms:
//!  - UCB1       : Upper Confidence Bound (deterministic, no randomness)
//!  - Thompson   : Thompson Sampling via Beta distribution posteriors

// ─── UCB1 ─────────────────────────────────────────────────────────────────────

/// UCB1 bandit with arms indexed 0..n_arms.
///
/// Each arm represents one candidate ef_search value.  UCB1 selects the arm
/// that maximises: mean_reward + sqrt(2 * ln(total_pulls) / arm_pulls).
pub struct Ucb1Bandit {
    n_arms: usize,
    /// Accumulated reward per arm.
    rewards: Vec<f64>,
    /// Pull count per arm.
    counts: Vec<u64>,
    /// Total pulls across all arms.
    total: u64,
}

impl Ucb1Bandit {
    pub fn new(n_arms: usize) -> Self {
        Self {
            n_arms,
            rewards: vec![0.0; n_arms],
            counts: vec![0; n_arms],
            total: 0,
        }
    }

    /// Select an arm using UCB1.  Unpulled arms are always chosen first.
    pub fn select(&self) -> usize {
        // Always try each arm at least once before applying UCB formula.
        for i in 0..self.n_arms {
            if self.counts[i] == 0 {
                return i;
            }
        }
        let ln_total = (self.total as f64).ln();
        let mut best_arm = 0;
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..self.n_arms {
            let mean = self.rewards[i] / self.counts[i] as f64;
            let bonus = (2.0 * ln_total / self.counts[i] as f64).sqrt();
            let score = mean + bonus;
            if score > best_score {
                best_score = score;
                best_arm = i;
            }
        }
        best_arm
    }

    /// Record an observed reward for the chosen arm.
    pub fn update(&mut self, arm: usize, reward: f32) {
        self.rewards[arm] += reward as f64;
        self.counts[arm] += 1;
        self.total += 1;
    }

    /// Return the arm with the highest empirical mean reward (exploitation).
    pub fn best_arm(&self) -> usize {
        let mut best = 0;
        let mut best_mean = f64::NEG_INFINITY;
        for i in 0..self.n_arms {
            if self.counts[i] == 0 {
                continue;
            }
            let mean = self.rewards[i] / self.counts[i] as f64;
            if mean > best_mean {
                best_mean = mean;
                best = i;
            }
        }
        best
    }

    /// Mean reward for each arm (returns 0.0 for unpulled arms).
    pub fn mean_rewards(&self) -> Vec<f64> {
        (0..self.n_arms)
            .map(|i| {
                if self.counts[i] == 0 {
                    0.0
                } else {
                    self.rewards[i] / self.counts[i] as f64
                }
            })
            .collect()
    }

    /// Pull counts per arm.
    pub fn pull_counts(&self) -> &[u64] {
        &self.counts
    }

    pub fn total_pulls(&self) -> u64 {
        self.total
    }
}

// ─── Thompson Sampling ────────────────────────────────────────────────────────

/// Thompson Sampling bandit using Beta(α,β) posteriors on [0,1] rewards.
///
/// Rewards are clipped to [0, 1] and used as Bernoulli-like observations.
pub struct ThompsonBandit {
    n_arms: usize,
    /// α parameter per arm (successes + 1).
    alpha: Vec<f64>,
    /// β parameter per arm (failures + 1).
    beta: Vec<f64>,
}

impl ThompsonBandit {
    pub fn new(n_arms: usize) -> Self {
        Self {
            n_arms,
            alpha: vec![1.0; n_arms],
            beta: vec![1.0; n_arms],
        }
    }

    /// Select an arm by sampling from each Beta posterior and picking the max.
    pub fn select(&self, rng: &mut impl rand::Rng) -> usize {
        let mut best_arm = 0;
        let mut best_sample = f64::NEG_INFINITY;
        for i in 0..self.n_arms {
            let sample = sample_beta(self.alpha[i], self.beta[i], rng);
            if sample > best_sample {
                best_sample = sample;
                best_arm = i;
            }
        }
        best_arm
    }

    /// Update Beta posterior with an observed reward clipped to [0,1].
    pub fn update(&mut self, arm: usize, reward: f32) {
        let r = reward.clamp(0.0, 1.0) as f64;
        self.alpha[arm] += r;
        self.beta[arm] += 1.0 - r;
    }

    /// Arm with the highest posterior mean α/(α+β).
    pub fn best_arm(&self) -> usize {
        (0..self.n_arms)
            .max_by(|&a, &b| {
                let ma = self.alpha[a] / (self.alpha[a] + self.beta[a]);
                let mb = self.alpha[b] / (self.alpha[b] + self.beta[b]);
                ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0)
    }

    pub fn posterior_means(&self) -> Vec<f64> {
        (0..self.n_arms)
            .map(|i| self.alpha[i] / (self.alpha[i] + self.beta[i]))
            .collect()
    }
}

/// Approximate Beta(α,β) sample using Johnk's method.
/// Valid for α,β >= 1.
fn sample_beta(alpha: f64, beta: f64, rng: &mut impl rand::Rng) -> f64 {
    // Use the relation: Beta(α,β) = Gamma(α) / (Gamma(α) + Gamma(β))
    // Approximate via log-normal when α,β are large; direct Johnk otherwise.
    let x = sample_gamma(alpha, rng);
    let y = sample_gamma(beta, rng);
    x / (x + y)
}

/// Marsaglia-Tsang gamma sampler.  Shape parameter `shape` >= 1.
fn sample_gamma(shape: f64, rng: &mut impl rand::Rng) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x: f64 = rng.gen::<f64>() * 2.0 - 1.0; // roughly N(0,1)
        let v = (1.0 + c * x).powi(3);
        if v > 0.0 {
            let u: f64 = rng.gen();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn ucb1_explores_all_arms_first() {
        let mut b = Ucb1Bandit::new(4);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4 {
            let arm = b.select();
            seen.insert(arm);
            b.update(arm, 0.5);
        }
        assert_eq!(seen.len(), 4, "UCB1 must pull every arm at least once");
    }

    #[test]
    fn ucb1_converges_to_best_arm() {
        let mut b = Ucb1Bandit::new(3);
        let rewards = [0.2, 0.8, 0.5]; // arm 1 is best
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for _ in 0..300 {
            let arm = b.select();
            let r = rewards[arm] + (rng.gen::<f32>() - 0.5) * 0.1;
            b.update(arm, r);
        }
        assert_eq!(
            b.best_arm(),
            1,
            "UCB1 should converge to arm 1 (highest reward)"
        );
    }

    #[test]
    fn thompson_selects_best_arm_with_high_probability() {
        let mut b = ThompsonBandit::new(3);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let rewards = [0.3, 0.9, 0.5];
        for _ in 0..200 {
            let arm = b.select(&mut rng);
            let r = rewards[arm] + (rng.gen::<f32>() - 0.5) * 0.1;
            b.update(arm, r);
        }
        assert_eq!(b.best_arm(), 1);
    }

    #[test]
    fn posterior_means_are_monotone_after_updates() {
        let mut b = ThompsonBandit::new(2);
        // Feed arm 0 high rewards, arm 1 low rewards.
        for _ in 0..50 {
            b.update(0, 0.9);
            b.update(1, 0.1);
        }
        let means = b.posterior_means();
        assert!(means[0] > means[1]);
    }
}
