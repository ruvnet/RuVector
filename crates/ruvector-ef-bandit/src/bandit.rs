//! UCB1 and ε-greedy multi-armed bandit policies for ef-search adaptation.

use rand::Rng;

/// State of a single bandit arm (one ef candidate).
#[derive(Debug, Clone)]
pub struct Arm {
    /// The ef value this arm represents.
    pub ef: usize,
    /// Number of times this arm has been pulled.
    pub n_pulls: u64,
    /// Cumulative reward received.
    cumulative_reward: f64,
}

impl Arm {
    pub fn new(ef: usize) -> Self {
        Self {
            ef,
            n_pulls: 0,
            cumulative_reward: 0.0,
        }
    }

    /// Sample-mean reward estimate.
    pub fn mean_reward(&self) -> f64 {
        if self.n_pulls == 0 {
            0.0
        } else {
            self.cumulative_reward / self.n_pulls as f64
        }
    }

    pub fn update(&mut self, reward: f64) {
        self.n_pulls += 1;
        self.cumulative_reward += reward;
    }
}

/// UCB1 bandit: Upper Confidence Bound policy.
///
/// Selects the arm maximising Q(a) + c * sqrt(ln(N) / n(a)).
/// Each arm is pulled once before exploitation begins.
pub struct Ucb1Bandit {
    arms: Vec<Arm>,
    total_pulls: u64,
    /// Exploration constant c (default √2).
    exploration_c: f64,
    /// Round-robin index used during the initial exploration phase.
    init_cursor: usize,
}

impl Ucb1Bandit {
    pub fn new(ef_values: &[usize], exploration_c: f64) -> Self {
        assert!(!ef_values.is_empty(), "bandit needs at least one arm");
        Self {
            arms: ef_values.iter().copied().map(Arm::new).collect(),
            total_pulls: 0,
            exploration_c,
            init_cursor: 0,
        }
    }

    /// Select the arm index and its ef value.
    pub fn select(&mut self) -> (usize, usize) {
        // Pull each arm once before applying UCB1.
        if self.init_cursor < self.arms.len() {
            let idx = self.init_cursor;
            self.init_cursor += 1;
            return (idx, self.arms[idx].ef);
        }

        let ln_n = (self.total_pulls as f64).ln();
        let best = self
            .arms
            .iter()
            .enumerate()
            .map(|(i, arm)| {
                let bonus = self.exploration_c * (ln_n / arm.n_pulls as f64).sqrt();
                (i, arm.mean_reward() + bonus)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        (best.0, self.arms[best.0].ef)
    }

    /// Record a reward for the given arm.
    pub fn update(&mut self, arm_idx: usize, reward: f64) {
        self.arms[arm_idx].update(reward);
        self.total_pulls += 1;
    }

    /// Index and ef of the arm with the highest mean reward (exploitation only).
    pub fn best_arm(&self) -> (usize, usize) {
        let best = self
            .arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.mean_reward().partial_cmp(&b.mean_reward()).unwrap())
            .unwrap();
        (best.0, best.1.ef)
    }

    pub fn arms(&self) -> &[Arm] {
        &self.arms
    }

    pub fn total_pulls(&self) -> u64 {
        self.total_pulls
    }

    /// Bytes used by the bandit state.
    pub fn memory_bytes(&self) -> usize {
        // Each Arm: ef (8B) + n_pulls (8B) + reward (8B) = 24B + overhead ≈ 32B
        self.arms.len() * 32 + std::mem::size_of::<Self>()
    }
}

/// ε-Greedy bandit with exponential epsilon decay.
///
/// With probability ε, selects a random arm (exploration).
/// With probability 1-ε, selects the arm with the highest mean reward.
/// ε decays multiplicatively after every query.
pub struct EpsilonGreedyBandit {
    arms: Vec<Arm>,
    epsilon: f64,
    decay: f64,
    total_pulls: u64,
}

impl EpsilonGreedyBandit {
    pub fn new(ef_values: &[usize], epsilon_init: f64, decay: f64) -> Self {
        assert!(!ef_values.is_empty(), "bandit needs at least one arm");
        assert!(
            (0.0..=1.0).contains(&epsilon_init),
            "epsilon must be in [0, 1]"
        );
        assert!((0.0..=1.0).contains(&decay), "decay must be in [0, 1]");
        Self {
            arms: ef_values.iter().copied().map(Arm::new).collect(),
            epsilon: epsilon_init,
            decay,
            total_pulls: 0,
        }
    }

    /// Select an arm index and ef value, updating epsilon.
    pub fn select(&mut self, rng: &mut impl Rng) -> (usize, usize) {
        let idx = if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..self.arms.len())
        } else {
            self.arms
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.mean_reward().partial_cmp(&b.mean_reward()).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        self.epsilon *= self.decay;
        (idx, self.arms[idx].ef)
    }

    pub fn update(&mut self, arm_idx: usize, reward: f64) {
        self.arms[arm_idx].update(reward);
        self.total_pulls += 1;
    }

    pub fn best_arm(&self) -> (usize, usize) {
        let best = self
            .arms
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.mean_reward().partial_cmp(&b.mean_reward()).unwrap())
            .unwrap();
        (best.0, best.1.ef)
    }

    pub fn arms(&self) -> &[Arm] {
        &self.arms
    }

    pub fn current_epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn total_pulls(&self) -> u64 {
        self.total_pulls
    }

    pub fn memory_bytes(&self) -> usize {
        self.arms.len() * 32 + std::mem::size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn ucb1_explores_all_arms_first() {
        let mut bandit = Ucb1Bandit::new(&[10, 50, 100], 1.414);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..3 {
            let (_, ef) = bandit.select();
            bandit.update(bandit.arms.iter().position(|a| a.ef == ef).unwrap(), 0.5);
            seen.insert(ef);
        }
        assert_eq!(
            seen.len(),
            3,
            "UCB1 must pull each arm once before exploiting"
        );
    }

    #[test]
    fn ucb1_converges_to_best_arm() {
        // Arm 1 (ef=50) always gives reward 0.9; others give 0.5.
        let mut bandit = Ucb1Bandit::new(&[10, 50, 100], 1.414);
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..200 {
            let (idx, ef) = bandit.select();
            let reward = if ef == 50 { 0.9 } else { 0.5 };
            bandit.update(idx, reward + rng.gen::<f64>() * 0.05);
        }
        let (_, best_ef) = bandit.best_arm();
        assert_eq!(best_ef, 50, "UCB1 should converge to highest-reward arm");
    }

    #[test]
    fn epsilon_greedy_decays() {
        let mut bandit = EpsilonGreedyBandit::new(&[10, 50, 100], 0.5, 0.99);
        let mut rng = SmallRng::seed_from_u64(7);
        let eps_before = bandit.current_epsilon();
        bandit.select(&mut rng);
        bandit.update(0, 0.8);
        let eps_after = bandit.current_epsilon();
        assert!(eps_after < eps_before, "epsilon must decay after each pull");
    }

    #[test]
    fn epsilon_greedy_converges_to_best_arm() {
        let mut bandit = EpsilonGreedyBandit::new(&[10, 50, 100], 0.30, 0.995);
        let mut rng = SmallRng::seed_from_u64(99);
        for _ in 0..500 {
            let (idx, ef) = bandit.select(&mut rng);
            let reward = if ef == 100 { 0.95 } else { 0.6 };
            bandit.update(idx, reward);
        }
        let (_, best_ef) = bandit.best_arm();
        assert_eq!(
            best_ef, 100,
            "ε-greedy should converge to highest-reward arm"
        );
    }
}
