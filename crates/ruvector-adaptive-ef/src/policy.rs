//! Adaptive search policies for HNSW ef-search tuning.

const EF_MIN: u32 = 8;
const EF_MAX: u32 = 512;

/// Core trait: recommend an ef before each query, receive feedback after.
pub trait SearchPolicy: Send + Sync {
    fn recommend_ef(&mut self, latency_budget_us: u64) -> u32;
    fn observe(&mut self, latency_us: u64, recall: f32, ef_used: u32);
    fn name(&self) -> &str;
    fn current_ef(&self) -> u32;
}

// ── 1. Baseline: Fixed ────────────────────────────────────────────────────────

pub struct FixedPolicy { ef: u32 }

impl FixedPolicy {
    pub fn new(ef: u32) -> Self { Self { ef: ef.clamp(EF_MIN, EF_MAX) } }
}

impl SearchPolicy for FixedPolicy {
    fn recommend_ef(&mut self, _budget: u64) -> u32 { self.ef }
    fn observe(&mut self, _lat: u64, _rec: f32, _ef: u32) {}
    fn name(&self) -> &str { "Fixed" }
    fn current_ef(&self) -> u32 { self.ef }
}

// ── 2. EWMA greedy hill-climb ─────────────────────────────────────────────────

/// Exponentially weighted moving average of recent latency; adjusts ef via a
/// simple hill-climbing step triggered on each `recommend_ef` call.
pub struct EwmaGreedy {
    ef: u32,
    ewma_us: f64,
    alpha: f64,
    step: u32,
    observations: u64,
}

impl EwmaGreedy {
    pub fn new(init_ef: u32, alpha: f64) -> Self {
        Self {
            ef: init_ef.clamp(EF_MIN, EF_MAX),
            ewma_us: 0.0,
            alpha: alpha.clamp(0.01, 0.9),
            step: 8,
            observations: 0,
        }
    }
}

impl SearchPolicy for EwmaGreedy {
    fn recommend_ef(&mut self, budget_us: u64) -> u32 {
        if self.observations == 0 { return self.ef; }
        let budget = budget_us as f64;
        let slack = (budget - self.ewma_us) / budget;
        if slack > 0.20 {
            self.ef = (self.ef + self.step).min(EF_MAX);
        } else if slack < -0.10 {
            self.ef = self.ef.saturating_sub(self.step).max(EF_MIN);
        }
        self.ef
    }

    fn observe(&mut self, latency_us: u64, _recall: f32, _ef_used: u32) {
        self.observations += 1;
        if self.observations == 1 {
            self.ewma_us = latency_us as f64;
        } else {
            self.ewma_us = self.alpha * latency_us as f64 + (1.0 - self.alpha) * self.ewma_us;
        }
    }

    fn name(&self) -> &str { "EwmaGreedy" }
    fn current_ef(&self) -> u32 { self.ef }
}

// ── 3. ε-greedy multi-armed bandit ───────────────────────────────────────────

pub struct BanditPolicy {
    arms: Vec<u32>,
    rewards: Vec<f64>,
    counts: Vec<u64>,
    epsilon: f64,
    last_arm: usize,
    rng: u64,
    total_steps: u64,
}

impl BanditPolicy {
    pub fn new(epsilon: f64) -> Self {
        let arms: Vec<u32> = vec![8, 16, 32, 48, 64, 96, 128, 192, 256];
        let n = arms.len();
        Self {
            arms,
            rewards: vec![0.5; n],
            counts: vec![0; n],
            epsilon,
            last_arm: 4, // start at ef=64
            rng: 0x517c_c1b7_2722_0a95,
            total_steps: 0,
        }
    }

    fn lcg(&mut self) -> f64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 11) as f64 / (u64::MAX >> 11) as f64
    }

    fn best_arm(&self) -> usize {
        self.rewards
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(4)
    }
}

impl SearchPolicy for BanditPolicy {
    fn recommend_ef(&mut self, _budget: u64) -> u32 {
        let eps = (self.epsilon * 200.0 / (self.total_steps as f64 + 200.0)).max(0.05);
        self.last_arm = if self.lcg() < eps {
            let n = self.arms.len();
            (self.lcg() * n as f64) as usize % n
        } else {
            self.best_arm()
        };
        self.total_steps += 1;
        self.arms[self.last_arm]
    }

    fn observe(&mut self, _latency_us: u64, recall: f32, ef_used: u32) {
        let arm_idx = self
            .arms
            .iter()
            .position(|&e| e == ef_used)
            .unwrap_or(self.last_arm);
        self.counts[arm_idx] += 1;
        let n = self.counts[arm_idx] as f64;
        // Incremental mean of recall-as-reward
        self.rewards[arm_idx] += (recall as f64 - self.rewards[arm_idx]) / n;
    }

    fn name(&self) -> &str { "Bandit" }
    fn current_ef(&self) -> u32 { self.arms[self.last_arm] }
}

// ── 4. PID controller ─────────────────────────────────────────────────────────

pub struct PidController {
    ef: f64,
    integral: f64,
    prev_error: f64,
    kp: f64,
    ki: f64,
    kd: f64,
    last_budget: u64,
    steps: u64,
}

impl PidController {
    pub fn new(init_ef: u32) -> Self {
        Self {
            ef: init_ef.clamp(EF_MIN, EF_MAX) as f64,
            integral: 0.0,
            prev_error: 0.0,
            kp: 0.30,
            ki: 0.01,
            kd: 0.05,
            last_budget: 200,
            steps: 0,
        }
    }

    pub fn with_gains(mut self, kp: f64, ki: f64, kd: f64) -> Self {
        self.kp = kp; self.ki = ki; self.kd = kd; self
    }
}

impl SearchPolicy for PidController {
    fn recommend_ef(&mut self, budget_us: u64) -> u32 {
        self.last_budget = budget_us;
        self.ef.round().clamp(EF_MIN as f64, EF_MAX as f64) as u32
    }

    fn observe(&mut self, latency_us: u64, _recall: f32, _ef_used: u32) {
        self.steps += 1;
        if self.steps < 4 { return; } // warmup
        let budget = self.last_budget as f64;
        let error = (latency_us as f64 - budget) / budget;
        self.integral = (self.integral + error).clamp(-10.0, 10.0);
        let derivative = error - self.prev_error;
        self.prev_error = error;
        let correction = self.kp * error + self.ki * self.integral + self.kd * derivative;
        // positive correction (too slow) → decrease ef
        self.ef -= correction * 32.0;
        self.ef = self.ef.clamp(EF_MIN as f64, EF_MAX as f64);
    }

    fn name(&self) -> &str { "PID" }
    fn current_ef(&self) -> u32 { self.ef.round() as u32 }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_stays_constant() {
        let mut p = FixedPolicy::new(64);
        assert_eq!(p.recommend_ef(100), 64);
        p.observe(500, 0.9, 64);
        assert_eq!(p.recommend_ef(100), 64);
    }

    #[test]
    fn ewma_decreases_under_pressure() {
        let mut p = EwmaGreedy::new(128, 0.5);
        for _ in 0..30 {
            p.observe(9999, 0.95, 128);
        }
        let ef = p.recommend_ef(100);
        assert!(ef < 128, "ef should have decreased; got {ef}");
    }

    #[test]
    fn ewma_increases_with_slack() {
        let mut p = EwmaGreedy::new(16, 0.5);
        for _ in 0..20 {
            p.observe(5, 0.99, 16);
        }
        let ef = p.recommend_ef(10_000);
        assert!(ef > 16, "ef should increase when budget is generous; got {ef}");
    }

    #[test]
    fn bandit_explores_multiple_arms() {
        let mut p = BanditPolicy::new(0.9);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..300 {
            let ef = p.recommend_ef(200);
            p.observe(100, 0.9, ef);
            seen.insert(ef);
        }
        assert!(seen.len() >= 4, "Bandit should explore ≥4 arms; saw {}", seen.len());
    }

    #[test]
    fn pid_reduces_ef_when_too_slow() {
        let mut p = PidController::new(256);
        for _ in 0..30 {
            let _ef = p.recommend_ef(100);
            p.observe(1000, 0.9, 256);
        }
        assert!(p.current_ef() < 256, "PID should reduce ef when overbudget");
    }

    #[test]
    fn pid_increases_ef_when_fast() {
        let mut p = PidController::new(32);
        for _ in 0..30 {
            let _ef = p.recommend_ef(500);
            p.observe(10, 0.9, 32);
        }
        assert!(p.current_ef() > 32, "PID should increase ef when well under budget");
    }
}
