use super::*;

/// Anytime-valid paired test by betting (ADR-004 gate 2; e-processes,
/// arXiv:2606.00878 and arXiv:2501.03982). For each *discordant* validation
/// pair — one model right, the other wrong — wealth updates
/// `W ← W · (1 + λ·(X − ½))`, where `X = 1` iff the champion won the pair.
/// Concordant pairs carry no paired information and are skipped (McNemar's
/// structure). Under the null "no improvement", `X ~ Bernoulli(½)` so `E[W]`
/// stays 1: `W` is a non-negative martingale and Ville's inequality gives
/// `P(sup_t W_t ≥ 1/α) ≤ α`. The rejection therefore latches on the first
/// crossing of `1/α` and never un-latches.
#[derive(Debug, Clone)]
pub struct PairedSequentialTest {
    alpha: f32,
    lambda: f32,
    wealth: f64,
    max_wealth: f64,
    n_champion_wins: u32,
    n_baseline_wins: u32,
    rejected: bool,
    n_discordant_at_rejection: Option<u32>,
}

impl PairedSequentialTest {
    /// `alpha` is the type-I bound (default 0.05); `lambda` is the betting
    /// fraction. `lambda` must lie in `(0, 2)` — outside it wealth can go
    /// negative and the martingale argument breaks — so it is clamped into a
    /// safe interior band.
    #[must_use]
    pub fn new(alpha: f32, lambda: f32) -> Self {
        let alpha = alpha.clamp(1e-6, 0.5);
        let lambda = lambda.clamp(0.01, 1.99);
        Self {
            alpha,
            lambda,
            wealth: 1.0,
            max_wealth: 1.0,
            n_champion_wins: 0,
            n_baseline_wins: 0,
            rejected: false,
            n_discordant_at_rejection: None,
        }
    }

    /// α = 0.05, λ = 0.5 (wealth factor in `{0.75, 1.25}` — never negative).
    #[must_use]
    pub fn standard() -> Self {
        Self::new(0.05, 0.5)
    }

    fn threshold(&self) -> f64 {
        1.0 / self.alpha as f64
    }

    /// Feed one paired outcome. Returns `true` once the rejection has latched.
    pub fn update(&mut self, baseline_correct: bool, champion_correct: bool) -> bool {
        if baseline_correct == champion_correct {
            return self.rejected; // concordant: no information
        }
        let champion_won = champion_correct && !baseline_correct;
        let x = if champion_won { 1.0 } else { 0.0 };
        if champion_won {
            self.n_champion_wins += 1;
        } else {
            self.n_baseline_wins += 1;
        }
        self.wealth *= 1.0 + self.lambda as f64 * (x - 0.5);
        if self.wealth > self.max_wealth {
            self.max_wealth = self.wealth;
        }
        if !self.rejected && self.max_wealth >= self.threshold() {
            self.rejected = true;
            self.n_discordant_at_rejection = Some(self.n_champion_wins + self.n_baseline_wins);
        }
        self.rejected
    }

    /// Feed a whole stream of `(baseline_correct, champion_correct)` pairs.
    pub fn update_all(&mut self, pairs: &[(bool, bool)]) -> bool {
        for &(b, c) in pairs {
            self.update(b, c);
        }
        self.rejected
    }

    /// `true` once wealth has ever reached `1/α` (latched, anytime-valid).
    #[must_use]
    pub fn rejected(&self) -> bool {
        self.rejected
    }

    /// Plain paired counts `(champion_wins, baseline_wins)` — the McNemar cells.
    #[must_use]
    pub fn discordant_counts(&self) -> (u32, u32) {
        (self.n_champion_wins, self.n_baseline_wins)
    }

    #[must_use]
    pub fn statistic(&self) -> TestStatistic {
        TestStatistic {
            alpha: self.alpha,
            lambda: self.lambda,
            wealth: self.wealth,
            max_wealth: self.max_wealth,
            threshold: self.threshold(),
            n_champion_wins: self.n_champion_wins,
            n_baseline_wins: self.n_baseline_wins,
            rejected: self.rejected,
            n_discordant_at_rejection: self.n_discordant_at_rejection,
        }
    }
}
