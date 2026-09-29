//! Value-of-information (VoI) routing primitive (PIR WP28, ADR-331).
//!
//! Routing as information economics, after "Pandora's AI Model Routing Box:
//! Efficient Allocation with Costly Value Estimation" (arXiv:2608.20316): a
//! cheap estimator may already predict which candidate will succeed, while a
//! stronger estimator gives a better routing decision but costs tokens and
//! latency. The router purchases the better estimate only when the purchase
//! has positive expected value:
//!
//! ```text
//! buy  ⇔  expected_quality_gain × value_of_success  >  estimator_cost + latency_cost
//! ```
//!
//! # Closed-form Gaussian case
//!
//! Belief over a candidate's utility is `U ~ N(μ, σ²)`. An estimator observes
//! `y = U + ε` with `ε ~ N(0, τ²)`. After observing `y`, the posterior mean is
//! `m′ = μ + k·(y − μ)` with gain `k = σ²/(σ² + τ²)`. Before the observation,
//! `m′` is itself Gaussian under the prior predictive: `m′ ~ N(μ, s²)` with
//!
//! ```text
//! s² = σ⁴ / (σ² + τ²)
//! ```
//!
//! Routing against a known outside option `a` (the best alternative's value),
//! the decision value without the estimate is `max(μ, a)`; with it, the router
//! decides after seeing `m′`, so the expected value is `E[max(m′, a)]`. For
//! `X ~ N(μ, s²)` the Gaussian expected-improvement integral gives the closed
//! form (with `z = (a − μ)/s`, standard normal cdf `Φ` and pdf `φ`):
//!
//! ```text
//! E[max(X, a)] = a·Φ(z) + μ·(1 − Φ(z)) + s·φ(z)
//! VoI(μ, σ, τ, a) = E[max(m′, a)] − max(μ, a)  ≥ 0
//! ```
//!
//! `VoI` is maximized at `a = μ`, giving the upper bound `s·φ(0) = s/√(2π)` —
//! no estimator purchase can ever be worth more than that in utility units.
//!
//! # Calibrating `value_of_success`
//!
//! That bound is roughly `0.4σ`, so on a `[0, 1]`-scale utility with a
//! typical `σ ≈ 0.05` an estimator is worth at most `value_of_success × 0.02`.
//! Leaving `value_of_success` at a nominal `1.0` therefore makes any
//! estimator costing more than about two cents permanently unpurchasable, and
//! a gate built on it degenerates into a never-buy switch that still looks
//! configured. Express `value_of_success` as the currency value of a correct
//! decision, and sanity-check with [`voi_upper_bound`].
//!
//! # Intended call sites
//!
//! [`decide`] is a standalone pure function so the same primitive can gate
//! model selection, retrieval depth, verifier invocations,
//! agent spawning, and escalation. This crate stays independent of those
//! consumers; each constructs its own [`EstimatorSpec`] ladder.
//!
//! # Numerical discipline
//!
//! Every input crosses [`Belief::new`] / [`EstimatorSpec::validate`] /
//! [`VoiConfig::validate`], which reject non-finite values *before* any
//! comparison is made — a NaN must never reach the decision inequality
//! (comparison-based gates silently pass NaN; see PIR Wave-3 finding on
//! non-finite bypass).

/// Errors from invalid VoI inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiError {
    /// An input was non-finite or outside the expected range.
    InvalidInput(String),
}

impl std::fmt::Display for VoiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(message) => write!(f, "Invalid input: {message}"),
        }
    }
}

impl std::error::Error for VoiError {}

/// Result of a value-of-information operation.
pub type Result<T> = std::result::Result<T, VoiError>;

/// 1/√(2π), the standard normal density at zero.
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

fn invalid(msg: impl Into<String>) -> VoiError {
    VoiError::InvalidInput(msg.into())
}

fn require_finite(name: &str, value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(format!("{name} must be finite, got {value}")))
    }
}

/// Standard normal pdf.
fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal cdf via the Abramowitz–Stegun 7.1.26 erf approximation
/// (absolute error ≤ 1.5e-7, ample for a purchase decision).
fn norm_cdf(x: f64) -> f64 {
    let z = x / std::f64::consts::SQRT_2;
    let (sign, z) = if z < 0.0 { (-1.0, -z) } else { (1.0, z) };
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = sign * (1.0 - poly * (-z * z).exp());
    0.5 * (1.0 + erf)
}

/// Gaussian belief `N(mean, std_dev²)` over a candidate's utility.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Belief {
    mean: f64,
    std_dev: f64,
}

impl Belief {
    /// Create a belief. Rejects non-finite inputs and `std_dev <= 0` — a
    /// degenerate (certain) prior leaves nothing for an estimator to reveal.
    pub fn new(mean: f64, std_dev: f64) -> Result<Self> {
        require_finite("belief mean", mean)?;
        require_finite("belief std_dev", std_dev)?;
        if std_dev <= 0.0 {
            return Err(invalid(format!(
                "belief std_dev must be > 0, got {std_dev}"
            )));
        }
        Ok(Self { mean, std_dev })
    }

    /// Posterior mean of the belief.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Posterior standard deviation of the belief.
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }
}

/// A purchasable estimator: pay `cost` and `latency_us`, observe the true
/// utility corrupted by `N(0, noise_std²)` noise. `noise_std == 0` models a
/// perfect oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EstimatorSpec {
    /// Direct cost of one invocation, in the same currency as
    /// [`VoiConfig::value_of_success`] (e.g. dollars or token-dollars).
    pub cost: f64,
    /// Added latency of one invocation, in microseconds.
    pub latency_us: f64,
    /// Observation noise standard deviation, in utility units.
    pub noise_std: f64,
}

impl EstimatorSpec {
    /// Reject non-finite or negative fields before any comparison.
    pub fn validate(&self) -> Result<()> {
        for (name, v) in [
            ("estimator cost", self.cost),
            ("estimator latency_us", self.latency_us),
            ("estimator noise_std", self.noise_std),
        ] {
            require_finite(name, v)?;
            if v < 0.0 {
                return Err(invalid(format!("{name} must be >= 0, got {v}")));
            }
        }
        Ok(())
    }

    /// Total monetized cost of one invocation under `config`.
    fn monetized_cost(&self, config: &VoiConfig) -> f64 {
        self.cost + self.latency_us * config.latency_price
    }
}

/// Economic parameters for the purchase decision.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VoiConfig {
    /// Currency value of one unit of routing utility (converts VoI from
    /// utility units into the same currency as [`EstimatorSpec::cost`]).
    pub value_of_success: f64,
    /// Currency price per microsecond of added latency.
    pub latency_price: f64,
}

impl VoiConfig {
    /// Reject non-finite, non-positive `value_of_success`, or negative
    /// `latency_price` before any comparison.
    pub fn validate(&self) -> Result<()> {
        require_finite("value_of_success", self.value_of_success)?;
        require_finite("latency_price", self.latency_price)?;
        if self.value_of_success <= 0.0 {
            return Err(invalid(format!(
                "value_of_success must be > 0, got {}",
                self.value_of_success
            )));
        }
        if self.latency_price < 0.0 {
            return Err(invalid(format!(
                "latency_price must be >= 0, got {}",
                self.latency_price
            )));
        }
        Ok(())
    }
}

/// Outcome of one VoI evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VoiDecision {
    /// Purchase estimator `estimators[idx]`; observe, update the belief with
    /// [`observe`], and re-evaluate.
    Buy(usize),
    /// No estimator purchase has positive expected value — route on the
    /// current belief.
    Route,
}

/// Standard deviation `s` of the prior-predictive posterior mean:
/// `s² = σ⁴/(σ² + τ²)`. `τ = 0` gives `s = σ` (a perfect estimator reveals
/// the full prior uncertainty).
///
/// Evaluated as `σ / √(1 + (τ/σ)²)` rather than literally: the σ⁴ form
/// underflows to `0/0 = NaN` for small-but-legal σ (σ = 1e-200 squares to
/// zero), and a NaN ceiling silently defeats every `cost > bound` check
/// downstream. The ratio form is exact for the same inputs and degrades to
/// `0` — "this estimator reveals nothing" — when τ/σ overflows.
fn predictive_mean_std(sigma: f64, tau: f64) -> f64 {
    if tau == 0.0 {
        return sigma;
    }
    let ratio = tau / sigma;
    sigma / (1.0 + ratio * ratio).sqrt()
}

/// `E[max(X, alt)]` for `X ~ N(mean, std²)`.
fn expected_max(mean: f64, std: f64, alt: f64) -> f64 {
    let z = (alt - mean) / std;
    alt * norm_cdf(z) + mean * (1.0 - norm_cdf(z)) + std * norm_pdf(z)
}

/// Value of information (utility units) of purchasing one observation with
/// noise `noise_std`, given `belief` and a known outside option `alternative`.
///
/// Guaranteed finite and non-negative: negatives (approximation error) and
/// non-finite intermediates both clamp to `0`, i.e. "not worth buying" — the
/// safe direction, since a clamped value can only suppress a purchase.
///
/// # Precision
///
/// [`norm_cdf`] uses the Abramowitz–Stegun 7.1.26 rational approximation
/// (absolute error ≤ 1.5e-7). In [`expected_max`] that error is scaled by
/// `mean`, so for utilities of large magnitude the deep tail (`|z|` large,
/// true VoI near zero) can be overestimated by orders of magnitude relative
/// to a vanishing true value. The bias is one-directional — it can trigger a
/// worthless purchase, never suppress a worthwhile one — and is immaterial
/// for `[0, 1]`-scale utilities like router scores. Callers reusing
/// this primitive with large-magnitude utilities should either rescale to
/// unit range or substitute a higher-precision cdf.
pub fn value_of_information(belief: Belief, alternative: f64, noise_std: f64) -> Result<f64> {
    require_finite("alternative", alternative)?;
    require_finite("noise_std", noise_std)?;
    if noise_std < 0.0 {
        return Err(invalid(format!("noise_std must be >= 0, got {noise_std}")));
    }
    let s = predictive_mean_std(belief.std_dev, noise_std);
    let voi = expected_max(belief.mean, s, alternative) - belief.mean.max(alternative);
    Ok(if voi.is_finite() { voi.max(0.0) } else { 0.0 })
}

/// Upper bound on [`value_of_information`] over every possible `alternative`:
/// `s·φ(0) = s/√(2π)`, attained at `alternative = mean`. No estimator
/// purchase can be worth more than this, so it is the natural sanity check
/// for a cost model (`if cost > bound { never purchasable }`).
///
/// Guaranteed finite and non-negative: a non-finite intermediate is clamped
/// to `0`, which reports "nothing is purchasable" rather than handing a
/// caller a NaN ceiling that makes every `cost > bound` comparison false.
pub fn voi_upper_bound(belief: Belief, noise_std: f64) -> Result<f64> {
    require_finite("noise_std", noise_std)?;
    if noise_std < 0.0 {
        return Err(invalid(format!("noise_std must be >= 0, got {noise_std}")));
    }
    let bound = predictive_mean_std(belief.std_dev, noise_std) * INV_SQRT_2PI;
    Ok(if bound.is_finite() {
        bound.max(0.0)
    } else {
        0.0
    })
}

/// One greedy step of the index policy: evaluate every estimator's net value
/// (`value_of_success × VoI − monetized cost`) and buy the best strictly
/// positive one; otherwise route. Ties break to the lowest index, so the
/// decision is deterministic for identical inputs. When the alternative is
/// many standard deviations from the mean, the mathematically-positive VoI
/// underflows to exactly 0 — a free estimator is then value-indifferent and
/// the policy routes rather than buys.
///
/// The caller purchases the returned estimator, feeds its observation through
/// [`observe`], and calls `decide` again; the posterior tightens each round
/// (see [`observe`]), so VoI decays toward zero while costs stay fixed.
///
/// # Termination
///
/// That decay guarantees termination **only when every estimator has strictly
/// positive monetized cost**. [`EstimatorSpec::validate`] permits `cost == 0`
/// and `latency_us == 0`, and a free estimator is bought for as long as VoI
/// stays strictly positive — which, decaying geometrically, it does
/// indefinitely (a probe ran 100,000 rounds against a free estimator and was
/// still returning `Buy`). A caller driving the repeated-purchase protocol
/// must therefore either require `cost > 0` or impose its own round cap. This
/// crate's router integration is unaffected: it makes a single bounded
/// `decide` call over a one-element ladder and never loops.
///
/// # Noiseless estimators
///
/// `decide` may return [`VoiDecision::Buy`] for a spec with `noise_std == 0`,
/// which [`observe`] deliberately refuses — a noiseless observation *is* the
/// exact utility, so there is no posterior left to represent. That case exits
/// the protocol rather than continuing it: purchase the oracle, take its
/// value as ground truth, and route on it directly without calling
/// [`observe`].
pub fn decide(
    belief: Belief,
    alternative: f64,
    estimators: &[EstimatorSpec],
    config: &VoiConfig,
) -> Result<VoiDecision> {
    require_finite("alternative", alternative)?;
    config.validate()?;
    let mut best: Option<(usize, f64)> = None;
    for (idx, est) in estimators.iter().enumerate() {
        est.validate()?;
        let voi = value_of_information(belief, alternative, est.noise_std)?;
        let net = config.value_of_success * voi - est.monetized_cost(config);
        if net > 0.0
            && match best {
                None => true,
                Some((_, best_net)) => net > best_net,
            }
        {
            best = Some((idx, net));
        }
    }
    Ok(match best {
        Some((idx, _)) => VoiDecision::Buy(idx),
        None => VoiDecision::Route,
    })
}

/// Bayesian update after purchasing `estimator` and observing `observation`:
/// gain `k = σ²/(σ² + τ²)`, posterior mean `μ + k·(y − μ)`, posterior
/// variance `σ²τ²/(σ² + τ²) ≤ σ²` (never increases). Requires
/// `noise_std > 0`; a perfect (`τ = 0`) observation leaves no residual
/// uncertainty to represent — the caller already holds the exact utility.
pub fn observe(belief: Belief, estimator: &EstimatorSpec, observation: f64) -> Result<Belief> {
    estimator.validate()?;
    require_finite("observation", observation)?;
    if estimator.noise_std <= 0.0 {
        return Err(invalid(
            "observe requires noise_std > 0; a noiseless observation is the exact utility",
        ));
    }
    let s2 = belief.std_dev * belief.std_dev;
    let t2 = estimator.noise_std * estimator.noise_std;
    let k = s2 / (s2 + t2);
    Belief::new(
        belief.mean + k * (observation - belief.mean),
        (s2 * t2 / (s2 + t2)).sqrt(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn cfg() -> VoiConfig {
        VoiConfig {
            value_of_success: 1.0,
            latency_price: 0.0,
        }
    }

    #[test]
    fn rejects_non_finite_at_the_choke_point() {
        assert!(Belief::new(f64::NAN, 1.0).is_err());
        assert!(Belief::new(0.0, f64::INFINITY).is_err());
        assert!(Belief::new(0.0, 0.0).is_err());
        assert!(Belief::new(0.0, -1.0).is_err());
        let b = Belief::new(0.5, 0.2).unwrap();
        assert!(value_of_information(b, f64::NAN, 0.1).is_err());
        assert!(value_of_information(b, 0.5, f64::NAN).is_err());
        let bad = EstimatorSpec {
            cost: f64::NAN,
            latency_us: 0.0,
            noise_std: 0.1,
        };
        assert!(decide(b, 0.5, &[bad], &cfg()).is_err());
        assert!(VoiConfig {
            value_of_success: f64::NAN,
            latency_price: 0.0
        }
        .validate()
        .is_err());
        assert!(observe(b, &bad, 0.5).is_err());
    }

    #[test]
    fn zero_cost_perfect_estimator_always_buys() {
        // Property (b): zero-noise, zero-cost estimator has strictly positive
        // net value for any uncertain prior, so it is always purchased —
        // whenever the VoI is floating-point representable (an alternative
        // dozens of σ away underflows VoI to exactly 0 = indifference).
        let free_oracle = EstimatorSpec {
            cost: 0.0,
            latency_us: 0.0,
            noise_std: 0.0,
        };
        for mean in [-2.0, 0.0, 0.5, 3.0] {
            for alt in [-1.0, 0.0, 0.5, 2.0] {
                let b = Belief::new(mean, 1.0).unwrap();
                assert_eq!(
                    decide(b, alt, &[free_oracle], &cfg()).unwrap(),
                    VoiDecision::Buy(0),
                    "mean={mean} alt={alt}"
                );
            }
        }
    }

    proptest! {
        // Property (a): never buys when the cost exceeds the VoI upper bound
        // value_of_success × s·φ(0).
        #[test]
        fn never_buys_above_the_voi_upper_bound(
            mean in -10.0f64..10.0,
            sigma in 0.01f64..5.0,
            alt in -10.0f64..10.0,
            tau in 0.0f64..5.0,
        ) {
            let b = Belief::new(mean, sigma).unwrap();
            let bound = voi_upper_bound(b, tau).unwrap();
            let est = EstimatorSpec { cost: bound + 1e-9, latency_us: 0.0, noise_std: tau };
            prop_assert_eq!(decide(b, alt, &[est], &cfg()).unwrap(), VoiDecision::Route);
        }

        // Property (a) corollary: VoI never exceeds its closed-form bound.
        #[test]
        fn voi_respects_its_upper_bound(
            mean in -10.0f64..10.0,
            // Range reaches into the subnormal-σ regime that made the
            // literal σ⁴/(σ²+τ²) form return NaN.
            sigma in prop::sample::select(vec![1e-300f64, 1e-200, 1e-8, 0.01, 0.5, 5.0]),
            alt in -10.0f64..10.0,
            tau in 0.0f64..5.0,
        ) {
            let b = Belief::new(mean, sigma).unwrap();
            let voi = value_of_information(b, alt, tau).unwrap();
            prop_assert!(voi >= 0.0);
            prop_assert!(voi <= voi_upper_bound(b, tau).unwrap() + 1e-9);
        }

        // Property (c): posterior variance is monotonically non-increasing
        // across a chain of purchases.
        #[test]
        fn posterior_variance_never_increases(
            mean in -5.0f64..5.0,
            sigma in 0.01f64..3.0,
            taus in prop::collection::vec(0.01f64..3.0, 1..6),
            ys in prop::collection::vec(-5.0f64..5.0, 6),
        ) {
            let mut b = Belief::new(mean, sigma).unwrap();
            for (tau, y) in taus.iter().zip(ys.iter()) {
                let est = EstimatorSpec { cost: 0.0, latency_us: 0.0, noise_std: *tau };
                let next = observe(b, &est, *y).unwrap();
                prop_assert!(next.std_dev() <= b.std_dev() + 1e-12);
                b = next;
            }
        }

        // Property (d): decisions are deterministic for identical inputs.
        #[test]
        fn decisions_are_deterministic(
            mean in -5.0f64..5.0,
            sigma in 0.01f64..3.0,
            alt in -5.0f64..5.0,
            specs in prop::collection::vec(
                (0.0f64..0.5, 0.0f64..100.0, 0.0f64..2.0), 0..5),
        ) {
            let b = Belief::new(mean, sigma).unwrap();
            let ests: Vec<EstimatorSpec> = specs.iter()
                .map(|&(cost, latency_us, noise_std)| EstimatorSpec { cost, latency_us, noise_std })
                .collect();
            let config = VoiConfig { value_of_success: 2.0, latency_price: 0.001 };
            let d1 = decide(b, alt, &ests, &config).unwrap();
            let d2 = decide(b, alt, &ests, &config).unwrap();
            prop_assert_eq!(d1, d2);
        }
    }

    #[test]
    fn noisier_estimators_are_worth_less() {
        let b = Belief::new(0.5, 0.3).unwrap();
        let sharp = value_of_information(b, 0.5, 0.05).unwrap();
        let blunt = value_of_information(b, 0.5, 1.0).unwrap();
        assert!(sharp > blunt);
    }

    #[test]
    fn greedy_step_prefers_the_higher_net_estimator() {
        let b = Belief::new(0.5, 0.4).unwrap();
        // Same noise, different price: the cheaper one must win.
        let pricey = EstimatorSpec {
            cost: 0.05,
            latency_us: 0.0,
            noise_std: 0.1,
        };
        let cheap = EstimatorSpec {
            cost: 0.01,
            latency_us: 0.0,
            noise_std: 0.1,
        };
        assert_eq!(
            decide(b, 0.5, &[pricey, cheap], &cfg()).unwrap(),
            VoiDecision::Buy(1)
        );
    }

    #[test]
    fn latency_is_priced_into_the_purchase() {
        let b = Belief::new(0.5, 0.4).unwrap();
        let slow = EstimatorSpec {
            cost: 0.0,
            latency_us: 1_000_000.0,
            noise_std: 0.1,
        };
        let config = VoiConfig {
            value_of_success: 1.0,
            latency_price: 1.0, // exorbitant: 1 currency unit per µs
        };
        assert_eq!(
            decide(b, 0.5, &[slow], &config).unwrap(),
            VoiDecision::Route
        );
    }

    #[test]
    fn tiny_sigma_never_yields_a_nan_ceiling() {
        // Regression: the literal σ⁴/(σ²+τ²) form underflowed to 0/0 here,
        // and a NaN bound makes every `cost > bound` guard false — the
        // caller then buys unconditionally.
        for sigma in [1e-300, 1e-200, 1e-160, 1e-8] {
            let b = Belief::new(0.0, sigma).unwrap();
            for tau in [0.0, 1e-200, 1.0] {
                let bound = voi_upper_bound(b, tau).unwrap();
                assert!(bound.is_finite() && bound >= 0.0, "sigma={sigma} tau={tau}");
                let voi = value_of_information(b, 0.0, tau).unwrap();
                assert!(voi.is_finite() && voi >= 0.0, "sigma={sigma} tau={tau}");
            }
        }
    }

    #[test]
    fn norm_cdf_matches_known_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((norm_cdf(1.959_964) - 0.975).abs() < 1e-5);
        assert!((norm_cdf(-1.959_964) - 0.025).abs() < 1e-5);
    }
}
