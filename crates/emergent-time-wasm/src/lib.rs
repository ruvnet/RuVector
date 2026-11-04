//! # `@ruvector/emergent-time` — WASM bindings for **Agentic Time**
//!
//! This crate wraps the *agentic-time layer* of the dependency-free
//! [`emergent-time`](https://crates.io/crates/emergent-time) Rust crate in a tiny
//! `wasm-bindgen` surface for the browser, the edge, and Node.
//!
//! Agentic time measures how much an AI agent has *changed internally*, not how
//! many seconds, steps, or tokens have elapsed. You feed it the six channel
//! deltas of a transition — belief, memory, retrieval, goal-graph, contradiction,
//! plan — and it returns an explainable [`Tick`], a cumulative internal-time
//! reading, the Agentic Time Index (ATI = progress per unit structural change),
//! and a 7-state health classification.
//!
//! The physics core of the parent crate (Wheeler–DeWitt, Page–Wootters, entropic
//! and thermal time, Structural Proper Time) is **not** wrapped here — it deals
//! in dense matrices that do not serialize cleanly or cheaply across the JS
//! boundary, and the agentic layer is the JS-useful product. Use the
//! `emergent-time` crate directly for the physics.
//!
//! ## Honest scope (mirrors the Rust crate / ADR-251)
//!
//! The agentic clock is a **diagnostic signal**. On real recorded traces it does
//! **not** establish an early-warning lead over a fair cheap baseline (a windowed
//! z-score on a single observable, or a Page–Hinkley detector). It is a useful,
//! explainable, per-channel decomposition of internal change and a health
//! classifier — not a proven predictor that beats a fair competitor. Both fair
//! competitors are exported here so you can run the same comparison yourself.

#![allow(clippy::new_without_default)]

use emergent_time::adaptive::PageHinkley as CorePageHinkley;
use emergent_time::agentic_time::{
    agentic_time_index, classify, AgentState, AgenticTime, AgenticWeights, HealthThresholds,
    Tick as CoreTick, TickClass,
};
use emergent_time::weight_learning::{FeatureMode, LearnedWeights as CoreLearnedWeights};
use wasm_bindgen::prelude::*;

// Tiny global allocator — far smaller than the default for short-lived wasm.
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

/// Optional: route Rust panics to the JS console with a readable message.
/// Call once after instantiation. No-op cost if never called.
#[wasm_bindgen(js_name = setPanicHook)]
pub fn set_panic_hook() {
    // Minimal hook without the console_error_panic_hook dependency: a panic in a
    // `panic = "abort"` wasm traps anyway; this just makes the abort explicit.
    // (Kept dependency-free to preserve the tiny bundle.)
}

// ---------------------------------------------------------------------------
// Channel deltas — the per-transition input.
// ---------------------------------------------------------------------------

/// The six per-transition channel deltas fed to [`AgenticClock::tick`].
///
/// Each field is the **already-computed scalar movement** of that channel over a
/// transition (e.g. the L2 distance between successive belief embeddings, or the
/// absolute change in a contradiction score). Keeping the JS boundary scalar — six
/// numbers, not six embedding vectors — keeps the wasm tiny and lets the caller
/// pick whatever distance metric and embedding model they like on the JS side.
///
/// `contradictionLevel` is the *current absolute* contradiction in `[0, 1]` (not a
/// delta); it drives the collapse / human-review health states.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub struct StateDelta {
    belief: f64,
    memory: f64,
    retrieval: f64,
    goal: f64,
    contradiction: f64,
    plan: f64,
    contradiction_level: f64,
    progress: f64,
}

#[wasm_bindgen]
impl StateDelta {
    /// Construct a transition's channel deltas.
    ///
    /// * `belief`, `memory`, `retrieval`, `plan` — non-negative scalar movements
    ///   (typically L2 distance between successive embeddings).
    /// * `goal` — absolute change in goal-graph mass (e.g. open-subgoal count).
    /// * `contradiction` — absolute change in the contradiction score.
    /// * `contradictionLevel` — current absolute contradiction in `[0, 1]`.
    /// * `progress` — absolute change in task progress over this transition
    ///   (used for the ATI and health classification; pass `0` if unknown).
    ///
    /// Non-finite inputs (`NaN`, `±Infinity`) are treated as `0` so one bad
    /// sample cannot poison the clock's cumulative and windowed state.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        belief: f64,
        memory: f64,
        retrieval: f64,
        goal: f64,
        contradiction: f64,
        plan: f64,
        contradiction_level: f64,
        progress: f64,
    ) -> StateDelta {
        let f = |x: f64| if x.is_finite() { x } else { 0.0 };
        let (belief, memory, retrieval, goal) = (f(belief), f(memory), f(retrieval), f(goal));
        let (contradiction, plan) = (f(contradiction), f(plan));
        let (contradiction_level, progress) = (f(contradiction_level), f(progress));
        StateDelta {
            belief: belief.max(0.0),
            memory: memory.max(0.0),
            retrieval: retrieval.max(0.0),
            goal: goal.abs(),
            contradiction: contradiction.abs(),
            plan: plan.max(0.0),
            contradiction_level: contradiction_level.clamp(0.0, 1.0),
            progress,
        }
    }
}

impl StateDelta {
    /// Build a pair of [`AgentState`]s whose deltas equal this struct, so the core
    /// (which works on state pairs) can compute the weighted tick. We pack each
    /// scalar channel movement into a 1-D embedding: `prev = [0]`, `cur = [d]` so
    /// `l2(prev, cur) == d`. Goal / contradiction are scalar fields on the state.
    fn as_state_pair(&self) -> (AgentState, AgentState) {
        let prev = AgentState {
            belief: vec![0.0],
            memory: vec![0.0],
            retrieval: vec![0.0],
            goal_graph: 0.0,
            contradiction: 0.0,
            plan: vec![0.0],
            tokens: 0,
        };
        let cur = AgentState {
            belief: vec![self.belief],
            memory: vec![self.memory],
            retrieval: vec![self.retrieval],
            goal_graph: self.goal,
            contradiction: self.contradiction,
            plan: vec![self.plan],
            tokens: 0,
        };
        (prev, cur)
    }
}

// ---------------------------------------------------------------------------
// Tick — explainable per-step result.
// ---------------------------------------------------------------------------

/// One agentic-time class (mirrors the Rust `TickClass`). Exposed as a small
/// enum so the JS side can `switch` on it without string parsing.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TickClassJs {
    /// Below the noise floor — no meaningful change.
    Idle = 0,
    /// Belief / plan / goal moved forward.
    Progress = 1,
    /// New information arrived (retrieval / memory moved).
    Learning = 2,
    /// Contradiction rose.
    Contradiction = 3,
    /// Contradiction is high — failure regime.
    Collapse = 4,
}

impl From<TickClass> for TickClassJs {
    fn from(c: TickClass) -> Self {
        match c {
            TickClass::Idle => TickClassJs::Idle,
            TickClass::Progress => TickClassJs::Progress,
            TickClass::Learning => TickClassJs::Learning,
            TickClass::Contradiction => TickClassJs::Contradiction,
            TickClass::Collapse => TickClassJs::Collapse,
        }
    }
}

/// An explainable agentic-time tick: the post-floor internal-time increment, its
/// class, a human-readable reason, and the raw (pre-floor) per-channel weighted
/// contributions. See the Rust crate's `Tick` docs for the post-floor /
/// pre-floor contract: `deltaTime == Σ channels` only when `noiseFloor == 0`.
#[wasm_bindgen]
pub struct Tick {
    inner: CoreTick,
}

#[wasm_bindgen]
impl Tick {
    /// Post-floor internal-time magnitude: `max(0, Σ channels − noiseFloor)`.
    #[wasm_bindgen(getter, js_name = deltaTime)]
    pub fn delta_time(&self) -> f64 {
        self.inner.delta
    }

    /// The tick class (Idle / Progress / Learning / Contradiction / Collapse).
    #[wasm_bindgen(getter)]
    pub fn class(&self) -> TickClassJs {
        self.inner.class.into()
    }

    /// A human-readable audit string explaining which channel dominated.
    #[wasm_bindgen(getter)]
    pub fn reason(&self) -> String {
        self.inner.reason.clone()
    }

    /// Raw (pre-floor) weighted belief contribution.
    #[wasm_bindgen(getter)]
    pub fn belief(&self) -> f64 {
        self.inner.belief
    }
    /// Raw (pre-floor) weighted memory contribution.
    #[wasm_bindgen(getter)]
    pub fn memory(&self) -> f64 {
        self.inner.memory
    }
    /// Raw (pre-floor) weighted retrieval contribution.
    #[wasm_bindgen(getter)]
    pub fn retrieval(&self) -> f64 {
        self.inner.retrieval
    }
    /// Raw (pre-floor) weighted goal-graph contribution.
    #[wasm_bindgen(getter, js_name = goalGraph)]
    pub fn goal_graph(&self) -> f64 {
        self.inner.goal_graph
    }
    /// Raw (pre-floor) weighted contradiction contribution.
    #[wasm_bindgen(getter)]
    pub fn contradiction(&self) -> f64 {
        self.inner.contradiction
    }
    /// Raw (pre-floor) weighted plan contribution.
    #[wasm_bindgen(getter)]
    pub fn plan(&self) -> f64 {
        self.inner.plan
    }
}

// ---------------------------------------------------------------------------
// Health classification.
// ---------------------------------------------------------------------------

/// The 7-state agent health verdict (mirrors the Rust `AgentHealth`).
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentHealthJs {
    /// Progress is keeping pace with internal change.
    Healthy = 0,
    /// Moving, but inefficiently (low progress per unit change).
    Drifting = 1,
    /// Neither changing nor progressing.
    Stuck = 2,
    /// Lots of internal churn, no progress — replan.
    NeedsReplan = 3,
    /// Losing ground (progress going backwards).
    Contradicting = 4,
    /// Contradiction is high and rising.
    Collapsing = 5,
    /// Contradiction is critical — escalate to a human.
    NeedsHumanReview = 6,
}

impl From<emergent_time::agentic_time::AgentHealth> for AgentHealthJs {
    fn from(h: emergent_time::agentic_time::AgentHealth) -> Self {
        use emergent_time::agentic_time::AgentHealth as H;
        match h {
            H::Healthy => AgentHealthJs::Healthy,
            H::Drifting => AgentHealthJs::Drifting,
            H::Stuck => AgentHealthJs::Stuck,
            H::NeedsReplan => AgentHealthJs::NeedsReplan,
            H::Contradicting => AgentHealthJs::Contradicting,
            H::Collapsing => AgentHealthJs::Collapsing,
            H::NeedsHumanReview => AgentHealthJs::NeedsHumanReview,
        }
    }
}

// ---------------------------------------------------------------------------
// AgenticClock — the centerpiece.
// ---------------------------------------------------------------------------

/// A stateful agentic-time clock. Construct it (optionally with custom channel
/// weights / health thresholds), feed transitions via [`AgenticClock::tick`], and
/// read back cumulative agentic time, the ATI, and the health classification.
///
/// The clock keeps a small amount of running state: cumulative agentic time,
/// cumulative progress, and a rolling window of the last `window` ticks /
/// progress used for the ATI and health classification.
#[wasm_bindgen]
pub struct AgenticClock {
    clock: AgenticTime,
    thresholds: HealthThresholds,
    noise_floor: f64,
    window: usize,
    // Running state.
    cumulative_time: f64,
    cumulative_progress: f64,
    recent_delta: Vec<f64>,
    recent_progress: Vec<f64>,
    last_contradiction: f64,
}

#[wasm_bindgen]
impl AgenticClock {
    /// Construct a clock with the default channel weights (contradiction 1.5,
    /// belief / goal / plan 1.0, memory / retrieval 0.5), default health
    /// thresholds, a noise floor of `1e-3`, and a window of 8 ticks.
    #[wasm_bindgen(constructor)]
    pub fn new() -> AgenticClock {
        AgenticClock {
            clock: AgenticTime::new(AgenticWeights::default()),
            thresholds: HealthThresholds::default(),
            noise_floor: 1e-3,
            window: 8,
            cumulative_time: 0.0,
            cumulative_progress: 0.0,
            recent_delta: Vec::new(),
            recent_progress: Vec::new(),
            last_contradiction: 0.0,
        }
    }

    /// Construct a clock with custom channel weights. Order:
    /// `belief, memory, retrieval, goalGraph, contradiction, plan`.
    #[wasm_bindgen(js_name = withWeights)]
    pub fn with_weights(
        belief: f64,
        memory: f64,
        retrieval: f64,
        goal_graph: f64,
        contradiction: f64,
        plan: f64,
    ) -> AgenticClock {
        let mut c = AgenticClock::new();
        c.clock = AgenticTime::new(AgenticWeights {
            belief,
            memory,
            retrieval,
            goal_graph,
            contradiction,
            plan,
        });
        c
    }

    /// Override the noise floor (jitter suppression). Ticks whose raw channel sum
    /// is below this floor report `deltaTime == 0`.
    #[wasm_bindgen(js_name = setNoiseFloor)]
    pub fn set_noise_floor(&mut self, floor: f64) {
        self.noise_floor = floor.max(0.0);
    }

    /// Override the rolling window length used for the ATI and health (default 8).
    #[wasm_bindgen(js_name = setWindow)]
    pub fn set_window(&mut self, window: usize) {
        self.window = window.max(1);
    }

    /// Override the health-classifier thresholds. Order:
    /// `idle, healthyAti, driftingAti, collapse, humanReview`.
    #[wasm_bindgen(js_name = setThresholds)]
    pub fn set_thresholds(
        &mut self,
        idle: f64,
        healthy_ati: f64,
        drifting_ati: f64,
        collapse: f64,
        human_review: f64,
    ) {
        self.thresholds = HealthThresholds {
            idle,
            healthy_ati,
            drifting_ati,
            collapse,
            human_review,
        };
    }

    /// Feed one transition's channel deltas and return the explainable [`Tick`].
    /// Advances the clock's cumulative agentic time and progress, and updates the
    /// rolling window used by [`AgenticClock::ati`] and
    /// [`AgenticClock::health`].
    pub fn tick(&mut self, delta: &StateDelta) -> Tick {
        let (prev, cur) = delta.as_state_pair();
        let core: CoreTick = self.clock.explain(&prev, &cur, self.noise_floor);

        self.cumulative_time += core.delta;
        self.cumulative_progress += delta.progress;
        self.last_contradiction = delta.contradiction_level;

        self.recent_delta.push(core.delta);
        self.recent_progress.push(delta.progress);
        // Drop *all* excess, not just one: after `setWindow` shrinks the window,
        // removing one per push would keep it at the old length forever.
        let excess = self.recent_delta.len().saturating_sub(self.window);
        self.recent_delta.drain(..excess);
        self.recent_progress.drain(..excess);

        Tick { inner: core }
    }

    /// Cumulative agentic time accrued across all ticks so far.
    #[wasm_bindgen(getter, js_name = cumulativeTime)]
    pub fn cumulative_time(&self) -> f64 {
        self.cumulative_time
    }

    /// Cumulative progress accrued across all ticks so far.
    #[wasm_bindgen(getter, js_name = cumulativeProgress)]
    pub fn cumulative_progress(&self) -> f64 {
        self.cumulative_progress
    }

    /// The Agentic Time Index over the current window: progress per unit of
    /// structural change. High ATI ⇒ learning and moving; near-zero ⇒ spinning;
    /// `Infinity` ⇒ progressing with no internal change.
    #[wasm_bindgen(getter)]
    pub fn ati(&self) -> f64 {
        let dt: f64 = self.recent_delta.iter().sum();
        let dp: f64 = self.recent_progress.iter().sum();
        agentic_time_index(dt, dp)
    }

    /// The current health verdict, classified over the rolling window of agentic
    /// time, progress, and the latest contradiction level.
    #[wasm_bindgen(getter)]
    pub fn health(&self) -> AgentHealthJs {
        let dt: f64 = self.recent_delta.iter().sum();
        let dp: f64 = self.recent_progress.iter().sum();
        classify(dt, dp, self.last_contradiction, &self.thresholds).into()
    }

    /// Reset all running state (cumulative time / progress, window) to zero,
    /// keeping the configured weights, thresholds, noise floor, and window length.
    pub fn reset(&mut self) {
        self.cumulative_time = 0.0;
        self.cumulative_progress = 0.0;
        self.recent_delta.clear();
        self.recent_progress.clear();
        self.last_contradiction = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Change-point detectors.
// ---------------------------------------------------------------------------

/// A **windowed z-score** change-point detector (rolling `mean + kσ`): push a
/// scalar each step, get back the z-score and an alarm flag. This is the *fair
/// baseline* the agentic clock is honestly compared against — a cheap one-signal
/// detector a practitioner would actually deploy.
#[wasm_bindgen]
pub struct WindowedDeltaClock {
    window: usize,
    k_sigma: f64,
    std_floor: f64,
    history: Vec<f64>,
    alarmed: bool,
    last_alarm_index: i64,
    index: usize,
}

#[wasm_bindgen]
impl WindowedDeltaClock {
    /// Construct a detector with a trailing `window`, a `kSigma` alarm multiplier
    /// (e.g. 4.0), and a `stdFloor` variance floor that prevents a near-constant
    /// stream from producing a spurious infinite z-score.
    #[wasm_bindgen(constructor)]
    pub fn new(window: usize, k_sigma: f64, std_floor: f64) -> WindowedDeltaClock {
        WindowedDeltaClock {
            window: window.max(2),
            k_sigma,
            std_floor: std_floor.max(0.0),
            history: Vec::new(),
            alarmed: false,
            last_alarm_index: -1,
            index: 0,
        }
    }

    /// Push the next scalar observable and return its rolling z-score (deviation
    /// from the trailing-window mean over the floored window std). Updates
    /// [`alarmed`](Self::alarmed) when the z-score exceeds `kSigma`.
    pub fn push(&mut self, value: f64) -> f64 {
        let start = self.history.len().saturating_sub(self.window);
        let win = &self.history[start..];
        let z = if win.len() < 2 {
            0.0
        } else {
            let mean = win.iter().sum::<f64>() / win.len() as f64;
            let var = win.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / win.len() as f64;
            let std = var.sqrt().max(self.std_floor);
            (value - mean).abs() / std
        };
        if z > self.k_sigma && !self.alarmed {
            self.alarmed = true;
            self.last_alarm_index = self.index as i64;
        }
        // Only the trailing `window` samples are ever read; keep no more, so a
        // long-lived detector does not grow without bound.
        if self.history.len() >= self.window {
            let excess = self.history.len() + 1 - self.window;
            self.history.drain(..excess);
        }
        self.history.push(value);
        self.index += 1;
        z
    }

    /// Whether the detector has fired (latched true on first alarm).
    #[wasm_bindgen(getter)]
    pub fn alarmed(&self) -> bool {
        self.alarmed
    }

    /// The 0-based index at which the detector first fired, or `-1` if it has not.
    #[wasm_bindgen(getter, js_name = alarmIndex)]
    pub fn alarm_index(&self) -> i64 {
        self.last_alarm_index
    }

    /// Reset the detector's history and alarm latch.
    pub fn reset(&mut self) {
        self.history.clear();
        self.alarmed = false;
        self.last_alarm_index = -1;
        self.index = 0;
    }
}

/// A **Page–Hinkley** adaptive change-point detector: a CUSUM test whose
/// reference is a *running* mean (so a noisy early phase does not permanently
/// raise the bar). Push a scalar each step, get back the current PH statistic and
/// an alarm flag. The adaptive counterpart of [`WindowedDeltaClock`]; both are the
/// fair competitors to the agentic clock.
#[wasm_bindgen]
pub struct PageHinkleyDetector {
    ph: CorePageHinkley,
    // Running state replicating the core's streaming statistic.
    count: f64,
    sum: f64,
    cum: f64,
    extreme: f64,
    alarmed: bool,
    last_alarm_index: i64,
    index: usize,
}

#[wasm_bindgen]
impl PageHinkleyDetector {
    /// Construct an **upward** (increase-detecting) Page–Hinkley detector with
    /// tolerance `delta` (deviations below this are treated as normal jitter) and
    /// alarm threshold `lambda` (larger ⇒ fewer false alarms, later detection).
    #[wasm_bindgen(constructor)]
    pub fn new(delta: f64, lambda: f64) -> PageHinkleyDetector {
        PageHinkleyDetector {
            ph: CorePageHinkley::upward(delta, lambda),
            count: 0.0,
            sum: 0.0,
            cum: 0.0,
            extreme: f64::INFINITY,
            alarmed: false,
            last_alarm_index: -1,
            index: 0,
        }
    }

    /// Construct a **downward** (decrease-detecting) Page–Hinkley detector.
    #[wasm_bindgen(js_name = downward)]
    pub fn downward(delta: f64, lambda: f64) -> PageHinkleyDetector {
        let mut d = PageHinkleyDetector::new(delta, lambda);
        d.ph = CorePageHinkley::downward(delta, lambda);
        d.extreme = f64::NEG_INFINITY;
        d
    }

    /// Push the next scalar and return the current Page–Hinkley statistic (the
    /// rise above the running minimum for the upward form, or the drop below the
    /// running maximum for the downward form). Updates [`alarmed`](Self::alarmed)
    /// when the statistic exceeds `lambda`.
    pub fn push(&mut self, value: f64) -> f64 {
        self.count += 1.0;
        self.sum += value;
        let mean = self.sum / self.count;
        let ph = if self.ph.upward {
            self.cum += value - mean - self.ph.delta;
            if self.cum < self.extreme {
                self.extreme = self.cum;
            }
            self.cum - self.extreme
        } else {
            self.cum += value - mean + self.ph.delta;
            if self.cum > self.extreme {
                self.extreme = self.cum;
            }
            self.extreme - self.cum
        };
        if ph > self.ph.lambda && !self.alarmed {
            self.alarmed = true;
            self.last_alarm_index = self.index as i64;
        }
        self.index += 1;
        ph
    }

    /// Whether the detector has fired (latched true on first alarm).
    #[wasm_bindgen(getter)]
    pub fn alarmed(&self) -> bool {
        self.alarmed
    }

    /// The 0-based index at which the detector first fired, or `-1` if it has not.
    #[wasm_bindgen(getter, js_name = alarmIndex)]
    pub fn alarm_index(&self) -> i64 {
        self.last_alarm_index
    }

    /// Reset the detector's running statistics and alarm latch.
    pub fn reset(&mut self) {
        self.count = 0.0;
        self.sum = 0.0;
        self.cum = 0.0;
        self.extreme = if self.ph.upward {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
        self.alarmed = false;
        self.last_alarm_index = -1;
        self.index = 0;
    }
}

// ---------------------------------------------------------------------------
// Learned weight scoring.
// ---------------------------------------------------------------------------

/// A fitted logistic-regression scorer over the channel-movement features (the
/// crate's `LearnedWeights`). Reconstruct it from persisted parameters and score
/// raw feature vectors to get a failure-approach probability in `[0, 1]`.
///
/// The training harness lives in the Rust crate; this binding exposes only the
/// cheap inference path (`predict`) so a model trained offline can run in the
/// browser without bundling the trainer.
#[wasm_bindgen]
pub struct LearnedWeights {
    inner: CoreLearnedWeights,
}

#[wasm_bindgen]
impl LearnedWeights {
    /// Reconstruct a scorer from persisted parameters. `coef`, `mean`, and `std`
    /// must each have length `dim` (the feature count: 6 for the full channel set,
    /// 5 for the contradiction-free "honest" set).
    #[wasm_bindgen(js_name = fromParams)]
    pub fn from_params(
        dim: usize,
        coef: Vec<f64>,
        bias: f64,
        mean: Vec<f64>,
        std: Vec<f64>,
    ) -> Result<LearnedWeights, JsError> {
        if coef.len() != dim || mean.len() != dim || std.len() != dim {
            return Err(JsError::new(
                "coef, mean, and std must each have length == dim",
            ));
        }
        // `predict` divides by each std; zero / non-finite values would turn the
        // documented [0, 1] probability into NaN. The trainer floors std at 1e-9.
        if std.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(JsError::new("std entries must be finite and > 0"));
        }
        Ok(LearnedWeights {
            inner: CoreLearnedWeights::from_params(dim, coef, bias, mean, std),
        })
    }

    /// Predicted failure-approach probability in `[0, 1]` for a raw feature vector
    /// (the per-channel movements in feature order). `features.length` must equal
    /// the model's `dim`.
    pub fn predict(&self, features: Vec<f64>) -> Result<f64, JsError> {
        if features.len() != self.inner.dim {
            return Err(JsError::new("features.length must equal the model dim"));
        }
        Ok(self.inner.predict(&features))
    }

    /// The model's feature dimensionality.
    #[wasm_bindgen(getter)]
    pub fn dim(&self) -> usize {
        self.inner.dim
    }

    /// Non-negative clock weights derived from the learned coefficients (the
    /// positive part), suitable for `AgenticClock.withWeights(...)`.
    #[wasm_bindgen(js_name = clockWeights)]
    pub fn clock_weights(&self) -> Vec<f64> {
        self.inner.clock_weights()
    }
}

/// The number of channel features for the full set (6) — for sizing
/// [`LearnedWeights`] parameter arrays.
#[wasm_bindgen(js_name = fullFeatureDim)]
pub fn full_feature_dim() -> usize {
    FeatureMode::Full.dim()
}

/// The number of channel features for the contradiction-free "honest" set (5).
#[wasm_bindgen(js_name = honestFeatureDim)]
pub fn honest_feature_dim() -> usize {
    FeatureMode::Honest.dim()
}

/// The package version (compile-time constant from Cargo).
#[wasm_bindgen(js_name = version)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests;
