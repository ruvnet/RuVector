//! Policy genome and score axes for the promotion flywheel (ADR-278).
//!
//! rvAgent does not implement a promotion engine. `@metaharness/flywheel`
//! already provides a frozen fingerprinted conjunctive gate, a holdout plus a
//! never-optimized-against anchor, Ed25519 receipts, independent replay
//! verification, and a compounding lineage DAG. This module is the Rust half of
//! that seam: the thing being evolved (a [`PolicyGenome`]) and the thing being
//! measured (a [`Score`]).
//!
//! The genome is deliberately shaped as the flywheel's `Policy` —
//! `Record<string, string>` — so there is no adapter impedance. It is also what
//! GEPA-style optimizers consume, whose candidate is likewise a named set of
//! text components.
//!
//! # Why policy and not memory
//!
//! Self-learning splits into two objects with opposite evidence: policy text
//! (positive) and accumulated episodic memory (negative — an inverted-U where
//! utility eventually falls below no-memory). ADR-278 moves new effort here.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::graph::GraphConfig;
use crate::masking::MaskConfig;

/// The levers this harness knows how to apply.
///
/// A genome naming anything outside this set is rejected rather than ignored —
/// see [`PolicyGenome::apply_to`].
pub const KNOWN_LEVERS: &[&str] = &[
    "max_iterations",
    "parallel_tools",
    "max_parallel_tools",
    "loop_repeat_threshold",
    "keep_last_observations",
    "max_tool_result_bytes",
    "system_prompt_suffix",
    "compaction_rubric",
];

/// An operating policy: named string levers, the unit the flywheel evolves.
///
/// `BTreeMap` rather than `HashMap` so serialization is deterministic —
/// iteration order feeds gate fingerprints and replay, and a nondeterministic
/// ordering would make identical genomes hash differently.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PolicyGenome {
    levers: BTreeMap<String, String>,
}

/// Why a genome could not be applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// The genome names a lever this harness does not implement.
    ///
    /// This must be an error, not a silent skip. A mutation to an unapplied
    /// lever produces a run identical to baseline, which the flywheel would
    /// score as "no effect" and burn generations on — while a scoring artifact
    /// could even promote it. Failing loudly keeps the search honest.
    UnknownLever(String),
    /// The value could not be parsed for that lever's type.
    BadValue { lever: String, value: String },
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolicyError::UnknownLever(name) => write!(
                f,
                "unknown policy lever '{name}' — this harness would ignore it, so the \
                 evaluation would be meaningless. Known levers: {}",
                KNOWN_LEVERS.join(", ")
            ),
            PolicyError::BadValue { lever, value } => {
                write!(f, "lever '{lever}' cannot take value {value:?}")
            }
        }
    }
}

impl std::error::Error for PolicyError {}

impl PolicyGenome {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a lever. Chainable.
    pub fn with(mut self, lever: impl Into<String>, value: impl Into<String>) -> Self {
        self.levers.insert(lever.into(), value.into());
        self
    }

    pub fn get(&self, lever: &str) -> Option<&str> {
        self.levers.get(lever).map(String::as_str)
    }

    pub fn is_empty(&self) -> bool {
        self.levers.is_empty()
    }

    pub fn len(&self) -> usize {
        self.levers.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &String)> {
        self.levers.iter()
    }

    /// Free-text levers, which shape prompts rather than numeric config.
    pub fn system_prompt_suffix(&self) -> Option<&str> {
        self.get("system_prompt_suffix")
    }

    pub fn compaction_rubric(&self) -> Option<&str> {
        self.get("compaction_rubric")
    }

    /// Apply the numeric and boolean levers onto a config.
    ///
    /// Rejects unknown levers (see [`PolicyError::UnknownLever`]). Text levers
    /// are validated as known but applied by the caller, since they belong to
    /// the prompt layer rather than the loop config.
    pub fn apply_to(&self, base: GraphConfig) -> Result<GraphConfig, PolicyError> {
        let mut config = base;
        let mut mask: MaskConfig = config.mask.clone();

        for (lever, value) in &self.levers {
            match lever.as_str() {
                "max_iterations" => config.max_iterations = parse(lever, value)?,
                "parallel_tools" => config.parallel_tools = parse(lever, value)?,
                "max_parallel_tools" => config.max_parallel_tools = parse(lever, value)?,
                "loop_repeat_threshold" => config.loop_repeat_threshold = parse(lever, value)?,
                "keep_last_observations" => mask.keep_last_observations = parse(lever, value)?,
                "max_tool_result_bytes" => mask.max_tool_result_bytes = parse(lever, value)?,
                // Known, but applied at the prompt layer.
                "system_prompt_suffix" | "compaction_rubric" => {}
                other => return Err(PolicyError::UnknownLever(other.to_string())),
            }
        }

        config.mask = mask;
        Ok(config)
    }
}

fn parse<T: std::str::FromStr>(lever: &str, value: &str) -> Result<T, PolicyError> {
    value.trim().parse::<T>().map_err(|_| PolicyError::BadValue {
        lever: lever.to_string(),
        value: value.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Cost-per-win when a policy won nothing.
///
/// **Not `f64::INFINITY`.** JSON has no infinity, so serde emits `null`, and the
/// gate's comparison `candidate.costPerWin > baseline.costPerWin` evaluates
/// `null > n` as `false` in JavaScript — meaning a policy that won nothing
/// would silently *pass* the cost clause. The largest finite double is the
/// honest encoding of "unboundedly bad" and compares correctly on both sides.
pub const COST_PER_WIN_NO_WINS: f64 = f64::MAX;

/// The outcome of one evaluated run.
///
/// Serializable so a headless run can emit it as JSON for the flywheel's
/// `Evaluator` to aggregate — that boundary is a process boundary, not a
/// function call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunOutcome {
    /// Which suite item this run covers, so an aggregator can detect a
    /// missing or duplicated item rather than silently averaging fewer runs.
    #[serde(default)]
    pub item_id: String,
    /// Did the run achieve the task (tests pass, issue resolved)?
    pub succeeded: bool,
    /// Did the run actually change anything?
    ///
    /// A run that ends without committing any change is a **no-op** even when
    /// it reports success — the agent talked itself to a stop. This is the
    /// signal `noop_rate` exists to catch.
    #[serde(rename = "madeChanges")]
    pub made_changes: bool,
    /// Total cost in USD.
    #[serde(rename = "costUsd")]
    pub cost_usd: f64,
    /// Hard safety or security regression. Any `true` blocks promotion.
    #[serde(default)]
    pub regressed: bool,
}

/// The four axes the flywheel's gate decides over.
///
/// Named generically on purpose — the gate is host- and benchmark-agnostic, and
/// projecting rvAgent's meaning onto these axes honestly is the trust boundary
/// for every downstream guarantee.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Score {
    /// Main quality signal — higher is better.
    pub primary: f64,
    /// Fraction of runs that committed nothing — lower is better.
    ///
    /// The default gate requires this to **strictly** improve: a policy earns
    /// promotion by making the executor commit more, not merely score higher.
    /// A policy that raises `primary` while leaving the agent more likely to
    /// end empty has found a scoring artifact, not an improvement.
    #[serde(rename = "noopRate")]
    pub noop_rate: f64,
    /// Resource cost per success — lower is better.
    #[serde(rename = "costPerWin")]
    pub cost_per_win: f64,
    /// Hard safety/security stop.
    pub regressed: bool,
}

impl Score {
    /// Aggregate run outcomes into the four axes.
    ///
    /// An empty set scores as maximally bad rather than perfect: zero runs must
    /// never look like a clean sweep to the gate.
    pub fn from_runs(runs: &[RunOutcome]) -> Self {
        if runs.is_empty() {
            return Self {
                primary: 0.0,
                noop_rate: 1.0,
                cost_per_win: COST_PER_WIN_NO_WINS,
                regressed: false,
            };
        }

        let total = runs.len() as f64;
        let wins = runs.iter().filter(|r| r.succeeded).count();
        let noops = runs.iter().filter(|r| !r.made_changes).count();
        let cost: f64 = runs.iter().map(|r| r.cost_usd).sum();

        Self {
            primary: wins as f64 / total,
            noop_rate: noops as f64 / total,
            cost_per_win: if wins == 0 {
                COST_PER_WIN_NO_WINS
            } else {
                cost / wins as f64
            },
            regressed: runs.iter().any(|r| r.regressed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(succeeded: bool, made_changes: bool, cost_usd: f64) -> RunOutcome {
        RunOutcome {
            item_id: String::new(),
            succeeded,
            made_changes,
            cost_usd,
            regressed: false,
        }
    }

    #[test]
    fn genome_serializes_as_a_flat_string_map() {
        let g = PolicyGenome::new()
            .with("max_iterations", "50")
            .with("system_prompt_suffix", "Be concise.");
        let json = serde_json::to_string(&g).unwrap();
        // Must match the flywheel's Policy = Record<string, string> exactly.
        assert_eq!(
            json,
            r#"{"max_iterations":"50","system_prompt_suffix":"Be concise."}"#
        );
        let back: PolicyGenome = serde_json::from_str(&json).unwrap();
        assert_eq!(back, g);
    }

    #[test]
    fn serialization_order_is_deterministic() {
        let a = PolicyGenome::new().with("zebra", "1").with("alpha", "2");
        let b = PolicyGenome::new().with("alpha", "2").with("zebra", "1");
        // Identical genomes must serialize identically, or fingerprints and
        // replay diverge for no reason.
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }

    #[test]
    fn applies_numeric_and_boolean_levers() {
        let g = PolicyGenome::new()
            .with("max_iterations", "42")
            .with("parallel_tools", "false")
            .with("loop_repeat_threshold", "5")
            .with("keep_last_observations", "3");
        let config = g.apply_to(GraphConfig::default()).unwrap();

        assert_eq!(config.max_iterations, 42);
        assert!(!config.parallel_tools);
        assert_eq!(config.loop_repeat_threshold, 5);
        assert_eq!(config.mask.keep_last_observations, 3);
    }

    #[test]
    fn unknown_lever_is_rejected_not_ignored() {
        let g = PolicyGenome::new().with("nonexistent_knob", "7");
        let err = g.apply_to(GraphConfig::default()).unwrap_err();
        assert_eq!(err, PolicyError::UnknownLever("nonexistent_knob".into()));
        // The message must name the valid set, or the optimizer cannot recover.
        assert!(err.to_string().contains("max_iterations"));
    }

    #[test]
    fn bad_value_is_reported_with_the_lever_name() {
        let g = PolicyGenome::new().with("max_iterations", "not-a-number");
        let err = g.apply_to(GraphConfig::default()).unwrap_err();
        assert!(matches!(err, PolicyError::BadValue { .. }));
        assert!(err.to_string().contains("max_iterations"));
    }

    #[test]
    fn text_levers_are_accepted_and_readable() {
        let g = PolicyGenome::new()
            .with("system_prompt_suffix", "Prefer small diffs.")
            .with("compaction_rubric", "Preserve failing tests.");
        assert!(g.apply_to(GraphConfig::default()).is_ok());
        assert_eq!(g.system_prompt_suffix(), Some("Prefer small diffs."));
        assert_eq!(g.compaction_rubric(), Some("Preserve failing tests."));
    }

    #[test]
    fn empty_genome_leaves_config_untouched() {
        let base = GraphConfig::default();
        let applied = PolicyGenome::new().apply_to(base.clone()).unwrap();
        assert_eq!(applied.max_iterations, base.max_iterations);
        assert_eq!(
            applied.mask.keep_last_observations,
            base.mask.keep_last_observations
        );
    }

    #[test]
    fn score_computes_the_four_axes() {
        let runs = vec![
            run(true, true, 1.0),
            run(true, true, 3.0),
            run(false, true, 2.0),
            run(false, false, 0.5),
        ];
        let s = Score::from_runs(&runs);
        assert_eq!(s.primary, 0.5);
        assert_eq!(s.noop_rate, 0.25);
        assert_eq!(s.cost_per_win, 6.5 / 2.0);
        assert!(!s.regressed);
    }

    #[test]
    fn a_successful_run_that_changed_nothing_still_counts_as_a_noop() {
        // The case the axis exists for: the agent reports success but committed
        // nothing. Scoring that as a win would reward talking over doing.
        let s = Score::from_runs(&[run(true, false, 1.0)]);
        assert_eq!(s.primary, 1.0);
        assert_eq!(s.noop_rate, 1.0);
    }

    #[test]
    fn zero_wins_gives_maximally_bad_cost_per_win_not_zero() {
        let s = Score::from_runs(&[run(false, true, 5.0)]);
        assert_eq!(s.cost_per_win, COST_PER_WIN_NO_WINS);
    }

    #[test]
    fn score_never_serializes_a_non_finite_number() {
        // JSON has no infinity: serde emits null, and the gate reads `null > n`
        // as false in JS, so a zero-win policy would pass the cost clause.
        for runs in [vec![], vec![run(false, true, 5.0)], vec![run(true, true, 1.0)]] {
            let json = serde_json::to_value(Score::from_runs(&runs)).unwrap();
            let cost = json.get("costPerWin").unwrap();
            assert!(
                cost.is_number(),
                "costPerWin must serialize as a number, got {cost}"
            );
            assert!(cost.as_f64().unwrap().is_finite());
        }
    }

    #[test]
    fn empty_run_set_scores_as_bad_not_perfect() {
        let s = Score::from_runs(&[]);
        assert_eq!(s.primary, 0.0);
        assert_eq!(s.noop_rate, 1.0);
        assert_eq!(s.cost_per_win, COST_PER_WIN_NO_WINS);
    }

    #[test]
    fn any_regression_sets_the_hard_stop() {
        let mut bad = run(true, true, 1.0);
        bad.regressed = true;
        let s = Score::from_runs(&[run(true, true, 1.0), bad]);
        assert!(s.regressed);
    }

    #[test]
    fn score_serializes_with_the_gate_field_names() {
        let s = Score::from_runs(&[run(true, true, 2.0)]);
        let json = serde_json::to_value(&s).unwrap();
        // These names are the gate's contract; renaming them silently breaks it.
        assert!(json.get("primary").is_some());
        assert!(json.get("noopRate").is_some());
        assert!(json.get("costPerWin").is_some());
        assert!(json.get("regressed").is_some());
    }


    #[test]
    fn run_outcome_round_trips_across_the_process_boundary() {
        let outcome = RunOutcome {
            item_id: "task-7".into(),
            succeeded: true,
            made_changes: true,
            cost_usd: 0.42,
            regressed: false,
        };
        let json = serde_json::to_string(&outcome).unwrap();
        // Field names are the contract with the JS evaluator; renaming them
        // silently would make every aggregated Score wrong rather than failing.
        assert!(json.contains("\"madeChanges\""));
        assert!(json.contains("\"costUsd\""));
        assert!(json.contains("\"item_id\""));
        let back: RunOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(back, outcome);
    }

    #[test]
    fn run_outcome_tolerates_a_missing_regressed_flag() {
        // An emitter that has nothing to report should not have to say so.
        let json = r#"{"item_id":"a","succeeded":false,"madeChanges":false,"costUsd":0.0}"#;
        let back: RunOutcome = serde_json::from_str(json).unwrap();
        assert!(!back.regressed);
    }

    #[test]
    fn every_known_lever_is_actually_applicable() {
        // Guards against KNOWN_LEVERS drifting out of sync with apply_to.
        for lever in KNOWN_LEVERS {
            let value = match *lever {
                "parallel_tools" => "true",
                "system_prompt_suffix" | "compaction_rubric" => "text",
                _ => "4",
            };
            let g = PolicyGenome::new().with(*lever, value);
            assert!(
                g.apply_to(GraphConfig::default()).is_ok(),
                "declared lever '{lever}' is not handled by apply_to"
            );
        }
    }
}
