//! Stage-level memory diagnostic gate (ADR-325, PIR WP22), informed by
//! **D²ACCI (Diagnostic-Driven Artifact-based Closed-loop Controlled
//! Iteration)** — arXiv:2608.17756, evidence grade A, no upstream code found,
//! so this is a first-party build from the paper's described mechanism, not a
//! port.
//!
//! # What this adds, and what it does not replace
//!
//! `research-gate` (ruvector ADR-282, adopted via ADR-306; the paired-bootstrap
//! statistics in `crates/ruvector-sota-bench/harness/src/statistics.ts`)
//! answers a single **outcome-level** question: *did this whole mutation beat
//! its parent?* It stays the promotion authority for that axis. This module
//! sits **underneath** it and adds a complementary **stage-level** axis:
//! when a memory mutation moves the end-to-end score, *which of the four
//! memory-pipeline stages* — ingestion, retrieval, filtering, generation — is
//! responsible? The outcome-level verdict is consumed here as an input
//! ([`PairedEvaluation`], produced by that harness), never recomputed or
//! weakened.
//!
//! ("Extraction" is one of the paper's *ablated interventions*, not a pipeline
//! stage; the abstract enumerates exactly four stages — ingestion, retrieval,
//! filtering, generation — and this module models those four, no more.)
//!
//! # The joint three-condition promotion bar (ADR-325 Decision §2)
//!
//! A memory-strategy artifact promotes **only when all three hold jointly**:
//!
//! - **(a) paired improvement** — `research-gate`'s outcome-level delta passes;
//! - **(b) no protected-slice regression** — a held-out slice the candidate is
//!   not tuned against does not regress;
//! - **(c) single-stage localizability** — the diagnostic trace attributes the
//!   behavior change to *exactly one* of the four stages.
//!
//! Any one failing blocks promotion. A **result-only log** — an end-to-end
//! outcome with no per-stage attribution — scores as *non-diagnosable*
//! (coverage 0.0) and can never satisfy (c); this is the paper's key result
//! ("diagnostic artifacts reach 98–100% DCR@3 versus 0% for results-only
//! logs"), that stage-attributed traces vastly outperform result-only logs for
//! fault localization.
//!
//! # Immutable versioned policy artifacts (ADR-325 Decision §1)
//!
//! Every memory strategy is a [`MemoryPolicy`] — an immutable, versioned
//! artifact, not a strategy mutated in place. A change is always a *new*
//! versioned candidate ([`PolicySet::with_mutation`] returns a new set, leaving
//! the original untouched). Retrieval algorithms (HNSW, graph traversal, BM25,
//! temporal, attractor, learned; see [`RetrievalStrategy`]) are Darwin
//! candidates registered this way — exactly the cautious posture the paper's
//! own BM25/RRF ablation models: a plausible retrieval strategy that does not
//! clear the bar is held as a monitored feature flag, not promoted.
//!
//! # The "DCR" metric name
//!
//! The paper uses the token *DCR* as a graded observability metric but does
//! **not** expand the acronym; this program does not invent an expansion for
//! it. The Rust surface names the graded metric [`diagnostic_coverage`]
//! descriptively.
//!
//! # Ledger integration (ADR-325 §, PIR WP4 reuse)
//!
//! A promotion is a *governed transition* through the existing WP4
//! [`TransactionalLedger`] — witness-first, no-witness-no-mutation — not a new
//! mutation path. [`apply_gated_promotion`] routes the gate's decision onto the
//! ledger: a promote is `add` + proof-gated `accept` (Accepted); a monitored
//! flag is `add` left `Pending`; a hard block is `ignore` (seen, deliberately
//! not retained), the distinct block reason witnessed in the transition record.

use serde::{Deserialize, Serialize};

use crate::ledger::{ProofGate, TransactionalLedger};
use crate::ops::{fnv1a, LedgerError, WitnessSink};

// ─────────────────────────────────────────────────────────────────────────────
// The four memory-pipeline stages
// ─────────────────────────────────────────────────────────────────────────────

/// The four memory-pipeline stages enumerated by the D²ACCI abstract
/// (verbatim: "its multi-stage pipeline (ingestion, retrieval, filtering,
/// generation)"). Exactly four — "extraction" is an ablated *intervention*,
/// not a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MemoryStage {
    Ingestion,
    Retrieval,
    Filtering,
    Generation,
}

impl MemoryStage {
    /// All four stages, in pipeline order.
    pub const ALL: [MemoryStage; 4] = [
        MemoryStage::Ingestion,
        MemoryStage::Retrieval,
        MemoryStage::Filtering,
        MemoryStage::Generation,
    ];

    /// Stable index into a four-slot, stage-keyed array.
    pub const fn index(self) -> usize {
        match self {
            MemoryStage::Ingestion => 0,
            MemoryStage::Retrieval => 1,
            MemoryStage::Filtering => 2,
            MemoryStage::Generation => 3,
        }
    }

    /// Canonical lowercase name (used in the ledger record's reason text).
    pub const fn as_str(self) -> &'static str {
        match self {
            MemoryStage::Ingestion => "ingestion",
            MemoryStage::Retrieval => "retrieval",
            MemoryStage::Filtering => "filtering",
            MemoryStage::Generation => "generation",
        }
    }
}

/// Well-known retrieval-stage strategies, each a Darwin mutation candidate
/// (ADR-325 Decision §3). A free-form strategy string on [`MemoryPolicy`] keeps
/// the set open; this enum names the ones the ADR calls out explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RetrievalStrategy {
    Hnsw,
    GraphTraversal,
    Bm25,
    Temporal,
    Attractor,
    Learned,
}

impl RetrievalStrategy {
    pub const fn as_str(self) -> &'static str {
        match self {
            RetrievalStrategy::Hnsw => "hnsw",
            RetrievalStrategy::GraphTraversal => "graph-traversal",
            RetrievalStrategy::Bm25 => "bm25",
            RetrievalStrategy::Temporal => "temporal",
            RetrievalStrategy::Attractor => "attractor",
            RetrievalStrategy::Learned => "learned",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Immutable versioned policy artifact
// ─────────────────────────────────────────────────────────────────────────────

/// An immutable, versioned memory-strategy artifact (ADR-325 Decision §1).
///
/// A strategy is never mutated in place: a change is a new `MemoryPolicy` with
/// an incremented `version` and a fresh `content_hash`. Fields are read-only
/// (private + accessors); construct via [`MemoryPolicy::new`] or
/// [`MemoryPolicy::retrieval`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPolicy {
    id: String,
    version: u32,
    stage: MemoryStage,
    strategy: String,
    content_hash: u64,
}

impl MemoryPolicy {
    /// Build a versioned artifact for `stage`. `content` is the canonical bytes
    /// of the strategy's configuration; its FNV-1a digest gives the artifact a
    /// stable identity to diagnose and promote/reject as a unit.
    pub fn new(
        id: impl Into<String>,
        version: u32,
        stage: MemoryStage,
        strategy: impl Into<String>,
        content: &[u8],
    ) -> Self {
        Self {
            id: id.into(),
            version,
            stage,
            strategy: strategy.into(),
            content_hash: fnv1a(content),
        }
    }

    /// Convenience for a retrieval-stage Darwin candidate.
    pub fn retrieval(strategy: RetrievalStrategy, version: u32, content: &[u8]) -> Self {
        Self::new(
            "retrieval",
            version,
            MemoryStage::Retrieval,
            strategy.as_str(),
            content,
        )
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn stage(&self) -> MemoryStage {
        self.stage
    }
    pub fn strategy(&self) -> &str {
        &self.strategy
    }
    pub fn content_hash(&self) -> u64 {
        self.content_hash
    }

    /// Canonical one-line identity written as the ledger entry's content, so a
    /// promotion/rejection is auditable back to the exact artifact version.
    pub fn canonical_string(&self) -> String {
        format!(
            "policy:{}@v{} stage={} strategy={} hash={:016x}",
            self.id,
            self.version,
            self.stage.as_str(),
            self.strategy,
            self.content_hash
        )
    }
}

/// A complete four-stage policy set — the configuration a mutation is evaluated
/// against. A mutation swaps exactly one stage; the set stays immutable
/// (`with_mutation` returns a new set).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicySet {
    policies: [MemoryPolicy; 4],
}

impl PolicySet {
    /// A set covering all four stages. Each policy must sit on its named stage.
    ///
    /// # Panics
    /// If any argument's `stage()` does not match its slot.
    pub fn new(
        ingestion: MemoryPolicy,
        retrieval: MemoryPolicy,
        filtering: MemoryPolicy,
        generation: MemoryPolicy,
    ) -> Self {
        assert_eq!(ingestion.stage(), MemoryStage::Ingestion, "wrong stage");
        assert_eq!(retrieval.stage(), MemoryStage::Retrieval, "wrong stage");
        assert_eq!(filtering.stage(), MemoryStage::Filtering, "wrong stage");
        assert_eq!(generation.stage(), MemoryStage::Generation, "wrong stage");
        Self {
            policies: [ingestion, retrieval, filtering, generation],
        }
    }

    /// The policy currently governing `stage`.
    pub fn get(&self, stage: MemoryStage) -> &MemoryPolicy {
        &self.policies[stage.index()]
    }

    /// Produce a **new** set with `candidate` replacing exactly its own stage's
    /// policy (Darwin candidate application). The receiver is left unchanged,
    /// enforcing artifact immutability at the set level.
    ///
    /// # Errors
    /// [`DiagnosticError::NonMonotonicVersion`] if the candidate does not
    /// strictly supersede the policy it replaces.
    pub fn with_mutation(&self, candidate: MemoryPolicy) -> Result<PolicySet, DiagnosticError> {
        let stage = candidate.stage();
        let current = self.get(stage);
        if candidate.version() <= current.version() {
            return Err(DiagnosticError::NonMonotonicVersion {
                stage,
                current: current.version(),
                candidate: candidate.version(),
            });
        }
        let mut policies = self.policies.clone();
        policies[stage.index()] = candidate;
        Ok(PolicySet { policies })
    }
}

/// Errors from constructing or mutating policy sets.
#[derive(Debug, PartialEq, Eq)]
pub enum DiagnosticError {
    /// A candidate did not strictly supersede the policy it replaces.
    NonMonotonicVersion {
        stage: MemoryStage,
        current: u32,
        candidate: u32,
    },
}

impl std::fmt::Display for DiagnosticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticError::NonMonotonicVersion {
                stage,
                current,
                candidate,
            } => write!(
                f,
                "candidate v{candidate} does not supersede current v{current} on stage {}",
                stage.as_str()
            ),
        }
    }
}

impl std::error::Error for DiagnosticError {}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostic trace + the graded coverage metric
// ─────────────────────────────────────────────────────────────────────────────

/// Per-stage attribution signal within a diagnostic trace.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StageSignal {
    pub stage: MemoryStage,
    /// Whether this stage's behavior was actually observed for this outcome.
    /// An unattributed stage contributes no localization mass.
    pub attributed: bool,
    /// Graded share of the outcome's behavior change localized to this stage,
    /// in `[0, 1]` (negatives are clamped to 0).
    pub localization: f32,
}

impl StageSignal {
    pub fn new(stage: MemoryStage, localization: f32) -> Self {
        Self {
            stage,
            attributed: true,
            localization,
        }
    }
}

/// The evidence attached to a memory outcome for stage localization.
///
/// The gate's central discriminator: a [`DiagnosticTrace::ResultOnly`] log
/// carries only the end-to-end delta and can never localize a stage (coverage
/// 0.0, non-diagnosable); a [`DiagnosticTrace::Staged`] trace carries per-stage
/// signals that can concentrate the change on one stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiagnosticTrace {
    /// End-to-end outcome with no per-stage attribution — the baseline the
    /// paper measures diagnostic traces against (0% localization).
    ResultOnly { outcome_delta: f32 },
    /// Per-stage attribution for the outcome.
    Staged {
        signals: Vec<StageSignal>,
        outcome_delta: f32,
    },
}

impl DiagnosticTrace {
    /// Build a staged trace from stage/localization pairs.
    pub fn staged(outcome_delta: f32, signals: impl IntoIterator<Item = StageSignal>) -> Self {
        DiagnosticTrace::Staged {
            signals: signals.into_iter().collect(),
            outcome_delta,
        }
    }
}

/// The dominant stage must hold a **strict majority** of the localization mass
/// (coverage *strictly greater* than this) for an outcome to count as localized
/// to *exactly one* stage — D²ACCI's headline property. A genuine single-stage
/// trace yields coverage near 1.0; four equally-attributed stages yield 0.25
/// (blocked); and a two-way 50/50 tie yields exactly 0.5, which is **not** a
/// strict majority and so blocks as `NonLocalizable` rather than being
/// arbitrarily attributed to one of the two tied stages.
pub const LOCALIZATION_THRESHOLD: f32 = 0.5;

/// The paper's graded **DCR** observability metric (the paper uses the token
/// *DCR* without expanding the acronym; this program does not invent one).
///
/// Returns the *concentration* of localization mass on the single most-implicated
/// stage, in `[0, 1]`:
///
/// - a **result-only** log scores `0.0` — non-diagnosable, matching the paper's
///   "0% for results-only logs";
/// - a trace whose mass sits entirely on one stage scores `1.0` (the paper's
///   98–100% DCR@3 regime);
/// - a trace spread evenly across all four stages scores `0.25`.
pub fn diagnostic_coverage(trace: &DiagnosticTrace) -> f32 {
    match trace {
        DiagnosticTrace::ResultOnly { .. } => 0.0,
        DiagnosticTrace::Staged { signals, .. } => {
            let (total, max) = signals
                .iter()
                .filter(|s| s.attributed)
                .map(|s| s.localization.max(0.0))
                .fold((0.0f32, 0.0f32), |(sum, mx), v| (sum + v, mx.max(v)));
            if total <= 0.0 {
                0.0
            } else {
                (max / total).clamp(0.0, 1.0)
            }
        }
    }
}

/// The stage a trace localizes to, or `None` when the outcome is not
/// localizable to exactly one stage — i.e. the dominant stage does not hold a
/// *strict majority* of the localization mass (coverage at or below
/// [`LOCALIZATION_THRESHOLD`]), or the trace is a result-only log.
///
/// The strict-majority test is deliberate: a two-way 50/50 tie yields coverage
/// exactly 0.5 and blocks as `NonLocalizable` rather than being arbitrarily
/// attributed to one of the two tied stages — D²ACCI localizes to *exactly one*
/// stage, and two equally-implicated stages are not one stage.
pub fn localized_stage(trace: &DiagnosticTrace) -> Option<MemoryStage> {
    let DiagnosticTrace::Staged { signals, .. } = trace else {
        return None;
    };
    if diagnostic_coverage(trace) <= LOCALIZATION_THRESHOLD {
        return None;
    }
    signals
        .iter()
        .filter(|s| s.attributed && s.localization > 0.0)
        .max_by(|a, b| a.localization.total_cmp(&b.localization))
        .map(|s| s.stage)
}

// ─────────────────────────────────────────────────────────────────────────────
// Outcome-level inputs (consumed from research-gate, never recomputed here)
// ─────────────────────────────────────────────────────────────────────────────

/// The outcome axis of a paired-bootstrap decision, as produced by
/// `research-gate` (`statistics.ts::pairedBootstrapDecision`). This gate
/// *consumes* the verdict; it never recomputes or overrides it — the
/// outcome-level gate remains the promotion authority for this axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PairedOutcome {
    /// Lower 95% bound clears the minimum effect — genuine improvement.
    Pass,
    /// Upper 95% bound at or below zero — a regression.
    Fail,
    /// Interval straddles the effect threshold — not yet significant.
    Inconclusive,
}

/// A research-gate paired-evaluation result carried into the stage gate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PairedEvaluation {
    pub mean_delta: f32,
    pub lower95: f32,
    pub outcome: PairedOutcome,
}

impl PairedEvaluation {
    pub fn improved(&self) -> bool {
        self.outcome == PairedOutcome::Pass
    }
}

/// Held-out protected-slice replay result (ADR-325 Decision §2(b)). The slice
/// is isolated from whatever the candidate was tuned/evaluated against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtectedSliceResult {
    /// Per-workload score deltas (candidate − baseline) on the protected slice.
    pub deltas: Vec<f32>,
    /// A workload counts as regressed when its delta drops below `-tolerance`.
    pub tolerance: f32,
}

impl ProtectedSliceResult {
    pub fn new(deltas: impl IntoIterator<Item = f32>, tolerance: f32) -> Self {
        Self {
            deltas: deltas.into_iter().collect(),
            tolerance,
        }
    }

    /// True if any protected workload regressed beyond tolerance.
    pub fn regressed(&self) -> bool {
        self.deltas.iter().any(|&d| d < -self.tolerance)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The joint gate
// ─────────────────────────────────────────────────────────────────────────────

/// Why a candidate did not promote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BlockReason {
    /// (a) research-gate's outcome-level delta did not clear the bar.
    NoPairedImprovement,
    /// (b) a held-out protected workload regressed.
    ProtectedRegression,
    /// (c) a staged trace could not localize the change to one stage.
    NonLocalizable,
    /// (c) a result-only log with no per-stage attribution at all.
    NonDiagnosable,
}

impl BlockReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            BlockReason::NoPairedImprovement => "no-paired-improvement",
            BlockReason::ProtectedRegression => "protected-regression",
            BlockReason::NonLocalizable => "non-localizable",
            BlockReason::NonDiagnosable => "non-diagnosable",
        }
    }
}

/// The gate's verdict for a candidate mutation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PromotionDecision {
    /// All three conditions held: promote, with the localized stage and the
    /// graded coverage attached.
    Promote { stage: MemoryStage, coverage: f32 },
    /// A plausible candidate that did not clear the outcome-level bar — held as
    /// a *monitored feature flag*, not promoted (the paper's BM25/RRF worked
    /// example). Retained for observation, never served as accepted knowledge.
    MonitoredFlag { reason: BlockReason, coverage: f32 },
    /// A hard block: the candidate is skipped, with a distinct reason.
    Blocked { reason: BlockReason, coverage: f32 },
}

impl PromotionDecision {
    pub fn is_promoted(&self) -> bool {
        matches!(self, PromotionDecision::Promote { .. })
    }
    /// The graded coverage metric attached to every decision record.
    pub fn coverage(&self) -> f32 {
        match self {
            PromotionDecision::Promote { coverage, .. }
            | PromotionDecision::MonitoredFlag { coverage, .. }
            | PromotionDecision::Blocked { coverage, .. } => *coverage,
        }
    }
}

/// Evaluate the joint three-condition promotion bar (ADR-325 Decision §2),
/// underneath research-gate's outcome-level verdict.
///
/// Order of judgement — each condition independently blocks:
/// 1. **(a)** paired improvement not `Pass` → **monitored feature flag** (a
///    plausible-but-unproven candidate is held, not hard-rejected — the paper's
///    BM25/RRF posture).
/// 2. **(b)** protected slice regressed → **hard block** `ProtectedRegression`
///    (the silent-cross-workload-regression failure mode).
/// 3. **(c)** trace non-diagnosable (result-only) → **hard block**
///    `NonDiagnosable`; or staged-but-not-single-stage → **hard block**
///    `NonLocalizable`.
///
/// Only when all three pass does the candidate promote, tagged with its
/// localized stage and coverage.
pub fn evaluate_promotion(
    paired: &PairedEvaluation,
    protected: &ProtectedSliceResult,
    trace: &DiagnosticTrace,
) -> PromotionDecision {
    let coverage = diagnostic_coverage(trace);

    // (a) outcome-level improvement is research-gate's authority. A candidate
    // that does not clear it is a monitored flag, not a hard rejection.
    if !paired.improved() {
        return PromotionDecision::MonitoredFlag {
            reason: BlockReason::NoPairedImprovement,
            coverage,
        };
    }

    // (b) protected-slice non-regression.
    if protected.regressed() {
        return PromotionDecision::Blocked {
            reason: BlockReason::ProtectedRegression,
            coverage,
        };
    }

    // (c) single-stage localizability. Result-only logs are non-diagnosable by
    // construction (coverage 0.0) — the paper's key discriminator.
    match localized_stage(trace) {
        Some(stage) => PromotionDecision::Promote { stage, coverage },
        None => {
            let reason = match trace {
                DiagnosticTrace::ResultOnly { .. } => BlockReason::NonDiagnosable,
                DiagnosticTrace::Staged { .. } => BlockReason::NonLocalizable,
            };
            PromotionDecision::Blocked { reason, coverage }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Governed application through the WP4 TransactionalLedger
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of routing a gate decision onto the ledger.
#[derive(Debug, Clone)]
pub struct GatedPromotion {
    pub decision: PromotionDecision,
    /// The ledger entry created for the artifact, if any: `Some` for a promote
    /// (ends `Accepted`) or a monitored flag (stays `Pending`); `None` for a
    /// hard block (witnessed via `ignore`, no entry retained).
    pub ledger_entry: Option<u64>,
    /// The graded coverage metric attached to the promotion record.
    pub coverage: f32,
}

/// Apply a gate `decision` for `candidate` as a **governed transition** through
/// the existing WP4 [`TransactionalLedger`] (witness-first,
/// no-witness-no-mutation) — reusing the ledger, not rebuilding a mutation
/// path. The distinct decision reason and the graded coverage are written into
/// the transition record's reason text, so every promotion, monitored flag, and
/// block is auditable.
///
/// - **Promote** → `add` (Pending) then proof-gated `accept` (Accepted).
/// - **MonitoredFlag** → `add`, left `Pending` — retained, never accepted.
/// - **Blocked** → `ignore` — seen and deliberately not retained, the block
///   reason witnessed; no entry created.
pub fn apply_gated_promotion<S: WitnessSink, G: ProofGate>(
    ledger: &mut TransactionalLedger<S, G>,
    candidate: &MemoryPolicy,
    decision: PromotionDecision,
    actor: &str,
) -> Result<GatedPromotion, LedgerError> {
    let content = candidate.canonical_string();
    let coverage = decision.coverage();
    let ledger_entry = match decision {
        PromotionDecision::Promote { stage, coverage } => {
            let reason = format!(
                "d2acci-promote: stage={} dcr={coverage:.4} (paired+protected+localizable)",
                stage.as_str()
            );
            let id = ledger.add(content, &[], actor, &reason)?;
            ledger.accept(id, actor, &reason)?;
            Some(id)
        }
        PromotionDecision::MonitoredFlag { reason, coverage } => {
            let note = format!(
                "d2acci-monitored-flag: reason={} dcr={coverage:.4} (held Pending, not promoted)",
                reason.as_str()
            );
            let id = ledger.add(content, &[], actor, &note)?;
            Some(id)
        }
        PromotionDecision::Blocked { reason, coverage } => {
            let note = format!(
                "d2acci-blocked: reason={} dcr={coverage:.4} (skipped, not promoted)",
                reason.as_str()
            );
            ledger.ignore(&content, actor, &note)?;
            None
        }
    };
    Ok(GatedPromotion {
        decision,
        ledger_entry,
        coverage,
    })
}

/// Evaluate the joint bar and apply the resulting decision in one governed step.
pub fn gate_and_apply<S: WitnessSink, G: ProofGate>(
    ledger: &mut TransactionalLedger<S, G>,
    candidate: &MemoryPolicy,
    paired: &PairedEvaluation,
    protected: &ProtectedSliceResult,
    trace: &DiagnosticTrace,
    actor: &str,
) -> Result<GatedPromotion, LedgerError> {
    let decision = evaluate_promotion(paired, protected, trace);
    apply_gated_promotion(ledger, candidate, decision, actor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::{AlwaysAdmitGate, TransactionalLedger};
    use crate::ops::{LedgerState, MemoryWitnessLog};

    fn stage_policy(stage: MemoryStage, v: u32, strat: &str) -> MemoryPolicy {
        MemoryPolicy::new(stage.as_str(), v, stage, strat, strat.as_bytes())
    }

    fn baseline_set() -> PolicySet {
        PolicySet::new(
            stage_policy(MemoryStage::Ingestion, 1, "chunk-512"),
            stage_policy(MemoryStage::Retrieval, 1, "hnsw"),
            stage_policy(MemoryStage::Filtering, 1, "forget-guard"),
            stage_policy(MemoryStage::Generation, 1, "template-v1"),
        )
    }

    fn passing() -> PairedEvaluation {
        PairedEvaluation {
            mean_delta: 0.031,
            lower95: 0.012,
            outcome: PairedOutcome::Pass,
        }
    }

    fn no_regression() -> ProtectedSliceResult {
        ProtectedSliceResult::new([0.004, 0.001, 0.010], 0.01)
    }

    /// A trace concentrating the change on one stage (near-1.0 coverage).
    fn localizable_to(stage: MemoryStage) -> DiagnosticTrace {
        let signals = MemoryStage::ALL
            .iter()
            .map(|&s| StageSignal::new(s, if s == stage { 0.90 } else { 0.03 }));
        DiagnosticTrace::staged(0.031, signals)
    }

    // ── 1. Immutable versioned four-stage policy set ─────────────────────────

    #[test]
    fn four_stage_policy_set_is_immutable() {
        let base = baseline_set();
        // A retrieval Darwin candidate: a *new* version, not an in-place edit.
        let candidate = MemoryPolicy::retrieval(RetrievalStrategy::GraphTraversal, 2, b"graph-cfg");
        let mutated = base.with_mutation(candidate.clone()).unwrap();

        // Original set untouched (immutability).
        assert_eq!(base.get(MemoryStage::Retrieval).version(), 1);
        assert_eq!(base.get(MemoryStage::Retrieval).strategy(), "hnsw");
        // New set carries the superseding artifact; other stages unchanged.
        assert_eq!(mutated.get(MemoryStage::Retrieval).version(), 2);
        assert_eq!(
            mutated.get(MemoryStage::Retrieval).strategy(),
            "graph-traversal"
        );
        assert_eq!(
            mutated.get(MemoryStage::Ingestion),
            base.get(MemoryStage::Ingestion)
        );
        // Distinct content hashes = distinct diagnosable identities.
        assert_ne!(
            base.get(MemoryStage::Retrieval).content_hash(),
            mutated.get(MemoryStage::Retrieval).content_hash()
        );
        // A non-superseding version is refused.
        assert_eq!(
            base.with_mutation(MemoryPolicy::retrieval(RetrievalStrategy::Bm25, 1, b"x")),
            Err(DiagnosticError::NonMonotonicVersion {
                stage: MemoryStage::Retrieval,
                current: 1,
                candidate: 1,
            })
        );
    }

    // ── 2a. improve + no regression + localizable → PROMOTED (the proof) ─────

    #[test]
    fn improve_no_regression_localizable_promotes() {
        let candidate = MemoryPolicy::retrieval(RetrievalStrategy::Learned, 2, b"learned-cfg");
        let trace = localizable_to(MemoryStage::Retrieval);

        let decision = evaluate_promotion(&passing(), &no_regression(), &trace);
        // Localizes to exactly the retrieval stage, with high coverage.
        match decision {
            PromotionDecision::Promote { stage, coverage } => {
                assert_eq!(stage, MemoryStage::Retrieval);
                assert!(coverage >= LOCALIZATION_THRESHOLD, "coverage {coverage}");
                assert!(coverage > 0.9, "single-stage coverage should approach 1.0");
            }
            other => panic!("expected Promote, got {other:?}"),
        }

        // Governed application: add + proof-gated accept through the WP4 ledger.
        let mut ledger =
            TransactionalLedger::new(MemoryWitnessLog::default(), AlwaysAdmitGate::default());
        let applied = gate_and_apply(
            &mut ledger,
            &candidate,
            &passing(),
            &no_regression(),
            &trace,
            "wp22-gate",
        )
        .unwrap();

        let id = applied
            .ledger_entry
            .expect("promotion creates a ledger entry");
        assert!(applied.decision.is_promoted());
        // The promoted artifact is Accepted knowledge.
        assert_eq!(ledger.state_of(id), Some(LedgerState::Accepted));
        // Witness-first discipline held: chain verifies, and an Accept was recorded.
        assert!(ledger.witness_sink().verify_chain());
        assert!(ledger
            .history()
            .iter()
            .any(|r| matches!(r.kind, crate::ops::TransitionKind::Accept)));
    }

    // ── 2b. failure not attributable to one stage → BLOCKED ──────────────────

    #[test]
    fn non_localizable_change_is_blocked() {
        // Localization spread evenly across all four stages: coverage 0.25.
        let diffuse = DiagnosticTrace::staged(
            -0.02,
            MemoryStage::ALL.iter().map(|&s| StageSignal::new(s, 0.25)),
        );
        assert!((diagnostic_coverage(&diffuse) - 0.25).abs() < 1e-6);
        assert_eq!(localized_stage(&diffuse), None);

        let decision = evaluate_promotion(&passing(), &no_regression(), &diffuse);
        assert_eq!(
            decision,
            PromotionDecision::Blocked {
                reason: BlockReason::NonLocalizable,
                coverage: 0.25,
            }
        );
        assert!(!decision.is_promoted());

        // Applied as a witnessed skip — no entry retained, nothing accepted.
        let candidate = MemoryPolicy::retrieval(RetrievalStrategy::Attractor, 2, b"attr");
        let mut ledger = TransactionalLedger::unguarded();
        let applied =
            apply_gated_promotion(&mut ledger, &candidate, decision, "wp22-gate").unwrap();
        assert!(applied.ledger_entry.is_none());
        assert!(ledger.entries_in(LedgerState::Accepted).is_empty());
    }

    // ── 2c. strict majority: a 50/50 tie blocks; a clear dominant promotes ───

    #[test]
    fn strict_majority_required_two_way_tie_blocks() {
        // Exactly two stages, equally implicated (coverage 0.5). Not a strict
        // majority → NonLocalizable, never arbitrarily attributed to one.
        let tie = DiagnosticTrace::staged(
            0.02,
            [
                StageSignal::new(MemoryStage::Retrieval, 0.5),
                StageSignal::new(MemoryStage::Filtering, 0.5),
            ],
        );
        assert!((diagnostic_coverage(&tie) - 0.5).abs() < 1e-6);
        assert_eq!(localized_stage(&tie), None);
        assert_eq!(
            evaluate_promotion(&passing(), &no_regression(), &tie),
            PromotionDecision::Blocked {
                reason: BlockReason::NonLocalizable,
                coverage: 0.5,
            }
        );

        // A clear dominant stage (0.6 vs 0.4, coverage 0.6 > 0.5) IS a strict
        // majority → promotes, localized to the dominant stage.
        let dominant = DiagnosticTrace::staged(
            0.02,
            [
                StageSignal::new(MemoryStage::Retrieval, 0.6),
                StageSignal::new(MemoryStage::Filtering, 0.4),
            ],
        );
        assert!((diagnostic_coverage(&dominant) - 0.6).abs() < 1e-6);
        assert_eq!(localized_stage(&dominant), Some(MemoryStage::Retrieval));
        match evaluate_promotion(&passing(), &no_regression(), &dominant) {
            PromotionDecision::Promote { stage, coverage } => {
                assert_eq!(stage, MemoryStage::Retrieval);
                assert!((coverage - 0.6).abs() < 1e-6);
            }
            other => panic!("expected Promote on strict majority, got {other:?}"),
        }
    }

    // ── 3. result-only log → non-diagnosable → BLOCKED ───────────────────────

    #[test]
    fn result_only_log_is_non_diagnosable_and_blocks() {
        // Even with paired improvement AND no protected regression, a
        // result-only log cannot localize a stage — the paper's 0% baseline.
        let result_only = DiagnosticTrace::ResultOnly {
            outcome_delta: 0.031,
        };
        assert_eq!(diagnostic_coverage(&result_only), 0.0);
        assert_eq!(localized_stage(&result_only), None);

        let decision = evaluate_promotion(&passing(), &no_regression(), &result_only);
        assert_eq!(
            decision,
            PromotionDecision::Blocked {
                reason: BlockReason::NonDiagnosable,
                coverage: 0.0,
            }
        );

        let candidate = MemoryPolicy::retrieval(RetrievalStrategy::Temporal, 2, b"temporal");
        let mut ledger =
            TransactionalLedger::new(MemoryWitnessLog::default(), AlwaysAdmitGate::default());
        let applied =
            apply_gated_promotion(&mut ledger, &candidate, decision, "wp22-gate").unwrap();
        assert!(applied.ledger_entry.is_none());
        assert!(ledger.entries_in(LedgerState::Accepted).is_empty());
        // The block decision itself is still witnessed (governed skip).
        assert!(ledger.witness_sink().verify_chain());
        assert!(!ledger.witness_sink().records.is_empty());
    }

    // ── 4. BM25/RRF worked example → monitored flag, not promoted ────────────

    #[test]
    fn bm25_rrf_candidate_stays_monitored_flag() {
        // A plausible retrieval strategy whose paired delta does NOT clear the
        // bar (inconclusive). Even with a perfectly localizable trace, it is
        // held as a monitored feature flag — the gate correctly declining to
        // promote a weak candidate.
        let bm25 = MemoryPolicy::retrieval(RetrievalStrategy::Bm25, 2, b"bm25-rrf");
        let inconclusive = PairedEvaluation {
            mean_delta: 0.004,
            lower95: -0.002,
            outcome: PairedOutcome::Inconclusive,
        };
        let trace = localizable_to(MemoryStage::Retrieval);

        let decision = evaluate_promotion(&inconclusive, &no_regression(), &trace);
        assert_eq!(
            decision,
            PromotionDecision::MonitoredFlag {
                reason: BlockReason::NoPairedImprovement,
                coverage: decision.coverage(),
            }
        );
        assert!(!decision.is_promoted());

        // Retained Pending (monitored), never accepted.
        let mut ledger = TransactionalLedger::unguarded();
        let applied = apply_gated_promotion(&mut ledger, &bm25, decision, "wp22-gate").unwrap();
        let id = applied
            .ledger_entry
            .expect("monitored flag is retained as a candidate");
        assert_eq!(ledger.state_of(id), Some(LedgerState::Pending));
        assert!(ledger.entries_in(LedgerState::Accepted).is_empty());
    }

    // ── 5. protected-slice regression hard-blocks even with improvement ──────

    #[test]
    fn protected_regression_blocks() {
        let regressing = ProtectedSliceResult::new([0.02, -0.05, 0.01], 0.01);
        assert!(regressing.regressed());
        let trace = localizable_to(MemoryStage::Filtering);

        let decision = evaluate_promotion(&passing(), &regressing, &trace);
        assert_eq!(
            decision,
            PromotionDecision::Blocked {
                reason: BlockReason::ProtectedRegression,
                coverage: decision.coverage(),
            }
        );
    }

    // ── 6. the graded metric itself ──────────────────────────────────────────

    #[test]
    fn diagnostic_coverage_grades_localization() {
        // Result-only: 0.0 (non-diagnosable).
        assert_eq!(
            diagnostic_coverage(&DiagnosticTrace::ResultOnly { outcome_delta: 0.1 }),
            0.0
        );
        // Single attributed stage: 1.0.
        let single = DiagnosticTrace::staged(0.1, [StageSignal::new(MemoryStage::Generation, 0.7)]);
        assert!((diagnostic_coverage(&single) - 1.0).abs() < 1e-6);
        assert_eq!(localized_stage(&single), Some(MemoryStage::Generation));
        // Two equal stages: coverage 0.5 — NOT a strict majority, so this is a
        // two-way tie and does NOT localize to one stage (blocks downstream).
        let two = DiagnosticTrace::staged(
            0.1,
            [
                StageSignal::new(MemoryStage::Retrieval, 0.4),
                StageSignal::new(MemoryStage::Filtering, 0.4),
            ],
        );
        assert!((diagnostic_coverage(&two) - 0.5).abs() < 1e-6);
        assert_eq!(localized_stage(&two), None, "a 50/50 tie is not one stage");
        // Unattributed stages contribute no mass.
        let unattributed = DiagnosticTrace::staged(
            0.1,
            [
                StageSignal {
                    stage: MemoryStage::Ingestion,
                    attributed: false,
                    localization: 5.0,
                },
                StageSignal::new(MemoryStage::Retrieval, 0.9),
            ],
        );
        assert!((diagnostic_coverage(&unattributed) - 1.0).abs() < 1e-6);
        assert_eq!(localized_stage(&unattributed), Some(MemoryStage::Retrieval));
    }
}
