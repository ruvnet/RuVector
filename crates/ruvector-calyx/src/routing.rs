//! Adaptive lens routing (the retrieval analogue of model routing).
//!
//! Instead of always measuring every lens, the router consults lenses
//! **cheapest-first** and stops as soon as the *calibrated* confidence clears a
//! threshold — escalating to the expensive lenses only when the cheap ones are
//! not decisive, and abstaining when even the full panel is not. Every decision
//! emits a [`Witness`]: which lenses were consulted, their tops, the confidence,
//! the action, and why it escalated or abstained.

use std::collections::BTreeMap;

use crate::calibrate::{raw_confidence, Calibrator, ConfidenceSignals};
use crate::constellation::ConstellationStore;
use crate::engine::Query;
use crate::fusion::{weighted_rrf, DEFAULT_RRF_K};
use crate::ledger::grounded_path;
use crate::loom::{min_agreement, top_dissenter};
use crate::ward::GuardProfile;

/// How the router consults lenses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingMode {
    /// Always consult every lens in `order` (the un-routed baseline).
    BruteForce,
    /// Always consult the first `max_lenses` in `order` (fixed, non-adaptive).
    Static { max_lenses: usize },
    /// Consult cheapest-first, stopping early once calibrated confidence clears
    /// `stop_threshold` (per-query adaptive).
    Adaptive,
}

/// Router thresholds.
#[derive(Debug, Clone)]
pub struct RoutingConfig {
    /// Stop consulting more lenses once calibrated confidence ≥ this.
    pub stop_threshold: f32,
    /// Emit an answer only if final calibrated confidence ≥ this (else abstain).
    pub answer_threshold: f32,
    /// Require at least this many corroborating lenses before committing to an
    /// answer or stopping early — "don't trust a lone lens". Clamped to ≥1.
    pub min_lenses_to_answer: usize,
    /// Top-k per lens.
    pub top_k: usize,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            stop_threshold: 0.80,
            answer_threshold: 0.55,
            min_lenses_to_answer: 2,
            top_k: 10,
        }
    }
}

/// The provenance of one routed decision.
#[derive(Debug, Clone)]
pub struct Witness {
    pub lenses_consulted: Vec<String>,
    /// The fused top candidate, recorded even when the router abstains (so
    /// calibration and evaluation can score the decision that *would* be made).
    pub top_candidate: Option<u64>,
    /// `(lens, top_id, top_score)` per consulted lens.
    pub per_lens_top: Vec<(String, u64, f32)>,
    pub signals: ConfidenceSignals,
    pub confidence: f32,
    /// "answer" | "abstain".
    pub action: &'static str,
    pub escalation_reason: String,
    /// Summed measurement cost (µs) of the consulted lenses.
    pub cost_us: f32,
    /// Summed measurement latency (µs) of the consulted lenses.
    pub latency_us: f32,
}

/// The outcome of routing one query.
#[derive(Debug, Clone)]
pub struct RoutingOutcome {
    /// `Some(id)` if answered, `None` if abstained.
    pub answered: Option<u64>,
    pub confidence: f32,
    pub witness: Witness,
}

/// Route one query. `order` is lens names sorted cheapest-first.
pub fn route(
    store: &ConstellationStore,
    order: &[String],
    query: &Query,
    calibrator: &Calibrator,
    cfg: &RoutingConfig,
    guard: &GuardProfile,
    mode: RoutingMode,
) -> RoutingOutcome {
    let planned = match mode {
        RoutingMode::BruteForce => order.len(),
        RoutingMode::Static { max_lenses } => max_lenses.clamp(1, order.len()),
        RoutingMode::Adaptive => order.len(),
    };

    let mut rankings: Vec<Vec<(u64, f32)>> = Vec::new();
    let mut consulted: Vec<String> = Vec::new();
    let mut per_lens_top: Vec<(String, u64, f32)> = Vec::new();
    let mut cost_us = 0.0f32;
    let mut latency_us = 0.0f32;

    let mut confidence = 0.0f32;
    let mut signals = ConfidenceSignals::default();
    let mut fused: Vec<(u64, f32)> = Vec::new();
    let mut escalation_reason = String::from("full-panel");

    for (step, lens_name) in order.iter().take(planned).enumerate() {
        let Some(m) = store.lenses().iter().find(|l| &l.name == lens_name) else {
            continue;
        };
        let Some(qv) = query.slots.get(lens_name) else {
            continue;
        };
        let ranking = store.search_lens(lens_name, qv, cfg.top_k);
        if let Some(&(id, score)) = ranking.first() {
            per_lens_top.push((lens_name.clone(), id, score));
        }
        rankings.push(ranking);
        consulted.push(lens_name.clone());
        cost_us += m.cost_us;
        latency_us += m.latency_us;

        let weights = vec![1.0f32; rankings.len()];
        fused = weighted_rrf(&rankings, &weights, DEFAULT_RRF_K);
        signals = compute_signals(store, query, &consulted, &fused, guard);
        confidence = calibrator.calibrate(raw_confidence(&signals));

        // Adaptive early-exit: stop once we are confident enough — but only
        // after enough lenses corroborate (never commit on a single lens).
        if mode == RoutingMode::Adaptive
            && consulted.len() >= cfg.min_lenses_to_answer.max(1)
            && confidence >= cfg.stop_threshold
            && step + 1 < order.len()
        {
            escalation_reason = format!("early-exit@{}lenses(conf={:.2})", consulted.len(), confidence);
            break;
        }
    }

    // Decide: fail closed on grounding, then apply the confidence gate.
    let top = fused.first().copied();
    let grounded = top
        .and_then(|(id, _)| store.get(id))
        .map(|rec| grounded_path(rec.id, &rec.anchors, guard.min_grounding_weight).is_some())
        .unwrap_or(false);

    let (answered, action) = match top {
        Some((id, _))
            if consulted.len() >= cfg.min_lenses_to_answer.max(1)
                && confidence >= cfg.answer_threshold
                && signals.cross_lens_agreement >= guard.min_cross_lens_agreement
                && (!guard.require_grounding || grounded) =>
        {
            (Some(id), "answer")
        }
        _ => {
            if escalation_reason == "full-panel" {
                escalation_reason = format!("abstain(conf={:.2}<{:.2})", confidence, cfg.answer_threshold);
            }
            (None, "abstain")
        }
    };

    RoutingOutcome {
        answered,
        confidence,
        witness: Witness {
            lenses_consulted: consulted,
            top_candidate: top.map(|(id, _)| id),
            per_lens_top,
            signals,
            confidence,
            action,
            escalation_reason,
            cost_us,
            latency_us,
        },
    }
}

/// Confidence signals over the lenses consulted so far.
fn compute_signals(
    store: &ConstellationStore,
    query: &Query,
    consulted: &[String],
    fused: &[(u64, f32)],
    guard: &GuardProfile,
) -> ConfidenceSignals {
    let top_score = fused.first().map(|&(_, s)| s).unwrap_or(0.0);
    let second = fused.get(1).map(|&(_, s)| s).unwrap_or(0.0);
    let margin_ratio = if top_score > 1e-9 {
        (top_score - second) / top_score
    } else {
        0.0
    };

    // Restrict the query's slots to the consulted lenses for agreement/dissent.
    let sub: BTreeMap<String, Vec<f32>> = query
        .slots
        .iter()
        .filter(|(k, _)| consulted.iter().any(|c| c == *k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    let (agreement, contradiction, grounding) = match fused.first().and_then(|&(id, _)| store.get(id)) {
        Some(rec) => {
            let agr = min_agreement(&sub, rec);
            let contra = top_dissenter(&sub, rec).map(|(_, gap)| gap).unwrap_or(0.0);
            let grd = grounded_path(rec.id, &rec.anchors, guard.min_grounding_weight)
                .map(|g| g.best_weight)
                .unwrap_or(0.0);
            (agr, contra, grd)
        }
        None => (0.0, 0.0, 0.0),
    };

    ConfidenceSignals {
        top_score,
        margin_ratio,
        cross_lens_agreement: agreement,
        grounding,
        contradiction,
        lens_count: consulted.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constellation::Constellation;
    use crate::ledger::{Anchor, AnchorKind};
    use crate::lens::{LensKind, LensManifest};

    fn store() -> ConstellationStore {
        let lenses = vec![
            LensManifest::new("cheap", "v1", LensKind::Structural, 2, 1.0, 1.0),
            LensManifest::new("pricey", "v1", LensKind::DenseSemantic, 2, 10.0, 10.0),
        ];
        let mut s = ConstellationStore::new(lenses);
        s.insert(
            Constellation::new(1)
                .with_slot("cheap", vec![1.0, 0.0])
                .with_slot("pricey", vec![1.0, 0.0])
                .with_anchor(Anchor::new(AnchorKind::PassedTest, "t1", 0.9)),
        )
        .unwrap();
        s.insert(
            Constellation::new(2)
                .with_slot("cheap", vec![0.0, 1.0])
                .with_slot("pricey", vec![0.0, 1.0])
                .with_anchor(Anchor::new(AnchorKind::PassedTest, "t2", 0.9)),
        )
        .unwrap();
        s
    }

    #[test]
    fn adaptive_stops_early_when_cheap_lens_is_decisive() {
        let s = store();
        let order = vec!["cheap".to_string(), "pricey".to_string()];
        let q = Query::new()
            .with_slot("cheap", vec![1.0, 0.0])
            .with_slot("pricey", vec![1.0, 0.0]);
        // A calibrator that trusts high raw confidence.
        let cal = Calibrator::identity(10);
        let cfg = RoutingConfig {
            stop_threshold: 0.5,
            answer_threshold: 0.4,
            min_lenses_to_answer: 1, // this test checks single-lens early-exit
            top_k: 5,
        };
        let out = route(&s, &order, &q, &cal, &cfg, &GuardProfile::default(), RoutingMode::Adaptive);
        assert_eq!(out.answered, Some(1));
        // Should have stopped after the single cheap lens (skipped the pricey one).
        assert_eq!(out.witness.lenses_consulted, vec!["cheap".to_string()]);
        assert!(out.witness.cost_us < 2.0);
    }

    #[test]
    fn brute_force_consults_all() {
        let s = store();
        let order = vec!["cheap".to_string(), "pricey".to_string()];
        let q = Query::new()
            .with_slot("cheap", vec![1.0, 0.0])
            .with_slot("pricey", vec![1.0, 0.0]);
        let cal = Calibrator::identity(10);
        let out = route(
            &s,
            &order,
            &q,
            &cal,
            &RoutingConfig::default(),
            &GuardProfile::default(),
            RoutingMode::BruteForce,
        );
        assert_eq!(out.witness.lenses_consulted.len(), 2);
    }
}
