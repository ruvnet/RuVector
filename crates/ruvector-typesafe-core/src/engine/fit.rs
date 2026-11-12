//! Pure fitters and scorers, free of any cache or engine state (ADR-004). Both
//! `decide` (through the artifact cache) and `optimize` (per proposal, on
//! pre-embedded bank examples) call these, so a campaign never touches the
//! decision-path caches and a decision never re-implements the campaign's fit.
//!
//! An [`Artifact`] is a trained-and-calibrated head for one question under one
//! [`EngineOptions`]. `fit_class_artifact` / `fit_noul_artifact` build it from a
//! (train, calibration) example split; `class_answer` / `noul_answer` turn a
//! state embedding into a Jev-shaped [`Answer`] under the same options.

use crate::calibration::{fit_temperature, Platt};
use crate::engine::options::{EngineOptions, HeadChoice};
use crate::heads::logistic::{BinaryLogistic, LogisticConfig};
use crate::heads::probe::{MultiProbe, ProbeConfig};
use crate::heads::{
    apply_catch_all, assemble_noul, geometry, oos_logit_excluding, similarity_to_unit, ClassProtos,
    Classified,
};
use crate::{Answer, Head};

/// Fewest examples of a class (in the training slice) before the linear probe
/// takes over from the nearest-prototype head (ADR-003).
pub(crate) const MIN_EXAMPLES_PER_CLASS: usize = 4;

/// A per-question trained artifact. `Class` covers `choice`/`score`; `Noul`
/// covers the binary predicate. Built together with the matching [`Compiled`].
pub(crate) enum Artifact {
    Class {
        probe: Option<MultiProbe>,
        temperature: f32,
        calibrated: bool,
        head: Head,
    },
    Noul {
        model: Option<BinaryLogistic>,
        platt: Option<Platt>,
        calibrated: bool,
        head: Head,
    },
}

impl Artifact {
    /// The head this artifact will report.
    pub(crate) fn head(&self) -> Head {
        match self {
            Artifact::Class { head, .. } | Artifact::Noul { head, .. } => *head,
        }
    }

    /// Fitted temperature (class head) or `1.0` (noul, whose calibration is
    /// Platt, not temperature) — recorded in a campaign receipt.
    pub(crate) fn temperature(&self) -> f32 {
        match self {
            Artifact::Class { temperature, .. } => *temperature,
            Artifact::Noul { .. } => 1.0,
        }
    }
}

fn probe_config(opts: &EngineOptions) -> ProbeConfig {
    ProbeConfig {
        lr: opts.probe_learning_rate,
        l2: opts.probe_l2,
        iters: opts.probe_iterations,
        class_balanced: opts.probe_class_balanced,
    }
}

/// Decide whether to fit a probe given the option's head choice and the
/// per-class training counts.
fn use_probe(opts: &EngineOptions, k: usize, counts: &[usize]) -> bool {
    match opts.head {
        HeadChoice::Prototype => false,
        HeadChoice::Probe => k >= 2 && counts.iter().all(|&c| c >= 1),
        HeadChoice::Auto => k >= 2 && counts.iter().all(|&c| c >= MIN_EXAMPLES_PER_CLASS),
    }
}

/// Fit a class head + temperature from a (train, calibration) split. The logit
/// scale is applied consistently: temperature is fitted over `scale·logits`, so
/// `class_answer` (which also scales before temperature) is calibrated on the
/// same quantity.
pub(crate) fn fit_class_artifact(
    opts: &EngineOptions,
    cp: &ClassProtos,
    train_ex: &[(Vec<f32>, usize)],
    calib: &[(Vec<f32>, usize)],
    dims: usize,
    allow_calibration: bool,
) -> Artifact {
    let mut counts = vec![0usize; cp.keys.len()];
    for (_, ci) in train_ex {
        if *ci < counts.len() {
            counts[*ci] += 1;
        }
    }
    let probe = if use_probe(opts, cp.keys.len(), &counts) {
        Some(MultiProbe::train(
            train_ex,
            cp.keys.len(),
            dims,
            &probe_config(opts),
        ))
    } else {
        None
    };
    let head = if probe.is_some() {
        Head::LinearProbe
    } else {
        Head::NearestPrototype
    };

    let scale = opts.logit_scale;
    let (temperature, calibrated) = if calib.len() >= opts.min_calibration && allow_calibration {
        let mut logits = Vec::with_capacity(calib.len());
        let mut labels = Vec::with_capacity(calib.len());
        for (emb, ci) in calib {
            let row = match &probe {
                Some(p) => p.logits(emb),
                None => geometry(emb, cp, opts.not_for_lambda).proto_scores,
            };
            logits.push(row.iter().map(|l| l * scale).collect::<Vec<f32>>());
            labels.push(*ci);
        }
        (fit_temperature(&logits, &labels), true)
    } else if opts.crossfit_calibration && allow_calibration {
        match crossfit_class_logits(opts, cp, train_ex, calib, dims, probe.is_some()) {
            Some((logits, labels)) => (fit_temperature(&logits, &labels), true),
            None => (1.0, false),
        }
    } else {
        (1.0, false)
    };

    Artifact::Class {
        probe,
        temperature,
        calibrated,
        head,
    }
}

/// Fit the binary `noul` head + Platt layer from a (train, calibration) split.
pub(crate) fn fit_noul_artifact(
    opts: &EngineOptions,
    train_ex: &[(Vec<f32>, f32)],
    calib: &[(Vec<f32>, f32)],
    dims: usize,
    allow_calibration: bool,
) -> Artifact {
    let has_pos = train_ex.iter().any(|(_, y)| *y >= 0.5);
    let has_neg = train_ex.iter().any(|(_, y)| *y < 0.5);
    if train_ex.is_empty() || !has_pos || !has_neg {
        return Artifact::Noul {
            model: None,
            platt: None,
            calibrated: false,
            head: Head::SimilarityUncalibrated,
        };
    }
    let cfg = LogisticConfig {
        lr: opts.probe_learning_rate,
        l2: opts.probe_l2,
        iters: opts.probe_iterations,
    };
    let model = BinaryLogistic::train(train_ex, dims, &cfg);
    let (platt, calibrated) = if calib.len() >= opts.min_calibration && allow_calibration {
        let scores: Vec<f32> = calib.iter().map(|(x, _)| model.raw(x)).collect();
        let labels: Vec<f32> = calib.iter().map(|(_, y)| *y).collect();
        (Some(Platt::fit(&scores, &labels)), true)
    } else if opts.crossfit_calibration && allow_calibration {
        match crossfit_noul_scores(train_ex, calib, dims, &cfg, opts.min_calibration) {
            Some((scores, labels)) => (Some(Platt::fit(&scores, &labels)), true),
            None => (None, false),
        }
    } else {
        (None, false)
    };
    Artifact::Noul {
        model: Some(model),
        platt,
        calibrated,
        head: Head::Logistic,
    }
}

/// Folds for cross-fitted calibration (`EngineOptions::crossfit_calibration`).
pub(crate) const CROSSFIT_FOLDS: usize = 5;

/// Out-of-fold, scale-multiplied class logits over `train ∪ calib`, for fitting
/// a temperature when the calibration slice alone is below `min_calibration`.
/// Each example is scored by a head trained without it (positional folds, so
/// the result is deterministic). The final artifact's head is still the one
/// trained on `train`; these logits only set its temperature. Returns `None`
/// when the pool is below `min_calibration` or a fold cannot train the head
/// the final artifact uses.
fn crossfit_class_logits(
    opts: &EngineOptions,
    cp: &ClassProtos,
    train_ex: &[(Vec<f32>, usize)],
    calib: &[(Vec<f32>, usize)],
    dims: usize,
    with_probe: bool,
) -> Option<(Vec<Vec<f32>>, Vec<usize>)> {
    let pool: Vec<&(Vec<f32>, usize)> = train_ex.iter().chain(calib).collect();
    if pool.len() < opts.min_calibration.max(CROSSFIT_FOLDS) {
        return None;
    }
    let k = cp.keys.len();
    let scale = opts.logit_scale;
    let mut logits = Vec::with_capacity(pool.len());
    let mut labels = Vec::with_capacity(pool.len());
    for fold in 0..CROSSFIT_FOLDS {
        let probe = if with_probe {
            let fit: Vec<(Vec<f32>, usize)> = pool
                .iter()
                .enumerate()
                .filter(|(i, _)| i % CROSSFIT_FOLDS != fold)
                .map(|(_, e)| (*e).clone())
                .collect();
            let mut counts = vec![0usize; k];
            for (_, ci) in &fit {
                if *ci < k {
                    counts[*ci] += 1;
                }
            }
            if k < 2 || counts.contains(&0) {
                return None;
            }
            Some(MultiProbe::train(&fit, k, dims, &probe_config(opts)))
        } else {
            None
        };
        for (_, (emb, ci)) in pool
            .iter()
            .enumerate()
            .filter(|(i, _)| i % CROSSFIT_FOLDS == fold)
        {
            let row = match &probe {
                Some(p) => p.logits(emb),
                None => geometry(emb, cp, opts.not_for_lambda).proto_scores,
            };
            logits.push(row.iter().map(|l| l * scale).collect());
            labels.push(*ci);
        }
    }
    Some((logits, labels))
}

/// Out-of-fold raw logistic scores over `train ∪ calib` for a Platt layer, as
/// [`crossfit_class_logits`] does for the class head. `None` when the pool is
/// too small or a fold lacks a positive or a negative example.
fn crossfit_noul_scores(
    train_ex: &[(Vec<f32>, f32)],
    calib: &[(Vec<f32>, f32)],
    dims: usize,
    cfg: &LogisticConfig,
    min_calibration: usize,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let pool: Vec<&(Vec<f32>, f32)> = train_ex.iter().chain(calib).collect();
    if pool.len() < min_calibration.max(CROSSFIT_FOLDS) {
        return None;
    }
    let mut scores = Vec::with_capacity(pool.len());
    let mut labels = Vec::with_capacity(pool.len());
    for fold in 0..CROSSFIT_FOLDS {
        let fit: Vec<(Vec<f32>, f32)> = pool
            .iter()
            .enumerate()
            .filter(|(i, _)| i % CROSSFIT_FOLDS != fold)
            .map(|(_, e)| (*e).clone())
            .collect();
        if !fit.iter().any(|(_, y)| *y >= 0.5) || !fit.iter().any(|(_, y)| *y < 0.5) {
            return None;
        }
        let model = BinaryLogistic::train(&fit, dims, cfg);
        for (_, (x, y)) in pool
            .iter()
            .enumerate()
            .filter(|(i, _)| i % CROSSFIT_FOLDS == fold)
        {
            scores.push(model.raw(x));
            labels.push(*y);
        }
    }
    Some((scores, labels))
}

/// Score a `choice`/`score` state embedding into a Jev-shaped answer under
/// `opts`. Panics only on an `Artifact::Noul` paired with class prototypes,
/// which the engine never constructs.
pub(crate) fn class_answer(
    opts: &EngineOptions,
    cp: &ClassProtos,
    art: &Artifact,
    state_emb: &[f32],
    model: &str,
) -> Answer {
    let Artifact::Class {
        probe,
        temperature,
        calibrated,
        head,
    } = art
    else {
        unreachable!("class_answer called with a noul artifact");
    };
    let mut g = geometry(state_emb, cp, opts.not_for_lambda);
    let mut head_logits = match probe {
        Some(p) => p.logits(state_emb),
        None => g.proto_scores.clone(),
    };
    // Opt-in catch-all option (EngineOptions::catch_all): its text is never
    // scored as a prototype or probe class; its probability comes from the
    // out-of-scope logit over the real options.
    let catch = catch_all_index(opts, cp);
    if let Some(k) = catch {
        head_logits[k] = f32::NEG_INFINITY;
        g.proto_scores[k] = f32::NEG_INFINITY;
    }
    let answer = Classified {
        keys: &cp.keys,
        kind: cp.kind,
        head_logits,
        proto_scores: &g.proto_scores,
        abstain_logit: g.abstain_logit(opts.abstain_tau, opts.abstain_scale),
        head: *head,
        temperature: *temperature,
        logit_scale: opts.logit_scale,
        calibrated: *calibrated,
        model,
        abstain_mode: opts.abstain_mode,
    }
    .into_answer();
    match catch {
        Some(k) => {
            let z = oos_logit_excluding(state_emb, cp, k, opts.abstain_tau, opts.abstain_scale);
            let p_oos = 1.0 / (1.0 + (-z).exp());
            apply_catch_all(answer, k, &cp.keys, p_oos, opts.catch_all_threshold)
        }
        None => answer,
    }
}

/// Index of the configured catch-all option, when this is a `choice` question
/// that has it and at least two real options besides it.
fn catch_all_index(opts: &EngineOptions, cp: &ClassProtos) -> Option<usize> {
    let key = opts.catch_all.as_deref()?;
    if !matches!(cp.kind, crate::heads::ClassKind::Choice) || cp.keys.len() < 3 {
        return None;
    }
    cp.keys.iter().position(|k| k == key)
}

/// Score a `noul` state embedding into an answer under `opts`.
pub(crate) fn noul_answer(
    art: &Artifact,
    predicate: &[f32],
    state_emb: &[f32],
    model: &str,
) -> Answer {
    let Artifact::Noul {
        model: probe,
        platt,
        calibrated,
        head,
    } = art
    else {
        unreachable!("noul_answer called with a class artifact");
    };
    let noul = match probe {
        Some(m) => match platt {
            Some(p) => p.apply(m.raw(state_emb)),
            None => m.prob(state_emb),
        },
        None => similarity_to_unit(crate::embedder::dot(state_emb, predicate)),
    };
    assemble_noul(noul, *head, *calibrated, model)
}
