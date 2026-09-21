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
use crate::heads::{assemble_noul, geometry, similarity_to_unit, ClassProtos, Classified};
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
    let g = geometry(state_emb, cp, opts.not_for_lambda);
    let head_logits = match probe {
        Some(p) => p.logits(state_emb),
        None => g.proto_scores.clone(),
    };
    Classified {
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
    }
    .into_answer()
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
