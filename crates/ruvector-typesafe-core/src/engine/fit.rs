//! Pure fitters and scorers, free of any cache or engine state (ADR-004). Both
//! `decide` (through the artifact cache) and `optimize` (per proposal, on
//! pre-embedded bank examples) call these, so a campaign never touches the
//! decision-path caches and a decision never re-implements the campaign's fit.
//!
//! An [`Artifact`] is a trained-and-calibrated head for one question under one
//! [`EngineOptions`]. `fit_class_artifact` / `fit_noul_artifact` build it from a
//! (train, calibration) example split; `class_answer` / `noul_answer` turn a
//! state embedding into a Jev-shaped [`Answer`] under the same options.

use crate::calibration::{apply_temperature, fit_temperature, Platt, T_MAX, T_MIN};
use crate::engine::options::{EngineOptions, HeadChoice};
use crate::heads::logistic::{BinaryLogistic, LogisticConfig};
use crate::heads::probe::{MultiProbe, ProbeConfig};
use crate::heads::{assemble_noul, geometry, similarity_to_unit, softmax, ClassProtos, Classified};
use crate::{Answer, Head};

/// Fewest examples of a class (in the training slice) before the linear probe
/// takes over from the nearest-prototype head (ADR-003).
pub(crate) const MIN_EXAMPLES_PER_CLASS: usize = 4;

/// The two distributions used at decision time form a K+1 distribution:
/// `P(class) = P(class | in-scope) * (1 - P(abstain))`. Calibrating only the
/// conditional head logits ignores the abstain mass and systematically
/// miscalibrates the reported confidence.
struct ClassCalibrationRow {
    head_logits: Vec<f32>,
    proto_logits: Vec<f32>,
    abstain_logit: f32,
    label: usize,
}

fn class_nll(rows: &[ClassCalibrationRow], temperature: f32) -> f32 {
    let mut loss = 0.0;
    for row in rows {
        let shares = softmax(&apply_temperature(&row.head_logits, temperature));
        let mut proto_full = row.proto_logits.clone();
        proto_full.push(row.abstain_logit);
        let proto_masses = softmax(&apply_temperature(&proto_full, temperature));
        let in_scope = 1.0 - proto_masses.last().copied().unwrap_or(0.0);
        let p = shares.get(row.label).copied().unwrap_or(0.0) * in_scope;
        loss -= p.max(1e-9).ln();
    }
    loss / rows.len().max(1) as f32
}

/// Fit the temperature against the same in-scope class probability reported
/// by `Classified`. All rows come from the disjoint calibration slice.
fn fit_class_temperature(rows: &[ClassCalibrationRow]) -> f32 {
    if rows.is_empty() {
        return 1.0;
    }
    // Retain the old conditional optimum as a candidate in addition to the
    // fixed log-spaced grid, so the new objective can never be worse than that
    // temperature on the calibration rows due to grid resolution alone.
    let head_logits: Vec<Vec<f32>> = rows.iter().map(|row| row.head_logits.clone()).collect();
    let labels: Vec<usize> = rows.iter().map(|row| row.label).collect();
    let mut best_t = fit_temperature(&head_logits, &labels);
    let mut best_loss = class_nll(rows, best_t);
    const GRID: usize = 120;
    let ln_min = T_MIN.ln();
    let ln_max = T_MAX.ln();
    for i in 0..GRID {
        let fraction = i as f32 / (GRID - 1) as f32;
        let temperature = (ln_min + fraction * (ln_max - ln_min)).exp().clamp(T_MIN, T_MAX);
        let loss = class_nll(rows, temperature);
        if loss < best_loss {
            best_loss = loss;
            best_t = temperature;
        }
    }
    best_t
}

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
/// scale is applied consistently to both head and prototype logits, including
/// the abstain logit, just as `class_answer` applies it at decision time.
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
        let mut rows = Vec::with_capacity(calib.len());
        for (emb, ci) in calib {
            let g = geometry(emb, cp, opts.not_for_lambda);
            let head_logits = match &probe {
                Some(p) => p.logits(emb),
                None => g.proto_scores.clone(),
            };
            let abstain_logit = g.abstain_logit(opts.abstain_tau, opts.abstain_scale) * scale;
            rows.push(ClassCalibrationRow {
                head_logits: head_logits.into_iter().map(|l| l * scale).collect(),
                proto_logits: g.proto_scores.into_iter().map(|l| l * scale).collect(),
                abstain_logit,
                label: *ci,
            });
        }
        (fit_class_temperature(&rows), true)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heads::ClassKind;

    #[test]
    fn calibration_uses_reported_class_mass_including_abstain() {
        let row = ClassCalibrationRow {
            head_logits: vec![2.0, 0.5],
            proto_logits: vec![0.5, 0.2],
            abstain_logit: 0.1,
            label: 0,
        };
        let keys = vec!["alpha".to_string(), "beta".to_string()];
        let response = Classified {
            keys: &keys,
            kind: ClassKind::Choice,
            head_logits: row.head_logits.clone(),
            proto_scores: &row.proto_logits,
            abstain_logit: row.abstain_logit,
            head: Head::LinearProbe,
            temperature: 1.7,
            logit_scale: 1.0,
            calibrated: true,
            model: "test",
        }
        .into_answer();
        let Answer::Choice { choice, meta, .. } = response else {
            panic!("expected class answer")
        };
        assert_eq!(choice, "alpha");
        let effective_probability = (-class_nll(&[row], 1.7)).exp();
        assert!((effective_probability - meta.confidence).abs() < 1e-6);
    }

    #[test]
    fn fitted_temperature_accounts_for_abstention_on_calibration_rows() {
        // The head is right most of the time, but these in-scope examples look
        // far from the prototypes. The conditional-only optimum overstates
        // abstention; the fitted objective must include that lost class mass.
        let rows: Vec<ClassCalibrationRow> = (0..24)
            .map(|i| ClassCalibrationRow {
                head_logits: vec![3.0, 0.0],
                proto_logits: vec![0.0, -0.2],
                abstain_logit: 2.0,
                label: if i < 20 { 0 } else { 1 },
            })
            .collect();
        let old_temperature = fit_temperature(
            &rows.iter().map(|row| row.head_logits.clone()).collect::<Vec<_>>(),
            &rows.iter().map(|row| row.label).collect::<Vec<_>>(),
        );
        let fitted = fit_class_temperature(&rows);
        assert!(
            class_nll(&rows, fitted) + 0.05 < class_nll(&rows, old_temperature),
            "fitted={fitted}, conditional={old_temperature}"
        );
        assert!((T_MIN..=T_MAX).contains(&fitted));
        assert_eq!(fit_class_temperature(&rows), fitted);
    }
}
