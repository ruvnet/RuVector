use super::{Candidate, MetricsReport};
use crate::{
    artifact::{ModelBundle, Origin},
    content_hash,
};
use ruvector_language::error::{Error, Result};

/// Verifies integrity and an operator-selected MRR floor, then creates a NEW bundle.
/// This neither changes a running model nor establishes artifact authenticity.
pub fn promote(candidate: &Candidate, minimum_mrr: f32) -> Result<ModelBundle> {
    candidate.model.validate()?;
    let r = &candidate.report;
    if candidate.version != 1
        || r.version != 1
        || r.model_hash != content_hash(&candidate.model)?
        || !minimum_mrr.is_finite()
        || !(0.0..=1.0).contains(&minimum_mrr)
        || r.evaluation_count == 0
        || r.evaluation_count > 2000
        || r.train_count == 0
        || r.train_count > 10_000
        || r.epochs == 0
        || r.epochs > 1000
        || !r.learning_rate.is_finite()
        || r.learning_rate <= 0.0
        || r.learning_rate > 1.0
        || r.estimated_work > 1_000_000_000
        || r.losses.len() != r.epochs
        || r.losses.iter().any(|v| !v.is_finite())
    {
        return Err(Error::new(
            "PROMOTION",
            "invalid candidate or promotion threshold",
        ));
    }
    match &candidate.model.origin {
        Origin::Trained {
            request_hash,
            training_hash,
            evaluation_hash,
            epochs,
        } if request_hash == &r.request_hash
            && training_hash == &r.training_hash
            && evaluation_hash == &r.evaluation_hash
            && epochs == &r.epochs => {}
        _ => {
            return Err(Error::new(
                "PROMOTION",
                "candidate training provenance mismatch",
            ))
        }
    }
    validate_metrics(
        &r.initial,
        r.evaluation_count,
        candidate.model.entities.len(),
    )?;
    validate_metrics(
        &r.trained,
        r.evaluation_count,
        candidate.model.entities.len(),
    )?;
    if !r.trained.combined.mrr.is_finite()
        || r.trained.combined.count != 2 * r.evaluation_count
        || r.trained.combined.mrr < minimum_mrr
        || r.trained.combined.mrr > 1.0
    {
        return Err(Error::new(
            "PROMOTION",
            "held out MRR failed the explicit promotion floor",
        ));
    }
    Ok(ModelBundle {
        version: 1,
        models: vec![candidate.model.clone()],
    })
}

fn validate_metrics(report: &MetricsReport, count: usize, entities: usize) -> Result<()> {
    for (metrics, expected) in [
        (&report.head, count),
        (&report.tail, count),
        (&report.combined, 2 * count),
    ] {
        let probabilities = [metrics.mrr, metrics.hits1, metrics.hits3, metrics.hits10];
        if metrics.count != expected
            || !metrics.mr.is_finite()
            || metrics.mr < 1.0
            || metrics.mr > entities as f32
            || probabilities
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || metrics.hits1 > metrics.hits3
            || metrics.hits3 > metrics.hits10
        {
            return Err(Error::new("PROMOTION", "invalid evaluation metrics"));
        }
    }
    let head = &report.head;
    let tail = &report.tail;
    let combined = &report.combined;
    for (h, t, c) in [
        (head.mr, tail.mr, combined.mr),
        (head.mrr, tail.mrr, combined.mrr),
        (head.hits1, tail.hits1, combined.hits1),
        (head.hits3, tail.hits3, combined.hits3),
        (head.hits10, tail.hits10, combined.hits10),
    ] {
        if ((h + t) * 0.5 - c).abs() > 1e-5 * c.abs().max(1.0) {
            return Err(Error::new(
                "PROMOTION",
                "inconsistent combined evaluation metrics",
            ));
        }
    }
    Ok(())
}
