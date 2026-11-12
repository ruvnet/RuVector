//! One optimizer step's loss (plan Step 3 "total loss"):
//! SupCon + head CE (+ urgent + frustration on ticket steps) + oe·OE (CLINC).

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use candle_core::Tensor;

use crate::config::Config;
use crate::data::{Row, CLINC150, TICKETS};
use crate::embed::pad_rows;
use crate::loss::{inverse_frequency, smoothed, soft_ce, supcon};
use crate::model::{Bert, Heads};
use crate::sampler::StepBatch;

pub const PAD_ID: u32 = 0; // [PAD] in the pinned bge tokenizer.json
pub const H_DEPT: &str = "tickets.department";
pub const H_URGENT: &str = "tickets.urgent";
pub const H_FRUST: &str = "tickets.frustration";

pub fn head_name(dataset: &str) -> &str {
    if dataset == TICKETS {
        H_DEPT
    } else {
        dataset
    }
}

/// Everything a step needs that is fixed for the run.
pub struct StepCtx {
    pub rows: Vec<Row>,
    /// Pre-tokenized train rows (no padding), parallel to `rows`.
    pub tokens: Vec<Vec<u32>>,
    /// (dataset, label) → tokenized description text.
    pub desc_tokens: BTreeMap<(String, String), Vec<u32>>,
    /// dataset → label → class index (sorted label order).
    pub label_index: BTreeMap<String, BTreeMap<String, usize>>,
    /// Class weights: urgent [neg, pos], frustration [0, 1, 2].
    pub urgent_w: Vec<f32>,
    pub frust_w: Vec<f32>,
}

impl StepCtx {
    pub fn class_weights(rows: &[Row]) -> (Vec<f32>, Vec<f32>) {
        let t: Vec<&Row> = rows.iter().filter(|r| r.dataset == TICKETS).collect();
        let pos = t.iter().filter(|r| r.urgent == Some(true)).count();
        let neg = t.len() - pos;
        let urgent_w = vec![
            1.0,
            if pos == 0 {
                1.0
            } else {
                neg as f32 / pos as f32
            },
        ];
        let mut fc = [0usize; 3];
        for r in &t {
            if let Some(f) = r.frustration {
                fc[(f as usize).min(2)] += 1;
            }
        }
        (urgent_w, inverse_frequency(&fc))
    }

    pub fn n_classes(&self, dataset: &str) -> usize {
        self.label_index[dataset].len()
    }
}

pub struct StepOut {
    pub loss: Tensor,
    pub parts: BTreeMap<&'static str, f32>,
    pub texts: usize,
}

fn part(parts: &mut BTreeMap<&'static str, f32>, k: &'static str, t: &Tensor) -> Result<()> {
    parts.insert(k, crate::loss::scalar(t)?);
    Ok(())
}

pub fn step_loss(
    bert: &Bert,
    heads: &Heads,
    ctx: &StepCtx,
    cfg: &Config,
    b: &StepBatch,
) -> Result<StepOut> {
    let ds = b.dataset;
    let lidx = &ctx.label_index[ds];
    let c = lidx.len();
    let mut seqs: Vec<&[u32]> = b
        .rows
        .iter()
        .map(|(i, _)| ctx.tokens[*i].as_slice())
        .collect();
    // Descriptions excluded by the leakage check (equal to a held-out text) are
    // absent from `desc_tokens`; those labels simply get no extra positive.
    let descs: Vec<&String> = b
        .desc_labels
        .iter()
        .filter(|l| {
            ctx.desc_tokens
                .contains_key(&(ds.to_string(), (*l).clone()))
        })
        .collect();
    for l in &descs {
        seqs.push(ctx.desc_tokens[&(ds.to_string(), (*l).clone())].as_slice());
    }
    seqs.extend(b.oos_rows.iter().map(|i| ctx.tokens[*i].as_slice()));
    let (n_rows, n_desc, n_oos) = (b.rows.len(), descs.len(), b.oos_rows.len());
    let batch = pad_rows(&seqs, PAD_ID, &bert.device)?;
    let emb = bert.embed(&batch.ids, &batch.types, &batch.mask)?;
    let mut parts = BTreeMap::new();

    // SupCon over examples + label descriptions (descriptions are positives).
    let labels: Vec<u32> = b
        .rows
        .iter()
        .map(|(_, l)| l)
        .chain(descs.iter().copied())
        .map(|l| lidx[l] as u32)
        .collect();
    let mut total = match supcon(
        &emb.narrow(0, 0, n_rows + n_desc)?,
        &labels,
        cfg.train.supcon_temperature,
    )? {
        Some(s) => {
            part(&mut parts, "supcon", &s)?;
            s
        }
        None => Tensor::zeros((), candle_core::DType::F32, &bert.device)?,
    };

    // Main head: smoothed one-hot, or the soft ambiguous target (no extra smoothing).
    let er = emb.narrow(0, 0, n_rows)?;
    let eps = cfg.train.label_smoothing;
    let mut targets = Vec::with_capacity(n_rows * c);
    for (i, l) in &b.rows {
        match &ctx.rows[*i].soft {
            Some(soft) if ds == TICKETS && soft.keys().any(|k| k != l) => {
                // primary gets `ambiguous_primary`; the rest share 1 − that by weight.
                let p = cfg.train.ambiguous_primary;
                let others: f32 = soft.iter().filter(|(k, _)| *k != l).map(|(_, w)| *w).sum();
                let mut t = vec![0f32; c];
                t[lidx[l]] = p;
                for (k, w) in soft.iter().filter(|(k, _)| *k != l) {
                    let slot = lidx.get(k).with_context(|| format!("soft label {k}"))?;
                    t[*slot] += (1.0 - p) * w / others.max(1e-6);
                }
                targets.extend(t);
            }
            _ => targets.extend(smoothed(lidx[l], c, eps)),
        }
    }
    let ce = soft_ce(
        &heads.logits(head_name(ds), &er)?,
        targets,
        vec![1.0; n_rows],
    )?;
    part(&mut parts, "head", &ce)?;
    total = (total + ce)?;

    if ds == TICKETS {
        let (mut tu, mut wu, mut tf, mut wf) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, _) in &b.rows {
            let r = &ctx.rows[*i];
            let u = usize::from(r.urgent.unwrap_or(false));
            tu.extend(smoothed(u, 2, eps));
            wu.push(ctx.urgent_w[u]);
            let f = (r.frustration.unwrap_or(0) as usize).min(2);
            tf.extend(smoothed(f, 3, eps));
            wf.push(ctx.frust_w[f]);
        }
        let lu = soft_ce(&heads.logits(H_URGENT, &er)?, tu, wu)?;
        let lf = soft_ce(&heads.logits(H_FRUST, &er)?, tf, wf)?;
        part(&mut parts, "urgent", &lu)?;
        part(&mut parts, "frustration", &lf)?;
        total = ((total + lu)? + lf)?;
    }

    if ds == CLINC150 && n_oos > 0 {
        let eo = emb.narrow(0, n_rows + n_desc, n_oos)?;
        let uni = vec![1.0 / c as f32; n_oos * c];
        let oe = soft_ce(&heads.logits(CLINC150, &eo)?, uni, vec![1.0; n_oos])?;
        part(&mut parts, "oe", &oe)?;
        total = (total + (oe * cfg.train.oe_weight)?)?;
    }
    part(&mut parts, "total", &total)?;
    Ok(StepOut {
        loss: total,
        parts,
        texts: seqs.len(),
    })
}
