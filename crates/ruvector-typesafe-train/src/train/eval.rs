//! In-training validation (plan Step 3 "in-training val metric"): mean of
//! tickets department accuracy, urgent AUROC, frustration accuracy (heads on
//! tickets validation) and head accuracy on each public validation slice.
//! Selection only — cross-arm claims come from the bench (ADR-008 §1b).

use std::collections::BTreeMap;

use anyhow::Result;
use candle_core::Tensor;
use serde::Serialize;

use super::step::{head_name, StepCtx, H_FRUST, H_URGENT, PAD_ID};
use crate::data::{Row, CLINC150, TICKETS};
use crate::embed::pad_rows;
use crate::model::{Bert, Heads};

#[derive(Debug, Clone, Serialize, Default)]
pub struct EvalMetrics {
    pub selection: f64,
    pub components: BTreeMap<String, f64>,
    /// Informational: tickets department head ECE (10 bins), CLINC OOS AUROC.
    pub info: BTreeMap<String, f64>,
    pub rows: usize,
}

fn softmax_rows(logits: &[Vec<f32>]) -> Vec<Vec<f32>> {
    logits
        .iter()
        .map(|r| {
            let m = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = r.iter().map(|x| (x - m).exp()).collect();
            let s: f32 = e.iter().sum();
            e.iter().map(|x| x / s).collect()
        })
        .collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

/// Mann–Whitney AUROC with average ranks for ties; 0.5 if a class is empty.
pub fn auroc(scores: &[f32], positive: &[bool]) -> f64 {
    let n_pos = positive.iter().filter(|p| **p).count();
    let n_neg = positive.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0f64; scores.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[idx[k]] = r;
        }
        i = j + 1;
    }
    let sum_pos: f64 = ranks
        .iter()
        .zip(positive)
        .filter(|(_, p)| **p)
        .map(|(r, _)| r)
        .sum();
    (sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos * n_neg) as f64
}

/// 10-bin expected calibration error of top-1 confidence.
pub fn ece(conf: &[f32], correct: &[bool]) -> f64 {
    let mut bins = [(0f64, 0f64, 0usize); 10];
    for (c, ok) in conf.iter().zip(correct) {
        let b = ((*c as f64 * 10.0) as usize).min(9);
        bins[b].0 += *c as f64;
        bins[b].1 += f64::from(u8::from(*ok));
        bins[b].2 += 1;
    }
    let n = conf.len().max(1) as f64;
    bins.iter()
        .filter(|b| b.2 > 0)
        .map(|b| (b.2 as f64 / n) * ((b.0 - b.1) / b.2 as f64).abs())
        .sum()
}

/// Embed rows and return head logits for each named head.
fn head_logits(
    bert: &Bert,
    heads: &Heads,
    tokens: &[Vec<u32>],
    names: &[&str],
    batch: usize,
) -> Result<Vec<Vec<Vec<f32>>>> {
    let mut out = vec![Vec::with_capacity(tokens.len()); names.len()];
    for chunk in tokens.chunks(batch.max(1)) {
        let seqs: Vec<&[u32]> = chunk.iter().map(|t| t.as_slice()).collect();
        let b = pad_rows(&seqs, PAD_ID, &bert.device)?;
        let e = bert.embed(&b.ids, &b.types, &b.mask)?.detach();
        for (k, n) in names.iter().enumerate() {
            let l: Tensor = heads.logits_frozen(n, &e)?;
            out[k].extend(l.to_vec2::<f32>()?);
        }
    }
    Ok(out)
}

pub fn evaluate(
    bert: &Bert,
    heads: &Heads,
    ctx: &StepCtx,
    datasets: &[&str],
    val: &[(Row, Vec<u32>)],
    batch: usize,
) -> Result<EvalMetrics> {
    let mut m = EvalMetrics::default();
    for &ds in datasets {
        let rows: Vec<&(Row, Vec<u32>)> = val.iter().filter(|(r, _)| r.dataset == ds).collect();
        if rows.is_empty() {
            continue;
        }
        m.rows += rows.len();
        let toks: Vec<Vec<u32>> = rows.iter().map(|(_, t)| t.clone()).collect();
        let lidx = &ctx.label_index[ds];
        if ds == TICKETS {
            let l = head_logits(
                bert,
                heads,
                &toks,
                &[head_name(ds), H_URGENT, H_FRUST],
                batch,
            )?;
            let pd = softmax_rows(&l[0]);
            let pu = softmax_rows(&l[1]);
            let pf = softmax_rows(&l[2]);
            let mut ok = Vec::new();
            let mut conf = Vec::new();
            let (mut u_scores, mut u_pos, mut f_ok) = (Vec::new(), Vec::new(), 0usize);
            for (i, (r, _)) in rows.iter().enumerate() {
                let pred = argmax(&pd[i]);
                ok.push(Some(&pred) == lidx.get(&r.label));
                conf.push(pd[i][pred]);
                u_scores.push(pu[i][1]);
                u_pos.push(r.urgent == Some(true));
                f_ok += usize::from(argmax(&pf[i]) == r.frustration.unwrap_or(0) as usize);
            }
            let n = rows.len() as f64;
            m.components.insert(
                "tickets.department_acc".into(),
                ok.iter().filter(|o| **o).count() as f64 / n,
            );
            m.components
                .insert("tickets.urgent_auroc".into(), auroc(&u_scores, &u_pos));
            m.components
                .insert("tickets.frustration_acc".into(), f_ok as f64 / n);
            m.info
                .insert("tickets.department_head_ece".into(), ece(&conf, &ok));
        } else {
            let l = head_logits(bert, heads, &toks, &[ds], batch)?;
            let p = softmax_rows(&l[0]);
            let (mut hits, mut n_in) = (0usize, 0usize);
            let (mut oos_score, mut is_oos) = (Vec::new(), Vec::new());
            for (i, (r, _)) in rows.iter().enumerate() {
                let pred = argmax(&p[i]);
                if ds == CLINC150 {
                    oos_score.push(1.0 - p[i][pred]);
                    is_oos.push(r.oos);
                }
                if r.oos {
                    continue;
                }
                n_in += 1;
                hits += usize::from(Some(&pred) == lidx.get(&r.label));
            }
            m.components
                .insert(format!("{ds}.acc"), hits as f64 / n_in.max(1) as f64);
            if ds == CLINC150 {
                m.info.insert(
                    "clinc150.oos_auroc_maxprob".into(),
                    auroc(&oos_score, &is_oos),
                );
            }
        }
    }
    m.selection = m.components.values().sum::<f64>() / m.components.len().max(1) as f64;
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auroc_known_values() {
        assert_eq!(
            auroc(&[0.1, 0.2, 0.8, 0.9], &[false, false, true, true]),
            1.0
        );
        assert_eq!(
            auroc(&[0.9, 0.8, 0.2, 0.1], &[false, false, true, true]),
            0.0
        );
        assert_eq!(auroc(&[0.5, 0.5], &[false, true]), 0.5);
    }

    #[test]
    fn ece_perfectly_calibrated_is_zero() {
        assert!(ece(&[1.0, 1.0], &[true, true]) < 1e-12);
        assert!((ece(&[0.95, 0.95], &[false, false]) - 0.95).abs() < 1e-6);
    }
}
