//! `openjev train`: prep-check (pins, Assertion A, token lengths) → P×K
//! multi-task fine-tune → best-validation checkpoint + run records.

pub mod eval;
pub mod step;

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use serde_json::json;

use crate::config::Config;
use crate::data::{read_hashes, write_hashes, DataDir, Row, TICKETS};
use crate::embed::{load_tokenizer, pretokenize};
use crate::leakage::{assert_no_leakage, colliding_descriptions, LeakageReport};
use crate::model::{frozen_view, load_encoder, BertConfig, Heads};
use crate::norm::{sha256_hex, sha256_norm};
use crate::optim::{lr_scale, AdamW};
use crate::pins;
use crate::prep::{heldout_hashes, Sources};
use crate::sampler::Sampler;
use step::{head_name, step_loss, StepCtx, H_FRUST, H_URGENT};

struct StepSummary {
    parts: BTreeMap<&'static str, f32>,
    texts: usize,
}

pub struct TrainArgs {
    pub config: Config,
    pub config_sha256: String,
    pub data: PathBuf,
    pub sources: Sources,
    /// Base safetensors, verified against `pins::BASE_SAFETENSORS`.
    pub base: PathBuf,
    pub tokenizer: PathBuf,
    pub seed: u64,
    pub device: Device,
    pub device_name: String,
    pub out: PathBuf,
}

pub struct PrepCheck {
    pub data: DataDir,
    pub leakage: LeakageReport,
    pub token_p50: usize,
    pub token_p99: usize,
}

fn pct(v: &mut [usize], p: f64) -> usize {
    v.sort_unstable();
    v.get(((v.len() as f64 - 1.0) * p).round() as usize)
        .copied()
        .unwrap_or(0)
}

/// Plan `prep-check`: verifies input pins, runs Assertion A against held-out
/// hashes recomputed from the sources (∪ the data dir's heldout-hashes.txt),
/// and checks token lengths against `max_seq_len`. Cannot be skipped by `train`.
pub fn prep_check(a: &TrainArgs) -> Result<PrepCheck> {
    pins::read_verified(&a.base, pins::BASE_SAFETENSORS.sha256)?;
    pins::read_verified(&a.tokenizer, pins::TOKENIZER.sha256)?;
    let data = DataDir::load(&a.data)?;
    let (mut heldout, counts) = heldout_hashes(&a.sources)?;
    let exported = a.data.join("heldout-hashes.txt");
    if exported.exists() {
        heldout.extend(read_hashes(&exported)?);
    }
    // Carry prep's dropped-duplicate counts into this run's report (model card).
    let mut dropped: BTreeMap<String, usize> = BTreeMap::new();
    let prep_report = a.data.join("leakage-report.json");
    if prep_report.exists() {
        let v: serde_json::Value = serde_json::from_slice(&fs::read(&prep_report)?)?;
        if let Some(per) = v["per_dataset"].as_object() {
            for (ds, d) in per {
                dropped.insert(
                    ds.clone(),
                    d["dropped_duplicates"].as_u64().unwrap_or(0) as usize,
                );
            }
        }
    }
    let mut leakage = assert_no_leakage(&data.train, &data.val, &heldout, &counts, &dropped)
        .map_err(anyhow::Error::new)?;
    leakage.dropped_descriptions = colliding_descriptions(&data.labels, &heldout);
    let tok = load_tokenizer(&a.tokenizer, 512)?;
    let active: HashSet<&str> = a.config.active().into_iter().collect();
    let texts: Vec<&str> = data
        .train
        .iter()
        .filter(|r| active.contains(r.dataset.as_str()))
        .map(|r| r.text.as_str())
        .collect();
    let mut lens: Vec<usize> = pretokenize(&tok, &texts)?.iter().map(|t| t.len()).collect();
    let (p50, p99) = (pct(&mut lens, 0.5), pct(&mut lens, 0.99));
    if p99 + 4 > a.config.max_seq_len {
        bail!(
            "token p99 {p99} too close to max_seq_len {} (plan: fall back to 128)",
            a.config.max_seq_len
        );
    }
    Ok(PrepCheck {
        data,
        leakage,
        token_p50: p50,
        token_p99: p99,
    })
}

fn git_sha() -> String {
    std::env::var("OPENJEV_GIT_SHA").ok().unwrap_or_else(|| {
        std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".into())
    })
}

pub fn run(a: &TrainArgs) -> Result<serde_json::Value> {
    let t_start = Instant::now();
    let cfg = &a.config;
    let pc = prep_check(a)?; // Assertion A happens here, before any step.
    fs::create_dir_all(&a.out)?;
    fs::write(
        a.out.join("leakage-report.json"),
        serde_json::to_vec_pretty(&pc.leakage)?,
    )?;
    let active = cfg.active();
    let train: Vec<Row> = pc
        .data
        .train
        .iter()
        .filter(|r| active.contains(&r.dataset.as_str()))
        .cloned()
        .collect();
    let val_all: Vec<Row> = pc
        .data
        .val
        .iter()
        .filter(|r| active.contains(&r.dataset.as_str()))
        .cloned()
        .collect();

    // Label descriptions that equal a held-out text are never trained on.
    let banned = |ds: &str, label: &str| {
        pc.leakage
            .dropped_descriptions
            .get(ds)
            .is_some_and(|v| v.iter().any(|l| l == label))
    };
    // Assertion B input: every text that produced a gradient (rows + the label
    // descriptions actually used as SupCon positives).
    let mut grad_texts: Vec<String> = train.iter().map(|r| sha256_norm(&r.text)).collect();
    for ds in &active {
        for (label, d) in &pc.data.labels[*ds] {
            if !banned(ds, label) {
                grad_texts.push(sha256_norm(d));
            }
        }
    }
    let n_hashes = write_hashes(&a.out.join("train-text-hashes.txt"), grad_texts)?;

    let tok = load_tokenizer(&a.tokenizer, cfg.max_seq_len)?;
    let texts: Vec<&str> = train.iter().map(|r| r.text.as_str()).collect();
    let tokens = pretokenize(&tok, &texts)?;
    let mut label_index = BTreeMap::new();
    let mut desc_tokens = BTreeMap::new();
    for ds in &active {
        let labels = &pc.data.labels[*ds];
        label_index.insert(
            ds.to_string(),
            labels
                .keys()
                .enumerate()
                .map(|(i, k)| (k.clone(), i))
                .collect(),
        );
        let keys: Vec<&String> = labels.keys().collect();
        let dt: Vec<&str> = labels.values().map(|s| s.as_str()).collect();
        for (k, t) in keys.into_iter().zip(pretokenize(&tok, &dt)?) {
            if !banned(ds, k) {
                desc_tokens.insert((ds.to_string(), k.clone()), t);
            }
        }
    }
    let (urgent_w, frust_w) = StepCtx::class_weights(&train);
    let ctx = StepCtx {
        rows: train,
        tokens,
        desc_tokens,
        label_index,
        urgent_w,
        frust_w,
    };

    // Validation rows (optionally capped per dataset for smoke runs), pre-tokenized.
    let mut val = Vec::new();
    for ds in &active {
        let rows: Vec<&Row> = val_all.iter().filter(|r| r.dataset == *ds).collect();
        let cap = if cfg.train.val_limit == 0 {
            rows.len()
        } else {
            cfg.train.val_limit.min(rows.len())
        };
        let t: Vec<&str> = rows[..cap].iter().map(|r| r.text.as_str()).collect();
        for (r, tk) in rows[..cap].iter().zip(pretokenize(&tok, &t)?) {
            val.push(((*r).clone(), tk));
        }
    }

    let (enc_vm, bert) = load_encoder(&a.base, BertConfig::bge_small(), &a.device)?;
    let eval_bert = frozen_view(&enc_vm, BertConfig::bge_small(), &a.device)?;
    let mut spec: Vec<(String, usize)> = active
        .iter()
        .map(|d| (head_name(d).to_string(), ctx.n_classes(d)))
        .collect();
    if active.contains(&TICKETS) {
        spec.push((H_URGENT.into(), 2));
        spec.push((H_FRUST.into(), 3));
    }
    let heads = Heads::new(&spec, 384, cfg.train.head_scale, a.seed, &a.device)?;
    let mut opt = AdamW::new(
        cfg.train.beta1,
        cfg.train.beta2,
        cfg.train.adam_eps,
        cfg.train.weight_decay,
    );
    opt.add_group(&enc_vm, cfg.train.encoder_lr)?;
    opt.add_group(&heads.vm, cfg.train.head_lr)?;
    let mut sampler = Sampler::new(&ctx.rows, cfg, a.seed);

    let mut curves = fs::File::create(a.out.join("curves.jsonl"))?;
    let base_eval = eval::evaluate(
        &eval_bert,
        &heads,
        &ctx,
        &active,
        &val,
        cfg.train.eval_batch,
    )?;
    writeln!(
        curves,
        "{}",
        json!({"kind": "eval", "step": 0, "metrics": base_eval})
    )?;
    crate::gpu::trim(&a.device)?;
    eprintln!(
        "[train] step 0 val selection {:.4} {:?}",
        base_eval.selection, base_eval.components
    );
    let (mut best, mut best_step, mut bad) = (f64::NEG_INFINITY, 0usize, 0usize);
    let (mut texts_seen, mut train_secs, mut steps_done) = (0usize, 0f64, 0usize);
    let mut first_loss = None;
    let mut last_loss = 0f32;
    for s in 0..cfg.train.max_steps {
        let t0 = Instant::now();
        let batch = sampler.next_batch();
        let lr = lr_scale(s, cfg.train.warmup_steps, cfg.train.max_steps);
        // Scoped so the step's graph and gradients are freed before any trim.
        let (out, gnorm) = {
            let o = step_loss(&bert, &heads, &ctx, cfg, &batch)?;
            let grads = o.loss.backward()?;
            let g = opt.step(&grads, lr, cfg.train.grad_clip)?;
            (
                StepSummary {
                    parts: o.parts,
                    texts: o.texts,
                },
                g,
            )
        };
        let dt = t0.elapsed().as_secs_f64();
        let total = out.parts["total"];
        if !total.is_finite() {
            bail!("non-finite loss at step {s}: {:?}", out.parts);
        }
        first_loss.get_or_insert(total);
        last_loss = total;
        texts_seen += out.texts;
        train_secs += dt;
        steps_done = s + 1;
        writeln!(
            curves,
            "{}",
            json!({"kind": "step", "step": s + 1, "dataset": batch.dataset,
            "loss": out.parts, "lr_scale": lr, "grad_norm": gnorm, "texts": out.texts, "secs": dt})
        )?;
        if (s + 1) % 25 == 0 {
            let mem = crate::gpu::used_mib(&a.device);
            crate::gpu::trim(&a.device)?;
            let after = crate::gpu::used_mib(&a.device);
            writeln!(
                curves,
                "{}",
                json!({"kind": "mem", "step": s + 1, "used_mib": mem, "after_trim_mib": after})
            )?;
            eprintln!(
                "[train] step {} {} loss {:.4} gnorm {:.3} {:.1} seq/s gpu {:?}->{:?} MiB",
                s + 1,
                batch.dataset,
                total,
                gnorm,
                texts_seen as f64 / train_secs,
                mem,
                after
            );
        }
        if (s + 1) % cfg.train.eval_every == 0 || s + 1 == cfg.train.max_steps {
            let m = eval::evaluate(
                &eval_bert,
                &heads,
                &ctx,
                &active,
                &val,
                cfg.train.eval_batch,
            )?;
            crate::gpu::trim(&a.device)?;
            writeln!(
                curves,
                "{}",
                json!({"kind": "eval", "step": s + 1, "metrics": m})
            )?;
            eprintln!(
                "[train] step {} val selection {:.4} {:?}",
                s + 1,
                m.selection,
                m.components
            );
            if m.selection > best {
                best = m.selection;
                best_step = s + 1;
                bad = 0;
                enc_vm.save(a.out.join("encoder.safetensors"))?;
                heads.vm.save(a.out.join("heads.safetensors"))?;
            } else {
                bad += 1;
                if bad >= cfg.train.patience {
                    eprintln!("[train] early stop at step {} (best {best_step})", s + 1);
                    break;
                }
            }
        }
    }
    fs::write(a.out.join("heads-discarded.txt"),
        "heads.safetensors holds the auxiliary multi-task heads. They only shape the embedding space and are NOT shipped (ADR-008 §3): the engine fits its own probe/logistic/temperature heads at bench time.\n")?;
    let enc_sha =
        sha256_hex(&fs::read(a.out.join("encoder.safetensors")).context("no checkpoint written")?);
    let record = json!({
        "config": cfg, "config_sha256": a.config_sha256, "seed": a.seed, "git_sha": git_sha(),
        "device": a.device_name, "data_manifest_sha256": fs::read(a.data.join("data-manifest.json")).ok().map(|b| sha256_hex(&b)),
        "base_safetensors_sha256": pins::BASE_SAFETENSORS.sha256, "tokenizer_sha256": pins::TOKENIZER.sha256,
        "encoder_sha256": enc_sha, "best_step": best_step, "best_val_selection": best, "baseline_val": base_eval,
        "steps_done": steps_done, "first_loss": first_loss, "last_loss": last_loss,
        "train_seconds": train_secs, "wall_seconds": t_start.elapsed().as_secs_f64(),
        "texts_per_second": texts_seen as f64 / train_secs.max(1e-9),
        "token_p50": pc.token_p50, "token_p99": pc.token_p99, "train_text_hashes": n_hashes,
        "train_rows": ctx.rows.len(), "val_rows": val.len(),
    });
    fs::write(
        a.out.join("training-run.json"),
        serde_json::to_vec_pretty(&record)?,
    )?;
    Ok(record)
}

pub fn config_sha(path: &Path) -> Result<String> {
    Ok(sha256_hex(&fs::read(path)?))
}
