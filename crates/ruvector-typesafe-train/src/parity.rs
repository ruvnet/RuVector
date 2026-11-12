//! `openjev parity` (ADR-008 §5 step 4): the transplanted ONNX run through the
//! engine's own `OrtEmbedder` must agree with the candle forward pass on the
//! fine-tuned safetensors — cosine ≥ threshold per probe text — and an engine
//! built on each must make identical `department` decisions.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use candle_core::Device;
use ruvector_embed_core::{ModelManifest, OrtEmbedder, Pooling};
use ruvector_typesafe_core::engine::{Engine, LabeledExample};
use ruvector_typesafe_core::{Answer, DecisionRequest, Embedder};
use serde::Serialize;
use serde_json::json;

use crate::data::{DataDir, TICKETS};
use crate::embed::{load_tokenizer, CandleEmbedder};
use crate::model::{frozen_view, load_encoder, BertConfig};
use crate::norm::sha256_hex;

pub struct ParityArgs {
    pub onnx: PathBuf,
    pub weights: PathBuf,
    pub tokenizer: PathBuf,
    pub data: Option<PathBuf>,
    pub probes: usize,
    pub threshold: f32,
    pub max_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ParityReport {
    pub onnx_sha256: String,
    pub probes: usize,
    pub threshold: f32,
    pub cosine_min: f64,
    pub cosine_mean: f64,
    pub cosine_p01: f64,
    pub below_threshold: usize,
    pub decisions_compared: usize,
    pub decisions_identical: usize,
    pub pass: bool,
    pub seconds: f64,
}

/// Deterministic probe set: every dataset's validation rows, interleaved, then
/// tickets train rows; falls back to a fixed list when no data dir is given.
fn probe_texts(data: Option<&DataDir>, n: usize) -> Vec<String> {
    let Some(d) = data else {
        return FALLBACK
            .iter()
            .map(|s| s.to_string())
            .cycle()
            .take(n.min(FALLBACK.len()))
            .collect();
    };
    let mut per: Vec<Vec<&str>> = crate::data::DATASETS
        .iter()
        .map(|ds| {
            d.val
                .iter()
                .filter(|r| r.dataset == *ds)
                .map(|r| r.text.as_str())
                .collect()
        })
        .collect();
    let mut out = Vec::with_capacity(n);
    let mut i = 0;
    while out.len() < n && per.iter().any(|v| i < v.len()) {
        for v in per.iter_mut() {
            if let Some(t) = v.get(i) {
                if out.len() < n {
                    out.push(t.to_string());
                }
            }
        }
        i += 1;
    }
    out
}

const FALLBACK: &[&str] = &[
    "I was charged twice for my subscription this month.",
    "The app crashes every time I open settings.",
    "Where is my parcel? It was due on Monday.",
    "I want to return the shoes, they don't fit.",
    "I can't log in after resetting my password.",
    "How much does the enterprise tier cost?",
    "Please send the signed DPA for our compliance review.",
    "Love the new dashboard, great job!",
];

fn manifest_for(onnx_bytes: &[u8], tok_bytes: &[u8], max_tokens: usize) -> ModelManifest {
    ModelManifest {
        name: "openjev-candidate".into(),
        file: "model.onnx".into(),
        sha256: sha256_hex(onnx_bytes),
        dims: 384,
        license: "MIT".into(),
        source_url: "local".into(),
        added: "2026-09-26".into(),
        review_by: "2027-03-26".into(),
        pooling: Pooling::Cls,
        tokenizer_file: "tokenizer.json".into(),
        tokenizer_sha256: Some(sha256_hex(tok_bytes)),
        max_tokens,
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-24)
}

fn department_decisions<E: Embedder>(
    engine: &mut Engine<E>,
    data: &DataDir,
    texts: &[String],
) -> Result<Vec<String>> {
    let examples: Vec<LabeledExample> = data
        .train
        .iter()
        .filter(|r| r.dataset == TICKETS)
        .map(|r| LabeledExample {
            text: r.text.clone(),
            label: r.label.clone(),
        })
        .collect();
    engine
        .train("department", &examples)
        .map_err(|e| anyhow::anyhow!("engine train: {e}"))?;
    let criteria = data
        .labels
        .get(TICKETS)
        .context("labels.json lacks tickets")?;
    let mut out = Vec::with_capacity(texts.len());
    for t in texts {
        let req: DecisionRequest = serde_json::from_value(json!({
            "state": t,
            "questions": {"department": {"type": "choice",
                "instructions": "Which team should own this message", "criteria": criteria}}
        }))?;
        let resp = engine
            .decide(&req)
            .map_err(|e| anyhow::anyhow!("decide: {e}"))?;
        match resp.answers.get("department") {
            Some(Answer::Choice { choice, .. }) => out.push(choice.clone()),
            other => bail!("unexpected department answer {other:?}"),
        }
    }
    Ok(out)
}

pub fn run(a: &ParityArgs, device: &Device) -> Result<ParityReport> {
    let t0 = std::time::Instant::now();
    crate::pins::check_extension(&a.onnx)?;
    let onnx = std::fs::read(&a.onnx).with_context(|| format!("read {}", a.onnx.display()))?;
    let tok_bytes = crate::pins::read_verified(&a.tokenizer, crate::pins::TOKENIZER.sha256)?;
    let manifest = manifest_for(&onnx, &tok_bytes, a.max_tokens);
    let ort = OrtEmbedder::from_bytes(&onnx, &tok_bytes, &manifest)
        .map_err(|e| anyhow::anyhow!("ort load: {e}"))?;
    let (vm, _) = load_encoder(&a.weights, BertConfig::bge_small(), device)?;
    let bert = frozen_view(&vm, BertConfig::bge_small(), device)?;
    // The engine tokenizes at the manifest's max_tokens, so must the candle side.
    let candle = CandleEmbedder {
        bert,
        tok: load_tokenizer(&a.tokenizer, a.max_tokens)?,
        id: "candle".into(),
        dims: 384,
    };
    let data = a.data.as_deref().map(DataDir::load).transpose()?;
    let texts = probe_texts(data.as_ref(), a.probes);
    let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let mut cos = Vec::with_capacity(refs.len());
    for chunk in refs.chunks(32) {
        let eo = ort
            .embed(chunk)
            .map_err(|e| anyhow::anyhow!("ort embed: {e}"))?;
        let ec = candle
            .embed(chunk)
            .map_err(|e| anyhow::anyhow!("candle embed: {e}"))?;
        cos.extend(eo.iter().zip(&ec).map(|(x, y)| cosine(x, y)));
    }
    let mut sorted = cos.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let below = cos.iter().filter(|&&c| c < f64::from(a.threshold)).count();
    let (mut compared, mut identical) = (0, 0);
    if let Some(d) = data.as_ref() {
        let n = texts.len().min(256);
        let mut eo = Engine::new(ort);
        let mut ec = Engine::new(candle);
        let po = department_decisions(&mut eo, d, &texts[..n])?;
        let pc = department_decisions(&mut ec, d, &texts[..n])?;
        compared = n;
        identical = po.iter().zip(&pc).filter(|(x, y)| x == y).count();
    }
    let rep = ParityReport {
        onnx_sha256: manifest.sha256.clone(),
        probes: cos.len(),
        threshold: a.threshold,
        cosine_min: sorted.first().copied().unwrap_or(0.0),
        cosine_mean: cos.iter().sum::<f64>() / cos.len().max(1) as f64,
        cosine_p01: sorted.get(sorted.len() / 100).copied().unwrap_or(0.0),
        below_threshold: below,
        decisions_compared: compared,
        decisions_identical: identical,
        pass: below == 0 && !cos.is_empty() && compared == identical,
        seconds: t0.elapsed().as_secs_f64(),
    };
    Ok(rep)
}

pub fn write_report(path: &Path, rep: &ParityReport) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(rep)?)?;
    Ok(())
}
