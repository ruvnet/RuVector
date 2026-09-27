//! Export: candle safetensors → the engine's pinned ONNX graph (ADR-008 §5).
//! INT8 (`quantize`) is a stretch goal and not implemented in v0.

pub mod pbwalk;
pub mod transplant;

use std::fs;
use std::path::Path;

use anyhow::Result;

use crate::model::BertConfig;
use crate::pins;
use transplant::{load_named, TransplantReport};

/// Transplant `weights` (or the base itself when `weights` is None: identity)
/// into the template; writes the ONNX and returns the report.
pub fn run(
    template: &Path,
    base: &Path,
    weights: Option<&Path>,
    out: &Path,
) -> Result<TransplantReport> {
    let tbytes = pins::read_verified(template, pins::TEMPLATE_ONNX.sha256)?;
    pins::read_verified(base, pins::BASE_SAFETENSORS.sha256)?;
    let cfg = BertConfig::bge_small();
    let base_named = load_named(base, cfg)?;
    let new_named = match weights {
        Some(w) => load_named(w, cfg)?,
        None => base_named.clone(),
    };
    let (bytes, rep) = transplant::transplant(&tbytes, &base_named, &new_named)?;
    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(out, bytes)?;
    Ok(rep)
}

/// Stage a run for `bench/run.mjs --model-dir OUT --model NAME` (plan Step 4):
/// `OUT/NAME/{model.onnx,tokenizer.json,train-text-hashes.txt}` + `OUT/manifest.json`
/// with every file sha256-pinned (the bench refuses unpinned entries).
pub fn stage(
    onnx: &Path,
    tokenizer: &Path,
    train_hashes: &Path,
    name: &str,
    out: &Path,
) -> Result<serde_json::Value> {
    use crate::norm::sha256_hex;
    if name.is_empty()
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.' || b == b'_')
        || name.contains("..")
    {
        anyhow::bail!("stage: model name {name:?} must be [A-Za-z0-9._-]");
    }
    let hashes = crate::data::read_hashes(train_hashes)?;
    if hashes.is_empty() {
        anyhow::bail!("stage: {} is empty", train_hashes.display());
    }
    let dir = out.join(name);
    fs::create_dir_all(&dir)?;
    let onnx_b = fs::read(onnx)?;
    let tok_b = pins::read_verified(tokenizer, pins::TOKENIZER.sha256)?;
    let th_b = fs::read(train_hashes)?;
    fs::write(dir.join("model.onnx"), &onnx_b)?;
    fs::write(dir.join("tokenizer.json"), &tok_b)?;
    fs::write(dir.join("train-text-hashes.txt"), &th_b)?;
    let entry = serde_json::json!({
        "name": name,
        "file": format!("{name}/model.onnx"),
        "sha256": sha256_hex(&onnx_b),
        "dims": 384,
        "license": "UNRELEASED",
        "source_url": "local:unpublished",
        "added": "2026-09-26",
        "review_by": "2026-12-26",
        "pooling": "cls",
        "tokenizer_file": format!("{name}/tokenizer.json"),
        "tokenizer_sha256": sha256_hex(&tok_b),
        "max_tokens": 256,
        "train_hashes_file": format!("{name}/train-text-hashes.txt"),
        "train_hashes_sha256": sha256_hex(&th_b),
    });
    let manifest = serde_json::json!({ "models": [entry] });
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}
