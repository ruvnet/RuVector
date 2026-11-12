//! Numerical parity gate (ADR-002 §7): native `ort` and wasm-path `tract` must
//! agree to cosine >= 0.9999 per text. `#[ignore]`d — needs model files. Run:
//!   cargo test -p ruvector-embed-core --features native,wasm -- --include-ignored
#![cfg(all(feature = "native", feature = "wasm"))]

use std::path::{Path, PathBuf};

use ruvector_embed_core::{diagnose_load, Embedder, ManifestFile, OrtEmbedder, TractEmbedder};

fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../npm/packages/typesafe/models")
}

fn manifest() -> ManifestFile {
    let p = models_dir().join("manifest.json");
    ManifestFile::from_json(&std::fs::read_to_string(p).expect("manifest present")).unwrap()
}

fn skip_if_missing() -> bool {
    if !models_dir().join("bge-small-en-v1.5/model.onnx").exists() {
        eprintln!("SKIP: models absent; run scripts/fetch-models.mjs");
        return true;
    }
    false
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// First 50 distinct fixture texts from the ticket corpus (same source the
/// spike uses), so the CI gate is 50 real texts, not repeats.
fn fixtures_50() -> Vec<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../npm/packages/typesafe/bench/fixtures/tickets-corpus.json");
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    v["docs"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|d| d["text"].as_str().map(str::to_string))
        .take(50)
        .collect()
}

fn parity_for(model_name: &str) {
    if skip_if_missing() {
        return;
    }
    let mf = manifest();
    let m = mf.get(model_name).unwrap();

    let ort = OrtEmbedder::from_manifest(models_dir(), m).expect("ort load");
    let model_bytes = std::fs::read(models_dir().join(&m.file)).unwrap();
    let tok_bytes = std::fs::read(models_dir().join(&m.tokenizer_file)).unwrap();
    let tract = TractEmbedder::from_bytes(&model_bytes, &tok_bytes, m).expect("tract load");

    let texts = fixtures_50();
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let ov = ort.embed(&refs).expect("ort embed");
    let tv = tract.embed(&refs).expect("tract embed");
    assert_eq!(ov.len(), tv.len());
    assert_eq!(ov.len(), 50);

    let mut min_cos = f32::INFINITY;
    for (a, b) in ov.iter().zip(&tv) {
        min_cos = min_cos.min(cosine(a, b));
    }
    assert!(
        min_cos >= 0.9999,
        "{model_name}: min cosine {min_cos} < 0.9999"
    );
}

#[test]
#[ignore = "requires model files"]
fn ort_tract_parity_bge_fp32() {
    parity_for("bge-small-en-v1.5");
}

#[test]
#[ignore = "requires model files"]
fn ort_tract_parity_minilm_fp32() {
    parity_for("all-MiniLM-L6-v2");
}

#[test]
#[ignore = "requires model files"]
fn int8_tract_load_outcome_is_recorded() {
    if skip_if_missing() {
        return;
    }
    let mf = manifest();
    let m = mf.get("bge-small-en-v1.5-int8").unwrap();
    let bytes = std::fs::read(models_dir().join(&m.file)).unwrap();
    // This does not assert pass/fail — the point is that the outcome is a typed
    // value we can record. The spike JSON captures which it was.
    let outcome = diagnose_load(&bytes, m.max_tokens);
    eprintln!("INT8 tract 0.23 load outcome: {outcome:?}");
}
