//! Integration tests for the native `ort` embedder. `#[ignore]`d because they
//! need the model files (`scripts/fetch-models.mjs`); run with:
//!   cargo test -p ruvector-embed-core --features native -- --include-ignored
#![cfg(feature = "native")]

use std::path::{Path, PathBuf};

use ruvector_embed_core::{Embedder, ManifestFile, OrtEmbedder};

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

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
#[ignore = "requires model files"]
fn dims_and_normalisation() {
    if skip_if_missing() {
        return;
    }
    let mf = manifest();
    let m = mf.get("bge-small-en-v1.5").unwrap();
    let emb = OrtEmbedder::from_manifest(models_dir(), m).expect("load");

    assert_eq!(emb.dims(), 384);
    assert!(emb.id().starts_with("bge-small-en-v1.5@"));

    let out = emb
        .embed(&[
            "a helpdesk ticket about a broken laptop",
            "the invoice was paid late",
        ])
        .expect("embed");
    assert_eq!(out.len(), 2);
    for v in &out {
        assert_eq!(v.len(), 384);
        assert!(
            (l2(v) - 1.0).abs() < 1e-4,
            "not L2-normalised: |v|={}",
            l2(v)
        );
    }
}

#[test]
#[ignore = "requires model files"]
fn batch_equals_single() {
    if skip_if_missing() {
        return;
    }
    let mf = manifest();
    let m = mf.get("all-MiniLM-L6-v2").unwrap();
    let emb = OrtEmbedder::from_manifest(models_dir(), m).expect("load");

    let texts = [
        "the comet tail was faint before dawn",
        "we whisked the sourdough starter twice",
        "the gravel tire was swapped before the climb",
    ];
    let batched = emb.embed(&texts).expect("batched");
    for (i, t) in texts.iter().enumerate() {
        let single = emb.embed(&[*t]).expect("single");
        let c = cosine(&batched[i], &single[0]);
        assert!(
            c >= 0.9999,
            "batch vs single cosine {c} < 0.9999 for text {i}"
        );
    }
}

#[test]
#[ignore = "requires model files"]
fn int8_native_loads_and_embeds() {
    if skip_if_missing() {
        return;
    }
    let mf = manifest();
    let m = mf.get("bge-small-en-v1.5-int8").unwrap();
    // ort must load the quantized graph natively (INT8 is native-only if tract
    // cannot — see the spike). Here we assert native INT8 works.
    let emb = OrtEmbedder::from_manifest(models_dir(), m).expect("int8 native load");
    let out = emb
        .embed(&["a quantized model still embeds"])
        .expect("embed");
    assert_eq!(out[0].len(), 384);
    assert!((l2(&out[0]) - 1.0).abs() < 1e-4);
}
