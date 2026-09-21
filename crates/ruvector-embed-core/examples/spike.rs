//! Embedder spike (ADR-002 §3, §7): FP32/INT8 × ort/tract × batch 1/32 timing,
//! the INT8-on-tract-0.23 load result, and ort-vs-tract cosine parity.
//!
//! Native build of both backends (the wasm `tract` path built for the host is
//! fine for timing). Writes `bench/embedder-spike-2026-09-21.json` and prints a
//! table. Requires the models fetched by `scripts/fetch-models.mjs`.
//!
//! Run: `cargo run -p ruvector-embed-core --features native,wasm --example spike --release`

use std::path::{Path, PathBuf};
use std::time::Instant;

use ruvector_embed_core::{
    diagnose_load, Embedder, LoadOutcome, ManifestFile, OrtEmbedder, TractEmbedder,
};
use serde_json::{json, Value};

const SPIKE_SEQ_TEXTS: usize = 100;
const PARITY_TEXTS: usize = 50;

fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../npm/packages/typesafe/models")
}
fn bench_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../npm/packages/typesafe/bench")
}

fn load_fixtures(n: usize) -> Vec<String> {
    let path = bench_dir().join("fixtures/tickets-corpus.json");
    let v: Value = serde_json::from_slice(&std::fs::read(&path).expect("read corpus")).unwrap();
    v["docs"]
        .as_array()
        .expect("docs array")
        .iter()
        .filter_map(|d| d["text"].as_str().map(str::to_string))
        .take(n)
        .collect()
}

/// Mean ms per embed over `texts`, in chunks of `batch`. Cache is warmed by
/// the caller first, so this measures inference, not model/plan build.
fn time_ms_per_embed(emb: &dyn Embedder, texts: &[String], batch: usize) -> f64 {
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let start = Instant::now();
    let mut count = 0usize;
    for chunk in refs.chunks(batch) {
        let _ = emb.embed(chunk).expect("embed");
        count += chunk.len();
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    elapsed / count as f64
}

fn warm(emb: &dyn Embedder, texts: &[String]) {
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    for chunk in refs.chunks(32) {
        let _ = emb.embed(chunk).expect("warm embed");
    }
    for r in &refs {
        let _ = emb.embed(&[r]).expect("warm embed single");
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() {
    let dir = models_dir();
    let manifest: ManifestFile = ManifestFile::from_json(
        &std::fs::read_to_string(dir.join("manifest.json")).expect("read manifest"),
    )
    .expect("parse manifest");

    let texts = load_fixtures(SPIKE_SEQ_TEXTS);
    assert!(
        texts.len() >= PARITY_TEXTS,
        "need >= {PARITY_TEXTS} fixtures"
    );

    let mut results: Vec<Value> = Vec::new();
    let mut int8_tract: Vec<Value> = Vec::new();
    let mut parity: Vec<Value> = Vec::new();

    // Group the four manifest entries into (model, fp32-entry, int8-entry).
    let groups: &[(&str, &str, &str)] = &[
        (
            "bge-small-en-v1.5",
            "bge-small-en-v1.5",
            "bge-small-en-v1.5-int8",
        ),
        (
            "all-MiniLM-L6-v2",
            "all-MiniLM-L6-v2",
            "all-MiniLM-L6-v2-int8",
        ),
    ];

    println!(
        "{:<20} {:<6} {:<6} {:>12} {:>12}",
        "model", "prec", "backend", "batch1 ms", "batch32 ms"
    );

    for (model, fp32_name, int8_name) in groups {
        let fp32 = manifest.get(fp32_name).expect("fp32 entry");
        let int8 = manifest.get(int8_name).expect("int8 entry");

        // Keep fp32 ort/tract embedders to compute parity later.
        let mut fp32_ort_vecs: Vec<Vec<f32>> = Vec::new();
        let mut fp32_tract_vecs: Vec<Vec<f32>> = Vec::new();

        for (prec, m) in [("fp32", fp32), ("int8", int8)] {
            // ---- ort (native) ----
            match OrtEmbedder::from_manifest(&dir, m) {
                Ok(emb) => {
                    warm(&emb, &texts);
                    let b1 = time_ms_per_embed(&emb, &texts, 1);
                    let b32 = time_ms_per_embed(&emb, &texts, 32);
                    row(&mut results, model, prec, "ort", Some(b1), Some(b32), None);
                    println!("{model:<20} {prec:<6} {:<6} {b1:>12.3} {b32:>12.3}", "ort");
                    if prec == "fp32" {
                        fp32_ort_vecs = embed_all(&emb, &texts[..PARITY_TEXTS]);
                    }
                }
                Err(e) => row(
                    &mut results,
                    model,
                    prec,
                    "ort",
                    None,
                    None,
                    Some(e.to_string()),
                ),
            }

            // ---- tract (wasm path, native build) ----
            let model_bytes = std::fs::read(dir.join(&m.file)).expect("read model");
            let tok_bytes = std::fs::read(dir.join(&m.tokenizer_file)).expect("read tok");
            let outcome = diagnose_load(&model_bytes, m.max_tokens);
            if prec == "int8" {
                int8_tract.push(json!({ "model": model, "outcome": describe(&outcome) }));
            }
            if outcome.is_runnable() {
                let emb =
                    TractEmbedder::from_bytes(&model_bytes, &tok_bytes, m).expect("tract embedder");
                warm(&emb, &texts);
                let b1 = time_ms_per_embed(&emb, &texts, 1);
                let b32 = time_ms_per_embed(&emb, &texts, 32);
                row(
                    &mut results,
                    model,
                    prec,
                    "tract",
                    Some(b1),
                    Some(b32),
                    None,
                );
                println!(
                    "{model:<20} {prec:<6} {:<6} {b1:>12.3} {b32:>12.3}",
                    "tract"
                );
                if prec == "fp32" {
                    fp32_tract_vecs = embed_all(&emb, &texts[..PARITY_TEXTS]);
                }
            } else {
                row(
                    &mut results,
                    model,
                    prec,
                    "tract",
                    None,
                    None,
                    Some(describe(&outcome)),
                );
                println!(
                    "{model:<20} {prec:<6} {:<6} {:>12} {:>12}",
                    "tract", "LOAD-FAIL", "-"
                );
            }
        }

        // Parity on fp32 (int8 differs numerically by design).
        if fp32_ort_vecs.len() == fp32_tract_vecs.len() && !fp32_ort_vecs.is_empty() {
            let min_cos = fp32_ort_vecs
                .iter()
                .zip(&fp32_tract_vecs)
                .map(|(a, b)| cosine(a, b))
                .fold(f32::INFINITY, f32::min);
            parity.push(json!({
                "model": model,
                "precision": "fp32",
                "texts": PARITY_TEXTS,
                "min_cosine_ort_vs_tract": min_cos,
            }));
            println!("parity {model} fp32: min cosine(ort,tract) = {min_cos:.6}");
        }
    }

    let out = json!({
        "date": "2026-09-21",
        "spike": "ruvector-embed-core ort/tract embedder (ADR-002)",
        "ort_version": "2.0.0-rc.13",
        "tract_version": "0.23",
        "fixtures": SPIKE_SEQ_TEXTS,
        "parity_texts": PARITY_TEXTS,
        "single_thread": true,
        "results": results,
        "int8_tract_0_23": int8_tract,
        "parity": parity,
    });
    let out_path = bench_dir().join("embedder-spike-2026-09-21.json");
    std::fs::write(
        &out_path,
        serde_json::to_string_pretty(&out).unwrap() + "\n",
    )
    .unwrap();
    println!("\nWrote {}", out_path.display());
}

fn embed_all(emb: &dyn Embedder, texts: &[String]) -> Vec<Vec<f32>> {
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    emb.embed(&refs).expect("embed_all")
}

#[allow(clippy::too_many_arguments)]
fn row(
    results: &mut Vec<Value>,
    model: &str,
    prec: &str,
    backend: &str,
    b1: Option<f64>,
    b32: Option<f64>,
    note: Option<String>,
) {
    results.push(json!({
        "model": model,
        "precision": prec,
        "backend": backend,
        "batch1_ms_per_embed": b1,
        "batch32_ms_per_embed": b32,
        "note": note,
    }));
}

fn describe(o: &LoadOutcome) -> String {
    match o {
        LoadOutcome::Runnable => "Runnable".to_string(),
        LoadOutcome::FailedParse(e) => format!("FailedParse: {e}"),
        LoadOutcome::FailedOptimize(e) => format!("FailedOptimize: {e}"),
        LoadOutcome::FailedRunnable(e) => format!("FailedRunnable: {e}"),
    }
}
