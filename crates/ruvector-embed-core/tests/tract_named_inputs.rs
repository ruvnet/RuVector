//! Regression tests for the tract (`wasm` feature) embedder's graph handling:
//!
//!  * symbolic `batch_size` / `sequence_length` inputs (nomic-embed-text-v1.5
//!    failed at `optimize` with `Impossible to unify Sym(batch_size) with
//!    Val(1)` when the inputs were pinned to a concrete `[1, seq_len]`);
//!  * binding inputs by NAME rather than position (nomic declares
//!    `input_ids, token_type_ids, attention_mask`; a positional
//!    `[ids, mask, types]` silently fed the mask into `token_type_ids`).
//!
//! All tests are `#[ignore]`d: they need real model files, never stand-ins.
//!   cargo test -p ruvector-embed-core --features wasm --test tract_named_inputs -- --include-ignored
//!
//! Models are read from `$RUVECTOR_EMBED_MODELS_DIR` (default
//! `npm/packages/typesafe/models`, filled by `scripts/fetch-models.mjs`):
//!   bge-small-en-v1.5/{model.onnx,tokenizer.json}
//!   all-MiniLM-L6-v2/{model.onnx,tokenizer.json}
//!   nomic-embed-text-v1.5/{model.onnx,tokenizer.json}   (HF nomic-ai/nomic-embed-text-v1.5, onnx/model.onnx)
//! Hashes are pinned below and verified before use. A missing model skips
//! (prints SKIP); a wrong hash fails.
//!
//! Fixtures (tests/fixtures):
//!  * `nomic-embed-text-v1.5-ort-reference.json` - vectors from onnxruntime;
//!  * `tract-prepatch-golden.json` - bge / MiniLM vectors from the tract
//!    backend BEFORE the symbolic-dims + named-inputs change. Regenerate (on an
//!    unchanged tree only) with `RUVECTOR_WRITE_GOLDEN=<path>` and the ignored
//!    `write_prepatch_golden` test.
#![cfg(feature = "wasm")]

use std::path::{Path, PathBuf};

use ruvector_embed_core::manifest::sha256_hex;
use ruvector_embed_core::{Embedder, ModelManifest, Pooling, TractEmbedder};

const BGE_SHA: &str = "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35";
const MINILM_SHA: &str = "759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e";
const NOMIC_SHA: &str = "147d5aa88c2101237358e17796cf3a227cead1ec304ec34b465bb08e9d952965";

struct Model {
    dir: PathBuf,
    name: &'static str,
    sha: &'static str,
    dims: usize,
    pooling: Pooling,
    max_tokens: usize,
}

impl Model {
    fn bge() -> Self {
        Self::new("bge-small-en-v1.5", BGE_SHA, 384, Pooling::Cls, 256)
    }
    fn minilm() -> Self {
        Self::new("all-MiniLM-L6-v2", MINILM_SHA, 384, Pooling::Mean, 256)
    }
    fn nomic() -> Self {
        Self::new("nomic-embed-text-v1.5", NOMIC_SHA, 768, Pooling::Mean, 512)
    }
    fn new(
        name: &'static str,
        sha: &'static str,
        dims: usize,
        pooling: Pooling,
        max_tokens: usize,
    ) -> Self {
        let root = std::env::var_os("RUVECTOR_EMBED_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../../npm/packages/typesafe/models")
            });
        Self {
            dir: root.join(name),
            name,
            sha,
            dims,
            pooling,
            max_tokens,
        }
    }

    /// `None` (and a SKIP line) when the files are absent.
    fn load_bytes(&self) -> Option<(Vec<u8>, Vec<u8>)> {
        let (m, t) = (self.dir.join("model.onnx"), self.dir.join("tokenizer.json"));
        if !m.exists() || !t.exists() {
            eprintln!(
                "SKIP {}: {} lacks model.onnx / tokenizer.json",
                self.name,
                self.dir.display()
            );
            return None;
        }
        let model = std::fs::read(m).unwrap();
        assert_eq!(
            sha256_hex(&model),
            self.sha,
            "{}: model.onnx hash differs from the pin",
            self.name
        );
        Some((model, std::fs::read(t).unwrap()))
    }

    fn manifest(&self, model_sha: &str, tok: &[u8]) -> ModelManifest {
        ModelManifest {
            name: self.name.into(),
            file: "model.onnx".into(),
            sha256: model_sha.into(),
            dims: self.dims,
            license: "see source".into(),
            source_url: "https://huggingface.co".into(),
            added: "2026-11-13".into(),
            review_by: "2027-05-13".into(),
            pooling: self.pooling,
            tokenizer_file: "tokenizer.json".into(),
            tokenizer_sha256: Some(sha256_hex(tok)),
            max_tokens: self.max_tokens,
        }
    }

    fn embedder(&self, model: &[u8], tok: &[u8]) -> TractEmbedder {
        TractEmbedder::from_bytes(model, tok, &self.manifest(&sha256_hex(model), tok))
            .unwrap_or_else(|e| panic!("{}: tract load failed: {e}", self.name))
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum() // both L2-normalised
}

/// Texts of different token lengths, embedded one at a time AND as one batch.
const TEXTS: [&str; 5] = [
    "The borrow checker rejected my Rust code with a lifetime error.",
    "why does the borrow checker complain",
    "a helpdesk ticket about a broken laptop",
    "ok",
    "WeftOS keeps memory, governance and a chain of decisions, and runs on laptops, servers and small boards, \
     so an agent can remember what it did yesterday, explain why it did it, and ask before it does anything \
     it was not told it could do, while every decision is written to a log that cannot be quietly edited later.",
];

fn embed_each(e: &TractEmbedder, texts: &[&str]) -> Vec<Vec<f32>> {
    texts
        .iter()
        .map(|t| e.embed(&[t]).expect("embed").remove(0))
        .collect()
}

fn fixture(name: &str) -> serde_json::Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    serde_json::from_slice(&std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
        .unwrap()
}

fn vec_of(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

// ---------------------------------------------------------------- nomic vs ort

#[test]
#[ignore = "requires model files"]
fn nomic_tract_matches_ort_reference() {
    let m = Model::nomic();
    let Some((model, tok)) = m.load_bytes() else {
        return;
    };
    let fx = fixture("nomic-embed-text-v1.5-ort-reference.json");
    assert_eq!(fx["model_sha256"], NOMIC_SHA);
    let e = m.embedder(&model, &tok);

    let mut min_cos = f32::INFINITY;
    for t in fx["texts"].as_array().unwrap() {
        let prefix = if t["lane"] == "query" {
            "search_query: "
        } else {
            "search_document: "
        };
        let text = format!("{prefix}{}", t["text"].as_str().unwrap());
        let got = e.embed(&[text.as_str()]).expect("embed").remove(0);
        let want = vec_of(&t["vector"]);
        assert_eq!(got.len(), 768);
        min_cos = min_cos.min(cosine(&got, &want));
    }
    eprintln!("nomic tract vs ort min cosine {min_cos}");
    assert!(
        min_cos >= 0.9999,
        "nomic tract vs ort min cosine {min_cos} < 0.9999"
    );
}

// ----------------------------------------------- bge / MiniLM unchanged by patch

fn golden_check(m: Model) {
    let Some((model, tok)) = m.load_bytes() else {
        return;
    };
    let e = m.embedder(&model, &tok);
    let golden = fixture("tract-prepatch-golden.json");
    let rows = golden["models"][m.name]
        .as_array()
        .unwrap_or_else(|| panic!("no golden for {}", m.name));
    assert_eq!(rows.len(), TEXTS.len());
    let each = embed_each(&e, &TEXTS);
    let batched = e.embed(&TEXTS).expect("batch embed");
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row["text"], TEXTS[i]);
        let want = vec_of(&row["vector"]);
        let c1 = cosine(&each[i], &want);
        let c2 = cosine(&batched[i], &want);
        eprintln!(
            "{} text {i}: cosine vs pre-patch golden single {c1} batched {c2}",
            m.name
        );
        assert!(
            c1 >= 0.99999,
            "{} text {i}: single vs pre-patch golden cosine {c1}",
            m.name
        );
        assert!(
            c2 >= 0.9999,
            "{} text {i}: batched vs pre-patch golden cosine {c2}",
            m.name
        );
    }
}

#[test]
#[ignore = "requires model files"]
fn bge_small_unchanged_vs_prepatch_golden() {
    golden_check(Model::bge());
}

#[test]
#[ignore = "requires model files"]
fn minilm_unchanged_vs_prepatch_golden() {
    golden_check(Model::minilm());
}

/// Run on an UNPATCHED tree to (re)generate the golden file.
#[test]
#[ignore = "requires model files and RUVECTOR_WRITE_GOLDEN=<out.json>"]
fn write_prepatch_golden() {
    let Some(out) = std::env::var_os("RUVECTOR_WRITE_GOLDEN") else {
        eprintln!("SKIP: RUVECTOR_WRITE_GOLDEN not set");
        return;
    };
    let mut models = serde_json::Map::new();
    for m in [Model::bge(), Model::minilm()] {
        let (model, tok) = m.load_bytes().expect("model files needed to write golden");
        let e = m.embedder(&model, &tok);
        let rows: Vec<_> = embed_each(&e, &TEXTS)
            .into_iter()
            .zip(TEXTS)
            .map(|(v, t)| serde_json::json!({ "text": t, "vector": v }))
            .collect();
        models.insert(m.name.into(), rows.into());
    }
    let doc = serde_json::json!({
        "engine": "ruvector-embed-core TractEmbedder, tract 0.23, before symbolic-dims + named-inputs (upstream/main 86fda361f)",
        "models": models,
    });
    std::fs::write(out, serde_json::to_vec_pretty(&doc).unwrap()).unwrap();
}

// ------------------------------------------------------------- bind by name

/// Rewrite the ONNX `graph.input` list so the named inputs appear in `order`
/// (weights and every other field untouched). Minimal protobuf walk:
/// ModelProto.graph = field 7, GraphProto.input = field 11, ValueInfoProto.name = field 1.
fn reorder_graph_inputs(onnx: &[u8], order: &[&str]) -> Vec<u8> {
    fn varint(b: &[u8], i: &mut usize) -> u64 {
        let (mut v, mut s) = (0u64, 0);
        loop {
            let x = b[*i];
            *i += 1;
            v |= u64::from(x & 0x7f) << s;
            if x & 0x80 == 0 {
                return v;
            }
            s += 7;
        }
    }
    fn put_varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8 & 0x7f) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    /// Split a message into (field number, wire type, full record bytes, payload range).
    fn fields(b: &[u8]) -> Vec<(u64, Vec<u8>, std::ops::Range<usize>)> {
        let (mut i, mut v) = (0, vec![]);
        while i < b.len() {
            let start = i;
            let tag = varint(b, &mut i);
            let range = match tag & 7 {
                0 => {
                    varint(b, &mut i);
                    i..i
                }
                1 => {
                    i += 8;
                    i..i
                }
                2 => {
                    let n = varint(b, &mut i) as usize;
                    i += n;
                    i - n..i
                }
                5 => {
                    i += 4;
                    i..i
                }
                w => panic!("unexpected wire type {w}"),
            };
            v.push((tag >> 3, b[start..i].to_vec(), range));
        }
        v
    }
    let name_of = |rec: &[u8], payload: std::ops::Range<usize>| -> String {
        // record = tag + len + payload; recover the payload slice from the record tail.
        let p = &rec[rec.len() - (payload.end - payload.start)..];
        let f = fields(p)
            .into_iter()
            .find(|f| f.0 == 1)
            .expect("input has a name");
        String::from_utf8(p[f.2].to_vec()).unwrap()
    };

    let mut model_out = Vec::new();
    for (num, rec, payload) in fields(onnx) {
        if num != 7 {
            model_out.extend_from_slice(&rec);
            continue;
        }
        let graph = &rec[rec.len() - (payload.end - payload.start)..];
        let (mut rest, mut inputs) = (Vec::new(), Vec::new());
        for (gnum, grec, gpay) in fields(graph) {
            if gnum == 11 {
                inputs.push((name_of(&grec, gpay), grec));
            } else {
                rest.extend_from_slice(&grec);
            }
        }
        let pos = |n: &str| order.iter().position(|o| *o == n).unwrap_or(usize::MAX);
        inputs.sort_by_key(|(n, _)| pos(n));
        for (_, rec) in inputs {
            rest.extend_from_slice(&rec);
        }
        put_varint(&mut model_out, (7 << 3) | 2);
        put_varint(&mut model_out, rest.len() as u64);
        model_out.extend_from_slice(&rest);
    }
    model_out
}

/// A real model (bge-small weights) whose declared input order differs from the
/// old positional `[ids, mask, types]` assumption must embed the same as the
/// stock order. Before the fix the permuted graphs return garbage.
#[test]
#[ignore = "requires model files"]
fn input_binding_is_by_name_not_position() {
    let m = Model::bge();
    let Some((model, tok)) = m.load_bytes() else {
        return;
    };
    let base = embed_each(&m.embedder(&model, &tok), &TEXTS);

    let orders: [[&str; 3]; 3] = [
        ["input_ids", "token_type_ids", "attention_mask"], // nomic-embed-text's order
        ["token_type_ids", "attention_mask", "input_ids"],
        ["attention_mask", "input_ids", "token_type_ids"],
    ];
    for order in orders {
        let permuted = reorder_graph_inputs(&model, &order);
        assert_ne!(permuted, model, "{order:?}: rewrite changed nothing");
        let got = embed_each(&m.embedder(&permuted, &tok), &TEXTS);
        for (i, (g, b)) in got.iter().zip(&base).enumerate() {
            let c = cosine(g, b);
            eprintln!("input order {order:?}, text {i}: cosine vs stock order {c}");
            assert!(
                c >= 0.99999,
                "input order {order:?}, text {i}: cosine vs stock order {c}"
            );
        }
    }
}

/// Same property on a second model family (mean pooling, different graph).
#[test]
#[ignore = "requires model files"]
fn input_binding_by_name_minilm_reversed() {
    let m = Model::minilm();
    let Some((model, tok)) = m.load_bytes() else {
        return;
    };
    let base = embed_each(&m.embedder(&model, &tok), &TEXTS);
    let permuted = reorder_graph_inputs(&model, &["token_type_ids", "attention_mask", "input_ids"]);
    assert_ne!(permuted, model);
    let got = embed_each(&m.embedder(&permuted, &tok), &TEXTS);
    for (i, (g, b)) in got.iter().zip(&base).enumerate() {
        let c = cosine(g, b);
        assert!(c >= 0.99999, "MiniLM reversed inputs, text {i}: cosine {c}");
    }
}
