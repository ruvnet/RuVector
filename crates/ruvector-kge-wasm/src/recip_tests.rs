//! Plan M3 binding tests: the `complex` scorer and reciprocal models.
//! Duplicated VERBATIM in the ffi and wasm crates (next to `recip.rs`).

use super::{copy_rows, query_route};
use crate::model::KgeModel;
use ruvector_kge::scorer::ComplEx;
use ruvector_kge::{evaluate, EvalConfig, Scorer, Side, Tables, TieBreak, TripleStore};
use serde_json::Value;

const RECIPE: &str = r#"{"loss":{"kind":"one_vs_all"},"n3_form":"moduli",
    "loss_reduction":"mean","init":{"kind":"normal","scale":0.001},
    "optimizer":{"kind":"adagrad"},"optim_state":"dense","n3_lambda":0.01,
    "rp_weight":0.05,"one_n_kernel":"gemm","epochs":3,"batch_size":16,"lr":0.1,"seed":3}"#;

fn graph(ne: usize, nr: usize, nt: usize, tag: &str) -> String {
    let t: Vec<String> = (0..nt)
        .map(|i| {
            let (s, r) = (i % ne, (i / ne) % nr);
            let o = (s * 7 + r * 13 + 1) % ne;
            format!(r#"{{"s":"e{s}","r":"{tag}{r}","o":"e{o}"}}"#)
        })
        .collect();
    format!("[{}]", t.join(","))
}

fn complex(reciprocal: bool) -> KgeModel {
    let opts = format!(r#"{{"scorer":"complex","dims":8,"seed":5,"reciprocal":{reciprocal}}}"#);
    let mut m = KgeModel::new(&opts).unwrap();
    let added: Value = serde_json::from_str(&m.add_triples_json(&graph(30, 3, 90, "r"))).unwrap();
    assert_eq!(added["added"], 90);
    m
}

fn json(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|e| panic!("bad JSON {s}: {e}"))
}

fn kind(s: &str) -> String {
    json(s)["error"]["kind"].as_str().unwrap_or("").to_string()
}

#[test]
fn complex_reciprocal_model_holds_two_r_rows() {
    let mut m = complex(true);
    m.ensure_built();
    assert_eq!(m.tables.as_ref().unwrap().num_relations(), 6);
    let stats = json(&m.stats_json());
    assert_eq!(stats["scorer"], "complex");
    assert_eq!(stats["reciprocal"], true);
    assert_eq!(stats["relations"], 3);
    let mut plain = complex(false);
    plain.ensure_built();
    assert_eq!(plain.tables.as_ref().unwrap().num_relations(), 3);
    assert_eq!(json(&plain.stats_json())["reciprocal"], false);
}

#[test]
fn non_reciprocal_models_serialize_without_the_flag() {
    let plain = complex(false);
    assert!(
        !plain.to_json().contains("reciprocal"),
        "hash compatibility"
    );
    let recip = complex(true);
    let back = KgeModel::from_json(&recip.to_json()).unwrap();
    assert!(back.config.reciprocal);
    assert_eq!(back.to_json(), recip.to_json());
}

#[test]
fn copy_rows_moves_the_inverse_block() {
    let d = 2;
    let mut src = Tables::new(3, 4, d, 1); // R = 2
    for (i, x) in src.relations_raw_mut().iter_mut().enumerate() {
        *x = 100.0 + i as f32;
    }
    let mut dst = Tables::new(5, 6, d, 2); // R = 3
    let fresh = dst.relations_raw().to_vec();
    copy_rows(&src, &mut dst, true);
    let r = |t: &Tables, i: u32| t.relation(i).unwrap().to_vec();
    assert_eq!(r(&dst, 0), r(&src, 0));
    assert_eq!(r(&dst, 1), r(&src, 1));
    assert_eq!(r(&dst, 3), r(&src, 2), "inverse of r0 moves 2 -> 3");
    assert_eq!(r(&dst, 4), r(&src, 3), "inverse of r1 moves 3 -> 4");
    assert_eq!(
        r(&dst, 2),
        fresh[4..6].to_vec(),
        "new base row keeps its init"
    );
    assert_eq!(
        r(&dst, 5),
        fresh[10..12].to_vec(),
        "new inverse row keeps its init"
    );
    assert_eq!(dst.entity(2).unwrap(), src.entity(2).unwrap());
    // Non-reciprocal: plain prefix.
    let mut dst = Tables::new(5, 6, d, 2);
    copy_rows(&src, &mut dst, false);
    assert_eq!(&dst.relations_raw()[..8], src.relations_raw());
}

#[test]
fn growth_keeps_inverse_rows_paired() {
    let mut m = complex(true);
    m.ensure_built();
    let marker = vec![7.0f32; 8];
    m.tables
        .as_mut()
        .unwrap()
        .relation_mut(3)
        .unwrap()
        .copy_from_slice(&marker); // r0⁻¹
    m.add_triples_json(r#"[{"s":"e1","r":"new","o":"e2"}]"#);
    let t = m.tables.as_ref().unwrap();
    assert_eq!(t.num_relations(), 8);
    assert_eq!(t.relation(4).unwrap(), &marker[..], "r0⁻¹ now at R = 4");
    assert_ne!(
        t.relation(3).unwrap(),
        &marker[..],
        "row 3 is the new base relation"
    );
}

#[test]
fn reciprocal_head_predict_scores_the_inverse_tail_query() {
    let mut m = complex(true);
    m.ensure_built();
    let (o, r) = (
        m.entities.get("e4").unwrap(),
        m.relations.get("r1").unwrap(),
    );
    let t = m.tables.as_ref().unwrap();
    let (inv, side) = query_route(t, r, Side::Head, true);
    assert_eq!((inv, side), (r + 3, Side::Tail));
    let sc = ComplEx::new(8).unwrap();
    let mut want: Vec<(u32, f32)> = (0..t.num_entities() as u32)
        .map(|e| {
            let (a, rv) = (t.entity(o).unwrap(), t.relation(inv).unwrap());
            (e, sc.score(a, rv, t.entity(e).unwrap()))
        })
        .collect();
    want.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let got = json(&m.predict_json(r#"{"r":"r1","o":"e4","k":5,"useIndex":false}"#));
    let got: Vec<String> = got["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["entity"].as_str().unwrap().to_string())
        .collect();
    let want: Vec<String> = want[..5]
        .iter()
        .map(|(e, _)| m.entities.label(*e).unwrap().to_string())
        .collect();
    assert_eq!(got, want);
    // The ANN route answers the same routed query.
    assert_eq!(json(&m.build_index_json())["indexed"], true);
    let ann = json(&m.predict_json(r#"{"r":"r1","o":"e4","k":1}"#));
    assert_eq!(ann["ann"], true);
    assert_eq!(
        ann["candidates"][0]["entity"],
        Value::String(want[0].clone())
    );
}

#[test]
fn recipe_trains_and_evaluates_a_reciprocal_complex_model() {
    let mut m = complex(true);
    let rep = json(&m.train_json(RECIPE));
    assert_eq!(rep["status"], "trained", "{rep}");
    assert!(rep["loss"].as_f64().unwrap().is_finite());
    let ev = json(&m.eval_json(r#"{"tieBreak":"bottom","seed":1}"#));
    assert_eq!(ev["reciprocal"], true, "{ev}");
    // The binding's report is exactly core eval with the reciprocal protocol.
    let store = TripleStore::new(m.triples.clone()).unwrap();
    let cfg = EvalConfig {
        tie_break: TieBreak::Bottom,
        filtered: true,
        seed: 1,
        reciprocal: true,
    };
    let sc = ComplEx::new(8).unwrap();
    let want = evaluate(m.tables.as_ref().unwrap(), &sc, &store, &m.triples, &cfg).unwrap();
    let got: ruvector_kge::EvalReport = serde_json::from_value(ev["report"].clone()).unwrap();
    assert_eq!(got, want);
}

#[test]
fn train_reciprocal_must_match_the_model() {
    let mut recip = complex(true);
    assert_eq!(
        kind(&recip.train_json(r#"{"reciprocal":false,"epochs":1}"#)),
        "invalid"
    );
    let ok = json(&recip.train_json(r#"{"reciprocal":true,"epochs":1}"#));
    assert_eq!(ok["status"], "trained", "{ok}");
    let mut plain = complex(false);
    assert_eq!(
        kind(&plain.train_json(r#"{"reciprocal":true,"epochs":1}"#)),
        "invalid"
    );
    // Backwards compatible: omitted inherits; the old config shape still trains.
    let ok = json(&plain.train_json(r#"{"epochs":1,"lr":0.05}"#));
    assert_eq!(ok["status"], "trained", "{ok}");
}

#[test]
fn snapshot_swap_relocates_trained_inverse_rows() {
    let mut m = complex(true);
    let mut job = m.begin_train(RECIPE).unwrap();
    assert!(!job.in_place());
    m.add_triples_json(r#"[{"s":"e1","r":"late","o":"e2"}]"#);
    job.run().unwrap();
    let trained = job.tables_for_test().clone();
    let out = json(&m.finish_train(job, Ok(())));
    assert_eq!(out["replayed"]["relations"], 1, "{out}");
    let live = m.tables.as_ref().unwrap();
    assert_eq!(live.num_relations(), 8);
    for r in 0..3u32 {
        assert_eq!(live.relation(r).unwrap(), trained.relation(r).unwrap());
        assert_eq!(
            live.relation(4 + r).unwrap(),
            trained.relation(3 + r).unwrap()
        );
    }
}

#[test]
fn reciprocal_loads_fail_closed_on_a_wrong_row_count() {
    let mut m = complex(true);
    m.ensure_built();
    assert!(m.validate_loaded().is_ok());
    m.tables = Some(Tables::new(30, 3, 8, 1));
    assert!(m.validate_loaded().is_err());
}

#[test]
fn complex_and_reciprocal_surface_limits() {
    let mut m = complex(true);
    m.ensure_built();
    assert_eq!(kind(&m.optimize_json("{}")), "unsupported");
    assert_eq!(
        kind(&m.compose_json(r#"{"r1":"r0","r2":"r1","s":"e1"}"#)),
        "unsupported"
    );
    let sim = json(&m.similar_relations_json(r#"{"r":"r0","k":10}"#));
    let labels: Vec<&str> = sim["relations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["relation"].as_str().unwrap())
        .collect();
    assert_eq!(labels.len(), 2, "only labelled relations: {labels:?}");
    assert!(labels.iter().all(|l| !l.is_empty()));
}
