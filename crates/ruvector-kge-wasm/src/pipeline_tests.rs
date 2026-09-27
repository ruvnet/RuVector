//! F2 tests: training on a snapshot without holding the model lock.
//! Duplicated VERBATIM in the ffi and wasm crates (next to `pipeline.rs`).

use crate::model::KgeModel;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// A dense-ish synthetic graph: `ne` entities, `nr` relations, `nt` triples.
fn graph_json(ne: usize, nr: usize, nt: usize) -> String {
    let mut out = Vec::with_capacity(nt);
    for i in 0..nt {
        let s = i % ne;
        let r = (i / ne) % nr;
        let o = (s * 7 + r * 13 + 1) % ne;
        out.push(format!(r#"{{"s":"e{s}","r":"r{r}","o":"e{o}"}}"#));
    }
    format!("[{}]", out.join(","))
}

fn model(ne: usize, nr: usize, nt: usize) -> KgeModel {
    let mut m = KgeModel::new(r#"{"scorer":"hole","dims":32,"seed":7}"#).unwrap();
    let added: serde_json::Value =
        serde_json::from_str(&m.add_triples_json(&graph_json(ne, nr, nt))).unwrap();
    assert_eq!(added["added"], nt);
    m
}

fn candidates(json: &str) -> Vec<(String, f64)> {
    let v: serde_json::Value = serde_json::from_str(json).unwrap();
    v["candidates"]
        .as_array()
        .unwrap_or_else(|| panic!("no candidates in {json}"))
        .iter()
        .map(|c| {
            (
                c["entity"].as_str().unwrap().to_string(),
                c["score"].as_f64().unwrap(),
            )
        })
        .collect()
}

const Q: &str = r#"{"s":"e1","r":"r0","k":5,"useIndex":false}"#;

#[test]
fn predict_is_served_while_training_runs() {
    // Calibrate epochs so the unlocked run takes > 1 s on this build profile.
    let probe = {
        let mut m = model(200, 4, 2000);
        let mut job = m
            .begin_train(r#"{"epochs":1,"batch_size":256,"lr":0.05}"#)
            .unwrap();
        let t = Instant::now();
        job.run().unwrap();
        t.elapsed()
    };
    let epochs = (Duration::from_millis(1500).as_secs_f64() / probe.as_secs_f64().max(1e-4))
        .ceil()
        .max(2.0) as usize;

    let shared = Arc::new(RwLock::new(model(200, 4, 2000)));
    let before = candidates(&shared.write().unwrap().predict_json(Q));
    let cfg = format!(r#"{{"epochs":{epochs},"batch_size":256,"lr":0.05}}"#);
    let mut job = shared.write().unwrap().begin_train(&cfg).unwrap();
    let trainer = std::thread::spawn(move || {
        let t = Instant::now();
        let fit = job.run();
        (job, fit, t.elapsed())
    });

    let mut served = 0usize;
    let mut worst = Duration::ZERO;
    while !trainer.is_finished() {
        let t = Instant::now();
        let out = shared.write().unwrap().predict_json(Q);
        let dt = t.elapsed();
        worst = worst.max(dt);
        assert!(
            dt < Duration::from_millis(50),
            "predict took {dt:?} during training"
        );
        let c = candidates(&out);
        assert_eq!(c.len(), 5);
        // Served from the pre-training tables: identical to the pre-run answer.
        assert_eq!(c, before, "predict during training must use the old tables");
        served += 1;
        std::thread::sleep(Duration::from_millis(5));
    }
    let (job, fit, took) = trainer.join().unwrap();
    assert!(
        took > Duration::from_secs(1),
        "training too short to prove anything: {took:?}"
    );
    assert!(
        served >= 10,
        "only {served} predicts ran during {took:?} of training"
    );
    let report: serde_json::Value =
        serde_json::from_str(&shared.write().unwrap().finish_train(job, fit)).unwrap();
    eprintln!("F2: {served} predicts during {took:?} of training; worst {worst:?}");
    assert_eq!(report["status"], "trained", "{report}");
    assert!(worst < Duration::from_millis(50));
}

#[test]
fn post_train_predictions_reflect_trained_tables() {
    let mut m = model(60, 3, 300);
    let before = candidates(&m.predict_json(Q));
    let out: serde_json::Value =
        serde_json::from_str(&m.train_json(r#"{"epochs":20,"batch_size":64,"lr":0.1}"#)).unwrap();
    assert_eq!(out["status"], "trained", "{out}");
    let after = candidates(&m.predict_json(Q));
    assert_ne!(before, after, "predictions must change after training");
    // The swapped-in tables are exactly the ones predict scores with: a second
    // identical query is stable, and the true tail (e1 --r0--> e8) ranks top-5.
    assert_eq!(after, candidates(&m.predict_json(Q)));
    assert!(
        after.iter().any(|(e, _)| e == "e8"),
        "trained tail missing: {after:?}"
    );
}

#[test]
fn second_train_is_single_flight() {
    let mut m = model(30, 2, 60);
    let job = m.begin_train(r#"{"epochs":1}"#).unwrap();
    let busy = m.begin_train(r#"{"epochs":1}"#).err().unwrap();
    assert_eq!(busy, r#"{"status":"training"}"#);
    // The composed sync form honours the same flag.
    assert_eq!(m.train_json(r#"{"epochs":1}"#), r#"{"status":"training"}"#);
    let mut job = job;
    let fit = job.run();
    assert!(m.finish_train(job, fit).contains("\"trained\""));
    // Flag cleared: training may start again; abort also clears it.
    let _j = m.begin_train(r#"{"epochs":1}"#).unwrap();
    m.abort_train();
    assert!(m.begin_train(r#"{"epochs":1}"#).is_ok());
}

#[test]
fn growth_during_training_is_replayed() {
    let mut m = model(30, 2, 60);
    let mut job = m.begin_train(r#"{"epochs":5,"lr":0.1}"#).unwrap();
    job.run().unwrap();
    let seed_row_31 = {
        // A row interned mid-run: it must survive the swap as seed-init.
        m.add_triples_json(r#"[{"s":"new-a","r":"new-r","o":"new-b"}]"#);
        let grown = m.tables.as_ref().unwrap();
        assert_eq!(grown.num_entities(), 32);
        grown.entity(31).unwrap().to_vec() // seed-init row of a mid-run entity
    };
    let out: serde_json::Value = serde_json::from_str(&m.finish_train(job, Ok(()))).unwrap();
    assert_eq!(out["status"], "trained", "{out}");
    assert_eq!(out["replayed"]["entities"], 2);
    assert_eq!(out["replayed"]["relations"], 1);
    let t = m.tables.as_ref().unwrap();
    assert_eq!(t.num_entities(), 32);
    assert_eq!(t.num_relations(), 3);
    assert_eq!(
        t.entity(31).unwrap(),
        &seed_row_31[..],
        "mid-run rows keep seed-init"
    );
    assert_eq!(m.triples.len(), 61, "mid-run triples are kept");
    // The mid-run entity is immediately predictable.
    assert_eq!(
        candidates(&m.predict_json(r#"{"s":"new-a","r":"new-r","k":3}"#)).len(),
        3
    );
}

#[test]
fn trained_rows_land_in_the_live_tables_after_growth() {
    let mut m = model(30, 2, 60);
    let mut job = m.begin_train(r#"{"epochs":5,"lr":0.1}"#).unwrap();
    job.run().unwrap();
    let expect = job_entity_row(&job, 3);
    m.add_triples_json(r#"[{"s":"x","r":"r0","o":"y"}]"#);
    m.finish_train(job, Ok(()));
    assert_eq!(m.tables.as_ref().unwrap().entity(3).unwrap(), &expect[..]);
}

fn job_entity_row(job: &crate::pipeline::TrainJob, id: u32) -> Vec<f32> {
    job.tables_for_test().entity(id).unwrap().to_vec()
}

#[test]
fn wholesale_replacement_during_training_refuses_the_swap() {
    let mut m = model(30, 2, 60);
    let mut job = m.begin_train(r#"{"epochs":2}"#).unwrap();
    job.run().unwrap();
    let live = m.tables.clone().unwrap();
    m.replace_tables(live.clone()); // what an optimize champion install does
    let out: serde_json::Value = serde_json::from_str(&m.finish_train(job, Ok(()))).unwrap();
    assert_eq!(out["error"]["kind"], "unavailable", "{out}");
    assert_eq!(
        m.tables.as_ref().unwrap(),
        &live,
        "live tables untouched on refusal"
    );
    assert!(
        m.begin_train(r#"{"epochs":1}"#).is_ok(),
        "flag cleared on refusal"
    );
}

#[test]
fn transient_fields_do_not_change_the_serialized_model() {
    let mut m = model(10, 2, 12);
    m.ensure_built();
    let before = m.to_json();
    let _job = m.begin_train(r#"{"epochs":1}"#).unwrap();
    // training flag + generation are #[serde(skip)]: same bytes, same hash.
    assert_eq!(m.to_json(), before);
    assert!(KgeModel::from_json(&before).is_ok());
}

/// 30 entities + 2 relations at 32 dims = 32 rows x 128 B = 4096 B of tables.
fn capped(cap: u64) -> KgeModel {
    let mut m = KgeModel::new(&format!(
        r#"{{"scorer":"hole","dims":32,"seed":7,"maxTableBytes":{cap}}}"#
    ))
    .unwrap();
    let out = m.add_triples_json(&graph_json(30, 2, 60));
    assert!(!out.contains("\"error\""), "{out}");
    m.ensure_built();
    assert_eq!(m.tables.as_ref().unwrap().num_entities(), 30);
    m
}

#[test]
fn near_cap_model_trains_in_place_without_a_copy() {
    // Tables use 4096 B of a 6000 B cap: one copy fits, two do not.
    let mut m = capped(6000);
    let job = m.begin_train(r#"{"epochs":1}"#).unwrap();
    assert!(
        job.in_place(),
        "a doubled working set must not be snapshotted"
    );
    // No single-flight flag for an in-place job (the caller holds the model).
    let mut job = job;
    assert!(job.run().is_err(), "an in-place job has no snapshot to run");
    let fit = job.run_in_place(m.tables.as_mut().unwrap());
    let out: serde_json::Value = serde_json::from_str(&m.finish_train(job, fit)).unwrap();
    assert_eq!(out["status"], "trained", "{out}");
    assert_eq!(out["mode"], "in-place");

    // The composed (wasm) form trains in place too and the result is live.
    let before = candidates(&m.predict_json(Q));
    let out: serde_json::Value =
        serde_json::from_str(&m.train_json(r#"{"epochs":20,"batch_size":64,"lr":0.1}"#)).unwrap();
    assert_eq!(out["status"], "trained", "{out}");
    assert_eq!(out["mode"], "in-place");
    assert_ne!(before, candidates(&m.predict_json(Q)));
    assert!(m.begin_train(r#"{"epochs":1}"#).is_ok(), "no flag left set");
}

#[test]
fn snapshot_taken_only_when_two_copies_fit_the_cap() {
    // Exactly 2 x 4096 B: the snapshot fits beside the live tables.
    let mut m = capped(8192);
    let mut job = m.begin_train(r#"{"epochs":1}"#).unwrap();
    assert!(!job.in_place());
    let fit = job.run();
    let out: serde_json::Value = serde_json::from_str(&m.finish_train(job, fit)).unwrap();
    assert_eq!(out["mode"], "snapshot", "{out}");
    // One byte less and it falls back to in place.
    assert!(capped(8191)
        .begin_train(r#"{"epochs":1}"#)
        .unwrap()
        .in_place());
}

#[test]
fn wasm_style_train_json_never_snapshots() {
    // Even an uncapped model trains in place through the composed form.
    let mut m = model(30, 2, 60);
    let out: serde_json::Value = serde_json::from_str(&m.train_json(r#"{"epochs":2}"#)).unwrap();
    assert_eq!(out["mode"], "in-place", "{out}");
}
