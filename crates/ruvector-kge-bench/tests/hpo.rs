//! HPO plan validation and `selection.json` assembly (ADR-007 §2.6, §3).

use ruvector_kge_bench::config::RunConfig;
use ruvector_kge_bench::hpo::HpoPlan;
use ruvector_kge_bench::receipt::seal;
use ruvector_kge_bench::selection::{self, argmax, contains_key, sd_rr, ConfigResult};
use serde_json::{json, Value};

const N_TEST: u64 = 3134;

fn run_cfg(id: &str, seed: u64) -> RunConfig {
    RunConfig {
        dataset: "wn18rr".into(),
        config_id: id.into(),
        recipe: None,
        seed,
        max_epochs: 10,
        early_stop_patience: Some(3),
        eval_every: 1,
        threads: Some(2),
    }
}

/// A sealed valid-mode receipt with the given Bottom ranks (RANDOM = Bottom).
fn receipt(id: &str, ranks: &[usize]) -> Value {
    receipt_with(id, ranks, 100, "valid", false)
}

fn receipt_with(id: &str, ranks: &[usize], seed: u64, mode: &str, test_scored: bool) -> Value {
    let cfg = run_cfg(id, seed);
    seal(json!({
        "mode": mode,
        "generated_at": "2026-09-27T00:00:00Z",
        "provenance": {},
        "dataset": { "name": "wn18rr", "file_hashes": {}, "splits_hash": "x", "entity_vocab_hash": "y",
                     "counts": { "train": 1, "valid": ranks.len() / 2, "test": N_TEST } },
        "config": { "config_hash": cfg.config_hash().unwrap(), "canonical": cfg.recipe_canonical().unwrap(),
                    "run": cfg, "complex_rank": 1000 },
        "seed": seed,
        "eval": { "split": "valid", "test_scored": test_scored },
        "epochs": [{}, {}],
        "stopped": "early_stop",
        "best": { "epoch": 1, "metrics": {}, "valid_ranks": { "bottom": ranks, "random": ranks } },
        "timings": { "invocation_wall_secs": 1.0 },
    }))
    .unwrap()
}

fn plan() -> Value {
    json!({ "plan": { "dataset": "wn18rr", "configs": ["C1", "C2", "C3"] } })
}

#[test]
fn guards_reject_non_hpo_receipts() {
    let r = vec![1, 2, 3, 4];
    assert!(ConfigResult::from_receipt("wn18rr", receipt("C1", &r)).is_ok());
    assert!(ConfigResult::from_receipt("fb15k237", receipt("C1", &r)).is_err());
    assert!(
        ConfigResult::from_receipt("wn18rr", receipt_with("C1", &r, 0, "valid", false)).is_err()
    );
    assert!(
        ConfigResult::from_receipt("wn18rr", receipt_with("C1", &r, 100, "valid", true)).is_err()
    );
    // A tampered body fails the receipt hash.
    let mut t = receipt("C2", &r);
    t["best"]["valid_ranks"]["bottom"] = json!([1, 1, 1, 1]);
    assert!(ConfigResult::from_receipt("wn18rr", t).is_err());
    // A config_hash that is not the grid's is refused (resealed so only the hash check fires).
    let mut h = receipt("C2", &r);
    h["config"]["config_hash"] = json!("0".repeat(64));
    h.as_object_mut().unwrap().remove("receipt_sha256");
    assert!(ConfigResult::from_receipt("wn18rr", seal(h).unwrap()).is_err());
}

#[test]
fn argmax_and_lowest_index_tiebreak() {
    let a = ConfigResult::from_receipt("wn18rr", receipt("C3", &[1, 2, 1, 2])).unwrap();
    let b = ConfigResult::from_receipt("wn18rr", receipt("C2", &[1, 2, 1, 2])).unwrap();
    let c = ConfigResult::from_receipt("wn18rr", receipt("C1", &[2, 2, 2, 2])).unwrap();
    let v = vec![a, b, c];
    assert_eq!(v[argmax(&v).unwrap()].config_id, "C2");
}

#[test]
fn selection_has_winner_delta_gate_and_no_test_numbers() {
    // 400 queries: C2 wins 300 discordant pairs vs C1, loses none → gate rejects.
    let c1: Vec<usize> = (0..400).map(|i| if i < 300 { 5 } else { 1 }).collect();
    let c2: Vec<usize> = (0..400).map(|_| 1).collect();
    let c3: Vec<usize> = (0..400).map(|i| if i % 2 == 0 { 1 } else { 10 }).collect();
    let rs = [("C3", &c3), ("C1", &c1), ("C2", &c2)]
        .iter()
        .map(|(id, r)| ConfigResult::from_receipt("wn18rr", receipt(id, r)).unwrap())
        .collect();
    let s = selection::build("wn18rr", rs, &plan(), &json!({}), json!({})).unwrap();
    assert_eq!(s["selected"]["config_id"], "C2");
    assert_eq!(s["complete"], false);
    assert_eq!(s["configs_run"], json!(["C1", "C2", "C3"]));
    assert_eq!(s["decision_rule"]["alpha_per_proposal"], json!(0.00625));
    assert_eq!(s["decision_rule"]["selected_passes_gate_vs_c1"], true);
    let c3row = &s["configs"][2];
    assert_eq!(c3row["config_id"], "C3");
    assert_eq!(s["delta"]["n_test_queries"], json!(2 * N_TEST));
    let want = sd_rr(&c3) / ((2 * N_TEST) as f64).sqrt();
    assert!((c3row["delta_bottom"].as_f64().unwrap() - want).abs() < 1e-15);
    assert!(
        s["configs"][0]["gate_vs_c1"].is_null(),
        "C1 is not tested against itself"
    );
    assert!(!contains_key(&s, "test"));
    assert_eq!(s["test_scored"], false);
    assert_eq!(s["selection_sha256"].as_str().unwrap().len(), 64);
}

#[test]
fn sd_rr_is_sample_sd() {
    // RR = [1, 0.5]: mean 0.75, sample var = 2·0.0625/1.
    assert!((sd_rr(&[1, 2]) - 0.125f64.sqrt()).abs() < 1e-15);
    assert_eq!(sd_rr(&[3]), 0.0);
}

#[test]
fn plan_validation() {
    let ok: HpoPlan = serde_json::from_value(json!({
        "dataset": "wn18rr", "configs": ["C1", "C5", "C3"], "max_epochs": 100, "early_stop_patience": 5
    }))
    .unwrap();
    ok.validate().unwrap();
    let rc = ok.run_config("C5", 8).unwrap();
    assert_eq!(
        (rc.seed, rc.threads, rc.config_id.as_str()),
        (100, Some(8), "C5")
    );
    for bad in [
        json!({"dataset":"wn18rr","configs":["C1"],"max_epochs":101,"early_stop_patience":5}),
        json!({"dataset":"wn18rr","configs":["C9"],"max_epochs":10,"early_stop_patience":5}),
        json!({"dataset":"wn18rr","configs":["custom"],"max_epochs":10,"early_stop_patience":5}),
        json!({"dataset":"wn18rr","configs":["C1","C1"],"max_epochs":10,"early_stop_patience":5}),
        json!({"dataset":"yago310","configs":["C1"],"max_epochs":10,"early_stop_patience":5}),
        json!({"dataset":"wn18rr","configs":[],"max_epochs":10,"early_stop_patience":5}),
    ] {
        let p: HpoPlan = serde_json::from_value(bad.clone()).unwrap();
        assert!(p.validate().is_err(), "{bad}");
    }
    assert!(serde_json::from_value::<HpoPlan>(
        json!({"dataset":"wn18rr","configs":["C1"],"max_epochs":10,"early_stop_patience":5,"seed":0})
    )
    .is_err(), "the seed is fixed at 100, not a plan field");
}

#[test]
fn reuse_refuses_receipts_from_another_plan() {
    let plan: HpoPlan = serde_json::from_value(json!({
        "dataset": "wn18rr", "configs": ["C1"], "max_epochs": 10, "early_stop_patience": 3
    }))
    .unwrap();
    // `receipt()` uses max_epochs 10, patience 3, eval_every 1, threads 2.
    let r = receipt("C1", &[1, 2]);
    ruvector_kge_bench::hpo::check_matches_plan(&r, &plan, "C1").unwrap();
    let mut other = plan.clone();
    other.max_epochs = 6;
    assert!(ruvector_kge_bench::hpo::check_matches_plan(&r, &other, "C1").is_err());
    let mut other = plan.clone();
    other.early_stop_patience = 5;
    assert!(ruvector_kge_bench::hpo::check_matches_plan(&r, &other, "C1").is_err());
    assert!(ruvector_kge_bench::hpo::check_matches_plan(&r, &plan, "C2").is_err());
}
