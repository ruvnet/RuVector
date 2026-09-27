//! HPO selection (ADR-007 §2.6, §3; plan M4): from the **valid-only**
//! receipts of one dataset's C1–C8 runs (seed 100), build `selection.json`.
//!
//! - **Selection rule** (ADR-007 §2.6): the valid-MRR argmax under the verdict
//!   tie policy (Bottom, worst rank). An exact MRR tie between configs goes to
//!   the lowest config index (C1 before C2 …). No significance test overrides
//!   the argmax.
//! - **Decision-rule output** (ADR-007 §3, ADR-004 gate): every Ck is also run
//!   through the anytime paired sequential test (`PairedSequentialTest`, test
//!   by betting) against the anchor C1 at α/N = 0.05/8 = 0.00625. Pairs are the
//!   per-query Bottom ranks in dataset query order (tail at 2i, head at 2i+1),
//!   built exactly as `ruvector_kge::optimize::trainer_eval::pair_ranks` does
//!   (strictly better rank wins; equal ranks are concordant). This is reported
//!   beside the argmax; it does not change it. N stays the pre-registered 8
//!   even when fewer configs ran.
//! - **δ sizing** (ADR-007 §1 item 4, P0-3): sd(RR) of the per-query valid
//!   reciprocal ranks at the best epoch (sample sd, n−1), projected to one
//!   test-set standard error δ = sd(RR)/√(n_test_queries) with
//!   n_test_queries = 2 × |test| (a split *size*, not a score).
//!
//! `selection.json` never contains a test number: every input receipt must be
//! `mode = "valid"` with `eval.test_scored = false`, and the output is checked
//! for a `test` key before it is returned.

use crate::canon::{canonical_json, sha256_hex, utc_now};
use crate::config::RunConfig;
use crate::receipt::validate;
use anyhow::{bail, Context, Result};
use ruvector_typesafe_core::loop_gate::PairedSequentialTest;
use serde_json::{json, Value};

pub const SELECTION_SCHEMA: &str = "ruvector-kge-bench/selection@1";
/// ADR-007 §3: HPO seed, disjoint from the final seeds {0..4}.
pub const HPO_SEED: u64 = 100;
/// ADR-007 §3: pre-registered configs per dataset.
pub const N_CONFIGS: usize = 8;
/// ADR-007 §1 item 5 / §3: family α.
pub const FAMILY_ALPHA: f64 = 0.05;
/// ADR-004 anytime-test betting fraction (campaign default, λ = 0.5).
pub const GATE_LAMBDA: f32 = 0.5;

/// One config's valid-only HPO outcome, read from its receipt.
#[derive(Debug, Clone)]
pub struct ConfigResult {
    pub config_id: String,
    pub receipt: Value,
    pub mrr_bottom: f64,
    pub mrr_random: f64,
    pub ranks_bottom: Vec<usize>,
    pub ranks_random: Vec<usize>,
}

fn index_of(id: &str) -> Option<usize> {
    let n: usize = id.strip_prefix('C')?.parse().ok()?;
    (1..=N_CONFIGS).contains(&n).then_some(n)
}

fn ranks(v: &Value) -> Result<Vec<usize>> {
    v.as_array()
        .context("rank vector is not an array")?
        .iter()
        .map(|x| {
            x.as_u64()
                .filter(|&r| r >= 1)
                .map(|r| r as usize)
                .context("rank is not an integer >= 1")
        })
        .collect()
}

impl ConfigResult {
    /// Read and guard one HPO receipt: valid-only, seed 100, a grid config of
    /// `dataset` whose `config_hash` matches the frozen grid.
    pub fn from_receipt(dataset: &str, receipt: Value) -> Result<Self> {
        let mode = validate(&receipt)?;
        if mode != "valid" || receipt["eval"]["test_scored"] != json!(false) {
            bail!(
                "HPO selection reads valid-only receipts (mode '{mode}', test_scored {})",
                receipt["eval"]["test_scored"]
            );
        }
        if receipt.get("test").is_some() {
            bail!("receipt carries a test block; refusing");
        }
        if receipt["dataset"]["name"] != json!(dataset) {
            bail!(
                "receipt is for dataset {} not {dataset}",
                receipt["dataset"]["name"]
            );
        }
        if receipt["seed"] != json!(HPO_SEED) {
            bail!(
                "HPO receipts must use seed {HPO_SEED} (got {})",
                receipt["seed"]
            );
        }
        let run: RunConfig = serde_json::from_value(receipt["config"]["run"].clone())
            .context("receipt config.run")?;
        let id = run.config_id.clone();
        if index_of(&id).is_none() {
            bail!("config '{id}' is not one of C1..C{N_CONFIGS}");
        }
        if receipt["config"]["config_hash"] != json!(run.config_hash()?) {
            bail!("{id}: receipt config_hash does not match the frozen grid");
        }
        let best = &receipt["best"];
        if best.is_null() {
            bail!("{id}: receipt has no best-on-valid epoch");
        }
        let ranks_bottom = ranks(&best["valid_ranks"]["bottom"])?;
        let ranks_random = ranks(&best["valid_ranks"]["random"])?;
        if ranks_bottom.len() != ranks_random.len() || ranks_bottom.is_empty() {
            bail!("{id}: malformed valid rank vectors");
        }
        let mrr = |r: &[usize]| r.iter().map(|&k| 1.0 / k as f64).sum::<f64>() / r.len() as f64;
        Ok(Self {
            mrr_bottom: mrr(&ranks_bottom),
            mrr_random: mrr(&ranks_random),
            config_id: id,
            receipt,
            ranks_bottom,
            ranks_random,
        })
    }
}

/// Sample sd (n−1) of the per-query reciprocal ranks.
pub fn sd_rr(ranks: &[usize]) -> f64 {
    let n = ranks.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let rr: Vec<f64> = ranks.iter().map(|&k| 1.0 / k as f64).collect();
    let m = rr.iter().sum::<f64>() / n;
    (rr.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// `trainer_eval::pair_ranks`: (baseline wins, candidate wins) per query.
pub fn pair_ranks(baseline: &[usize], candidate: &[usize]) -> Vec<(bool, bool)> {
    baseline
        .iter()
        .zip(candidate)
        .map(|(&b, &c)| (c > b, c < b))
        .collect()
}

/// Index of the argmax (valid MRR Bottom); exact ties → lowest config index.
pub fn argmax(results: &[ConfigResult]) -> Result<usize> {
    let mut best: Option<usize> = None;
    for (i, r) in results.iter().enumerate() {
        best = Some(match best {
            None => i,
            Some(b) => {
                let (rb, ri) = (&results[b], r);
                let better = ri.mrr_bottom > rb.mrr_bottom
                    || (ri.mrr_bottom == rb.mrr_bottom
                        && index_of(&ri.config_id) < index_of(&rb.config_id));
                if better {
                    i
                } else {
                    b
                }
            }
        });
    }
    best.context("no HPO results to select from")
}

/// Assemble `selection.json` for one dataset. `plan` is the pre-registered
/// HPO plan (hashed in), `jobs` the per-config job statuses, `extra` a
/// provenance block from the caller.
pub fn build(
    dataset: &str,
    mut results: Vec<ConfigResult>,
    plan: &Value,
    jobs: &Value,
    provenance: Value,
) -> Result<Value> {
    results.sort_by_key(|r| index_of(&r.config_id));
    for w in results.windows(2) {
        if w[0].config_id == w[1].config_id {
            bail!("duplicate receipt for {}", w[0].config_id);
        }
    }
    let n_test_queries = results
        .first()
        .and_then(|r| r.receipt["dataset"]["counts"]["test"].as_u64())
        .context("receipt dataset.counts.test")?
        * 2;
    let alpha_n = FAMILY_ALPHA / N_CONFIGS as f64;
    let anchor = results.iter().find(|r| r.config_id == "C1");
    let win = argmax(&results)?;
    let delta = |sd: f64| sd / (n_test_queries as f64).sqrt();

    let mut configs = Vec::new();
    for r in &results {
        let rc = &r.receipt;
        let (sdb, sdr) = (sd_rr(&r.ranks_bottom), sd_rr(&r.ranks_random));
        let gate = match anchor {
            Some(a) if a.config_id != r.config_id => {
                if a.ranks_bottom.len() != r.ranks_bottom.len() {
                    bail!("{} and C1 rank different query sets", r.config_id);
                }
                let mut t = PairedSequentialTest::new(alpha_n as f32, GATE_LAMBDA);
                t.update_all(&pair_ranks(&a.ranks_bottom, &r.ranks_bottom));
                json!(t.statistic())
            }
            _ => Value::Null,
        };
        configs.push(json!({
            "config_id": r.config_id,
            "config_hash": rc["config"]["config_hash"],
            "complex_rank": rc["config"]["complex_rank"],
            "recipe": rc["config"]["canonical"],
            "receipt_sha256": rc["receipt_sha256"],
            "valid_mrr_bottom": r.mrr_bottom,
            "valid_mrr_random": r.mrr_random,
            "valid_metrics": rc["best"]["metrics"],
            "best_epoch": rc["best"]["epoch"],
            "epochs_run": rc["epochs"].as_array().map(|a| a.len()),
            "stopped": rc["stopped"],
            "wall_secs": rc["timings"]["invocation_wall_secs"],
            "valid_queries": r.ranks_bottom.len(),
            "sd_rr_bottom": sdb,
            "sd_rr_random": sdr,
            "delta_bottom": delta(sdb),
            "delta_random": delta(sdr),
            "gate_vs_c1": gate,
        }));
    }
    let w = &results[win];
    let complete = results.len() == N_CONFIGS
        && results
            .iter()
            .all(|r| r.receipt["stopped"] != json!("interrupted"));
    let passes = configs[win]["gate_vs_c1"]["rejected"].as_bool();
    let mut body = json!({
        "schema": SELECTION_SCHEMA,
        "generated_at": utc_now(),
        "dataset": dataset,
        "hpo_seed": HPO_SEED,
        "split": "valid",
        "test_scored": false,
        "complete": complete,
        "complete_note": "complete = all 8 pre-registered configs ran to early stop or max_epochs. \
            An incomplete selection is a local M4 measurement, NOT the M7 record pushed to kge-final-ledger.",
        "configs_run": results.iter().map(|r| r.config_id.clone()).collect::<Vec<_>>(),
        "plan_sha256": sha256_hex(canonical_json(plan)),
        "plan": plan,
        "jobs": jobs,
        "selected": {
            "config_id": w.config_id,
            "config_hash": w.receipt["config"]["config_hash"],
            "receipt_sha256": w.receipt["receipt_sha256"],
            "valid_mrr_bottom": w.mrr_bottom,
            "valid_mrr_random": w.mrr_random,
        },
        "hpo_receipt_hashes": results.iter()
            .map(|r| (r.config_id.clone(), r.receipt["receipt_sha256"].clone()))
            .collect::<serde_json::Map<_, _>>(),
        "decision_rule": {
            "selection": "argmax valid MRR (Bottom / worst-rank tie policy) over the configs run; exact ties -> lowest config index",
            "gate": "ADR-004 anytime paired sequential test (betting) of each Ck vs anchor C1 on per-query valid Bottom ranks, dataset query order; reported, never overrides the argmax",
            "family_alpha": FAMILY_ALPHA,
            "n_preregistered": N_CONFIGS,
            "alpha_per_proposal": alpha_n,
            "lambda": GATE_LAMBDA,
            "selected_is_anchor": w.config_id == "C1",
            "selected_passes_gate_vs_c1": passes,
            "margin_over_c1_mrr_bottom": anchor.map(|a| w.mrr_bottom - a.mrr_bottom),
        },
        "delta": {
            "definition": "delta = sd(RR_valid) / sqrt(n_test_queries), sample sd at the best-on-valid epoch; ADR-007 §1 item 4 (frozen at M5)",
            "n_test_queries": n_test_queries,
            "selected_bottom": delta(sd_rr(&w.ranks_bottom)),
            "selected_random": delta(sd_rr(&w.ranks_random)),
        },
        "configs": configs,
        "provenance": provenance,
    });
    if contains_key(&body, "test") {
        bail!("internal: selection body contains a `test` key");
    }
    let h = sha256_hex(canonical_json(&body));
    body.as_object_mut()
        .unwrap()
        .insert("selection_sha256".into(), json!(h));
    Ok(body)
}

/// Recursive key search (the no-test-numbers guard).
pub fn contains_key(v: &Value, key: &str) -> bool {
    match v {
        Value::Object(m) => m.iter().any(|(k, x)| k == key || contains_key(x, key)),
        Value::Array(a) => a.iter().any(|x| contains_key(x, key)),
        _ => false,
    }
}
