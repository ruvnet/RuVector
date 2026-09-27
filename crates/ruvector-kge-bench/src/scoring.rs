//! The data side of `--final` / `--verify-final`: read a finished run
//! directory, score **test** on its best-on-valid tables (filter = train ∪
//! valid ∪ transfer ∪ test, ADR-007 §2.1), and seal the receipt; or recompute
//! the ranks from an exported table and compare their hashes with a receipt
//! already recorded on origin's ledger.
//! The ledger/tag gate lives in `final_mode`; git is reached only through
//! `FinalGit`.

use crate::canon::{atomic_write, utc_now};
use crate::config::{RunConfig, EVAL_SEED};
use crate::datasets::Dataset;
use crate::export::{load_tables, Manifest};
use crate::final_mode::{run_final, Authorized, FinalGit, FinalRequest, Scored};
use crate::ledger::{self, Line};
use crate::metrics::{rank_split, RankVectors};
use crate::receipt::{provenance, seal, validate};
use crate::runner::{config_block, dataset_block, with_threads};
use anyhow::{bail, Context, Result};
use ruvector_kge::Tables;
use serde_json::{json, Value};
use std::path::Path;

/// Final runs train the anchors' 500 epochs with no early stop (ADR-007 §3).
pub const FINAL_EPOCHS: usize = 500;

/// A finished valid-mode run, ready to be scored.
pub struct RunDir {
    pub config: RunConfig,
    pub receipt: Value,
    pub tables: Tables,
    pub manifest: Manifest,
}

/// Load and check `dir` (its `receipt.json` and `best/` export) against `ds`.
/// With `require_final_protocol`, the run must be a 500-epoch, no-early-stop,
/// uninterrupted run (the pre-registered final shape).
pub fn load_run_dir(dir: &Path, ds: &Dataset, require_final_protocol: bool) -> Result<RunDir> {
    let receipt: Value = serde_json::from_slice(
        &std::fs::read(dir.join("receipt.json")).context("read run receipt.json")?,
    )?;
    if validate(&receipt)? != "valid" {
        bail!("{} is not a training (valid-mode) receipt", dir.display());
    }
    let config: RunConfig =
        serde_json::from_value(receipt["config"]["run"].clone()).context("receipt config.run")?;
    if config.config_hash()? != receipt["config"]["config_hash"] {
        bail!("receipt config hash does not match its config");
    }
    if receipt["dataset"]["splits_hash"] != json!(ds.splits_hash)
        || receipt["dataset"]["file_hashes"] != json!(ds.file_hashes)
    {
        bail!("run receipt was produced on different dataset bytes");
    }
    if require_final_protocol
        && (config.max_epochs != FINAL_EPOCHS
            || config.early_stop_patience.is_some()
            || receipt["stopped"] != json!("max_epochs"))
    {
        bail!(
            "refusing --final: the run is not a completed {FINAL_EPOCHS}-epoch, no-early-stop run"
        );
    }
    let (tables, manifest) = load_tables(&dir.join("best"), Some(ds))?;
    if receipt["best"]["weights_sha256"] != json!(manifest.weights_sha256)
        || manifest.config_hash != config.config_hash()?
    {
        bail!("best/ export does not match the run receipt's best-on-valid tables");
    }
    Ok(RunDir {
        config,
        receipt,
        tables,
        manifest,
    })
}

fn test_block(ranks: &RankVectors) -> Value {
    json!({ "split": "test", "metrics": ranks.metrics(), "ranks": ranks })
}

fn eval_block() -> Value {
    json!({
        "split": "test", "filter": "train+valid+transfer+test", "tie_breaks": ["bottom", "random"],
        "verdict_tie_break": "bottom", "eval_seed": EVAL_SEED, "reciprocal_head_queries": true,
        "rank_hash": "sha256 over u32 little-endian ranks; tail query at 2i, head at 2i+1",
    })
}

/// Score test once on `run`'s best tables and build the sealed `final`
/// receipt (no file written, no git). Called only from inside the gate
/// (`run_final`'s scoring step), after the intent is on origin.
pub fn score_test_receipt(
    run: &RunDir,
    ds: &Dataset,
    auth: &Authorized,
    repo: &Path,
) -> Result<Value> {
    let filter = ds.filter_store(true)?;
    let ranks = with_threads(run.config.threads, |_| {
        rank_split(&run.tables, &filter, &ds.test)
    })?;
    seal(json!({
        "mode": "final",
        "generated_at": utc_now(),
        "provenance": provenance(repo, run.config.threads.unwrap_or(0)),
        "dataset": dataset_block(ds),
        "config": config_block(&run.config)?,
        "seed": run.config.seed,
        "eval": eval_block(),
        "test": test_block(&ranks),
        "tables": { "weights_sha256": run.manifest.weights_sha256, "manifest": run.manifest, "dir_hint": "best" },
        "source_run_receipt_sha256": run.receipt["receipt_sha256"],
        "final": {
            "tag": crate::ledger::PREREG_TAG, "tag_sha": auth.tag.tag_object, "tag_commit": auth.tag.commit,
            "ledger_branch": crate::ledger::LEDGER_BRANCH, "ledger_base": auth.origin.head, "retry": auth.retry,
        },
    }))
}

/// `--final`: gate through `git`, then score test once on `run`'s best
/// tables and write `out` (the sealed `final` receipt).
pub fn final_score(
    git: &dyn FinalGit,
    run: &RunDir,
    ds: &Dataset,
    out: &Path,
    repo: &Path,
) -> Result<Value> {
    let req = FinalRequest {
        dataset: ds.name.clone(),
        seed: run.config.seed,
        config_hash: run.config.config_hash()?,
        config_is_grid: run.config.is_grid(),
        checkpoint_sha256: run.manifest.weights_sha256.clone(),
    };
    let mut sealed = Value::Null;
    let score = |auth: &Authorized| -> Result<Scored> {
        let r = score_test_receipt(run, ds, auth, repo)?;
        atomic_write(out, serde_json::to_string_pretty(&r)?.as_bytes())?;
        let scored = Scored {
            receipt_sha256: r["receipt_sha256"].as_str().unwrap_or_default().into(),
            bottom_ranks_sha256: r["test"]["ranks"]["bottom_sha256"]
                .as_str()
                .unwrap_or_default()
                .into(),
            random_ranks_sha256: r["test"]["ranks"]["random_sha256"]
                .as_str()
                .unwrap_or_default()
                .into(),
        };
        sealed = r;
        Ok(scored)
    };
    run_final(git, &req, &utc_now(), score)?;
    Ok(sealed)
}

/// `--verify-final`: recompute the test ranks from the exported `tables_dir`
/// and compare their sha256 with `final_receipt`. Returns a sealed
/// `verification` receipt carrying only the hash comparisons and `pass` — no
/// metrics, no rank vectors — so a failing verification reveals nothing about
/// test. Writes no ledger line and is never a new scoring.
///
/// Refuses **before** any test query is ranked (ADR-007 §2.8, P0-6) unless
/// (1) the tables' `weights_sha256` equals the receipt's, and (2) origin's
/// `kge-final-ledger` (read through `git`) holds a `result` line for this
/// receipt: same `receipt_sha256`, `checkpoint_sha256`, dataset, seed, config
/// hash and rank hashes. So only a scoring already on the ledger can be
/// re-derived, and only from the very tables it was scored on; an HPO
/// candidate paired with a real or forged receipt is never scored on test.
pub fn verify_final(
    git: &dyn FinalGit,
    final_receipt: &Value,
    ds: &Dataset,
    tables_dir: &Path,
    threads: Option<usize>,
    repo: &Path,
) -> Result<Value> {
    if validate(final_receipt)? != "final" {
        bail!("--verify-final needs a `final` receipt");
    }
    if final_receipt["dataset"]["name"] != json!(ds.name)
        || final_receipt["dataset"]["file_hashes"] != json!(ds.file_hashes)
        || final_receipt["dataset"]["splits_hash"] != json!(ds.splits_hash)
    {
        bail!("dataset bytes differ from the receipt's");
    }
    let want_w = final_receipt["tables"]["weights_sha256"]
        .as_str()
        .context("receipt tables.weights_sha256 is not a string")?;
    let (tables, manifest) = load_tables(tables_dir, Some(ds))?;
    // (1) Only the tables the receipt was scored on.
    if manifest.weights_sha256 != want_w {
        bail!("refusing --verify-final: tables weights_sha256 {} != receipt's {want_w}; nothing was scored", manifest.weights_sha256);
    }
    // (2) Only a scoring already recorded on origin's ledger.
    let seed = final_receipt["seed"]
        .as_u64()
        .context("receipt seed is not an integer")?;
    let want_r = final_receipt["receipt_sha256"]
        .as_str()
        .context("receipt_sha256 is not a string")?;
    let want_cfg = final_receipt["config"]["config_hash"]
        .as_str()
        .context("receipt config.config_hash is not a string")?;
    let want = &final_receipt["test"]["ranks"];
    let origin = git.fetch_origin_ledger().context(
        "refusing --verify-final: cannot fetch origin's ledger branch; nothing was scored",
    )?;
    let ledgered = ledger::parse(&origin.ledger)
        .context("refusing --verify-final: origin's ledger does not parse; nothing was scored")?
        .iter()
        .any(|l| match l {
            Line::Result {
                dataset,
                seed: s,
                config_hash,
                checkpoint_sha256,
                receipt_sha256,
                bottom_ranks_sha256,
                random_ranks_sha256,
                ..
            } => {
                *dataset == ds.name
                    && *s == seed
                    && config_hash == want_cfg
                    && checkpoint_sha256 == want_w
                    && receipt_sha256 == want_r
                    && want["bottom_sha256"] == json!(bottom_ranks_sha256)
                    && want["random_sha256"] == json!(random_ranks_sha256)
            }
            Line::Intent { .. } => false,
        });
    if !ledgered {
        bail!(
            "refusing --verify-final: origin's {} has no `result` line for receipt {want_r} ({}, seed {seed}, checkpoint {want_w}); nothing was scored",
            ledger::LEDGER_BRANCH,
            ds.name
        );
    }
    let filter = ds.filter_store(true)?;
    let ranks = with_threads(threads, |_| rank_split(&tables, &filter, &ds.test))?;
    let checks = json!({
        "weights_sha256": true,
        "ledger_result": true,
        "bottom_ranks_sha256": want["bottom_sha256"] == json!(ranks.bottom_sha256),
        "random_ranks_sha256": want["random_sha256"] == json!(ranks.random_sha256),
    });
    drop(ranks);
    let pass = checks
        .as_object()
        .unwrap()
        .values()
        .all(|v| v == &json!(true));
    seal(json!({
        "mode": "verification",
        "generated_at": utc_now(),
        "provenance": provenance(repo, threads.unwrap_or(0)),
        "dataset": final_receipt["dataset"],
        "config": final_receipt["config"],
        "seed": final_receipt["seed"],
        "eval": eval_block(),
        "tables": { "weights_sha256": manifest.weights_sha256 },
        "verifies_receipt_sha256": want_r,
        "ledger_head": origin.head,
        "checks": checks,
        "pass": pass,
    }))
}
