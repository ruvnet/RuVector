//! The default (valid-only) mode: train ComplEx-N3-R epoch by epoch, evaluate
//! **valid** (filtered by train ∪ valid ∪ transfer; Bottom + RANDOM), keep the
//! best-on-valid tables, early-stop when configured, checkpoint every epoch,
//! and write a receipt. Test triples are never scored here.
//!
//! Resume (`--resume`) reloads the checkpoint and continues bit-identically:
//! the core `TrainSession` carries the permutation, RNG and optimizer rows,
//! and evaluation is deterministic, so per-epoch metrics and final tables
//! equal an uninterrupted run's (asserted in `tests/resume.rs`).

use crate::canon::{atomic_write, utc_now};
use crate::checkpoint::{self, Header};
use crate::config::{RunConfig, EVAL_SEED};
use crate::datasets::Dataset;
use crate::export::{export_tables, ExportMeta, Manifest};
use crate::metrics::{rank_split, Best, EpochRecord, RunProgress, StopReason};
use crate::receipt::{provenance, seal};
use anyhow::{bail, Result};
use ruvector_kge::kernel::GemmOneToN;
use ruvector_kge::scorer::ComplEx;
use ruvector_kge::train::TrainSession;
use ruvector_kge::Tables;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Where and how a run executes.
#[derive(Debug, Clone)]
pub struct RunOptions {
    /// Output directory: checkpoint, `best/`, `last/`, `receipt.json`.
    pub out: PathBuf,
    /// Continue from `out/checkpoint.bin`.
    pub resume: bool,
    /// Stop (resumably) after this many completed epochs — an interruption
    /// drill for the resume-equivalence check.
    pub stop_after_epoch: Option<usize>,
    /// Repository whose git state is recorded.
    pub repo: PathBuf,
    /// Print one progress line per epoch to stderr.
    pub verbose: bool,
}

/// Run `f` on a rayon pool of `threads` (the kernel and evaluator use the
/// current pool). Returns the thread count actually used.
pub fn with_threads<T>(
    threads: Option<usize>,
    f: impl FnOnce(usize) -> Result<T> + Send,
) -> Result<T>
where
    T: Send,
{
    #[cfg(feature = "parallel")]
    {
        let mut b = rayon::ThreadPoolBuilder::new();
        if let Some(n) = threads {
            b = b.num_threads(n);
        }
        let pool = b.build()?;
        let n = pool.current_num_threads();
        pool.install(|| f(n))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = threads;
        f(1)
    }
}

fn dataset_json(ds: &Dataset) -> Value {
    json!({
        "name": ds.name, "licence": ds.licence, "sources": ds.sources, "file_hashes": ds.file_hashes,
        "splits_hash": ds.splits_hash, "split_hashes": ds.split_hashes,
        "entity_vocab_hash": ds.entity_vocab_hash, "relation_vocab_hash": ds.relation_vocab_hash,
        "counts": ds.counts(),
    })
}

/// The dataset block of a receipt (shared with `final_mode`).
pub fn dataset_block(ds: &Dataset) -> Value {
    dataset_json(ds)
}

/// The config block of a receipt.
pub fn config_block(cfg: &RunConfig) -> Result<Value> {
    Ok(json!({
        "config_id": cfg.config_id, "config_hash": cfg.config_hash()?, "canonical": cfg.recipe_canonical()?,
        "run": cfg, "train_config": cfg.train_config()?, "complex_rank": cfg.recipe()?.complex_rank,
    }))
}

fn header(cfg: &RunConfig, ds: &Dataset, threads: usize, progress: &RunProgress) -> Result<Header> {
    Ok(Header {
        config_hash: cfg.config_hash()?,
        run: cfg.clone(),
        dataset: ds.name.clone(),
        splits_hash: ds.splits_hash.clone(),
        entity_vocab_hash: ds.entity_vocab_hash.clone(),
        relation_vocab_hash: ds.relation_vocab_hash.clone(),
        threads,
        num_entities: 0,
        num_relation_rows: 0,
        dims: 0,
        next_epoch: 0,
        sample_rng: 0,
        optim_t: 0,
        order_len: 0,
        optim_rows: vec![],
        progress: progress.clone(),
    })
}

/// Train + validate per `cfg` on `ds`. Returns the sealed receipt (also
/// written to `out/receipt.json`).
pub fn run(cfg: &RunConfig, ds: &Dataset, opts: &RunOptions) -> Result<Value> {
    cfg.validate()?;
    if cfg.dataset != ds.name {
        bail!(
            "config names dataset '{}' but '{}' was loaded",
            cfg.dataset,
            ds.name
        );
    }
    with_threads(cfg.threads, |threads| run_on_pool(cfg, ds, opts, threads))
}

fn run_on_pool(cfg: &RunConfig, ds: &Dataset, opts: &RunOptions, threads: usize) -> Result<Value> {
    let wall = Instant::now();
    let tcfg = cfg.train_config()?;
    let config_hash = cfg.config_hash()?;
    let train = ds.train_store()?;
    let filter = ds.filter_store(false)?;
    let scorer = ComplEx::new(tcfg.dims)?;
    let kernel = GemmOneToN::new();
    let out = &opts.out;

    let (mut tables, mut session, mut progress) = if opts.resume {
        let ck = checkpoint::read(out)?;
        let h = &ck.header;
        if h.config_hash != config_hash || h.run != *cfg {
            bail!("checkpoint belongs to a different run config (hash {} vs {config_hash}); refusing to resume", h.config_hash);
        }
        if h.dataset != ds.name
            || h.splits_hash != ds.splits_hash
            || h.entity_vocab_hash != ds.entity_vocab_hash
            || h.relation_vocab_hash != ds.relation_vocab_hash
        {
            bail!("checkpoint belongs to a different dataset or vocabulary; refusing to resume");
        }
        if h.threads != threads {
            bail!("checkpoint was written with {} threads, this run has {threads}; pass --threads {} to resume", h.threads, h.threads);
        }
        let mut progress = h.progress.clone();
        progress.resumed_at.push(ck.state.next_epoch);
        let tables = ck.tables;
        let session = TrainSession::resume(&tables, &scorer, &train, &tcfg, &kernel, &ck.state)?;
        (tables, session, progress)
    } else {
        if out.join(checkpoint::CHECKPOINT_FILE).exists() {
            bail!(
                "{} already holds a checkpoint; pass --resume or use a new --out",
                out.display()
            );
        }
        let mut tables = Tables::try_new(
            ds.num_entities,
            2 * ds.num_relations,
            tcfg.dims,
            cfg.seed,
            None,
        )?;
        let session = TrainSession::new(&mut tables, &scorer, &train, &tcfg, &kernel)?;
        (tables, session, RunProgress::default())
    };

    let meta = |epoch| ExportMeta {
        dataset: ds,
        seed: cfg.seed,
        config_hash: &config_hash,
        epoch,
    };
    let mut stopped = if progress.early_stopped {
        StopReason::EarlyStop
    } else {
        StopReason::MaxEpochs
    };
    while !progress.early_stopped && session.next_epoch() < cfg.max_epochs {
        let epoch = session.next_epoch();
        let t0 = Instant::now();
        let p = session.run_epoch(&mut tables)?;
        let train_secs = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut rec = EpochRecord {
            epoch,
            loss: p.loss,
            n3_penalty: p.n3_penalty,
            rp_loss: p.rp_loss,
            train_secs,
            eval_secs: 0.0,
            valid: None,
        };
        if (epoch + 1) % cfg.eval_every == 0 || epoch + 1 == cfg.max_epochs {
            let ranks = rank_split(&tables, &filter, &ds.valid)?;
            let m = ranks.metrics();
            let mrr = m.bottom.combined.mrr;
            if progress
                .best
                .as_ref()
                .is_none_or(|b| mrr > b.valid_mrr_bottom)
            {
                let man = export_tables(&out.join("best"), &tables, &meta(epoch))?;
                progress.best = Some(Best {
                    epoch,
                    valid_mrr_bottom: mrr,
                    metrics: m,
                    valid_ranks: ranks,
                    weights_sha256: man.weights_sha256,
                });
                progress.evals_since_best = 0;
            } else {
                progress.evals_since_best += 1;
            }
            if cfg
                .early_stop_patience
                .is_some_and(|p| progress.evals_since_best >= p)
            {
                progress.early_stopped = true;
                stopped = StopReason::EarlyStop;
            }
            rec.valid = Some(m);
        }
        rec.eval_secs = t1.elapsed().as_secs_f64();
        if opts.verbose {
            let v = rec
                .valid
                .map(|m| {
                    format!(
                        " valid MRR bottom {:.4} random {:.4}",
                        m.bottom.combined.mrr, m.random.combined.mrr
                    )
                })
                .unwrap_or_default();
            eprintln!(
                "[{}] epoch {epoch} loss {:.5} ({:.1}s train, {:.1}s eval){v}",
                ds.name, rec.loss, rec.train_secs, rec.eval_secs
            );
        }
        progress.history.push(rec);
        checkpoint::write(
            out,
            header(cfg, ds, threads, &progress)?,
            &tables,
            &session.export_state(),
        )?;
        if opts.stop_after_epoch == Some(epoch + 1)
            && session.next_epoch() < cfg.max_epochs
            && !progress.early_stopped
        {
            stopped = StopReason::Interrupted;
            break;
        }
    }
    let last_epoch = session.next_epoch().saturating_sub(1);
    let last: Manifest = export_tables(&out.join("last"), &tables, &meta(last_epoch))?;
    let receipt = seal(json!({
        "mode": "valid",
        "generated_at": utc_now(),
        "provenance": provenance(&opts.repo, threads),
        "dataset": dataset_json(ds),
        "config": config_block(cfg)?,
        "seed": cfg.seed,
        "eval": {
            "split": "valid", "filter": "train+valid+transfer", "tie_breaks": ["bottom", "random"],
            "verdict_tie_break": "bottom", "selection_metric": "valid_mrr_bottom", "eval_seed": EVAL_SEED,
            "reciprocal_head_queries": true, "test_scored": false,
        },
        "epochs": progress.history,
        "best": progress.best,
        "stopped": stopped,
        "resumed_at": progress.resumed_at,
        "timings": {
            "invocation_wall_secs": wall.elapsed().as_secs_f64(),
            "train_secs_total": progress.history.iter().map(|e| e.train_secs).sum::<f64>(),
            "eval_secs_total": progress.history.iter().map(|e| e.eval_secs).sum::<f64>(),
        },
        "exports": { "last": last, "best_weights_sha256": progress.best.as_ref().map(|b| b.weights_sha256.clone()) },
        "cost_usd": 0.0,
    }))?;
    atomic_write(
        &out.join("receipt.json"),
        serde_json::to_string_pretty(&receipt)?.as_bytes(),
    )?;
    Ok(receipt)
}

/// Default output directory for a config.
pub fn default_out(cfg: &RunConfig) -> PathBuf {
    Path::new("target/kge-bench").join(format!("{}-{}-s{}", cfg.dataset, cfg.config_id, cfg.seed))
}
