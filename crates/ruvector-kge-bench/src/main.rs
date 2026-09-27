//! CLI for `ruvector-kge-bench`. Modes (exactly one):
//!
//! ```text
//! ruvector-kge-bench --config run.json [--out DIR] [--resume] [--stop-after-epoch N]
//! ruvector-kge-bench --final --run DIR [--out FILE]
//! ruvector-kge-bench --verify-final FINAL_RECEIPT [--tables DIR] [--out FILE]
//! ruvector-kge-bench --check-dataset NAME
//! ruvector-kge-bench --validate-receipt RECEIPT
//! ```
//!
//! The default (`--config`) mode trains and scores **valid only**. Test is
//! read only by `--final`, which is gated on origin's tag and ledger.

use anyhow::{bail, Context, Result};
use clap::Parser;
use ruvector_kge_bench::config::RunConfig;
use ruvector_kge_bench::datasets::{self, default_cache_dir};
use ruvector_kge_bench::gitops::GitCli;
use ruvector_kge_bench::receipt::validate;
use ruvector_kge_bench::runner::{self, RunOptions};
use ruvector_kge_bench::scoring::{final_score, load_run_dir, verify_final};
use serde_json::Value;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "ruvector-kge-bench",
    about = "ADR-007 KGE bench: valid-only training, --final, --verify-final"
)]
struct Args {
    /// Train + validate from a JSON run config (dataset, config_id, seed, max_epochs, ...).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Override the config's seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Override the config's max_epochs.
    #[arg(long)]
    max_epochs: Option<usize>,
    /// Override the config's thread count.
    #[arg(long)]
    threads: Option<usize>,
    /// Output directory (train) or receipt file (final / verify-final).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Resume from OUT/checkpoint.bin.
    #[arg(long)]
    resume: bool,
    /// Stop (resumably) after N completed epochs.
    #[arg(long)]
    stop_after_epoch: Option<usize>,
    /// One-time test scoring of a finished run (gated on origin's tag + ledger).
    #[arg(long = "final")]
    final_: bool,
    /// Run directory for --final.
    #[arg(long)]
    run: Option<PathBuf>,
    /// Recompute test ranks from exported tables and compare with a final receipt.
    #[arg(long)]
    verify_final: Option<PathBuf>,
    /// Exported tables directory for --verify-final (default: <receipt dir>/best).
    #[arg(long)]
    tables: Option<PathBuf>,
    /// Load, verify and summarise a dataset.
    #[arg(long)]
    check_dataset: Option<String>,
    /// Validate a receipt against receipt@1.
    #[arg(long)]
    validate_receipt: Option<PathBuf>,
    /// Dataset cache (default: $RVKGE_CACHE_DIR or the JS harness cache).
    #[arg(long)]
    cache_dir: Option<PathBuf>,
    /// Repository for git provenance and the --final gate (default: cwd).
    #[arg(long, default_value = ".")]
    repo: PathBuf,
    /// Quiet: no per-epoch progress on stderr.
    #[arg(long)]
    quiet: bool,
}

fn read_json(p: &PathBuf) -> Result<Value> {
    serde_json::from_slice(&std::fs::read(p).with_context(|| format!("read {}", p.display()))?)
        .with_context(|| format!("parse {}", p.display()))
}

fn main() -> Result<()> {
    let a = Args::parse();
    let modes = [
        a.config.is_some(),
        a.final_,
        a.verify_final.is_some(),
        a.check_dataset.is_some(),
        a.validate_receipt.is_some(),
    ];
    if modes.iter().filter(|&&m| m).count() != 1 {
        bail!("choose exactly one of --config, --final, --verify-final, --check-dataset, --validate-receipt (see --help)");
    }
    let cache = a.cache_dir.clone().unwrap_or_else(default_cache_dir);

    if let Some(name) = &a.check_dataset {
        let ds = datasets::load(name, &cache)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&runner::dataset_block(&ds))?
        );
        return Ok(());
    }
    if let Some(p) = &a.validate_receipt {
        let mode = validate(&read_json(p)?)?;
        println!("receipt@1 OK (mode {mode})");
        return Ok(());
    }
    if let Some(cfg_path) = &a.config {
        let mut cfg = RunConfig::from_json(&std::fs::read_to_string(cfg_path)?)?;
        if let Some(s) = a.seed {
            cfg.seed = s;
        }
        if let Some(e) = a.max_epochs {
            cfg.max_epochs = e;
        }
        if let Some(t) = a.threads {
            cfg.threads = Some(t);
        }
        cfg.validate()?;
        let ds = datasets::load(&cfg.dataset, &cache)?;
        let out = a.out.clone().unwrap_or_else(|| runner::default_out(&cfg));
        let opts = RunOptions {
            out: out.clone(),
            resume: a.resume,
            stop_after_epoch: a.stop_after_epoch,
            repo: a.repo.clone(),
            verbose: !a.quiet,
        };
        let r = runner::run(&cfg, &ds, &opts)?;
        eprintln!(
            "receipt: {} (stopped: {}, best valid MRR bottom: {})",
            out.join("receipt.json").display(),
            r["stopped"],
            r["best"]["valid_mrr_bottom"]
        );
        return Ok(());
    }
    if a.final_ {
        let run = a.run.clone().context("--final needs --run <dir>")?;
        let rec = read_json(&run.join("receipt.json"))?;
        let name = rec["dataset"]["name"]
            .as_str()
            .context("run receipt has no dataset name")?;
        let ds = datasets::load(name, &cache)?;
        let rd = load_run_dir(&run, &ds, true)?;
        let out = a
            .out
            .clone()
            .unwrap_or_else(|| run.join("final-receipt.json"));
        let r = final_score(&GitCli::new(&a.repo), &rd, &ds, &out, &a.repo)?;
        eprintln!(
            "final receipt: {} (test MRR bottom {})",
            out.display(),
            r["test"]["metrics"]["bottom"]["combined"]["mrr"]
        );
        return Ok(());
    }
    let rp = a.verify_final.clone().expect("mode checked");
    let rec = read_json(&rp)?;
    let name = rec["dataset"]["name"]
        .as_str()
        .context("receipt has no dataset name")?;
    let ds = datasets::load(name, &cache)?;
    let tables = a.tables.clone().unwrap_or_else(|| {
        rp.parent()
            .unwrap_or(std::path::Path::new("."))
            .join("best")
    });
    let v = verify_final(
        &GitCli::new(&a.repo),
        &rec,
        &ds,
        &tables,
        a.threads,
        &a.repo,
    )?;
    let text = serde_json::to_string_pretty(&v)?;
    match &a.out {
        Some(p) => ruvector_kge_bench::canon::atomic_write(p, text.as_bytes())?,
        None => println!("{text}"),
    }
    // `verify_final` refuses (Err, nothing scored) unless the tables and the
    // receipt are on origin's ledger; `v` holds only hash checks + pass.
    if v["pass"] != Value::Bool(true) {
        bail!("verification FAILED: {}", v["checks"]);
    }
    eprintln!(
        "verification PASS (recorded as `verification`, not `final`; no ledger line written)"
    );
    Ok(())
}
