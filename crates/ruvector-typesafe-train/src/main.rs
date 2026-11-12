//! `openjev` CLI (plan Step 2): fetch | prep | prep-check | train | transplant | parity.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand};
use ruvector_typesafe_train::config::Config;
use ruvector_typesafe_train::leakage::{LeakageError, LEAKAGE_EXIT};
use ruvector_typesafe_train::{device, export, parity, pins, prep, train};

#[derive(Parser)]
#[command(name = "openjev", about = "OpenJev v0 trainer (ADR-008)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// Where pinned inputs live. Defaults match a repo checkout.
#[derive(Args, Clone)]
struct Inputs {
    /// Cache of pinned downloads (datasets, base safetensors, template ONNX, tokenizer).
    #[arg(long, default_value = "runs/cache")]
    cache: PathBuf,
    /// Frozen tickets fixture.
    #[arg(
        long,
        default_value = "npm/packages/typesafe/bench/fixtures/tickets-decisions.json"
    )]
    fixture: PathBuf,
    /// Download missing pinned inputs (sha256-verified).
    #[arg(long)]
    download: bool,
}

impl Inputs {
    fn sources(&self) -> prep::Sources {
        prep::Sources {
            tickets_fixture: self.fixture.clone(),
            cache: self.cache.clone(),
            allow_download: self.download,
            datasets: Vec::new(),
        }
    }
    fn model(&self, pin: &pins::Pin) -> Result<PathBuf> {
        pins::ensure(&self.cache, pin, self.download)
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Download + verify every pinned input into --cache.
    Fetch {
        #[command(flatten)]
        inputs: Inputs,
    },
    /// Build the data dir (train.jsonl, val.jsonl, labels.json, heldout-hashes.txt, …).
    Prep {
        #[command(flatten)]
        inputs: Inputs,
        #[arg(long, default_value = "runs/data")]
        out: PathBuf,
    },
    /// Pins + Assertion A + token lengths, without training. Exit 3 on leakage.
    PrepCheck {
        #[command(flatten)]
        common: TrainCommon,
    },
    /// Fine-tune; keeps the best-validation checkpoint in --out.
    Train {
        #[command(flatten)]
        common: TrainCommon,
        #[arg(long)]
        out: PathBuf,
    },
    /// Write fine-tuned (or, with --identity, base) weights into the pinned ONNX.
    Transplant {
        #[command(flatten)]
        inputs: Inputs,
        #[arg(long, required_unless_present = "identity")]
        weights: Option<PathBuf>,
        #[arg(long)]
        identity: bool,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Stage a run as a bench model dir (manifest.json + NAME/{model.onnx,tokenizer.json,train-text-hashes.txt}).
    Stage {
        #[command(flatten)]
        inputs: Inputs,
        #[arg(long)]
        onnx: PathBuf,
        #[arg(long)]
        train_hashes: PathBuf,
        #[arg(long, default_value = "openjev-small-v0")]
        name: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// ort(ONNX) vs candle(safetensors): cosine per probe + engine decisions.
    Parity {
        #[command(flatten)]
        inputs: Inputs,
        #[arg(long)]
        onnx: PathBuf,
        /// Defaults to the pinned base safetensors.
        #[arg(long)]
        weights: Option<PathBuf>,
        #[arg(long, default_value = "runs/data")]
        data: PathBuf,
        #[arg(long, default_value_t = 512)]
        probes: usize,
        #[arg(long, default_value_t = 0.9999)]
        threshold: f32,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[derive(Args, Clone)]
struct TrainCommon {
    #[command(flatten)]
    inputs: Inputs,
    #[arg(
        long,
        default_value = "crates/ruvector-typesafe-train/configs/openjev-small-v0.toml"
    )]
    config: PathBuf,
    #[arg(long, default_value = "runs/data")]
    data: PathBuf,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Overrides for smoke runs.
    #[arg(long)]
    max_steps: Option<usize>,
    #[arg(long)]
    eval_every: Option<usize>,
    #[arg(long)]
    warmup_steps: Option<usize>,
    #[arg(long)]
    val_limit: Option<usize>,
    #[arg(long)]
    encoder_lr: Option<f64>,
    #[arg(long)]
    mix_tickets: Option<f64>,
    /// Train on tickets only (plan sweep ablation).
    #[arg(long)]
    tickets_only: bool,
}

impl TrainCommon {
    fn args(&self, out: &Path) -> Result<train::TrainArgs> {
        let mut c = Config::load(&self.config)?;
        if let Some(v) = self.max_steps {
            c.train.max_steps = v;
        }
        if let Some(v) = self.eval_every {
            c.train.eval_every = v;
        }
        if let Some(v) = self.warmup_steps {
            c.train.warmup_steps = v;
        }
        if let Some(v) = self.val_limit {
            c.train.val_limit = v;
        }
        if let Some(v) = self.encoder_lr {
            c.train.encoder_lr = v;
        }
        if let Some(v) = self.mix_tickets {
            c.mix.insert("tickets".into(), v);
        }
        if self.tickets_only {
            for d in ["banking77", "clinc150", "hwu64"] {
                c.mix.insert(d.into(), 0.0);
            }
        }
        c.validate()?;
        Ok(train::TrainArgs {
            config_sha256: train::config_sha(&self.config)?,
            config: c,
            data: self.data.clone(),
            sources: self.inputs.sources(),
            base: self.inputs.model(&pins::BASE_SAFETENSORS)?,
            tokenizer: self.inputs.model(&pins::TOKENIZER)?,
            seed: self.seed,
            device: device(&self.device)?,
            device_name: self.device.clone(),
            out: out.to_path_buf(),
        })
    }
}

fn print_json<T: serde::Serialize>(v: &T) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

fn run(cli: Cli) -> Result<()> {
    match cli.cmd {
        Cmd::Fetch { inputs } => {
            for p in pins::MODEL_PINS.iter().chain(pins::DATASET_PINS.iter()) {
                let path = pins::ensure(&inputs.cache, p, inputs.download)?;
                println!("ok {} {}", p.sha256, path.display());
            }
        }
        Cmd::Prep { inputs, out } => {
            let o = prep::run(&inputs.sources(), &out)?;
            print_json(&o.report)?;
        }
        Cmd::PrepCheck { common } => {
            let a = common.args(Path::new("runs/prep-check"))?;
            let pc = train::prep_check(&a)?;
            print_json(
                &serde_json::json!({"leakage": pc.leakage, "token_p50": pc.token_p50, "token_p99": pc.token_p99}),
            )?;
        }
        Cmd::Train { common, out } => {
            let a = common.args(&out)?;
            let rec = train::run(&a)?;
            print_json(&rec)?;
        }
        Cmd::Transplant {
            inputs,
            weights,
            identity,
            out,
            report,
        } => {
            if identity && weights.is_some() {
                bail!("--identity and --weights are mutually exclusive");
            }
            let template = inputs.model(&pins::TEMPLATE_ONNX)?;
            let base = inputs.model(&pins::BASE_SAFETENSORS)?;
            let rep = export::run(&template, &base, weights.as_deref(), &out)?;
            if let Some(r) = report {
                std::fs::write(r, serde_json::to_vec_pretty(&rep)?)?;
            }
            eprintln!(
                "[transplant] {} float initializers mapped ({} named, {} transposed), {} non-float skipped, identical={} sha256={}",
                rep.float_initializers, rep.named, rep.anonymous_transposed, rep.skipped_non_float.len(),
                rep.byte_identical_to_template, rep.output_sha256
            );
        }
        Cmd::Stage {
            inputs,
            onnx,
            train_hashes,
            name,
            out,
        } => {
            let tok = inputs.model(&pins::TOKENIZER)?;
            print_json(&export::stage(&onnx, &tok, &train_hashes, &name, &out)?)?;
        }
        Cmd::Parity {
            inputs,
            onnx,
            weights,
            data,
            probes,
            threshold,
            device: d,
            out,
        } => {
            let weights = match weights {
                Some(w) => w,
                None => inputs.model(&pins::BASE_SAFETENSORS)?,
            };
            let args = parity::ParityArgs {
                onnx,
                weights,
                tokenizer: inputs.model(&pins::TOKENIZER)?,
                data: data.exists().then_some(data),
                probes,
                threshold,
                max_tokens: 256,
            };
            let rep = parity::run(&args, &device(&d)?)?;
            if let Some(o) = out {
                parity::write_report(&o, &rep)?;
            }
            print_json(&rep)?;
            if !rep.pass {
                bail!(
                    "parity FAILED: {} probes below {threshold}, decisions {}/{}",
                    rep.below_threshold,
                    rep.decisions_identical,
                    rep.decisions_compared
                );
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            if e.downcast_ref::<LeakageError>().is_some() {
                return ExitCode::from(LEAKAGE_EXIT as u8);
            }
            ExitCode::FAILURE
        }
    }
}
