//! Run configuration and the pre-registered C1–C8 grid (ADR-007 §3).
//!
//! A run is `{dataset, config_id, seed, max_epochs, early_stop_patience, …}`.
//! `config_id` is one of the dataset's `C1`..`C8` (resolved from the ADR table
//! below) or `custom` with explicit hyperparameters (smoke tests; never
//! selectable for `--final`, see `final_mode`).
//!
//! **`config_hash`** identifies the *recipe*: sha256 of the canonical JSON of
//! `{schema, dataset, config_id, recipe}`. It deliberately excludes the seed,
//! `max_epochs`, early stopping, the eval cadence and the thread count, so the
//! HPO run (seed 100, ≤ 100 epochs, early stop) and the five final runs
//! (seeds 0–4, 500 epochs, no early stop) of one config share one hash — the
//! key `selection.json` and the ledger bind.

use crate::canon::canonical_hash;
use anyhow::{bail, Context, Result};
use ruvector_kge::train::{Init, LossKind, N3Form, OneNKernel, OptimKind, Reduction, StateLayout};
use ruvector_kge::TrainConfig;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Schema tag folded into every config hash.
pub const CONFIG_SCHEMA: &str = "ruvector-kge-bench/config@1";
/// RANDOM tie-break base seed for every evaluation (per-query seeds derive
/// from it and the triple). Fixed by protocol, recorded in receipts.
pub const EVAL_SEED: u64 = 0;
/// torch `Adagrad` default epsilon (kbc / ssl-RP use torch's defaults).
pub const ADAGRAD_EPS: f32 = 1e-10;
/// kbc `init_size`.
pub const INIT_SCALE: f32 = 1e-3;

/// The recipe hyperparameters that vary across C1–C8 (ADR-007 §3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
    /// Complex coordinates per embedding (`dims = 2 · complex_rank`).
    pub complex_rank: usize,
    pub batch_size: usize,
    pub lr: f32,
    /// N3 weight λ.
    pub n3_lambda: f32,
    /// Relation-prediction weight `w_rel` (0 = RP off).
    pub rp_weight: f32,
}

/// A run as given on the command line / in the JSON config file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub dataset: String,
    /// `C1`..`C8`, or `custom` (then `recipe` is required).
    pub config_id: String,
    /// Only for `config_id = "custom"`.
    #[serde(default)]
    pub recipe: Option<Recipe>,
    pub seed: u64,
    pub max_epochs: usize,
    /// Stop after this many evaluations without a strictly better valid MRR
    /// (Bottom). `None` = no early stop (final runs: best-on-valid only).
    #[serde(default)]
    pub early_stop_patience: Option<usize>,
    /// Evaluate valid every N epochs (the last epoch is always evaluated).
    #[serde(default = "one")]
    pub eval_every: usize,
    /// Worker threads (rayon pool). `None` = rayon's default.
    #[serde(default)]
    pub threads: Option<usize>,
}

fn one() -> usize {
    1
}

/// The ADR-007 §3 table: C1 is ssl-RP's per-dataset "Best Run"; C2–C8 change
/// one or two axes of **that dataset's** C1. All use complex rank 1000, lr 0.1.
pub fn grid(dataset: &str, id: &str) -> Result<Recipe> {
    let c1 = match dataset {
        "fb15k237" => Recipe {
            complex_rank: 1000,
            batch_size: 1000,
            lr: 0.1,
            n3_lambda: 0.05,
            rp_weight: 4.0,
        },
        "wn18rr" => Recipe {
            complex_rank: 1000,
            batch_size: 100,
            lr: 0.1,
            n3_lambda: 0.10,
            rp_weight: 0.05,
        },
        "codexm" => Recipe {
            complex_rank: 1000,
            batch_size: 500,
            lr: 0.1,
            n3_lambda: 0.01,
            rp_weight: 0.125,
        },
        other => {
            bail!("dataset '{other}' has no pre-registered C1–C8 grid (yago310 is informational)")
        }
    };
    // (C2 w_rel, C3 λ, C4 λ, C5 w_rel, C6 batch, C7 (batch, λ))
    let (c3, c4, c5, c6, c7) = match dataset {
        "fb15k237" => (0.01, 0.1, 1.0, 100, (100, 0.1)),
        "wn18rr" => (0.05, 0.5, 0.5, 1000, (500, 0.05)),
        _ => (0.005, 0.05, 0.0625, 1000, (100, 0.005)),
    };
    Ok(match id {
        "C1" => c1,
        "C2" => Recipe {
            rp_weight: 0.0,
            ..c1
        },
        "C3" => Recipe {
            n3_lambda: c3,
            ..c1
        },
        "C4" => Recipe {
            n3_lambda: c4,
            ..c1
        },
        "C5" => Recipe {
            rp_weight: c5,
            ..c1
        },
        "C6" => Recipe {
            batch_size: c6,
            ..c1
        },
        "C7" => Recipe {
            batch_size: c7.0,
            n3_lambda: c7.1,
            ..c1
        },
        "C8" => Recipe {
            complex_rank: 2000,
            ..c1
        },
        other => bail!("unknown config id '{other}' (C1..C8 or custom)"),
    })
}

impl RunConfig {
    /// Parse a JSON config file's contents.
    pub fn from_json(text: &str) -> Result<Self> {
        let c: Self = serde_json::from_str(text).context("parse run config JSON")?;
        c.validate()?;
        Ok(c)
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_epochs == 0 {
            bail!("max_epochs must be >= 1");
        }
        if self.eval_every == 0 {
            bail!("eval_every must be >= 1");
        }
        if self.early_stop_patience == Some(0) {
            bail!("early_stop_patience must be >= 1 (omit it for no early stop)");
        }
        if self.threads == Some(0) {
            bail!("threads must be >= 1");
        }
        let r = self.recipe()?;
        if r.complex_rank == 0 || r.batch_size == 0 {
            bail!("complex_rank and batch_size must be >= 1");
        }
        if !(r.lr.is_finite() && r.lr > 0.0)
            || !(r.n3_lambda.is_finite() && r.n3_lambda >= 0.0)
            || !(r.rp_weight.is_finite() && r.rp_weight >= 0.0)
        {
            bail!("lr must be > 0; n3_lambda and rp_weight must be finite and >= 0");
        }
        Ok(())
    }

    /// Is `config_id` one of the pre-registered grid entries?
    pub fn is_grid(&self) -> bool {
        self.config_id != "custom"
    }

    /// The resolved recipe (grid lookup, or the custom block).
    pub fn recipe(&self) -> Result<Recipe> {
        match (self.config_id.as_str(), &self.recipe) {
            ("custom", Some(r)) => Ok(*r),
            ("custom", None) => bail!("config_id \"custom\" needs a \"recipe\" block"),
            (_, Some(_)) => bail!(
                "\"recipe\" is only allowed with config_id \"custom\" (grid configs are frozen)"
            ),
            (id, None) => grid(&self.dataset, id),
        }
    }

    /// The core `TrainConfig` for this run (ComplEx-N3-R, ADR-007 §3).
    pub fn train_config(&self) -> Result<TrainConfig> {
        let r = self.recipe()?;
        Ok(TrainConfig {
            dims: 2 * r.complex_rank,
            epochs: self.max_epochs,
            batch_size: r.batch_size,
            lr: r.lr,
            optimizer: OptimKind::Adagrad {
                epsilon: ADAGRAD_EPS,
            },
            loss: LossKind::OneVsAll,
            n3_lambda: r.n3_lambda,
            seed: self.seed,
            reciprocal: true,
            init: Init::Normal { scale: INIT_SCALE },
            n3_form: N3Form::Moduli,
            loss_reduction: Reduction::Mean,
            rp_weight: r.rp_weight,
            optim_state: StateLayout::Dense,
            one_n_kernel: OneNKernel::Gemm,
        })
    }

    /// The canonical recipe object hashed into `config_hash`: every
    /// `TrainConfig` field except `seed` and `epochs`.
    pub fn recipe_canonical(&self) -> Result<serde_json::Value> {
        let mut v = serde_json::to_value(self.train_config()?)?;
        let m = v.as_object_mut().context("TrainConfig is an object")?;
        m.remove("seed");
        m.remove("epochs");
        m.insert("scorer".into(), json!("complex"));
        m.insert("complex_rank".into(), json!(self.recipe()?.complex_rank));
        Ok(v)
    }

    /// `config_hash` (see the module docs).
    pub fn config_hash(&self) -> Result<String> {
        Ok(canonical_hash(&json!({
            "schema": CONFIG_SCHEMA,
            "dataset": self.dataset,
            "config_id": self.config_id,
            "recipe": self.recipe_canonical()?,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(ds: &str, id: &str) -> RunConfig {
        RunConfig {
            dataset: ds.into(),
            config_id: id.into(),
            recipe: None,
            seed: 100,
            max_epochs: 100,
            early_stop_patience: Some(5),
            eval_every: 1,
            threads: Some(8),
        }
    }

    #[test]
    fn c1_matches_adr_table() {
        let fb = grid("fb15k237", "C1").unwrap();
        assert_eq!(
            (fb.batch_size, fb.n3_lambda, fb.rp_weight, fb.complex_rank),
            (1000, 0.05, 4.0, 1000)
        );
        let wn = grid("wn18rr", "C1").unwrap();
        assert_eq!(
            (wn.batch_size, wn.n3_lambda, wn.rp_weight),
            (100, 0.10, 0.05)
        );
        let cx = grid("codexm", "C7").unwrap();
        assert_eq!(
            (cx.batch_size, cx.n3_lambda, cx.rp_weight),
            (100, 0.005, 0.125)
        );
        assert_eq!(grid("wn18rr", "C8").unwrap().complex_rank, 2000);
        assert!(grid("yago310", "C1").is_err());
        assert!(grid("wn18rr", "C9").is_err());
    }

    #[test]
    fn config_hash_ignores_seed_epochs_threads_and_early_stop() {
        let hpo = run("wn18rr", "C1");
        let mut fin = hpo.clone();
        fin.seed = 3;
        fin.max_epochs = 500;
        fin.early_stop_patience = None;
        fin.threads = Some(32);
        fin.eval_every = 5;
        assert_eq!(hpo.config_hash().unwrap(), fin.config_hash().unwrap());
        // …but every recipe axis and the dataset change it.
        let hashes: std::collections::BTreeSet<String> = (1..=8)
            .map(|i| run("wn18rr", &format!("C{i}")).config_hash().unwrap())
            .chain([run("fb15k237", "C1").config_hash().unwrap()])
            .collect();
        assert_eq!(hashes.len(), 9);
    }

    #[test]
    fn custom_requires_recipe_and_grid_forbids_it() {
        let mut c = run("wn18rr", "custom");
        assert!(c.validate().is_err());
        c.recipe = Some(Recipe {
            complex_rank: 8,
            batch_size: 4,
            lr: 0.1,
            n3_lambda: 0.1,
            rp_weight: 0.0,
        });
        c.validate().unwrap();
        let mut g = run("wn18rr", "C1");
        g.recipe = c.recipe;
        assert!(g.validate().is_err());
        assert!(RunConfig::from_json(
            r#"{"dataset":"wn18rr","config_id":"C1","seed":1,"max_epochs":1,"bogus":1}"#
        )
        .is_err());
        let p = RunConfig::from_json(
            r#"{"dataset":"wn18rr","config_id":"C2","seed":1,"max_epochs":3}"#,
        )
        .unwrap();
        assert_eq!(p.train_config().unwrap().rp_weight, 0.0);
        assert!(p.train_config().unwrap().reciprocal);
    }
}
