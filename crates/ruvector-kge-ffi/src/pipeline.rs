//! Training, evaluation and ANN index build for [`KgeModel`]. Duplicated
//! VERBATIM in `ruvector-kge-ffi/src/pipeline.rs` and
//! `ruvector-kge-wasm/src/pipeline.rs`.
//!
//! `train` and `eval` run over the current triples through the landed HolE /
//! RotatE scorers (both `Differentiable`). `buildIndex` builds a
//! `DistanceMetric::DotProduct` HNSW that `predict` then uses. `optimize` is
//! still a stub — the core `Campaign` needs an `Evaluator` the binding does not
//! provide yet.

use crate::model::{err_json, kge_error_json, KgeModel, SplitLabel};
use ruvector_kge::scorer::{HolE, RotatE};
use ruvector_kge::{
    evaluate, AnnIndex, EvalConfig, ScorerKind, TieBreak, TrainConfig, Trainer, Triple, TripleStore,
};
use serde::Deserialize;

fn default_true() -> bool {
    true
}

#[derive(Deserialize)]
struct EvalCfgInput {
    #[serde(default)]
    split: Option<String>,
    #[serde(default = "default_true")]
    filtered: bool,
    #[serde(default)]
    seed: u64,
    #[serde(rename = "tieBreak", default)]
    tie_break: Option<String>,
}

impl KgeModel {
    /// Train the tables from the current triples. `configJson` is a
    /// [`TrainConfig`] (all fields optional); `dims` is forced to the model's.
    /// If any triple carries a split tag, training uses ONLY the `train` split
    /// (never valid/test/transfer). Trains in place (preserving the growth
    /// invariant), then drops any ANN index (the tables changed).
    pub fn train_json(&mut self, config_json: &str) -> String {
        let mut cfg: TrainConfig = match serde_json::from_str(config_json) {
            Ok(c) => c,
            Err(e) => return err_json("invalid", &format!("train config parse error: {e}")),
        };
        cfg.dims = self.config.dims;
        if self.triples.is_empty() {
            return err_json("invalid", "no triples to train on");
        }
        // With ingested split tags, train ONLY on the "train" split — never on
        // valid/test/transfer (no leakage, ADR-006). Untagged: train on all.
        let train_triples = if self.has_split_tags() {
            self.triples_with_split(SplitLabel::Train)
        } else {
            self.triples.clone()
        };
        if train_triples.is_empty() {
            return err_json("invalid", "no 'train'-labelled triples to train on");
        }
        let store = match TripleStore::new(train_triples) {
            Ok(s) => s,
            Err(e) => return kge_error_json(&e),
        };
        self.ensure_built();
        self.invalidate_index();
        let kind = self.config.scorer;
        let dims = self.config.dims;
        // (epoch, epochs, num_batches, loss, n3_penalty) of the last epoch.
        let mut last: Option<(usize, usize, usize, f32, f32)> = None;
        let fit_result = {
            let tables = self.tables.as_mut().unwrap();
            let mut record = |p: &ruvector_kge::Progress| {
                last = Some((p.epoch, p.epochs, p.num_batches, p.loss, p.n3_penalty));
            };
            match kind {
                ScorerKind::Hole => match HolE::new(dims) {
                    Ok(sc) => Trainer::fit(tables, &sc, &store, &cfg, &mut record),
                    Err(e) => Err(e),
                },
                ScorerKind::Rotate => match RotatE::new(dims) {
                    Ok(sc) => Trainer::fit(tables, &sc, &store, &cfg, &mut record),
                    Err(e) => Err(e),
                },
            }
        };
        if let Err(e) = fit_result {
            return kge_error_json(&e);
        }
        match last {
            Some((epoch, epochs, batches, loss, n3)) => serde_json::json!({
                "epoch": epoch,
                "epochs": epochs,
                "batches": batches,
                "loss": loss,
                "n3Penalty": n3,
                "triples": self.triples.len(),
                "entities": self.entities.len(),
                "relations": self.relations.len(),
            })
            .to_string(),
            None => serde_json::json!({
                "epoch": 0, "epochs": cfg.epochs, "loss": 0.0,
                "note": "no epochs run (epochs=0)",
            })
            .to_string(),
        }
    }

    /// Filtered ranking metrics: `{"split":"train|valid|test|transfer",
    /// "filtered":true,"seed":0,"tieBreak":"random|top|bottom"}`. If triples
    /// carry ingested split tags the requested split is honoured VERBATIM (the
    /// caller's frozen split, e.g. the bench harness's — ADR-006). Otherwise a
    /// `split` is a *derived* 80/10/10 partition, a smoke check only. Filtering
    /// is always against the full triple set.
    pub fn eval_json(&mut self, config_json: &str) -> String {
        let ec: EvalCfgInput = match serde_json::from_str(config_json) {
            Ok(c) => c,
            Err(e) => return err_json("invalid", &format!("eval config parse error: {e}")),
        };
        if self.triples.is_empty() {
            return err_json("invalid", "no triples to evaluate");
        }
        let tie = match ec.tie_break.as_deref() {
            None | Some("random") => TieBreak::Random,
            Some("top") => TieBreak::Top,
            Some("bottom") => TieBreak::Bottom,
            Some(_) => return err_json("invalid", "tieBreak must be random|top|bottom"),
        };
        let store = match TripleStore::new(self.triples.clone()) {
            Ok(s) => s,
            Err(e) => return kge_error_json(&e),
        };
        let has_tags = self.has_split_tags();
        // note is the report's split provenance: the frozen per-triple split
        // when tags are present, else the derived-split disclaimer.
        let (eval_triples, split_source, note): (Vec<Triple>, &str, &str) = match ec
            .split
            .as_deref()
        {
            None => (
                self.triples.clone(),
                "all triples",
                "all known triples (no split requested)",
            ),
            Some(name) => {
                let label = match SplitLabel::from_name(name) {
                    Some(l) => l,
                    None => return err_json("invalid", "split must be train|valid|test|transfer"),
                };
                if has_tags {
                    (
                        self.triples_with_split(label),
                        "per-triple",
                        "frozen per-triple split",
                    )
                } else {
                    match name {
                        "train" | "valid" | "test" => match store.split(ec.seed, [0.8, 0.1, 0.1]) {
                            Ok(sp) => {
                                let picked = match name {
                                    "train" => sp.train,
                                    "valid" => sp.valid,
                                    _ => sp.test,
                                };
                                (
                                            picked,
                                            "derived",
                                            "derived 80/10/10 split, not the ADR-006 frozen content-hashed split",
                                        )
                            }
                            Err(e) => return kge_error_json(&e),
                        },
                        _ => {
                            return err_json(
                                "invalid",
                                "split \"transfer\" requires ingested split tags",
                            )
                        }
                    }
                }
            }
        };
        if eval_triples.is_empty() {
            return err_json("invalid", "the requested split is empty");
        }
        let cfg = EvalConfig {
            tie_break: tie,
            filtered: ec.filtered,
            seed: ec.seed,
        };
        self.ensure_built();
        let kind = self.config.scorer;
        let dims = self.config.dims;
        let tables = self.tables.as_ref().unwrap();
        let report = match kind {
            ScorerKind::Hole => match HolE::new(dims) {
                Ok(sc) => evaluate(tables, &sc, &store, &eval_triples, &cfg),
                Err(e) => return kge_error_json(&e),
            },
            ScorerKind::Rotate => match RotatE::new(dims) {
                Ok(sc) => evaluate(tables, &sc, &store, &eval_triples, &cfg),
                Err(e) => return kge_error_json(&e),
            },
        };
        match report {
            Ok(r) => serde_json::json!({
                "report": r,
                "evalTriples": eval_triples.len(),
                "split": ec.split,
                "splitSource": split_source,
                "filtered": cfg.filtered,
                "note": note,
            })
            .to_string(),
            Err(e) => kge_error_json(&e),
        }
    }

    /// Build the `DistanceMetric::DotProduct` HNSW over the entity table so
    /// `predict` retrieves candidates by ANN instead of scanning all entities.
    /// The index is transient: it is not saved and is dropped on any table
    /// change, so re-run after `train` or `addTriples`.
    pub fn build_index_json(&mut self) -> String {
        if self.entities.len() == 0 {
            return err_json("invalid", "no entities to index");
        }
        self.ensure_built();
        let kind = self.config.scorer;
        let dims = self.config.dims;
        let build_result: ruvector_kge::Result<AnnIndex> = {
            let tables = self.tables.as_ref().unwrap();
            match kind {
                ScorerKind::Hole => HolE::new(dims).and_then(|sc| AnnIndex::build(tables, &sc)),
                ScorerKind::Rotate => RotatE::new(dims).and_then(|sc| AnnIndex::build(tables, &sc)),
            }
        };
        match build_result {
            Ok(idx) => {
                let entities = self.entities.len();
                self.set_index(idx);
                serde_json::json!({ "indexed": true, "entities": entities }).to_string()
            }
            Err(e) => kge_error_json(&e),
        }
    }

    /// Self-optimization campaign. Still a stub: the core `Campaign` needs an
    /// `Evaluator` implementation the binding does not provide yet (ADR-004).
    pub fn optimize_json(&mut self, _campaign_json: &str) -> String {
        err_json(
            "unavailable",
            "optimize is unavailable: the core Campaign needs an Evaluator implementation the binding does not provide yet",
        )
    }
}
