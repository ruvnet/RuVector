//! Training, evaluation and ANN index build for [`KgeModel`]. Duplicated
//! VERBATIM in `ruvector-kge-ffi/src/pipeline.rs` and
//! `ruvector-kge-wasm/src/pipeline.rs`.
//!
//! `train` and `eval` run over the current triples through the landed HolE /
//! RotatE scorers (both `Differentiable`). `buildIndex` builds a
//! `DistanceMetric::DotProduct` HNSW that `predict` then uses. The
//! self-optimization campaign lives in the sibling `optimize` module.

use crate::model::{err_json, kge_error_json, KgeModel, SplitLabel};
use ruvector_kge::scorer::{HolE, RotatE};
use ruvector_kge::{
    effective_max_bytes, evaluate, AnnIndex, EvalConfig, ScorerKind, Tables, TieBreak, TrainConfig,
    Trainer, Triple, TripleStore,
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

/// `(epoch, epochs, num_batches, loss, n3_penalty)` of a job's last epoch.
type LastEpoch = (usize, usize, usize, f32, f32);

/// A training run (F2). Two modes:
///
/// - **Snapshot** (`begin_train`, used by the native async/sync wrappers):
///   the tables, the training triples and the config are cloned under a brief
///   lock; [`TrainJob::run`] trains the copy with NO model lock held, so
///   `predict` & co. keep answering from the pre-training tables;
///   `finish_train` swaps the trained rows back in under a brief lock.
/// - **In place** (`train_json`, the wasm path, and the native fallback when
///   a copy would not fit): no copy is made; [`TrainJob::run_in_place`] trains
///   the live tables directly and the caller holds the model (lock) for the
///   whole run, exactly as before F2.
///
/// Memory (F1 invariant): a snapshot doubles the table working set. It is taken
/// only when `2 x table bytes <= effective_max_bytes(maxTableBytes)` (see
/// `ruvector_kge::MAX_TABLE_BYTES`), so snapshot + live tables together stay
/// within the single-model cap; bigger models fall back to in-place training
/// (predict then blocks for the run, as before F2). The one excess is
/// `addTriples` growth *during* a snapshot run: `grow_tables` briefly holds
/// the old and the grown live table next to the snapshot, a transient peak of
/// up to `cap/2 + cap` beyond the snapshot's own `cap/2`. wasm is single
/// threaded and always trains in place, so it never pays for a copy.
///
/// Concurrency contract (snapshot mode; decision: **replay**, not refuse, for
/// growth):
/// - `addTriples` during a run only appends vocab/triples and grows the tables
///   preserving existing rows. At swap time the trained rows overwrite ids
///   `< snapshot size`; rows added during the run keep their seed-init. The
///   triples added mid-run are kept (they simply were not trained on), and the
///   report says how many rows were replayed (`replayed`).
/// - A *wholesale* table replacement during a run (an `optimize` champion
///   install, which may also change `dims`/`scorer`) bumps the table
///   generation; the swap is then refused with an `unavailable` error and the
///   trained snapshot is discarded — the live tables are left untouched.
/// - Single flight: while a snapshot job is outstanding a second `begin_train`
///   (or `train_json`) returns `{"status":"training"}` (not an error) without
///   starting work.
pub struct TrainJob {
    cfg: TrainConfig,
    store: TripleStore,
    /// `Some` = snapshot copy; `None` = train the live tables in place.
    tables: Option<Tables>,
    kind: ScorerKind,
    dims: usize,
    gen: u64,
    last: Option<LastEpoch>,
}

impl TrainJob {
    /// True when this job trains the live tables (no snapshot was taken): the
    /// caller must keep the model locked and use [`TrainJob::run_in_place`].
    pub fn in_place(&self) -> bool {
        self.tables.is_none()
    }

    /// Train the snapshot tables. Touches nothing but the job itself, so the
    /// caller must NOT hold the model lock while this runs.
    // Snapshot jobs are native-only (wasm always trains in place).
    #[allow(dead_code)]
    pub fn run(&mut self) -> ruvector_kge::Result<()> {
        let Some(mut tables) = self.tables.take() else {
            return Err(ruvector_kge::KgeError::Invalid(
                "in-place training job needs run_in_place".into(),
            ));
        };
        let fit = self.fit(&mut tables);
        self.tables = Some(tables);
        fit
    }

    /// Train the model's live `tables` directly (in-place mode). The caller
    /// holds the model for the whole run.
    pub fn run_in_place(&mut self, tables: &mut Tables) -> ruvector_kge::Result<()> {
        if self.tables.is_some() {
            return Err(ruvector_kge::KgeError::Invalid(
                "snapshot training job needs run".into(),
            ));
        }
        self.fit(tables)
    }

    fn fit(&mut self, tables: &mut Tables) -> ruvector_kge::Result<()> {
        let last = &mut self.last;
        let mut record = |p: &ruvector_kge::Progress| {
            *last = Some((p.epoch, p.epochs, p.num_batches, p.loss, p.n3_penalty));
        };
        match self.kind {
            ScorerKind::Hole => HolE::new(self.dims)
                .and_then(|sc| Trainer::fit(tables, &sc, &self.store, &self.cfg, &mut record)),
            ScorerKind::Rotate => RotatE::new(self.dims)
                .and_then(|sc| Trainer::fit(tables, &sc, &self.store, &self.cfg, &mut record)),
        }
    }

    /// The snapshot tables (tests only; panics for an in-place job).
    #[cfg(test)]
    pub(crate) fn tables_for_test(&self) -> &Tables {
        self.tables.as_ref().expect("snapshot job")
    }
}

impl KgeModel {
    /// Train the tables from the current triples. `configJson` is a
    /// [`TrainConfig`] (all fields optional); `dims` is forced to the model's.
    /// If any triple carries a split tag, training uses ONLY the `train` split
    /// (never valid/test/transfer). Trains the live tables IN PLACE (no copy,
    /// the caller holds `&mut self` throughout) — the wasm path. The native
    /// wrappers use `begin_train` / [`TrainJob::run`] / `finish_train` instead
    /// so the model lock is released around the run.
    // The native wrapper composes the phases itself, so this form is used only
    // by the wasm build (and tests).
    #[allow(dead_code)]
    pub fn train_json(&mut self, config_json: &str) -> String {
        let mut job = match self.begin_job(config_json, false) {
            Ok(job) => job,
            Err(json) => return json,
        };
        let fit = match self.tables.as_mut() {
            Some(tables) => job.run_in_place(tables),
            None => Err(ruvector_kge::KgeError::Invalid("tables not built".into())),
        };
        self.finish_train(job, fit)
    }

    /// Phase 1 (brief lock): validate, snapshot, and mark the model training.
    /// Falls back to an in-place job (no snapshot, no flag) when the doubled
    /// table working set would exceed the model's byte cap — check
    /// [`TrainJob::in_place`]. `Err` carries the JSON to return verbatim — a
    /// request error, or `{"status":"training"}` when a job is outstanding.
    // Native-only entry point (wasm composes `train_json`).
    #[allow(dead_code)]
    pub fn begin_train(&mut self, config_json: &str) -> Result<TrainJob, String> {
        self.begin_job(config_json, true)
    }

    fn begin_job(&mut self, config_json: &str, snapshot: bool) -> Result<TrainJob, String> {
        if self.is_training() {
            return Err(serde_json::json!({ "status": "training" }).to_string());
        }
        let mut cfg: TrainConfig = serde_json::from_str(config_json)
            .map_err(|e| err_json("invalid", &format!("train config parse error: {e}")))?;
        cfg.dims = self.config.dims;
        if self.triples.is_empty() {
            return Err(err_json("invalid", "no triples to train on"));
        }
        // With ingested split tags, train ONLY on the "train" split — never on
        // valid/test/transfer (no leakage, ADR-006). Untagged: train on all.
        let train_triples = if self.has_split_tags() {
            self.triples_with_split(SplitLabel::Train)
        } else {
            self.triples.clone()
        };
        if train_triples.is_empty() {
            return Err(err_json(
                "invalid",
                "no 'train'-labelled triples to train on",
            ));
        }
        let store = TripleStore::new(train_triples).map_err(|e| kge_error_json(&e))?;
        self.ensure_built();
        // The ANN index is NOT dropped here: predict keeps using it (and the
        // pre-training tables) until the swap in `finish_train`.
        let tables = if snapshot && self.snapshot_fits() {
            self.set_training(true);
            self.tables.clone()
        } else {
            None
        };
        Ok(TrainJob {
            cfg,
            store,
            tables,
            kind: self.config.scorer,
            dims: self.config.dims,
            gen: self.table_gen(),
            last: None,
        })
    }

    /// True when a snapshot copy of the live tables fits beside them under the
    /// byte cap: `2 x bytes <= effective_max_bytes(maxTableBytes)`.
    fn snapshot_fits(&self) -> bool {
        let Some(t) = self.tables.as_ref() else {
            return false;
        };
        let doubled = (t.num_entities() as u64)
            .checked_add(t.num_relations() as u64)
            .and_then(|rows| rows.checked_mul(t.dims() as u64))
            .and_then(|n| n.checked_mul(8)); // 4 B per f32, twice
        matches!(doubled, Some(b) if b <= effective_max_bytes(self.config.max_table_bytes))
    }

    /// Phase 3 (brief lock): swap the trained rows in (replaying any growth
    /// that happened mid-run) or refuse on a stale generation. An in-place job
    /// already trained the live tables; this only publishes them (new
    /// generation, index dropped). Always clears the single-flight flag.
    pub fn finish_train(&mut self, job: TrainJob, fit: ruvector_kge::Result<()>) -> String {
        let in_place = job.in_place();
        if !in_place {
            self.set_training(false);
        }
        if let Err(e) = fit {
            if in_place {
                // Like pre-F2 in-place training, the live rows may be partly
                // updated: drop the now-stale index.
                self.invalidate_index();
            }
            return kge_error_json(&e);
        }
        let (replayed_e, replayed_r) = match job.tables {
            None => {
                // In place: the live tables ARE the trained tables (moved, not
                // copied, through replace_tables to bump the generation).
                if let Some(t) = self.tables.take() {
                    self.replace_tables(t);
                }
                (0, 0)
            }
            Some(trained) => {
                if job.gen != self.table_gen() || job.dims != self.config.dims {
                    return err_json(
                        "unavailable",
                        "the model's tables were replaced while training ran (e.g. by optimize); the training result was discarded - retrain",
                    );
                }
                self.ensure_built();
                let current = self.tables.take().unwrap();
                let (base_e, base_r) = (trained.num_entities(), trained.num_relations());
                let (cur_e, cur_r) = (current.num_entities(), current.num_relations());
                let merged = if cur_e == base_e && cur_r == base_r {
                    trained
                } else {
                    // Replay growth: trained rows for the snapshot ids, the
                    // live rows (seed-init) for ids interned while training ran.
                    let mut merged = current;
                    let d = job.dims;
                    let ne = base_e.min(cur_e) * d;
                    let nr = base_r.min(cur_r) * d;
                    merged.entities_raw_mut()[..ne].copy_from_slice(&trained.entities_raw()[..ne]);
                    merged.relations_raw_mut()[..nr]
                        .copy_from_slice(&trained.relations_raw()[..nr]);
                    merged
                };
                self.replace_tables(merged);
                (cur_e.saturating_sub(base_e), cur_r.saturating_sub(base_r))
            }
        };
        let replayed = serde_json::json!({ "entities": replayed_e, "relations": replayed_r });
        let mode = if in_place { "in-place" } else { "snapshot" };
        match job.last {
            Some((epoch, epochs, batches, loss, n3)) => serde_json::json!({
                "status": "trained",
                "epoch": epoch,
                "epochs": epochs,
                "batches": batches,
                "loss": loss,
                "n3Penalty": n3,
                "triples": self.triples.len(),
                "entities": self.entities.len(),
                "relations": self.relations.len(),
                "replayed": replayed,
                "mode": mode,
            })
            .to_string(),
            None => serde_json::json!({
                "status": "trained",
                "epoch": 0, "epochs": job.cfg.epochs, "loss": 0.0,
                "note": "no epochs run (epochs=0)",
                "mode": mode,
            })
            .to_string(),
        }
    }

    /// Clear the single-flight flag without touching the tables (the native
    /// wrapper's drop guard calls this if `run` panics).
    // Used only by the native wrapper (wasm has no unwinding worker thread).
    #[allow(dead_code)]
    pub fn abort_train(&mut self) {
        self.set_training(false);
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
}

#[cfg(test)]
#[path = "pipeline_tests.rs"]
mod tests;
