//! Query and mutation operations on [`KgeModel`]. Duplicated VERBATIM in
//! `ruvector-kge-ffi/src/ops.rs` and `ruvector-kge-wasm/src/ops.rs`.
//!
//! Input limits (ADR-005) are enforced here, before anything reaches the core.
//! `predict` (exhaustive or ANN-accelerated), `similarRelations` and `compose`
//! are wired to the landed HolE / RotatE scorers through the [`build_scorer`]
//! seam. Training, evaluation and index build live in `pipeline.rs`; only
//! `optimize` remains a stub (it needs an `Evaluator` the binding does not
//! provide yet).

use crate::model::{err_json, kge_error_json, KgeModel};
use ruvector_kge::scorer::{HolE, RotatE};
use ruvector_kge::{AnnIndex, Scorer, ScorerKind, Side, Triple};
use serde::Deserialize;

// ---- ADR-005 input limits -------------------------------------------------

const MAX_ENTITIES: usize = 1_000_000;
const MAX_RELATIONS: usize = 100_000;
const MAX_LABEL_BYTES: usize = 1024;
const MAX_K: usize = 1000;

fn default_k() -> usize {
    10
}

fn check_k(k: usize) -> Option<String> {
    if !(1..=MAX_K).contains(&k) {
        return Some(err_json("limit", "k must be in 1..=1000"));
    }
    None
}

// ---- scorer seam ----------------------------------------------------------

/// Construct the trait object for the model's scorer. Returns `None` only if
/// the core constructor rejects `dims` (odd/zero) — which the model
/// constructor already prevents, so `predict` treats `None` as a `scorer`
/// error, not a routine outcome. This is the single seam every scoring path
/// goes through (ADR-002 §1).
fn build_scorer(kind: ScorerKind, dims: usize) -> Option<Box<dyn Scorer>> {
    match kind {
        ScorerKind::Hole => HolE::new(dims).ok().map(|s| Box::new(s) as Box<dyn Scorer>),
        ScorerKind::Rotate => RotatE::new(dims)
            .ok()
            .map(|s| Box::new(s) as Box<dyn Scorer>),
    }
}

fn scorer_name(kind: ScorerKind) -> &'static str {
    match kind {
        ScorerKind::Hole => "hole",
        ScorerKind::Rotate => "rotate",
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

fn by_score_desc(a: &(u32, f32), b: &(u32, f32)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
}

// ---- operations -----------------------------------------------------------

#[derive(Deserialize)]
struct TripleInput {
    s: String,
    r: String,
    o: String,
}

#[derive(Deserialize)]
struct PredictQuery {
    #[serde(default)]
    s: Option<String>,
    #[serde(default)]
    r: Option<String>,
    #[serde(default)]
    o: Option<String>,
    #[serde(default = "default_k")]
    k: usize,
}

#[derive(Deserialize)]
struct SimilarQuery {
    r: String,
    #[serde(default = "default_k")]
    k: usize,
}

#[derive(Deserialize)]
struct ComposeQuery {
    r1: String,
    r2: String,
    s: String,
    #[serde(default = "default_k")]
    k: usize,
}

impl KgeModel {
    /// Admit `[{"s","r","o"}]`. Interns labels, appends triples, grows the
    /// tables preserving existing rows, and drops the ANN index. Enforces the
    /// ADR-005 table and label-size caps.
    pub fn add_triples_json(&mut self, triples_json: &str) -> String {
        let items: Vec<TripleInput> = match serde_json::from_str(triples_json) {
            Ok(v) => v,
            Err(e) => return err_json("invalid", &format!("triples JSON parse error: {e}")),
        };
        for t in &items {
            for label in [&t.s, &t.r, &t.o] {
                if label.is_empty() {
                    return err_json("invalid", "triple label must not be empty");
                }
                if label.len() > MAX_LABEL_BYTES {
                    return err_json("limit", "triple label exceeds 1 KiB");
                }
            }
        }
        let mut added = 0usize;
        for t in &items {
            if self.entities.len().saturating_add(2) > MAX_ENTITIES {
                return err_json("limit", "entity table would exceed 1,000,000");
            }
            if self.relations.len().saturating_add(1) > MAX_RELATIONS {
                return err_json("limit", "relation table would exceed 100,000");
            }
            let s = self.intern_entity(&t.s);
            let o = self.intern_entity(&t.o);
            let r = self.intern_relation(&t.r);
            self.triples.push(Triple::new(s, r, o));
            added += 1;
        }
        if added > 0 {
            self.grow_tables();
            self.invalidate_index();
        }
        serde_json::json!({
            "added": added,
            "entities": self.entities.len(),
            "relations": self.relations.len(),
            "triples": self.triples.len(),
        })
        .to_string()
    }

    /// `{"s","r","k"}` (open tail) or `{"r","o","k"}` (open head). Ranks
    /// candidates for the open slot: ANN-accelerated when an index is present
    /// (`"ann":true,"exact":false`), else exhaustive top-k through the
    /// [`Scorer`](ruvector_kge::Scorer) trait (`"ann":false,"exact":true`).
    pub fn predict_json(&mut self, query_json: &str) -> String {
        let q: PredictQuery = match serde_json::from_str(query_json) {
            Ok(v) => v,
            Err(e) => return err_json("invalid", &format!("query JSON parse error: {e}")),
        };
        if let Some(e) = check_k(q.k) {
            return e;
        }
        let r_label = match &q.r {
            Some(r) => r,
            None => return err_json("invalid", "predict requires a relation \"r\""),
        };
        let (anchor_label, side) = match (&q.s, &q.o) {
            (Some(s), None) => (s, Side::Tail),
            (None, Some(o)) => (o, Side::Head),
            (Some(_), Some(_)) => {
                return err_json("invalid", "predict leaves exactly one of \"s\"/\"o\" open")
            }
            (None, None) => return err_json("invalid", "predict needs one of \"s\"/\"o\""),
        };
        let r_id = match self.relations.get(r_label) {
            Some(id) => id,
            None => return err_json("invalid", "unknown relation label"),
        };
        let a_id = match self.entities.get(anchor_label) {
            Some(id) => id,
            None => return err_json("invalid", "unknown entity label"),
        };
        let scorer = match build_scorer(self.config.scorer, self.config.dims) {
            Some(sc) => sc,
            None => {
                return err_json(
                    "scorer",
                    &format!(
                        "scorer \"{}\" could not be constructed",
                        scorer_name(self.config.scorer)
                    ),
                )
            }
        };
        self.ensure_built();
        let tables = self.tables.as_ref().unwrap();
        let r_vec = match tables.relation(r_id) {
            Ok(v) => v,
            Err(e) => return kge_error_json(&e),
        };
        let a_vec = match tables.entity(a_id) {
            Ok(v) => v,
            Err(e) => return kge_error_json(&e),
        };
        let n = tables.num_entities();
        let k = q.k.min(n);
        let exact = |e: u32| match side {
            Side::Tail => scorer.score(a_vec, r_vec, tables.entity(e).unwrap()),
            Side::Head => scorer.score(tables.entity(e).unwrap(), r_vec, a_vec),
        };
        // ANN path when an index is present; exhaustive otherwise.
        let (ranked, ann_used): (Vec<(u32, f32)>, bool) = if let Some(index) = self.ann.as_ref() {
            let query = scorer.query_vector(r_vec, a_vec, side);
            let ef = (k * 4).max(64);
            match index.candidates(&query, k, ef) {
                Ok(cands) => (
                    AnnIndex::rerank(&cands, exact, k)
                        .into_iter()
                        .map(|c| (c.entity, c.score))
                        .collect(),
                    true,
                ),
                Err(e) => return kge_error_json(&e),
            }
        } else {
            let mut scored: Vec<(u32, f32)> = (0..n as u32).map(|e| (e, exact(e))).collect();
            scored.sort_by(by_score_desc);
            scored.truncate(k);
            (scored, false)
        };
        let candidates = self.label_candidates(ranked);
        serde_json::json!({ "candidates": candidates, "exact": !ann_used, "ann": ann_used })
            .to_string()
    }

    /// `{"r","k"}` -> relations ranked by cosine of their vectors. Works today
    /// (needs only the relation table); values are meaningful once trained.
    pub fn similar_relations_json(&mut self, query_json: &str) -> String {
        let q: SimilarQuery = match serde_json::from_str(query_json) {
            Ok(v) => v,
            Err(e) => return err_json("invalid", &format!("query JSON parse error: {e}")),
        };
        if let Some(e) = check_k(q.k) {
            return e;
        }
        let r_id = match self.relations.get(&q.r) {
            Some(id) => id,
            None => return err_json("invalid", "unknown relation label"),
        };
        self.ensure_built();
        let tables = self.tables.as_ref().unwrap();
        let base = tables.relation(r_id).unwrap();
        let nr = tables.num_relations();
        let mut scored: Vec<(u32, f32)> = Vec::new();
        for rr in 0..nr as u32 {
            if rr == r_id {
                continue;
            }
            scored.push((rr, cosine(base, tables.relation(rr).unwrap())));
        }
        scored.sort_by(by_score_desc);
        let relations: Vec<_> = scored
            .into_iter()
            .take(q.k)
            .map(|(id, score)| {
                serde_json::json!({
                    "relation": self.relations.label(id).unwrap_or(""),
                    "score": score,
                })
            })
            .collect();
        serde_json::json!({ "relations": relations }).to_string()
    }

    /// `{"r1","r2","s","k"}`. Composes `r1 ∘ r2` into a synthetic relation
    /// vector and ranks tails for `(s, r1∘r2, ?)`. RotatE only: HolE/ComplEx
    /// cannot represent relation composition (kind `unsupported`).
    pub fn compose_json(&mut self, query_json: &str) -> String {
        let q: ComposeQuery = match serde_json::from_str(query_json) {
            Ok(v) => v,
            Err(e) => return err_json("invalid", &format!("query JSON parse error: {e}")),
        };
        if let Some(e) = check_k(q.k) {
            return e;
        }
        if matches!(self.config.scorer, ScorerKind::Hole) {
            return err_json(
                "unsupported",
                "compose is defined only for the rotate scorer (HolE/ComplEx cannot represent relation composition)",
            );
        }
        let r1_id = match self.relations.get(&q.r1) {
            Some(id) => id,
            None => return err_json("invalid", "unknown relation label for \"r1\""),
        };
        let r2_id = match self.relations.get(&q.r2) {
            Some(id) => id,
            None => return err_json("invalid", "unknown relation label for \"r2\""),
        };
        let s_id = match self.entities.get(&q.s) {
            Some(id) => id,
            None => return err_json("invalid", "unknown entity label"),
        };
        let rot = match RotatE::new(self.config.dims) {
            Ok(r) => r,
            Err(e) => return kge_error_json(&e),
        };
        self.ensure_built();
        let tables = self.tables.as_ref().unwrap();
        let composed = rot.compose(
            tables.relation(r1_id).unwrap(),
            tables.relation(r2_id).unwrap(),
        );
        let s_vec = tables.entity(s_id).unwrap();
        let n = tables.num_entities();
        let mut scored: Vec<(u32, f32)> = Vec::with_capacity(n);
        for e in 0..n as u32 {
            scored.push((e, rot.score(s_vec, &composed, tables.entity(e).unwrap())));
        }
        scored.sort_by(by_score_desc);
        scored.truncate(q.k.min(n));
        let candidates = self.label_candidates(scored);
        serde_json::json!({ "candidates": candidates, "exact": true, "ann": false }).to_string()
    }

    /// Map ranked `(entity-id, score)` pairs to `{entity,score}` JSON, resolving
    /// ids back to labels (the binding owns the vocab).
    pub(crate) fn label_candidates(&self, ranked: Vec<(u32, f32)>) -> Vec<serde_json::Value> {
        ranked
            .into_iter()
            .map(|(id, score)| {
                serde_json::json!({
                    "entity": self.entities.label(id).unwrap_or(""),
                    "score": score,
                })
            })
            .collect()
    }
}
