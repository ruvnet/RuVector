//! The self-optimization campaign (ADR-004) wired to the core `Campaign` over a
//! real [`TrainerEvaluator`]. Duplicated VERBATIM in
//! `ruvector-kge-ffi/src/optimize.rs` and `ruvector-kge-wasm/src/optimize.rs`
//! (`cmp` them to verify).
//!
//! `optimizeJson` fits the HPO/model arms over the model's frozen splits, gates
//! each against the incumbent, then **installs the champion's exact trained
//! tables** into the model (no retraining) and returns the `CampaignReport`
//! plus the hash-chained receipts JSONL. ADR-005 input limits (budget cap, grid
//! size, valid `alpha`) are enforced here as `{"error":{...}}` envelopes.

use crate::model::{err_json, KgeModel, SplitLabel};
use ruvector_kge::optimize::{Campaign, CampaignSpec, HpoGrid, Knobs, Loss, TrainerEvaluator};
use ruvector_kge::{Split4, Tables, TripleStore};
use serde::Deserialize;

/// Hard cap on the per-campaign evaluation budget (ADR-005: bounded work).
const MAX_BUDGET: u32 = 64;
/// Hard cap on the expanded HPO grid size (`dims × lrs × losses × n3`).
const MAX_GRID: usize = 64;

/// The `optimizeJson` request. Every field is optional; the defaults are a
/// small, promotion-friendly sweep at the model's own dimension.
#[derive(Deserialize)]
struct OptimizeSpec {
    #[serde(default = "default_budget")]
    budget: u32,
    #[serde(default = "default_alpha")]
    alpha: f32,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(rename = "splitRatios", default = "default_ratios")]
    split_ratios: [f64; 4],
    #[serde(rename = "transferTolerance", default = "default_transfer_tol")]
    transfer_tolerance: f32,
    #[serde(rename = "lambdaCost", default = "default_lambda_cost")]
    lambda_cost: f32,
    /// Override the HPO grid (`{"dims":[..],"lrs":[..],"losses":[..],"n3_lambdas":[..]}`).
    #[serde(default)]
    grid: Option<HpoGrid>,
}

fn default_budget() -> u32 {
    16
}
fn default_alpha() -> f32 {
    0.05
}
fn default_ratios() -> [f64; 4] {
    [0.7, 0.1, 0.1, 0.1]
}
fn default_transfer_tol() -> f32 {
    0.05
}
fn default_lambda_cost() -> f32 {
    0.1
}

impl KgeModel {
    /// Run one self-optimization campaign and install its champion. Returns the
    /// report JSON, or an `{"error":{...}}` envelope on a bad request.
    pub fn optimize_json(&mut self, campaign_json: &str) -> String {
        let spec: OptimizeSpec = match serde_json::from_str(campaign_json) {
            Ok(s) => s,
            Err(e) => return err_json("invalid", &format!("optimize spec parse error: {e}")),
        };
        if self.triples.is_empty() {
            return err_json("invalid", "no triples to optimize over");
        }
        if spec.budget == 0 {
            return err_json("invalid", "budget must be >= 1");
        }
        if !(spec.alpha > 0.0 && spec.alpha < 1.0) {
            return err_json("invalid", "alpha must be in (0, 1)");
        }
        let seed = spec.seed.unwrap_or(self.config.seed);

        // The filter store: the full graph (filtered ranking sees every fact).
        let store = match TripleStore::new(self.triples.clone()) {
            Ok(s) => s,
            Err(e) => return crate::model::kge_error_json(&e),
        };

        // Frozen splits: per-triple tags verbatim when present (ADR-006), else a
        // seeded `split4` (whose transfer holdout is whole-relation — see README).
        let (split, split_source) = if self.has_split_tags() {
            (
                Split4 {
                    train: self.triples_with_split(SplitLabel::Train),
                    valid: self.triples_with_split(SplitLabel::Valid),
                    transfer: self.triples_with_split(SplitLabel::Transfer),
                    test: self.triples_with_split(SplitLabel::Test),
                },
                "per-triple",
            )
        } else {
            match store.split4(seed, spec.split_ratios) {
                Ok(s) => (s, "split4"),
                Err(e) => return crate::model::kge_error_json(&e),
            }
        };
        if split.valid.is_empty() || split.test.is_empty() {
            return err_json(
                "invalid",
                "optimize needs a non-empty valid and test split (tag triples or pass splitRatios)",
            );
        }

        // Baseline knobs = the model's scorer/dims at trainer defaults; the grid
        // varies dims/lr/loss/n3 over it (ADR-004 loop 1 ranked first).
        let baseline = Knobs {
            dims: self.config.dims,
            scorer: self.config.scorer,
            ..Knobs::default()
        };
        let grid = spec.grid.unwrap_or_else(|| default_grid(self.config.dims));
        if let Some(msg) = validate_grid(&grid) {
            return err_json("invalid", &msg);
        }

        let campaign_spec = CampaignSpec {
            baseline,
            grid,
            eta: 2,
            alpha: spec.alpha,
            lambda: 0.5,
            lambda_cost: spec.lambda_cost,
            transfer_tolerance: spec.transfer_tolerance,
            per_day_evals: spec.budget.min(MAX_BUDGET),
            day_key: "optimize".to_string(),
        };

        let baseline_id =
            ruvector_kge::optimize::Proposal::baseline(campaign_spec.baseline.clone()).id;
        let mut evaluator = TrainerEvaluator::new(store, split, seed);
        let report = Campaign::run(&campaign_spec, &mut evaluator);

        // Install the champion's exact trained tables (no retraining). Guard the
        // shape so a mismatched table set is reported, never installed blindly.
        let installed = self.install_champion(&evaluator, report.champion_id, &report.champion);

        let promoted = report.champion_id != baseline_id;
        let receipts_count = report
            .receipts_jsonl
            .lines()
            .filter(|l| !l.trim().is_empty())
            .count();
        let proposals = serde_json::to_value(&report.proposals).unwrap_or(serde_json::Value::Null);
        let champion = serde_json::to_value(&report.champion).unwrap_or(serde_json::Value::Null);

        serde_json::json!({
            "champion": champion,
            // String id (full u64 would lose precision as a JS number), so it
            // matches the receipts' `knobs_hash` and `proposals[].id` exactly.
            "championId": report.champion_id.to_string(),
            "promoted": promoted,
            "installed": installed,
            "paused": report.paused,
            "budgetConsumed": report.budget_consumed,
            "splitSource": split_source,
            "proposals": proposals,
            "proposalCount": report.proposals.len(),
            "val": { "baselineMrr": report.baseline_val_mrr, "championMrr": report.champion_val_mrr },
            "transfer": {
                "baselineMrr": report.baseline_transfer_mrr,
                "championMrr": report.champion_transfer_mrr,
            },
            "test": {
                "baselineMrr": report.baseline_test.baseline_accuracy,
                "championMrr": report.champion_test.champion_accuracy,
            },
            "scorer": self.config.scorer,
            "dims": self.config.dims,
            "receiptsCount": receipts_count,
            "receipts": report.receipts_jsonl,
        })
        .to_string()
    }

    /// Install the champion's trained tables and knobs. Returns `true` on a
    /// successful install; `false` if the retained tables do not match the
    /// model's vocab shape (defensive — never installs an incompatible table).
    fn install_champion(
        &mut self,
        evaluator: &TrainerEvaluator,
        champion_id: u64,
        champion: &Knobs,
    ) -> bool {
        let Some(tables) = evaluator.tables(champion_id) else {
            return false;
        };
        let ne = self.entities.len().max(1);
        let nr = self.relations.len().max(1);
        if tables.num_entities() != ne
            || tables.num_relations() != nr
            || tables.dims() != champion.dims
        {
            return false;
        }
        self.config.scorer = champion.scorer;
        self.config.dims = champion.dims;
        self.tables = Some(Tables::clone(&tables));
        self.invalidate_index();
        true
    }
}

/// The default HPO grid: the model's dimension at two healthy learning rates,
/// so the campaign has a plausibly-better arm than the `lr=0.01` baseline.
fn default_grid(dims: usize) -> HpoGrid {
    HpoGrid {
        dims: vec![dims],
        lrs: vec![0.05, 0.1],
        losses: vec![Loss::CrossEntropy],
        n3_lambdas: vec![0.0],
    }
}

/// ADR-005 grid caps: every axis non-empty, dims positive and even (HolE/RotatE
/// require it), and the expanded product within [`MAX_GRID`].
fn validate_grid(grid: &HpoGrid) -> Option<String> {
    if grid.dims.is_empty()
        || grid.lrs.is_empty()
        || grid.losses.is_empty()
        || grid.n3_lambdas.is_empty()
    {
        return Some("grid axes must each be non-empty".to_string());
    }
    if grid.dims.iter().any(|&d| d == 0 || !d.is_multiple_of(2)) {
        return Some("grid dims must be positive and even".to_string());
    }
    let size = grid.dims.len() * grid.lrs.len() * grid.losses.len() * grid.n3_lambdas.len();
    if size > MAX_GRID {
        return Some(format!("grid expands to {size} configs; max is {MAX_GRID}"));
    }
    None
}
