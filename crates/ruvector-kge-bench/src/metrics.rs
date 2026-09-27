//! Evaluation (valid or test) through the core's filtered, reciprocal-aware
//! ranking, and the metric / progress records receipts and checkpoints carry.

use crate::canon::sha256_u32s;
use crate::config::EVAL_SEED;
use anyhow::Result;
use ruvector_kge::eval::{evaluate_rank_pair_with, EvalConfig, RankPair, TieBreak};
use ruvector_kge::scorer::ComplEx;
use ruvector_kge::{Tables, Triple, TripleStore};
use serde::{Deserialize, Serialize};

/// MR / MRR / Hits over a set of 1-based ranks (f64 accumulation).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub count: usize,
    pub mrr: f64,
    pub mr: f64,
    pub hits1: f64,
    pub hits3: f64,
    pub hits10: f64,
}

impl Summary {
    pub fn of<'a>(ranks: impl IntoIterator<Item = &'a usize>) -> Self {
        let (mut n, mut rr, mut r, mut h1, mut h3, mut h10) =
            (0usize, 0f64, 0f64, 0usize, 0usize, 0usize);
        for &k in ranks {
            n += 1;
            rr += 1.0 / k as f64;
            r += k as f64;
            h1 += (k <= 1) as usize;
            h3 += (k <= 3) as usize;
            h10 += (k <= 10) as usize;
        }
        let d = n.max(1) as f64;
        Self {
            count: n,
            mrr: rr / d,
            mr: r / d,
            hits1: h1 as f64 / d,
            hits3: h3 as f64 / d,
            hits10: h10 as f64 / d,
        }
    }
}

/// Combined and per-side metrics under one tie policy. Index layout of the
/// rank vectors (core `RankPair`): tail query at `2i`, head query at `2i+1`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sided {
    pub combined: Summary,
    pub tail: Summary,
    pub head: Summary,
}

impl Sided {
    pub fn of(ranks: &[usize]) -> Self {
        Self {
            combined: Summary::of(ranks),
            tail: Summary::of(ranks.iter().step_by(2)),
            head: Summary::of(ranks.iter().skip(1).step_by(2)),
        }
    }
}

/// Both tie policies (ADR-007 §2.2: verdicts on Bottom, RANDOM alongside).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvalMetrics {
    pub bottom: Sided,
    pub random: Sided,
}

/// Per-query rank vectors with their sha256 (u32-LE, `canon::sha256_u32s`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankVectors {
    pub bottom: Vec<usize>,
    pub random: Vec<usize>,
    pub bottom_sha256: String,
    pub random_sha256: String,
}

impl RankVectors {
    pub fn from_pair(p: RankPair) -> Self {
        Self {
            bottom_sha256: sha256_u32s(&p.bottom),
            random_sha256: sha256_u32s(&p.random),
            bottom: p.bottom,
            random: p.random,
        }
    }

    pub fn metrics(&self) -> EvalMetrics {
        EvalMetrics {
            bottom: Sided::of(&self.bottom),
            random: Sided::of(&self.random),
        }
    }
}

/// Filtered, reciprocal (head query = `(o, r⁻¹, ?)`) Bottom + RANDOM ranks of
/// `triples` against `filter`, on the core GEMM evaluation path.
pub fn rank_split(
    tables: &Tables,
    filter: &TripleStore,
    triples: &[Triple],
) -> Result<RankVectors> {
    let scorer = ComplEx::new(tables.dims())?;
    let cfg = EvalConfig {
        tie_break: TieBreak::Bottom,
        filtered: true,
        seed: EVAL_SEED,
        reciprocal: true,
    };
    let pair = evaluate_rank_pair_with(tables, &scorer, filter, triples, &cfg)?;
    Ok(RankVectors::from_pair(pair))
}

/// One epoch of a run, as recorded in checkpoints and receipts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochRecord {
    pub epoch: usize,
    pub loss: f32,
    pub n3_penalty: f32,
    pub rp_loss: f32,
    pub train_secs: f64,
    pub eval_secs: f64,
    /// Valid metrics when this epoch was evaluated.
    pub valid: Option<EvalMetrics>,
}

/// The best-on-valid epoch so far (selection metric: Bottom MRR, strict `>`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Best {
    pub epoch: usize,
    pub valid_mrr_bottom: f64,
    pub metrics: EvalMetrics,
    /// Per-query valid ranks at the best epoch (M4 sizes δ from these).
    pub valid_ranks: RankVectors,
    pub weights_sha256: String,
}

/// Why a run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    MaxEpochs,
    EarlyStop,
    /// `--stop-after-epoch` (simulated interruption; resumable).
    Interrupted,
}

/// Everything about a run's history that must survive a resume.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RunProgress {
    pub history: Vec<EpochRecord>,
    pub best: Option<Best>,
    /// Evaluations since the best one (early-stop counter).
    pub evals_since_best: usize,
    /// Epoch indices at which the run was resumed from a checkpoint.
    pub resumed_at: Vec<usize>,
    pub early_stopped: bool,
}
