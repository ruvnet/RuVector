//! Training configuration: the JSON the bindings accept (`trainJson`) and the
//! HPO loop varies. Every field has a serde default equal to [`Default`], so a
//! partial (or empty) object deserializes, and the defaults reproduce the
//! pre-M1 trainer exactly (summed loss, elementwise N3, no reciprocals, no RP,
//! tables kept as built, sparse optimizer state).

use super::init::Init;
use super::optim::{OptimKind, StateLayout};
use serde::{Deserialize, Serialize};

/// Which training regime to run (ADR-003 §1). A bandit arm, not hardcoded.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LossKind {
    /// Self-adversarial negative sampling (RotatE).
    SelfAdversarial {
        #[serde(default = "neg_count_default")]
        neg_count: usize,
        #[serde(default = "temperature_default")]
        temperature: f32,
        #[serde(default = "margin_default")]
        margin: f32,
    },
    /// 1-vs-all cross-entropy over every entity (ComplEx-N3): both sides, or
    /// tail-only over the reciprocal-augmented examples when
    /// [`TrainConfig::reciprocal`] is set.
    #[default]
    OneVsAll,
}

fn neg_count_default() -> usize {
    16
}
fn temperature_default() -> f32 {
    1.0
}
fn margin_default() -> f32 {
    9.0
}

/// Form of the N3 regulariser (weight [`TrainConfig::n3_lambda`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum N3Form {
    /// `Σ_i |x_i|³` over the raw reals (pre-M1 behaviour).
    #[default]
    Elementwise,
    /// `Σ_j |z_j|³` over complex moduli `|z_j| = sqrt(re_j² + im_j²)` of the
    /// split `[re; im]` layout (Lacroix 2018). Needs even `dims`.
    Moduli,
}

/// How per-example terms combine within a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Reduction {
    /// Sum over the batch (pre-M1 behaviour).
    #[default]
    Sum,
    /// Average over the batch's examples: loss **and** regulariser are both
    /// divided by the batch size (kbc / ssl-RP). The last short batch
    /// divides by its own size.
    Mean,
}

/// Which [`OneToN`](super::OneToN) implementation the batched 1-vs-all path
/// uses. Consulted only when the scorer is bilinear with an identity index
/// (ComplEx) and the loss is [`LossKind::OneVsAll`]; every other scorer/loss
/// ignores it, so HPO arms on HolE / RotatE never fail on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OneNKernel {
    /// [`NaiveOneToN`](super::NaiveOneToN): fixed-order reference loops.
    #[default]
    Naive,
    /// [`GemmOneToN`](crate::kernel::GemmOneToN): three SGEMMs per batch,
    /// multi-threaded under the `parallel` feature (plan M2).
    Gemm,
}

/// Training hyperparameters. Every field has a serde default so a partial
/// config deserializes (the HPO loop varies a subset per arm, ADR-004).
///
/// The ComplEx-N3-R recipe (ADR-007 §3) is, on a `ComplEx` scorer:
/// `{"loss":{"kind":"one_vs_all"}, "reciprocal":true, "n3_form":"moduli",
/// "loss_reduction":"mean", "init":{"kind":"normal","scale":1e-3},
/// "optimizer":{"kind":"adagrad"}, "optim_state":"dense", "lr":0.1,
/// "rp_weight":<w_rel>}`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    #[serde(default = "dims_default")]
    pub dims: usize,
    #[serde(default = "epochs_default")]
    pub epochs: usize,
    #[serde(default = "batch_size_default")]
    pub batch_size: usize,
    #[serde(default = "lr_default")]
    pub lr: f32,
    #[serde(default)]
    pub optimizer: OptimKind,
    #[serde(default)]
    pub loss: LossKind,
    #[serde(default = "n3_lambda_default")]
    pub n3_lambda: f32,
    #[serde(default)]
    pub seed: u64,
    /// Train with reciprocal relations (`2·R` relation rows; see
    /// [`super::reciprocal`] for the id convention).
    #[serde(default)]
    pub reciprocal: bool,
    /// Table (re-)initialisation at the start of `fit`.
    #[serde(default)]
    pub init: Init,
    /// Elementwise or complex-moduli N3.
    #[serde(default)]
    pub n3_form: N3Form,
    /// Sum or batch-mean reduction of loss and regulariser.
    #[serde(default)]
    pub loss_reduction: Reduction,
    /// Relation-prediction auxiliary loss weight (`w_rel`, Chen 2021); 0 = off.
    #[serde(default)]
    pub rp_weight: f32,
    /// Sparse (per-row map) or dense (flat) gradient and optimizer state.
    #[serde(default)]
    pub optim_state: StateLayout,
    /// Batched 1-N kernel (`"naive"` | `"gemm"`); see [`OneNKernel`].
    #[serde(default)]
    pub one_n_kernel: OneNKernel,
}

fn dims_default() -> usize {
    128
}
fn epochs_default() -> usize {
    100
}
fn batch_size_default() -> usize {
    256
}
fn lr_default() -> f32 {
    0.1
}
fn n3_lambda_default() -> f32 {
    1e-3
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            dims: dims_default(),
            epochs: epochs_default(),
            batch_size: batch_size_default(),
            lr: lr_default(),
            optimizer: OptimKind::default(),
            loss: LossKind::default(),
            n3_lambda: n3_lambda_default(),
            seed: 0,
            reciprocal: false,
            init: Init::default(),
            n3_form: N3Form::default(),
            loss_reduction: Reduction::default(),
            rp_weight: 0.0,
            optim_state: StateLayout::default(),
            one_n_kernel: OneNKernel::default(),
        }
    }
}

impl TrainConfig {
    /// Complex coordinates per embedding for complex-valued scorers
    /// (`dims / 2`; "rank" in kbc / ssl-RP). Recorded in receipts; derived,
    /// not stored, so it can never disagree with `dims`.
    pub fn complex_rank(&self) -> usize {
        self.dims / 2
    }
}
