//! The scoring seam (ADR-002 §1). Three implementations:
//! `hole` (default), `rotate` (opt-in), `complex` (the ComplEx-N3-R recipe,
//! ADR-007; also the HolE ≡ ComplEx gate). Everything else — training,
//! evaluation, ANN — goes through this trait and never special-cases a scorer.

pub trait Scorer: Send + Sync {
    /// Embedding dimension every vector passed to this scorer must have.
    fn dims(&self) -> usize;

    /// Score one triple; higher is more plausible.
    fn score(&self, s: &[f32], r: &[f32], o: &[f32]) -> f32;

    /// The vector `q` such that `score(s, r, o)` equals a plain dot product of
    /// `q` with the *indexed* representation of the open-slot entity (see
    /// [`Scorer::index_vector`]). `anchor` is the known entity; `side` says
    /// which slot is open. This is what makes candidate retrieval a
    /// `DistanceMetric::DotProduct` HNSW search (ADR-001 §3, ADR-002 §2).
    fn query_vector(&self, r: &[f32], anchor: &[f32], side: crate::Side) -> Vec<f32>;

    /// The representation of an entity that is stored in the ANN index (for
    /// HolE: `[Re F(e); Im F(e)]`). Its length is [`Scorer::index_dims`].
    fn index_vector(&self, e: &[f32]) -> Vec<f32>;

    /// Length of [`Scorer::index_vector`] output.
    fn index_dims(&self) -> usize;

    /// Stable identifier recorded in receipts, e.g. `hole-fft@d256`.
    fn id(&self) -> &str;

    /// Whether the ANN dot product of [`Scorer::query_vector`] with
    /// [`Scorer::index_vector`] equals the exact [`Scorer::score`]. HolE keeps
    /// the default `true` (its Fourier index is an exact inner-product
    /// factorisation). RotatE overrides to `false` — its appended-norm MIPS
    /// form is order-preserving but not the score, so the exact rerank is
    /// mandatory (ADR-002 §4). Provided method; adding it does not change any
    /// pinned signature.
    fn ann_exact(&self) -> bool {
        true
    }
}

/// A [`Differentiable`](crate::Differentiable) scorer whose score is a plain
/// inner product `⟨q(r, anchor, side), index(e)⟩` in the open-slot entity —
/// the seam the batched 1-N kernel (plan M2, `train::one_to_n::OneToN`) plugs
/// into. Reached from `&dyn Differentiable` through
/// [`Differentiable::as_bilinear`](crate::Differentiable::as_bilinear), so the
/// trainer never special-cases a scorer.
///
/// Implemented by [`ComplEx`]. HolE qualifies through its frequency view
/// (`index_is_identity() == false`), which the kernel lane may add.
pub trait Bilinear: crate::Differentiable {
    /// Write the query `q` for `(anchor, r, ?)` (`Side::Tail`, anchor = s) or
    /// `(?, r, anchor)` (`Side::Head`, anchor = o) into `out` (length
    /// [`Scorer::index_dims`]). Equals [`Scorer::query_vector`].
    fn query_into(&self, r: &[f32], anchor: &[f32], side: crate::Side, out: &mut [f32]);

    /// Backpropagate `d_q = dL/dq` through [`Bilinear::query_into`],
    /// **accumulating** (`+=`) into `d_r` and `d_anchor` (length
    /// [`Scorer::dims`]).
    fn query_backward(
        &self,
        r: &[f32],
        anchor: &[f32],
        side: crate::Side,
        d_q: &[f32],
        d_r: &mut [f32],
        d_anchor: &mut [f32],
    );

    /// True when [`Scorer::index_vector`] is the identity (the raw entity
    /// table *is* the `E × index_dims` candidate matrix, and `dL/d index(e)`
    /// is `dL/d e`). Only such scorers take the batched 1-N training path.
    fn index_is_identity(&self) -> bool;
}

pub mod complex;
pub mod fft;
pub mod hole;
pub mod rotate;

pub use complex::ComplEx;
pub use hole::HolE;
pub use rotate::RotatE;
