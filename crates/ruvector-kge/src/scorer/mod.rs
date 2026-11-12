//! The scoring seam (ADR-002 §1). Three implementations are planned:
//! `hole` (default), `rotate` (opt-in), `complex` (test-only, to assert
//! HolE ≡ ComplEx). Everything else — training, evaluation, ANN — goes through
//! this trait and never special-cases a scorer.

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

#[cfg(test)]
mod complex;
pub mod fft;
pub mod hole;
pub mod rotate;

pub use hole::HolE;
pub use rotate::RotatE;
