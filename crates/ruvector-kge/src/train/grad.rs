//! The training gradient seam. [`Scorer`] (ADR-002) exposes only a forward
//! `score`; training needs analytic gradients, so a scorer that can be trained
//! implements [`Differentiable`] in addition. The scorer crate implements it
//! for HolE and RotatE; training here depends only on this trait.

use crate::Scorer;

/// A [`Scorer`] that can return the gradient of its score with respect to each
/// input embedding. All three returned vectors have length [`Scorer::dims`].
pub trait Differentiable: Scorer {
    /// Gradient of `score(s, r, o)` w.r.t. each input vector, returned as
    /// `(d_score/d_s, d_score/d_r, d_score/d_o)`. No side effects.
    fn grad(&self, s: &[f32], r: &[f32], o: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>);
}

#[cfg(test)]
pub(crate) mod testing {
    //! A trivially-differentiable DistMult-like scorer used across the train
    //! and eval tests so they run before the real scorers land. Not compiled
    //! into the shipped crate.
    use super::Differentiable;
    use crate::{Scorer, Side};

    /// `score(s, r, o) = sum_i s_i * r_i * o_i` — bilinear, symmetric in s/o,
    /// and fits the dot-product retrieval seam exactly.
    #[derive(Debug, Clone)]
    pub(crate) struct DistMult {
        dims: usize,
        id: String,
    }

    impl DistMult {
        pub(crate) fn new(dims: usize) -> Self {
            Self {
                dims,
                id: format!("distmult-test@d{dims}"),
            }
        }
    }

    impl Scorer for DistMult {
        fn dims(&self) -> usize {
            self.dims
        }
        fn score(&self, s: &[f32], r: &[f32], o: &[f32]) -> f32 {
            s.iter().zip(r).zip(o).map(|((&a, &b), &c)| a * b * c).sum()
        }
        fn query_vector(&self, r: &[f32], anchor: &[f32], _side: Side) -> Vec<f32> {
            // score = sum_i (r_i * anchor_i) * open_i, symmetric in the two
            // entity slots, so the query is r ∘ anchor regardless of side.
            r.iter().zip(anchor).map(|(&a, &b)| a * b).collect()
        }
        fn index_vector(&self, e: &[f32]) -> Vec<f32> {
            e.to_vec()
        }
        fn index_dims(&self) -> usize {
            self.dims
        }
        fn id(&self) -> &str {
            &self.id
        }
    }

    impl Differentiable for DistMult {
        fn grad(&self, s: &[f32], r: &[f32], o: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let ds: Vec<f32> = r.iter().zip(o).map(|(&b, &c)| b * c).collect();
            let dr: Vec<f32> = s.iter().zip(o).map(|(&a, &c)| a * c).collect();
            let do_: Vec<f32> = s.iter().zip(r).map(|(&a, &b)| a * b).collect();
            (ds, dr, do_)
        }
    }

    /// A scorer that returns the same value for every triple — used by the
    /// tie-breaking gate (ADR-006 §2).
    #[derive(Debug, Clone)]
    pub(crate) struct Constant {
        dims: usize,
        id: String,
    }
    impl Constant {
        pub(crate) fn new(dims: usize) -> Self {
            Self {
                dims,
                id: "constant-test".into(),
            }
        }
    }
    impl Scorer for Constant {
        fn dims(&self) -> usize {
            self.dims
        }
        fn score(&self, _s: &[f32], _r: &[f32], _o: &[f32]) -> f32 {
            0.0
        }
        fn query_vector(&self, _r: &[f32], anchor: &[f32], _side: Side) -> Vec<f32> {
            vec![0.0; anchor.len()]
        }
        fn index_vector(&self, e: &[f32]) -> Vec<f32> {
            vec![0.0; e.len()]
        }
        fn index_dims(&self) -> usize {
            self.dims
        }
        fn id(&self) -> &str {
            &self.id
        }
    }
}
