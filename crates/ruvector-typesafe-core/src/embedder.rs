//! The one seam between the engine and an inference backend (ADR-002 §1).

use crate::Result;

pub trait Embedder: Send + Sync {
    /// Embed a batch of texts. Each vector has `dims()` entries and is
    /// L2-normalised so cosine similarity is a dot product.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dims(&self) -> usize;
    /// `model-name@manifest-hash` — recorded in every receipt.
    fn id(&self) -> &str;
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
