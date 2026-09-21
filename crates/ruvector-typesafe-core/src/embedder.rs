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

/// Blanket impls so a runtime-chosen backend (hash vs ONNX) can be held behind
/// a trait object and still satisfy `Engine<E: Embedder>` without a wrapper
/// type: `Engine::new(boxed)` just works. `dyn Embedder` carries the trait's
/// `Send + Sync` supertraits, so the smart pointers stay `Send + Sync` too.
impl Embedder for Box<dyn Embedder> {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (**self).embed(texts)
    }
    fn dims(&self) -> usize {
        (**self).dims()
    }
    fn id(&self) -> &str {
        (**self).id()
    }
}

impl Embedder for std::sync::Arc<dyn Embedder> {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        (**self).embed(texts)
    }
    fn dims(&self) -> usize {
        (**self).dims()
    }
    fn id(&self) -> &str {
        (**self).id()
    }
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
