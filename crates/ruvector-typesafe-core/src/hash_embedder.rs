//! Deterministic hashed bag-of-words embedder (feature `hash-embedder`).
//! Exists so heads, calibration, bindings and CI can be tested without ONNX
//! weights. It is a *test double*: `id()` says so, and the engine refuses to
//! report `calibrated: true` for it.

use crate::embedder::{l2_normalize, Embedder};
use crate::Result;

pub struct HashEmbedder {
    dims: usize,
    id: String,
}

impl HashEmbedder {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            id: format!("hash-bow-{dims}@test-double"),
        }
    }
}

fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl Embedder for HashEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; self.dims];
                for tok in t
                    .to_lowercase()
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|w| w.len() > 1)
                {
                    let h = fnv1a(tok);
                    let idx = (h % self.dims as u64) as usize;
                    let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
                    v[idx] += sign;
                }
                l2_normalize(&mut v);
                v
            })
            .collect())
    }
    fn dims(&self) -> usize {
        self.dims
    }
    fn id(&self) -> &str {
        &self.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::dot;

    #[test]
    fn is_deterministic_and_normalised() {
        let e = HashEmbedder::new(64);
        let a = e.embed(&["refund my duplicate charge"]).unwrap();
        let b = e.embed(&["refund my duplicate charge"]).unwrap();
        assert_eq!(a, b);
        assert!((dot(&a[0], &a[0]) - 1.0).abs() < 1e-5);
    }
}
