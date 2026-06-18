//! Resonant identity layer: continuity recognition for physical objects.
//!
//! Every physical object emits a *resonant response signature* — a vibration,
//! acoustic, or RF-reflection embedding that depends on its mass, geometry,
//! material, fastening, and contents. This layer does not ask *"what is this?"*;
//! it asks *"is this STILL the same physical thing?"*
//!
//! By enrolling a known signature and comparing fresh observations against it,
//! we detect **identity drift**: a panel has loosened, a pipe has filled with
//! water, a bearing has worn, a casing has been tampered with. Small, gradual
//! changes (aging, ambient noise) are absorbed by an exponentially-weighted
//! moving average (EWMA) so the stored signature tracks slow drift, while a
//! sudden large change trips the `changed` flag.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The result of comparing a fresh signature against an enrolled one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityDrift {
    /// The object identifier this observation pertains to.
    pub id: String,
    /// Cosine distance (1 - cosine similarity) clamped to `[0, 1]`.
    pub drift: f32,
    /// Whether the drift exceeded the configured threshold (identity changed).
    pub changed: bool,
}

/// A trusted memory of resonant signatures keyed by object identity.
///
/// Stores one EWMA-smoothed signature per enrolled object. Observations that
/// stay within the drift threshold slowly update the stored signature; large
/// jumps are flagged and left to update the memory (the stored signature is
/// preserved so a transient tamper does not poison the baseline).
#[derive(Debug, Clone)]
pub struct IdentityMemory {
    signatures: HashMap<String, Vec<f32>>,
    drift_threshold: f32,
    alpha: f32,
}

impl IdentityMemory {
    /// Create an empty identity memory.
    ///
    /// `drift_threshold` in `[0, 1]`: cosine-distance above which identity is
    /// considered changed. `alpha` in `[0, 1]`: EWMA update rate for the stored
    /// signature when identity is unchanged (higher = faster adaptation). Both
    /// are clamped to `[0, 1]` defensively.
    pub fn new(drift_threshold: f32, alpha: f32) -> Self {
        Self {
            signatures: HashMap::new(),
            drift_threshold: drift_threshold.clamp(0.0, 1.0),
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Enroll (or overwrite) a known object's resonant signature embedding.
    pub fn enroll(&mut self, id: impl Into<String>, signature: Vec<f32>) {
        self.signatures.insert(id.into(), signature);
    }

    /// Whether an id is enrolled.
    pub fn contains(&self, id: &str) -> bool {
        self.signatures.contains_key(id)
    }

    /// Compare a fresh signature to the stored one.
    ///
    /// Returns `drift` = cosine distance (1 - cosine similarity) clamped to
    /// `[0, 1]`, and `changed` = `drift > threshold`. If unchanged, the stored
    /// signature is EWMA-updated (slow adaptation to aging/noise). If the id is
    /// unknown, the signature is auto-enrolled and `drift = 0.0`,
    /// `changed = false` is returned. A length mismatch against the stored
    /// signature is treated as a change (`drift = 1.0`) without updating.
    pub fn observe(&mut self, id: &str, signature: &[f32]) -> IdentityDrift {
        let Some(stored) = self.signatures.get(id) else {
            self.signatures.insert(id.to_string(), signature.to_vec());
            return IdentityDrift {
                id: id.to_string(),
                drift: 0.0,
                changed: false,
            };
        };

        if stored.len() != signature.len() {
            return IdentityDrift {
                id: id.to_string(),
                drift: 1.0,
                changed: true,
            };
        }

        let drift = cosine_distance(stored, signature);
        let changed = drift > self.drift_threshold;

        if !changed {
            let alpha = self.alpha;
            // Update in place: stored = (1 - alpha) * stored + alpha * signature.
            if let Some(stored_mut) = self.signatures.get_mut(id) {
                for (s, &fresh) in stored_mut.iter_mut().zip(signature.iter()) {
                    *s = (1.0 - alpha) * *s + alpha * fresh;
                }
            }
        }

        IdentityDrift {
            id: id.to_string(),
            drift,
            changed,
        }
    }
}

/// Cosine distance `1 - cos_sim`, clamped to `[0, 1]`.
///
/// Guards zero norms: if either vector has (near-)zero norm, distance is `1.0`
/// when the other vector is non-zero, else `0.0` (both effectively silent =
/// indistinguishable).
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let norm_a = dot(a, a).sqrt();
    let norm_b = dot(b, b).sqrt();
    const EPS: f32 = 1e-12;

    let a_zero = norm_a <= EPS;
    let b_zero = norm_b <= EPS;
    if a_zero || b_zero {
        return if a_zero && b_zero { 0.0 } else { 1.0 };
    }

    let cos_sim = dot(a, b) / (norm_a * norm_b);
    (1.0 - cos_sim).clamp(0.0, 1.0)
}

/// Dot product of two equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_identical_signature_is_unchanged() {
        let mut mem = IdentityMemory::new(0.1, 0.2);
        mem.enroll("pump-7", vec![1.0, 2.0, 3.0, 4.0]);

        // Tiny perturbation (sensor noise): same physical object.
        let result = mem.observe("pump-7", &[1.01, 1.99, 3.02, 3.98]);

        assert!(
            !result.changed,
            "near-identical signature should be unchanged"
        );
        assert!(
            result.drift < 0.1,
            "drift should be low, got {}",
            result.drift
        );
        assert_eq!(result.id, "pump-7");
    }

    #[test]
    fn large_change_trips_changed() {
        let mut mem = IdentityMemory::new(0.2, 0.2);
        mem.enroll("panel-3", vec![1.0, 0.0, 0.0, 0.0]);

        // Orthogonal signature — panel loosened, resonance shifted entirely.
        let result = mem.observe("panel-3", &[0.0, 1.0, 0.0, 0.0]);

        assert!(result.changed, "orthogonal signature should be a change");
        assert!(
            result.drift > 0.2,
            "drift should be high, got {}",
            result.drift
        );
        // Cosine distance of orthogonal vectors is exactly 1.0.
        assert!((result.drift - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unknown_id_auto_enrolls() {
        let mut mem = IdentityMemory::new(0.1, 0.2);
        assert!(!mem.contains("valve-1"));

        let result = mem.observe("valve-1", &[0.5, 0.5, 0.5]);

        assert!(!result.changed);
        assert_eq!(result.drift, 0.0);
        assert!(
            mem.contains("valve-1"),
            "observing unknown id should enroll it"
        );
    }

    #[test]
    fn length_mismatch_is_a_change() {
        let mut mem = IdentityMemory::new(0.1, 0.2);
        mem.enroll("bearing-2", vec![1.0, 2.0, 3.0]);

        let result = mem.observe("bearing-2", &[1.0, 2.0]);

        assert!(result.changed);
        assert_eq!(result.drift, 1.0);
    }

    #[test]
    fn gradual_drift_absorbed_then_sudden_change_trips() {
        let mut mem = IdentityMemory::new(0.15, 0.3);
        mem.enroll("casing-9", vec![1.0, 1.0, 1.0, 1.0]);

        // A slow walk of small perturbations: each step is tiny relative to the
        // current baseline, so EWMA absorbs it and identity stays the same.
        let mut current = vec![1.0_f32, 1.0, 1.0, 1.0];
        for step in 0..20 {
            let nudge = (step as f32) * 0.01;
            current = vec![
                1.0 + nudge,
                1.0 - nudge * 0.5,
                1.0 + nudge * 0.3,
                1.0 - nudge * 0.2,
            ];
            let r = mem.observe("casing-9", &current);
            assert!(
                !r.changed,
                "gradual step {step} should stay unchanged (drift {})",
                r.drift
            );
        }

        // Sudden large change — casing tampered: resonance inverts.
        let tampered = vec![-1.0, -1.0, -1.0, -1.0];
        let r = mem.observe("casing-9", &tampered);
        assert!(
            r.changed,
            "sudden inversion should trip changed (drift {})",
            r.drift
        );
        assert!(r.drift > 0.15);
    }
}
