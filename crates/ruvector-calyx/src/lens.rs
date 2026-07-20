//! Lenses and their frozen, content-addressed manifests.
//!
//! A **lens** is one frozen way of measuring an object: a dense semantic
//! embedder, a sparse lexical index, a code model, a structural/domain encoder,
//! a temporal recurrence model, and so on.  Each lens produces one typed
//! **slot** vector.  Calyx's core invariant is that slots are kept *distinct*
//! and **never flattened** into a single vector — relationships are derived
//! *between* slots instead.
//!
//! Lenses are reproducible: a [`LensManifest`] records the measurement lineage
//! (name, version, kind, dimensionality, declared cost/latency) and is
//! content-addressed by a fingerprint hash so the same measurement can be
//! replayed and audited later (the paper's *Registry*).

use crate::scoring::fnv1a64;

/// The family a lens belongs to. Branding aside, the only thing the engine
/// needs from a lens is "what kind of signal is this and how big".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LensKind {
    /// Dense semantic embedding (topic / meaning).
    DenseSemantic,
    /// Sparse lexical / BM25-style surface-term signal (stored dense here).
    Lexical,
    /// Structural / domain / jurisdiction / code-shape signal.
    Structural,
    /// Temporal recurrence signal.
    Temporal,
    /// Non-text sensor signal (RF, audio, vibration, pose, ...).
    Sensor,
}

impl LensKind {
    pub fn as_str(self) -> &'static str {
        match self {
            LensKind::DenseSemantic => "dense_semantic",
            LensKind::Lexical => "lexical",
            LensKind::Structural => "structural",
            LensKind::Temporal => "temporal",
            LensKind::Sensor => "sensor",
        }
    }
}

/// Reproducible measurement lineage for one lens.
///
/// The `fingerprint` content-addresses the lens: identical
/// `(name, version, kind, dims)` always produce the same fingerprint, so a
/// retrieval path can be replayed against a byte-identical measurement set.
#[derive(Debug, Clone, PartialEq)]
pub struct LensManifest {
    /// Stable lens identifier (also the slot key on every constellation).
    pub name: String,
    /// Frozen model/version string.
    pub version: String,
    /// Signal family.
    pub kind: LensKind,
    /// Output dimensionality.
    pub dims: usize,
    /// Declared cost to measure one object, in microseconds (signal economics).
    pub cost_us: f32,
    /// Declared latency to measure one object, in microseconds.
    pub latency_us: f32,
    /// Content-addressed fingerprint over the lineage fields.
    pub fingerprint: u64,
}

impl LensManifest {
    /// Build a manifest, computing its content-addressed fingerprint.
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        kind: LensKind,
        dims: usize,
        cost_us: f32,
        latency_us: f32,
    ) -> Self {
        let name = name.into();
        let version = version.into();
        let fingerprint = Self::compute_fingerprint(&name, &version, kind, dims);
        Self {
            name,
            version,
            kind,
            dims,
            cost_us,
            latency_us,
            fingerprint,
        }
    }

    fn compute_fingerprint(name: &str, version: &str, kind: LensKind, dims: usize) -> u64 {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(version.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(kind.as_str().as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&(dims as u64).to_le_bytes());
        fnv1a64(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_is_content_addressed() {
        let a = LensManifest::new("semantic", "v1", LensKind::DenseSemantic, 48, 1.0, 1.0);
        let b = LensManifest::new("semantic", "v1", LensKind::DenseSemantic, 48, 9.0, 9.0);
        // Cost/latency are economics, not lineage — same fingerprint.
        assert_eq!(a.fingerprint, b.fingerprint);
        let c = LensManifest::new("semantic", "v2", LensKind::DenseSemantic, 48, 1.0, 1.0);
        assert_ne!(a.fingerprint, c.fingerprint);
        let d = LensManifest::new("semantic", "v1", LensKind::DenseSemantic, 64, 1.0, 1.0);
        assert_ne!(a.fingerprint, d.fingerprint);
    }
}
