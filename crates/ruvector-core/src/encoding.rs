//! Unified vector-representation plane (ADR-297 §1).
//!
//! One interface — [`VectorCodec`] / [`EncodedQuery`] — for every way
//! RuVector can hold a vector: FP32, FP16, Int8, Turbo4 (and, from their own
//! crates, PQ and RaBitQ). Storage, indexes, snapshots, WASM, and bindings
//! consume `&dyn VectorCodec` and opaque blobs; they never name a concrete
//! codec. This is what lets precision become a per-vector, per-query
//! *decision* instead of a compile-time choice.
//!
//! Also home to [`VectorProvenance`] (ADR-297 §7): the metadata every stored
//! vector must carry so that migrations are auditable and snapshots stay
//! deterministic and independently verifiable.

pub mod codecs;

use crate::error::{Result, RuvectorError};
use crate::types::DistanceMetric;
use serde::{Deserialize, Serialize};

/// Every representation the plane knows about. `Pq` and `RaBitQ1` are
/// reserved here (stable blob-header / provenance values) but their codecs
/// live in `ruvector-pq-search` / `ruvector-rabitq` (ADR-297 phase C).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodecKind {
    /// 4 bytes/dim, exact.
    Fp32,
    /// 2 bytes/dim, ~1e-3 relative error. Hot-tier / verification codec.
    Fp16,
    /// 1 byte/dim + 8 B header (per-vector min/scale uniform quantization).
    Int8,
    /// 0.5 bytes/dim + 8 B constants (ADR-296 Lloyd-Max rotated codes).
    Turbo4,
    /// Product quantization (external crate; reserved).
    Pq,
    /// 1-bit RaBitQ (external crate; reserved).
    RaBitQ1,
}

impl CodecKind {
    /// Bytes per encoded vector at dimension `dim` (None for reserved kinds
    /// whose layout is owned by external crates).
    pub fn encoded_len(&self, dim: usize) -> Option<usize> {
        match self {
            CodecKind::Fp32 => Some(dim * 4),
            CodecKind::Fp16 => Some(dim * 2),
            CodecKind::Int8 => Some(dim + 8),
            CodecKind::Turbo4 => Some(dim / 2 + 8),
            CodecKind::Pq | CodecKind::RaBitQ1 => None,
        }
    }
}

/// A vector codec: encodes f32 vectors into opaque blobs and scores blobs
/// without the caller knowing the representation.
pub trait VectorCodec: Send + Sync {
    /// Which representation this codec produces.
    fn kind(&self) -> CodecKind;
    /// Dimensionality of vectors this codec accepts.
    fn dim(&self) -> usize;
    /// Bytes per encoded vector.
    fn encoded_len(&self) -> usize;
    /// Encode a vector. The blob is self-sufficient for scoring.
    fn encode(&self, v: &[f32]) -> Result<Vec<u8>>;
    /// Reconstruct an approximation (exact only for `Fp32`).
    fn decode(&self, blob: &[u8]) -> Result<Vec<f32>>;
    /// Symmetric blob×blob distance under this codec's metric.
    fn distance(&self, a: &[u8], b: &[u8]) -> Result<f32>;
    /// Prepare a query for repeated asymmetric scoring against blobs.
    fn make_query(&self, q: &[f32]) -> Result<Box<dyn EncodedQuery>>;
}

/// A prepared query bound to one codec + metric.
pub trait EncodedQuery: Send + Sync {
    /// Distance from the query to an encoded vector. Conventions match the
    /// index plane: Euclidean root, `1 − cos` (clamped ≥ 0), `−dot`
    /// (clamped ≥ 0), Manhattan sum.
    fn distance_to(&self, blob: &[u8]) -> f32;
}

/// Construct a codec by kind. The single entry point bindings and storage
/// use, so adding a codec never touches consumers.
pub fn codec_for(
    kind: CodecKind,
    dim: usize,
    metric: DistanceMetric,
    rotation_seed: u64,
) -> Result<Box<dyn VectorCodec>> {
    match kind {
        CodecKind::Fp32 => Ok(Box::new(codecs::Fp32Codec::new(dim, metric))),
        CodecKind::Fp16 => Ok(Box::new(codecs::Fp16Codec::new(dim, metric))),
        CodecKind::Int8 => Ok(Box::new(codecs::Int8Codec::new(dim, metric))),
        CodecKind::Turbo4 => Ok(Box::new(codecs::Turbo4PlaneCodec::new(
            dim,
            metric,
            rotation_seed,
        )?)),
        CodecKind::Pq | CodecKind::RaBitQ1 => Err(RuvectorError::InvalidParameter(format!(
            "{kind:?} codec is provided by its own crate and not yet wired \
             into the core plane (ADR-297 phase C)"
        ))),
    }
}

/// Provenance every stored vector carries (ADR-297 §7). Snapshots must be
/// reproducible from (source, provenance) alone — which is why rotation
/// seeds are recorded and codecs are versioned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorProvenance {
    /// Embedding model identity (e.g. "text-embedding-3-large").
    pub model_id: Option<String>,
    /// Representation of the stored blob.
    pub codec: CodecKind,
    /// Version of that codec's layout/tables. Bump on any change that alters
    /// encoded bytes for identical input.
    pub codec_version: u16,
    /// Rotation seed for rotated codecs (Turbo4, RaBitQ).
    pub rotation_seed: Option<u64>,
    /// Dimensionality.
    pub dim: usize,
    /// Distance metric the codes were prepared for.
    pub metric: DistanceMetric,
    /// Hex SHA-256 of the source object, when known.
    pub source_hash: Option<String>,
    /// Migration lineage, oldest first (e.g. "fp32→turbo4@2026-08-06").
    pub lineage: Vec<String>,
}

/// Current layout version of the in-core codecs.
pub const CODEC_VERSION: u16 = 1;

impl VectorProvenance {
    /// Provenance for a freshly encoded vector under `codec`.
    pub fn new(codec: &dyn VectorCodec, metric: DistanceMetric) -> Self {
        Self {
            model_id: None,
            codec: codec.kind(),
            codec_version: CODEC_VERSION,
            rotation_seed: None,
            dim: codec.dim(),
            metric,
            source_hash: None,
            lineage: Vec::new(),
        }
    }

    /// Record a migration step (old codec → new codec).
    pub fn push_migration(&mut self, entry: impl Into<String>) {
        self.lineage.push(entry.into());
    }
}

/// Shared scalar distance with the same conventions as the HNSW plane
/// (`index::hnsw::DistanceFn`): callers across codecs must agree on score
/// semantics or adaptive escalation would compare incomparable numbers.
pub(crate) fn metric_distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    use crate::simd_intrinsics as si;
    match metric {
        DistanceMetric::Euclidean => si::euclidean_distance_simd(a, b),
        DistanceMetric::Cosine => (1.0_f32 - si::cosine_similarity_simd(a, b)).max(0.0),
        DistanceMetric::DotProduct => (-si::dot_product_simd(a, b)).max(0.0),
        DistanceMetric::Manhattan => si::manhattan_distance_simd(a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoded_len_matches_kind_table() {
        for (kind, dim, expect) in [
            (CodecKind::Fp32, 128, 512),
            (CodecKind::Fp16, 128, 256),
            (CodecKind::Int8, 128, 136),
            (CodecKind::Turbo4, 128, 72),
        ] {
            assert_eq!(kind.encoded_len(dim), Some(expect));
            let codec = codec_for(kind, dim, DistanceMetric::Euclidean, 42).unwrap();
            assert_eq!(codec.encoded_len(), expect);
            assert_eq!(codec.kind(), kind);
        }
        assert_eq!(CodecKind::Pq.encoded_len(128), None);
        assert!(codec_for(CodecKind::Pq, 128, DistanceMetric::Euclidean, 42).is_err());
    }

    #[test]
    fn provenance_roundtrips_serde() {
        let codec = codec_for(CodecKind::Turbo4, 64, DistanceMetric::Cosine, 42).unwrap();
        let mut p = VectorProvenance::new(codec.as_ref(), DistanceMetric::Cosine);
        p.rotation_seed = Some(42);
        p.model_id = Some("test-model".into());
        p.push_migration("fp32→turbo4@2026-08-06");
        let json = serde_json::to_string(&p).unwrap();
        let back: VectorProvenance = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
        assert_eq!(back.lineage.len(), 1);
    }
}
