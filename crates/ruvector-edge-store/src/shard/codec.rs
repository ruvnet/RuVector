//! Row and op-log encodings, and the shard `meta` configuration rows.

use crate::distance::Metric;
use crate::ports::StoreError;
use serde_json::{Map, Value as Json};

/// f32 slab row → little-endian bytes (`vectors.f32`).
pub fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Inverse of [`encode_f32`]; the length must be exactly `dim * 4`.
pub fn decode_f32(bytes: &[u8], dim: usize) -> Result<Vec<f32>, StoreError> {
    if bytes.len() != dim.checked_mul(4).ok_or(StoreError::Corrupt("dim"))? {
        return Err(StoreError::Corrupt("vectors.f32 length"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Upsert op body: `dim u32 LE ‖ f32 LE × dim ‖ metadata JSON (may be empty)`.
pub fn encode_upsert_body(values: &[f32], meta_text: Option<&str>) -> Vec<u8> {
    let dim = u32::try_from(values.len()).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(4 + values.len() * 4 + meta_text.map_or(0, str::len));
    out.extend_from_slice(&dim.to_le_bytes());
    out.extend_from_slice(&encode_f32(values));
    if let Some(m) = meta_text {
        out.extend_from_slice(m.as_bytes());
    }
    out
}

/// Decoded upsert op body.
pub type UpsertBody = (Vec<f32>, Option<String>);

/// Inverse of [`encode_upsert_body`], checked against the shard dimension.
pub fn decode_upsert_body(body: &[u8], dim: usize) -> Result<UpsertBody, StoreError> {
    let head: [u8; 4] = body
        .get(..4)
        .and_then(|h| h.try_into().ok())
        .ok_or(StoreError::Corrupt("ops.body header"))?;
    if u32::from_le_bytes(head) as usize != dim {
        return Err(StoreError::Corrupt("ops.body dim"));
    }
    let end = 4 + dim * 4;
    let vals = decode_f32(
        body.get(4..end).ok_or(StoreError::Corrupt("ops.body"))?,
        dim,
    )?;
    let rest = &body[end..];
    let meta = if rest.is_empty() {
        None
    } else {
        Some(
            core::str::from_utf8(rest)
                .map_err(|_| StoreError::Corrupt("ops.body metadata"))?
                .to_string(),
        )
    };
    Ok((vals, meta))
}

/// Parse stored metadata text into an object.
pub fn parse_meta(text: &str) -> Result<Map<String, Json>, StoreError> {
    match serde_json::from_str::<Json>(text) {
        Ok(Json::Object(m)) => Ok(m),
        _ => Err(StoreError::Corrupt("metadata")),
    }
}

/// `meta` keys for the shard configuration and counters.
pub mod keys {
    /// Dimension.
    pub const DIM: &str = "dim";
    /// Metric wire name.
    pub const METRIC: &str = "metric";
    /// Declared filterable keys (JSON array).
    pub const FILTERABLE: &str = "filterable_keys";
    /// Per-shard float cap.
    pub const FLOAT_CAP: &str = "float_cap";
    /// Index kind (`flat` at M1).
    pub const INDEX_KIND: &str = "index_kind";
    /// Schema version.
    pub const SCHEMA_VER: &str = "schema_ver";
    /// Last applied op sequence.
    pub const WRITE_SEQ: &str = "write_seq";
    /// Next internal id.
    pub const NEXT_IID: &str = "next_iid";
    /// Highest op-log sequence pruned (ops at or below it are gone; the
    /// `vectors` table is the snapshot they were folded into).
    pub const SNAPSHOT_SEQ: &str = "snapshot_seq";
    /// Index epoch (M2b index-chunk generation; constant 0 for `flat`).
    pub const INDEX_EPOCH: &str = "index_epoch";
}

/// Current shard schema version.
pub const SCHEMA_VERSION: &str = "1";

/// Stored configuration of a shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardConfig {
    /// Fixed dimension, `1..=1536`.
    pub dim: u32,
    /// Metric.
    pub metric: Metric,
    /// Declared filterable metadata keys (≤ 8).
    pub filterable_keys: Vec<String>,
    /// Resident float cap for this shard (M1: 3,000,000).
    pub float_cap: u64,
}

impl ShardConfig {
    /// `meta` rows for this configuration.
    pub fn to_kv(&self) -> Vec<(&'static str, String)> {
        vec![
            (keys::DIM, self.dim.to_string()),
            (keys::METRIC, self.metric.as_str().to_string()),
            (
                keys::FILTERABLE,
                serde_json::to_string(&self.filterable_keys).unwrap_or_else(|_| "[]".into()),
            ),
            (keys::FLOAT_CAP, self.float_cap.to_string()),
            (keys::INDEX_KIND, "flat".to_string()),
            (keys::SCHEMA_VER, SCHEMA_VERSION.to_string()),
        ]
    }

    /// Read the configuration from `meta` pairs; `Ok(None)` if absent.
    pub fn from_kv(kv: &[(String, String)]) -> Result<Option<Self>, StoreError> {
        let get = |k: &str| kv.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str());
        let Some(dim) = get(keys::DIM) else {
            return Ok(None);
        };
        let dim: u32 = dim.parse().map_err(|_| StoreError::Corrupt("meta.dim"))?;
        let metric = get(keys::METRIC)
            .and_then(Metric::parse)
            .ok_or(StoreError::Corrupt("meta.metric"))?;
        let filterable_keys: Vec<String> = get(keys::FILTERABLE)
            .and_then(|s| serde_json::from_str(s).ok())
            .ok_or(StoreError::Corrupt("meta.filterable_keys"))?;
        let float_cap = get(keys::FLOAT_CAP)
            .and_then(|s| s.parse().ok())
            .ok_or(StoreError::Corrupt("meta.float_cap"))?;
        if !(1..=1536).contains(&dim) || filterable_keys.len() > crate::filter::MAX_FILTERABLE_KEYS
        {
            return Err(StoreError::Corrupt("meta config range"));
        }
        Ok(Some(ShardConfig {
            dim,
            metric,
            filterable_keys,
            float_cap,
        }))
    }
}

/// Parse an optional `u64` counter from `meta`, defaulting to `default`.
pub fn counter(kv: &[(String, String)], key: &str, default: u64) -> Result<u64, StoreError> {
    match kv.iter().find(|(k, _)| k == key) {
        None => Ok(default),
        Some((_, v)) => v.parse().map_err(|_| StoreError::Corrupt("meta counter")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_round_trip_and_corruption() {
        let v = [1.5f32, -2.0, 0.25];
        let b = encode_upsert_body(&v, Some("{\"a\":1}"));
        assert_eq!(
            decode_upsert_body(&b, 3).unwrap(),
            (v.to_vec(), Some("{\"a\":1}".into()))
        );
        let b2 = encode_upsert_body(&v, None);
        assert_eq!(decode_upsert_body(&b2, 3).unwrap().1, None);
        assert!(decode_upsert_body(&b2, 4).is_err());
        assert!(decode_upsert_body(&b2[..7], 3).is_err());
        assert!(decode_f32(&[0; 5], 1).is_err());
    }
}
