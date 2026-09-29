//! Row and op-log encodings, and the shard `meta` configuration rows.

use crate::distance::Metric;
use crate::ports::StoreError;
use serde::{Deserialize, Serialize};
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

/// High bit of the upsert body header: v2 (an `iid i64 LE` follows).
const BODY_V2: u32 = 0x8000_0000;

/// Upsert op body. v2 (written since M2):
/// `(0x8000_0000 | dim) u32 LE ‖ iid i64 LE ‖ f32 LE × dim ‖ metadata JSON`;
/// v1 (M1) had no iid: `dim u32 LE ‖ f32 LE × dim ‖ metadata JSON`.
pub fn encode_upsert_body(iid: i64, values: &[f32], meta_text: Option<&str>) -> Vec<u8> {
    let dim = u32::try_from(values.len()).unwrap_or(0) & !BODY_V2;
    let mut out = Vec::with_capacity(12 + values.len() * 4 + meta_text.map_or(0, str::len));
    out.extend_from_slice(&(BODY_V2 | dim).to_le_bytes());
    out.extend_from_slice(&iid.to_le_bytes());
    out.extend_from_slice(&encode_f32(values));
    if let Some(m) = meta_text {
        out.extend_from_slice(m.as_bytes());
    }
    out
}

/// Decoded upsert op body: `(iid if v2, values, metadata)`.
pub type UpsertBody = (Option<i64>, Vec<f32>, Option<String>);

/// Inverse of [`encode_upsert_body`] (both versions), checked against the
/// shard dimension.
pub fn decode_upsert_body(body: &[u8], dim: usize) -> Result<UpsertBody, StoreError> {
    let head: [u8; 4] = body
        .get(..4)
        .and_then(|h| h.try_into().ok())
        .ok_or(StoreError::Corrupt("ops.body header"))?;
    let head = u32::from_le_bytes(head);
    let (iid, at) = if head & BODY_V2 != 0 {
        let b: [u8; 8] = body
            .get(4..12)
            .and_then(|h| h.try_into().ok())
            .ok_or(StoreError::Corrupt("ops.body iid"))?;
        (Some(i64::from_le_bytes(b)), 12)
    } else {
        (None, 4)
    };
    if (head & !BODY_V2) as usize != dim {
        return Err(StoreError::Corrupt("ops.body dim"));
    }
    let end = at + dim * 4;
    let vals = decode_f32(
        body.get(at..end).ok_or(StoreError::Corrupt("ops.body"))?,
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
    Ok((iid, vals, meta))
}

/// Delete op body (v2): the deleted row's `iid i64 LE`. M1 wrote `NULL`.
pub fn encode_delete_body(iid: i64) -> Vec<u8> {
    iid.to_le_bytes().to_vec()
}

/// Inverse of [`encode_delete_body`]; `None` for an M1 (`NULL`) body.
pub fn decode_delete_body(body: Option<&[u8]>) -> Result<Option<i64>, StoreError> {
    match body {
        None => Ok(None),
        Some(b) => {
            let a: [u8; 8] = b
                .try_into()
                .map_err(|_| StoreError::Corrupt("ops.body delete"))?;
            Ok(Some(i64::from_le_bytes(a)))
        }
    }
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
    /// Index kind (`flat` | `hnsw`).
    pub const INDEX_KIND: &str = "index_kind";
    /// HNSW `m` (links per upper-layer node; `m0 = 2m`).
    pub const HNSW_M: &str = "hnsw_m";
    /// HNSW `ef_construction`.
    pub const HNSW_EFC: &str = "hnsw_ef_construction";
    /// Schema version.
    pub const SCHEMA_VER: &str = "schema_ver";
    /// Last applied op sequence.
    pub const WRITE_SEQ: &str = "write_seq";
    /// Next internal id.
    pub const NEXT_IID: &str = "next_iid";
    /// Highest op-log sequence pruned (ops at or below it are gone; the
    /// `vectors` table is the snapshot they were folded into).
    pub const SNAPSHOT_SEQ: &str = "snapshot_seq";
    /// Latest persisted index epoch (`index_chunks.epoch`).
    pub const INDEX_EPOCH: &str = "index_epoch";
    /// Persisted index epochs (JSON, see `shard::persist`).
    pub const INDEX_STATE: &str = "index_state";
    /// Quantizer parameters (JSON, see `shard::ann`).
    pub const QUANT: &str = "quant";
    /// Set by a collection drop's wipe: the DO answers `404` forever.
    pub const WIPED: &str = "wiped";
}

/// Current shard schema version (2: M2 index, v2 op bodies).
pub const SCHEMA_VERSION: &str = "2";

/// Default HNSW `m` (ADR §7 range `8..=48`).
pub const HNSW_M_DEFAULT: u16 = 16;
/// Default HNSW `ef_construction` (ADR §7 range `32..=200`).
pub const HNSW_EFC_DEFAULT: u16 = 128;

/// Collection index kind (ADR §7 `index`). `flat` (default) is an int8
/// quantized scan with an exact f32 rerank from SQLite; `hnsw` traverses
/// the same codes through a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum IndexConfig {
    /// Quantized flat scan + rerank.
    #[default]
    Flat,
    /// HNSW over codes + rerank.
    Hnsw {
        /// Links per upper-layer node (`8..=48`); layer 0 holds `2m`.
        m: u16,
        /// Construction beam (`32..=200`).
        ef_construction: u16,
    },
}

impl IndexConfig {
    /// Default HNSW parameters.
    pub const HNSW_DEFAULT: IndexConfig = IndexConfig::Hnsw {
        m: HNSW_M_DEFAULT,
        ef_construction: HNSW_EFC_DEFAULT,
    };

    /// Wire name of the kind.
    pub fn kind_str(self) -> &'static str {
        match self {
            IndexConfig::Flat => "flat",
            IndexConfig::Hnsw { .. } => "hnsw",
        }
    }

    /// `true` when the parameters are inside the ADR §7 ranges.
    pub fn in_range(self) -> bool {
        match self {
            IndexConfig::Flat => true,
            IndexConfig::Hnsw { m, ef_construction } => {
                (8..=48).contains(&m) && (32..=200).contains(&ef_construction)
            }
        }
    }

    /// Catalog `kind` column form: `flat` or `hnsw:<m>:<ef_construction>`.
    pub fn to_catalog(self) -> String {
        match self {
            IndexConfig::Flat => "flat".into(),
            IndexConfig::Hnsw { m, ef_construction } => format!("hnsw:{m}:{ef_construction}"),
        }
    }

    /// Inverse of [`IndexConfig::to_catalog`] (bare `hnsw` = defaults).
    pub fn from_catalog(s: &str) -> Option<IndexConfig> {
        let mut it = s.split(':');
        let cfg = match (it.next()?, it.next(), it.next()) {
            ("flat", None, None) => IndexConfig::Flat,
            ("hnsw", None, None) => IndexConfig::HNSW_DEFAULT,
            ("hnsw", Some(m), Some(e)) => IndexConfig::Hnsw {
                m: m.parse().ok()?,
                ef_construction: e.parse().ok()?,
            },
            _ => return None,
        };
        (it.next().is_none() && cfg.in_range()).then_some(cfg)
    }
}

/// Stored configuration of a shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardConfig {
    /// Fixed dimension, `1..=1536`.
    pub dim: u32,
    /// Metric.
    pub metric: Metric,
    /// Declared filterable metadata keys (≤ 8).
    pub filterable_keys: Vec<String>,
    /// Stored float cap for this shard (f32 rows live in SQLite at M2; the
    /// resident limit is `SHARD_RESIDENT_CAP_BYTES`).
    pub float_cap: u64,
    /// Index kind.
    pub index: IndexConfig,
}

impl ShardConfig {
    /// `meta` rows for this configuration.
    pub fn to_kv(&self) -> Vec<(&'static str, String)> {
        let mut kv = vec![
            (keys::DIM, self.dim.to_string()),
            (keys::METRIC, self.metric.as_str().to_string()),
            (
                keys::FILTERABLE,
                serde_json::to_string(&self.filterable_keys).unwrap_or_else(|_| "[]".into()),
            ),
            (keys::FLOAT_CAP, self.float_cap.to_string()),
            (keys::INDEX_KIND, self.index.kind_str().to_string()),
            (keys::SCHEMA_VER, SCHEMA_VERSION.to_string()),
        ];
        if let IndexConfig::Hnsw { m, ef_construction } = self.index {
            kv.push((keys::HNSW_M, m.to_string()));
            kv.push((keys::HNSW_EFC, ef_construction.to_string()));
        }
        kv
    }

    /// Read the configuration from `meta` pairs; `Ok(None)` if absent. An
    /// M1 shard (`index_kind = flat`, schema 1) reads as the M2 `flat`.
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
        let num = |k| get(k).and_then(|s| s.parse::<u16>().ok());
        let index = match get(keys::INDEX_KIND).unwrap_or("flat") {
            "flat" => IndexConfig::Flat,
            "hnsw" => IndexConfig::Hnsw {
                m: num(keys::HNSW_M).ok_or(StoreError::Corrupt("meta.hnsw_m"))?,
                ef_construction: num(keys::HNSW_EFC).ok_or(StoreError::Corrupt("meta.hnsw_efc"))?,
            },
            _ => return Err(StoreError::Corrupt("meta.index_kind")),
        };
        if !(1..=1536).contains(&dim)
            || filterable_keys.len() > crate::filter::MAX_FILTERABLE_KEYS
            || !index.in_range()
        {
            return Err(StoreError::Corrupt("meta config range"));
        }
        Ok(Some(ShardConfig {
            dim,
            metric,
            filterable_keys,
            float_cap,
            index,
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
        let b = encode_upsert_body(7, &v, Some("{\"a\":1}"));
        assert_eq!(
            decode_upsert_body(&b, 3).unwrap(),
            (Some(7), v.to_vec(), Some("{\"a\":1}".into()))
        );
        let b2 = encode_upsert_body(9, &v, None);
        assert_eq!(decode_upsert_body(&b2, 3).unwrap().2, None);
        assert!(decode_upsert_body(&b2, 4).is_err());
        assert!(decode_upsert_body(&b2[..11], 3).is_err());
        assert!(decode_f32(&[0; 5], 1).is_err());
        // An M1 (v1) body still decodes, without an iid.
        let mut v1 = 3u32.to_le_bytes().to_vec();
        v1.extend_from_slice(&encode_f32(&v));
        assert_eq!(
            decode_upsert_body(&v1, 3).unwrap(),
            (None, v.to_vec(), None)
        );
        assert_eq!(
            decode_delete_body(Some(&encode_delete_body(5))),
            Ok(Some(5))
        );
        assert_eq!(decode_delete_body(None), Ok(None));
        assert!(decode_delete_body(Some(&[1, 2])).is_err());
    }

    #[test]
    fn index_config_forms() {
        for c in [IndexConfig::Flat, IndexConfig::HNSW_DEFAULT] {
            assert_eq!(IndexConfig::from_catalog(&c.to_catalog()), Some(c));
        }
        assert_eq!(
            IndexConfig::from_catalog("hnsw"),
            Some(IndexConfig::HNSW_DEFAULT)
        );
        for bad in ["hnsw:4:128", "hnsw:16:500", "q8", "flat:1", "hnsw:16"] {
            assert_eq!(IndexConfig::from_catalog(bad), None, "{bad}");
        }
    }
}
