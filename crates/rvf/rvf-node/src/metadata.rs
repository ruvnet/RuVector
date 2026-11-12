//! Per-vector and file-level metadata for the Node binding (issue #704).
//!
//! `ingestBatchWithMetadata` maps onto `RvfStore::ingest_batch_with_metadata`,
//! which commits each vector's record in the same META_SEG as the batch, so
//! the record survives close/reopen. Records are read back with
//! `getMetadata` and are attached to every query hit.

use napi::bindgen_prelude::*;
use napi::{Env, Task};
use napi_derive::napi;

use rvf_runtime::options::{
    MetadataEntry as RustMetadataEntry, MetadataValue as RustMetadataValue, VectorMetadata,
};
use rvf_runtime::RvfStore;

use crate::{map_rvf_err, parse_metadata_entry, RvfIngestResult, RvfMetadataEntry};

/// One vector's metadata record for `ingestBatchWithMetadata`.
#[napi(object)]
pub struct RvfVectorMetadata {
    /// The vector ID this record belongs to (must be in the same batch or
    /// already live in the store).
    pub id: i64,
    /// The record's fields. An empty array stores an empty record.
    pub fields: Vec<RvfMetadataEntry>,
}

/// Convert a stored value back into the binding's `{ valueType, value }`
/// string form. Bytes are hex-encoded.
pub(crate) fn entry_to_js(entry: &RustMetadataEntry) -> RvfMetadataEntry {
    let (value_type, value) = match &entry.value {
        RustMetadataValue::Null => ("null", String::new()),
        RustMetadataValue::U64(v) => ("u64", v.to_string()),
        RustMetadataValue::I64(v) => ("i64", v.to_string()),
        // `{:?}` keeps a round-trippable representation (e.g. `1.0`, `1e300`).
        RustMetadataValue::F64(v) => ("f64", format!("{v:?}")),
        RustMetadataValue::String(v) => ("string", v.clone()),
        RustMetadataValue::Bool(v) => ("bool", v.to_string()),
        RustMetadataValue::Bytes(v) => ("bytes", hex_encode(v)),
        RustMetadataValue::DeleteField => ("null", String::new()),
    };
    RvfMetadataEntry {
        field_id: u32::from(entry.field_id),
        value_type: value_type.to_string(),
        value,
    }
}

pub(crate) fn record_to_js(store: &RvfStore, id: u64) -> Option<Vec<RvfMetadataEntry>> {
    store
        .get_metadata(id)
        .map(|fields| fields.iter().map(entry_to_js).collect())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[usize::from(b >> 4)] as char);
        out.push(HEX[usize::from(b & 0x0f)] as char);
    }
    out
}

pub(crate) fn hex_decode(s: &str) -> Result<Vec<u8>> {
    let bad = || napi::Error::from_reason(format!("Cannot parse '{s}' as hex bytes"));
    if !s.len().is_multiple_of(2) {
        return Err(bad());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2).ok_or_else(bad)?, 16).map_err(|_| bad()))
        .collect()
}

fn split_vectors(vectors: &[f32], n: usize) -> Result<Vec<&[f32]>> {
    if n == 0 || !vectors.len().is_multiple_of(n) {
        return Err(napi::Error::from_reason(format!(
            "vectors length ({}) must be divisible by ids length ({n})",
            vectors.len()
        )));
    }
    let dim = vectors.len() / n;
    Ok((0..n).map(|i| &vectors[i * dim..(i + 1) * dim]).collect())
}

fn non_negative_id(id: i64) -> Result<u64> {
    u64::try_from(id)
        .map_err(|_| napi::Error::from_reason(format!("vector id {id} must be non-negative")))
}

pub(crate) fn run_ingest_with_metadata(
    store: &mut RvfStore,
    vectors: &[f32],
    ids: &[i64],
    records: &[RvfVectorMetadata],
) -> Result<RvfIngestResult> {
    if ids.is_empty() {
        return Ok(RvfIngestResult {
            accepted: 0,
            rejected: 0,
            epoch: 0,
        });
    }
    let vec_slices = split_vectors(vectors, ids.len())?;
    let rust_ids: Vec<u64> = ids.iter().map(|&id| id as u64).collect();
    let rust_records = records
        .iter()
        .map(|r| {
            Ok(VectorMetadata {
                vector_id: non_negative_id(r.id)?,
                fields: r
                    .fields
                    .iter()
                    .map(parse_metadata_entry)
                    .collect::<Result<_>>()?,
                delete_record: false,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let result = store
        .ingest_batch_with_metadata(&vec_slices, &rust_ids, &rust_records)
        .map_err(map_rvf_err)?;
    Ok(RvfIngestResult {
        accepted: result.accepted as i64,
        rejected: result.rejected as i64,
        epoch: result.epoch,
    })
}

/// `ingestBatchWithMetadataAsync` work item.
pub struct IngestWithMetadataTask {
    pub(crate) inner: crate::tasks::SharedStore,
    pub(crate) vectors: Vec<f32>,
    pub(crate) ids: Vec<i64>,
    pub(crate) records: Vec<RvfVectorMetadata>,
}

impl Task for IngestWithMetadataTask {
    type Output = RvfIngestResult;
    type JsValue = RvfIngestResult;

    fn compute(&mut self) -> Result<Self::Output> {
        crate::tasks::with_store_mut(&self.inner, |s| {
            run_ingest_with_metadata(s, &self.vectors, &self.ids, &self.records)
        })
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

/// File-level metadata values are exposed as strings; other stored types
/// are rendered with the same encoding as per-vector values.
pub(crate) fn file_value_to_js(value: &RustMetadataValue) -> String {
    match value {
        RustMetadataValue::String(v) => v.clone(),
        other => {
            entry_to_js(&RustMetadataEntry {
                field_id: 0,
                value: other.clone(),
            })
            .value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip() {
        let bytes = vec![0u8, 1, 0xab, 0xff];
        assert_eq!(hex_encode(&bytes), "0001abff");
        assert_eq!(hex_decode("0001abff").unwrap(), bytes);
        assert!(hex_decode("abc").is_err());
        assert!(hex_decode("zz").is_err());
    }

    #[test]
    fn f64_round_trips_through_string() {
        for v in [0.0, 1.0, -2.5, 1e300, f64::MIN_POSITIVE, 0.1 + 0.2] {
            let js = entry_to_js(&RustMetadataEntry {
                field_id: 3,
                value: RustMetadataValue::F64(v),
            });
            assert_eq!(js.value_type, "f64");
            assert_eq!(js.value.parse::<f64>().unwrap(), v);
        }
    }
}
