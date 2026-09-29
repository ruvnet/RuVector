//! The batch sink of the `ruvector-edge-ingest` consumer (split out of
//! `ingest`): where a queued import's batches go, the submitter's
//! re-authorized caller context, and which sink errors fail the values: vec![v; dim as usize],ob.

use crate::backend::Backend;
use crate::idem::{self, Seen, Slot};
use crate::jobs::Job;
use crate::service::{self, Call};
use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_snapshot::{FailCode, Row};
use ruvector_edge_store::filter::MAX_METADATA_BYTES;
use ruvector_edge_store::{CallerContext, ErrorCode, Op, OpError};
use ruvector_edge_tenancy::validate::{MAX_COLLECTION_NAME_LEN, MAX_VECTOR_ID_BYTES};
use ruvector_edge_tenancy::TenantKey;
use serde::Serialize;

/// Largest internal upsert body a queued-import batch may encode: the
/// 1 MiB body a client upsert may send (`api::MAX_BODY_BYTES`), so the
/// `VectorShard` path never parses more than it does for a client.
pub const SINK_BODY_MAX: u64 = crate::api::MAX_BODY_BYTES as u64;
/// Worst-case JSON bytes of one value: `f32` serialised by `ryu` (shortest
/// round-trip, positional up to 13 integer digits, so ≤ 16 chars such as
/// `-9985025000000.0`; `f32_text_never_exceeds_the_per_float_allowance`
/// sweeps the bit patterns) plus its comma.
pub const JSON_BYTES_PER_FLOAT: u64 = 17;
/// Fixed JSON of one row: `{"id":"","values":[],"metadata":null},`.
const ROW_OVERHEAD: u64 = 48;
/// `{"collection":"<name>","vectors":[]}` with an escaped 63-byte name.
const ENVELOPE: u64 = 32 + 2 * MAX_COLLECTION_NAME_LEN as u64;

/// Worst-case encoded bytes of one `dim`-dimensional row: every value at
/// [`JSON_BYTES_PER_FLOAT`], a 256-byte id fully escaped (ids carry no
/// control characters, so ≤ 2×), metadata at its 4 KiB serialised cap
/// (re-checked by [`upsert_args`] before encoding).
pub fn worst_row_bytes(dim: u32) -> u64 {
    u64::from(dim) * JSON_BYTES_PER_FLOAT
        + MAX_METADATA_BYTES as u64
        + 2 * MAX_VECTOR_ID_BYTES as u64
        + ROW_OVERHEAD
}

/// Rows of a `dim`-dimensional batch whose body is ≤ [`SINK_BODY_MAX`]
/// whatever their ids, values and metadata (at least 1).
pub fn rows_within_body(dim: u32) -> u64 {
    ((SINK_BODY_MAX - ENVELOPE) / worst_row_bytes(dim)).max(1)
}

#[derive(Serialize)]
struct SinkRow<'a> {
    id: &'a str,
    values: &'a [f32],
    metadata: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct SinkArgs<'a> {
    collection: &'a str,
    vectors: Vec<SinkRow<'a>>,
}

/// The `vectors:upsert` arguments of one batch. Values keep their `f32`
/// text (never widened through `serde_json::Value`, which would print up
/// to 17 digits), and metadata over the store's serialised cap is refused
/// here (`413`, as the store would) so [`worst_row_bytes`] is a bound.
pub(crate) fn upsert_args(collection: &str, rows: &[Row]) -> Result<String, OpError> {
    let mut vectors = Vec::with_capacity(rows.len());
    for r in rows {
        let metadata = match &r.metadata {
            None => None,
            Some(t) => {
                let v = serde_json::from_str::<serde_json::Value>(t)
                    .map_err(|_| OpError::invalid("row metadata"))?;
                ruvector_edge_store::filter::validate_metadata(&v)?;
                Some(v)
            }
        };
        vectors.push(SinkRow {
            id: &r.id,
            values: &r.values,
            metadata,
        });
    }
    serde_json::to_string(&SinkArgs {
        collection,
        vectors,
    })
    .map_err(|_| service::unexpected())
}

/// Where batches go.
#[allow(async_fn_in_trait)]
pub trait BatchSink {
    /// Apply `rows` under `op_id` (idempotent per `op_id`).
    async fn apply(&self, job: &Job, op_id: &str, rows: Vec<Row>) -> Result<(), OpError>;
}

/// The production sink: the M1 upsert executor, re-authorized as the
/// submitter (scope `write` ∩ current role) and deduplicated through the
/// ledger's `op_id` table under `import:{op_id}`.
pub struct UpsertSink<'a, B> {
    /// DO transport.
    pub b: &'a B,
    /// Unix seconds.
    pub now: u64,
}

pub(crate) fn submitter(j: &Job) -> Result<CallerContext, OpError> {
    let t = TenantKey::parse(&j.job.tenant_key).map_err(|_| OpError::invalid("tenant"))?;
    let mut caps = CapabilitySet::EMPTY;
    caps.insert(Capability::Write);
    let m = &j.meta;
    Ok(CallerContext::new(
        t,
        m.sub.clone(),
        m.client_id.clone(),
        m.jti.clone(),
        m.family_id.clone(),
        m.act_sub.clone(),
        caps,
    ))
}

impl<B: Backend> BatchSink for UpsertSink<'_, B> {
    async fn apply(&self, job: &Job, op_id: &str, rows: Vec<Row>) -> Result<(), OpError> {
        use sha2::{Digest, Sha256};
        let ctx = submitter(job)?;
        // Encoded before the idempotency slot is claimed: a refused row
        // must not leave the slot pending.
        let args = upsert_args(&job.meta.collection, &rows)?;
        let slot = Slot {
            sub: ctx.sub().to_string(),
            key: format!("import:{op_id}"),
            sha256: Sha256::digest(format!("{}|{op_id}", job.job.job_id)).into(),
        };
        let reused = "import op_id reused";
        if let Seen::Replay(_) = idem::check(self.b, &ctx, &slot, true, reused, self.now).await? {
            return Ok(());
        }
        let call = Call {
            b: self.b,
            ctx: &ctx,
            dry_run: false,
            now: self.now,
        };
        let res = match service::authorize_op(self.b, &ctx, Op::VectorUpsert).await {
            Ok(a) => service::run(&call, a, Op::VectorUpsert, &args).await,
            Err(e) => Err(e),
        };
        match res {
            Ok(_) => {
                idem::finish(self.b, &ctx, slot, Some("ok".into()), self.now).await;
                Ok(())
            }
            Err(e) => {
                idem::finish(self.b, &ctx, slot, None, self.now).await;
                Err(e)
            }
        }
    }
}

/// `None` = transient (redeliver); otherwise the job fails with this code.
pub(crate) fn sink_fail(e: &OpError) -> Option<FailCode> {
    use ErrorCode as C;
    match e.code {
        C::QuotaExceeded | C::BudgetExceeded | C::PayloadTooLarge => Some(FailCode::QuotaExceeded),
        C::DimensionMismatch => Some(FailCode::Incompatible),
        C::InvalidRequest | C::NonFiniteValue => Some(FailCode::Malformed),
        C::OpReplayed => Some(FailCode::Integrity),
        C::InsufficientScope | C::RoleRequired | C::NotClaimed | C::NotFound => {
            Some(FailCode::Cancelled)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::batch_rows;
    use ruvector_edge_store::IndexConfig;

    /// Metadata text that re-serialises to exactly `n` bytes, every string
    /// byte an escaped `"`.
    fn metadata_of(n: usize) -> String {
        let base = serde_json::to_string(&serde_json::json!({ "k": "" })).unwrap();
        let s = "\"".repeat((n - base.len()) / 2);
        let t = serde_json::to_string(&serde_json::json!({ "k": s })).unwrap();
        assert_eq!(t.len(), n);
        t
    }

    fn worst_rows(dim: u32, n: usize) -> Vec<Row> {
        // The longest `ryu` f32 text (16 chars).
        let v = -9_985_025e6_f32;
        assert_eq!(serde_json::to_string(&v).unwrap(), "-9985025000000.0");
        (0..n)
            .map(|i| Row {
                id: format!("{i:0>6}{}", "\"\\".repeat(125)),
                values: vec![v; dim as usize],
                metadata: Some(metadata_of(MAX_METADATA_BYTES)),
            })
            .collect()
    }

    #[test]
    fn f32_text_never_exceeds_the_per_float_allowance() {
        // A strided sweep of every exponent and sign (≈ 4.3M patterns).
        for x in (0..=u32::MAX).step_by(997) {
            let f = f32::from_bits(x);
            if f.is_finite() {
                let t = serde_json::to_string(&f).unwrap();
                assert!((t.len() as u64) < JSON_BYTES_PER_FLOAT, "{t}");
                assert_eq!(t.parse::<f32>().unwrap().to_bits(), x, "{t}");
            }
        }
    }

    /// Review finding (Workers Paid): 65,536 floats per batch could encode
    /// 1.5–3 MB bodies once metadata and widened floats were counted. A
    /// metadata-heavy batch of `batch_rows` rows now encodes to ≤ 1 MiB at
    /// every dim.
    #[test]
    fn metadata_heavy_batches_encode_within_one_mib() {
        let name = "c".repeat(MAX_COLLECTION_NAME_LEN);
        for dim in [1u32, 8, 128, 384, 768, 1536, 4096] {
            let n = batch_rows(dim, IndexConfig::Flat);
            assert_eq!(
                n as u64,
                rows_within_body(dim).min(65_536 / u64::from(dim)).min(500)
            );
            let body = upsert_args(&name, &worst_rows(dim, n)).unwrap();
            assert!(body.len() as u64 <= SINK_BODY_MAX, "{dim}: {}", body.len());
            // The bound is not slack by more than one row.
            assert!(body.len() as u64 + 2 * worst_row_bytes(dim) > SINK_BODY_MAX || n == 500);
        }
        assert_eq!(
            (
                batch_rows(384, IndexConfig::Flat),
                batch_rows(1536, IndexConfig::Flat)
            ),
            (93, 34)
        );
    }

    #[test]
    fn metadata_that_grows_past_the_cap_when_re_encoded_is_413() {
        // `1e15` re-serialises as `1000000000000000.0`: under 4 KiB as
        // stored text, over it once parsed and encoded again.
        let items = vec!["1e15"; 800].join(",");
        let t = format!("{{\"a\":[{items}]}}");
        assert!(t.len() <= MAX_METADATA_BYTES);
        let row = Row {
            id: "v".into(),
            values: vec![0.5],
            metadata: Some(t),
        };
        let e = upsert_args("c", &[row]).unwrap_err();
        assert_eq!((e.code, e.code.status()), (ErrorCode::PayloadTooLarge, 413));
    }
}
