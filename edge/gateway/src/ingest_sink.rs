//! The batch sink of the `ruvector-edge-ingest` consumer (split out of
//! `ingest`): where a queued import's batches go, the submitter's
//! re-authorized caller context, and which sink errors fail the job.

use crate::backend::Backend;
use crate::idem::{self, Seen, Slot};
use crate::jobs::Job;
use crate::service::{self, Call};
use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_snapshot::{FailCode, Row};
use ruvector_edge_store::{CallerContext, ErrorCode, Op, OpError};
use ruvector_edge_tenancy::TenantKey;
use serde_json::json;

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
        let slot = Slot {
            sub: ctx.sub().to_string(),
            key: format!("import:{op_id}"),
            sha256: Sha256::digest(format!("{}|{op_id}", job.job.job_id)).into(),
        };
        let reused = "import op_id reused";
        if let Seen::Replay(_) = idem::check(self.b, &ctx, &slot, true, reused, self.now).await? {
            return Ok(());
        }
        let mut vectors = Vec::with_capacity(rows.len());
        for r in rows {
            let metadata = match r.metadata {
                None => None,
                Some(t) => Some(
                    serde_json::from_str::<serde_json::Value>(&t)
                        .map_err(|_| OpError::invalid("row metadata"))?,
                ),
            };
            vectors.push(json!({ "id": r.id, "values": r.values, "metadata": metadata }));
        }
        let args = json!({ "collection": job.meta.collection, "vectors": vectors }).to_string();
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
