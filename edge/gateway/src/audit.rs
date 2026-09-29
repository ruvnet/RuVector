//! Audit shipping (ADR-351 §6.3 `audit/`, §6.5).
//!
//! **What is shipped.** One [`AuditEvent`] per mutating or privileged
//! request — every M3 route except a plain (no `text`) query handed to M1,
//! every upsert (REST, text or not), and the other M1 writes
//! (`audit_http`: claim, delete, mutating `/v1/ops` ops and MCP tools) — plus one per finished import job
//! (`job.done` / `job.failed` / `job.cancelled`). Plain reads (query, fetch,
//! get, list, me, usage) are not shipped: §6.1 puts the audit row in the
//! write path, and a Queue message per read would share the per-queue
//! message rate across every tenant's hot path. Logged: tenant, `sub`,
//! `client_id`, `act.sub`, `jti`, `family_id`, route, scope and role used,
//! outcome, rows, bytes, work units. Never tokens, vectors, metadata or IPs.
//! Gateway requests send through `ctx.wait_until`, so auditing adds no
//! latency; an audit outage never fails a request (best effort).
//!
//! **How.** The consumer groups a delivered batch by tenant and hands each
//! group to the tenant's ledger (`m3_audit_ledger`), which deduplicates by
//! queue message id, chains the lines onto the tenant's single audit chain
//! and names the object `audit/{tenant}/{yyyy}/{mm}/{dd}/{seq:012}.ndjson`;
//! the consumer PUTs every pending object and then confirms it.

use crate::m3_ports::{Blob, QueueName, Queues};
use crate::m3_wire::{ledger3, AuditLine, M3Backend, M3LedgerCall, M3LedgerOut};
use ruvector_edge_store::{CallerContext, OpError};
use ruvector_edge_tenancy::{Role, TenantKey};
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::BTreeMap;

/// One audit record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuditEvent {
    /// Schema version (1).
    pub v: u8,
    /// Unix ms.
    pub ts_ms: u64,
    /// Tenant (from the verified token).
    pub tenant_key: String,
    /// Edge subject.
    pub sub: String,
    /// OAuth client.
    pub client_id: String,
    /// Acting adapter, if exchanged.
    #[serde(default)]
    pub act_sub: Option<String>,
    /// Token id.
    pub jti: String,
    /// Grant family.
    pub family_id: String,
    /// Route or job event (`snapshot.create`, `job.done`, …).
    pub route: String,
    /// Scope the route required (`ruvector:write`, …), if checked.
    #[serde(default)]
    pub scope: Option<String>,
    /// Ledger role the caller acted with, if resolved.
    #[serde(default)]
    pub role: Option<String>,
    /// HTTP status or job outcome code.
    pub outcome: u16,
    /// Rows touched.
    pub rows: u64,
    /// Bytes moved.
    pub bytes: u64,
    /// Work units charged.
    pub work_units: u64,
}

impl AuditEvent {
    /// An event for `ctx`.
    pub fn new(ctx: &CallerContext, route: &str, outcome: u16, now_ms: u64) -> Self {
        AuditEvent {
            v: 1,
            ts_ms: now_ms,
            tenant_key: ctx.tenant_key().as_str().to_string(),
            sub: ctx.sub().to_string(),
            client_id: ctx.client_id().to_string(),
            act_sub: ctx.act_sub().map(str::to_string),
            jti: ctx.jti().to_string(),
            family_id: ctx.family_id().to_string(),
            route: route.to_string(),
            scope: None,
            role: None,
            outcome,
            rows: 0,
            bytes: 0,
            work_units: 0,
        }
    }

    /// Fill scope, role, rows, bytes and work units from a request's note.
    pub fn with_note(mut self, n: &Note) -> Self {
        self.scope = n.scope.get().map(str::to_string);
        self.role = n.role.get().map(|r| r.as_str().to_string());
        self.rows = n.rows.get();
        self.bytes = n.bytes.get();
        self.work_units = n.work_units.get();
        self
    }
}

/// What one request did, filled in as it runs (single-threaded isolate).
#[derive(Debug, Default)]
pub struct Note {
    /// Scope checked.
    pub scope: Cell<Option<&'static str>>,
    /// Role acted with.
    pub role: Cell<Option<Role>>,
    /// Work units admitted.
    pub work_units: Cell<u64>,
    /// Rows touched.
    pub rows: Cell<u64>,
    /// Bytes moved.
    pub bytes: Cell<u64>,
}

impl Note {
    /// Add admitted work units.
    pub fn charged(&self, wu: u64) {
        self.work_units
            .set(self.work_units.get().saturating_add(wu));
    }
}

/// Best-effort send (an audit outage never fails the request).
pub async fn emit<Q: Queues>(q: &Q, ev: &AuditEvent) {
    if let Ok(body) = serde_json::to_value(ev) {
        let _sent = q.send(QueueName::Audit, body).await;
    }
}

/// `(yyyy, mm, dd)` of a Unix ms timestamp (UTC, proleptic Gregorian).
pub fn ymd(ts_ms: u64) -> (i64, u32, u32) {
    let days = (ts_ms / 86_400_000) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// Consumer: chain and write one delivered batch of `(message id, event)`.
/// Events of a malformed tenant are dropped (they cannot be placed). Any
/// error means "retry the batch": every step is idempotent. Returns the
/// objects written.
pub async fn ship<B: M3Backend, R: Blob>(
    b: &B,
    blob: &R,
    events: Vec<(String, AuditEvent)>,
    now_ms: u64,
) -> Result<usize, OpError> {
    let mut groups: BTreeMap<String, Vec<(String, AuditEvent)>> = BTreeMap::new();
    for (mid, e) in events {
        groups
            .entry(e.tenant_key.clone())
            .or_default()
            .push((mid, e));
    }
    let mut written = 0;
    for (tenant, mut evs) in groups {
        let Ok(t) = TenantKey::parse(&tenant) else {
            continue;
        };
        evs.sort_by(|a, b| (a.1.ts_ms, &a.0).cmp(&(b.1.ts_ms, &b.0)));
        let mut lines = Vec::with_capacity(evs.len());
        for (mid, e) in evs {
            let json = serde_json::to_string(&e).map_err(|_| crate::service::unexpected())?;
            lines.push(AuditLine {
                mid,
                ts_ms: e.ts_ms,
                json,
            });
        }
        let call = M3LedgerCall::AuditAppend { lines, now_ms };
        let M3LedgerOut::AuditPending { objects } = ledger3(b, &t, call).await? else {
            return Err(crate::service::unexpected());
        };
        let mut seqs = Vec::with_capacity(objects.len());
        for o in objects {
            blob.put(&o.key, o.body.into_bytes(), None).await?;
            seqs.push(o.seq);
            written += 1;
        }
        if !seqs.is_empty() {
            ledger3(b, &t, M3LedgerCall::AuditCommit { seqs }).await?;
        }
    }
    Ok(written)
}

/// The tenant's audit chain position `(next seq, head)`.
pub async fn head<B: M3Backend>(b: &B, t: &TenantKey) -> Result<(u64, [u8; 32]), OpError> {
    match ledger3(b, t, M3LedgerCall::AuditHead).await? {
        M3LedgerOut::AuditHead { next, head } => Ok((next, crate::m3_wire::unhex32(&head)?)),
        _ => Err(crate::service::unexpected()),
    }
}
