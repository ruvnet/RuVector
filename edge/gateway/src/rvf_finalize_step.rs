//! Stepped finalize of a large upload (ADR-351 §3 rv-registry, M5).
//!
//! The Worker runs on the default per-invocation CPU limit (no
//! `limits.cpu_ms`: the account's plan refuses it), so a finalize must not
//! hash a whole package in one request. An upload of at most
//! [`INLINE_FINALIZE_BYTES`] (every single-part upload included) keeps the
//! one-request path of `rvf_finalize`. A larger one is validated in steps
//! of at most [`STEP_BYTES`] of staged parts each, **inside its scope's
//! `RegistryScope` DO**, the only place whose memory outlives a request:
//! the live [`StreamValidator`] (hash states cannot be serialized) stays in
//! the object's [`Jobs`], and a durable cursor (`gw/fz/{upload_id}`)
//! records progress. If the object was evicted between steps the cursor
//! is ahead of the (missing) job: validation restarts from byte 0 and
//! `restarts` counts it, so nothing is ever trusted that was not hashed by
//! the live validator.
//!
//! One `POST …/uploads/{id}:finalize` drives the steps back to back
//! (`rvf_finalize::FINALIZE_STEPS`), so the object is never idle long
//! enough to hibernate mid-finalize; each step is its own DO invocation.
//! The last step writes the blob (a streamed R2 put of the staging object
//! with the declared SHA-256, so R2 verifies the bytes and no Worker-side
//! hash pass is needed), commits and answers `201` with the manifest. A
//! request that stops early (another request is stepping the upload, or
//! the step cap ran out) answers `202` with progress, and the client
//! repeats the same request (same body) until `201`.
//!
//! **CPU: assumes Workers Paid.** The byte budgets below assume the Paid
//! plan's 30 s default CPU per invocation; the Free plan's 10 ms cannot
//! hash even one 8 MiB part, so registry uploads need Workers Paid.
//! Retries are safe: a committed session answers with its manifest, a step
//! in flight answers with progress, and a failed session answers its error.

use crate::registry_core::{scope_reply, serve_scope};
use crate::registry_kv::SqlKv;
use crate::registry_ports::{BlobStore, RegistryRpc};
use crate::registry_wire::{
    decode, encode, CallerWire, Coords, PlanWire, Reply, ReportWire, RvfError, ScopeCall, ScopeOut,
};
use crate::rvf_finalize::complete_staging;
use crate::rvf_upload::{hex32, sha};
use ruvector_edge_auth::Clock;
use ruvector_edge_registry::keys::registry_do_name;
use ruvector_edge_registry::ports::KvStore;
use ruvector_edge_registry::registry::RegistryConfig;
use ruvector_edge_registry::upload::{ObjectEvidence, PartRecord};
use ruvector_edge_registry::{
    ManifestView, Provenance, RegistryError, Scope, StreamValidator, ValidatedRvf, ValidationError,
};
use ruvector_edge_store::SqlStore;
use ruvector_edge_tenancy::{EntropySource, ProblemCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// Largest upload finalized in one request (on Workers Paid; hashing it takes well under a
/// second of wasm CPU; every single-part upload, ≤ 8 MiB, is below it).
pub const INLINE_FINALIZE_BYTES: u64 = 32 << 20;
/// Staged bytes one step validates at most (a single part may exceed it
/// only if it alone is larger; parts are ≤ 8 MiB). Three hash passes over
/// 64 MiB stay within a few seconds of wasm CPU, far below the Workers
/// **Paid** 30 s default limit (on Free, 10 ms, no step size fits).
pub const STEP_BYTES: u64 = 64 << 20;

/// One finalize step, Worker → `RegistryScope`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepCall {
    /// Caller.
    pub caller: CallerWire,
    /// Coordinates.
    pub at: Coords,
    /// Session.
    pub upload_id: String,
    /// Optional witness root (taken from the step that commits).
    pub provenance: Option<Provenance>,
}

/// What a step answers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "out", rename_all = "snake_case")]
pub enum StepOut {
    /// Not done yet: repeat the finalize.
    Progress {
        /// Staged bytes validated so far.
        bytes_done: u64,
        /// Declared size.
        size: u64,
        /// Times validation started over (object evicted between steps).
        restarts: u32,
    },
    /// Committed.
    Manifest {
        /// View.
        manifest: Box<ManifestView>,
    },
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct Cursor {
    next: usize,
    bytes_done: u64,
    restarts: u32,
}

/// Durable cursor of an upload's stepped finalize.
pub fn k_cursor(upload_id: &str) -> String {
    format!("gw/fz/{upload_id}")
}

/// Streaming state of one finalize.
struct Streaming {
    v: StreamValidator,
    streamed: Vec<PartRecord>,
    next: usize,
    offset: u64,
    complete: bool,
}

/// Validated (or refused) and, if required, copied: only the commit is
/// left (it could not run while the sweep held the object).
struct Ready {
    validated: Result<ValidatedRvf, ValidationError>,
    streamed: Vec<PartRecord>,
    copied: Option<ObjectEvidence>,
    stopped_invalid: bool,
}

enum Job {
    Streaming(Box<Streaming>),
    Ready(Box<Ready>),
}

/// Finalize jobs held in a `RegistryScope`'s memory, by upload id.
#[derive(Default)]
pub struct Jobs {
    live: RefCell<HashMap<String, Job>>,
    busy: RefCell<HashSet<String>>,
}

impl Jobs {
    /// Jobs in memory (tests: an evicted object has none).
    #[cfg(test)]
    pub fn live(&self) -> usize {
        self.live.borrow().len()
    }

    /// Forget every job (tests: the object was evicted).
    #[cfg(test)]
    pub fn evict(&self) {
        self.live.borrow_mut().clear();
    }

    fn put(&self, id: &str, job: Job) {
        self.live.borrow_mut().insert(id.to_string(), job);
    }
}

/// Marks a job as stepping; released on drop (a cancelled step included).
struct Busy<'a>(&'a Jobs, String);

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.busy.borrow_mut().remove(&self.1);
    }
}

/// A streaming job taken out of [`Jobs`] for one step; put back on drop
/// unless the step consumed (or deliberately dropped) it.
struct Held<'a>(&'a Jobs, String, Option<Box<Streaming>>);

impl Drop for Held<'_> {
    fn drop(&mut self) {
        if let Some(s) = self.2.take() {
            self.0.put(&self.1, Job::Streaming(s));
        }
    }
}

/// The ports and limits of one step, inside the DO.
pub struct Step<'a, Q, C, S> {
    /// The object's SQLite.
    pub sql: Q,
    /// Clock.
    pub clock: &'a C,
    /// Entropy.
    pub entropy: &'a dyn EntropySource,
    /// Registry configuration.
    pub cfg: RegistryConfig,
    /// R2.
    pub r2: &'a S,
    /// Byte budget of one step ([`STEP_BYTES`] in production).
    pub step_bytes: u64,
    /// `false` while the sweep holds the object: no index call then.
    pub open: &'a dyn Fn() -> bool,
}

/// Reply of one step, plus whether the object must keep an alarm armed.
pub struct Stepped {
    /// Encoded `Reply<StepOut>`.
    pub body: String,
    /// Sessions or released blobs are pending.
    pub maintenance: bool,
}

impl<Q: SqlStore + Copy, C: Clock, S: BlobStore> Step<'_, Q, C, S> {
    fn index(&self, call: &ScopeCall, maintenance: &mut bool) -> Reply<ScopeOut> {
        if !(self.open)() {
            return Err(RvfError::unavailable());
        }
        let body = serde_json::to_vec(call).map_err(|_| RvfError::unexpected())?;
        let served = serve_scope(self.sql, self.clock, self.entropy, self.cfg, &body);
        *maintenance |= served.maintenance;
        scope_reply(&served.body)
    }

    fn kv(&self) -> Result<SqlKv<Q>, RvfError> {
        SqlKv::open(self.sql).map_err(|e| RegistryError::from(e).into())
    }

    fn cursor(&self, id: &str) -> Result<Cursor, RvfError> {
        let raw = self.kv()?.get(&k_cursor(id)).map_err(RegistryError::from)?;
        Ok(raw
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default())
    }

    fn save(&self, id: &str, c: Cursor) -> Result<(), RvfError> {
        let v = serde_json::to_vec(&c).map_err(|_| RvfError::unexpected())?;
        self.kv()?
            .put(&k_cursor(id), &v)
            .map_err(|e| RegistryError::from(e).into())
    }

    fn clear(&self, id: &str) {
        if let Ok(kv) = self.kv() {
            let _ = kv.delete(&k_cursor(id));
        }
    }
}

/// Serve one step inside the `RegistryScope` DO.
pub async fn serve_step<Q: SqlStore + Copy, C: Clock, S: BlobStore>(
    jobs: &Jobs,
    st: &Step<'_, Q, C, S>,
    body: &[u8],
) -> Stepped {
    let mut maintenance = false;
    let out = match serde_json::from_slice::<StepCall>(body) {
        Ok(call) => step(jobs, st, call, &mut maintenance).await,
        Err(_) => Err(RvfError::new(ProblemCode::InvalidRequest, "malformed call")),
    };
    Stepped {
        body: encode(&out),
        maintenance,
    }
}

fn progress(c: Cursor, size: u64) -> StepOut {
    StepOut::Progress {
        bytes_done: c.bytes_done,
        size,
        restarts: c.restarts,
    }
}

async fn step<Q: SqlStore + Copy, C: Clock, S: BlobStore>(
    jobs: &Jobs,
    st: &Step<'_, Q, C, S>,
    call: StepCall,
    m: &mut bool,
) -> Reply<StepOut> {
    let plan_call = ScopeCall::Plan {
        caller: call.caller.clone(),
        at: call.at.clone(),
        upload_id: call.upload_id.clone(),
    };
    let plan = match st.index(&plan_call, m) {
        Ok(ScopeOut::Plan { plan }) => plan,
        Ok(ScopeOut::Manifest { manifest }) => {
            st.clear(&call.upload_id);
            return Ok(StepOut::Manifest { manifest });
        }
        Ok(_) => return Err(RvfError::unexpected()),
        Err(e) => {
            // Not finalizable (failed, expired, …): nothing to keep.
            if !(st.open)() || e.status >= 500 {
                return Err(e);
            }
            jobs.live.borrow_mut().remove(&call.upload_id);
            st.clear(&call.upload_id);
            return Err(e);
        }
    };
    let id = call.upload_id.clone();
    let mut cur = st.cursor(&id)?;
    if !jobs.busy.borrow_mut().insert(id.clone()) {
        // Another step of this upload is running: report where it is.
        return Ok(progress(cur, plan.size));
    }
    let _busy = Busy(jobs, id.clone());
    let taken = jobs.live.borrow_mut().remove(&id);
    let fresh = match taken {
        Some(Job::Ready(r)) => return commit(jobs, st, &call, &plan, &plan_call, *r, m).await,
        Some(Job::Streaming(s)) => s,
        None => {
            if cur.next > 0 {
                // The object lost the live validator: start over.
                cur.restarts = cur.restarts.saturating_add(1);
            }
            cur.next = 0;
            cur.bytes_done = 0;
            let complete = complete_staging(st.r2, &plan).await?;
            Box::new(Streaming {
                v: StreamValidator::new(st.cfg.validation),
                streamed: Vec::with_capacity(plan.parts.len()),
                next: 0,
                offset: 0,
                complete,
            })
        }
    };
    // Put back on every exit that does not consume it (errors and a
    // cancelled step included), so only eviction ever restarts a job.
    let mut held = Held(jobs, id.clone(), Some(fresh));
    let Some(s) = held.2.as_mut() else {
        return Err(RvfError::unexpected());
    };
    let mut spent = 0u64;
    let mut stop: Option<bool> = None;
    while s.complete && s.next < plan.parts.len() {
        let p = &plan.parts[s.next].0;
        if spent > 0 && spent.saturating_add(p.size) > st.step_bytes {
            break;
        }
        let bytes = match st.r2.get_range(&plan.staging, s.offset, p.size).await {
            Ok(Some(b)) => b,
            Ok(None) => {
                // A concurrent finalize or the sweep removed the staging.
                held.2 = None;
                st.clear(&id);
                return match st.index(&plan_call, m) {
                    Ok(ScopeOut::Manifest { manifest }) => Ok(StepOut::Manifest { manifest }),
                    _ => Err(RvfError::storage()),
                };
            }
            // Transient: the guard keeps the progress made so far.
            Err(e) => return Err(e),
        };
        s.offset += p.size;
        s.next += 1;
        spent += p.size;
        let measured = PartRecord {
            number: p.number,
            size: bytes.len() as u64,
            sha256: sha(&bytes),
        };
        let changed = measured != *p;
        s.streamed.push(measured);
        // Stop at the first invalid byte or changed part: the session fails
        // either way.
        let invalid = s.v.push(&bytes).is_err();
        if invalid || changed {
            stop = Some(invalid && !changed);
            break;
        }
    }
    if stop.is_none() && s.complete && s.next < plan.parts.len() {
        cur.next = s.next;
        cur.bytes_done = s.offset;
        st.save(&id, cur)?;
        return Ok(progress(cur, plan.size));
    }
    let Some(s) = held.2.take() else {
        return Err(RvfError::unexpected());
    };
    let ready = finish(st, &plan, *s, stop).await?;
    commit(jobs, st, &call, &plan, &plan_call, ready, m).await
}

/// All parts streamed (or streaming stopped): finish the validator and
/// write the blob if the plan asks for it and the bytes proved valid.
async fn finish<Q: SqlStore + Copy, C: Clock, S: BlobStore>(
    st: &Step<'_, Q, C, S>,
    plan: &PlanWire,
    s: Streaming,
    stop: Option<bool>,
) -> Result<Ready, RvfError> {
    let declared = hex32(&plan.sha256)?;
    let validated = if s.complete {
        s.v.finish()
    } else {
        Err(ValidationError::Empty)
    };
    let ok = stop.is_none()
        && s.streamed.len() == plan.parts.len()
        && validated.as_ref().is_ok_and(|r| r.sha256 == declared);
    // A storage failure here drops the job: the next step starts over.
    let copied = if plan.copy_required && ok {
        Some(
            st.r2
                .copy_checked(&plan.staging, &plan.blob, declared)
                .await?,
        )
    } else {
        None
    };
    Ok(Ready {
        validated,
        streamed: s.streamed,
        copied,
        stopped_invalid: stop == Some(true),
    })
}

async fn commit<Q: SqlStore + Copy, C: Clock, S: BlobStore>(
    jobs: &Jobs,
    st: &Step<'_, Q, C, S>,
    call: &StepCall,
    plan: &PlanWire,
    plan_call: &ScopeCall,
    r: Ready,
    m: &mut bool,
) -> Reply<StepOut> {
    if !(st.open)() {
        jobs.put(&call.upload_id, Job::Ready(Box::new(r)));
        return Err(RvfError::unavailable());
    }
    let report = ReportWire {
        validated: r.validated.as_ref().ok().cloned(),
        streamed: r.streamed,
        copied: r.copied,
        provenance: call.provenance.clone(),
    };
    let fin = ScopeCall::Finalize {
        caller: call.caller.clone(),
        at: call.at.clone(),
        upload_id: call.upload_id.clone(),
        report,
    };
    let res = st.index(&fin, m);
    st.clear(&call.upload_id);
    match res {
        Ok(ScopeOut::Manifest { manifest }) => {
            // Best effort: the sweep deletes it at expiry otherwise.
            let _ = st.r2.delete(&plan.staging).await;
            Ok(StepOut::Manifest { manifest })
        }
        Ok(_) => Err(RvfError::unexpected()),
        Err(e) if e.validation || r.stopped_invalid => Err(match r.validated {
            Err(local) => RegistryError::Validation(local).into(),
            Ok(_) => e,
        }),
        Err(e) if e.status == 409 => match st.index(plan_call, m) {
            Ok(ScopeOut::Manifest { manifest }) => Ok(StepOut::Manifest { manifest }),
            _ => Err(e),
        },
        Err(e) => Err(e),
    }
}

/// Worker side: run one step of `upload_id`'s finalize in its scope's DO.
/// `202` with progress, or `201` with the manifest.
pub async fn drive<R: RegistryRpc>(
    r: &R,
    scope: &Scope,
    call: StepCall,
) -> Result<(u16, Json), RvfError> {
    let body = serde_json::to_string(&call).map_err(|_| RvfError::unexpected())?;
    let text = r.call_scope_step(&registry_do_name(scope), body).await?;
    match decode::<StepOut>(&text)? {
        StepOut::Progress {
            bytes_done,
            size,
            restarts,
        } => Ok((
            202,
            json!({
                "status": "finalizing",
                "upload_id": call.upload_id,
                "bytes_validated": bytes_done,
                "size": size,
                "restarts": restarts,
            }),
        )),
        StepOut::Manifest { manifest } => Ok((
            201,
            serde_json::to_value(&manifest).map_err(|_| RvfError::unexpected())?,
        )),
    }
}
