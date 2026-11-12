//! The upload-session arms of the `RegistryScope` core (ADR-351 §3
//! rv-registry, M5): begin, attach, part targets, part records, plan and
//! finalize, plus the gateway's per-session multipart record
//! `gw/mp/{upload_id}`.
//!
//! **R2 part numbers are unique per attempt.** Logical part `n` may be
//! uploaded up to [`ATTEMPTS_PER_PART`] times; attempt `a` goes to R2 part
//! number `(n-1)·K + a + 1`. R2 never overwrites one attempt's bytes with
//! another's, so whatever order concurrent retries of a part reach R2 and
//! this index, the recorded (R2 part number, etag, SHA-256) always names
//! bytes R2 holds, and ascending logical order is ascending R2 order.

use crate::registry_core::{err, session, view_out, Reg};
use crate::registry_kv::SqlKv;
use crate::registry_wire::{PlanWire, Reply, RvfError, ScopeCall, ScopeOut};
use ruvector_edge_auth::{Capability, Clock};
use ruvector_edge_registry::keys::staging_key;
use ruvector_edge_registry::ports::KvStore;
use ruvector_edge_registry::registry::FinalizeReport;
use ruvector_edge_registry::upload::{UploadId, UploadSession, UploadState};
use ruvector_edge_registry::{RegistryError, ValidationError};
use ruvector_edge_store::{SqlStore, StoreError};
use ruvector_edge_tenancy::ProblemCode;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Highest R2 multipart part number.
pub const R2_MAX_PART_NUMBER: u32 = 10_000;
/// Uploads of one logical part a session accepts (distinct R2 parts).
pub const ATTEMPTS_PER_PART: u32 = 10;
/// Sessions a scope keeps at once (open, or ended but not yet swept).
pub const MAX_SESSIONS_PER_SCOPE: usize = 1000;

/// The gateway's multipart record of one session.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MpRecord {
    /// R2 multipart upload id of the staging object.
    pub multipart: String,
    /// Logical part → (R2 part number, R2 etag) of the recorded attempt.
    #[serde(default)]
    pub parts: BTreeMap<u32, (u16, String)>,
    /// Logical part → attempts handed out.
    #[serde(default)]
    pub attempts: BTreeMap<u32, u32>,
}

pub(crate) fn k_mp(id: &UploadId) -> String {
    format!("gw/mp/{}", id.as_str())
}

pub(crate) fn load_mp<K: KvStore>(kv: &K, id: &UploadId) -> Result<Option<MpRecord>, RvfError> {
    match kv.get(&k_mp(id)).map_err(|e| err(e.into()))? {
        None => Ok(None),
        Some(b) => serde_json::from_slice(&b)
            .map(Some)
            .map_err(|_| err(StoreError::Corrupt("mp record").into())),
    }
}

fn save_mp<K: KvStore>(kv: &K, id: &UploadId, r: &MpRecord) -> Result<(), RvfError> {
    let b = serde_json::to_vec(r).map_err(|_| RvfError::unexpected())?;
    kv.put(&k_mp(id), &b).map_err(|e| err(e.into()))
}

/// R2 part number of attempt `a` (0-based) of logical part `n`.
pub fn r2_part(n: u32, a: u32) -> Option<u16> {
    let r = n
        .checked_sub(1)?
        .checked_mul(ATTEMPTS_PER_PART)?
        .checked_add(a + 1)?;
    (a < ATTEMPTS_PER_PART && r <= R2_MAX_PART_NUMBER)
        .then(|| u16::try_from(r).ok())
        .flatten()
}

/// R2 multipart rule the registry crate does not enforce: every part but
/// the last has the same size.
fn uniform_parts(s: &UploadSession) -> Result<(), RvfError> {
    if let Some((_, init)) = s.parts.split_last() {
        if init.windows(2).any(|w| w[0].size != w[1].size)
            || init
                .last()
                .zip(s.parts.last())
                .is_some_and(|(a, l)| l.size > a.size)
        {
            return Err(err(RegistryError::PartsIncomplete(
                "every part but the last must have the same size, the last no larger",
            )));
        }
    }
    Ok(())
}

/// Serve an upload-session call (`None` for any other call).
pub(crate) fn upload_call<S: SqlStore, C: Clock>(
    reg: &Reg<'_, S, C>,
    kv: &SqlKv<S>,
    now: u64,
    call: ScopeCall,
) -> Option<Reply<ScopeOut>> {
    Some(match call {
        ScopeCall::Begin { caller, req } => (|| -> Reply<ScopeOut> {
            let open = kv
                .count_keys("upload/", MAX_SESSIONS_PER_SCOPE)
                .map_err(|e| err(e.into()))?;
            if open >= MAX_SESSIONS_PER_SCOPE {
                return Err(RvfError::new(
                    ProblemCode::RateLimited,
                    "too many upload sessions in this scope; finish or let some expire",
                ));
            }
            let s = reg.begin_upload(&caller.caller()?, req).map_err(err)?;
            Ok(ScopeOut::Session {
                staging: staging_key(&s.tenant, &s.id).to_string(),
                upload_id: s.id.as_str().to_string(),
                expires_at: s.expires_at,
            })
        })(),
        ScopeCall::Attach {
            caller,
            upload_id,
            multipart,
        } => (|| -> Reply<ScopeOut> {
            let c = caller.caller()?;
            let id = UploadId::parse(&upload_id).map_err(err)?;
            reg.upload(&c, &id).map_err(err)?;
            if load_mp(kv, &id)?.is_some() {
                return Err(err(RegistryError::UploadNotOpen));
            }
            let rec = MpRecord {
                multipart,
                ..Default::default()
            };
            save_mp(kv, &id, &rec)?;
            Ok(ScopeOut::Done)
        })(),
        ScopeCall::PartTarget {
            caller,
            at,
            upload_id,
            number,
        } => (|| -> Reply<ScopeOut> {
            let (c, id, s) = session(reg, &caller, &at, &upload_id)?;
            if !c.caps.contains(Capability::Write) {
                return Err(err(RegistryError::Forbidden(Capability::Write)));
            }
            s.check_open(now).map_err(err)?;
            let lim = reg.config().upload;
            if number == 0 || number > lim.max_parts || r2_part(number, 0).is_none() {
                return Err(err(RegistryError::UploadLimit("part number out of range")));
            }
            let mut mp = load_mp(kv, &id)?.ok_or(err(RegistryError::UploadNotOpen))?;
            let a = mp.attempts.get(&number).copied().unwrap_or(0);
            let r2 = r2_part(number, a).ok_or_else(|| {
                RvfError::new(
                    ProblemCode::Conflict,
                    "part uploaded too many times; begin a new upload",
                )
            })?;
            mp.attempts.insert(number, a + 1);
            save_mp(kv, &id, &mp)?;
            Ok(ScopeOut::PartTarget {
                staging: staging_key(&s.tenant, &id).to_string(),
                multipart: mp.multipart,
                r2_part: r2,
            })
        })(),
        ScopeCall::RecordPart {
            caller,
            at,
            upload_id,
            part,
            r2_part: r2,
            etag,
        } => (|| -> Reply<ScopeOut> {
            let (c, id, _) = session(reg, &caller, &at, &upload_id)?;
            let mut mp = load_mp(kv, &id)?.ok_or(err(RegistryError::UploadNotOpen))?;
            let n = part.number;
            let handed_out = mp.attempts.get(&n).copied().unwrap_or(0);
            if !(0..handed_out).any(|a| r2_part(n, a) == Some(r2)) {
                return Err(RvfError::new(
                    ProblemCode::InvalidRequest,
                    "part attempt was never handed out",
                ));
            }
            reg.record_part(&c, &id, part).map_err(err)?;
            mp.parts.insert(n, (r2, etag));
            save_mp(kv, &id, &mp)?;
            Ok(ScopeOut::Done)
        })(),
        ScopeCall::Plan {
            caller,
            at,
            upload_id,
        } => (|| -> Reply<ScopeOut> {
            let (c, id, s) = session(reg, &caller, &at, &upload_id)?;
            match s.state {
                UploadState::Open => uniform_parts(&s)?,
                // A retried finalize whose first attempt committed: answer
                // with the committed version instead of a conflict.
                UploadState::Finalized => {
                    let m = reg
                        .get(&c, &s.target.name, &s.target.version)
                        .map_err(err)?;
                    return Ok(ScopeOut::Manifest {
                        manifest: Box::new(m),
                    });
                }
                _ => {}
            }
            let plan = reg.finalize_plan(&c, &id).map_err(err)?;
            let mp = load_mp(kv, &id)?.unwrap_or_default();
            let parts = plan
                .parts
                .into_iter()
                .map(|p| {
                    let e = mp.parts.get(&p.number).cloned();
                    (p, e)
                })
                .collect();
            Ok(ScopeOut::Plan {
                plan: PlanWire {
                    staging: plan.staging.to_string(),
                    multipart: (!mp.multipart.is_empty()).then_some(mp.multipart),
                    parts,
                    blob: plan.blob.to_string(),
                    copy_required: plan.copy_required,
                    size: plan.size,
                    sha256: hex::encode(plan.sha256),
                },
            })
        })(),
        ScopeCall::Finalize {
            caller,
            at,
            upload_id,
            report,
        } => (|| -> Reply<ScopeOut> {
            let (c, id, _) = session(reg, &caller, &at, &upload_id)?;
            // A validation failure is reported without its (non-serializable)
            // cause; the Worker answers with its own precise error.
            let validated = report.validated.ok_or(ValidationError::Empty);
            let r = FinalizeReport {
                validated,
                streamed: report.streamed,
                copied: report.copied,
                provenance: report.provenance,
            };
            let m = reg.finalize(&c, &id, r).map_err(err)?;
            // The staging multipart was completed: nothing left to abort.
            kv.delete(&k_mp(&id)).map_err(|e| err(e.into()))?;
            Ok(view_out(m, &c))
        })(),
        _ => return None,
    })
}
