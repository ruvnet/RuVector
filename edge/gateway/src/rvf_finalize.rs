//! Worker side of `POST …/uploads/{upload_id}:finalize` (ADR-351 §3
//! rv-registry, M5): `Plan` freezes the session; the Worker completes the
//! staging multipart with exactly the frozen parts' recorded R2 part
//! numbers and etags, re-reads each part by range through a
//! [`StreamValidator`], re-measuring it, and only if the RVF is valid and
//! its SHA-256 is the declared one (and the plan says `copy_required`)
//! writes the blob from those same bytes (single part: an R2 put with the
//! expected SHA-256, whose result is the evidence; several: a multipart
//! upload that is aborted unless valid, then measured).
//!
//! Retries are safe: a finalize whose first attempt committed answers with
//! the committed version (the DO's `Plan` returns it), and so does one that
//! lost a race with a concurrent duplicate (its staging object is gone, or
//! the DO says the session is no longer open). Reading stops at the first
//! validation error or changed part, so a bad package costs one pass.

use crate::registry_ports::{scope, BlobStore, RegistryRpc};
use crate::registry_wire::{CallerWire, PlanWire, ReportWire, RvfError, ScopeCall, ScopeOut};
use crate::rvf_upload::{body_json, hex32, sha, Deps, Target};
use ruvector_edge_registry::upload::{ObjectEvidence, PartRecord};
use ruvector_edge_registry::{
    Caller, Provenance, RegistryError, StreamValidator, ValidatedRvf, ValidationError,
};
use serde::Deserialize;
use serde_json::Value as Json;

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct FinalizeBody {
    #[serde(default)]
    provenance: Option<Provenance>,
}

fn committed(manifest: impl serde::Serialize) -> Result<(u16, Json), RvfError> {
    Ok((
        201,
        serde_json::to_value(&manifest).map_err(|_| RvfError::unexpected())?,
    ))
}

/// Ask the (already frozen) session again: `Some` if it committed.
async fn recheck<R: RegistryRpc>(r: &R, t: &Target, call: &ScopeCall) -> Option<(u16, Json)> {
    match scope(r, t.name.scope(), call.clone()).await {
        Ok(ScopeOut::Manifest { manifest }) => committed(manifest).ok(),
        _ => None,
    }
}

/// Complete the staging multipart with the frozen parts. `Ok(false)` when
/// that is impossible for good (a frozen part has no recorded upload, or no
/// multipart exists); a refused complete with no staging object is a
/// storage error, so a transient R2 failure leaves the session frozen and
/// the finalize retryable.
async fn complete_staging<S: BlobStore>(s: &S, p: &PlanWire) -> Result<bool, RvfError> {
    let etags: Option<Vec<(u16, String)>> = p.parts.iter().map(|(_, e)| e.clone()).collect();
    let (Some(mp), Some(etags)) = (&p.multipart, etags) else {
        return Ok(false);
    };
    if s.mp_complete(&p.staging, mp, &etags).await.is_ok() {
        return Ok(true);
    }
    // A retry after an earlier complete: the object is already there.
    match s.size(&p.staging).await? {
        Some(n) => Ok(n == p.size),
        None => Err(RvfError::storage()),
    }
}

/// The blob write of one finalize.
enum Copy {
    None,
    Single(Option<Vec<u8>>),
    Multi(String, Vec<(u16, String)>),
}

/// `POST …/uploads/{upload_id}:finalize`.
pub async fn finalize<B, R: RegistryRpc, S: BlobStore>(
    d: &Deps<'_, B, R, S>,
    c: &Caller,
    t: &Target,
    upload_id: &str,
    body: &[u8],
) -> Result<(u16, Json), RvfError> {
    let fb: FinalizeBody = body_json(body)?;
    let cw = CallerWire::from(c);
    let plan_call = ScopeCall::Plan {
        caller: cw.clone(),
        at: t.coords(),
        upload_id: upload_id.to_string(),
    };
    let plan = match scope(d.r, t.name.scope(), plan_call.clone()).await? {
        ScopeOut::Plan { plan } => plan,
        ScopeOut::Manifest { manifest } => return committed(manifest),
        _ => return Err(RvfError::unexpected()),
    };
    let declared = hex32(&plan.sha256)?;
    let complete = complete_staging(d.s, &plan).await?;
    let mut v = StreamValidator::new(d.cfg.validation);
    let mut streamed = Vec::with_capacity(plan.parts.len());
    let mut copy = match (plan.copy_required, plan.parts.len()) {
        (false, _) => Copy::None,
        (true, 1) => Copy::Single(None),
        (true, _) => Copy::Multi(d.s.mp_create(&plan.blob).await?, Vec::new()),
    };
    let (mut offset, mut intact, mut stopped_invalid) = (0u64, true, false);
    for (p, _) in plan.parts.iter().filter(|_| complete) {
        let Some(bytes) = d.s.get_range(&plan.staging, offset, p.size).await? else {
            // A concurrent duplicate finalize won and removed the staging.
            if let Copy::Multi(up, _) = &copy {
                let _ = d.s.mp_abort(&plan.blob, up).await;
            }
            return recheck(d.r, t, &plan_call)
                .await
                .ok_or_else(RvfError::storage);
        };
        offset += p.size;
        let measured = PartRecord {
            number: p.number,
            size: bytes.len() as u64,
            sha256: sha(&bytes),
        };
        let changed = measured != *p;
        streamed.push(measured);
        // Stop at the first invalid byte or changed part: the session fails
        // either way, and the rest would only burn CPU.
        let invalid = v.push(&bytes).is_err();
        if invalid || changed {
            intact = false;
            stopped_invalid = invalid && !changed;
            break;
        }
        match &mut copy {
            Copy::Single(slot) => *slot = Some(bytes),
            Copy::Multi(up, etags) => {
                let n = p.number as u16;
                match d.s.mp_part(&plan.blob, up, n, bytes).await {
                    Ok(e) => etags.push((n, e)),
                    Err(e) => {
                        let _ = d.s.mp_abort(&plan.blob, up).await;
                        return Err(e);
                    }
                }
            }
            Copy::None => {}
        }
    }
    let validated: Result<ValidatedRvf, ValidationError> = if complete {
        v.finish()
    } else {
        Err(ValidationError::Empty)
    };
    let ok = intact
        && streamed.len() == plan.parts.len()
        && validated.as_ref().is_ok_and(|r| r.sha256 == declared);
    let copied = write_blob(d.s, &plan, copy, ok, declared).await?;
    let report = ReportWire {
        validated: validated.as_ref().ok().cloned(),
        streamed,
        copied,
        provenance: fb.provenance,
    };
    let call = ScopeCall::Finalize {
        caller: cw,
        at: t.coords(),
        upload_id: upload_id.to_string(),
        report,
    };
    match scope(d.r, t.name.scope(), call).await {
        Ok(ScopeOut::Manifest { manifest }) => {
            // Best effort: the sweep deletes it at expiry otherwise.
            let _ = d.s.delete(&plan.staging).await;
            committed(manifest)
        }
        Ok(_) => Err(RvfError::unexpected()),
        // The local validation error is the precise one (a stop at the
        // first invalid byte leaves the streamed parts short, which the DO
        // reports as changed parts).
        Err(e) if e.validation || stopped_invalid => Err(match validated {
            Err(local) => RegistryError::Validation(local).into(),
            Ok(_) => e,
        }),
        // A concurrent duplicate committed first.
        Err(e) if e.status == 409 => Ok(recheck(d.r, t, &plan_call).await.ok_or(e)?),
        Err(e) => Err(e),
    }
}

/// Write the blob if the plan asked for it and the bytes proved valid;
/// return the stored object. Storage failures are returned (the session
/// stays frozen, so the finalize can be retried) rather than reported as
/// missing evidence, which would fail the session.
async fn write_blob<S: BlobStore>(
    s: &S,
    plan: &PlanWire,
    copy: Copy,
    ok: bool,
    declared: [u8; 32],
) -> Result<Option<ObjectEvidence>, RvfError> {
    match copy {
        Copy::None => Ok(None),
        // R2 verified the SHA-256 on the put: its answer is the evidence.
        Copy::Single(Some(bytes)) if ok => {
            s.put_checked(&plan.blob, bytes, declared).await.map(Some)
        }
        Copy::Multi(up, etags) if ok => {
            if let Err(e) = s.mp_complete(&plan.blob, &up, &etags).await {
                let _ = s.mp_abort(&plan.blob, &up).await;
                return Err(e);
            }
            s.measure(&plan.blob).await
        }
        Copy::Multi(up, _) => {
            let _ = s.mp_abort(&plan.blob, &up).await;
            Ok(None)
        }
        Copy::Single(_) => Ok(None),
    }
}
