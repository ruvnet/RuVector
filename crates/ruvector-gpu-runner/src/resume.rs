//! `--resume-from`: resolve a run id or `gs://` object to one checkpoint
//! object in the artifact bucket (plus its sha256 when known). The launcher
//! then mints a short-lived signed GET URL for exactly that object; the
//! instance downloads it, verifies the sha256, and extracts it into the
//! checkpoint dir before the job command runs. No credential reaches the
//! instance.

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::gcs::{GcsPlan, CKPT_POINTER};

#[derive(Debug, Clone, PartialEq)]
pub enum ResumeSource {
    /// A previous run id (`rvgr-YYYYMMDDtHHMMSS-<8 hex>`): resolved through
    /// that run's `ckpt-latest.json` pointer.
    Run(String),
    /// An explicit object `gs://<bucket>/<path>.tar.gz` in the artifact bucket.
    Object(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedResume {
    pub gs_url: String,
    /// Expected sha256 of the tarball (from the pointer); None for a raw object.
    pub sha256: Option<String>,
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// `rvgr-` + 8 digits + `t` + 6 digits + `-` + 8 hex (see `main::launch`).
pub fn is_run_id(s: &str) -> bool {
    let Some(rest) = s.strip_prefix("rvgr-") else {
        return false;
    };
    let b = rest.as_bytes();
    b.len() == 24
        && b[..8].iter().all(u8::is_ascii_digit)
        && b[8] == b't'
        && b[9..15].iter().all(u8::is_ascii_digit)
        && b[15] == b'-'
        && is_hex(&rest[16..], 8)
}

fn safe_object_path(p: &str) -> bool {
    !p.is_empty()
        && !p.starts_with('/')
        && p.split('/')
            .all(|seg| !seg.is_empty() && seg != "." && seg != "..")
        && p.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '-' | '_' | '.'))
}

pub fn parse(s: &str, bucket: &str) -> Result<ResumeSource> {
    if is_run_id(s) {
        return Ok(ResumeSource::Run(s.to_string()));
    }
    let Some(rest) = s.strip_prefix("gs://") else {
        bail!("--resume-from must be a run id (rvgr-YYYYMMDDtHHMMSS-xxxxxxxx) or gs://{bucket}/…tar.gz");
    };
    let Some((b, path)) = rest.split_once('/') else {
        bail!("--resume-from {s:?}: missing object path");
    };
    if b != bucket {
        bail!("--resume-from must be in the artifact bucket gs://{bucket}/ (got gs://{b}/)");
    }
    if !safe_object_path(path) || !path.ends_with(".tar.gz") {
        bail!("--resume-from {s:?}: object path must be a plain …/*.tar.gz without '..'");
    }
    Ok(ResumeSource::Object(s.to_string()))
}

/// Parse a run's `ckpt-latest.json` pointer (written by the onstart script).
pub fn parse_pointer(json: &str, bucket: &str, run_id: &str) -> Result<ResolvedResume> {
    let v: Value = serde_json::from_str(json).context("checkpoint pointer is not JSON")?;
    let obj = v.get("object").and_then(Value::as_str).unwrap_or("");
    let sha = v.get("sha256").and_then(Value::as_str).unwrap_or("");
    let valid_obj = obj
        .strip_prefix("ckpt-")
        .and_then(|r| r.strip_suffix(".tar.gz"))
        .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
    if !valid_obj {
        bail!("checkpoint pointer names an invalid object {obj:?}");
    }
    if !is_hex(sha, 64) {
        bail!("checkpoint pointer has no valid sha256");
    }
    if v.get("run_id").and_then(Value::as_str) != Some(run_id) {
        bail!("checkpoint pointer run_id does not match {run_id}");
    }
    Ok(ResolvedResume {
        gs_url: format!("gs://{bucket}/runs/{run_id}/{obj}"),
        sha256: Some(sha.to_ascii_lowercase()),
    })
}

/// Resolve with the operator's own credentials (reads the pointer object),
/// and confirm the checkpoint object exists, so a typo or an overwritten slot
/// is a preflight blocker rather than a failure after the instance bills.
pub fn resolve(src: &ResumeSource, gcs: &GcsPlan) -> Result<ResolvedResume> {
    let r = resolve_ref(src, gcs)?;
    if !GcsPlan::exists(&r.gs_url) {
        bail!("checkpoint object {} not found", r.gs_url);
    }
    Ok(r)
}

fn resolve_ref(src: &ResumeSource, gcs: &GcsPlan) -> Result<ResolvedResume> {
    match src {
        ResumeSource::Object(u) => Ok(ResolvedResume {
            gs_url: u.clone(),
            sha256: None,
        }),
        ResumeSource::Run(id) => {
            let ptr = format!("gs://{}/runs/{id}/{CKPT_POINTER}", gcs.bucket);
            let body = GcsPlan::cat(&ptr).with_context(|| format!("reading {ptr}"))?;
            parse_pointer(&body, &gcs.bucket, id)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUN: &str = "rvgr-20260927t120000-d3e29a23";

    #[test]
    fn parses_run_ids_and_bucket_objects() {
        assert_eq!(parse(RUN, "b").unwrap(), ResumeSource::Run(RUN.into()));
        let o = format!("gs://b/runs/{RUN}/ckpt-2.tar.gz");
        assert_eq!(parse(&o, "b").unwrap(), ResumeSource::Object(o.clone()));
        for bad in [
            "rvgr-2026-bad",
            "gs://other/runs/x/ckpt-0.tar.gz",
            "gs://b/runs/../secret.tar.gz",
            "gs://b//x.tar.gz",
            "gs://b/runs/x/ckpt-0.zip",
            "gs://b/runs/x/$(id).tar.gz",
            "https://evil/x.tar.gz",
            "/tmp/x.tar.gz",
        ] {
            assert!(parse(bad, "b").is_err(), "{bad}");
        }
    }

    #[test]
    fn pointer_resolves_to_slot_with_sha() {
        let sha = "a".repeat(64);
        let j = format!(
            r#"{{"run_id":"{RUN}","seq":7,"slot":1,"object":"ckpt-1.tar.gz","sha256":"{sha}","at":"x"}}"#
        );
        let r = parse_pointer(&j, "b", RUN).unwrap();
        assert_eq!(r.gs_url, format!("gs://b/runs/{RUN}/ckpt-1.tar.gz"));
        assert_eq!(r.sha256.as_deref(), Some(sha.as_str()));
    }

    #[test]
    fn pointer_rejects_tampering() {
        let sha = "a".repeat(64);
        let mk = |obj: &str, sha: &str, run: &str| {
            format!(r#"{{"run_id":"{run}","object":"{obj}","sha256":"{sha}"}}"#)
        };
        assert!(parse_pointer(&mk("../x.tar.gz", &sha, RUN), "b", RUN).is_err());
        assert!(parse_pointer(&mk("DONE", &sha, RUN), "b", RUN).is_err());
        assert!(parse_pointer(&mk("ckpt-1.tar.gz", "zz", RUN), "b", RUN).is_err());
        assert!(parse_pointer(&mk("ckpt-1.tar.gz", &sha, "rvgr-other"), "b", RUN).is_err());
        assert!(parse_pointer("not json", "b", RUN).is_err());
    }
}
