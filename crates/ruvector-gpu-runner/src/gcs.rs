//! GCS artifact sink: V4 signed URLs minted locally, one per fixed object.
//!
//! Strategy: `gcloud storage sign-url --impersonate-service-account=<SA>`.
//! The uploader SA holds only `roles/storage.objectCreator` on the artifact
//! bucket and has NO keys; impersonation signs via IAM signBlob, so no
//! credential file ever exists on disk or on the instance. The instance gets
//! PUT URLs, each scoped to one object and expiring after the job's max
//! duration. gcloud caps impersonated signed URLs at 12h, which bounds
//! `--max-hours`.
//!
//! Resume (`--resume-from`) needs a read: the uploader cannot read, so a
//! second keyless SA (`reader_sa`, `roles/storage.objectViewer`) signs one
//! short-lived GET URL for exactly one checkpoint object.

use anyhow::{bail, Context, Result};
use std::process::Command;

/// Max validity of an impersonation-signed URL (gcloud limit).
pub const MAX_SIGNED_HOURS: f64 = 12.0;
pub const OBJECTS: [&str; 4] = ["artifacts.tar.gz", "job.log", "DONE", "FAILED"];
/// Pointer object naming the newest checkpoint slot (`{seq,slot,object,sha256}`).
pub const CKPT_POINTER: &str = "ckpt-latest.json";
/// Upper bound on the rolling checkpoint ring.
pub const MAX_CKPT_RING: usize = 8;

/// Name of rolling checkpoint slot `i` (`ckpt-<i>.tar.gz`).
pub fn ckpt_slot(i: usize) -> String {
    format!("ckpt-{i}.tar.gz")
}

#[derive(Debug, Clone)]
pub struct GcsPlan {
    pub bucket: String,
    pub signer_sa: String,
    /// Keyless SA with objectViewer, used only to sign resume GET URLs.
    pub reader_sa: String,
    pub project: String,
    pub region: String,
    pub run_id: String,
}

pub struct SignedUrls {
    pub artifacts: String,
    pub log: String,
    pub done: String,
    pub failed: String,
    /// One PUT URL per ring slot (empty when checkpointing is off).
    pub ckpt_slots: Vec<String>,
    /// PUT URL for [`CKPT_POINTER`] (None when checkpointing is off).
    pub ckpt_pointer: Option<String>,
}

/// `gcloud storage sign-url` arguments. Pure so tests can pin the shape:
/// GET URLs must not carry a signed content-type header (a plain
/// `curl -o` would then fail signature validation).
pub fn sign_args(gs_url: &str, verb: &str, secs: u64, sa: &str, region: &str) -> Vec<String> {
    let mut a = vec![
        "storage".to_string(),
        "sign-url".into(),
        gs_url.into(),
        format!("--http-verb={verb}"),
        format!("--duration={secs}s"),
    ];
    if verb == "PUT" {
        a.push("--headers=content-type=application/octet-stream".into());
    }
    a.extend([
        format!("--impersonate-service-account={sa}"),
        format!("--region={region}"),
        "--format=value(signed_url)".into(),
        "--quiet".into(),
    ]);
    a
}

impl GcsPlan {
    pub fn prefix(&self) -> String {
        format!("gs://{}/runs/{}", self.bucket, self.run_id)
    }

    fn gcloud<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> Result<std::process::Output> {
        Command::new("gcloud")
            .args(args)
            .output()
            .context("running gcloud")
    }

    fn sa_exists(&self, sa: &str) -> bool {
        matches!(Self::gcloud(&[
            "iam",
            "service-accounts",
            "describe",
            sa,
            &format!("--project={}", self.project),
            "--format=value(email)",
        ]), Ok(o) if o.status.success())
    }

    /// Readiness checks; returns the list of problems (empty = ready).
    /// The reader SA is only required when a resume is requested.
    pub fn readiness(&self, need_reader: bool) -> Vec<String> {
        let mut out = Vec::new();
        let b = format!("gs://{}", self.bucket);
        match Self::gcloud(&["storage", "buckets", "describe", &b, "--format=value(name)"]) {
            Ok(o) if o.status.success() => {}
            _ => out.push(format!("bucket {b} missing or unreadable")),
        }
        if !self.sa_exists(&self.signer_sa) {
            out.push(format!("signer SA {} does not exist", self.signer_sa));
        }
        if need_reader && !self.sa_exists(&self.reader_sa) {
            out.push(format!(
                "resume reader SA {} does not exist",
                self.reader_sa
            ));
        }
        out
    }

    pub fn setup_commands(&self, need_reader: bool) -> String {
        let sa_name = self.signer_sa.split('@').next().unwrap_or("rvgr-uploader");
        let mut s = format!(
            "gcloud iam service-accounts create {sa_name} --project={p} --display-name='ruvector-gpu-runner uploader (no keys)'\n\
             gcloud storage buckets add-iam-policy-binding gs://{b} --member=serviceAccount:{sa} --role=roles/storage.objectCreator\n\
             gcloud iam service-accounts add-iam-policy-binding {sa} --project={p} --member=user:$(gcloud config get-value account) --role=roles/iam.serviceAccountTokenCreator",
            p = self.project, b = self.bucket, sa = self.signer_sa
        );
        if need_reader {
            let rd = self.reader_sa.split('@').next().unwrap_or("rvgr-reader");
            s.push_str(&format!(
                "\n# resume reader (read-only, keyless):\n\
                 gcloud iam service-accounts create {rd} --project={p} --display-name='ruvector-gpu-runner resume reader (no keys)'\n\
                 gcloud storage buckets add-iam-policy-binding gs://{b} --member=serviceAccount:{sa} --role=roles/storage.objectViewer\n\
                 gcloud iam service-accounts add-iam-policy-binding {sa} --project={p} --member=user:$(gcloud config get-value account) --role=roles/iam.serviceAccountTokenCreator",
                p = self.project, b = self.bucket, sa = self.reader_sa
            ));
        }
        s
    }

    fn sign(&self, gs_url: &str, verb: &str, hours: f64, sa: &str) -> Result<String> {
        if !(hours > 0.0 && hours <= MAX_SIGNED_HOURS) {
            bail!("signed URL duration {hours}h outside (0, {MAX_SIGNED_HOURS}]");
        }
        let secs = (hours * 3600.0).ceil() as u64;
        let o = Self::gcloud(&sign_args(gs_url, verb, secs, sa, &self.region))?;
        let name = gs_url.rsplit('/').next().unwrap_or("");
        if !o.status.success() {
            bail!(
                "sign-url {verb} for {name} failed (stderr suppressed: may reference credentials)"
            );
        }
        let url = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if !url.starts_with("https://") {
            bail!("sign-url {verb} for {name} returned no URL");
        }
        Ok(url)
    }

    fn sign_put(&self, object: &str, hours: f64) -> Result<String> {
        self.sign(
            &format!("{}/{object}", self.prefix()),
            "PUT",
            hours,
            &self.signer_sa,
        )
    }

    /// Short-lived GET URL for one existing object (resume). Signed by the
    /// read-only SA; `gs_url` must already be validated to lie in `bucket`.
    pub fn sign_get(&self, gs_url: &str, hours: f64) -> Result<String> {
        if !gs_url.starts_with(&format!("gs://{}/", self.bucket)) {
            bail!("refusing to sign GET outside gs://{}/", self.bucket);
        }
        self.sign(gs_url, "GET", hours, &self.reader_sa)
    }

    /// `ckpt_ring == 0` disables the checkpoint URLs.
    pub fn sign_all(&self, hours: f64, ckpt_ring: usize) -> Result<SignedUrls> {
        if hours > MAX_SIGNED_HOURS {
            bail!("signed URL duration {hours}h exceeds {MAX_SIGNED_HOURS}h");
        }
        if ckpt_ring > MAX_CKPT_RING {
            bail!("checkpoint ring {ckpt_ring} > {MAX_CKPT_RING}");
        }
        let ckpt_slots = (0..ckpt_ring)
            .map(|i| self.sign_put(&ckpt_slot(i), hours))
            .collect::<Result<Vec<_>>>()?;
        let ckpt_pointer = if ckpt_ring > 0 {
            Some(self.sign_put(CKPT_POINTER, hours)?)
        } else {
            None
        };
        Ok(SignedUrls {
            artifacts: self.sign_put(OBJECTS[0], hours)?,
            log: self.sign_put(OBJECTS[1], hours)?,
            done: self.sign_put(OBJECTS[2], hours)?,
            failed: self.sign_put(OBJECTS[3], hours)?,
            ckpt_slots,
            ckpt_pointer,
        })
    }

    /// Read a small object with the operator's own gcloud credentials.
    pub fn cat(gs_url: &str) -> Result<String> {
        let o = Self::gcloud(&["storage", "cat", gs_url])?;
        if !o.status.success() {
            bail!("gcloud storage cat {gs_url} failed");
        }
        Ok(String::from_utf8_lossy(&o.stdout).into_owned())
    }

    /// True when the object exists (operator creds, read-only).
    pub fn exists(gs_url: &str) -> bool {
        matches!(Self::gcloud(&["storage", "ls", gs_url]), Ok(o) if o.status.success())
    }

    /// Poll for a terminal marker using the operator's own gcloud creds.
    pub fn marker(&self) -> Option<&'static str> {
        let o = Self::gcloud(&["storage", "ls", &format!("{}/", self.prefix())]).ok()?;
        let s = String::from_utf8_lossy(&o.stdout);
        if s.lines().any(|l| l.ends_with("/DONE")) {
            Some("DONE")
        } else if s.lines().any(|l| l.ends_with("/FAILED")) {
            Some("FAILED")
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_args_are_read_only_and_headerless() {
        let a = sign_args(
            "gs://b/runs/r/ckpt-1.tar.gz",
            "GET",
            3000,
            "rd@p.iam",
            "us-central1",
        );
        assert!(a.contains(&"--http-verb=GET".to_string()));
        assert!(a.contains(&"--duration=3000s".to_string()));
        assert!(a.contains(&"--impersonate-service-account=rd@p.iam".to_string()));
        assert!(!a.iter().any(|x| x.starts_with("--headers")));
        let p = sign_args("gs://b/runs/r/DONE", "PUT", 60, "up@p.iam", "us-central1");
        assert!(p.contains(&"--headers=content-type=application/octet-stream".to_string()));
    }

    #[test]
    fn sign_get_refuses_other_buckets_and_bad_durations() {
        let g = GcsPlan {
            bucket: "b".into(),
            signer_sa: "up@p".into(),
            reader_sa: "rd@p".into(),
            project: "p".into(),
            region: "us-central1".into(),
            run_id: "rvgr-x".into(),
        };
        // Both refusals happen before any gcloud call.
        assert!(g.sign_get("gs://other/runs/x/ckpt-0.tar.gz", 1.0).is_err());
        assert!(g.sign_get("gs://bx/runs/x/ckpt-0.tar.gz", 1.0).is_err());
        assert!(g
            .sign("gs://b/runs/x/ckpt-0.tar.gz", "GET", 13.0, "rd@p")
            .is_err());
        assert!(g
            .setup_commands(true)
            .contains("roles/storage.objectViewer"));
        assert!(!g.setup_commands(false).contains("objectViewer"));
    }
}
