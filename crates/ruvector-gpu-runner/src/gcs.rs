//! GCS artifact sink: V4 signed PUT URLs minted locally, one per fixed object.
//!
//! Strategy: `gcloud storage sign-url --impersonate-service-account=<SA>`.
//! The SA holds only `roles/storage.objectCreator` on the artifact bucket and
//! has NO keys; impersonation signs via IAM signBlob, so no credential file
//! ever exists on disk or on the instance. The instance gets four URLs, each
//! scoped to one object, PUT-only, expiring after the job's max duration.
//! gcloud caps impersonated signed URLs at 12h, which bounds `--max-hours`.

use anyhow::{bail, Context, Result};
use std::process::Command;

/// Max validity of an impersonation-signed URL (gcloud limit).
pub const MAX_SIGNED_HOURS: f64 = 12.0;
pub const OBJECTS: [&str; 4] = ["artifacts.tar.gz", "job.log", "DONE", "FAILED"];

#[derive(Debug, Clone)]
pub struct GcsPlan {
    pub bucket: String,
    pub signer_sa: String,
    pub project: String,
    pub run_id: String,
}

pub struct SignedUrls {
    pub artifacts: String,
    pub log: String,
    pub done: String,
    pub failed: String,
}

impl GcsPlan {
    pub fn prefix(&self) -> String {
        format!("gs://{}/runs/{}", self.bucket, self.run_id)
    }

    fn gcloud(args: &[&str]) -> Result<std::process::Output> {
        Command::new("gcloud")
            .args(args)
            .output()
            .context("running gcloud")
    }

    /// Readiness checks; returns the list of problems (empty = ready).
    pub fn readiness(&self) -> Vec<String> {
        let mut out = Vec::new();
        let b = format!("gs://{}", self.bucket);
        match Self::gcloud(&["storage", "buckets", "describe", &b, "--format=value(name)"]) {
            Ok(o) if o.status.success() => {}
            _ => out.push(format!("bucket {b} missing or unreadable")),
        }
        match Self::gcloud(&[
            "iam",
            "service-accounts",
            "describe",
            &self.signer_sa,
            &format!("--project={}", self.project),
            "--format=value(email)",
        ]) {
            Ok(o) if o.status.success() => {}
            _ => out.push(format!("signer SA {} does not exist", self.signer_sa)),
        }
        out
    }

    pub fn setup_commands(&self) -> String {
        let sa_name = self.signer_sa.split('@').next().unwrap_or("rvgr-uploader");
        format!(
            "gcloud iam service-accounts create {sa_name} --project={p} --display-name='ruvector-gpu-runner uploader (no keys)'\n\
             gcloud storage buckets add-iam-policy-binding gs://{b} --member=serviceAccount:{sa} --role=roles/storage.objectCreator\n\
             gcloud iam service-accounts add-iam-policy-binding {sa} --project={p} --member=user:$(gcloud config get-value account) --role=roles/iam.serviceAccountTokenCreator",
            p = self.project, b = self.bucket, sa = self.signer_sa
        )
    }

    fn sign_put(&self, object: &str, hours: f64) -> Result<String> {
        let secs = (hours * 3600.0).ceil() as u64;
        let o = Self::gcloud(&[
            "storage",
            "sign-url",
            &format!("{}/{object}", self.prefix()),
            "--http-verb=PUT",
            &format!("--duration={secs}s"),
            "--headers=content-type=application/octet-stream",
            &format!("--impersonate-service-account={}", self.signer_sa),
            "--format=value(signed_url)",
            "--quiet",
        ])?;
        if !o.status.success() {
            bail!("sign-url for {object} failed (stderr suppressed: may reference credentials)");
        }
        let url = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if !url.starts_with("https://") {
            bail!("sign-url for {object} returned no URL");
        }
        Ok(url)
    }

    pub fn sign_all(&self, hours: f64) -> Result<SignedUrls> {
        if hours > MAX_SIGNED_HOURS {
            bail!("signed URL duration {hours}h exceeds {MAX_SIGNED_HOURS}h");
        }
        Ok(SignedUrls {
            artifacts: self.sign_put(OBJECTS[0], hours)?,
            log: self.sign_put(OBJECTS[1], hours)?,
            done: self.sign_put(OBJECTS[2], hours)?,
            failed: self.sign_put(OBJECTS[3], hours)?,
        })
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
