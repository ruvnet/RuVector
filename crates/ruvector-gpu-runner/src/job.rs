//! Job contract: pinned image, pinned git SHA, onstart script rendering.
//!
//! The onstart script receives signed upload URLs via env vars
//! (`RVGR_URL_ARTIFACTS`, `RVGR_URL_LOG`, `RVGR_URL_DONE`, `RVGR_URL_FAILED`).
//! It never echoes them and never runs with `set -x`. No vast.ai API key and
//! no GCP credential is ever placed on the instance.

use anyhow::{bail, Context, Result};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct JobSpec {
    pub run_id: String,
    pub image: String,
    pub repo: String,
    pub git_ref: String,
    pub git_sha: String,
    pub command: String,
    pub artifact_dir: String,
    /// Hard in-instance timeout for the job command, seconds.
    pub job_timeout_secs: u64,
}

pub fn validate_sha(sha: &str) -> Result<()> {
    if sha.len() != 40 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("--git-sha must be a full 40-char hex commit SHA, got {sha:?}");
    }
    Ok(())
}

/// Require `name[:tag]@sha256:<64 hex>` so the container can't drift.
pub fn validate_image(image: &str) -> Result<()> {
    let Some((_, digest)) = image.split_once("@sha256:") else {
        bail!("--image must be pinned by digest (…@sha256:<64 hex>), got {image:?}");
    };
    if digest.len() != 64 || !digest.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("image digest must be 64 hex chars");
    }
    Ok(())
}

pub fn validate_artifact_dir(dir: &str) -> Result<()> {
    if !dir.starts_with('/') || dir.contains("..") || dir == "/" {
        bail!("--artifact-dir must be an absolute path without '..' and not '/'");
    }
    Ok(())
}

/// Verify the SHA is reachable from `refs/heads/<ref>` on the remote the
/// instance will clone from. Returns the remote tip SHA.
pub fn verify_remote_sha(
    local_repo: &str,
    repo_url: &str,
    git_ref: &str,
    sha: &str,
) -> Result<String> {
    let out = Command::new("git")
        .args(["ls-remote", repo_url, &format!("refs/heads/{git_ref}")])
        .output()
        .context("running git ls-remote")?;
    let tip = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string();
    if !out.status.success() || tip.is_empty() {
        bail!("branch {git_ref:?} does not exist on {repo_url} (not pushed?) — the instance could not clone it");
    }
    if tip == sha {
        return Ok(tip);
    }
    let fetch = Command::new("git")
        .args([
            "-C",
            local_repo,
            "fetch",
            "--quiet",
            repo_url,
            &format!("refs/heads/{git_ref}"),
        ])
        .status()
        .context("git fetch")?;
    if !fetch.success() {
        bail!("git fetch of {git_ref} failed");
    }
    let anc = Command::new("git")
        .args([
            "-C",
            local_repo,
            "merge-base",
            "--is-ancestor",
            sha,
            "FETCH_HEAD",
        ])
        .status()
        .context("git merge-base")?;
    if !anc.success() {
        bail!("sha {sha} is not reachable from remote {git_ref} (tip {tip}) — push it first");
    }
    Ok(tip)
}

/// POSIX single-quote a string for safe embedding in bash.
pub fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// Render the onstart script. Order is load-bearing: artifacts, then log,
/// then marker — so a marker's presence implies the log/artifacts exist.
pub fn render_onstart(j: &JobSpec) -> String {
    format!(
        r#"#!/bin/bash
# ruvector-gpu-runner onstart (run {run_id}). Never enable xtrace: env holds signed URLs.
set -uo pipefail
W=/workspace/rvgr; mkdir -p "$W" {art}; LOG="$W/job.log"
exec > >(tee -a "$LOG") 2>&1
put() {{ curl -fsS --retry 5 --retry-delay 5 -X PUT -H 'Content-Type: application/octet-stream' --upload-file "$1" "$2" >/dev/null; }}
main() {{
  set -e
  echo "[rvgr] start $(date -u +%FT%TZ) run={run_id}"
  if ! command -v git >/dev/null || ! command -v curl >/dev/null; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends git curl ca-certificates
  fi
  nvidia-smi || true
  git clone --quiet --filter=blob:none --branch {git_ref} {repo} "$W/src"
  cd "$W/src"
  git checkout --quiet {sha}
  test "$(git rev-parse HEAD)" = {sha}
  echo "[rvgr] HEAD verified {sha}"
  export RVGR_ARTIFACT_DIR={art}
  timeout --signal=TERM --kill-after=60 {timeout} bash -c {cmd}
}}
( main ); rc=$?
if [ "$rc" -eq 0 ]; then status=DONE; url="$RVGR_URL_DONE"; else status=FAILED; url="$RVGR_URL_FAILED"; fi
echo "[rvgr] job finished rc=$rc status=$status $(date -u +%FT%TZ)"
tar -czf "$W/artifacts.tar.gz" -C {art} . || echo "[rvgr] tar failed"
[ -f "$W/artifacts.tar.gz" ] && put "$W/artifacts.tar.gz" "$RVGR_URL_ARTIFACTS" || echo "[rvgr] artifact upload failed"
exec >/dev/null 2>&1; sleep 2; sync  # close the tee pipe so job.log is fully flushed
put "$LOG" "$RVGR_URL_LOG" || true
printf '{{"run_id":"%s","status":"%s","rc":%d,"sha":"%s","finished_at":"%s"}}\n' {run_id} "$status" "$rc" {sha} "$(date -u +%FT%TZ)" > "$W/marker.json"
put "$W/marker.json" "$url"
"#,
        run_id = j.run_id,
        art = sh_quote(&j.artifact_dir),
        git_ref = sh_quote(&j.git_ref),
        repo = sh_quote(&j.repo),
        sha = j.git_sha,
        timeout = j.job_timeout_secs,
        cmd = sh_quote(&j.command),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> JobSpec {
        JobSpec {
            run_id: "rvgr-test".into(),
            image: "nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:".to_string() + &"a".repeat(64),
            repo: "https://github.com/ruvnet/ruvector.git".into(),
            git_ref: "feat/openjev".into(),
            git_sha: "f".repeat(40),
            command: "echo 'hi'; cargo build".into(),
            artifact_dir: "/workspace/out".into(),
            job_timeout_secs: 3600,
        }
    }

    #[test]
    fn validators() {
        assert!(validate_sha(&"a".repeat(40)).is_ok());
        assert!(validate_sha("abc123").is_err());
        assert!(validate_image(&spec().image).is_ok());
        assert!(validate_image("nvidia/cuda:latest").is_err());
        assert!(validate_artifact_dir("/workspace/out").is_ok());
        assert!(validate_artifact_dir("/workspace/../etc").is_err());
    }

    #[test]
    fn onstart_is_safe() {
        let s = render_onstart(&spec());
        assert!(!s.contains("set -x"));
        assert!(s.contains("timeout --signal=TERM --kill-after=60 3600"));
        assert!(s.contains(r"'echo '\''hi'\''; cargo build'"));
        // marker uploaded last
        let log_pos = s.find("RVGR_URL_LOG\" || true").unwrap();
        let marker_pos = s.rfind("put \"$W/marker.json\"").unwrap();
        assert!(log_pos < marker_pos);
        assert!(!s.to_lowercase().contains("vast_api_key"));
    }
}
