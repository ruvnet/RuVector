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
    /// Directory checkpointed periodically and restored by a resume.
    pub checkpoint_dir: String,
    /// Checkpoint upload interval in seconds; 0 disables periodic uploads.
    pub checkpoint_secs: u64,
    /// Number of rolling `ckpt-<i>.tar.gz` slots (2..=MAX_CKPT_RING when enabled).
    pub checkpoint_ring: usize,
    /// Fetch a checkpoint (URL in `RVGR_URL_RESUME`) before the command runs.
    pub resume: Option<ResumeFetch>,
    /// Wall-clock window for the post-job transfers (artifacts + final
    /// checkpoint). Past it they are cut so log + marker still land before
    /// the local watchdog; the launcher reserves this on top of 600 s.
    pub final_xfer_secs: u64,
}

#[derive(Debug, Clone)]
pub struct ResumeFetch {
    /// Expected tarball sha256 (64 hex), verified on the instance when known.
    pub sha256: Option<String>,
}

/// Instance work dir holding the log and staging tarballs.
pub const WORKDIR: &str = "/workspace/rvgr";

/// Same directory up to trailing slashes.
pub fn same_dir(a: &str, b: &str) -> bool {
    a.trim_end_matches('/') == b.trim_end_matches('/')
}

/// Checkpoint dir: absolute, no `..`, and neither inside nor an ancestor of
/// [`WORKDIR`] (the checkpoint tarball is staged there and must not include itself).
pub fn validate_checkpoint_dir(dir: &str) -> Result<()> {
    validate_artifact_dir(dir)?;
    let d = dir.trim_end_matches('/');
    if d == WORKDIR
        || d.starts_with(&format!("{WORKDIR}/"))
        || WORKDIR.starts_with(&format!("{d}/"))
    {
        bail!("--checkpoint-dir must not be {WORKDIR}, inside it, or one of its parents");
    }
    Ok(())
}

impl JobSpec {
    pub fn checkpointing(&self) -> bool {
        self.checkpoint_secs > 0
    }
    /// A separate final checkpoint is uploaded only when the checkpoint dir
    /// differs from the artifact dir; otherwise `artifacts.tar.gz` already
    /// holds the same final state and a second tar + upload is skipped.
    pub fn final_checkpoint(&self) -> bool {
        self.checkpointing() && !same_dir(&self.checkpoint_dir, &self.artifact_dir)
    }
    /// Env var names the create body must fill with signed URLs.
    pub fn url_env_names(&self) -> Vec<String> {
        let mut v: Vec<String> = ["ARTIFACTS", "LOG", "DONE", "FAILED"]
            .iter()
            .map(|s| format!("RVGR_URL_{s}"))
            .collect();
        if self.checkpointing() {
            v.extend((0..self.checkpoint_ring).map(|i| format!("RVGR_URL_CKPT_{i}")));
            v.push("RVGR_URL_CKPT_LATEST".into());
        }
        if self.resume.is_some() {
            v.push("RVGR_URL_RESUME".into());
        }
        v
    }
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

/// Checkpoint helpers, rendered only when periodic checkpoints are enabled
/// (the `${!v}` indirection would trip `set -u` on unset slot vars).
/// Uploads start only once `$W/job.started` exists, so a half-cloned tree or
/// a resume in progress is never captured. The pointer is written after the
/// slot upload succeeds and the sequence advances only when the pointer
/// lands, so the pointer never names a slot that is being overwritten.
fn render_ckpt_fns(j: &JobSpec) -> String {
    format!(
        r#"ckpt_upload() {{
  [ -f "$W/job.started" ] || return 0
  [ -n "$(ls -A {ck} 2>/dev/null)" ] || return 0
  local n slot v sum trc
  n=$(cat "$W/ckpt.seq" 2>/dev/null || echo 0); slot=$(( n % {ring} ))
  bounded tar -czf "$W/ckpt.tar.gz" -C {ck} . ; trc=$?
  if [ "$trc" -gt 1 ]; then echo "[rvgr] checkpoint $n: tar failed/cut rc=$trc"; return 0; fi
  sum=$(sha256sum "$W/ckpt.tar.gz" | cut -d' ' -f1)
  v="RVGR_URL_CKPT_$slot"
  put "$W/ckpt.tar.gz" "${{!v}}" || {{ echo "[rvgr] checkpoint $n: upload failed/cut"; return 0; }}
  printf '{{"run_id":"%s","seq":%d,"slot":%d,"object":"ckpt-%d.tar.gz","sha256":"%s","at":"%s"}}\n' {run_id} "$n" "$slot" "$slot" "$sum" "$(date -u +%FT%TZ)" > "$W/ckpt-latest.json"
  # The tiny pointer is never cut: a slot is named only once fully uploaded.
  if put_raw "$W/ckpt-latest.json" "$RVGR_URL_CKPT_LATEST"; then
    echo $(( n + 1 )) > "$W/ckpt.seq"
    echo "[rvgr] checkpoint $n -> ckpt-$slot.tar.gz sha256=$sum"
  else
    echo "[rvgr] checkpoint $n: pointer upload failed"
  fi
}}
ckpt_loop() {{
  local i
  while :; do
    for (( i = 0; i < {secs}; i++ )); do [ -f "$W/ckpt.stop" ] && return 0; sleep 1; done
    ckpt_upload
  done
}}
"#,
        ck = sh_quote(&j.checkpoint_dir),
        ring = j.checkpoint_ring,
        secs = j.checkpoint_secs,
        run_id = j.run_id,
    )
}

/// Resume block (inside `main`, under `set -e`): any failure fails the job
/// rather than silently training from scratch on the paid instance.
fn render_resume(j: &JobSpec, r: &ResumeFetch) -> String {
    let check = match &r.sha256 {
        Some(s) => format!(
            "  echo {}'  '\"$W/resume.tar.gz\" | sha256sum -c --quiet -\n  echo \"[rvgr] resume: sha256 verified\"\n",
            sh_quote(s)
        ),
        None => "  echo \"[rvgr] resume: no sha256 recorded (explicit object)\"\n".into(),
    };
    format!(
        r#"  echo "[rvgr] resume: fetching checkpoint"
  curl -fsS --retry 5 --retry-delay 5 -o "$W/resume.tar.gz" "$RVGR_URL_RESUME"
{check}  tar -xzf "$W/resume.tar.gz" --no-same-owner -C {ck}
  rm -f "$W/resume.tar.gz"
  export RVGR_RESUMED=1
  echo "[rvgr] resume: extracted into "{ck}
"#,
        ck = sh_quote(&j.checkpoint_dir),
    )
}

/// Render the onstart script. Order is load-bearing: resume before the
/// command; after it, artifacts, then the final checkpoint (both inside the
/// `final_xfer_secs` window), then log, then marker, so a marker's presence
/// implies the log exists and the transfers finished or were cut.
pub fn render_onstart(j: &JobSpec) -> String {
    let ck = j.checkpointing();
    let fns = if ck {
        render_ckpt_fns(j)
    } else {
        String::new()
    };
    let resume = j
        .resume
        .as_ref()
        .map(|r| render_resume(j, r))
        .unwrap_or_default();
    // Stopping the loop aborts an in-flight periodic tar/upload (`bounded`
    // watches ckpt.stop) instead of waiting for it; the aborted slot is not
    // the one the pointer names, and the sequence did not advance.
    let (start, stop) = if ck {
        (
            "ckpt_loop & CKPT_PID=$!\n",
            "touch \"$W/ckpt.stop\"; wait \"$CKPT_PID\"; rm -f \"$W/ckpt.stop\"\n",
        )
    } else {
        ("", "")
    };
    let fin = if j.final_checkpoint() {
        "ckpt_upload  # final checkpoint\n"
    } else if ck {
        "echo \"[rvgr] no separate final checkpoint: artifacts.tar.gz holds the final checkpoint dir\"\n"
    } else {
        ""
    };
    format!(
        r#"#!/bin/bash
# ruvector-gpu-runner onstart (run {run_id}). Never enable xtrace: env holds signed URLs.
set +x
set -uo pipefail
W="${{RVGR_WORKDIR:-{workdir}}}"; mkdir -p "$W" {art} {ckd}; LOG="$W/job.log"
exec > >(tee -a "$LOG") 2>&1
# Run "$@" in the background; cut it (rc 124) when the checkpoint loop is told
# to stop or the post-job transfer window ($DEADLINE, epoch secs) has passed.
bounded() {{
  "$@" & local p=$!
  while kill -0 "$p" 2>/dev/null; do
    if [ -f "$W/ckpt.stop" ] || {{ [ -n "${{DEADLINE:-}}" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; }}; then
      kill -TERM "$p" 2>/dev/null; wait "$p" 2>/dev/null; return 124
    fi
    sleep 0.2
  done
  wait "$p"
}}
put_raw() {{ curl -fsS --retry 5 --retry-delay 5 -X PUT -H 'Content-Type: application/octet-stream' --upload-file "$1" "$2" >/dev/null; }}
# `bounded` runs curl itself (not a function), so the kill reaches curl.
put() {{ bounded curl -fsS --retry 5 --retry-delay 5 -X PUT -H 'Content-Type: application/octet-stream' --upload-file "$1" "$2" >/dev/null; }}
{fns}main() {{
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
  export RVGR_ARTIFACT_DIR={art} RVGR_CHECKPOINT_DIR={ckd}
{resume}  for v in $(compgen -e | grep '^RVGR_URL_' || true); do unset "$v"; done  # the job never sees URLs
  touch "$W/job.started"
  timeout --signal=TERM --kill-after=60 {timeout} bash -c {cmd}
}}
{start}( main ); rc=$?
{stop}if [ "$rc" -eq 0 ]; then status=DONE; url="$RVGR_URL_DONE"; else status=FAILED; url="$RVGR_URL_FAILED"; fi
echo "[rvgr] job finished rc=$rc status=$status $(date -u +%FT%TZ)"
# Artifacts first (the deliverable), then the final checkpoint, both cut at
# the transfer window so the log + marker below always get their reserve.
DEADLINE=$(( $(date +%s) + {xfer} ))
rm -f "$W/artifacts.tar.gz"; bounded tar -czf "$W/artifacts.tar.gz" -C {art} . ; trc=$?
if [ "$trc" -le 1 ]; then put "$W/artifacts.tar.gz" "$RVGR_URL_ARTIFACTS" || echo "[rvgr] artifact upload failed/cut"; else echo "[rvgr] artifact tar failed/cut rc=$trc"; fi
{fin}unset DEADLINE
exec >/dev/null 2>&1; sleep 2; sync  # close the tee pipe so job.log is fully flushed
put_raw "$LOG" "$RVGR_URL_LOG" || true
printf '{{"run_id":"%s","status":"%s","rc":%d,"sha":"%s","finished_at":"%s"}}\n' {run_id} "$status" "$rc" {sha} "$(date -u +%FT%TZ)" > "$W/marker.json"
put_raw "$W/marker.json" "$url"
"#,
        run_id = j.run_id,
        workdir = WORKDIR,
        art = sh_quote(&j.artifact_dir),
        ckd = sh_quote(&j.checkpoint_dir),
        git_ref = sh_quote(&j.git_ref),
        repo = sh_quote(&j.repo),
        sha = j.git_sha,
        timeout = j.job_timeout_secs,
        xfer = j.final_xfer_secs,
        cmd = sh_quote(&j.command),
    )
}

#[cfg(test)]
pub(crate) fn test_spec() -> JobSpec {
    JobSpec {
        run_id: "rvgr-test".into(),
        image: "nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:".to_string() + &"a".repeat(64),
        repo: "https://github.com/ruvnet/ruvector.git".into(),
        git_ref: "feat/openjev".into(),
        git_sha: "f".repeat(40),
        command: "echo 'hi'; cargo build".into(),
        artifact_dir: "/workspace/out".into(),
        job_timeout_secs: 3600,
        checkpoint_dir: "/workspace/out".into(),
        checkpoint_secs: 0,
        checkpoint_ring: 3,
        resume: None,
        final_xfer_secs: 900,
    }
}

#[cfg(test)]
#[path = "job_tests.rs"]
mod tests;
