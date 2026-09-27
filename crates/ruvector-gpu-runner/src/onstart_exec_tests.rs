//! Executes the rendered onstart script under bash with a stub `git` and a
//! stub `curl` whose "bucket" is a local directory (the mocked store), and a
//! 1-second checkpoint clock. Covers the M6a acceptance tests: an upload
//! happens mid-run, resume places files before the command starts, and a job
//! killed without warning (watchdog destroy) leaves its last checkpoint
//! uploaded. No network, no credentials.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::job::{render_onstart, test_spec, JobSpec, ResumeFetch};

const GIT_STUB: &str = r#"#!/bin/bash
case "$1" in
  clone) mkdir -p "${@: -1}" ;;
  rev-parse) echo "$STUB_SHA" ;;
  *) : ;;
esac
"#;

// PUT: copy --upload-file to $STORE/<name> and log it. GET (-o): copy from
// $STORE/<name>, exit 22 (like curl -f) when absent.
const CURL_STUB: &str = r#"#!/bin/bash
up=""; out=""; url="${@: -1}"
while [ $# -gt 0 ]; do
  case "$1" in --upload-file) up="$2"; shift ;; -o) out="$2"; shift ;; esac; shift
done
name="${url#stub://}"
if [ -n "$up" ]; then
  # One slow PUT for the object named in $STORE/slow.once (then fast again).
  if [ "$(cat "$STORE/slow.once" 2>/dev/null)" = "$name" ]; then
    rm -f "$STORE/slow.once"; trap 'kill $s 2>/dev/null; exit 143' TERM
    sleep 60 & s=$!; wait $s
  fi
  cp "$up" "$STORE/$name.tmp" && mv "$STORE/$name.tmp" "$STORE/$name"
  echo "$(date +%s%N) $name" >> "$STORE/uploads.log"
elif [ -n "$out" ]; then
  [ -f "$STORE/$name" ] || exit 22
  cp "$STORE/$name" "$out"
fi
"#;

struct Sandbox {
    root: PathBuf,
}

impl Sandbox {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!("rvgr-exec-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for d in ["bin", "store/src", "w"] {
            fs::create_dir_all(root.join(d)).unwrap();
        }
        for (n, body) in [("git", GIT_STUB), ("curl", CURL_STUB)] {
            let p = root.join("bin").join(n);
            fs::write(&p, body).unwrap();
            Command::new("chmod").arg("+x").arg(&p).status().unwrap();
        }
        Self { root }
    }
    fn p(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }
    fn store(&self) -> PathBuf {
        self.p("store")
    }
    fn spec(&self, cmd: &str) -> JobSpec {
        let mut j = test_spec();
        j.artifact_dir = self.p("out").display().to_string();
        j.checkpoint_dir = self.p("ckpt").display().to_string();
        j.checkpoint_secs = 1;
        j.checkpoint_ring = 3;
        j.job_timeout_secs = 30;
        j.command = cmd.into();
        j
    }
    fn command(&self, j: &JobSpec) -> Command {
        let script = self.p("onstart.sh");
        fs::write(&script, render_onstart(j)).unwrap();
        let mut c = Command::new("bash");
        c.arg(&script)
            .env(
                "PATH",
                format!(
                    "{}:{}",
                    self.p("bin").display(),
                    std::env::var("PATH").unwrap()
                ),
            )
            .env("RVGR_WORKDIR", self.p("w"))
            .env("STORE", self.store())
            .env("STUB_SHA", &j.git_sha)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for n in j.url_env_names() {
            let obj = match n.as_str() {
                "RVGR_URL_ARTIFACTS" => "artifacts.tar.gz".to_string(),
                "RVGR_URL_LOG" => "job.log".into(),
                "RVGR_URL_DONE" => "DONE".into(),
                "RVGR_URL_FAILED" => "FAILED".into(),
                "RVGR_URL_CKPT_LATEST" => "ckpt-latest.json".into(),
                "RVGR_URL_RESUME" => "src/resume.tar.gz".into(),
                s => format!("ckpt-{}.tar.gz", s.trim_start_matches("RVGR_URL_CKPT_")),
            };
            c.env(n, format!("stub://{obj}"));
        }
        c
    }
    fn run(&self, j: &JobSpec) {
        let st = self.command(j).status().unwrap();
        assert!(st.success(), "onstart exited {st}; log:\n{}", self.log());
    }
    fn log(&self) -> String {
        fs::read_to_string(self.p("w/job.log")).unwrap_or_default()
    }
    fn uploads(&self) -> Vec<String> {
        fs::read_to_string(self.store().join("uploads.log"))
            .unwrap_or_default()
            .lines()
            .map(|l| l.split_once(' ').unwrap().1.to_string())
            .collect()
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn sha256(p: &Path) -> String {
    let o = Command::new("sha256sum").arg(p).output().unwrap();
    String::from_utf8_lossy(&o.stdout)
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
}

/// Checks the pointer's object + sha against the store and returns `file`
/// extracted from the checkpoint it names.
fn read_latest(store: &Path, file: &str) -> String {
    let ptr: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(store.join("ckpt-latest.json")).unwrap()).unwrap();
    let obj = ptr["object"].as_str().unwrap();
    assert_eq!(sha256(&store.join(obj)), ptr["sha256"].as_str().unwrap());
    assert_eq!(ptr["run_id"], "rvgr-test");
    let o = Command::new("tar")
        .args(["-xzOf"])
        .arg(store.join(obj))
        .arg(format!("./{file}"))
        .output()
        .unwrap();
    String::from_utf8_lossy(&o.stdout).trim().to_string()
}

#[test]
fn checkpoint_uploaded_mid_run_then_final_then_marker() {
    let sb = Sandbox::new("mid");
    // The job itself asserts that a checkpoint of its own state reached the
    // store while it was still running, and that it cannot see any URL.
    let cmd = r#"echo step1 > "$RVGR_CHECKPOINT_DIR/state"
for i in $(seq 1 100); do [ -f "$STORE/ckpt-latest.json" ] && break; sleep 0.1; done
obj=$(sed -n 's/.*"object":"\([^"]*\)".*/\1/p' "$STORE/ckpt-latest.json")
tar -xzOf "$STORE/$obj" ./state | grep -qx step1
! env | grep -q RVGR_URL_
echo step2 > "$RVGR_CHECKPOINT_DIR/state"
echo ok > "$RVGR_ARTIFACT_DIR/result""#;
    sb.run(&sb.spec(cmd));
    assert!(sb.store().join("DONE").exists(), "log:\n{}", sb.log());
    assert!(!sb.store().join("FAILED").exists());
    // The final checkpoint carries the state written after the mid-run one.
    assert_eq!(read_latest(&sb.store(), "state"), "step2");
    let up = sb.uploads();
    let last_ckpt = up.iter().rposition(|n| n == "ckpt-latest.json").unwrap();
    let art = up.iter().position(|n| n == "artifacts.tar.gz").unwrap();
    assert!(
        up.iter().filter(|n| *n == "ckpt-latest.json").count() >= 2,
        "{up:?}"
    );
    // Artifacts go first, then the final checkpoint, inside the transfer window.
    assert!(art < last_ckpt, "{up:?}");
    assert_eq!(up.last().unwrap(), "DONE");
    assert!(!sb.log().contains("stub://"), "a URL reached the log");
}

#[test]
fn resume_places_files_before_command() {
    let sb = Sandbox::new("resume");
    let seed = sb.p("seed");
    fs::create_dir_all(&seed).unwrap();
    fs::write(seed.join("resumed.txt"), "hello\n").unwrap();
    let tarball = sb.store().join("src/resume.tar.gz");
    assert!(Command::new("tar")
        .arg("-czf")
        .arg(&tarball)
        .arg("-C")
        .arg(&seed)
        .arg(".")
        .status()
        .unwrap()
        .success());
    let cmd = r#"test "$(cat "$RVGR_CHECKPOINT_DIR/resumed.txt")" = hello
test "$RVGR_RESUMED" = 1
touch "$RVGR_ARTIFACT_DIR/ran""#;
    let mut j = sb.spec(cmd);
    j.resume = Some(ResumeFetch {
        sha256: Some(sha256(&tarball)),
    });
    sb.run(&j);
    assert!(sb.store().join("DONE").exists(), "log:\n{}", sb.log());
    assert!(sb.p("out/ran").exists());

    // A tampered/mismatched checkpoint fails the job before the command runs.
    let sb2 = Sandbox::new("resume-bad");
    fs::copy(&tarball, sb2.store().join("src/resume.tar.gz")).unwrap();
    let mut j2 = sb2.spec("touch \"$RVGR_ARTIFACT_DIR/ran\"");
    j2.resume = Some(ResumeFetch {
        sha256: Some("0".repeat(64)),
    });
    sb2.run(&j2);
    assert!(sb2.store().join("FAILED").exists());
    assert!(
        !sb2.p("out/ran").exists(),
        "command ran despite a bad checkpoint"
    );

    // A missing checkpoint also fails rather than training from scratch.
    let sb3 = Sandbox::new("resume-missing");
    let mut j3 = sb3.spec("touch \"$RVGR_ARTIFACT_DIR/ran\"");
    j3.resume = Some(ResumeFetch { sha256: None });
    sb3.run(&j3);
    assert!(sb3.store().join("FAILED").exists());
    assert!(!sb3.p("out/ran").exists());
}

#[test]
fn killed_job_leaves_last_checkpoint_uploaded() {
    use std::os::unix::process::CommandExt;
    let sb = Sandbox::new("kill");
    // Atomic checkpoint writes (tmp + rename), as jobs are expected to do.
    // The tag lets the test reap the job too: `timeout` moves the command
    // into its own process group, which a real instance destroy also kills.
    let tag = format!("rvgr-kill-tag-{}", std::process::id());
    let cmd = format!(
        r#": {tag}; i=0; while :; do i=$((i+1)); echo $i > "$RVGR_CHECKPOINT_DIR/step.tmp"; mv "$RVGR_CHECKPOINT_DIR/step.tmp" "$RVGR_CHECKPOINT_DIR/step"; sleep 0.2; done"#
    );
    let mut child = sb.command(&sb.spec(&cmd)).process_group(0).spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    while sb
        .uploads()
        .iter()
        .filter(|n| *n == "ckpt-latest.json")
        .count()
        < 2
    {
        assert!(
            Instant::now() < deadline,
            "no checkpoint within 20s:\n{}",
            sb.log()
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    // Watchdog destroy: the whole instance vanishes with no chance to clean up.
    let killed = Command::new("kill")
        .args(["-s", "KILL", "--", &format!("-{}", child.id())])
        .status()
        .unwrap();
    assert!(killed.success());
    let _ = Command::new("pkill").args(["-KILL", "-f", &tag]).status();
    let _ = child.wait();
    std::thread::sleep(Duration::from_millis(300));
    let step: u64 = read_latest(&sb.store(), "step").parse().unwrap();
    assert!(step >= 1);
    assert!(!sb.store().join("DONE").exists() && !sb.store().join("FAILED").exists());
}

#[test]
fn in_instance_timeout_still_uploads_final_checkpoint() {
    let sb = Sandbox::new("timeout");
    let mut j = sb.spec(r#"echo 7 > "$RVGR_CHECKPOINT_DIR/step"; sleep 600"#);
    j.job_timeout_secs = 2;
    j.checkpoint_secs = 60; // no periodic tick before the timeout
    let t = Instant::now();
    sb.run(&j);
    assert!(t.elapsed() < Duration::from_secs(20));
    assert!(sb.store().join("FAILED").exists());
    assert_eq!(read_latest(&sb.store(), "step"), "7");
}

#[test]
fn stop_cuts_in_flight_periodic_upload() {
    let sb = Sandbox::new("cut");
    // The second periodic upload (slot 1) hangs for 60 s; the job ends while
    // it is in flight. The loop stop must cut it rather than wait.
    fs::write(sb.store().join("slow.once"), "ckpt-1.tar.gz").unwrap();
    let cmd = r#"echo a > "$RVGR_CHECKPOINT_DIR/state"
for i in $(seq 1 100); do [ -f "$STORE/ckpt-latest.json" ] && break; sleep 0.1; done
echo a2 > "$RVGR_CHECKPOINT_DIR/state"
sleep 2.5
echo b > "$RVGR_CHECKPOINT_DIR/state""#;
    let t = Instant::now();
    sb.run(&sb.spec(cmd));
    assert!(
        t.elapsed() < Duration::from_secs(20),
        "waited on the hung upload"
    );
    assert!(sb.store().join("DONE").exists(), "log:\n{}", sb.log());
    // The final checkpoint rewrote the cut slot and the pointer names it.
    assert_eq!(read_latest(&sb.store(), "state"), "b");
    assert!(sb.log().contains("upload failed/cut"), "log:\n{}", sb.log());
}

#[test]
fn transfer_window_cut_still_lands_log_and_marker() {
    let sb = Sandbox::new("deadline");
    fs::write(sb.store().join("slow.once"), "artifacts.tar.gz").unwrap();
    let mut j = sb.spec(r#"echo 1 > "$RVGR_ARTIFACT_DIR/r""#);
    j.checkpoint_secs = 60;
    j.final_xfer_secs = 2;
    let t = Instant::now();
    sb.run(&j);
    assert!(t.elapsed() < Duration::from_secs(20));
    assert!(sb.store().join("DONE").exists(), "log:\n{}", sb.log());
    assert!(sb.store().join("job.log").exists());
    assert!(!sb.store().join("artifacts.tar.gz").exists());
    assert!(sb.log().contains("artifact upload failed/cut"));
    assert_eq!(sb.uploads().last().unwrap(), "DONE");
}

#[test]
fn same_dir_uploads_artifacts_once_at_the_end() {
    let sb = Sandbox::new("samedir");
    let mut j = sb.spec(r#"echo z > "$RVGR_ARTIFACT_DIR/z""#);
    j.checkpoint_dir = j.artifact_dir.clone();
    j.checkpoint_secs = 60;
    sb.run(&j);
    assert!(sb.store().join("DONE").exists(), "log:\n{}", sb.log());
    let up = sb.uploads();
    assert!(!up.iter().any(|n| n.starts_with("ckpt-")), "{up:?}");
    assert_eq!(up.iter().filter(|n| *n == "artifacts.tar.gz").count(), 1);
}

#[test]
fn unchanged_checkpoint_is_not_reuploaded() {
    let sb = Sandbox::new("unchanged");
    // One write, then ~4 idle ticks, then one more write: exactly two
    // checkpoint uploads (the idle ticks and the final one are skipped).
    let cmd = r#"echo 1 > "$RVGR_CHECKPOINT_DIR/state"
for i in $(seq 1 100); do [ -f "$STORE/ckpt-latest.json" ] && break; sleep 0.1; done
sleep 4
echo 2 > "$RVGR_CHECKPOINT_DIR/state.tmp"; mv "$RVGR_CHECKPOINT_DIR/state.tmp" "$RVGR_CHECKPOINT_DIR/state"
sleep 2.5"#;
    sb.run(&sb.spec(cmd));
    assert!(sb.store().join("DONE").exists(), "log:\n{}", sb.log());
    let up = sb.uploads();
    let n = up.iter().filter(|n| *n == "ckpt-latest.json").count();
    assert_eq!(n, 2, "{up:?}\n{}", sb.log());
    assert_eq!(read_latest(&sb.store(), "state"), "2");
}

#[test]
fn tar_rc1_tick_is_a_failed_checkpoint() {
    let sb = Sandbox::new("tarrc1");
    // A tar wrapper that reports rc=1 ("file changed as we read it") on the
    // first checkpoint tar: that tick must not name a slot; a later one does.
    let wrap = r#"#!/bin/bash
/usr/bin/tar "$@"; rc=$?
case " $* " in *ckpt.tar.gz*)
  [ -f "$STORE/tar.flaked" ] || { touch "$STORE/tar.flaked"; exit 1; } ;;
esac
exit $rc
"#;
    let p = sb.p("bin/tar");
    fs::write(&p, wrap).unwrap();
    Command::new("chmod").arg("+x").arg(&p).status().unwrap();
    let cmd = r#"echo 1 > "$RVGR_CHECKPOINT_DIR/state"
for i in $(seq 1 100); do [ -f "$STORE/ckpt-latest.json" ] && break; sleep 0.1; done"#;
    sb.run(&sb.spec(cmd));
    assert!(sb.store().join("tar.flaked").exists());
    assert!(
        sb.log().contains("tar failed/cut rc=1"),
        "log:\n{}",
        sb.log()
    );
    // The flaked tick uploaded nothing and did not advance seq.
    let up = sb.uploads();
    assert_eq!(
        up.iter().find(|n| n.starts_with("ckpt-")).unwrap(),
        "ckpt-0.tar.gz",
        "{up:?}"
    );
    assert_eq!(read_latest(&sb.store(), "state"), "1");
}
