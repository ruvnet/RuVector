//! Unit tests for `job.rs`: validators and the rendered onstart script.

use super::*;

fn spec() -> JobSpec {
    test_spec()
}

#[test]
fn validators() {
    assert!(validate_sha(&"a".repeat(40)).is_ok());
    assert!(validate_sha("abc123").is_err());
    assert!(validate_image(&spec().image).is_ok());
    assert!(validate_image("nvidia/cuda:latest").is_err());
    assert!(validate_artifact_dir("/workspace/out").is_ok());
    assert!(validate_artifact_dir("/workspace/../etc").is_err());
    assert!(validate_checkpoint_dir("/workspace/ckpt").is_ok());
    for bad in [
        "/workspace/rvgr",
        "/workspace/rvgr/x",
        "/workspace",
        "/",
        "rel",
        "/a/../b",
    ] {
        assert!(validate_checkpoint_dir(bad).is_err(), "{bad}");
    }
}

#[test]
fn onstart_is_safe() {
    let s = render_onstart(&spec());
    assert!(!s.contains("set -x"));
    assert!(s.contains("timeout --signal=TERM --kill-after=60 3600"));
    assert!(s.contains(r"'echo '\''hi'\''; cargo build'"));
    // marker uploaded last
    let log_pos = s.find("RVGR_URL_LOG\" || true").unwrap();
    let marker_pos = s.rfind("put_raw \"$W/marker.json\"").unwrap();
    assert!(log_pos < marker_pos);
    assert!(!s.to_lowercase().contains("vast_api_key"));
}

#[test]
fn checkpoint_loop_rendered_only_when_enabled() {
    let off = render_onstart(&spec());
    assert!(!off.contains("ckpt_loop"));
    assert!(!off.contains("RVGR_URL_CKPT"));
    let mut j = spec();
    j.checkpoint_secs = 600;
    j.checkpoint_dir = "/workspace/ckpt".into();
    let s = render_onstart(&j);
    assert!(s.contains("for (( i = 0; i < 600; i++ ))"));
    assert!(s.contains("slot=$(( n % 3 ))"));
    assert!(s.contains("-C '/workspace/ckpt' ."));
    // Loop started at top level before main, stopped after main, then
    // artifacts -> final checkpoint (inside the transfer window) -> log -> marker.
    let start = s.find("ckpt_loop & CKPT_PID=$!").unwrap();
    let main_call = s.find("( main ); rc=$?").unwrap();
    let stop = s
        .find("touch \"$W/ckpt.stop\"; wait \"$CKPT_PID\"")
        .unwrap();
    let fin = s.find("ckpt_upload  # final").unwrap();
    let art = s.find("put \"$W/artifacts.tar.gz\"").unwrap();
    let dl = s.find("DEADLINE=$(( $(date +%s) + 900 ))").unwrap();
    let undl = s.find("unset DEADLINE").unwrap();
    let log = s.find("put_raw \"$LOG\"").unwrap();
    assert!(start < main_call && main_call < stop && stop < dl && dl < art);
    assert!(art < fin && fin < undl && undl < log);
    // Uploads gated on the job having started.
    assert!(s.contains("[ -f \"$W/job.started\" ] || return 0"));
    assert!(s.find("touch \"$W/job.started\"").unwrap() < s.find("timeout --signal=TERM").unwrap());
}

#[test]
fn final_transfers_bounded_and_not_duplicated() {
    // Same dir: no second tar + upload of identical data after artifacts.
    let mut j = spec();
    j.checkpoint_secs = 600;
    j.checkpoint_dir = "/workspace/out/".into();
    assert!(!j.final_checkpoint());
    let s = render_onstart(&j);
    assert!(s.contains("ckpt_loop & CKPT_PID=$!"));
    assert!(!s.contains("ckpt_upload  # final"));
    assert!(s.contains("artifacts.tar.gz holds the final checkpoint dir"));
    // Stopping the loop cuts an in-flight upload instead of waiting on it.
    assert!(s.contains("if [ -f \"$W/ckpt.stop\" ] ||"));
    assert!(s.contains("rm -f \"$W/ckpt.stop\"\n"));
    // Slot + artifact uploads are cut; the pointer, log and marker never are.
    assert!(s.contains("put() {{ bounded curl".replace("{{", "{").as_str()));
    assert!(s.contains("bounded tar -czf \"$W/ckpt.tar.gz\""));
    assert!(s.contains("bounded tar -czf \"$W/artifacts.tar.gz\""));
    assert!(s.contains("put_raw \"$W/ckpt-latest.json\""));
    assert!(s.contains("put_raw \"$W/marker.json\" \"$url\""));
    j.checkpoint_dir = "/workspace/ckpt".into();
    assert!(j.final_checkpoint());
}

#[test]
fn urls_never_echoed_and_xtrace_off() {
    let mut j = spec();
    j.checkpoint_secs = 60;
    j.resume = Some(ResumeFetch {
        sha256: Some("b".repeat(64)),
    });
    let s = render_onstart(&j);
    let lines: Vec<&str> = s.lines().collect();
    assert_eq!(
        lines[2], "set +x",
        "xtrace must be switched off before anything else"
    );
    assert!(!s.contains("set -x") && !s.contains("-o xtrace") && !s.contains("bash -x"));
    // Every expansion of a URL var is an argument to put/curl or a plain
    // assignment; nothing prints one (echo/printf only in `|| echo "[rvgr]…"`).
    for l in &lines {
        let t = l.trim_start();
        let uses_url = t.contains("\"$RVGR_URL") || t.contains("${!v}");
        if !uses_url {
            continue;
        }
        let head = t.split("||").next().unwrap();
        let via_transfer =
            head.contains("put \"") || head.contains("put_raw \"") || head.starts_with("curl ");
        let assignment = head.contains("url=\"$RVGR_URL_");
        assert!(via_transfer || assignment, "URL used outside put/curl: {t}");
        for bad in ["echo", "printf", "tee", "cat ", ">>", "logger"] {
            assert!(!head.contains(bad), "line may print a URL: {t}");
        }
    }
    // URLs are dropped from the job's environment before the command.
    let unset = s.find("unset \"$v\"; done").unwrap();
    assert!(unset < s.find("timeout --signal=TERM").unwrap());
}

#[test]
fn resume_before_command_and_verified() {
    let mut j = spec();
    j.resume = Some(ResumeFetch {
        sha256: Some("c".repeat(64)),
    });
    let s = render_onstart(&j);
    let head = s.find("test \"$(git rev-parse HEAD)\"").unwrap();
    let fetch = s
        .find("-o \"$W/resume.tar.gz\" \"$RVGR_URL_RESUME\"")
        .unwrap();
    let verify = s.find("sha256sum -c --quiet -").unwrap();
    let extract = s
        .find("tar -xzf \"$W/resume.tar.gz\" --no-same-owner -C '/workspace/out'")
        .unwrap();
    let cmd = s.find("timeout --signal=TERM").unwrap();
    assert!(head < fetch && fetch < verify && verify < extract && extract < cmd);
    assert!(s.contains(&"c".repeat(64)));
    j.resume = Some(ResumeFetch { sha256: None });
    assert!(!render_onstart(&j).contains("sha256sum -c"));
    assert!(!render_onstart(&spec()).contains("RVGR_URL_RESUME"));
}

#[test]
fn env_names_match_script() {
    let mut j = spec();
    j.checkpoint_secs = 60;
    j.checkpoint_ring = 2;
    j.resume = Some(ResumeFetch { sha256: None });
    let names = j.url_env_names();
    assert_eq!(names.len(), 4 + 2 + 1 + 1);
    let s = render_onstart(&j);
    for n in ["RVGR_URL_CKPT_LATEST", "RVGR_URL_RESUME", "RVGR_URL_DONE"] {
        assert!(names.iter().any(|x| x == n) && s.contains(n), "{n}");
    }
    // Slot vars are reached through RVGR_URL_CKPT_$slot.
    assert!(s.contains("v=\"RVGR_URL_CKPT_$slot\""));
}
