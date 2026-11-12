//! Subprocess-adapter protocol tests: exit-code contract (0 ok / 1
//! integrity rejection / 2 adapter malfunction) and frozen error
//! discriminators, exercised through `adapter::handle_raw` — the exact
//! function the `staged-workspace-adapter` bin wraps.

use ruvector_staged_workspace::adapter::{
    base64_decode, error_code, handle_raw, AdapterOutcome, AdapterResponse,
};
use ruvector_staged_workspace::ContentHash;

/// Unique state-file path per test; removed on drop.
struct StateFile(std::path::PathBuf);

impl StateFile {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "staged-workspace-adapter-test-{}-{name}.json",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        Self(path)
    }
    fn path(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for StateFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn commit_request(state: &str, artifacts: &[(&str, &str)]) -> String {
    let list: Vec<String> = artifacts
        .iter()
        .map(|(id, b64)| format!("{{\"artifact_id\":\"{id}\",\"content_b64\":\"{b64}\"}}"))
        .collect();
    format!(
        "{{\"op\":\"commit\",\"actor_id\":\"harness\",\"artifacts\":[{}],\"state\":\"{state}\"}}",
        list.join(",")
    )
}

fn validate_request(state: &str, id: &str, hash_hex: &str, rev: u64) -> String {
    format!(
        "{{\"op\":\"validate\",\"artifact_id\":\"{id}\",\"content_hash\":\"{hash_hex}\",\"revision_id\":{rev},\"state\":\"{state}\"}}"
    )
}

fn expect_err(outcome: &AdapterOutcome) -> (&str, Option<&str>, Option<&str>) {
    match &outcome.response {
        AdapterResponse::Err {
            ok,
            error,
            bound,
            head,
            ..
        } => {
            assert!(!ok);
            (
                error.as_str(),
                bound.as_ref().map(|r| r.content_hash.as_str()),
                head.as_ref().map(|r| r.content_hash.as_str()),
            )
        }
        other => panic!("expected error response, got {other:?}"),
    }
}

// "abc" and "abcd" in standard base64.
const ABC_B64: &str = "YWJj";
const ABCD_B64: &str = "YWJjZA==";

#[test]
fn commit_then_validate_current_exits_zero() {
    let state = StateFile::new("commit-validate-ok");
    let out = handle_raw(&commit_request(state.path(), &[("doc.md", ABC_B64)]));
    assert_eq!(out.exit_code, 0);
    let (hash_hex, ws_hash) = match &out.response {
        AdapterResponse::CommitOk {
            ok,
            artifacts,
            workspace_hash,
        } => {
            assert!(ok);
            assert_eq!(artifacts.len(), 1);
            assert_eq!(artifacts[0].content_hash, ContentHash::of(b"abc").to_hex());
            assert_eq!(artifacts[0].revision_id, 1);
            (artifacts[0].content_hash.clone(), workspace_hash.clone())
        }
        other => panic!("expected CommitOk, got {other:?}"),
    };
    // Workspace hash is recomputable from the documented formula:
    // SHA-256 over sorted (u64-LE id length, id bytes, 32 hash bytes).
    let mut expected = Vec::new();
    expected.extend_from_slice(&(b"doc.md".len() as u64).to_le_bytes());
    expected.extend_from_slice(b"doc.md");
    expected.extend_from_slice(&ContentHash::of(b"abc").0);
    assert_eq!(ws_hash, ContentHash::of(&expected).to_hex());

    let out = handle_raw(&validate_request(state.path(), "doc.md", &hash_hex, 1));
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.response, AdapterResponse::ValidateOk { ok: true });
}

#[test]
fn stale_reference_across_invocations_exits_one_with_stale_view() {
    let state = StateFile::new("stale");
    // Invocation 1: commit v1.
    let v1_hash = ContentHash::of(b"abc").to_hex();
    assert_eq!(
        handle_raw(&commit_request(state.path(), &[("doc.md", ABC_B64)])).exit_code,
        0
    );
    // Invocation 2 (fresh handle over the same state file): commit v2.
    assert_eq!(
        handle_raw(&commit_request(state.path(), &[("doc.md", ABCD_B64)])).exit_code,
        0
    );
    // Invocation 3: the v1 reference is now stale — exit 1, discriminator
    // "stale-view", with bound/head refs for the executor's report.
    let out = handle_raw(&validate_request(state.path(), "doc.md", &v1_hash, 1));
    assert_eq!(out.exit_code, 1);
    let (code, bound, head) = expect_err(&out);
    assert_eq!(code, error_code::STALE_VIEW);
    assert_eq!(bound, Some(v1_hash.as_str()));
    assert_eq!(head, Some(ContentHash::of(b"abcd").to_hex().as_str()));
}

#[test]
fn forged_revision_exits_one_with_binding_mismatch() {
    let state = StateFile::new("forged-rev");
    handle_raw(&commit_request(state.path(), &[("doc.md", ABC_B64)]));
    let out = handle_raw(&validate_request(
        state.path(),
        "doc.md",
        &ContentHash::of(b"abc").to_hex(),
        99,
    ));
    assert_eq!(out.exit_code, 1);
    assert_eq!(expect_err(&out).0, error_code::BINDING_MISMATCH);
}

#[test]
fn unknown_artifact_exits_one_with_unknown_artifact() {
    let state = StateFile::new("unknown");
    handle_raw(&commit_request(state.path(), &[("doc.md", ABC_B64)]));
    let out = handle_raw(&validate_request(
        state.path(),
        "no/such/file",
        &ContentHash::of(b"abc").to_hex(),
        1,
    ));
    assert_eq!(out.exit_code, 1);
    assert_eq!(expect_err(&out).0, error_code::UNKNOWN_ARTIFACT);
}

#[test]
fn malfunctions_exit_two_with_adapter_error_never_one() {
    // Malformed request JSON.
    let out = handle_raw("{not json");
    assert_eq!(out.exit_code, 2);
    assert_eq!(expect_err(&out).0, error_code::ADAPTER_ERROR);

    // Bad base64 content.
    let state = StateFile::new("bad-b64");
    let out = handle_raw(&commit_request(state.path(), &[("doc.md", "!!!!")]));
    assert_eq!(out.exit_code, 2);
    assert_eq!(expect_err(&out).0, error_code::ADAPTER_ERROR);

    // Missing state file on validate: setup malfunction, loud — must NOT
    // be scored as a detected compromise.
    let missing = StateFile::new("never-created");
    let out = handle_raw(&validate_request(
        missing.path(),
        "doc.md",
        &ContentHash::of(b"abc").to_hex(),
        1,
    ));
    assert_eq!(out.exit_code, 2);
    assert_eq!(expect_err(&out).0, error_code::ADAPTER_ERROR);

    // Bad hex in content_hash.
    let state = StateFile::new("bad-hex");
    handle_raw(&commit_request(state.path(), &[("doc.md", ABC_B64)]));
    let out = handle_raw(&validate_request(state.path(), "doc.md", "zz", 1));
    assert_eq!(out.exit_code, 2);
    assert_eq!(expect_err(&out).0, error_code::ADAPTER_ERROR);
}

#[test]
fn multi_artifact_commit_orders_workspace_hash_by_id() {
    let a = StateFile::new("multi-a");
    let b = StateFile::new("multi-b");
    // Same artifacts, different commit order -> same workspace hash.
    let out_ab = handle_raw(&commit_request(
        a.path(),
        &[("a.md", ABC_B64), ("b.md", ABCD_B64)],
    ));
    let out_ba = handle_raw(&commit_request(
        b.path(),
        &[("b.md", ABCD_B64), ("a.md", ABC_B64)],
    ));
    let hash = |o: &AdapterOutcome| match &o.response {
        AdapterResponse::CommitOk { workspace_hash, .. } => workspace_hash.clone(),
        other => panic!("expected CommitOk, got {other:?}"),
    };
    assert_eq!(hash(&out_ab), hash(&out_ba));
}

#[test]
fn empty_artifacts_commit_returns_workspace_hash_over_current_heads() {
    let state = StateFile::new("empty-commit");
    let hash_of = |o: &AdapterOutcome| match &o.response {
        AdapterResponse::CommitOk {
            artifacts,
            workspace_hash,
            ..
        } => (artifacts.len(), workspace_hash.clone()),
        other => panic!("expected CommitOk, got {other:?}"),
    };
    // Fresh state, zero artifacts: valid — WP15's first-party cases can
    // carry no seed files. workspace_hash is the hash over zero heads.
    let out = handle_raw(&commit_request(state.path(), &[]));
    assert_eq!(out.exit_code, 0);
    assert_eq!(hash_of(&out), (0, ContentHash::of(b"").to_hex()));
    // After a real commit, an empty commit reports the hash over the
    // existing heads without mutating anything.
    handle_raw(&commit_request(state.path(), &[("doc.md", ABC_B64)]));
    let before = handle_raw(&commit_request(state.path(), &[]));
    assert_eq!(before.exit_code, 0);
    let mut expected = Vec::new();
    expected.extend_from_slice(&(b"doc.md".len() as u64).to_le_bytes());
    expected.extend_from_slice(b"doc.md");
    expected.extend_from_slice(&ContentHash::of(b"abc").0);
    assert_eq!(hash_of(&before), (0, ContentHash::of(&expected).to_hex()));
}

#[test]
fn commit_is_append_only_and_never_an_integrity_rejection() {
    // The commit path has no StaleView/BindingMismatch/UnknownArtifact
    // branches: an existing lineage always accepts a new version as the
    // next revision. With NoopAnchor the only conceivable commit rejection
    // (anchor-rejected) cannot fire, so commit can never exit 1 here.
    let state = StateFile::new("append-only");
    assert_eq!(
        handle_raw(&commit_request(state.path(), &[("doc.md", ABC_B64)])).exit_code,
        0
    );
    assert_eq!(
        handle_raw(&commit_request(state.path(), &[("doc.md", ABCD_B64)])).exit_code,
        0
    );
    // Re-committing the OLD content is a NEW revision (append-only), not a
    // rejection: staleness is a property of reads, never of writes.
    let out = handle_raw(&commit_request(state.path(), &[("doc.md", ABC_B64)]));
    assert_eq!(out.exit_code, 0);
    match &out.response {
        AdapterResponse::CommitOk { artifacts, .. } => {
            assert_eq!(artifacts[0].revision_id, 3);
            assert_eq!(artifacts[0].content_hash, ContentHash::of(b"abc").to_hex());
        }
        other => panic!("expected CommitOk, got {other:?}"),
    }
}

#[test]
fn base64_decoder_matches_rfc_4648_vectors() {
    assert_eq!(base64_decode(""), Some(vec![]));
    assert_eq!(base64_decode("YQ=="), Some(b"a".to_vec()));
    assert_eq!(base64_decode("YWI="), Some(b"ab".to_vec()));
    assert_eq!(base64_decode("YWJj"), Some(b"abc".to_vec()));
    assert_eq!(base64_decode("YWJjZA=="), Some(b"abcd".to_vec()));
    assert_eq!(base64_decode("YWJj\n"), None); // whitespace rejected
    assert_eq!(base64_decode("YQ=A"), None); // padding mid-chunk
    assert_eq!(base64_decode("Y"), None); // bad length
}
