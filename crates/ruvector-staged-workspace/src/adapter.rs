//! Subprocess adapter protocol for cross-language consumers (WP15's
//! TypeScript `RvfWorkspaceBinder` in `crates/ruvector-sota-bench/harness`).
//!
//! One JSON request on stdin, one JSON response on stdout (see the
//! `staged-workspace-adapter` bin). The **exit code** carries the
//! fail-closed semantics, and the `error` discriminator is a stable,
//! machine-readable contract:
//!
//! | Exit | Meaning | `error` values |
//! |------|---------|----------------|
//! | 0 | operation succeeded | — |
//! | 1 | **integrity rejection** (ADR-318 hard rejection) | `stale-view`, `binding-mismatch`, `unknown-artifact`, `anchor-rejected` |
//! | 2 | **adapter malfunction** (bad input, missing/corrupt state file, IO) | `adapter-error` |
//!
//! A consumer maps exit 1 to "detected compromise" and must treat exit 2
//! (or a crash — no JSON at all) as a loud harness error, never as a
//! silently scored detection. The discriminator strings above are frozen;
//! new failure modes get new strings, existing ones do not change.
//!
//! State is persisted between invocations as a [`WorkspaceSnapshot`] JSON
//! file passed in the request's `state` field.

// Lint suggests `as_chunks`, stable only since Rust 1.88 — past the declared
// `rust-version = "1.77"`. `unknown_lints` keeps older toolchains quiet.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use serde::{Deserialize, Serialize};

use crate::anchor::NoopAnchor;
use crate::artifact::{ArtifactRef, ContentHash};
use crate::workspace::{WorkspaceError, WorkspaceSnapshot, WorkspaceState};

/// Stable error discriminators (see module docs; frozen contract).
pub mod error_code {
    /// View bound to a hash that is no longer the live head.
    pub const STALE_VIEW: &str = "stale-view";
    /// Hash matches the head but the revision id does not (forged binding).
    pub const BINDING_MISMATCH: &str = "binding-mismatch";
    /// No such artifact lineage in the workspace state.
    pub const UNKNOWN_ARTIFACT: &str = "unknown-artifact";
    /// The ADR-312 anchor refused the transition (commit aborted).
    pub const ANCHOR_REJECTED: &str = "anchor-rejected";
    /// Adapter malfunction: malformed request, bad hex/base64, missing or
    /// corrupt state file, IO failure. NOT an integrity signal.
    pub const ADAPTER_ERROR: &str = "adapter-error";
}

/// One artifact to commit: lineage id plus standard base64 (RFC 4648, with
/// padding) of its content bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitArtifact {
    /// Lineage identity (e.g. a workspace-relative path).
    pub artifact_id: String,
    /// Standard base64 of the artifact's content bytes.
    pub content_b64: String,
}

/// A request to the adapter (tagged by `op`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "kebab-case")]
pub enum AdapterRequest {
    /// Commit one or more artifacts, then report per-artifact refs and the
    /// whole-workspace hash.
    Commit {
        /// Actor recorded on the transition records.
        actor_id: String,
        /// Artifacts to commit (applied in the order given).
        artifacts: Vec<CommitArtifact>,
        /// Path to the snapshot JSON file (created if absent, updated on
        /// success).
        state: String,
    },
    /// Validate a bound (artifact_id, content_hash, revision_id) reference
    /// against the live head in the state file.
    Validate {
        /// Lineage the reference points at.
        artifact_id: String,
        /// 64-char hex of the bound content hash.
        content_hash: String,
        /// The bound revision id.
        revision_id: u64,
        /// Path to the snapshot JSON file (must exist — a missing state
        /// file is an adapter error, not an integrity rejection).
        state: String,
    },
}

/// A version reference in responses (hex-rendered hash).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefJson {
    /// Lineage identity.
    pub artifact_id: String,
    /// 64-char lowercase hex of the content hash.
    pub content_hash: String,
    /// Revision id.
    pub revision_id: u64,
}

impl From<&ArtifactRef> for RefJson {
    fn from(r: &ArtifactRef) -> Self {
        Self {
            artifact_id: r.artifact_id.clone(),
            content_hash: r.content_hash.to_hex(),
            revision_id: r.revision_id,
        }
    }
}

/// Adapter response. `ok: true` ⇔ exit code 0.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AdapterResponse {
    /// Successful commit: per-artifact refs plus the workspace hash
    /// (SHA-256 over all sorted lineage heads).
    CommitOk {
        ok: bool,
        artifacts: Vec<RefJson>,
        workspace_hash: String,
    },
    /// Successful validate: the reference is current.
    ValidateOk { ok: bool },
    /// Rejection or malfunction; `error` is a frozen [`error_code`] string.
    Err {
        ok: bool,
        error: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        bound: Option<RefJson>,
        #[serde(skip_serializing_if = "Option::is_none")]
        head: Option<RefJson>,
    },
}

/// Outcome of handling a request: the response plus the process exit code
/// the bin must use (0 ok / 1 integrity rejection / 2 adapter error).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterOutcome {
    /// JSON-serializable response body.
    pub response: AdapterResponse,
    /// 0, 1, or 2 per the module-level table.
    pub exit_code: i32,
}

fn adapter_error(detail: String) -> AdapterOutcome {
    AdapterOutcome {
        response: AdapterResponse::Err {
            ok: false,
            error: error_code::ADAPTER_ERROR.to_string(),
            detail: Some(detail),
            bound: None,
            head: None,
        },
        exit_code: 2,
    }
}

fn integrity_rejection(err: &WorkspaceError) -> AdapterOutcome {
    let (code, bound, head) = match err {
        WorkspaceError::StaleView(c) => (
            error_code::STALE_VIEW,
            Some(RefJson::from(&c.bound)),
            Some(RefJson::from(&c.head)),
        ),
        WorkspaceError::BindingMismatch(c) => (
            error_code::BINDING_MISMATCH,
            Some(RefJson::from(&c.bound)),
            Some(RefJson::from(&c.head)),
        ),
        WorkspaceError::UnknownArtifact(_) => (error_code::UNKNOWN_ARTIFACT, None, None),
        WorkspaceError::AnchorRejected(_) => (error_code::ANCHOR_REJECTED, None, None),
    };
    AdapterOutcome {
        response: AdapterResponse::Err {
            ok: false,
            error: code.to_string(),
            detail: Some(err.to_string()),
            bound,
            head,
        },
        exit_code: 1,
    }
}

/// Decode standard base64 (RFC 4648 §4, `+/` alphabet, `=` padding).
/// Whitespace is not accepted. Returns `None` on any malformed input.
pub fn base64_decode(input: &str) -> Option<Vec<u8>> {
    fn val(b: u8) -> Option<u32> {
        match b {
            b'A'..=b'Z' => Some((b - b'A') as u32),
            b'a'..=b'z' => Some((b - b'a') as u32 + 26),
            b'0'..=b'9' => Some((b - b'0') as u32 + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let last = i == bytes.len() / 4 - 1;
        let pad = if last {
            chunk.iter().rev().take_while(|&&b| b == b'=').count()
        } else {
            0
        };
        if pad > 2 || chunk[..4 - pad].contains(&b'=') {
            return None;
        }
        let mut acc: u32 = 0;
        for &b in &chunk[..4 - pad] {
            acc = (acc << 6) | val(b)?;
        }
        acc <<= 6 * pad as u32;
        let full = [(acc >> 16) as u8, (acc >> 8) as u8, acc as u8];
        out.extend_from_slice(&full[..3 - pad]);
    }
    Some(out)
}

fn load_state(path: &str, must_exist: bool) -> Result<WorkspaceSnapshot, Box<AdapterOutcome>> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text)
            .map_err(|e| Box::new(adapter_error(format!("corrupt state file {path}: {e}")))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !must_exist => {
            Ok(WorkspaceSnapshot {
                lineages: Default::default(),
                transitions: Vec::new(),
                next_sequence: 0,
                next_view_id: 0,
            })
        }
        Err(e) => Err(Box::new(adapter_error(format!(
            "cannot read state file {path}: {e}"
        )))),
    }
}

/// Handle one parsed request against the state file. Pure of process
/// concerns (stdin/stdout/exit) so it is unit-testable; the bin wraps it.
pub fn handle(request: AdapterRequest) -> AdapterOutcome {
    match request {
        AdapterRequest::Commit {
            actor_id,
            artifacts,
            state,
        } => {
            let snapshot = match load_state(&state, false) {
                Ok(s) => s,
                Err(out) => return *out,
            };
            let mut ws = WorkspaceState::from_snapshot(NoopAnchor, snapshot);
            let mut refs = Vec::with_capacity(artifacts.len());
            for a in &artifacts {
                let content = match base64_decode(&a.content_b64) {
                    Some(c) => c,
                    None => {
                        return adapter_error(format!(
                            "invalid base64 content for artifact {:?}",
                            a.artifact_id
                        ))
                    }
                };
                match ws.commit(&a.artifact_id, &content, &actor_id) {
                    Ok(r) => refs.push(RefJson::from(&r)),
                    Err(e) => return integrity_rejection(&e),
                }
            }
            let serialized = match serde_json::to_string(&ws.snapshot()) {
                Ok(s) => s,
                Err(e) => return adapter_error(format!("cannot serialize state: {e}")),
            };
            if let Err(e) = std::fs::write(&state, serialized) {
                return adapter_error(format!("cannot write state file {state}: {e}"));
            }
            AdapterOutcome {
                response: AdapterResponse::CommitOk {
                    ok: true,
                    artifacts: refs,
                    workspace_hash: ws.workspace_hash().to_hex(),
                },
                exit_code: 0,
            }
        }
        AdapterRequest::Validate {
            artifact_id,
            content_hash,
            revision_id,
            state,
        } => {
            let snapshot = match load_state(&state, true) {
                Ok(s) => s,
                Err(out) => return *out,
            };
            let hash = match ContentHash::from_hex(&content_hash) {
                Some(h) => h,
                None => return adapter_error(format!("invalid content_hash hex {content_hash:?}")),
            };
            let ws = WorkspaceState::from_snapshot(NoopAnchor, snapshot);
            let bound = ArtifactRef {
                artifact_id,
                content_hash: hash,
                revision_id,
            };
            match ws.validate_ref(&bound) {
                Ok(()) => AdapterOutcome {
                    response: AdapterResponse::ValidateOk { ok: true },
                    exit_code: 0,
                },
                Err(e) => integrity_rejection(&e),
            }
        }
    }
}

/// Parse a raw request string and handle it (malformed JSON is an adapter
/// error, exit 2 — never an integrity signal).
pub fn handle_raw(input: &str) -> AdapterOutcome {
    match serde_json::from_str::<AdapterRequest>(input) {
        Ok(req) => handle(req),
        Err(e) => adapter_error(format!("malformed request: {e}")),
    }
}
