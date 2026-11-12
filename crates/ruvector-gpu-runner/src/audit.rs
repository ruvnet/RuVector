//! Append-only JSONL audit log + live-instance state files (for SIGKILL
//! recovery via `reap`). Callers must never pass secrets or signed URLs.

use anyhow::{Context, Result};
use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn state_dir() -> PathBuf {
    let base = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("ruvector-gpu-runner")
}

pub struct Audit {
    path: PathBuf,
}

impl Audit {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&self, mut rec: Value) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        rec["ts"] = Value::String(chrono::Utc::now().to_rfc3339());
        let line = serde_json::to_string(&rec)?;
        debug_assert!(
            !line.contains("X-Goog-Signature"),
            "signed URL leaked into audit"
        );
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening audit log {}", self.path.display()))?;
        writeln!(f, "{line}")?;
        Ok(())
    }
}

fn live_dir() -> PathBuf {
    state_dir().join("live")
}

/// Record a live instance immediately after create returns its id.
pub fn mark_live(id: u64, rec: &Value) -> Result<()> {
    fs::create_dir_all(live_dir())?;
    fs::write(
        live_dir().join(format!("{id}.json")),
        serde_json::to_vec_pretty(rec)?,
    )?;
    Ok(())
}

pub fn clear_live(id: u64) {
    let _ = fs::remove_file(live_dir().join(format!("{id}.json")));
}

pub fn live_ids() -> Vec<u64> {
    fs::read_dir(live_dir())
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().to_str()?.strip_suffix(".json")?.parse().ok())
                .collect()
        })
        .unwrap_or_default()
}
