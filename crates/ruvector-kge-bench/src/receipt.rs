//! Receipts (`ruvector-kge-bench/receipt@1`, ADR-007 §2.7): provenance (git
//! SHA, dirty flag, binary sha256, host), config + hash, dataset pins and
//! hashes, thread count, timings, per-epoch valid metrics, and — for `final`
//! only — the per-query Bottom/RANDOM test rank vectors plus their sha256.
//! A `verification` carries only hash comparisons and `pass`.
//! `receipt_sha256` is the sha256 of the canonical JSON of the body without
//! that field. No absolute paths, no labels, no triples.

use crate::canon::{canonical_json, sha256_hex};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;

pub const RECEIPT_SCHEMA: &str = "ruvector-kge-bench/receipt@1";

/// Receipt modes: training/selection (`valid`), one-time test scoring
/// (`final`), and post-hoc recomputation (`verification`, never `final`).
pub const MODES: [&str; 3] = ["valid", "final", "verification"];

/// `git rev-parse HEAD` and whether the work tree has uncommitted changes.
pub fn git_state(repo: &Path) -> Value {
    let run = |args: &[&str]| {
        Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let sha = run(&["rev-parse", "HEAD"]);
    let dirty = run(&["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.is_empty());
    json!({ "sha": sha, "dirty": dirty })
}

/// sha256 of the running executable.
pub fn binary_sha256() -> Option<String> {
    std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::read(p).ok())
        .map(sha256_hex)
}

/// Host facts for reproducibility (no hostnames, no paths).
pub fn host() -> Value {
    let cpu = std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
        s.lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split(':').nth(1))
            .map(|m| m.trim().to_string())
    });
    json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "cpu_model": cpu,
        "available_parallelism": std::thread::available_parallelism().map(|n| n.get()).ok(),
    })
}

/// Common provenance block.
pub fn provenance(repo: &Path, threads: usize) -> Value {
    json!({
        "git": git_state(repo),
        "binary_sha256": binary_sha256(),
        "crate": { "name": env!("CARGO_PKG_NAME"), "version": env!("CARGO_PKG_VERSION") },
        "parallel_feature": cfg!(feature = "parallel"),
        "threads": threads,
        "host": host(),
    })
}

/// Stamp `schema` and `receipt_sha256` onto `body` (an object) and return it.
pub fn seal(mut body: Value) -> Result<Value> {
    let m = body
        .as_object_mut()
        .context("receipt body must be an object")?;
    m.insert("schema".into(), json!(RECEIPT_SCHEMA));
    m.remove("receipt_sha256");
    let h = sha256_hex(canonical_json(&body));
    body.as_object_mut()
        .unwrap()
        .insert("receipt_sha256".into(), json!(h));
    Ok(body)
}

/// Validate a receipt against `receipt@1`: required fields by mode and a
/// `receipt_sha256` that matches the body. Returns the mode.
pub fn validate(r: &Value) -> Result<String> {
    let o = r.as_object().context("receipt is not an object")?;
    if o.get("schema") != Some(&json!(RECEIPT_SCHEMA)) {
        bail!("schema is not {RECEIPT_SCHEMA}");
    }
    let mode = o
        .get("mode")
        .and_then(Value::as_str)
        .context("missing mode")?
        .to_string();
    if !MODES.contains(&mode.as_str()) {
        bail!("unknown mode '{mode}'");
    }
    let need = |k: &str| {
        o.get(k)
            .filter(|v| !v.is_null())
            .with_context(|| format!("receipt missing '{k}'"))
    };
    for k in [
        "generated_at",
        "provenance",
        "dataset",
        "config",
        "seed",
        "eval",
    ] {
        need(k)?;
    }
    for k in ["config_hash", "canonical"] {
        need("config")?
            .get(k)
            .with_context(|| format!("config.{k} missing"))?;
    }
    for k in ["file_hashes", "splits_hash", "entity_vocab_hash", "counts"] {
        need("dataset")?
            .get(k)
            .with_context(|| format!("dataset.{k} missing"))?;
    }
    match mode.as_str() {
        "valid" => {
            need("epochs")?
                .as_array()
                .context("epochs must be an array")?;
            need("stopped")?;
        }
        "verification" => {
            // Hash comparisons and a verdict only: a verification never
            // carries test metrics or rank vectors (ADR-007 §2.8).
            if o.contains_key("test") {
                bail!("a verification receipt must not carry test metrics or ranks");
            }
            need("verifies_receipt_sha256")?;
            need("checks")?
                .as_object()
                .context("checks must be an object")?;
            need("pass")?.as_bool().context("pass must be a boolean")?;
            need("tables")?
                .get("weights_sha256")
                .context("tables.weights_sha256 missing")?;
        }
        _ => {
            let t = need("test")?;
            for k in ["bottom", "random", "bottom_sha256", "random_sha256"] {
                t.get("ranks")
                    .and_then(|x| x.get(k))
                    .with_context(|| format!("test.ranks.{k} missing"))?;
            }
            need("tables")?
                .get("weights_sha256")
                .context("tables.weights_sha256 missing")?;
        }
    }
    if mode == "final" {
        need("final")?
            .get("tag_sha")
            .context("final.tag_sha missing")?;
    }
    let claimed = need("receipt_sha256")?
        .as_str()
        .context("receipt_sha256 not a string")?
        .to_string();
    let mut body = r.clone();
    body.as_object_mut().unwrap().remove("receipt_sha256");
    let got = sha256_hex(canonical_json(&body));
    if got != claimed {
        bail!("receipt_sha256 mismatch: body hashes to {got}, receipt claims {claimed}");
    }
    Ok(mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal(mode: &str) -> Value {
        let mut v = json!({
            "mode": mode, "generated_at": "2026-09-27T00:00:00Z", "provenance": {"git": {}},
            "dataset": {"file_hashes": {}, "splits_hash": "x", "entity_vocab_hash": "y", "counts": {}},
            "config": {"config_hash": "h", "canonical": {}}, "seed": 1, "eval": {},
        });
        if mode == "valid" {
            v["epochs"] = json!([]);
            v["stopped"] = json!("max_epochs");
        } else if mode == "verification" {
            v["tables"] = json!({"weights_sha256": "w"});
            v["verifies_receipt_sha256"] = json!("r");
            v["checks"] = json!({"bottom_ranks_sha256": true});
            v["pass"] = json!(true);
        } else {
            v["test"] = json!({"ranks": {"bottom": [1], "random": [1], "bottom_sha256": "a", "random_sha256": "b"}});
            v["tables"] = json!({"weights_sha256": "w"});
            v["final"] = json!({"tag_sha": "t"});
        }
        seal(v).unwrap()
    }

    #[test]
    fn sealed_receipts_validate_and_tamper_fails() {
        for m in MODES {
            let r = minimal(m);
            assert_eq!(validate(&r).unwrap(), m);
            let mut bad = r.clone();
            bad["seed"] = json!(2);
            assert!(validate(&bad).is_err(), "tamper must fail for {m}");
        }
        let mut r = minimal("valid");
        r.as_object_mut().unwrap().remove("epochs");
        assert!(validate(&r).is_err());
        let mut v = minimal("verification");
        v["test"] = json!({"metrics": {}});
        assert!(
            validate(&seal(v).unwrap()).is_err(),
            "verification must not carry test data"
        );
    }
}
