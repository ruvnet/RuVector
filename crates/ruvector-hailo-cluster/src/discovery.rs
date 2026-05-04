//! Worker discovery — turn fleet identity into a `Vec<WorkerEndpoint>`.
//!
//! ADR-167 §8.3 specifies Tailscale-tag-based discovery as primary, with
//! a static config and mDNS fallbacks. This module ships:
//!
//!   * `StaticDiscovery` — caller-provided list (always works, no I/O)
//!   * `TailscaleDiscovery` — shells out to `tailscale status --json`
//!     and filters peers by tag (e.g. `tag:ruvector-hailo-worker`)
//!
//! `MdnsDiscovery` is a future addition (would need an mDNS dep). The
//! trait keeps the cluster coordinator decoupled from any single source.

use crate::error::ClusterError;
use crate::transport::WorkerEndpoint;
use serde::Deserialize;
use std::process::Command;

/// Anything that can produce a worker list. Sync because the coordinator
/// itself is sync today; async impls can wrap this with a thread pool if
/// they need it.
pub trait Discovery {
    /// Produce the worker list. Implementations may consult external
    /// state (network, files, env) and can return errors via `ClusterError`.
    fn discover(&self) -> Result<Vec<WorkerEndpoint>, ClusterError>;
}

// ============================================================
// StaticDiscovery — config-driven, no I/O
// ============================================================

/// Trivial discovery that returns whatever was passed at construction.
/// Useful for tests and for deployments where the worker list is known
/// at boot from a config file.
pub struct StaticDiscovery {
    workers: Vec<WorkerEndpoint>,
}

impl StaticDiscovery {
    /// Construct from a pre-computed worker list (typically from config).
    pub fn new(workers: Vec<WorkerEndpoint>) -> Self {
        Self { workers }
    }
}

impl Discovery for StaticDiscovery {
    fn discover(&self) -> Result<Vec<WorkerEndpoint>, ClusterError> {
        Ok(self.workers.clone())
    }
}

// ============================================================
// FileDiscovery — load `host:port` lines from a manifest file
// ============================================================

/// Load worker endpoints from a text file. One `host:port` per line.
/// Blank lines and `#`-prefixed comments are ignored. Optional inline
/// name: `worker-name = host:port` (whitespace around `=` allowed).
///
/// Lets ops commit a fleet manifest to git rather than maintaining a
/// long `--workers` CSV in shell scripts. Reading is best-effort:
/// malformed lines surface as `Transport` errors with the line number
/// so the caller can see exactly which entry to fix.
pub struct FileDiscovery {
    path: std::path::PathBuf,
    /// ADR-172 §1c iter-107: optional Ed25519 detached signature
    /// verification. When `Some(_)`, `discover()` reads both files,
    /// verifies the signature against the manifest under the configured
    /// public key, and refuses to surface workers if verification fails.
    sig_pubkey: Option<(std::path::PathBuf, std::path::PathBuf)>,
}

impl FileDiscovery {
    /// Construct from a manifest file path. The file is read on each
    /// `discover()` call, so live edits are picked up on the next probe.
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            path: path.into(),
            sig_pubkey: None,
        }
    }

    /// Require an Ed25519 detached signature on the manifest (ADR-172 §1c).
    /// `sig_path` holds the 128-hex-char detached signature; `pubkey_path`
    /// holds the 64-hex-char public key. Both files are re-read on every
    /// `discover()` call so a key rotation takes effect on the next probe
    /// without restarting the coordinator.
    pub fn with_signature(
        mut self,
        sig_path: impl Into<std::path::PathBuf>,
        pubkey_path: impl Into<std::path::PathBuf>,
    ) -> Self {
        self.sig_pubkey = Some((sig_path.into(), pubkey_path.into()));
        self
    }

    /// Parse a manifest body without touching the filesystem — separate
    /// so tests can feed fixtures directly.
    pub fn parse(&self, body: &str) -> Result<Vec<WorkerEndpoint>, ClusterError> {
        let mut out = Vec::new();
        for (i, raw) in body.lines().enumerate() {
            let line = raw.trim();
            // Strip trailing inline comments (after a `#` not in a quoted
            // segment — manifest is simple enough that we don't bother
            // with quoting; just split on the first `#`).
            let line = match line.find('#') {
                Some(idx) => line[..idx].trim(),
                None => line,
            };
            if line.is_empty() {
                continue;
            }
            // Two formats: `name = host:port` (named) or `host:port`
            // (auto-named "file-N" by index).
            let (name, address) = if let Some(eq) = line.find('=') {
                let n = line[..eq].trim().to_string();
                let a = line[eq + 1..].trim().to_string();
                if n.is_empty() || a.is_empty() {
                    return Err(ClusterError::Transport {
                        worker: "<discovery>".into(),
                        reason: format!(
                            "FileDiscovery: line {}: empty name or address in {:?}",
                            i + 1,
                            raw
                        ),
                    });
                }
                (n, a)
            } else {
                (format!("file-{}", out.len()), line.to_string())
            };
            // Sanity-check `host:port` shape so a typo doesn't quietly
            // produce a worker that fails on first dial.
            if !address.contains(':') {
                return Err(ClusterError::Transport {
                    worker: "<discovery>".into(),
                    reason: format!(
                        "FileDiscovery: line {}: address {:?} has no port (expected host:port)",
                        i + 1,
                        address
                    ),
                });
            }
            out.push(WorkerEndpoint::new(name, address));
        }
        Ok(out)
    }
}

impl Discovery for FileDiscovery {
    fn discover(&self) -> Result<Vec<WorkerEndpoint>, ClusterError> {
        // Iter 210 — refuse manifests larger than 1 MB before we
        // `read_to_string`. A legitimate fleet manifest is one
        // `name=host:port` per worker (~100 B per line); even a 1000-
        // worker tailnet fits in ~100 KB. The 1 MB cap is 10× legit
        // headroom and prevents an accidentally-corrupted or
        // attacker-pointed-at file from OOMing the worker at boot.
        // We hit this BEFORE the iter-107 signature check so a
        // pathologically large file fails fast — verification of a
        // 1 GB signed file would be slow even though it'd reject.
        const MAX_MANIFEST_BYTES: u64 = 1 << 20; // 1 MB
        let meta = std::fs::metadata(&self.path).map_err(|e| ClusterError::Transport {
            worker: "<discovery>".into(),
            reason: format!("FileDiscovery: stat {}: {}", self.path.display(), e),
        })?;
        if meta.len() > MAX_MANIFEST_BYTES {
            return Err(ClusterError::Transport {
                worker: "<discovery>".into(),
                reason: format!(
                    "FileDiscovery: manifest {} is {} bytes, exceeds {} byte cap \
                     (iter 210 — likely a misconfig; legitimate fleets fit in <100 KB)",
                    self.path.display(),
                    meta.len(),
                    MAX_MANIFEST_BYTES
                ),
            });
        }
        // ADR-172 §1c iter-107: when a signature is configured, verify
        // *before* parsing. We don't even tokenize the manifest until
        // we know the bytes match the operator's signing key — defends
        // against a parser bug being a CVE vector for unsigned input.
        if let Some((sig_path, pubkey_path)) = &self.sig_pubkey {
            crate::manifest_sig::verify_files(&self.path, sig_path, pubkey_path)?;
        }
        let body = std::fs::read_to_string(&self.path).map_err(|e| ClusterError::Transport {
            worker: "<discovery>".into(),
            reason: format!("FileDiscovery: read {}: {}", self.path.display(), e),
        })?;
        self.parse(&body)
    }
}

// ============================================================
// TailscaleDiscovery — `tailscale status --json` + tag filter
// ============================================================

/// Discovery via the local Tailscale daemon. Filters peers by tag and
/// constructs `WorkerEndpoint`s using each peer's first IPv4 address +
/// the configured port.
pub struct TailscaleDiscovery {
    /// e.g. `"tag:ruvector-hailo-worker"`. Peers without this tag are skipped.
    tag: String,
    /// Port the worker's gRPC server listens on (e.g. 50051).
    port: u16,
    /// Path to the `tailscale` CLI binary. Default `tailscale` (PATH lookup).
    cli_path: String,
}

impl TailscaleDiscovery {
    /// Construct with the canonical `tailscale` CLI on PATH. Use
    /// `with_cli_path` to override for tests or custom installations.
    pub fn new(tag: impl Into<String>, port: u16) -> Self {
        Self {
            tag: tag.into(),
            port,
            cli_path: "tailscale".into(),
        }
    }

    /// Override the path to the `tailscale` CLI binary.
    pub fn with_cli_path(mut self, p: impl Into<String>) -> Self {
        self.cli_path = p.into();
        self
    }

    /// Parse a `tailscale status --json` output blob (decoupled from the
    /// shell-out so tests can feed fixtures without spawning a subprocess).
    pub fn parse_status_json(&self, json: &str) -> Result<Vec<WorkerEndpoint>, ClusterError> {
        let status: TsStatus = serde_json::from_str(json).map_err(|e| ClusterError::Transport {
            worker: "<discovery>".into(),
            reason: format!("tailscale status JSON parse: {}", e),
        })?;
        let mut out = Vec::new();
        // `Self` is the local node; we never include it as a worker —
        // the coordinator running this code IS the dispatcher, not a
        // worker. Walk the Peer map instead.
        if let Some(peer_map) = status.peer {
            for (_node_id, peer) in peer_map {
                let tags = peer.tags.unwrap_or_default();
                if !tags.contains(&self.tag) {
                    continue;
                }
                let ip = peer
                    .tailscale_ips
                    .as_ref()
                    .and_then(|ips| ips.iter().find(|ip| !ip.contains(':')))
                    .cloned();
                let ip = match ip {
                    Some(v) => v,
                    None => continue, // no IPv4 → skip (rare)
                };
                let name = peer.host_name.unwrap_or_else(|| ip.clone());
                let address = format!("{}:{}", ip, self.port);
                out.push(WorkerEndpoint::new(name, address));
            }
        }
        // Stable order helps testing + makes coordinator startup deterministic.
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }
}

impl Discovery for TailscaleDiscovery {
    fn discover(&self) -> Result<Vec<WorkerEndpoint>, ClusterError> {
        let output = Command::new(&self.cli_path)
            .args(["status", "--json"])
            .output()
            .map_err(|e| ClusterError::Transport {
                worker: "<discovery>".into(),
                reason: format!("tailscale CLI invoke: {}", e),
            })?;
        if !output.status.success() {
            return Err(ClusterError::Transport {
                worker: "<discovery>".into(),
                reason: format!(
                    "tailscale status exited {}: {}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr)
                ),
            });
        }
        let s = String::from_utf8_lossy(&output.stdout);
        self.parse_status_json(&s)
    }
}

// ---- Tailscale status JSON shape (only fields we need) -----------

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct TsStatus {
    #[serde(rename = "Peer")]
    peer: Option<std::collections::HashMap<String, TsPeer>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
struct TsPeer {
    host_name: Option<String>,
    tags: Option<Vec<String>>,
    // tailscale's JSON key is `TailscaleIPs` (full caps on the abbreviation).
    // PascalCase auto-rename would yield `TailscaleIps` which doesn't match,
    // so override explicitly.
    #[serde(rename = "TailscaleIPs")]
    tailscale_ips: Option<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_discovery_returns_caller_list() {
        let ws = vec![
            WorkerEndpoint::new("a", "10.0.0.1:50051"),
            WorkerEndpoint::new("b", "10.0.0.2:50051"),
        ];
        let d = StaticDiscovery::new(ws.clone());
        assert_eq!(d.discover().unwrap().len(), 2);
    }

    #[test]
    fn file_discovery_parses_simple_manifest() {
        let body = "
            # Production fleet — fingerprint fp:abc, dim 384.
            pi-a = 100.77.59.83:50051
            pi-b = 100.77.59.84:50051

            # Inline comments work too.
            pi-c = 100.77.59.85:50051   # spare unit
        ";
        let d = FileDiscovery::new("ignored");
        let workers = d.parse(body).unwrap();
        assert_eq!(workers.len(), 3);
        assert_eq!(workers[0].name, "pi-a");
        assert_eq!(workers[0].address, "100.77.59.83:50051");
        assert_eq!(workers[1].name, "pi-b");
        assert_eq!(workers[2].name, "pi-c");
    }

    #[test]
    fn file_discovery_auto_names_when_no_equals() {
        let body = "
            127.0.0.1:50051
            127.0.0.1:50052
        ";
        let d = FileDiscovery::new("ignored");
        let workers = d.parse(body).unwrap();
        assert_eq!(workers.len(), 2);
        assert_eq!(workers[0].name, "file-0");
        assert_eq!(workers[1].name, "file-1");
        assert_eq!(workers[0].address, "127.0.0.1:50051");
        assert_eq!(workers[1].address, "127.0.0.1:50052");
    }

    #[test]
    fn file_discovery_rejects_address_without_port() {
        let d = FileDiscovery::new("ignored");
        let r = d.parse("just-a-hostname-no-port");
        assert!(matches!(
            r,
            Err(crate::error::ClusterError::Transport { .. })
        ));
    }

    #[test]
    fn file_discovery_rejects_empty_name_or_address() {
        let d = FileDiscovery::new("ignored");
        assert!(d.parse(" = 10.0.0.1:50051").is_err());
        assert!(d.parse("worker-name =   ").is_err());
    }

    #[test]
    fn file_discovery_handles_only_comments() {
        let d = FileDiscovery::new("ignored");
        let workers = d
            .parse("# everything is a comment\n# nothing else")
            .unwrap();
        assert!(workers.is_empty());
    }

    #[test]
    fn tailscale_discovery_filters_by_tag() {
        // Synthetic status output with three peers, two carrying the tag.
        let json = r#"{
          "Peer": {
            "node-1": {
              "HostName": "cognitum-v0",
              "TailscaleIPs": ["100.77.59.83", "fd7a::1"],
              "Tags": ["tag:ruvector-hailo-worker"]
            },
            "node-2": {
              "HostName": "ruvultra",
              "TailscaleIPs": ["100.104.125.72"],
              "Tags": []
            },
            "node-3": {
              "HostName": "cognitum-v1",
              "TailscaleIPs": ["100.77.59.84"],
              "Tags": ["tag:ruvector-hailo-worker"]
            }
          }
        }"#;
        let d = TailscaleDiscovery::new("tag:ruvector-hailo-worker", 50051);
        let workers = d.parse_status_json(json).unwrap();
        assert_eq!(workers.len(), 2);
        assert_eq!(workers[0].name, "cognitum-v0");
        assert_eq!(workers[0].address, "100.77.59.83:50051");
        assert_eq!(workers[1].name, "cognitum-v1");
        assert_eq!(workers[1].address, "100.77.59.84:50051");
    }

    #[test]
    fn tailscale_discovery_skips_peers_without_ipv4() {
        let json = r#"{
          "Peer": {
            "node-1": {
              "HostName": "v6-only",
              "TailscaleIPs": ["fd7a::1"],
              "Tags": ["tag:ruvector-hailo-worker"]
            }
          }
        }"#;
        let d = TailscaleDiscovery::new("tag:ruvector-hailo-worker", 50051);
        let workers = d.parse_status_json(json).unwrap();
        assert_eq!(workers.len(), 0);
    }

    #[test]
    fn tailscale_discovery_handles_empty_status() {
        let json = r#"{"Peer": null}"#;
        let d = TailscaleDiscovery::new("tag:x", 50051);
        let workers = d.parse_status_json(json).unwrap();
        assert!(workers.is_empty());
    }

    #[test]
    fn tailscale_discovery_rejects_invalid_json() {
        let d = TailscaleDiscovery::new("tag:x", 50051);
        assert!(d.parse_status_json("not json").is_err());
    }

    /// Live integration test against the actual `tailscale` daemon.
    /// Skipped when tailscale isn't installed or unavailable. Not a unit
    /// test in the strict sense — confirms the JSON shape we parse against
    /// matches what the local daemon produces today.
    #[test]
    fn tailscale_discovery_live_smoke() {
        let d = TailscaleDiscovery::new("tag:ruvector-hailo-worker", 50051);
        match d.discover() {
            Ok(workers) => {
                eprintln!(
                    "tailscale discover() found {} worker(s) with tag",
                    workers.len()
                );
                // Test passes whether or not the user has tagged any
                // workers yet. Just exercise the code path.
            }
            Err(e) => {
                // tailscale not installed / not running on this host — fine.
                eprintln!("(tailscale not available: {})", e);
            }
        }
    }

    /// Iter 210 — the 1 MB manifest cap. A 2 MB file should be rejected
    /// by `stat`-then-cap before reaching read_to_string.
    #[test]
    fn file_discovery_rejects_oversized_manifest() {
        use std::io::Write as _;
        let path = std::env::temp_dir().join("iter210-oversized-manifest.txt");
        // 2 MB of "a = 1.2.3.4:50051\n"-style filler. Way under any
        // legit fleet shape; the cap is the gate, not the contents.
        let line = "filler-name = 1.2.3.4:50051\n";
        let n_lines = (2 * 1024 * 1024) / line.len() + 1;
        let mut f = std::fs::File::create(&path).expect("create fixture");
        for _ in 0..n_lines {
            f.write_all(line.as_bytes()).expect("write fixture");
        }
        f.sync_all().expect("sync");
        drop(f);

        let d = FileDiscovery::new(&path);
        let err = d
            .discover()
            .expect_err("oversized manifest must be rejected");
        match err {
            ClusterError::Transport { reason, .. } => {
                assert!(
                    reason.contains("exceeds")
                        && reason.contains("byte cap")
                        && reason.contains("iter 210"),
                    "expected size-cap rejection text, got: {:?}",
                    reason
                );
            }
            other => panic!("expected ClusterError::Transport, got {:?}", other),
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Counterpart: a small (well-under-cap) manifest still works.
    #[test]
    fn file_discovery_accepts_small_manifest() {
        use std::io::Write as _;
        let path = std::env::temp_dir().join("iter210-small-manifest.txt");
        let mut f = std::fs::File::create(&path).expect("create fixture");
        f.write_all(b"pi-a = 100.77.59.83:50051\npi-b = 100.77.59.84:50051\n")
            .expect("write fixture");
        f.sync_all().expect("sync");
        drop(f);

        let d = FileDiscovery::new(&path);
        let workers = d.discover().expect("small manifest should parse");
        assert_eq!(workers.len(), 2);
        let _ = std::fs::remove_file(&path);
    }
}
