//! Worker configuration sourced from environment variables.
//!
//! All keys carry the `RUVIEW_VITALS_` prefix so they don't collide
//! with the iter-123 `ruview-csi-bridge` env knobs (`RUVECTOR_CSI_*`).

use crate::error::{Error, Result};

use std::net::SocketAddr;
use std::time::Duration;

/// Default UDP listen address — RuView's stock ESP32 broadcast port.
pub const DEFAULT_UDP_LISTEN: &str = "0.0.0.0:5005";
/// Default gRPC bind — port 50054 per ADR-183 Tier 1 (`:50051` is
/// hailo embed, `:50053` is ruvllm pi-worker, `:50054` is vitals).
pub const DEFAULT_GRPC_LISTEN: &str = "0.0.0.0:50054";
/// Default brain URL — cognitum-v0 over Tailscale. Workers POST
/// `/memories` here.
pub const DEFAULT_BRAIN_URL: &str = "http://cognitum-v0:9876";
/// Default sliding-window length (frames). 50 frames @ 30 fps ≈ 1.6 s.
pub const DEFAULT_WINDOW_FRAMES: usize = 50;
/// Default brain POST cadence in seconds — same as RuView's
/// `brain_bridge.rs`.
pub const DEFAULT_BRAIN_POST_INTERVAL_SECS: u64 = 60;

/// Worker configuration. Built once at startup via [`Config::from_env`].
#[derive(Debug, Clone)]
pub struct Config {
    /// UDP socket the worker binds for ADR-018 ingress.
    pub udp_listen: SocketAddr,
    /// gRPC socket for the [`crate::proto`] service.
    pub grpc_listen: SocketAddr,
    /// Brain endpoint (e.g. `http://cognitum-v0:9876`). The worker
    /// POSTs `/memories` here.
    pub brain_url: String,
    /// Sliding-window length in frames.
    pub window_frames: usize,
    /// Brain POST cadence — controls how often vital summaries get
    /// flushed to the brain.
    pub brain_post_interval: Duration,
    /// Optional override for `node_name` reported on Health RPCs.
    /// Defaults to `gethostname()` lossy.
    pub node_name: String,
    /// True when verbose per-frame `tracing::debug!` is desired.
    pub verbose: bool,
    /// Optional comma-separated UDP fan-out targets — every received
    /// ADR-018 datagram is forwarded unchanged to each address. Used
    /// by ADR-183 Tier 2 to route per-room CSI from worker Pis to
    /// the cognitum-v0 fusion master (`100.77.59.83:5005`).
    pub relay_targets: Vec<SocketAddr>,
    /// ADR-183 Tier 3: path to `model.safetensors` from `ruv/ruview`.
    /// When `Some`, the worker computes a 128-dim contrastive CSI
    /// embedding after each vitals reading and POSTs it to the brain
    /// as a `"spatial-csi-embedding"` memory.
    /// Typically set to `/usr/local/share/ruvector/model.safetensors`
    /// on cognitum-v0 after `deploy/compile-csi-encoder-hef.py` runs.
    ///
    /// Feature-gated: only parsed when built with `--features csi-embed`.
    pub csi_model_path: Option<std::path::PathBuf>,
    /// ADR-183 Tier 3 iter 18: path to a room-specific LoRA adapter JSON
    /// (e.g. `/usr/local/share/ruvector/node-1.json`). When set alongside
    /// `csi_model_path`, the base encoder embeddings are refined by a
    /// rank-4 residual transform before being posted to the brain.
    /// Env: `RUVIEW_CSI_LORA_ADAPTER`.
    ///
    /// Feature-gated: only parsed when built with `--features csi-embed`.
    pub csi_lora_path: Option<std::path::PathBuf>,
}

impl Config {
    /// Parse from env. Anything missing falls back to the documented
    /// defaults; bad values surface as [`Error::Config`] /
    /// [`Error::Address`].
    pub fn from_env() -> Result<Self> {
        let udp_listen = parse_addr(
            "RUVIEW_VITALS_UDP_LISTEN",
            DEFAULT_UDP_LISTEN,
        )?;
        let grpc_listen = parse_addr(
            "RUVIEW_VITALS_GRPC_LISTEN",
            DEFAULT_GRPC_LISTEN,
        )?;
        let brain_url = std::env::var("RUVIEW_VITALS_BRAIN_URL")
            .unwrap_or_else(|_| DEFAULT_BRAIN_URL.to_string());
        let window_frames = parse_usize(
            "RUVIEW_VITALS_WINDOW_FRAMES",
            DEFAULT_WINDOW_FRAMES,
        )?;
        let brain_post_interval = Duration::from_secs(parse_u64(
            "RUVIEW_VITALS_BRAIN_INTERVAL_SECS",
            DEFAULT_BRAIN_POST_INTERVAL_SECS,
        )?);
        let node_name = std::env::var("RUVIEW_VITALS_NODE_NAME").ok().unwrap_or_else(
            || {
                hostname_lossy()
            },
        );
        let verbose = std::env::var("RUVIEW_VITALS_VERBOSE")
            .ok()
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false);
        let relay_targets = parse_addr_list("RUVIEW_VITALS_RELAY_TARGETS")?;

        #[cfg(feature = "csi-embed")]
        let csi_model_path = std::env::var("RUVIEW_CSI_MODEL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(std::path::PathBuf::from);
        #[cfg(not(feature = "csi-embed"))]
        let csi_model_path: Option<std::path::PathBuf> = None;

        #[cfg(feature = "csi-embed")]
        let csi_lora_path = std::env::var("RUVIEW_CSI_LORA_ADAPTER")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(std::path::PathBuf::from);
        #[cfg(not(feature = "csi-embed"))]
        let csi_lora_path: Option<std::path::PathBuf> = None;

        if window_frames < 8 {
            return Err(Error::Config(
                "RUVIEW_VITALS_WINDOW_FRAMES must be ≥ 8 (need at least one breathing cycle)"
                    .into(),
            ));
        }
        if brain_url.is_empty() {
            return Err(Error::Config("RUVIEW_VITALS_BRAIN_URL is empty".into()));
        }

        Ok(Self {
            udp_listen,
            grpc_listen,
            brain_url,
            window_frames,
            brain_post_interval,
            node_name,
            verbose,
            relay_targets,
            csi_model_path,
            csi_lora_path,
        })
    }
}

/// Parse a comma-separated list of `SocketAddr` from an env var.
/// Returns an empty vec when the var is unset or empty. Bad entries
/// surface as [`Error::Address`] with the offending element preserved
/// so the operator can grep `journalctl` and find the typo.
fn parse_addr_list(key: &str) -> Result<Vec<SocketAddr>> {
    let raw = match std::env::var(key) {
        Ok(s) if !s.trim().is_empty() => s,
        _ => return Ok(Vec::new()),
    };
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<SocketAddr>()
                .map_err(|source| Error::Address {
                    addr: s.to_string(),
                    source,
                })
        })
        .collect()
}

fn parse_addr(key: &str, default: &str) -> Result<SocketAddr> {
    let raw = std::env::var(key).unwrap_or_else(|_| default.to_string());
    raw.parse::<SocketAddr>()
        .map_err(|source| Error::Address { addr: raw, source })
}

fn parse_usize(key: &str, default: usize) -> Result<usize> {
    match std::env::var(key) {
        Ok(s) => s
            .parse()
            .map_err(|e| Error::Config(format!("{key}={s}: {e}"))),
        Err(_) => Ok(default),
    }
}

fn parse_u64(key: &str, default: u64) -> Result<u64> {
    match std::env::var(key) {
        Ok(s) => s
            .parse()
            .map_err(|e| Error::Config(format!("{key}={s}: {e}"))),
        Err(_) => Ok(default),
    }
}

/// Best-effort hostname read. Falls back to `"unknown"` if the host
/// can't be resolved (extremely rare on Linux).
fn hostname_lossy() -> String {
    // Prefer the `HOSTNAME` env var (set by most shells). Fall back to
    // `/proc/sys/kernel/hostname`. Avoid pulling the `hostname` crate —
    // we only need a label for the gRPC Health response.
    if let Ok(h) = std::env::var("HOSTNAME") {
        if !h.is_empty() {
            return h;
        }
    }
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_resolve() {
        // Clean any pre-set vars so the test is deterministic. We can't
        // really mutate global env safely from a test, so just sanity-
        // check the parsers on the default strings.
        let addr: SocketAddr = DEFAULT_UDP_LISTEN.parse().unwrap();
        assert_eq!(addr.port(), 5005);
        let addr: SocketAddr = DEFAULT_GRPC_LISTEN.parse().unwrap();
        assert_eq!(addr.port(), 50054);
        assert!(DEFAULT_BRAIN_URL.starts_with("http://cognitum-v0"));
    }
}
