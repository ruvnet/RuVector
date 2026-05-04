//! TLS configuration for the cluster transport (ADR-172 §1a HIGH mitigation).
//!
//! Only compiled when the `tls` cargo feature is on, which propagates to
//! `tonic/tls` and pulls in rustls + tokio-rustls. The base build remains
//! unchanged so x86 dev hosts and Tailscale-only deploys (where the wire
//! is already encrypted) don't pay the rustls cost.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "tls")] {
//! use std::time::Duration;
//! use ruvector_hailo_cluster::{GrpcTransport, tls::TlsClient};
//!
//! let tls = TlsClient::from_pem_files("ca.pem", "worker.local")
//!     .expect("load CA");
//! let transport = GrpcTransport::with_tls(
//!     Duration::from_secs(5),
//!     Duration::from_secs(2),
//!     tls,
//! ).expect("build transport");
//! # }
//! ```

#![cfg(feature = "tls")]

use crate::error::ClusterError;
use std::path::Path;
use tonic::transport::{Certificate, ClientTlsConfig, Identity, ServerTlsConfig};

fn read_pem(path: &Path, what: &str) -> Result<Vec<u8>, ClusterError> {
    // Iter 212 — cap PEM reads at 1 MB. A typical PEM is ~2-10 KB
    // (single cert) or ~30 KB (full chain with intermediates); 1 MB
    // is ~100× legit headroom and matches iter-210's manifest cap.
    // A misconfig pointing the cert/key/CA path at /var/log/* or a
    // binary blob would otherwise OOM the worker at boot before
    // rustls ever sees the bytes. Same threat model as iter-210
    // (operator-controlled paths) and iter-211 (signed-manifest
    // sig/pubkey reads).
    const PEM_CAP: u64 = 1 << 20; // 1 MB
    let meta = std::fs::metadata(path).map_err(|e| ClusterError::Transport {
        worker: "<tls>".into(),
        reason: format!("stat {} pem at {}: {}", what, path.display(), e),
    })?;
    if meta.len() > PEM_CAP {
        return Err(ClusterError::Transport {
            worker: "<tls>".into(),
            reason: format!(
                "{} pem at {} is {} bytes, exceeds {} byte cap (iter 212 — \
                 likely a misconfig pointed at the wrong file)",
                what,
                path.display(),
                meta.len(),
                PEM_CAP
            ),
        });
    }
    std::fs::read(path).map_err(|e| ClusterError::Transport {
        worker: "<tls>".into(),
        reason: format!("read {} pem at {}: {}", what, path.display(), e),
    })
}

/// Client-side TLS material for [`crate::GrpcTransport`].
///
/// Wraps tonic's `ClientTlsConfig` with file-based constructors and a
/// `domain_from_address` helper that strips the `host:port` form into
/// just the hostname (rustls wants the SAN, not the port).
#[derive(Clone)]
pub struct TlsClient {
    inner: ClientTlsConfig,
}

impl TlsClient {
    /// Trust **only** the CA certificate(s) in `ca_pem_path`. `domain` is
    /// the SNI / SAN value to assert against the server cert; pass the
    /// hostname half of the worker's `host:port`. Use [`domain_from_address`]
    /// if you've only got the wire-level address handy.
    pub fn from_pem_files(
        ca_pem_path: impl AsRef<Path>,
        domain: impl Into<String>,
    ) -> Result<Self, ClusterError> {
        let ca = read_pem(ca_pem_path.as_ref(), "ca")?;
        Self::from_pem_bytes(&ca, domain)
    }

    /// In-memory variant for tests / embedded deploys that already have
    /// the CA bundle as bytes.
    pub fn from_pem_bytes(ca_pem: &[u8], domain: impl Into<String>) -> Result<Self, ClusterError> {
        let inner = ClientTlsConfig::new()
            .domain_name(domain)
            .ca_certificate(Certificate::from_pem(ca_pem));
        Ok(Self { inner })
    }

    /// Attach a client cert + key for mTLS (ADR-172 §1b mitigation, iter 100).
    /// The worker will accept the connection only if this cert chains
    /// to its `--client-ca` bundle (set via the `RUVECTOR_TLS_CLIENT_CA`
    /// env var on the server side).
    pub fn with_client_identity(
        self,
        cert_pem_path: impl AsRef<Path>,
        key_pem_path: impl AsRef<Path>,
    ) -> Result<Self, ClusterError> {
        let cert = read_pem(cert_pem_path.as_ref(), "client cert")?;
        let key = read_pem(key_pem_path.as_ref(), "client key")?;
        Ok(self.with_client_identity_bytes(&cert, &key))
    }

    /// In-memory variant of [`Self::with_client_identity`]. Useful for
    /// tests + embedded deploys where the client identity is generated
    /// at runtime rather than read from a file.
    pub fn with_client_identity_bytes(self, cert_pem: &[u8], key_pem: &[u8]) -> Self {
        let identity = Identity::from_pem(cert_pem, key_pem);
        Self {
            inner: self.inner.identity(identity),
        }
    }

    /// Unwrap to the underlying tonic `ClientTlsConfig`. Pub so the
    /// `ruvector-hailo-worker` bin (a separate crate from the lib) can
    /// hand it to `tonic::transport::Server::tls_config`.
    pub fn into_inner(self) -> ClientTlsConfig {
        self.inner
    }
}

/// Server-side TLS material for the worker.
///
/// Built from a cert + key pair on disk (typically issued by a fleet CA
/// run on the coordinator host). `with_client_ca` enforces mTLS by
/// requiring clients to present a cert signed by the supplied CA.
pub struct TlsServer {
    inner: ServerTlsConfig,
}

impl TlsServer {
    /// Load identity from PEM-encoded cert + key paths.
    pub fn from_pem_files(
        cert_pem_path: impl AsRef<Path>,
        key_pem_path: impl AsRef<Path>,
    ) -> Result<Self, ClusterError> {
        let cert = read_pem(cert_pem_path.as_ref(), "server cert")?;
        let key = read_pem(key_pem_path.as_ref(), "server key")?;
        Ok(Self::from_pem_bytes(&cert, &key))
    }

    /// In-memory variant — bypasses the filesystem so unit tests can
    /// stand up a TLS server with rcgen-issued material.
    pub fn from_pem_bytes(cert_pem: &[u8], key_pem: &[u8]) -> Self {
        let identity = Identity::from_pem(cert_pem, key_pem);
        Self {
            inner: ServerTlsConfig::new().identity(identity),
        }
    }

    /// Require client certs signed by `ca_pem_path` (mTLS — ADR-172 §1b
    /// mitigation, iter 100). Combined with the default
    /// `client_auth_optional = false` from `ServerTlsConfig::new`, any
    /// client lacking a CA-signed identity is rejected at handshake.
    pub fn with_client_ca(self, ca_pem_path: impl AsRef<Path>) -> Result<Self, ClusterError> {
        let ca = read_pem(ca_pem_path.as_ref(), "client ca")?;
        Ok(self.with_client_ca_bytes(&ca))
    }

    /// In-memory variant of [`Self::with_client_ca`].
    pub fn with_client_ca_bytes(self, ca_pem: &[u8]) -> Self {
        Self {
            inner: self.inner.client_ca_root(Certificate::from_pem(ca_pem)),
        }
    }

    /// Unwrap to the underlying tonic `ServerTlsConfig`. Pub for the
    /// same cross-crate reason as [`TlsClient::into_inner`].
    pub fn into_inner(self) -> ServerTlsConfig {
        self.inner
    }
}

/// Strip a `host:port` (or `https://host:port[/path]`) into just the
/// hostname so `TlsClient::from_pem_files` can use it as the SNI value.
/// IPv6 brackets are stripped; if no port is present the whole input
/// (minus scheme/path) is returned.
pub fn domain_from_address(addr: &str) -> &str {
    let s = addr
        .strip_prefix("https://")
        .or_else(|| addr.strip_prefix("http://"))
        .unwrap_or(addr);
    let s = s.split('/').next().unwrap_or(s);
    // [::1]:50051 → ::1
    if let Some(rest) = s.strip_prefix('[') {
        if let Some(end) = rest.find(']') {
            return &rest[..end];
        }
    }
    // host:port → host. Only split on the last ':' so IPv6 without
    // brackets is left alone (rustls accepts it).
    match s.rfind(':') {
        Some(i) if !s[i + 1..].contains(':') => &s[..i],
        _ => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_strip_host_port() {
        assert_eq!(
            domain_from_address("worker-a.local:50051"),
            "worker-a.local"
        );
    }

    #[test]
    fn domain_strip_https_url() {
        assert_eq!(domain_from_address("https://w.local:50051/x"), "w.local");
    }

    #[test]
    fn domain_strip_ipv6_brackets() {
        assert_eq!(domain_from_address("[::1]:50051"), "::1");
    }

    #[test]
    fn domain_strip_bare_host() {
        assert_eq!(domain_from_address("worker.local"), "worker.local");
    }

    /// Iter 212 — read_pem rejects > 1 MB files before reading them.
    #[test]
    fn read_pem_rejects_oversized_file() {
        use std::io::Write as _;
        let path =
            std::env::temp_dir().join(format!("iter212-oversized-pem-{}", std::process::id()));
        let mut f = std::fs::File::create(&path).expect("create fixture");
        // 2 MB filler — pem-shaped armor noise so a future read would
        // appear plausible if the cap weren't there.
        let chunk = b"-----BEGIN CERTIFICATE-----\nA\n";
        for _ in 0..((2 * 1024 * 1024) / chunk.len() + 1) {
            f.write_all(chunk).expect("write fixture");
        }
        f.sync_all().expect("sync");
        drop(f);

        let err = read_pem(&path, "test").expect_err("oversized pem must reject");
        match err {
            ClusterError::Transport { reason, .. } => {
                assert!(
                    reason.contains("exceeds")
                        && reason.contains("byte cap")
                        && reason.contains("iter 212"),
                    "expected size-cap rejection, got: {:?}",
                    reason
                );
            }
            other => panic!("expected ClusterError::Transport, got {:?}", other),
        }
        let _ = std::fs::remove_file(&path);
    }

    /// Iter 212 — read_pem still works for legit-size files.
    #[test]
    fn read_pem_accepts_small_file() {
        let path = std::env::temp_dir().join(format!("iter212-small-pem-{}", std::process::id()));
        // ~30 byte fake PEM — well under 1 MB cap.
        std::fs::write(&path, b"-----BEGIN CERTIFICATE-----\nx\n").expect("write fixture");
        let bytes = read_pem(&path, "test").expect("small pem must succeed");
        assert!(bytes.len() < 1024);
        assert!(bytes.starts_with(b"-----BEGIN"));
        let _ = std::fs::remove_file(&path);
    }
}
