//! `ruview-csi-bridge` — host-side daemon that receives RuView's ADR-018
//! binary CSI UDP frames and posts a **header-summary natural-language
//! string** to the hailo-backend cluster's embed RPC. Telemetry
//! indexing only — the I/Q payload (`bytes 20..`) is parsed for the
//! header but otherwise dropped.
//!
//! # Important: this bridge is *not* WiFi-DensePose pose embedding
//!
//! ADR-178 §3.2 C (iter 220) explicitly disambiguated this: the
//! cosine embeddings of the summary strings cluster by
//! `(channel, rssi-bucket, node_id)`, NOT by anything related to
//! actual WiFi-DensePose pose content. ADR-171's pipeline diagram
//! (`CSI tensor → preprocess → HEF → pose tensor`) implies pose
//! semantics; this bridge ships none of that — it bridges the
//! *transport* (UDP→gRPC) and uses the same text-encoder HEF that
//! serves regular sentence inputs.
//!
//! Real pose embedding requires a pose-specific HEF (Hailo Model Zoo
//! today ships only hailo15h/10h variants, ADR-167 line 184-194), a
//! `HailoPipeline<I, O>` generalization of `EmbeddingPipeline`, and
//! host-side I/Q preprocessing (magnitude / FFT). That work is
//! tracked as a separate ADR per ADR-178 §3.2 C's long-term
//! recommendation; **this bridge stays as it is, useful for telemetry
//! indexing**.
//!
//! Iter 123 (ADR-171 transport implementation, companion to iter-115's
//! mmwave-bridge). RuView's ESP32 CSI nodes broadcast
//! `0xC5110001` (raw I/Q) or `0xC5110006` (feature state) UDP packets
//! on a configurable port; this bin parses the header, derives a short
//! natural-language description (RSSI / channel / motion-flag), and
//! posts each description into the cluster via the same TLS/mTLS-
//! protected embed path the mmwave-bridge uses.
//!
//! # Wire format (ADR-018, v1/v6)
//!
//! ```text
//!   bytes 0..4   magic        u32 LE: 0xC5110001 or 0xC5110006
//!   byte  4      node_id      u8
//!   byte  5      n_antennas   u8 (treat as max(1))
//!   bytes 6..8   n_subcarriers u16 LE
//!   byte  8      channel      u8
//!   byte  9      rssi         i8 (dBm)
//!   byte 10      noise_floor  i8 (dBm)
//!   bytes 11..16 reserved
//!   bytes 16..20 timestamp_us u32 LE
//!   bytes 20..   I/Q data: n_subcarriers × 2 × n_antennas bytes
//! ```
//!
//! # Usage
//!
//! ```text
//!   ruview-csi-bridge --listen 0.0.0.0:5005 \
//!       --workers 100.77.59.83:50051 --fingerprint <hex>
//! ```
//!
//! All TLS/mTLS flags (`--tls-ca`, `--tls-domain`, `--tls-client-cert`,
//! `--tls-client-key`) work the same way as in `ruvector-mmwave-bridge`
//! (iter 120).

use std::net::UdpSocket;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ruvector_hailo_cluster::transport::{EmbeddingTransport, WorkerEndpoint};
use ruvector_hailo_cluster::{GrpcTransport, HailoClusterEmbedder};

/// ADR-018 magic + header size — match RuView's canonical parser.
const CSI_MAGIC_V1: u32 = 0xC511_0001;
const CSI_MAGIC_V6: u32 = 0xC511_0006;
const CSI_HEADER_SIZE: usize = 20;

/// Decoded CSI summary suitable for an NL description. Keep allocation
/// to the bare minimum — this is one packet's worth of stats, not the
/// whole I/Q payload.
#[derive(Debug, Clone, Copy)]
struct CsiSummary {
    magic: u32,
    node_id: u8,
    n_antennas: u8,
    n_subcarriers: u16,
    channel: u8,
    rssi: i8,
    noise_floor: i8,
    timestamp_us: u32,
}

/// Parse an ADR-018 binary CSI frame header. Returns None when the
/// buffer is too short or the magic is unrecognised. Pure header-
/// only parse — we don't materialise the I/Q payload because we don't
/// need it for the embed RPC's NL description.
fn parse_csi_header(data: &[u8]) -> Option<CsiSummary> {
    if data.len() < CSI_HEADER_SIZE {
        return None;
    }
    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    if magic != CSI_MAGIC_V1 && magic != CSI_MAGIC_V6 {
        return None;
    }
    Some(CsiSummary {
        magic,
        node_id: data[4],
        n_antennas: data[5].max(1),
        n_subcarriers: u16::from_le_bytes([data[6], data[7]]),
        channel: data[8],
        rssi: data[9] as i8,
        noise_floor: data[10] as i8,
        timestamp_us: u32::from_le_bytes([data[16], data[17], data[18], data[19]]),
    })
}

/// Convert a CSI summary into an embedding-friendly NL string.
/// Short, sentence-like, includes the header values that drive
/// telemetry indexing. Keeps the format parallel to mmwave-bridge's
/// `event_to_text` so cluster-side pattern matching can treat both
/// bridges uniformly.
///
/// **Note (iter 220, ADR-178 §3.2 C):** the I/Q payload at
/// `bytes 20..` is intentionally *not* part of this string. The
/// cosine embedding of the resulting NL clusters by
/// `(channel, rssi, node_id)`, useful for "all CSI from node 3 on
/// channel 6" queries — but it does NOT carry pose semantics. Real
/// WiFi-DensePose pose embedding requires a pose HEF + I/Q
/// preprocessing and is out of scope for this transport bridge;
/// see the module docstring for the deferral context.
fn summary_to_text(s: &CsiSummary) -> String {
    let kind = if s.magic == CSI_MAGIC_V6 {
        "feature-state"
    } else {
        "raw"
    };
    format!(
        "wifi csi {} packet from node {} channel {} rssi {} dBm noise {} dBm \
         antennas {} subcarriers {}",
        kind, s.node_id, s.channel, s.rssi, s.noise_floor, s.n_antennas, s.n_subcarriers,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut listen_addr: String = "0.0.0.0:5005".to_string();
    let mut workers_csv: Option<String> = None;
    let mut dim: usize = 384;
    let mut fingerprint: String = String::new();
    let mut allow_empty_fingerprint = false;
    let mut quiet = false;

    let mut tls_ca: Option<String> = None;
    let mut tls_domain: Option<String> = None;
    let mut tls_client_cert: Option<String> = None;
    let mut tls_client_key: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--listen" => {
                listen_addr = args
                    .get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| "0.0.0.0:5005".into());
                i += 2;
            }
            "--workers" => {
                workers_csv = args.get(i + 1).cloned();
                i += 2;
            }
            "--dim" => {
                dim = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(384);
                i += 2;
            }
            "--fingerprint" => {
                fingerprint = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--allow-empty-fingerprint" => {
                allow_empty_fingerprint = true;
                i += 1;
            }
            "--quiet" => {
                quiet = true;
                i += 1;
            }
            "--tls-ca" => {
                tls_ca = args.get(i + 1).cloned();
                i += 2;
            }
            "--tls-domain" => {
                tls_domain = args.get(i + 1).cloned();
                i += 2;
            }
            "--tls-client-cert" => {
                tls_client_cert = args.get(i + 1).cloned();
                i += 2;
            }
            "--tls-client-key" => {
                tls_client_key = args.get(i + 1).cloned();
                i += 2;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            "--version" | "-V" => {
                println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            other => return Err(format!("unknown arg: {}", other).into()),
        }
    }

    // Optional cluster sink. Same gates as mmwave-bridge: §2a fp+cache
    // gate enforced, §1b mTLS plumbing wired through.
    let cluster: Option<Arc<HailoClusterEmbedder>> = if let Some(csv) = workers_csv.as_ref() {
        if fingerprint.is_empty() && !allow_empty_fingerprint {
            return Err(
                "refusing --workers with empty --fingerprint (ADR-172 §2a); pass \
                 --fingerprint <hex> or --allow-empty-fingerprint"
                    .into(),
            );
        }
        let workers: Vec<WorkerEndpoint> = csv
            .split(',')
            .filter(|s| !s.is_empty())
            .enumerate()
            .map(|(idx, addr)| {
                WorkerEndpoint::new(format!("static-{}", idx), addr.trim().to_string())
            })
            .collect();
        if workers.is_empty() {
            return Err("--workers list is empty".into());
        }

        let transport: Arc<dyn EmbeddingTransport + Send + Sync> = if tls_ca.is_some()
            || tls_domain.is_some()
            || tls_client_cert.is_some()
            || tls_client_key.is_some()
        {
            #[cfg(not(feature = "tls"))]
            {
                return Err(
                    "TLS flags supplied but this build wasn't compiled with --features tls".into(),
                );
            }
            #[cfg(feature = "tls")]
            {
                let ca =
                    tls_ca.ok_or("--tls-ca <path> is required when any --tls-* flag is set")?;
                let domain = tls_domain.unwrap_or_else(|| {
                    ruvector_hailo_cluster::tls::domain_from_address(&workers[0].address)
                        .to_string()
                });
                let mut tls = ruvector_hailo_cluster::tls::TlsClient::from_pem_files(&ca, domain)?;
                match (&tls_client_cert, &tls_client_key) {
                    (Some(c), Some(k)) => {
                        tls = tls.with_client_identity(c, k)?;
                        if !quiet {
                            eprintln!("ruview-csi-bridge: mTLS active (cert={})", c);
                        }
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        return Err(
                            "--tls-client-cert and --tls-client-key must both be set or both unset (ADR-172 §1b)"
                                .into(),
                        );
                    }
                    (None, None) => {}
                }
                Arc::new(GrpcTransport::with_tls(
                    Duration::from_secs(5),
                    Duration::from_secs(2),
                    tls,
                )?)
            }
        } else {
            Arc::new(GrpcTransport::new()?)
        };

        let c = HailoClusterEmbedder::new(workers, transport, dim, fingerprint.clone())?;
        if !quiet {
            eprintln!(
                "ruview-csi-bridge: cluster sink active — {} worker(s), dim={}, fp={:?}",
                csv.split(',').filter(|s| !s.is_empty()).count(),
                dim,
                if fingerprint.is_empty() {
                    "<unset>"
                } else {
                    fingerprint.as_str()
                },
            );
        }
        Some(Arc::new(c))
    } else {
        None
    };

    if !quiet {
        eprintln!(
            "ruview-csi-bridge: binding UDP {} for ADR-018 CSI frames",
            listen_addr
        );
    }
    let socket = UdpSocket::bind(&listen_addr)?;
    socket.set_read_timeout(Some(Duration::from_secs(1)))?;

    // Buffer sized for the largest realistic ADR-018 frame:
    // 20-byte header + (256 subcarriers × 2 I/Q × 4 antennas) = 2068 bytes.
    // 4 KB gives headroom for a future protocol bump without a recompile.
    let mut buf = [0u8; 4096];
    let started = Instant::now();
    let mut total_frames = 0u64;
    let mut total_dropped = 0u64;
    let mut last_status = Instant::now();

    loop {
        match socket.recv_from(&mut buf) {
            Ok((n, peer)) => {
                if let Some(summary) = parse_csi_header(&buf[..n]) {
                    total_frames += 1;
                    emit_event(&summary, &peer, started.elapsed());
                    if let Some(c) = cluster.as_deref() {
                        let text = summary_to_text(&summary);
                        match c.embed_one_blocking(&text) {
                            Ok(v) => {
                                if !quiet {
                                    eprintln!(
                                        "ruview-csi-bridge: posted text={:?} dim={} ok",
                                        text,
                                        v.len()
                                    );
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                    "ruview-csi-bridge: cluster post failed for {:?}: {}",
                                    text, e
                                );
                            }
                        }
                    }
                } else {
                    total_dropped += 1;
                    if !quiet && total_dropped <= 3 {
                        eprintln!(
                            "ruview-csi-bridge: dropped malformed packet from {} ({} bytes)",
                            peer, n
                        );
                    }
                }
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                // Timeout — fall through to status print.
            }
            Err(e) => {
                eprintln!("ruview-csi-bridge: recv error: {}", e);
            }
        }

        if last_status.elapsed() >= Duration::from_secs(5) && !quiet {
            eprintln!(
                "ruview-csi-bridge: {} frames parsed, {} malformed dropped over {:?}",
                total_frames,
                total_dropped,
                started.elapsed(),
            );
            last_status = Instant::now();
        }
    }
}

/// Emit one decoded CSI frame as a JSONL line on stdout. Same shape
/// convention as mmwave-bridge so a downstream consumer can treat
/// both event streams uniformly.
fn emit_event(s: &CsiSummary, peer: &std::net::SocketAddr, t: Duration) {
    let kind = if s.magic == CSI_MAGIC_V6 {
        "feature_state"
    } else {
        "raw"
    };
    println!(
        r#"{{"t_ms":{},"src":"{}","kind":"csi_{}","node_id":{},"channel":{},"rssi_dbm":{},"noise_dbm":{},"antennas":{},"subcarriers":{},"timestamp_us":{}}}"#,
        t.as_millis(),
        peer,
        kind,
        s.node_id,
        s.channel,
        s.rssi,
        s.noise_floor,
        s.n_antennas,
        s.n_subcarriers,
        s.timestamp_us,
    );
}

fn print_help() {
    println!(
        "{} {} — host-side bridge for RuView ADR-018 CSI UDP → cluster embed RPC (ADR-171)\n\
\n\
USAGE:\n    ruview-csi-bridge [--listen <addr>] [--workers <csv>] [--fingerprint <hex>]\n\
\n\
INPUT:\n    \
    --listen <addr>              UDP bind address (default 0.0.0.0:5005,\n                                  RuView's stock port).\n\
\n\
OPTIONAL:\n    \
    --workers <addr1,addr2,...>  Cluster sink — post each CSI frame's NL\n                                  description to the embed RPC.\n    \
    --dim <N>                    Expected embedding dim (default 384).\n    \
    --fingerprint <hex>          Reject workers reporting a different fp.\n    \
    --allow-empty-fingerprint    Bypass the ADR-172 §2a empty-fp gate.\n    \
    --quiet                      Suppress informational stderr.\n    \
    --tls-ca <path>              Server CA bundle (PEM). Enables TLS\n                                  (ADR-172 §1a). Requires `--features tls`.\n    \
    --tls-domain <name>          SNI / cert-SAN to assert.\n    \
    --tls-client-cert <path>     PEM client cert for mTLS (ADR-172 §1b).\n    \
    --tls-client-key <path>      PEM private key matching client cert.\n    \
    --help                       This message.\n    \
    --version                    Print version.\n\
\n\
OUTPUT:\n    \
    JSONL on stdout, one line per parsed CSI frame, e.g.:\n    \
    {{\"t_ms\":350,\"src\":\"10.0.0.42:54321\",\"kind\":\"csi_feature_state\",\"node_id\":2,…}}\n",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
    );
}
