//! `ruvector-mmwave-bridge` — host-side daemon that reads a 60 GHz
//! mmWave radar (Seeed MR60BHA2 over USB-serial) and surfaces decoded
//! vital signs.
//!
//! Iter 115 (host-side companion to iter A on the ESP32). Shares the
//! same `ruvector_mmwave::Mr60Parser` state machine that runs on the
//! ESP32-S3 firmware — exactly one tested implementation, two callers.
//!
//! Architectural fit: the radar enumerates as a `/dev/ttyUSB*` (CH340
//! / CP210x bridge variants) or `/dev/ttyACM*` (native USB-CDC variants
//! like Seeed's pre-soldered USB stick). Either way the byte stream is
//! identical Seeed mmWave protocol; this bin is the host counterpart to
//! the ESP32 firmware's UART read loop.
//!
//! # Usage
//!
//! ```text
//!   ruvector-mmwave-bridge --device /dev/ttyUSB0 [--baud 115200]
//!   ruvector-mmwave-bridge --simulator [--rate 10]   # synthesised frames @ N Hz
//!   ruvector-mmwave-bridge --auto                    # scan tty nodes for an MR60 SOF
//! ```
//!
//! Iter 116 will add `--workers <addr>` + the existing TLS/mTLS flag
//! set (`--workers-file-sig` / `--workers-file-pubkey`) so each
//! decoded vital can be posted as an embed RPC into the cluster's
//! §1b-gated path. Today's bin logs to stdout/stderr only.

use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ruvector_hailo_cluster::transport::{EmbeddingTransport, WorkerEndpoint};
use ruvector_hailo_cluster::{GrpcTransport, HailoClusterEmbedder};
use ruvector_mmwave::{invert_xor_public, Event, Mr60Parser};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let mut device: Option<PathBuf> = None;
    let mut auto_scan = false;
    let mut simulator = false;
    let mut sim_rate_hz: u32 = 10;
    let mut baud: u32 = 115_200;
    let mut quiet = false;

    // Iter 116: cluster posting. When --workers is provided, every
    // decoded event is converted to a natural-language description and
    // posted to the hailo-backend cluster via the embed RPC. The
    // cluster's existing §1b mTLS gate, §2a fp+cache gate, and §3b
    // rate-limit interceptor all apply to this traffic the same way
    // they do to embed/bench.
    let mut workers_csv: Option<String> = None;
    let mut dim: usize = 384;
    let mut fingerprint: String = String::new();
    let mut allow_empty_fingerprint = false;

    // Iter 120: TLS / mTLS flag plumbing — parity with the cluster's
    // iter-99/100 stack. All four are only meaningful under
    // `--features tls`; without that feature, the binary still parses
    // the flags and errors loudly so an operator gets a clear "this
    // build doesn't have tls" message rather than silent plaintext.
    let mut tls_ca: Option<String> = None;
    let mut tls_domain: Option<String> = None;
    let mut tls_client_cert: Option<String> = None;
    let mut tls_client_key: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--device" => {
                device = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--auto" => {
                auto_scan = true;
                i += 1;
            }
            "--simulator" => {
                simulator = true;
                i += 1;
            }
            "--rate" => {
                sim_rate_hz = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(10);
                i += 2;
            }
            "--baud" => {
                baud = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(115_200);
                i += 2;
            }
            "--quiet" => {
                quiet = true;
                i += 1;
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

    // Optional cluster sink. None = log JSONL only (the iter-115
    // behaviour); Some = post each decoded event to the cluster.
    let cluster: Option<Arc<HailoClusterEmbedder>> = if let Some(csv) = workers_csv.as_ref() {
        // ADR-172 §2a same gate the embed/bench bins enforce.
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

        // Iter 120: TLS path. When --tls-ca is set we build a
        // `GrpcTransport::with_tls(...)` instead of plain
        // `GrpcTransport::new()`. mTLS happens via
        // `with_client_identity` when both --tls-client-cert and
        // --tls-client-key are also set; supplying just one half is
        // a misconfiguration and must fail loudly (same gate shape as
        // worker.rs's RUVECTOR_TLS_CERT/KEY pair).
        let transport: Arc<dyn EmbeddingTransport + Send + Sync> = if tls_ca.is_some()
            || tls_domain.is_some()
            || tls_client_cert.is_some()
            || tls_client_key.is_some()
        {
            #[cfg(not(feature = "tls"))]
            {
                return Err(
                    "TLS flags supplied but this build wasn't compiled with --features tls; \
                     rebuild with `cargo build --features tls` or drop the --tls-* flags"
                        .into(),
                );
            }
            #[cfg(feature = "tls")]
            {
                let ca =
                    tls_ca.ok_or("--tls-ca <path> is required when any --tls-* flag is set")?;
                // Operator can pin the SNI domain explicitly; otherwise
                // we extract it from the first worker's address (host
                // part). The cluster-side iter-99 path uses the same
                // helper.
                let domain = tls_domain.unwrap_or_else(|| {
                    ruvector_hailo_cluster::tls::domain_from_address(&workers[0].address)
                        .to_string()
                });
                let mut tls = ruvector_hailo_cluster::tls::TlsClient::from_pem_files(&ca, domain)?;
                match (&tls_client_cert, &tls_client_key) {
                    (Some(c), Some(k)) => {
                        tls = tls.with_client_identity(c, k)?;
                        if !quiet {
                            eprintln!(
                                "ruvector-mmwave-bridge: mTLS active (cert={}, key={})",
                                c, k
                            );
                        }
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        return Err(
                            "--tls-client-cert and --tls-client-key must both be set or both unset (ADR-172 §1b)"
                                .into(),
                        );
                    }
                    (None, None) => {
                        if !quiet {
                            eprintln!("ruvector-mmwave-bridge: TLS active (server-auth only; no client cert)");
                        }
                    }
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
                "ruvector-mmwave-bridge: cluster sink active — {} worker(s), dim={}, fp={:?}",
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

    // Mode selection precedence: --simulator wins (always — operator
    // explicitly asked for synthetic), then --device, then --auto.
    if simulator {
        if !quiet {
            eprintln!(
                "ruvector-mmwave-bridge: simulator mode @ {} Hz (no hardware required)",
                sim_rate_hz
            );
        }
        run_simulator(sim_rate_hz, quiet, cluster.as_deref())?;
        return Ok(());
    }

    let dev = if let Some(d) = device {
        d
    } else if auto_scan {
        scan_for_radar(baud, quiet)?
    } else {
        return Err("must pass exactly one of --device <path> / --simulator / --auto".into());
    };

    if !quiet {
        eprintln!(
            "ruvector-mmwave-bridge: opening {} @ {} baud",
            dev.display(),
            baud
        );
    }
    set_baud(&dev, baud)?;
    let mut file = std::fs::File::open(&dev)?;
    run_serial(&mut file, quiet, cluster.as_deref())
}

/// Convert a decoded radar event into a natural-language description
/// for the embed RPC. Iter-116 keeps the surface minimal — short
/// sentences are intentionally embedding-friendly (sentence-transformers
/// shines at this granularity). Downstream RAG queries like
/// "did the kitchen sensor see anyone in the last hour?" can match
/// against these strings via cosine similarity.
fn event_to_text(ev: &Event) -> Option<String> {
    Some(match ev {
        Event::Breathing { bpm } => {
            format!("breathing rate {} bpm at radar sensor", bpm)
        }
        Event::HeartRate { bpm } => {
            format!("heart rate {} bpm at radar sensor", bpm)
        }
        Event::Distance { cm } => {
            format!("nearest target distance {} cm at radar sensor", cm)
        }
        Event::Presence { present: true } => "person detected at radar sensor".to_string(),
        Event::Presence { present: false } => "no person detected at radar sensor".to_string(),
        // Unknown / ChecksumError / Resync don't get embedded — they're
        // protocol-layer noise, not semantic events.
        _ => return None,
    })
}

/// Post `text` to the cluster via the existing embed RPC. Errors are
/// logged but don't kill the bridge — radar events keep flowing into
/// JSONL on stdout regardless of cluster availability.
fn post_to_cluster(cluster: &HailoClusterEmbedder, text: &str, quiet: bool) {
    match cluster.embed_one_blocking(text) {
        Ok(v) => {
            if !quiet {
                eprintln!(
                    "ruvector-mmwave-bridge: posted text={:?} dim={} ok",
                    text,
                    v.len()
                );
            }
        }
        Err(e) => {
            // Always print errors regardless of --quiet — cluster
            // unavailability is the kind of thing operators need to see.
            eprintln!(
                "ruvector-mmwave-bridge: cluster post failed for {:?}: {}",
                text, e
            );
        }
    }
}

/// Configure the tty's line settings to raw + N81 + the requested baud.
/// Uses `stty` because pulling in `nix` or `serialport` for a single
/// `tcsetattr` call is overkill — this bin is host-only and stty is
/// universally available where /dev/ttyUSB* + /dev/ttyACM* live.
fn set_baud(dev: &std::path::Path, baud: u32) -> Result<(), Box<dyn std::error::Error>> {
    let status = std::process::Command::new("stty")
        .args([
            "-F",
            dev.to_str().ok_or("non-utf8 device path")?,
            &baud.to_string(),
            "raw",
            "-echo",
            "-echoe",
            "-echok",
            "cs8",
            "-parenb",
            "-cstopb",
            "-crtscts",
        ])
        .status()?;
    if !status.success() {
        return Err(format!("stty failed: exit {:?}", status.code()).into());
    }
    Ok(())
}

/// Drive the parser from a real serial device. Loops until EOF or
/// SIGINT (handled implicitly by std::io::Read returning Ok(0) /
/// errors). Logs decoded events to stdout one per line, and (when a
/// cluster is configured) posts each semantically-meaningful event
/// to the embed RPC.
fn run_serial<R: Read>(
    reader: &mut R,
    quiet: bool,
    cluster: Option<&HailoClusterEmbedder>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut parser = Mr60Parser::new();
    let mut buf = [0u8; 256];
    let started = Instant::now();
    let mut total_events = 0u64;
    let mut last_status = Instant::now();

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            // EOF from std::fs::File means the device disappeared
            // (radar unplugged). Exit cleanly with a non-zero code so
            // a supervisor (systemd, runit) can restart on hot-plug.
            if !quiet {
                eprintln!("ruvector-mmwave-bridge: EOF on serial — radar disconnected");
            }
            return Err("device disconnected".into());
        }
        parser.feed_slice(&buf[..n], |ev| {
            total_events += 1;
            emit_event(&ev, started.elapsed());
            if let Some(c) = cluster {
                if let Some(text) = event_to_text(&ev) {
                    post_to_cluster(c, &text, quiet);
                }
            }
        });

        // 1 Hz status when nothing has happened — keeps log scrapers
        // from thinking the bridge is wedged on a dead-radar stream.
        if last_status.elapsed() >= Duration::from_secs(1) {
            if !quiet && total_events == 0 {
                eprintln!(
                    "ruvector-mmwave-bridge: 0 events in {:?} — check radar power + baud",
                    started.elapsed()
                );
            }
            last_status = Instant::now();
        }
    }
}

/// Synthetic frame generator — bypasses real hardware. Useful for
/// (a) demoing the pipeline without the radar attached, (b) pumping
/// regression-test fixtures into a downstream consumer, (c) soak
/// testing the cluster path under a known-shape event stream.
fn run_simulator(
    rate_hz: u32,
    quiet: bool,
    cluster: Option<&HailoClusterEmbedder>,
) -> Result<(), Box<dyn std::error::Error>> {
    let interval = Duration::from_secs_f64(1.0 / rate_hz.max(1) as f64);
    let started = Instant::now();
    let mut parser = Mr60Parser::new();
    let mut tick: u64 = 0;
    loop {
        let frame_bytes = synthesise_frame(tick);
        parser.feed_slice(&frame_bytes, |ev| {
            emit_event(&ev, started.elapsed());
            if let Some(c) = cluster {
                if let Some(text) = event_to_text(&ev) {
                    post_to_cluster(c, &text, quiet);
                }
            }
        });
        tick = tick.wrapping_add(1);
        std::thread::sleep(interval);
    }
}

/// Build a synthetic MR60BHA2 frame whose contents cycle through the
/// four interesting frame types so a downstream consumer sees the full
/// event matrix. `tick % 4` picks the type.
fn synthesise_frame(tick: u64) -> Vec<u8> {
    // Random-walk vital signs so the simulator output looks like a
    // realistic radar trace, not a constant.
    let breathing_bpm = 12 + ((tick / 4) % 8) as u8; // 12..19 bpm
    let heart_rate_bpm = 60 + ((tick * 7) % 40) as u8; // 60..99 bpm
    let distance_cm = 80 + ((tick * 13) % 200) as u16; // 80..279 cm
    let presence: u8 = if tick % 8 < 6 { 1 } else { 0 };

    let (frame_type, payload): (u16, Vec<u8>) = match tick % 4 {
        0 => (0x0A14, vec![breathing_bpm]),
        1 => (0x0A15, vec![heart_rate_bpm]),
        2 => (0x0A16, vec![(distance_cm >> 8) as u8, distance_cm as u8]),
        _ => (0x0F09, vec![presence]),
    };

    let mut header = vec![
        0x01u8,
        0x00,
        0x00,
        (payload.len() >> 8) as u8,
        payload.len() as u8,
        (frame_type >> 8) as u8,
        frame_type as u8,
    ];
    let hcksum = invert_xor_public(&header);
    header.push(hcksum);
    let dcksum = invert_xor_public(&payload);
    let mut out = header;
    out.extend_from_slice(&payload);
    out.push(dcksum);
    out
}

/// Scan `/dev/ttyUSB*` + `/dev/ttyACM*` for the MR60BHA2 SOF byte
/// (`0x01`) followed by a valid header checksum. First match wins.
/// 1.5 second probe per device — enough for ~15-20 frames at the
/// MR60BHA2's typical 10 Hz output rate.
fn scan_for_radar(baud: u32, quiet: bool) -> Result<PathBuf, Box<dyn std::error::Error>> {
    use std::fs;
    let mut candidates: Vec<PathBuf> = Vec::new();
    for prefix in ["/dev/ttyUSB", "/dev/ttyACM"] {
        for n in 0..16 {
            let p = PathBuf::from(format!("{}{}", prefix, n));
            if p.exists() {
                candidates.push(p);
            }
        }
    }
    if candidates.is_empty() {
        return Err("--auto: no /dev/ttyUSB* or /dev/ttyACM* nodes found".into());
    }

    for cand in candidates {
        if !quiet {
            eprintln!("ruvector-mmwave-bridge: probing {}", cand.display());
        }
        if set_baud(&cand, baud).is_err() {
            continue;
        }
        let mut f = match fs::File::open(&cand) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mut parser = Mr60Parser::new();
        let mut buf = [0u8; 64];
        let deadline = Instant::now() + Duration::from_millis(1500);
        let mut got_real_event = false;
        while Instant::now() < deadline && !got_real_event {
            // Non-blocking read via a small chunk — the kernel will
            // return whatever's available.
            match f.read(&mut buf) {
                Ok(n) if n > 0 => {
                    parser.feed_slice(&buf[..n], |ev| {
                        if matches!(
                            ev,
                            Event::Breathing { .. }
                                | Event::HeartRate { .. }
                                | Event::Distance { .. }
                                | Event::Presence { .. }
                        ) {
                            got_real_event = true;
                        }
                    });
                }
                _ => std::thread::sleep(Duration::from_millis(20)),
            }
        }
        if got_real_event {
            if !quiet {
                eprintln!(
                    "ruvector-mmwave-bridge: --auto found radar on {}",
                    cand.display()
                );
            }
            return Ok(cand);
        }
    }
    Err("--auto: no MR60BHA2-shaped frames found on any tty node within probe window".into())
}

/// Emit one decoded event as a stdout line. JSON-shaped so log
/// scrapers + iter 116's cluster-poster can both consume it cleanly.
fn emit_event(ev: &Event, t: Duration) {
    let ts_ms = t.as_millis();
    match ev {
        Event::Breathing { bpm } => {
            println!(r#"{{"t_ms":{},"kind":"breathing","bpm":{}}}"#, ts_ms, bpm)
        }
        Event::HeartRate { bpm } => {
            println!(r#"{{"t_ms":{},"kind":"heart_rate","bpm":{}}}"#, ts_ms, bpm)
        }
        Event::Distance { cm } => println!(r#"{{"t_ms":{},"kind":"distance","cm":{}}}"#, ts_ms, cm),
        Event::Presence { present } => println!(
            r#"{{"t_ms":{},"kind":"presence","present":{}}}"#,
            ts_ms, present
        ),
        Event::Unknown {
            frame_type,
            payload_len,
        } => println!(
            r#"{{"t_ms":{},"kind":"unknown","frame_type":"0x{:04x}","payload_len":{}}}"#,
            ts_ms, frame_type, payload_len
        ),
        Event::ChecksumError | Event::Resync => {
            // Don't pollute the stream — these surface as counter
            // increments in iter 116's status path.
        }
    }
}

fn print_help() {
    println!(
        "{} {} — host-side bridge for MR60BHA2 60 GHz mmWave radar (ADR-063)\n\
\n\
USAGE:\n    ruvector-mmwave-bridge <MODE> [OPTIONS]\n\
\n\
MODE (exactly one):\n    \
    --device <path>      Read from a specific tty (e.g. /dev/ttyUSB0).\n    \
    --auto               Scan /dev/ttyUSB* + /dev/ttyACM* for the radar.\n    \
    --simulator          Generate synthetic frames; no hardware required.\n\
\n\
OPTIONS:\n    \
    --baud <N>                   UART baud (default 115200, MR60BHA2 stock).\n    \
    --rate <Hz>                  Simulator frame rate (default 10).\n    \
    --quiet                      Suppress informational stderr; keep stdout JSON.\n    \
    --workers <addr1,addr2,...>  Cluster sink — post each decoded event to\n                                  the hailo-backend cluster's embed RPC. Same\n                                  semantics as --workers in embed/bench.\n    \
    --dim <N>                    Expected embedding dim (default 384).\n    \
    --fingerprint <hex>          Reject workers reporting a different fp.\n    \
    --allow-empty-fingerprint    Bypass the ADR-172 §2a empty-fp gate.\n    \
    --tls-ca <path>              Server CA bundle (PEM). Enables TLS — coerces\n                                  workers to https:// (ADR-172 §1a). Requires\n                                  the binary to be built with --features tls.\n    \
    --tls-domain <name>          SNI / cert-SAN to assert. Default: hostname\n                                  of the first --workers entry.\n    \
    --tls-client-cert <path>     PEM client cert for mTLS (ADR-172 §1b).\n                                  Must be paired with --tls-client-key.\n    \
    --tls-client-key <path>      PEM private key matching --tls-client-cert.\n    \
    --help                       This message.\n    \
    --version                    Print version.\n\
\n\
OUTPUT:\n    \
    One JSON object per decoded event on stdout, e.g.:\n    \
    {{\"t_ms\":150,\"kind\":\"heart_rate\",\"bpm\":72}}\n\
    When --workers is set, semantically-meaningful events are also\n\
    converted to natural-language descriptions (e.g. \"heart rate 72\n\
    bpm at radar sensor\") and posted via the cluster's embed RPC.",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
    );
}
