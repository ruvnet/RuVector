//! `ruvector-hailo-stats` — fleet-wide GetStats snapshot.
//!
//! Calls `cluster.fleet_stats()` against every discovered worker and
//! prints a one-row-per-worker table on stdout. Useful for operators,
//! cron-driven dashboards, or piping into a status check.
//!
//! Usage mirrors `ruvector-hailo-embed`:
//!
//!   ruvector-hailo-stats --workers <addr1,addr2>
//!   ruvector-hailo-stats --tailscale-tag tag:ruvector-hailo-worker --port 50051
//!
//! Output (tab-separated for easy `awk`/`column -t` consumption):
//!
//!   worker  address               embeds  errors  avg_us  max_us  up_s
//!   pi-a    100.77.59.83:50051    3450    2       1234    9876    412
//!   pi-b    100.77.59.84:50051    ERROR: connect timeout
//!
//! Exit code:
//!   0   all workers reported stats cleanly
//!   1   bad CLI args / discovery failed
//!   2   one or more workers errored on the stats RPC

use std::sync::Arc;

use ruvector_hailo_cluster::transport::WorkerEndpoint;
use ruvector_hailo_cluster::{
    Discovery, FileDiscovery, GrpcTransport, HailoClusterEmbedder, StaticDiscovery,
    TailscaleDiscovery,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut workers_arg: Option<String> = None;
    let mut workers_file_arg: Option<String> = None;
    // ADR-172 §1c iter-107 — see embed.rs for the rationale.
    let mut workers_file_sig: Option<String> = None;
    let mut workers_file_pubkey: Option<String> = None;
    let mut tag_arg: Option<String> = None;
    let mut port_arg: u16 = 50051;
    let mut json_output = false;
    let mut prom_output = false;
    let mut prom_file: Option<String> = None;
    let mut watch_secs: Option<u64> = None;
    let mut watch_max_iters: u64 = 0;
    let mut strict_homogeneous = false;
    // When set, print discovered workers and exit — no health/stats RPCs.
    // Useful for verifying a --workers-file manifest expands as expected,
    // or for resolving a tailscale tag → IPs without hitting the workers.
    let mut list_workers = false;
    // Iter 189 — TLS / mTLS knobs (mirror of iter-187 bench + iter-188
    // embed). Lets ops snapshot fleet stats from a TLS-configured
    // worker with the same flag surface as the other client tools.
    #[cfg(feature = "tls")]
    let mut tls_ca: Option<String> = None;
    #[cfg(feature = "tls")]
    let mut tls_domain: Option<String> = None;
    #[cfg(feature = "tls")]
    let mut tls_client_cert: Option<String> = None;
    #[cfg(feature = "tls")]
    let mut tls_client_key: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--workers" => {
                workers_arg = args.get(i + 1).cloned();
                i += 2;
            }
            "--workers-file" => {
                workers_file_arg = args.get(i + 1).cloned();
                i += 2;
            }
            "--workers-file-sig" => {
                workers_file_sig = args.get(i + 1).cloned();
                i += 2;
            }
            "--workers-file-pubkey" => {
                workers_file_pubkey = args.get(i + 1).cloned();
                i += 2;
            }
            "--tailscale-tag" => {
                tag_arg = args.get(i + 1).cloned();
                i += 2;
            }
            "--port" => {
                port_arg = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(50051);
                i += 2;
            }
            "--json" => {
                json_output = true;
                i += 1;
            }
            "--prom" => {
                prom_output = true;
                i += 1;
            }
            "--prom-file" => {
                prom_file = args.get(i + 1).cloned();
                if prom_file.is_none() {
                    return Err("--prom-file needs a path argument".into());
                }
                i += 2;
            }
            "--watch" => {
                watch_secs = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .filter(|&n| n > 0);
                if watch_secs.is_none() {
                    return Err("--watch needs a positive integer (seconds)".into());
                }
                i += 2;
            }
            "--max-iters" => {
                watch_max_iters = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--strict-homogeneous" => {
                strict_homogeneous = true;
                i += 1;
            }
            "--list-workers" => {
                list_workers = true;
                i += 1;
            }
            #[cfg(feature = "tls")]
            "--tls-ca" => {
                tls_ca = args.get(i + 1).cloned();
                i += 2;
            }
            #[cfg(feature = "tls")]
            "--tls-domain" => {
                tls_domain = args.get(i + 1).cloned();
                i += 2;
            }
            #[cfg(feature = "tls")]
            "--tls-client-cert" => {
                tls_client_cert = args.get(i + 1).cloned();
                i += 2;
            }
            #[cfg(feature = "tls")]
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
    if json_output && (prom_output || prom_file.is_some()) {
        return Err("--json and --prom/--prom-file are mutually exclusive".into());
    }
    if prom_output && prom_file.is_some() {
        return Err("--prom (stdout) and --prom-file (atomic file) are mutually exclusive".into());
    }

    let discovery: Box<dyn Discovery> = match (workers_arg, workers_file_arg, tag_arg) {
        (Some(csv), None, None) => {
            let workers: Vec<WorkerEndpoint> = csv
                .split(',')
                .filter(|s| !s.is_empty())
                .enumerate()
                .map(|(i, addr)| {
                    WorkerEndpoint::new(format!("static-{}", i), addr.trim().to_string())
                })
                .collect();
            Box::new(StaticDiscovery::new(workers))
        }
        (None, Some(path), None) => {
            let mut fd = FileDiscovery::new(path);
            match (&workers_file_sig, &workers_file_pubkey) {
                (Some(s), Some(p)) => fd = fd.with_signature(s, p),
                (Some(_), None) | (None, Some(_)) => {
                    return Err(
                        "--workers-file-sig and --workers-file-pubkey must both be set or both unset (ADR-172 §1c)"
                            .into(),
                    );
                }
                (None, None) => {}
            }
            Box::new(fd)
        }
        (None, None, Some(tag)) => Box::new(TailscaleDiscovery::new(tag, port_arg)),
        (None, None, None) => {
            return Err(
                "must pass exactly one of --workers / --workers-file / --tailscale-tag".into(),
            );
        }
        _ => {
            return Err(
                "discovery flags are mutually exclusive: pick one of --workers, --workers-file, --tailscale-tag".into(),
            );
        }
    };

    let workers = discovery.discover()?;
    if workers.is_empty() {
        return Err("discovery returned 0 workers".into());
    }

    // --list-workers short-circuit: print the discovered list and exit.
    // Useful for verifying a manifest, debugging a tailscale tag query,
    // or driving downstream tooling without paying the health-probe RTT.
    if list_workers {
        println!("worker\taddress");
        for w in &workers {
            println!("{}\t{}", w.name, w.address);
        }
        return Ok(());
    }

    // Cluster is built solely to call fleet_stats — dim + fingerprint
    // values don't matter for that path.
    //
    // Iter 189 — TLS transport when --tls-ca is set. Same partial-config
    // + orphan-flag refusals as iter-187/188 client tools.
    #[cfg(feature = "tls")]
    let transport = {
        if let Some(ca_path) = tls_ca.as_deref() {
            let addr0 = workers
                .first()
                .map(|w| w.address.clone())
                .unwrap_or_default();
            let domain = tls_domain.clone().unwrap_or_else(|| {
                ruvector_hailo_cluster::tls::domain_from_address(&addr0).to_string()
            });
            let mut tls = ruvector_hailo_cluster::tls::TlsClient::from_pem_files(ca_path, &domain)
                .map_err(|e| format!("--tls-ca: {}", e))?;
            match (tls_client_cert.as_deref(), tls_client_key.as_deref()) {
                (Some(c), Some(k)) => {
                    tls = tls
                        .with_client_identity(c, k)
                        .map_err(|e| format!("--tls-client-cert/--tls-client-key: {}", e))?;
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(
                        "--tls-client-cert and --tls-client-key must both be set or both unset"
                            .into(),
                    );
                }
                (None, None) => {}
            }
            Arc::new(GrpcTransport::with_tls(
                std::time::Duration::from_secs(5),
                std::time::Duration::from_secs(2),
                tls,
            )?)
        } else {
            if tls_domain.is_some() || tls_client_cert.is_some() || tls_client_key.is_some() {
                return Err(
                    "--tls-domain / --tls-client-cert / --tls-client-key require --tls-ca".into(),
                );
            }
            Arc::new(GrpcTransport::new()?)
        }
    };
    #[cfg(not(feature = "tls"))]
    let transport = Arc::new(GrpcTransport::new()?);
    let cluster = HailoClusterEmbedder::new(workers, transport, 1, "")?;

    // Returns (had_error, unique_fingerprint_count). Drift is anything ≥ 2.
    let render_once = |cluster: &HailoClusterEmbedder,
                       json: bool,
                       prom: bool,
                       prom_file: Option<&str>|
     -> (bool, usize) {
        let mut had_error = false;
        // fleet_state pairs health() + stats() per worker so we get
        // fingerprint and counters in a single render pass — operators
        // see model drift in the same table as throughput.
        let snapshots = cluster.fleet_state();
        let unique_fingerprints: std::collections::HashSet<&str> = snapshots
            .iter()
            .filter_map(|m| m.fingerprint.as_deref())
            .filter(|fp| !fp.is_empty())
            .collect();
        let unique_count = unique_fingerprints.len();
        if let Some(path) = prom_file {
            let mut buf = String::new();
            buf.push_str(&prom_header_string());
            for m in &snapshots {
                match &m.stats {
                    Ok(s) => buf.push_str(&prom_row_string(
                        &m.endpoint.name,
                        &m.endpoint.address,
                        m.fingerprint.as_deref().unwrap_or(""),
                        m.npu_temp_ts0_celsius,
                        m.npu_temp_ts1_celsius,
                        s,
                    )),
                    Err(e) => {
                        eprintln!(
                            "ruvector-hailo-stats: worker {} stats failed: {}",
                            m.endpoint.name, e
                        );
                        had_error = true;
                    }
                }
            }
            if let Err(e) = atomic_write(path, &buf) {
                eprintln!(
                    "ruvector-hailo-stats: failed to write prom file {}: {}",
                    path, e
                );
                had_error = true;
            }
            return (had_error, unique_count);
        }
        if json {
            for m in &snapshots {
                match &m.stats {
                    Ok(s) => {
                        let line = serde_json::json!({
                            "worker":      m.endpoint.name,
                            "address":     m.endpoint.address,
                            "fingerprint": m.fingerprint,
                            "npu_temp_ts0_celsius": m.npu_temp_ts0_celsius,
                            "npu_temp_ts1_celsius": m.npu_temp_ts1_celsius,
                            "stats":       s,
                        });
                        println!("{}", line);
                    }
                    Err(e) => {
                        let line = serde_json::json!({
                            "worker":      m.endpoint.name,
                            "address":     m.endpoint.address,
                            "fingerprint": m.fingerprint,
                            "npu_temp_ts0_celsius": m.npu_temp_ts0_celsius,
                            "npu_temp_ts1_celsius": m.npu_temp_ts1_celsius,
                            "error":       e.to_string(),
                        });
                        println!("{}", line);
                        had_error = true;
                    }
                }
            }
        } else if prom {
            emit_prom_header();
            for m in &snapshots {
                match &m.stats {
                    Ok(s) => emit_prom_row(
                        &m.endpoint.name,
                        &m.endpoint.address,
                        m.fingerprint.as_deref().unwrap_or(""),
                        m.npu_temp_ts0_celsius,
                        m.npu_temp_ts1_celsius,
                        s,
                    ),
                    Err(e) => {
                        eprintln!(
                            "ruvector-hailo-stats: worker {} stats failed: {}",
                            m.endpoint.name, e
                        );
                        had_error = true;
                    }
                }
            }
        } else {
            // Iter-105: TSV gains `rl_denials` + `rl_peers` columns. New
            // columns are appended to the right so existing scripts that
            // index by column number keep working through the upgrade.
            println!("worker\taddress\tfingerprint\tnpu_t0\tnpu_t1\tembeds\terrors\tavg_us\tmax_us\tup_s\trl_denials\trl_peers");
            for m in &snapshots {
                let fp = m.fingerprint.as_deref().unwrap_or("?");
                let t0 = m
                    .npu_temp_ts0_celsius
                    .map(|t| format!("{:.1}", t))
                    .unwrap_or_else(|| "?".into());
                let t1 = m
                    .npu_temp_ts1_celsius
                    .map(|t| format!("{:.1}", t))
                    .unwrap_or_else(|| "?".into());
                match &m.stats {
                    Ok(s) => {
                        let avg_us = s
                            .average_latency()
                            .map(|d| d.as_micros() as u64)
                            .unwrap_or(0);
                        let max_us = s.latency_max.map(|d| d.as_micros() as u64).unwrap_or(0);
                        println!(
                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                            m.endpoint.name,
                            m.endpoint.address,
                            fp,
                            t0,
                            t1,
                            s.embed_count,
                            s.error_count,
                            avg_us,
                            max_us,
                            s.uptime.as_secs(),
                            s.rate_limit_denials,
                            s.rate_limit_tracked_peers,
                        );
                    }
                    Err(e) => {
                        println!(
                            "{}\t{}\t{}\t{}\t{}\tERROR: {}",
                            m.endpoint.name, m.endpoint.address, fp, t0, t1, e
                        );
                        had_error = true;
                    }
                }
            }
        }
        (had_error, unique_count)
    };

    if let Some(secs) = watch_secs {
        // Watch mode — clear-screen between TSV snapshots, append-only
        // for --json / --prom (so the output stays a stream); silent
        // file overwrite for --prom-file (textfile-collector contract).
        // `watch_max_iters > 0` bounds the run for CI scripts that need
        // a clean exit after a fixed sample window.
        // First render outside the loop — gives `last_*` real values
        // before any branch can read them, no dead-store warnings.
        if !json_output && !prom_output && prom_file.is_none() {
            print!("\x1b[2J\x1b[H");
        }
        let (mut last_had_error, mut last_unique) =
            render_once(&cluster, json_output, prom_output, prom_file.as_deref());
        if last_unique > 1 {
            eprintln!(
                "ruvector-hailo-stats: DRIFT: {} unique fingerprints across the fleet",
                last_unique
            );
        }
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let mut iter: u64 = 1;

        // `watch_max_iters > 0` bounds the run for CI scripts that need
        // a clean exit after a fixed sample window.
        while watch_max_iters == 0 || iter < watch_max_iters {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            if !json_output && !prom_output && prom_file.is_none() {
                print!("\x1b[2J\x1b[H");
            }
            let (had_error, unique) =
                render_once(&cluster, json_output, prom_output, prom_file.as_deref());
            last_had_error = had_error;
            last_unique = unique;
            if unique > 1 {
                eprintln!(
                    "ruvector-hailo-stats: DRIFT: {} unique fingerprints across the fleet",
                    unique
                );
            }
            std::io::Write::flush(&mut std::io::stdout()).ok();
            iter += 1;
        }
        // Apply the same exit-code logic as the one-shot path so a
        // bounded watch run integrates cleanly with CI gates.
        if last_had_error {
            std::process::exit(2);
        }
        if strict_homogeneous && last_unique > 1 {
            std::process::exit(3);
        }
        return Ok(());
    }

    let (had_error, unique) = render_once(&cluster, json_output, prom_output, prom_file.as_deref());

    if unique > 1 {
        eprintln!(
            "ruvector-hailo-stats: DRIFT: {} unique fingerprints across the fleet",
            unique
        );
    }

    if had_error {
        std::process::exit(2);
    }
    if strict_homogeneous && unique > 1 {
        std::process::exit(3);
    }
    Ok(())
}

/// Emit the HELP/TYPE preamble for the Prometheus output.
/// Single block at start so the textfile-collector can scrape it efficiently.
fn emit_prom_header() {
    for (name, help, kind) in PROM_METRIC_DEFS {
        println!("# HELP {} {}", name, help);
        println!("# TYPE {} {}", name, kind);
    }
}

/// Single source of truth for the textfile-collector metric catalogue.
/// Iter-105: added `ruvector_rate_limit_*` for ADR-172 §3b visibility.
const PROM_METRIC_DEFS: &[(&str, &str, &str)] = &[
    (
        "ruvector_embed_count_total",
        "Successful embed RPCs since worker boot.",
        "counter",
    ),
    (
        "ruvector_error_count_total",
        "Failed embed RPCs since worker boot.",
        "counter",
    ),
    (
        "ruvector_health_count_total",
        "Health probes received since worker boot.",
        "counter",
    ),
    (
        "ruvector_latency_microseconds_sum",
        "Cumulative microseconds spent in successful embed RPCs.",
        "counter",
    ),
    (
        "ruvector_latency_microseconds_min",
        "Smallest microsecond latency observed since boot.",
        "gauge",
    ),
    (
        "ruvector_latency_microseconds_max",
        "Largest microsecond latency observed since boot.",
        "gauge",
    ),
    (
        "ruvector_uptime_seconds",
        "Seconds since worker process started.",
        "gauge",
    ),
    (
        "ruvector_npu_temp_celsius",
        "Hailo-8 on-die thermal sensor reading (sensor=ts0|ts1).",
        "gauge",
    ),
    (
        "ruvector_rate_limit_denials_total",
        "ResourceExhausted returned by the per-peer rate limiter.",
        "counter",
    ),
    (
        "ruvector_rate_limit_tracked_peers",
        "Distinct peers seen by the rate limiter since boot.",
        "gauge",
    ),
];

/// Emit one row per metric for a worker. Fingerprint goes on every row
/// as a label so PromQL filters like `{fingerprint="fp:current"}` work
/// for fleet drift detection.
#[allow(clippy::too_many_arguments)]
fn emit_prom_row(
    name: &str,
    address: &str,
    fingerprint: &str,
    npu_t0: Option<f32>,
    npu_t1: Option<f32>,
    s: &ruvector_hailo_cluster::transport::StatsSnapshot,
) {
    let labels = format!(
        "{{worker={:?},address={:?},fingerprint={:?}}}",
        name, address, fingerprint
    );
    println!("ruvector_embed_count_total{} {}", labels, s.embed_count);
    println!("ruvector_error_count_total{} {}", labels, s.error_count);
    println!("ruvector_health_count_total{} {}", labels, s.health_count);
    println!(
        "ruvector_latency_microseconds_sum{} {}",
        labels,
        s.latency_sum.as_micros() as u64
    );
    if let Some(d) = s.latency_min {
        println!(
            "ruvector_latency_microseconds_min{} {}",
            labels,
            d.as_micros() as u64
        );
    }
    if let Some(d) = s.latency_max {
        println!(
            "ruvector_latency_microseconds_max{} {}",
            labels,
            d.as_micros() as u64
        );
    }
    println!("ruvector_uptime_seconds{} {}", labels, s.uptime.as_secs());
    if let Some(t) = npu_t0 {
        // Append `sensor` label so PromQL can split by ts0/ts1.
        println!(
            "ruvector_npu_temp_celsius{{worker={:?},address={:?},fingerprint={:?},sensor=\"ts0\"}} {:.3}",
            name, address, fingerprint, t
        );
    }
    if let Some(t) = npu_t1 {
        println!(
            "ruvector_npu_temp_celsius{{worker={:?},address={:?},fingerprint={:?},sensor=\"ts1\"}} {:.3}",
            name, address, fingerprint, t
        );
    }
    // Iter-105 (ADR-172 §3b follow-up): rate-limiter visibility. Always
    // emit (even when 0/0) so PromQL alerts on a *change* don't have to
    // discriminate between "metric missing" and "metric present at 0".
    println!(
        "ruvector_rate_limit_denials_total{} {}",
        labels, s.rate_limit_denials
    );
    println!(
        "ruvector_rate_limit_tracked_peers{} {}",
        labels, s.rate_limit_tracked_peers
    );
}

/// String version of `emit_prom_header` so the file path can build the
/// whole document in memory before atomic write.
fn prom_header_string() -> String {
    let mut s = String::new();
    for (name, help, kind) in PROM_METRIC_DEFS {
        s.push_str(&format!("# HELP {} {}\n", name, help));
        s.push_str(&format!("# TYPE {} {}\n", name, kind));
    }
    s
}

/// String version of `emit_prom_row`. See `emit_prom_row` for the
/// rationale on adding `fingerprint` as a label.
#[allow(clippy::too_many_arguments)]
fn prom_row_string(
    name: &str,
    address: &str,
    fingerprint: &str,
    npu_t0: Option<f32>,
    npu_t1: Option<f32>,
    s: &ruvector_hailo_cluster::transport::StatsSnapshot,
) -> String {
    let labels = format!(
        "{{worker={:?},address={:?},fingerprint={:?}}}",
        name, address, fingerprint
    );
    let mut out = String::new();
    out.push_str(&format!(
        "ruvector_embed_count_total{} {}\n",
        labels, s.embed_count
    ));
    out.push_str(&format!(
        "ruvector_error_count_total{} {}\n",
        labels, s.error_count
    ));
    out.push_str(&format!(
        "ruvector_health_count_total{} {}\n",
        labels, s.health_count
    ));
    out.push_str(&format!(
        "ruvector_latency_microseconds_sum{} {}\n",
        labels,
        s.latency_sum.as_micros() as u64
    ));
    if let Some(d) = s.latency_min {
        out.push_str(&format!(
            "ruvector_latency_microseconds_min{} {}\n",
            labels,
            d.as_micros() as u64
        ));
    }
    if let Some(d) = s.latency_max {
        out.push_str(&format!(
            "ruvector_latency_microseconds_max{} {}\n",
            labels,
            d.as_micros() as u64
        ));
    }
    out.push_str(&format!(
        "ruvector_uptime_seconds{} {}\n",
        labels,
        s.uptime.as_secs()
    ));
    if let Some(t) = npu_t0 {
        out.push_str(&format!(
            "ruvector_npu_temp_celsius{{worker={:?},address={:?},fingerprint={:?},sensor=\"ts0\"}} {:.3}\n",
            name, address, fingerprint, t
        ));
    }
    if let Some(t) = npu_t1 {
        out.push_str(&format!(
            "ruvector_npu_temp_celsius{{worker={:?},address={:?},fingerprint={:?},sensor=\"ts1\"}} {:.3}\n",
            name, address, fingerprint, t
        ));
    }
    // Iter-105: rate-limiter visibility (mirrors emit_prom_row).
    out.push_str(&format!(
        "ruvector_rate_limit_denials_total{} {}\n",
        labels, s.rate_limit_denials
    ));
    out.push_str(&format!(
        "ruvector_rate_limit_tracked_peers{} {}\n",
        labels, s.rate_limit_tracked_peers
    ));
    out
}

/// Write `contents` to `path` atomically: write to `<path>.tmp`, fsync,
/// then rename. node_exporter's textfile collector relies on this
/// guarantee to never see a half-written scrape.
fn atomic_write(path: &str, contents: &str) -> std::io::Result<()> {
    use std::io::Write as _;
    let tmp = format!("{}.tmp", path);
    let mut f = std::fs::File::create(&tmp)?;
    f.write_all(contents.as_bytes())?;
    f.sync_all()?;
    drop(f);
    std::fs::rename(&tmp, path)
}

fn print_help() {
    eprintln!(
        "ruvector-hailo-stats — Hailo cluster fleet stats snapshot

USAGE:
    ruvector-hailo-stats [OPTIONS]

DISCOVERY (exactly one):
    --workers <addr1,addr2,...>     Static worker list (csv of host:port).
    --workers-file <path>           Manifest file: one `host:port` or
                                     `name = host:port` per line; `#`
                                     comments allowed.
    --tailscale-tag <tag> [--port N]  Discover via tailscale; tag matches
                                     peers, port is the gRPC port.

OPTIONS:
    --json                          Emit NDJSON (one object per worker).
    --prom                          Emit Prometheus textfile-collector
                                    format on stdout.
    --prom-file <path>              Atomically write Prometheus output
                                    to <path>. Pair with --watch for
                                    drop-in node_exporter textfile-
                                    collector monitoring (overwrites
                                    every tick; scraper sees latest).
    --watch <N>                     Re-render every N seconds; clears
                                    the screen between TSV snapshots.
                                    Ctrl-C to exit.
    --max-iters <N>                 Bound a --watch run to N iterations
                                    then exit cleanly (default 0 =
                                    unbounded). Pairs with CI gates so
                                    --strict-homogeneous / had-error
                                    exits land after a fixed sample.
    --strict-homogeneous            Exit with code 3 if more than one
                                    unique fingerprint is observed
                                    across the fleet (drift detection).
                                    Always emits a DRIFT line to stderr
                                    when drift is detected; the flag
                                    only changes the exit code.
    --list-workers                  Print discovered workers (TSV
                                    name\taddress) and exit. No health
                                    probe, no stats RPC — pure discovery
                                    output. Useful for verifying a
                                    --workers-file manifest or resolving
                                    a tailscale tag without hitting
                                    workers.
    --tls-ca <path>                 Enable HTTPS by trusting the PEM CA
                                    bundle at <path>. Without this the
                                    stats CLI dials plaintext gRPC.
                                    (Requires --features tls.)
    --tls-domain <name>             SNI / SAN value to assert against
                                    the server cert. Defaults to the
                                    hostname half of the first worker
                                    address.
    --tls-client-cert <path>        mTLS client cert (PEM). Pair with
                                    --tls-client-key.
    --tls-client-key <path>         mTLS client private key (PEM). Both
                                    cert and key must be set or both
                                    unset.
    --help, -h                      Print this help and exit.
    --version, -V                   Print the binary name + version and exit.

OUTPUT:
    Default: TSV (header + one row per worker).
    --json:  NDJSON, one JSON object per worker.
    --prom:  Prometheus exposition format, ready for node_exporter
             textfile-collector scraping.

EXIT CODES:
    0   all workers reported stats cleanly
    1   bad CLI args / discovery failed
    2   one or more workers errored on the stats RPC
    3   --strict-homogeneous and fleet has > 1 unique fingerprint
"
    );
}
