//! `ruvector-hailo-embed` — CLI for the Hailo embedding path.
//!
//! Reads text from stdin (one document per line), embeds each via a
//! configured cluster of workers, prints one JSON line per input.
//!
//! Usage:
//!
//!   ruvector-hailo-embed --workers <addr1,addr2,...> --dim 384
//!   ruvector-hailo-embed --tailscale-tag tag:ruvector-hailo-worker --port 50051 --dim 384
//!   echo "hello world" | ruvector-hailo-embed --workers 100.77.59.83:50051 --dim 384
//!
//! Output schema (one JSON object per stdin line):
//!
//!   {"text": "...", "dim": 384, "latency_us": 1234, "vec_head": [0.0123, ...]}
//!
//! Only the first 8 components are emitted to keep stdout readable; the
//! full vector is `dim` components — this binary is a demo / smoke test,
//! not a production embedder. For production, embed `HailoClusterEmbedder`
//! directly into your binary via the lib API.

use std::io::{BufRead, Write};
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
    // ADR-172 §1c iter-107: optional Ed25519 detached signature on the
    // manifest. Both must be set together; either alone is a misconfig.
    let mut workers_file_sig: Option<String> = None;
    let mut workers_file_pubkey: Option<String> = None;
    let mut tag_arg: Option<String> = None;
    let mut port_arg: u16 = 50051;
    let mut dim: usize = 384;
    let mut fingerprint: String = String::new();
    let mut batch_size: usize = 0;
    let mut cache_cap: usize = 0;
    let mut cache_ttl_secs: u64 = 0;
    let mut validate_fleet = false;
    let mut validate_only = false;
    let mut auto_fingerprint = false;
    // ADR-172 §2b iter-102: minimum number of workers that must agree
    // on the fingerprint during --auto-fingerprint discovery. 0 = use
    // smart default (1 for single-worker fleets, 2 for ≥2-worker fleets).
    let mut auto_fingerprint_quorum: usize = 0;
    // ADR-172 §2a iter-101 gate: if --cache > 0 is requested but the
    // fingerprint is empty (and didn't get filled in by --auto-fingerprint),
    // refuse to start unless the operator explicitly opted in.
    let mut allow_empty_fingerprint = false;
    let mut request_id: String = String::new();
    // "head" (default), "full", "none". head = first 8 components;
    // full = entire vector; none = drop the vector, keep dim + latency.
    let mut output_mode: String = "head".to_string();
    // Quiet mode suppresses all informational stderr ("X workers", validation
    // summary, end-of-run stats). Errors and validation FAILED lines still
    // print so a CI gate can see why something broke.
    let mut quiet = false;
    // 0 = no background health-checker. >0 = probe every N seconds in
    // a background tokio task; mismatched fingerprints get hard-ejected
    // and the cache is auto-cleared via the cluster's wired callback.
    let mut health_check_secs: u64 = 0;
    // Optional inline texts — when set, stdin is NOT read; the supplied
    // texts are embedded in order and the binary exits. Repeat the
    // flag to embed multiple texts in one invocation.
    let mut inline_texts: Vec<String> = Vec::new();
    // Iter 188 — symmetric TLS plumbing (mirror of iter-187 bench
    // additions). Lets ops drive a single embed against a TLS-enabled
    // worker without building a custom client. All flags
    // `#[cfg(feature = "tls")]` so the no-tls build is unchanged.
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
            "--dim" => {
                dim = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(384);
                i += 2;
            }
            "--fingerprint" => {
                fingerprint = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--batch" => {
                batch_size = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--cache" => {
                cache_cap = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--cache-ttl" => {
                cache_ttl_secs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--validate-fleet" => {
                validate_fleet = true;
                i += 1;
            }
            "--validate-only" => {
                validate_only = true;
                validate_fleet = true;
                i += 1;
            }
            "--auto-fingerprint" => {
                auto_fingerprint = true;
                i += 1;
            }
            "--auto-fingerprint-quorum" => {
                auto_fingerprint_quorum = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--allow-empty-fingerprint" => {
                allow_empty_fingerprint = true;
                i += 1;
            }
            "--request-id" => {
                request_id = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--output" => {
                let v = args.get(i + 1).cloned().unwrap_or_default();
                if !matches!(v.as_str(), "head" | "full" | "none") {
                    return Err(format!("--output must be head|full|none, got {:?}", v).into());
                }
                output_mode = v;
                i += 2;
            }
            "--quiet" => {
                quiet = true;
                i += 1;
            }
            "--health-check" => {
                health_check_secs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--text" => {
                if let Some(t) = args.get(i + 1).cloned() {
                    inline_texts.push(t);
                } else {
                    return Err("--text needs a string argument".into());
                }
                i += 2;
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

    // Resolve discovery source. Mutually exclusive: pick exactly one.
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
            // ADR-172 §1c iter-107: if either of the sig flags is set,
            // both must be — refuse partial config so an operator can't
            // accidentally disable verification by forgetting one half.
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
                "must pass exactly one of --workers <csv> / --workers-file <path> / --tailscale-tag <tag>".into(),
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
        return Err("discovery returned 0 workers — nothing to dispatch to".into());
    }

    if !quiet {
        eprintln!("ruvector-hailo-embed: {} worker(s):", workers.len());
        for w in &workers {
            eprintln!("  {} -> {}", w.name, w.address);
        }
    }

    // Iter 188 — TLS transport when --tls-ca is set; mirrors iter-187
    // bench plumbing. Same partial-config + orphan-flag refusals so a
    // misconfigured invocation surfaces an early error instead of a
    // silent plaintext downgrade.
    #[cfg(feature = "tls")]
    let transport: Arc<
        dyn ruvector_hailo_cluster::transport::EmbeddingTransport + Send + Sync,
    > = {
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
                    if !quiet {
                        eprintln!("ruvector-hailo-embed: mTLS client identity attached");
                    }
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(
                        "--tls-client-cert and --tls-client-key must both be set or both unset"
                            .into(),
                    );
                }
                (None, None) => {}
            }
            if !quiet {
                eprintln!(
                    "ruvector-hailo-embed: TLS enabled ca={} domain={}",
                    ca_path, domain
                );
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
    let transport: Arc<
        dyn ruvector_hailo_cluster::transport::EmbeddingTransport + Send + Sync,
    > = Arc::new(GrpcTransport::new()?);

    // Auto-discover fingerprint from the fleet if requested. Quorum mode
    // (ADR-172 §2b iter 102): when fleet has ≥2 workers and operator
    // didn't pin --auto-fingerprint-quorum explicitly, default to 2 so a
    // single hostile/stale worker can't poison the discovered fp. Single-
    // worker dev fleets keep the legacy 1-of-1 behavior.
    if auto_fingerprint {
        let resolved_quorum: usize = if auto_fingerprint_quorum > 0 {
            auto_fingerprint_quorum
        } else if workers.len() >= 2 {
            2
        } else {
            1
        };
        // Need a transient cluster with no enforcement (empty fingerprint)
        // to probe; rebuild below with the discovered value.
        let probe = HailoClusterEmbedder::new(
            workers.clone(),
            Arc::clone(&transport),
            dim,
            "".to_string(),
        )?;
        match probe.discover_fingerprint_with_quorum(resolved_quorum) {
            Ok(fp) if !fp.is_empty() => {
                if !quiet {
                    eprintln!(
                        "ruvector-hailo-embed: --auto-fingerprint (quorum={}) discovered fp={:?} \
                         (overrides --fingerprint)",
                        resolved_quorum, fp
                    );
                }
                fingerprint = fp;
            }
            Ok(_) => {
                if !quiet {
                    eprintln!(
                        "ruvector-hailo-embed: --auto-fingerprint: worker reported empty fingerprint — \
                         skipping enforcement"
                    );
                }
                fingerprint.clear();
            }
            Err(e) => {
                // Errors stay visible even in quiet mode so a CI gate
                // can see *why* enforcement was skipped.
                eprintln!(
                    "ruvector-hailo-embed: --auto-fingerprint failed: {} (continuing without enforcement)",
                    e
                );
                fingerprint.clear();
            }
        }
    }

    // ADR-172 §2a HIGH-MEDIUM mitigation (iter 101): refuse to enable
    // the in-process cache without a fingerprint to bind it to. An empty
    // fingerprint means *any* worker can poison the cache (silent stale
    // serve from a mismatched HEF/vocab). Operators who explicitly want
    // the legacy behavior pass --allow-empty-fingerprint.
    if cache_cap > 0 && fingerprint.is_empty() && !allow_empty_fingerprint {
        return Err(
            "refusing --cache > 0 with empty fingerprint (ADR-172 §2a); pass \
             --fingerprint <hex> or --auto-fingerprint, or opt out with \
             --allow-empty-fingerprint"
                .into(),
        );
    }

    let cluster = {
        let c = HailoClusterEmbedder::new(workers, transport, dim, fingerprint)?;
        match (cache_cap, cache_ttl_secs) {
            (0, _) => c,
            (cap, 0) => c.with_cache(cap),
            (cap, ttl) => c.with_cache_ttl(cap, std::time::Duration::from_secs(ttl)),
        }
    };

    if validate_fleet {
        match cluster.validate_fleet() {
            Ok(report) => {
                if !quiet {
                    eprintln!(
                        "ruvector-hailo-embed: fleet validation: {} healthy, {} mismatched fp, {} not ready, {} unreachable",
                        report.healthy.len(),
                        report.fingerprint_mismatched.len(),
                        report.not_ready.len(),
                        report.unreachable.len(),
                    );
                    if !report.healthy.is_empty() {
                        eprintln!("  healthy: {}", report.healthy.join(", "));
                    }
                    for m in &report.fingerprint_mismatched {
                        eprintln!(
                            "  EJECTED {}: expected fp={:?}, actual fp={:?}",
                            m.worker, m.expected, m.actual
                        );
                    }
                    for n in &report.not_ready {
                        eprintln!("  not_ready: {}", n);
                    }
                    for (n, e) in &report.unreachable {
                        eprintln!("  unreachable: {} ({})", n, e);
                    }
                }
                if validate_only {
                    return Ok(());
                }
            }
            Err(e) => {
                // Validation FAILED stays visible in quiet — the exit
                // code alone isn't enough context for a CI alert.
                eprintln!("ruvector-hailo-embed: fleet validation FAILED: {}", e);
                std::process::exit(2);
            }
        }
    }

    // Background health-checker — when --health-check N is set, spawn
    // a tokio runtime for the lifetime of main. Bound to a name-prefixed
    // local so the runtime + checker don't drop until main returns;
    // dropping the runtime aborts the checker task cleanly.
    let _health_keepalive = if health_check_secs > 0 {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name("health-check")
            .build()
            .map_err(|e| format!("health-check runtime: {}", e))?;
        let cfg = ruvector_hailo_cluster::HealthCheckerConfig {
            interval: std::time::Duration::from_secs(health_check_secs),
            ..cluster.health_checker_config()
        };
        let checker = cluster.spawn_health_checker(rt.handle(), cfg);
        if !quiet {
            eprintln!(
                "ruvector-hailo-embed: --health-check spawned, interval={}s",
                health_check_secs
            );
        }
        Some((rt, checker))
    } else {
        None
    };

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let start_total = std::time::Instant::now();
    let mut count: u64 = 0;
    let mut total_us: u128 = 0;

    if batch_size > 0 {
        // Batch mode — buffer up to batch_size lines, then dispatch via
        // embed_batch_blocking. Latency is per-batch (ns of cache + RPC),
        // amortised across N inputs in the JSON output.
        let mut buf: Vec<String> = Vec::with_capacity(batch_size);
        let flush = |buf: &mut Vec<String>,
                     out: &mut std::io::StdoutLock,
                     count: &mut u64,
                     total_us: &mut u128|
         -> Result<(), Box<dyn std::error::Error>> {
            if buf.is_empty() {
                return Ok(());
            }
            let t0 = std::time::Instant::now();
            // --request-id, when set, threads the supplied id through
            // every batch RPC. Multiple batches in one run share the
            // same id — caller's responsibility to supply a unique
            // token per logical request if that matters.
            let result = if request_id.is_empty() {
                cluster.embed_batch_blocking(buf)
            } else {
                cluster.embed_batch_blocking_with_request_id(buf, &request_id)
            };
            let batch_us = t0.elapsed().as_micros();
            match result {
                Ok(vecs) => {
                    let per_item_us = if !buf.is_empty() {
                        batch_us / (buf.len() as u128)
                    } else {
                        0
                    };
                    for (text, vec) in buf.iter().zip(vecs.iter()) {
                        *total_us += per_item_us;
                        *count += 1;
                        writeln!(
                            out,
                            "{{\"text\":{:?},\"dim\":{},\"latency_us\":{}{}}}",
                            text,
                            vec.len(),
                            per_item_us,
                            format_vector_field(&output_mode, vec)
                        )?;
                    }
                }
                Err(e) => {
                    // One error per text so downstream tooling sees the
                    // structure even when the whole batch failed.
                    let err_str = e.to_string();
                    for text in buf.iter() {
                        writeln!(out, "{{\"text\":{:?},\"error\":{:?}}}", text, err_str)?;
                    }
                }
            }
            buf.clear();
            Ok(())
        };

        // --text inputs bypass stdin entirely — one-shot mode for
        // shell scripts and ad-hoc queries. When any --text is set,
        // stdin is NOT read.
        if !inline_texts.is_empty() {
            for text in &inline_texts {
                let text = text.trim().to_string();
                if text.is_empty() {
                    continue;
                }
                buf.push(text);
                if buf.len() >= batch_size {
                    flush(&mut buf, &mut out, &mut count, &mut total_us)?;
                }
            }
            flush(&mut buf, &mut out, &mut count, &mut total_us)?;
        } else {
            for line in stdin.lock().lines() {
                let text = line?;
                let text = text.trim().to_string();
                if text.is_empty() {
                    continue;
                }
                buf.push(text);
                if buf.len() >= batch_size {
                    flush(&mut buf, &mut out, &mut count, &mut total_us)?;
                }
            }
            flush(&mut buf, &mut out, &mut count, &mut total_us)?;
        }
    } else {
        // Per-line mode. Same source-or-stdin branch as batch mode.
        let inline_iter = inline_texts.iter().map(|t| Ok(t.clone()));
        let stdin_iter: Box<dyn Iterator<Item = std::io::Result<String>>> =
            if !inline_texts.is_empty() {
                Box::new(inline_iter)
            } else {
                Box::new(stdin.lock().lines())
            };
        for line in stdin_iter {
            let text = line?;
            let text = text.trim();
            if text.is_empty() {
                continue;
            }

            let t0 = std::time::Instant::now();
            let result = if request_id.is_empty() {
                cluster.embed_one_blocking(text)
            } else {
                cluster.embed_one_blocking_with_request_id(text, &request_id)
            };
            match result {
                Ok(vec) => {
                    let elapsed_us = t0.elapsed().as_micros();
                    total_us += elapsed_us;
                    count += 1;
                    writeln!(
                        out,
                        "{{\"text\":{:?},\"dim\":{},\"latency_us\":{}{}}}",
                        text,
                        vec.len(),
                        elapsed_us,
                        format_vector_field(&output_mode, &vec)
                    )?;
                }
                Err(e) => {
                    writeln!(out, "{{\"text\":{:?},\"error\":{:?}}}", text, e.to_string())?;
                }
            }
        }
    }

    if count > 0 && !quiet {
        let elapsed_total = start_total.elapsed();
        eprintln!(
            "ruvector-hailo-embed: {} embeds in {:.3}s (avg {} us/embed, {:.1} embeds/s)",
            count,
            elapsed_total.as_secs_f64(),
            total_us / (count as u128),
            (count as f64) / elapsed_total.as_secs_f64()
        );
        if cache_cap > 0 {
            let s = cluster.cache_stats();
            eprintln!(
                "ruvector-hailo-embed: cache cap={} size={} hits={} misses={} evictions={}",
                s.capacity, s.size, s.hits, s.misses, s.evictions
            );
        }
    }

    Ok(())
}

/// Format the JSON suffix for the vector field based on `--output`:
///   `head` (default): `,"vec_head":[0.0123,...]` (first 8 components)
///   `full`         : `,"vector":[0.0123,...]`     (all components)
///   `none`         : `` (no vector field at all — pure metadata)
/// Returns a leading-comma'd fragment so it can be slotted into a
/// pre-existing JSON object literal without rebuilding the whole line.
fn format_vector_field(mode: &str, vec: &[f32]) -> String {
    match mode {
        "full" => {
            let parts: Vec<String> = vec.iter().map(|f| format!("{:.6}", f)).collect();
            format!(",\"vector\":[{}]", parts.join(","))
        }
        "none" => String::new(),
        _ => {
            // "head" (default) — first 8 components, 4 decimals
            let parts: Vec<String> = vec.iter().take(8).map(|f| format!("{:.4}", f)).collect();
            format!(",\"vec_head\":[{}]", parts.join(","))
        }
    }
}

fn print_help() {
    eprintln!(
        "ruvector-hailo-embed — Hailo cluster embedding CLI

USAGE:
    ruvector-hailo-embed [OPTIONS]

DISCOVERY (exactly one):
    --workers <addr1,addr2,...>     Static worker list (csv of host:port).
    --workers-file <path>           Manifest file: one `host:port` or
                                     `name = host:port` per line; blank
                                     lines and `#` comments allowed.
    --workers-file-sig <path>       Optional Ed25519 detached signature
                                     on the manifest (128 hex chars in a
                                     text file). Pair with the matching
                                     pubkey flag to enforce manifest
                                     integrity (ADR-172 §1c).
    --workers-file-pubkey <path>    32-byte Ed25519 verifying key as 64
                                     hex chars in a text file. Required
                                     when --workers-file-sig is set.
    --tailscale-tag <tag> [--port N]  Discover via tailscale; tag matches
                                     peers, port is the gRPC port.

OPTIONS:
    --dim <N>                       Expected embedding dim (default 384).
    --fingerprint <hex>             Reject workers reporting different fp.
                                    Empty = skip the check.
    --batch <N>                     Buffer up to N stdin lines per batch
                                     and dispatch via the streaming RPC
                                     (1 RPC per batch, ordered output).
                                     0 = per-line mode (default).
    --cache <N>                     Enable in-process LRU cache of size N.
                                     Repeat texts (per-line or batched)
                                     return without RPC. 0 = disabled.
    --cache-ttl <secs>              Optional TTL for cached entries (only
                                     used if --cache > 0). Entries older
                                     than <secs> seconds are reported as
                                     misses + counted as evictions.
                                     0 = no TTL (default; LRU only).
    --validate-fleet                Probe every worker on startup; eject
                                     any whose model fingerprint doesn't
                                     match --fingerprint. Print summary
                                     to stderr; exit 2 if 0 workers OK.
    --validate-only                 Validate as above, then exit without
                                     reading stdin. CI-friendly health
                                     gate.
    --auto-fingerprint              Probe the fleet for its fingerprint
                                     and use that as the expected value.
                                     Pairs with --validate-fleet to
                                     auto-discover then enforce homogeneity.
    --auto-fingerprint-quorum <N>   Minimum workers that must agree on the
                                     fingerprint during --auto-fingerprint
                                     (ADR-172 §2b). Default: 2 if fleet has
                                     ≥2 workers, 1 otherwise. Set to 1 to
                                     bypass the quorum check.
    --allow-empty-fingerprint       Opt out of the ADR-172 §2a safety gate
                                     that refuses --cache > 0 when the
                                     fingerprint is empty. Useful only for
                                     legacy fleets that haven't published a
                                     fingerprint yet; risks silent stale-
                                     serve from a mismatched HEF.
    --request-id <id>               Caller-supplied tracing token sent
                                     with every embed RPC via gRPC
                                     metadata. Workers' tracing spans
                                     log this id verbatim, so it can be
                                     grepped from the bottom of a
                                     web-handler → coordinator →
                                     worker chain.
    --output head|full|none         Vector serialisation mode (default
                                     head). head = first 8 components
                                     in `vec_head`; full = entire vector
                                     in `vector`; none = no vector field
                                     (metadata-only output for I/O-free
                                     benchmarks).
    --quiet                         Suppress all informational stderr
                                     (worker list, validation summary,
                                     end-of-run stats). Errors and
                                     validation FAILED still print so
                                     CI gates can see why.
    --health-check <secs>           Spawn a background health-checker
                                     that probes every <secs> seconds.
                                     Mismatched fingerprints get hard-
                                     ejected from the dispatch pool +
                                     auto-clear the cache. 0 = disabled.
    --text <string>                 One-shot mode: embed the supplied
                                     text and exit (skips stdin). Repeat
                                     the flag to embed multiple texts
                                     in one invocation.
    --tls-ca <path>                 Enable HTTPS by trusting the PEM CA
                                     bundle at <path>. Without this the
                                     embed CLI dials plaintext gRPC.
                                     (Requires --features tls.)
    --tls-domain <name>             SNI / SAN value to assert against the
                                     server cert. Defaults to the hostname
                                     half of the first worker address.
    --tls-client-cert <path>        mTLS client cert (PEM). Pair with
                                     --tls-client-key.
    --tls-client-key <path>         mTLS client private key (PEM). Both
                                     cert and key must be set or both
                                     unset.
    --help, -h                      Print this help and exit.
    --version, -V                   Print the binary name + version and exit.

INPUT:
    Reads stdin one document per line. Empty lines skipped.

OUTPUT:
    One JSON object per stdin line on stdout:
        {{\"text\":..., \"dim\":..., \"latency_us\":..., \"vec_head\":[...]}}
    Errors per-line:
        {{\"text\":..., \"error\":...}}
    Summary stats on stderr at end.
"
    );
}
