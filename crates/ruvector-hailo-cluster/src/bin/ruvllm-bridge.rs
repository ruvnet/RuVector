//! `ruvllm-bridge` — JSONL stdin/stdout adapter from ruvllm-shaped
//! requests to the hailo-backend cluster's embed RPC (iter 124,
//! ADR-173 host-side seam).
//!
//! ruvllm processes that need RAG retrieval don't want to link a tonic
//! client; they want a thin local subprocess that takes JSON in, gives
//! JSON out. This bridge does exactly that — line-delimited requests
//! on stdin, line-delimited responses on stdout.
//!
//! Companion to `ruvector-mmwave-bridge` (iter 116) and
//! `ruview-csi-bridge` (iter 123). Same TLS/mTLS/§2a flag set; same
//! ADR-172 §1b mTLS / §3b rate-limit guards apply by inheritance.
//!
//! # Wire format
//!
//! Request (one JSON object per line):
//! ```json
//!   {"text": "the input string to embed"}
//!   {"text": "another", "request_id": "01HRZK..."}     // optional ID
//! ```
//!
//! Response (one JSON object per line, matching request order):
//! ```json
//!   {"dim": 384, "latency_us": 8147, "vector": [0.012, -0.045, ...]}
//!   {"error": "cluster unreachable: ..."}
//! ```
//!
//! Closing stdin shuts down cleanly (EOF → exit 0).
//!
//! # Usage
//!
//! ```text
//!   ruvllm-bridge --workers 100.77.59.83:50051 --fingerprint <hex>
//!   echo '{"text":"hello world"}' | ruvllm-bridge --workers ...
//! ```
//!
//! HEF compile pipeline lands → cluster's HailoEmbedder serves real
//! semantic vectors → no bridge changes needed; same input/output
//! contract just produces real embeddings instead of FNV-1a placeholders.

use std::io::{BufRead, Write};
use std::sync::Arc;
#[cfg(feature = "tls")]
use std::time::Duration;
use std::time::Instant;

use ruvector_hailo_cluster::transport::{EmbeddingTransport, WorkerEndpoint};
use ruvector_hailo_cluster::{GrpcTransport, HailoClusterEmbedder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

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

    let csv = workers_csv
        .ok_or("ruvllm-bridge requires --workers <csv> — there's no other reason to run it")?;

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
        .map(|(idx, addr)| WorkerEndpoint::new(format!("static-{}", idx), addr.trim().to_string()))
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
            let ca = tls_ca.ok_or("--tls-ca <path> is required when any --tls-* flag is set")?;
            let domain = tls_domain.unwrap_or_else(|| {
                ruvector_hailo_cluster::tls::domain_from_address(&workers[0].address).to_string()
            });
            let mut tls = ruvector_hailo_cluster::tls::TlsClient::from_pem_files(&ca, domain)?;
            match (&tls_client_cert, &tls_client_key) {
                (Some(c), Some(k)) => {
                    tls = tls.with_client_identity(c, k)?;
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

    let cluster = HailoClusterEmbedder::new(workers, transport, dim, fingerprint)?;

    if !quiet {
        eprintln!(
            "ruvllm-bridge: cluster sink active — {} worker(s), dim={}",
            csv.split(',').filter(|s| !s.is_empty()).count(),
            dim,
        );
        eprintln!("ruvllm-bridge: ready — send JSONL on stdin, EOF to exit");
    }

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();
    let mut total_ok = 0u64;
    let mut total_err = 0u64;
    let started = Instant::now();

    for line_res in stdin.lock().lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(e) => {
                writeln!(stdout_lock, r#"{{"error":"stdin read: {}"}}"#, e)?;
                total_err += 1;
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        match handle_request(&cluster, &line) {
            Ok(json) => {
                writeln!(stdout_lock, "{}", json)?;
                total_ok += 1;
            }
            Err(e) => {
                writeln!(
                    stdout_lock,
                    r#"{{"error":{}}}"#,
                    json_string(&e.to_string())
                )?;
                total_err += 1;
            }
        }
        stdout_lock.flush()?;
    }

    if !quiet {
        eprintln!(
            "ruvllm-bridge: stdin closed — exiting; {} ok, {} err over {:?}",
            total_ok,
            total_err,
            started.elapsed(),
        );
    }
    Ok(())
}

/// Process a single JSONL request line and produce the response body
/// (caller wraps with newline). Errors are bubbled as anyhow-shaped
/// strings; the caller renders them as `{"error":"..."}`.
fn handle_request(
    cluster: &HailoClusterEmbedder,
    line: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    // Minimal hand-rolled JSON parser for the request shape — pulling
    // serde_json into the bin would force std::io's Read/Write into
    // serde's mid-line reader, which mishandles line-buffered stdin.
    // The accepted shape is small enough to parse by string-search.
    let text = extract_json_string_field(line, "text")
        .ok_or("request must have a top-level \"text\":\"...\" field")?;
    let request_id = extract_json_string_field(line, "request_id");

    let started = Instant::now();
    let vec = match request_id.as_deref() {
        Some(id) => cluster.embed_one_blocking_with_request_id(&text, id),
        None => cluster.embed_one_blocking(&text),
    }?;
    let latency_us = started.elapsed().as_micros() as u64;

    // Build response by hand — same reasoning as request parser. The
    // vector is the only field that needs care; we float-format with
    // `:?` precision (Debug) so round-trip exactness is preserved.
    let mut out = String::with_capacity(64 + vec.len() * 12);
    out.push_str(&format!(
        r#"{{"dim":{},"latency_us":{},"#,
        vec.len(),
        latency_us
    ));
    if let Some(id) = request_id.as_deref() {
        out.push_str(&format!(r#""request_id":{},"#, json_string(id)));
    }
    out.push_str(r#""vector":["#);
    for (i, f) in vec.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("{:?}", f));
    }
    out.push_str("]}");
    Ok(out)
}

/// Extract `"<field>":"..."` from a JSON-like string. Doesn't handle
/// escaped quotes inside the value — operators sending text with
/// embedded `\"` need to escape the line at the language level. The
/// 99% case (unicode text, sentence-shaped strings) works.
fn extract_json_string_field(s: &str, field: &str) -> Option<String> {
    let needle = format!(r#""{}":"#, field);
    let start = s.find(&needle)? + needle.len();
    let mut iter = s[start..].chars().peekable();
    while iter.peek() == Some(&' ') || iter.peek() == Some(&'\t') {
        iter.next();
    }
    if iter.next() != Some('"') {
        return None;
    }
    let mut out = String::new();
    let mut chars = iter;
    while let Some(c) = chars.next() {
        if c == '\\' {
            // Naive escape handling — `\"`, `\\`, `\n`, `\t`. Anything
            // else passes through. Sufficient for real ruvllm requests.
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => return None,
            }
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

/// JSON-quote a string. Handles the four escape characters that show
/// up in real prompts; falls back to a literal pass-through for the
/// rest. Same reasoning as the request parser above — no serde_json
/// dep for one bin.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn print_help() {
    println!(
        "{} {} — ruvllm host-side bridge (ADR-173 seam)\n\
\n\
USAGE:\n    ruvllm-bridge --workers <csv> --fingerprint <hex> [OPTIONS]\n\
\n\
INPUT:  JSONL on stdin, one request per line:\n    \
    {{\"text\":\"...\"}}\n    \
    {{\"text\":\"...\",\"request_id\":\"01HRZK...\"}}\n\
\n\
OUTPUT: JSONL on stdout, matched 1:1 with input order:\n    \
    {{\"dim\":384,\"latency_us\":8147,\"vector\":[...]}}\n    \
    {{\"error\":\"...\"}}\n\
\n\
OPTIONAL:\n    \
    --workers <csv>              REQUIRED. Cluster worker endpoints.\n    \
    --fingerprint <hex>          Reject workers reporting different fp.\n    \
    --allow-empty-fingerprint    Bypass the ADR-172 §2a empty-fp gate.\n    \
    --dim <N>                    Expected embedding dim (default 384).\n    \
    --quiet                      Suppress informational stderr.\n    \
    --tls-ca <path>              Server CA bundle (PEM).\n    \
    --tls-domain <name>          SNI / cert-SAN to assert.\n    \
    --tls-client-cert <path>     PEM client cert for mTLS (ADR-172 §1b).\n    \
    --tls-client-key <path>      PEM private key matching client cert.\n    \
    --help                       This message.\n    \
    --version                    Print version.\n",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
    );
}
