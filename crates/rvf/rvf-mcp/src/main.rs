//! RVF MCP server binary — stdio transport.
//!
//! Reads JSON-RPC requests from stdin, dispatches to RvfMcpServer,
//! writes JSON-RPC responses to stdout.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use rvf_mcp::{RvfMcpServer, ServerConfig};

fn main() {
    let data_dir = std::env::var("RVF_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".rvf-mcp")
        });

    let dimension: u16 = std::env::var("RVF_DIMENSION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(384);

    let config = ServerConfig {
        data_dir,
        dimension,
    };

    let mut server = match RvfMcpServer::new(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rvf-mcp: failed to initialize: {e}");
            std::process::exit(1);
        }
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match server.handle_jsonrpc(trimmed) {
            Ok(response) => {
                let _ = writeln!(out, "{response}");
                let _ = out.flush();
            }
            Err(e) => {
                let err = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": { "code": -32700, "message": e }
                });
                let _ = writeln!(out, "{err}");
                let _ = out.flush();
            }
        }
    }
}
