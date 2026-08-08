//! Cross-implementation check: does the Rust crate accept an `.rvf` the
//! TypeScript CLI wrote?
//!
//! `npm/packages/rvforge` writes RVF containers; `rvf-forge-core` reads them.
//! Both implement the same wire format, and they were written independently —
//! so "both pass their own tests" says nothing about whether they interoperate.
//! This binary closes that gap the same way `rvforge-registry-check` does for
//! the registry: it takes a file the CLI produced and runs the crate's own
//! public API over it, with no fixtures and no shared code.
//!
//! Usage: `rvforge-rvf-check <agent.rvf> [--public-key <hex>] [--allow-unsigned]`
//!
//! With `--public-key`, segment signatures are checked against that key rather
//! than skipped, so the run proves the CLI's Ed25519 signatures verify under
//! this crate's verifier — not merely that they are well-formed.
//!
//! Exits 0 when inspection and verification both succeed, 1 otherwise.

use rvf_forge_core::{inspect_bytes, verify_bytes, VerifyOptions};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => {
            println!("\nRVF PARITY OK — rvf-forge-core accepts the container the CLI wrote");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("\nRVF PARITY FAILED — {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let mut path: Option<&str> = None;
    let mut public_key: Option<[u8; 32]> = None;
    let mut allow_unsigned = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--public-key" => {
                let hex = args.get(i + 1).ok_or("--public-key needs a hex value")?;
                public_key = Some(parse_hex_key(hex)?);
                i += 2;
            }
            "--allow-unsigned" => {
                allow_unsigned = true;
                i += 1;
            }
            other if other.starts_with('-') => return Err(format!("unknown flag {other}")),
            other => {
                path = Some(other);
                i += 1;
            }
        }
    }

    let path = path.ok_or("usage: rvforge-rvf-check <agent.rvf> [--public-key <hex>]")?;
    let data = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;

    let inspection = inspect_bytes(&data).map_err(|e| format!("inspection refused it: {e}"))?;
    println!("inspected {path}");
    println!("  identity     {}", inspection.rvf_identity);
    println!("  bytes        {}", inspection.byte_length);
    println!("  segments     {}", inspection.segments.len());
    for segment in &inspection.segments {
        println!(
            "    {:<10} id={} payload={} signed={} checksum_algo={}",
            segment.segment_type_name,
            segment.segment_id,
            segment.payload_length,
            segment.signed,
            segment.checksum_algo,
        );
    }
    println!(
        "  capabilities {}",
        if inspection.declared_capabilities.is_empty() {
            "none declared (default-deny)".to_string()
        } else {
            inspection
                .declared_capabilities
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        }
    );

    let options = VerifyOptions {
        trusted_keys: public_key.into_iter().collect(),
        allow_unsigned_executable: allow_unsigned,
        expected_identity: Some(inspection.rvf_identity.clone()),
    };
    let report =
        verify_bytes(&data, &options).map_err(|e| format!("verification refused it: {e}"))?;

    println!("\nverified {path}");
    for record in &report.records {
        println!(
            "  {:?} {:?}: {}",
            record.check, record.outcome, record.detail
        );
    }
    if !report.ok {
        return Err(format!(
            "{} check(s) failed: {}",
            report.failures().len(),
            report
                .failures()
                .iter()
                .map(|r| r.detail.clone())
                .collect::<Vec<_>>()
                .join("; ")
        ));
    }

    // A skipped signature check is not a passed one. When the caller supplied a
    // key, at least one signature must actually have been verified against it,
    // or the run proves nothing about the CLI's signing.
    if public_key.is_some() {
        let verified = report
            .records
            .iter()
            .filter(|r| {
                r.check == rvf_forge_core::CheckKind::Signature
                    && r.outcome == rvf_forge_core::Outcome::Pass
            })
            .count();
        if verified == 0 {
            return Err(
                "a public key was supplied but no segment signature verified against it".into(),
            );
        }
        println!("\n  {verified} segment signature(s) verified against the supplied key");
    }

    Ok(())
}

fn parse_hex_key(hex: &str) -> Result<[u8; 32], String> {
    let hex = hex.trim();
    if hex.len() != 64 {
        return Err(format!(
            "a public key is 32 bytes / 64 hex characters, got {}",
            hex.len()
        ));
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|_| "public key is not valid hex".to_string())?;
    }
    Ok(out)
}
