//! `rvf verify-witness` -- Verify the witness hash chain of a store.
//!
//! Scans the RVF file for WITNESS_SEG segments in file order and verifies
//! them as ONE SHAKE-256 hash chain (`rvf_crypto::verify_witness_chain_from`):
//! each entry must carry the hash of the entry before it, so altering,
//! removing or reordering an entry that is followed by another one is
//! detected. Every segment's content hash is checked too, which also covers
//! the newest entry.
//!
//! Outcomes and exit codes:
//! * all chains verify                       -> `valid`, exit 0
//! * any failure (broken link, bad segment
//!   hash, unrecognized payload)             -> `invalid`, exit 1
//! * no failure, but legacy records present  -> `legacy_unverifiable`, exit 2
//!   (files written by rvf-runtime <= 0.4.0 used a record that carries no
//!   verifiable hash; it is reported as such and never counted as valid).
//!   `--allow-legacy` turns exit 2 into exit 0.
//!
//! Limits: the chain is unkeyed and unsigned. It detects accidental damage and
//! edits that do not recompute the following hashes. It does NOT detect
//! (a) removal or replacement of the newest entries, (b) stripping every
//! witness segment (use `--require-witness` to fail on an empty trail, which
//! still cannot prove one ever existed), or (c) an attacker who recomputes the
//! tail. Attestation segments are accepted only if they fully verify as an
//! attestation witness; they form independent chains and do not move the
//! runtime chain head.

use clap::Args;
use std::io::{BufReader, Read};

use rvf_crypto::{
    classify_witness_payload, is_valid_attestation_witness, verify_witness_chain_from,
    WitnessEntry, WitnessPayloadKind,
};
use rvf_types::{SegmentType, SEGMENT_HEADER_SIZE, SEGMENT_MAGIC};

#[derive(Args)]
pub struct VerifyWitnessArgs {
    /// Path to the RVF store
    pub file: String,
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
    /// Exit 0 (instead of 2) when the only problem is legacy, non-verifiable
    /// witness records from rvf-runtime <= 0.4.0
    #[arg(long)]
    pub allow_legacy: bool,
    /// Exit 1 when the file has no witness segments at all. Without this a
    /// store whose audit trail was stripped verifies exactly like a store that
    /// never had one (an unkeyed hash chain cannot tell them apart).
    #[arg(long)]
    pub require_witness: bool,
}

/// One WITNESS_SEG found in the file.
struct Found {
    offset: usize,
    payload: Vec<u8>,
    /// Segment content hash matches the payload.
    hash_ok: bool,
}

/// Extract all WITNESS_SEG segments from the raw file bytes.
fn extract_witness_segments(raw: &[u8]) -> Vec<Found> {
    let magic_bytes = SEGMENT_MAGIC.to_le_bytes();
    let mut results = Vec::new();
    let mut i = 0usize;

    while i + SEGMENT_HEADER_SIZE <= raw.len() {
        if raw[i..i + 4] == magic_bytes {
            let seg_type = raw[i + 5];
            let payload_len = u64::from_le_bytes(raw[i + 0x10..i + 0x18].try_into().unwrap());
            let payload_start = i + SEGMENT_HEADER_SIZE;
            let payload_end = usize::try_from(payload_len)
                .ok()
                .and_then(|l| payload_start.checked_add(l));

            if let Some(payload_end) = payload_end.filter(|&e| e <= raw.len()) {
                // A torn header whose declared length is later covered by newly
                // appended segments would swallow them. If a valid segment header
                // sits at an aligned offset inside the declared payload, this
                // header is a stale torn one: skip it, do not follow its length.
                let swallows = seg_type == SegmentType::Witness as u8
                    && (payload_start..payload_end).step_by(64).any(|p| {
                        p > i
                            && p + SEGMENT_HEADER_SIZE <= raw.len()
                            && raw[p..p + 4] == magic_bytes
                            && rvf_wire::reader::read_segment_header(&raw[p..]).is_ok()
                    });
                if swallows {
                    i += SEGMENT_HEADER_SIZE;
                    continue;
                }
                if seg_type == SegmentType::Witness as u8 {
                    let payload = raw[payload_start..payload_end].to_vec();
                    let hash_ok = rvf_wire::reader::read_segment_header(&raw[i..])
                        .map(|h| {
                            // Legacy-dialect segments (checksum_algo 0) and unset
                            // hashes are not checked, as in the runtime reader.
                            h.checksum_algo == 0
                                || h.content_hash == [0u8; 16]
                                || rvf_wire::reader::validate_segment(&h, &payload).is_ok()
                        })
                        .unwrap_or(false);
                    results.push(Found {
                        offset: i,
                        payload,
                        hash_ok,
                    });
                }
                i = payload_end.max(i + 1);
                continue;
            }
        }
        i += 1;
    }
    results
}

/// Format a nanosecond timestamp as a human-readable UTC string.
fn format_timestamp_ns(ns: u64) -> String {
    if ns == 0 {
        return "0 (genesis)".to_string();
    }
    let secs = ns / 1_000_000_000;
    let sub_ns = ns % 1_000_000_000;
    format!("{secs}.{sub_ns:09}s (unix epoch)")
}

/// Map witness_type byte to a name.
fn witness_type_name(wt: u8) -> &'static str {
    match wt {
        0x01 => "PROVENANCE",
        0x02 => "COMPUTATION",
        0x03 => "PLATFORM_ATTESTATION",
        0x04 => "KEY_BINDING",
        0x05 => "DATA_PROVENANCE",
        _ => "UNKNOWN",
    }
}

struct SegResult {
    kind: &'static str,
    entries: usize,
    valid: bool,
    error: Option<String>,
}

pub fn run(args: VerifyWitnessArgs) -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open(&args.file)?;
    let mut reader = BufReader::new(file);
    let mut raw_bytes = Vec::new();
    reader.read_to_end(&mut raw_bytes)?;

    let segments = extract_witness_segments(&raw_bytes);

    if segments.is_empty() {
        if args.json {
            crate::output::print_json(&serde_json::json!({
                "status": "no_witnesses",
                "witness_segments": 0,
                "total_entries": 0,
            }));
        } else {
            println!("No witness segments found in file.");
            println!(
                "  WARNING: nothing was verified. A stripped audit trail looks the same; \
                 use --require-witness to fail on this."
            );
        }
        if args.require_witness {
            return Err("no witness segments found (--require-witness)".into());
        }
        return Ok(());
    }

    // Verify all chain segments as one chain, in file order.
    let mut head = [0u8; 32]; // hash the next entry must carry (genesis = 0)
    let mut broken = false; // after a failure, re-anchor on each segment itself
    let mut results: Vec<SegResult> = Vec::new();
    let mut all_entries: Vec<WitnessEntry> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut legacy = 0usize;
    let mut attestation = 0usize;

    for (idx, seg) in segments.iter().enumerate() {
        let mut errs: Vec<String> = Vec::new();
        if !seg.hash_ok {
            errs.push("segment content hash mismatch (payload altered or damaged)".into());
        }
        let p = &seg.payload;
        // Classification never guesses from the length or shape alone:
        // every accepted kind must fully verify. Order: (1) hash chain,
        // (2) a complete attestation payload (`verify-attestation` format;
        // its own chain, does not move the runtime chain head),
        // (3) the legacy 0.4.0 record, reported but never valid, (4) failure.
        let mut chain_err: Option<String> = None;
        let mut classified: Option<(&'static str, usize)> = None;
        if p.is_empty() || p.len().is_multiple_of(rvf_crypto::WITNESS_ENTRY_SIZE) {
            let start = if broken && p.len() >= 32 {
                // Re-anchor after an earlier failure so one break is reported
                // once instead of poisoning every later link.
                p[..32].try_into().unwrap()
            } else {
                head
            };
            match verify_witness_chain_from(p, &start) {
                Ok((entries, next)) => {
                    head = next;
                    let n = entries.len();
                    all_entries.extend(entries);
                    classified = Some(("chain", n));
                }
                Err(e) => {
                    chain_err = Some(format!(
                        "{e}: hash chain broken (entry altered, removed, or out of order)"
                    ))
                }
            }
        }
        if classified.is_none() && is_valid_attestation_witness(p) {
            attestation += 1;
            classified = Some(("attestation", 0));
        }
        if classified.is_none() && classify_witness_payload(p) == WitnessPayloadKind::Legacy {
            // Written by rvf-runtime <= 0.4.0: carries no verifiable hash. The
            // runtime restarts its chain after such a record. Never "valid".
            legacy += 1;
            head = [0u8; 32];
            broken = false;
            classified = Some(("legacy", 1));
        }
        let (kind, entries) = match classified {
            Some(c) => c,
            None => {
                errs.push(chain_err.unwrap_or_else(|| {
                    "payload is not a valid chain entry, attestation, or legacy record".into()
                }));
                ("unrecognized", 0)
            }
        };
        let valid = errs.is_empty() && kind == "chain";
        if !errs.is_empty() {
            broken = true;
            for e in &errs {
                failures.push(format!("Segment #{idx} at offset 0x{:X}: {e}", seg.offset));
            }
        }
        results.push(SegResult {
            kind,
            entries,
            valid,
            error: (!errs.is_empty()).then(|| errs.join("; ")),
        });
    }

    let verified_segments = results.iter().filter(|r| r.valid).count();
    let total_entries: usize = results.iter().map(|r| r.entries).sum();
    let status = if !failures.is_empty() {
        "invalid"
    } else if legacy > 0 {
        "legacy_unverifiable"
    } else {
        "valid"
    };
    let first_ts = all_entries.first().map(|e| e.timestamp_ns).unwrap_or(0);
    let last_ts = all_entries.last().map(|e| e.timestamp_ns).unwrap_or(0);

    if args.json {
        let segs: Vec<serde_json::Value> = results
            .iter()
            .zip(&segments)
            .enumerate()
            .map(|(i, (r, s))| {
                serde_json::json!({
                    "segment_index": i,
                    "segment_offset": format!("0x{:X}", s.offset),
                    "kind": r.kind,
                    "entry_count": r.entries,
                    "chain_valid": r.valid,
                    "error": r.error,
                })
            })
            .collect();
        crate::output::print_json(&serde_json::json!({
            "status": status,
            "witness_segments": segments.len(),
            "valid_chains": verified_segments,
            "legacy_segments": legacy,
            "attestation_segments": attestation,
            "total_entries": total_entries,
            "first_timestamp_ns": first_ts,
            "last_timestamp_ns": last_ts,
            "chain_breaks": failures,
            "segments": segs,
        }));
    } else {
        println!("Witness chain verification (cryptographic):");
        println!();
        crate::output::print_kv("Witness segments:", &segments.len().to_string());
        crate::output::print_kv(
            "Valid chain segments:",
            &format!("{verified_segments}/{}", segments.len()),
        );
        if legacy > 0 {
            crate::output::print_kv("Legacy (not verifiable):", &legacy.to_string());
        }
        if attestation > 0 {
            crate::output::print_kv(
                "Attestation (see verify-attestation):",
                &attestation.to_string(),
            );
        }
        crate::output::print_kv("Total entries:", &total_entries.to_string());

        if !all_entries.is_empty() {
            println!();
            crate::output::print_kv("First timestamp:", &format_timestamp_ns(first_ts));
            crate::output::print_kv("Last timestamp:", &format_timestamp_ns(last_ts));
            let mut type_counts = std::collections::HashMap::new();
            for entry in &all_entries {
                *type_counts.entry(entry.witness_type).or_insert(0u64) += 1;
            }
            println!();
            println!("  Entry types:");
            let mut types: Vec<_> = type_counts.iter().collect();
            types.sort_by_key(|(k, _)| **k);
            for (wt, count) in types {
                println!(
                    "    0x{:02X} ({:20}): {}",
                    wt,
                    witness_type_name(*wt),
                    count
                );
            }
        }

        println!();
        match status {
            "valid" => println!("  All witness hash chains verified successfully."),
            "legacy_unverifiable" => println!(
                "  LEGACY: {legacy} witness record(s) were written by rvf-runtime <= 0.4.0 and \
                 carry no verifiable hash chain. They are NOT verified. Re-run with \
                 --allow-legacy to accept this."
            ),
            _ => {
                println!(
                    "  FAILED: {} problem(s) found in the witness chain:",
                    failures.len()
                );
                for f in &failures {
                    println!("    - {f}");
                }
            }
        }
    }

    match status {
        "invalid" => Err(format!(
            "witness verification failed: {} problem(s) (see output)",
            failures.len()
        )
        .into()),
        "legacy_unverifiable" if !args.allow_legacy => std::process::exit(2),
        _ => Ok(()),
    }
}
