//! Generated tonic stubs for `proto/embedding.proto`.
//!
//! The proto package is `ruvector.hailo.v1`, so tonic-build emits a
//! `ruvector.hailo.v1` Rust module containing the message structs +
//! the `embedding_client` / `embedding_server` modules.

#![allow(clippy::pedantic)]
#![allow(missing_docs)]

tonic::include_proto!("ruvector.hailo.v1");

/// Canonical metadata header for cross-call request correlation.
/// W3C-style — workers and any intermediary tracing infra can grep
/// `x-request-id` without knowing the proto schema.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Crockford's base32 alphabet for ULID encoding (no I, L, O, U).
const CROCKFORD: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// 26-char ULID correlation ID — sortable, standard, 80 bits entropy.
///
/// Iter-109 (deferred-backlog item): switched from the legacy
/// `<16-hex-ms><8-hex-rand>` 24-char format to a spec-compliant ULID
/// (<https://github.com/ulid/spec>) so log-tooling that natively groks
/// ULIDs (Datadog, Honeycomb, Vector, etc.) can decode the timestamp
/// without a custom parser. Layout:
///
///   chars  0..10 : 48-bit unix-ms big-endian, base32 (Crockford)
///   chars 10..26 : 80-bit randomness, base32 (Crockford)
///
/// Lexicographic sort still matches chronological order — that
/// invariant is part of the ULID spec by construction. Random suffix
/// has 80 bits of entropy from xorshift64* (two pulls); collisions
/// within a single ms are astronomically unlikely.
///
/// Public so callers (web handlers, batch ingest CLIs, custom
/// transports) can generate matching IDs without going through
/// `GrpcTransport`. Not crypto-grade.
pub fn random_request_id() -> String {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0) };
    }
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    // Mask off bits above 48 — ULID timestamp is 48 bits big-endian.
    // `now_ms` won't overflow 48 bits until year 10889.
    let ts48 = now_ms & 0x0000_FFFF_FFFF_FFFF;

    // Two xorshift64* pulls give us 128 random bits; we take 80.
    let mut rng = STATE.with(|s| {
        let mut x = s.get();
        if x == 0 {
            x = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1);
            if x == 0 {
                x = 0x9E3779B97F4A7C15;
            }
        }
        x
    });
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let r1 = next();
    let r2 = next();
    STATE.with(|s| s.set(rng));

    // Compose: 48 ts bits + 80 rand bits = 128 bits, then base32-encode
    // 5 bits at a time (most-significant first), 26 chars total.
    // We pack into a 128-bit big integer split as (high u64, low u64):
    //   high = ts48 << 16 | (top 16 bits of r1)
    //   low  = (low 48 bits of r1) << 16 | (top 16 bits of r2)
    let high: u64 = (ts48 << 16) | (r1 >> 48);
    let low: u64 = (r1 << 16) | (r2 >> 48);

    let mut out = [0u8; 26];
    // Encode 26 characters of 5 bits each, MSB-first across the 130-bit
    // logical integer (we only have 128 bits; the top 2 bits of char 0
    // are zero, which is fine — ULID's first char is always 0..7).
    for (i, slot) in out.iter_mut().enumerate() {
        // bit position counted from the *most* significant bit of the
        // 130-bit value, which is `(25 - i) * 5` from the LSB.
        let shift = (25 - i) * 5;
        let bits: u8 = if shift >= 64 {
            // bits live entirely in `high` (offset shift-64 from its LSB)
            ((high >> (shift - 64)) & 0x1F) as u8
        } else if shift + 5 <= 64 {
            // bits live entirely in `low`
            ((low >> shift) & 0x1F) as u8
        } else {
            // straddle: take low part of `high` + high part of `low`
            let lo_part = low >> shift;
            let hi_part = high << (64 - shift);
            ((hi_part | lo_part) & 0x1F) as u8
        };
        *slot = CROCKFORD[bits as usize];
    }
    // SAFETY: every byte we wrote is from CROCKFORD which is ASCII.
    String::from_utf8(out.to_vec()).expect("CROCKFORD is ASCII")
}

/// Inject `request_id` as the `x-request-id` gRPC metadata header on
/// an outgoing tonic request. Best-effort: invalid characters silently
/// fall through (the proto field carries the same value as fallback).
pub fn inject_request_id<T>(req: &mut tonic::Request<T>, request_id: &str) {
    if let Ok(v) = request_id.parse::<tonic::metadata::MetadataValue<_>>() {
        req.metadata_mut().insert(REQUEST_ID_HEADER, v);
    }
}

/// Pull `request_id` out of an incoming tonic request's metadata,
/// falling back to a `proto_field` if the header is absent. Returns
/// owned `String` to decouple from the request lifetime.
///
/// **Security (ADR-172 §4 mitigation):** the returned id is run through
/// [`sanitize_request_id`] before reaching tracing logs — control chars
/// stripped, length capped at 64. A caller-supplied id full of newlines
/// or ANSI escapes can't log-forge multi-line entries; an oversized id
/// can't inflate log line size for resource burn.
pub fn extract_request_id<T>(req: &tonic::Request<T>, proto_field: &str) -> String {
    let raw = if let Some(v) = req.metadata().get(REQUEST_ID_HEADER) {
        if let Ok(s) = v.to_str() {
            if !s.is_empty() {
                s
            } else {
                proto_field
            }
        } else {
            proto_field
        }
    } else {
        proto_field
    };
    sanitize_request_id(raw)
}

/// Strip control characters (anything < 0x20 except space, plus 0x7F)
/// and cap at 64 chars. Used by [`extract_request_id`] to neutralise
/// log-forging attempts and length-amplification (ADR-172 §4a/4b).
///
/// Returns an empty `String` for empty input — callers that want a
/// random fallback should chain `if id.is_empty() { random_request_id() }`.
pub fn sanitize_request_id(raw: &str) -> String {
    const MAX_LEN: usize = 64;
    let mut out = String::with_capacity(raw.len().min(MAX_LEN));
    let mut byte_count = 0usize;
    for c in raw.chars() {
        // Strip C0 controls (0x00..0x1F) except space, and DEL (0x7F).
        if c == ' ' || (c >= '\u{0020}' && c != '\u{007F}' && !c.is_control()) {
            // Track *byte* length so multi-byte UTF-8 doesn't blow past
            // the cap. Stop at MAX_LEN bytes — never push past.
            let cl = c.len_utf8();
            if byte_count + cl > MAX_LEN {
                break;
            }
            out.push(c);
            byte_count += cl;
        }
        // else: silently drop. We don't log-forge ourselves to warn.
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_message_types_are_constructible() {
        let req = EmbedRequest {
            text: "hello world".into(),
            max_seq: 128,
            request_id: "abc123".into(),
        };
        assert_eq!(req.text, "hello world");
        assert_eq!(req.max_seq, 128);
        assert_eq!(req.request_id, "abc123");

        let resp = EmbedResponse {
            vector: vec![0.1, 0.2, 0.3],
            dim: 3,
            latency_us: 1234,
        };
        assert_eq!(resp.vector.len(), 3);
        assert_eq!(resp.dim, 3);

        let health = HealthResponse {
            version: "ruvector-hailo 0.1.0".into(),
            device_id: "0001:01:00.0".into(),
            model_fingerprint: "sha256:abc".into(),
            ready: true,
            npu_temp_ts0_celsius: 0.0,
            npu_temp_ts1_celsius: 0.0,
        };
        assert!(health.ready);
        assert_eq!(health.device_id, "0001:01:00.0");
    }

    #[test]
    fn generated_messages_roundtrip_protobuf() {
        use prost::Message;

        let req = EmbedRequest {
            text: "ruvector".into(),
            max_seq: 64,
            request_id: "deadbeef".into(),
        };
        let buf = req.encode_to_vec();
        let decoded = EmbedRequest::decode(&buf[..]).expect("protobuf roundtrip");
        assert_eq!(decoded, req);
    }

    #[test]
    fn sanitize_request_id_strips_control_chars() {
        // Control bytes — newline, tab, ESC, BEL, NUL — all must go.
        // The non-control chars after ESC (`[31m`) are normal ASCII and
        // survive intact: that's the right behaviour, since stripping
        // control chars (not "ANSI sequences") is what neutralises log
        // forging — the parser logger no longer sees the escape byte.
        let raw = "ok-id\n\twith\x1b[31mansi\x07bell\0nul";
        let s = sanitize_request_id(raw);
        assert!(!s.contains('\n'), "newline must be stripped");
        assert!(!s.contains('\t'), "tab must be stripped");
        assert!(!s.contains('\x1b'), "ESC must be stripped");
        assert!(!s.contains('\x07'), "BEL must be stripped");
        assert!(!s.contains('\0'), "NUL must be stripped");
        // The surviving chars must include the non-control text.
        assert!(s.contains("ok-id"));
        assert!(s.contains("nul"));
    }

    #[test]
    fn sanitize_request_id_caps_length_at_64_bytes() {
        // 200-char id capped to 64.
        let raw: String = "a".repeat(200);
        let s = sanitize_request_id(&raw);
        assert_eq!(s.len(), 64);
        assert!(s.chars().all(|c| c == 'a'));
    }

    #[test]
    fn sanitize_request_id_handles_multibyte_utf8_at_boundary() {
        // Cap at byte length, not char length — never split a UTF-8 codepoint.
        // 'é' is 2 bytes; build a string just past 64 bytes.
        let raw: String = "é".repeat(40); // 80 bytes, 40 chars
        let s = sanitize_request_id(&raw);
        assert!(s.len() <= 64);
        // Whatever made it in must still be valid UTF-8 (Rust enforces),
        // and must be a whole number of 'é' chars.
        assert!(s.chars().all(|c| c == 'é'));
    }

    #[test]
    fn sanitize_request_id_preserves_normal_id() {
        let s = sanitize_request_id("0000019de68b5707983b8745");
        assert_eq!(s, "0000019de68b5707983b8745");
    }

    #[test]
    fn extract_request_id_sanitises_metadata_value() {
        // Even if the metadata header carries something hostile-shaped,
        // the value reaching tracing logs is sanitised.
        let mut req = tonic::Request::new(EmbedRequest::default());
        // tonic's MetadataValue parsing rejects control chars at parse
        // time; verify the inject path stays clean. For the extract
        // sanitiser test, exercise via the proto-field fallback.
        inject_request_id(&mut req, "log\nforging\tid"); // bypassed by parse
        let extracted = extract_request_id(&req, "log\nforging-fallback\tid");
        // Either branch (metadata or fallback), output must be stripped.
        assert!(!extracted.contains('\n'));
        assert!(!extracted.contains('\t'));
    }

    #[test]
    fn inject_and_extract_request_id_roundtrips_via_metadata() {
        let mut req = tonic::Request::new(EmbedRequest::default());
        inject_request_id(&mut req, "abc-123");
        let extracted = extract_request_id(&req, "");
        assert_eq!(extracted, "abc-123");
    }

    #[test]
    fn extract_request_id_falls_back_to_proto_field_when_header_absent() {
        let req = tonic::Request::new(EmbedRequest::default());
        let extracted = extract_request_id(&req, "from-proto");
        assert_eq!(extracted, "from-proto");
    }

    #[test]
    fn extract_request_id_prefers_metadata_over_proto_field() {
        let mut req = tonic::Request::new(EmbedRequest::default());
        inject_request_id(&mut req, "from-meta");
        let extracted = extract_request_id(&req, "from-proto");
        assert_eq!(extracted, "from-meta", "metadata wins when both present");
    }

    #[test]
    fn inject_request_id_with_invalid_chars_is_silently_dropped() {
        // Newlines aren't valid in HTTP header values; injection is a
        // no-op and extract falls back to the proto field.
        let mut req = tonic::Request::new(EmbedRequest::default());
        inject_request_id(&mut req, "bad\nid");
        let extracted = extract_request_id(&req, "fallback");
        assert_eq!(
            extracted, "fallback",
            "invalid header value should be silently ignored"
        );
    }

    #[test]
    fn random_request_id_has_26_crockford_chars() {
        // Iter-109: ULID format is 26 chars from Crockford's base32
        // (no I, L, O, U).
        let id = random_request_id();
        assert_eq!(id.len(), 26, "expected 26-char ULID, got {:?}", id);
        for c in id.chars() {
            assert!(
                "0123456789ABCDEFGHJKMNPQRSTVWXYZ".contains(c),
                "char {:?} not in Crockford base32 alphabet (id={:?})",
                c,
                id
            );
        }
    }

    #[test]
    fn random_request_id_two_consecutive_ids_sort_chronologically() {
        // ULID guarantee: lexicographic order matches creation order.
        let id1 = random_request_id();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let id2 = random_request_id();
        assert!(
            id1 < id2,
            "expected id1 < id2, got id1={:?} id2={:?}",
            id1,
            id2
        );
        assert_eq!(id1.len(), id2.len(), "format stability");
    }

    #[test]
    fn random_request_id_uniqueness_within_same_ms() {
        let mut ids = std::collections::HashSet::new();
        for _ in 0..1000 {
            ids.insert(random_request_id());
        }
        // 80-bit randomness: collisions in 1000 same-ms calls are
        // astronomically unlikely. If this ever fails, the entropy
        // source has regressed.
        assert_eq!(ids.len(), 1000, "duplicate ULIDs in 1000 rapid calls");
    }

    #[test]
    fn random_request_id_prefix_decodes_to_recent_ms() {
        // ULID timestamp is the first 10 chars of Crockford base32 =
        // 48 bits = unix ms. Decode and check it's within ±5s of now.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let id = random_request_id();
        let prefix = &id[..10];
        let mut ts: u64 = 0;
        for c in prefix.chars() {
            // Crockford alphabet → 0..32; only the digits we encode are
            // present (we don't decode I/L/O/U aliases since we never
            // emitted them).
            let v = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
                .find(c)
                .expect("char must be from Crockford alphabet");
            ts = (ts << 5) | (v as u64);
        }
        let delta = ts.abs_diff(now_ms);
        assert!(
            delta < 5_000,
            "prefix ms {} differs from now {} by {}ms",
            ts,
            now_ms,
            delta
        );
    }

    #[test]
    fn generated_client_and_server_modules_exist() {
        // Compile-time checks — if these fail, tonic-build didn't emit the
        // expected modules. Just touching the type names is enough.
        fn _t<T>() {}
        _t::<embedding_client::EmbeddingClient<tonic::transport::Channel>>();
        // Server side is a generic over an `Embedding` trait impl; just
        // referencing the trait keeps the test trivially compile-only.
        fn _s<T: embedding_server::Embedding>() {}
    }
}
