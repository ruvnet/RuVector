//! Minimal RFC 4648 §4 standard base64 (with padding), strict/canonical.
//!
//! Both [tlog-checkpoint] (root hash line) and [signed-note] (signature
//! payload) require "the standard Base 64 encoding specified in RFC 4648,
//! Section 4" — standard alphabet (`A-Za-z0-9+/`), `=` padding.
//!
//! Decoding is *canonical*: input length must be a multiple of 4, padding
//! may appear only at the end, and unused trailing bits must be zero.
//! This is stricter than Go's `base64.StdEncoding` (which tolerates
//! non-zero trailing bits) and removes a signature-malleability surface.
//!
//! [tlog-checkpoint]: https://github.com/C2SP/C2SP/blob/main/tlog-checkpoint.md
//! [signed-note]: https://github.com/C2SP/C2SP/blob/main/signed-note.md

use crate::Error;

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Encode `data` as standard base64 with `=` padding.
pub fn encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(chunk.get(1).copied().unwrap_or(0));
        let b2 = u32::from(chunk.get(2).copied().unwrap_or(0));
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn decode_sextet(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some(u32::from(c - b'A')),
        b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
        b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Decode canonical standard base64 with padding.
///
/// Rejects: lengths not a multiple of 4, padding anywhere but the final
/// one or two positions, characters outside the standard alphabet, and
/// non-zero unused trailing bits (non-canonical encodings).
pub fn decode(s: &str) -> Result<Vec<u8>, Error> {
    let bytes = s.as_bytes();
    if bytes.len() % 4 != 0 {
        return Err(Error::InvalidBase64);
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    let chunks = bytes.len() / 4;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let last = i + 1 == chunks;
        let pad = if chunk[3] == b'=' {
            if chunk[2] == b'=' { 2 } else { 1 }
        } else {
            0
        };
        if pad > 0 && !last {
            return Err(Error::InvalidBase64);
        }
        // '=' must not appear outside the padding positions checked above.
        if chunk[0] == b'=' || chunk[1] == b'=' || (pad < 2 && chunk[2] == b'=') {
            return Err(Error::InvalidBase64);
        }
        let v0 = decode_sextet(chunk[0]).ok_or(Error::InvalidBase64)?;
        let v1 = decode_sextet(chunk[1]).ok_or(Error::InvalidBase64)?;
        match pad {
            0 => {
                let v2 = decode_sextet(chunk[2]).ok_or(Error::InvalidBase64)?;
                let v3 = decode_sextet(chunk[3]).ok_or(Error::InvalidBase64)?;
                let n = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
                out.push((n >> 16) as u8);
                out.push((n >> 8) as u8);
                out.push(n as u8);
            }
            1 => {
                let v2 = decode_sextet(chunk[2]).ok_or(Error::InvalidBase64)?;
                if v2 & 0b11 != 0 {
                    return Err(Error::InvalidBase64); // non-canonical
                }
                let n = (v0 << 18) | (v1 << 12) | (v2 << 6);
                out.push((n >> 16) as u8);
                out.push((n >> 8) as u8);
            }
            _ => {
                if v1 & 0b1111 != 0 {
                    return Err(Error::InvalidBase64); // non-canonical
                }
                let n = (v0 << 18) | (v1 << 12);
                out.push((n >> 16) as u8);
            }
        }
    }
    Ok(out)
}

// Unit tests for this module live in tests/unit.rs (see the note on
// `[lib] test = false` in Cargo.toml).
