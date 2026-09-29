//! The edge subject (ADR-351 §5.2): the only user identifier edge access
//! tokens carry. One function, used by the edge authorization server when it
//! mints and by the §5.5 upstream-first-party verifier when it normalises, so
//! both paths yield the same actor for the same upstream user.
//!
//! Derivation (30 characters):
//! `"es1_" + base32lower(sha256("ruvector-edge/sub/v1|" + iss + "|" + sub))[0..26]`
//! where `iss`/`sub` are the upstream issuer and subject. The raw upstream
//! `sub` never leaves the AS store.

use sha2::{Digest, Sha256};

/// Prefix of every edge subject (version 1 of the derivation).
pub const EDGE_SUBJECT_PREFIX: &str = "es1_";
/// Domain-separation tag hashed in front of the inputs.
pub const EDGE_SUBJECT_DOMAIN: &str = "ruvector-edge/sub/v1|";
/// Number of base32 characters kept after the prefix.
pub const EDGE_SUBJECT_HASH_CHARS: usize = 26;
/// Total length of an edge subject.
pub const EDGE_SUBJECT_LEN: usize = EDGE_SUBJECT_PREFIX.len() + EDGE_SUBJECT_HASH_CHARS;

const ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";

/// RFC 4648 base32, lowercase, no padding.
fn base32_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(5) * 8);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &b in bytes {
        acc = (acc << 8) | u32::from(b);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            out.push(char::from(ALPHABET[((acc >> bits) & 31) as usize]));
        }
        acc &= (1 << bits) - 1;
    }
    if bits > 0 {
        out.push(char::from(ALPHABET[((acc << (5 - bits)) & 31) as usize]));
    }
    out
}

/// Derive the edge subject for an upstream `(iss, sub)` pair.
///
/// `upstream_iss` is the **upstream** issuer (`https://auth.cognitum.one`),
/// never the edge AS issuer. Deterministic and pure.
pub fn edge_subject(upstream_iss: &str, upstream_sub: &str) -> String {
    let mut h = Sha256::new();
    h.update(EDGE_SUBJECT_DOMAIN.as_bytes());
    h.update(upstream_iss.as_bytes());
    h.update(b"|");
    h.update(upstream_sub.as_bytes());
    let encoded = base32_lower(&h.finalize());
    format!(
        "{EDGE_SUBJECT_PREFIX}{}",
        &encoded[..EDGE_SUBJECT_HASH_CHARS]
    )
}

/// Whether `s` has the shape of an edge subject (prefix, length, alphabet).
pub fn is_edge_subject(s: &str) -> bool {
    s.len() == EDGE_SUBJECT_LEN
        && s.starts_with(EDGE_SUBJECT_PREFIX)
        && s[EDGE_SUBJECT_PREFIX.len()..]
            .bytes()
            .all(|b| ALPHABET.contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base32_matches_rfc4648_vectors() {
        // RFC 4648 §10, lowercased, padding stripped.
        let cases = [
            ("", ""),
            ("f", "my"),
            ("fo", "mzxq"),
            ("foo", "mzxw6"),
            ("foob", "mzxw6yq"),
            ("fooba", "mzxw6ytb"),
            ("foobar", "mzxw6ytboi"),
        ];
        for (input, want) in cases {
            assert_eq!(base32_lower(input.as_bytes()), want, "{input}");
        }
    }

    #[test]
    fn subject_shape_and_determinism() {
        let a = edge_subject("https://auth.cognitum.one", "user-1");
        assert_eq!(a.len(), EDGE_SUBJECT_LEN);
        assert_eq!(a.len(), 30);
        assert!(a.starts_with("es1_"));
        assert!(is_edge_subject(&a));
        assert_eq!(a, edge_subject("https://auth.cognitum.one", "user-1"));
        assert!(!a.contains("user-1"));
    }

    #[test]
    fn subject_depends_on_issuer_and_sub() {
        let base = edge_subject("https://auth.cognitum.one", "user-1");
        assert_ne!(base, edge_subject("https://other.example", "user-1"));
        assert_ne!(base, edge_subject("https://auth.cognitum.one", "user-2"));
    }

    #[test]
    fn rejects_malformed_subjects() {
        assert!(!is_edge_subject("es1_short"));
        assert!(!is_edge_subject("es2_abcdefghijklmnopqrstuvwxyz"));
        assert!(!is_edge_subject("es1_ABCDEFGHIJKLMNOPQRSTUVWXYZ"));
    }
}
