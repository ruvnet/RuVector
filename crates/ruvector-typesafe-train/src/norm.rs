//! Text normalization + hashing shared with the JS bench (ADR-008 §2).
//!
//! `norm`: Unicode NFKC → lowercase → every run of code points that are not
//! `\p{L}` or `\p{N}` becomes one space → trim. This deliberately uses the
//! regex crate's general-category classes rather than `char::is_alphanumeric`
//! (which is `Alphabetic || Numeric` and so also keeps combining marks with the
//! Other_Alphabetic property), so it matches JS `/[^\p{L}\p{N}]+/gu` exactly.

use std::sync::OnceLock;

use regex::Regex;
use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

fn non_alnum() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^\p{L}\p{N}]+").expect("static regex"))
}

/// Normalize a text for leakage hashing.
pub fn norm(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    let lower = nfkc.to_lowercase();
    non_alnum().replace_all(&lower, " ").trim().to_string()
}

/// Lower-hex sha256 of arbitrary bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex(&h.finalize())
}

/// `sha256(norm(text))`, lower hex — the unit of both leakage assertions.
pub fn sha256_norm(text: &str) -> String {
    sha256_hex(norm(text).as_bytes())
}

pub fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inline golden vectors (the shared file `bench/test/norm-golden.json`
    /// is additionally checked by `tests/norm_golden.rs` when present).
    const GOLDEN: &[(&str, &str)] = &[
        ("Hello, World!", "hello world"),
        ("  leading and trailing  ", "leading and trailing"),
        ("can't", "can t"),
        ("tab\tseparated\nlines", "tab separated lines"),
        ("ＡＢＣ１２３", "abc123"), // full-width → NFKC ASCII
        ("Café", "café"),           // precomposed é kept (it is \p{L})
        ("Cafe\u{301}", "café"),    // NFKC composes e + U+0301
        ("ﬁle", "file"),            // ligature decomposes under NFKC
        ("price: $5.00!!", "price 5 00"),
        ("emoji 😀 here", "emoji here"),
        ("東京 タワー", "東京 タワー"), // CJK letters kept
        ("²³", "23"),                   // superscripts → digits (NFKC)
        ("Ⅻ", "xii"),                   // roman numeral → letters under NFKC
        ("---", ""),
        ("", ""),
        ("ÉCOLE", "école"),
        ("a_b-c.d", "a b c d"), // underscore is not \p{L}/\p{N}
        ("ΟΔΟΣ", "οδος"),       // final-sigma: Rust lowercases Σ→ς at word end
    ];

    #[test]
    fn golden_vectors() {
        for (input, want) in GOLDEN {
            assert_eq!(norm(input), *want, "norm({input:?})");
        }
    }

    #[test]
    fn hash_is_of_normalized_text() {
        assert_eq!(sha256_norm("Hello,   WORLD"), sha256_hex(b"hello world"));
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
