//! Canonical JSON: sorted keys, no floats, no insignificant whitespace.
//!
//! Every hash that appears in a build manifest, runtime contract, or
//! provenance record is a hash *of this encoding*. Two structurally equal
//! values must therefore produce byte-identical output on every platform and
//! in every process, which rules out three things `serde_json::to_string`
//! does not rule out on its own:
//!
//! - **Key order.** `serde_json`'s object map is a `BTreeMap` only while the
//!   `preserve_order` feature is off. Feature unification in a large workspace
//!   can turn it on from an unrelated crate, at which point key order silently
//!   becomes insertion order. This module sorts explicitly and is unaffected.
//! - **Floats.** No float has a single canonical decimal form, so a float in a
//!   hashed document is a reproducibility bug. Encoding rejects them outright
//!   rather than rounding.
//! - **Whitespace.** The compact form is the only form emitted.
//!
//! Keys sort by Rust's `str` ordering, which for UTF-8 is bytewise and
//! therefore also code-point order.

use crate::error::{ForgeError, Result};
use serde::Serialize;
use serde_json::Value;

/// Serialize `value` to canonical JSON.
///
/// # Errors
///
/// [`ForgeError::Manifest`] if the value fails to serialize, or if it contains
/// a floating-point number (see the module docs for why floats are refused).
pub fn to_canonical_json<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    let json = serde_json::to_value(value).map_err(|e| ForgeError::Manifest {
        detail: format!("value is not serializable: {e}"),
    })?;
    let mut out = String::new();
    write_canonical(&json, &mut out)?;
    Ok(out)
}

/// SHA-256, lowercase hex, of the canonical JSON encoding of `value`.
///
/// # Errors
///
/// As [`to_canonical_json`].
pub fn canonical_sha256_hex<T: Serialize + ?Sized>(value: &T) -> Result<String> {
    let canonical = to_canonical_json(value)?;
    Ok(hex_lower(&rvf_types::sha256(canonical.as_bytes())))
}

/// Lowercase hex encoding of a byte slice.
pub fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 0x0f) as usize] as char);
    }
    s
}

/// Decode lowercase or uppercase hex into bytes.
///
/// # Errors
///
/// [`ForgeError::Manifest`] if the input has odd length or a non-hex digit.
pub fn hex_decode(s: &str) -> Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        return Err(ForgeError::Manifest {
            detail: format!("hex string has odd length {}", s.len()),
        });
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let hi = hex_nibble(pair[0])?;
        let lo = hex_nibble(pair[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        other => Err(ForgeError::Manifest {
            detail: format!("invalid hex digit {:?}", other as char),
        }),
    }
}

fn write_canonical(value: &Value, out: &mut String) -> Result<()> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => {
            if n.is_f64() {
                return Err(ForgeError::Manifest {
                    detail: format!(
                        "floating-point value {n} has no canonical encoding; \
                         use an integer or a string"
                    ),
                });
            }
            out.push_str(&n.to_string());
        }
        Value::String(s) => write_json_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(key, out);
                out.push(':');
                write_canonical(&map[key], out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Write a JSON string literal, escaping per RFC 8259.
fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn object_keys_are_sorted_regardless_of_input_order() {
        let a = to_canonical_json(&json!({"zulu": 1, "alpha": 2, "mike": 3})).unwrap();
        let b = to_canonical_json(&json!({"alpha": 2, "mike": 3, "zulu": 1})).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, r#"{"alpha":2,"mike":3,"zulu":1}"#);
    }

    #[test]
    fn nested_objects_are_sorted_too() {
        let v = json!({"b": {"z": 1, "a": 2}, "a": [{"y": 1, "x": 2}]});
        assert_eq!(
            to_canonical_json(&v).unwrap(),
            r#"{"a":[{"x":2,"y":1}],"b":{"a":2,"z":1}}"#
        );
    }

    #[test]
    fn array_order_is_preserved() {
        let v = json!([3, 1, 2]);
        assert_eq!(to_canonical_json(&v).unwrap(), "[3,1,2]");
    }

    #[test]
    fn floats_are_rejected() {
        let err = to_canonical_json(&json!({"ratio": 0.5})).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Manifest);
        assert!(err.to_string().contains("floating-point"), "{err}");
    }

    #[test]
    fn nested_float_is_rejected() {
        let err = to_canonical_json(&json!({"a": {"b": [1, 2.5]}})).unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::Manifest);
    }

    #[test]
    fn strings_are_escaped() {
        let v = json!({"k": "a\"b\\c\nd\te\u{1}"});
        // The C0 control becomes a \u escape; the rest use their short forms.
        assert_eq!(
            to_canonical_json(&v).unwrap(),
            "{\"k\":\"a\\\"b\\\\c\\nd\\te\\u0001\"}"
        );
    }

    #[test]
    fn no_insignificant_whitespace() {
        let out = to_canonical_json(&json!({"a": [1, {"b": null}], "c": true})).unwrap();
        assert_eq!(out, r#"{"a":[1,{"b":null}],"c":true}"#);
    }

    #[test]
    fn whitespace_inside_string_values_is_preserved() {
        let out = to_canonical_json(&json!({"name": "Example Agent"})).unwrap();
        assert_eq!(out, r#"{"name":"Example Agent"}"#);
    }

    #[test]
    fn hex_round_trips() {
        let bytes = [0x00u8, 0x0f, 0xa5, 0xff];
        let s = hex_lower(&bytes);
        assert_eq!(s, "000fa5ff");
        assert_eq!(hex_decode(&s).unwrap(), bytes);
        assert_eq!(hex_decode("000FA5FF").unwrap(), bytes);
    }

    #[test]
    fn hex_decode_rejects_bad_input() {
        assert!(hex_decode("abc").is_err());
        assert!(hex_decode("zz").is_err());
    }

    #[test]
    fn canonical_sha256_is_order_independent() {
        let a = canonical_sha256_hex(&json!({"x": 1, "y": 2})).unwrap();
        let b = canonical_sha256_hex(&json!({"y": 2, "x": 1})).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }
}
