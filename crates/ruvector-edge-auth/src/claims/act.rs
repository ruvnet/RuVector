//! RFC 8693 §4.1 `act` claim on edge tokens minted by the exchange grant
//! (ADR-351 §5.2, §5.6, §16.3): `{"sub": "<adapter client_id>"}`.
//!
//! The edge AS never re-exchanges a token that already carries `act` (it
//! refuses even `act: null`), so a well-formed exchanged token has exactly
//! one level. The verifier mirrors that: an `act` that is not an object
//! whose only member is a bounded `sub` equal to `client_id` is refused.

use crate::error::AuthError;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

/// Serde helper: keep a present `null` as `Some(Value::Null)` so it is
/// refused instead of reading as "absent" (a missing claim stays `None`
/// through `#[serde(default)]`).
pub(crate) fn present<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(d).map(Some)
}

/// Validate a raw `act` claim for `kind` and return `act.sub`.
///
/// Contract: absent => `Ok(None)`; on an upstream first-party token any
/// `act` is refused; otherwise it must be a JSON object with exactly one
/// member, `sub`, a non-empty string of at most `max_len` bytes without
/// control characters, equal to `client_id` (the exchange grant mints
/// `client_id = act.sub`). Anything else (`null`, arrays, nested `act`,
/// extra members) => [`AuthError::InvalidClaim`]`("act")`.
pub(crate) fn validate_act(
    raw: Option<Value>,
    upstream: bool,
    client_id: &str,
    max_len: usize,
) -> Result<Option<String>, AuthError> {
    let Some(v) = raw else {
        return Ok(None);
    };
    let bad = AuthError::InvalidClaim("act");
    if upstream {
        return Err(bad);
    }
    let Value::Object(map) = v else {
        return Err(bad);
    };
    if map.len() != 1 {
        return Err(bad);
    }
    match map.get("sub") {
        Some(Value::String(s))
            if !s.is_empty()
                && s.len() <= max_len
                && !s.chars().any(char::is_control)
                && s == client_id =>
        {
            Ok(Some(s.clone()))
        }
        _ => Err(bad),
    }
}
