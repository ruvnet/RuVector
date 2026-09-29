//! Boundary parsing of `application/x-www-form-urlencoded` bodies and query
//! strings that the Worker has already percent-decoded into pairs.
//!
//! Rules shared by every endpoint (RFC 6749 §3.1): a parameter sent without a
//! value is treated as omitted; a repeated parameter is an error; each value
//! is bounded. Unknown parameters are ignored.

use crate::error::{OAuthError, OAuthErrorCode};
use std::collections::BTreeMap;

/// Maximum length of any single parameter value (bytes).
pub const MAX_PARAM_LEN: usize = 2048;
/// Maximum number of pairs accepted in one request.
pub const MAX_PARAMS: usize = 32;
/// Maximum length of a parameter name (bytes; visible ASCII only).
pub const MAX_PARAM_NAME_LEN: usize = 64;
/// Maximum number of distinct scope tokens.
pub const MAX_SCOPE_TOKENS: usize = 32;
/// Identity scopes accepted and dropped (ADR-351 §5.3: the edge AS issues no
/// ID token).
pub const IDENTITY_SCOPES: [&str; 3] = ["openid", "profile", "email"];

/// Decoded, de-duplicated parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Params {
    map: BTreeMap<String, String>,
}

impl Params {
    /// Build from decoded pairs.
    ///
    /// Errors (`invalid_request` unless noted): more than [`MAX_PARAMS`]
    /// pairs; any name (even with an empty value) that is empty, longer than
    /// [`MAX_PARAM_NAME_LEN`] or not visible ASCII; a non-empty value longer
    /// than [`MAX_PARAM_LEN`]; the same name twice with a value (a repeated
    /// `resource` is `invalid_target`, since only one target is supported).
    pub fn from_pairs(pairs: &[(String, String)]) -> Result<Self, OAuthError> {
        if pairs.len() > MAX_PARAMS {
            return Err(OAuthError::new(
                OAuthErrorCode::InvalidRequest,
                "too many parameters",
            ));
        }
        let mut map = BTreeMap::new();
        for (k, v) in pairs {
            if k.is_empty()
                || k.len() > MAX_PARAM_NAME_LEN
                || !k.bytes().all(|b| b.is_ascii_graphic())
            {
                return Err(OAuthError::new(
                    OAuthErrorCode::InvalidRequest,
                    "invalid parameter name",
                ));
            }
            if v.is_empty() {
                continue;
            }
            if v.len() > MAX_PARAM_LEN {
                return Err(OAuthError::new(
                    OAuthErrorCode::InvalidRequest,
                    "parameter too long",
                ));
            }
            if map.insert(k.clone(), v.clone()).is_some() {
                return Err(if k == "resource" {
                    OAuthError::new(
                        OAuthErrorCode::InvalidTarget,
                        "only one resource is supported",
                    )
                } else {
                    OAuthError::new(OAuthErrorCode::InvalidRequest, "repeated parameter")
                });
            }
        }
        Ok(Params { map })
    }

    /// Optional value.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.map.get(name).map(String::as_str)
    }

    /// Owned optional value.
    pub fn take(&self, name: &str) -> Option<String> {
        self.get(name).map(str::to_string)
    }

    /// Required value, else `invalid_request`.
    pub fn require(&self, name: &str, missing: &'static str) -> Result<String, OAuthError> {
        self.take(name)
            .ok_or(OAuthError::new(OAuthErrorCode::InvalidRequest, missing))
    }
}

/// Split a space-delimited scope string (RFC 6749 §3.3).
///
/// Contract: at most [`MAX_PARAM_LEN`] bytes (checked first, so the JSON DCR
/// path is bounded too); single spaces only (no empty tokens, no
/// leading/trailing space); each token is `%x21 / %x23-5B / %x5D-7E`;
/// duplicates are dropped keeping first-seen order; at most
/// [`MAX_SCOPE_TOKENS`] tokens (checked while parsing, so work is bounded).
/// Violations yield `invalid_scope`.
pub fn split_scope(scope: &str) -> Result<Vec<String>, OAuthError> {
    let err = OAuthError::new(OAuthErrorCode::InvalidScope, "malformed scope");
    if scope.len() > MAX_PARAM_LEN {
        return Err(err);
    }
    let mut out: Vec<String> = Vec::new();
    for tok in scope.split(' ') {
        let ok = !tok.is_empty()
            && tok
                .bytes()
                .all(|b| b == 0x21 || (0x23..=0x5B).contains(&b) || (0x5D..=0x7E).contains(&b));
        if !ok {
            return Err(err);
        }
        if !out.iter().any(|s| s == tok) {
            if out.len() == MAX_SCOPE_TOKENS {
                return Err(err);
            }
            out.push(tok.to_string());
        }
    }
    Ok(out)
}

/// Drop the identity scopes `openid`, `profile`, `email` (ADR-351 §5.3:
/// accepted and dropped at registration and authorization). Callers fall back
/// to their default ceiling when nothing remains.
pub fn strip_identity_scopes(scopes: Vec<String>) -> Vec<String> {
    scopes
        .into_iter()
        .filter(|s| !IDENTITY_SCOPES.contains(&s.as_str()))
        .collect()
}

/// `requested ⊆ ceiling`, else `invalid_scope`.
pub fn ensure_subset(requested: &[String], ceiling: &[String]) -> Result<(), OAuthError> {
    if requested.iter().all(|s| ceiling.contains(s)) {
        Ok(())
    } else {
        Err(OAuthError::new(
            OAuthErrorCode::InvalidScope,
            "scope exceeds the allowed ceiling",
        ))
    }
}
