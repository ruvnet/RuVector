//! Package names: `@scope/name`, both lowercase slugs.
//!
//! - scope: `[a-z0-9][a-z0-9-]*`, 2..=39 chars, no `--`, no trailing `-`.
//!   A scope is owned by exactly one tenant (claimed on first push), which
//!   makes every name globally unambiguous for cross-tenant public pulls.
//! - name: `[a-z0-9][a-z0-9._-]*`, 1..=64 chars, no `..`, not ending in `.`
//!   or `-`.
//! - The full `@scope/name` is at most [`MAX_FULL_NAME`] bytes.
//! - [`RESERVED_SCOPES`] and [`RESERVED_NAMES`] are refused, as is any scope
//!   whose [`scope_skeleton`] equals a reserved scope's, contains a
//!   [`BRAND_WORDS`] skeleton or starts with a [`BRAND_PREFIXES`] entry.
//!   Look-alikes of *claimed* scopes are refused by the scope directory.
//!
//! Every component that reaches an R2 key or a Durable Object name comes out
//! of these parsers, so no key component is attacker-shaped (ADR §6.3).

use core::fmt;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Longest scope.
pub const MAX_SCOPE_LEN: usize = 39;
/// Longest package name (without scope).
pub const MAX_NAME_LEN: usize = 64;
/// Longest `@scope/name`.
pub const MAX_FULL_NAME: usize = 1 + MAX_SCOPE_LEN + 1 + MAX_NAME_LEN;

/// Scopes nobody can claim: platform, brand and routing words.
pub const RESERVED_SCOPES: &[&str] = &[
    "admin",
    "anthropic",
    "api",
    "blobs",
    "cognitum",
    "edge",
    "internal",
    "latest",
    "null",
    "official",
    "private",
    "public",
    "registry",
    "root",
    "ruv",
    "ruvector",
    "rvf",
    "staging",
    "support",
    "system",
    "undefined",
    "uploads",
    "v1",
    "www",
];

/// Brand words no scope may contain (compared on [`scope_skeleton`]s, so
/// `ru-vect0r-ai` and `c0gnitum-one` are caught too).
pub const BRAND_WORDS: &[&str] = &["anthropic", "cognitum", "official", "ruvector", "ruvnet"];

/// Brand words no scope may start with (too short to refuse as substrings).
pub const BRAND_PREFIXES: &[&str] = &["rvf"];

/// The confusable skeleton of a scope: `-` dropped, look-alike digits and
/// letters folded (`0→o`, `1/i→l`, `3→e`, `4→a`, `5→s`, `7→t`, `8→b`,
/// `rn→m`, `vv→w`) and runs of one character collapsed. Two scopes with the
/// same skeleton are indistinguishable at a glance, so the scope directory
/// ([`crate::scopes`]) lets only one of them be claimed.
pub fn scope_skeleton(s: &str) -> String {
    let folded: String = s
        .chars()
        .filter(|c| *c != '-')
        .map(|c| match c {
            '0' => 'o',
            '1' | 'i' => 'l',
            '3' => 'e',
            '4' => 'a',
            '5' => 's',
            '7' => 't',
            '8' => 'b',
            c => c,
        })
        .collect();
    let folded = folded.replace("rn", "m").replace("vv", "w");
    let mut out = String::with_capacity(folded.len());
    for c in folded.chars() {
        if !out.ends_with(c) {
            out.push(c);
        }
    }
    out
}

fn reserved_scope(s: &str) -> bool {
    if RESERVED_SCOPES.contains(&s) {
        return true;
    }
    let sk = scope_skeleton(s);
    RESERVED_SCOPES.iter().any(|r| scope_skeleton(r) == sk)
        || BRAND_WORDS.iter().any(|b| sk.contains(&scope_skeleton(b)))
        || BRAND_PREFIXES
            .iter()
            .any(|b| s.starts_with(b) || sk.starts_with(&scope_skeleton(b)))
}

/// Package names nobody can use (they collide with routes or tooling).
pub const RESERVED_NAMES: &[&str] = &[
    "blobs",
    "favicon.ico",
    "index",
    "latest",
    "manifest",
    "node_modules",
    "null",
    "undefined",
    "uploads",
    "versions",
];

/// Why a name was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum NameError {
    /// Missing `@`, `/` or a component.
    #[error("package name must be @scope/name")]
    Shape,
    /// Length out of range.
    #[error("{0} length out of range")]
    Length(&'static str),
    /// Character outside the allowed set (including uppercase).
    #[error("{0} contains a disallowed character")]
    Charset(&'static str),
    /// `--`, `..`, or a bad first/last character.
    #[error("{0} has a disallowed sequence or boundary character")]
    Sequence(&'static str),
    /// A reserved word.
    #[error("{0} is reserved")]
    Reserved(&'static str),
}

/// A validated scope (without `@`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Scope(String);

impl Scope {
    /// Strict parse.
    pub fn parse(s: &str) -> Result<Self, NameError> {
        const W: &str = "scope";
        if !(2..=MAX_SCOPE_LEN).contains(&s.len()) {
            return Err(NameError::Length(W));
        }
        if !s
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        {
            return Err(NameError::Charset(W));
        }
        if s.starts_with('-') || s.ends_with('-') || s.contains("--") {
            return Err(NameError::Sequence(W));
        }
        if reserved_scope(s) {
            return Err(NameError::Reserved(W));
        }
        Ok(Scope(s.to_string()))
    }

    /// The scope text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A validated `@scope/name`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PackageName {
    scope: Scope,
    name: String,
}

impl PackageName {
    /// Strict parse of `@scope/name`.
    pub fn parse(full: &str) -> Result<Self, NameError> {
        if full.len() > MAX_FULL_NAME {
            return Err(NameError::Length("package name"));
        }
        let rest = full.strip_prefix('@').ok_or(NameError::Shape)?;
        let (scope, name) = rest.split_once('/').ok_or(NameError::Shape)?;
        let scope = Scope::parse(scope)?;
        Self::check_name(name)?;
        Ok(PackageName {
            scope,
            name: name.to_string(),
        })
    }

    fn check_name(n: &str) -> Result<(), NameError> {
        const W: &str = "name";
        if !(1..=MAX_NAME_LEN).contains(&n.len()) {
            return Err(NameError::Length(W));
        }
        let ok =
            |b: u8| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-');
        if !n.bytes().all(ok) {
            return Err(NameError::Charset(W));
        }
        let first = n.as_bytes()[0];
        if !(first.is_ascii_lowercase() || first.is_ascii_digit())
            || n.ends_with('.')
            || n.ends_with('-')
            || n.contains("..")
        {
            return Err(NameError::Sequence(W));
        }
        if RESERVED_NAMES.contains(&n) {
            return Err(NameError::Reserved(W));
        }
        Ok(())
    }

    /// The scope.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// The unscoped name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for PackageName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}/{}", self.scope.0, self.name)
    }
}

impl Serialize for PackageName {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for PackageName {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        PackageName::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_well_formed_names() {
        for ok in [
            "@acme/embeddings",
            "@a1/x",
            "@my-org/v2.index_small",
            "@ab/0",
        ] {
            let p = PackageName::parse(ok).unwrap();
            assert_eq!(p.to_string(), ok);
        }
        let long = format!("@{}/{}", "a".repeat(39), "b".repeat(64));
        assert!(PackageName::parse(&long).is_ok());
    }

    #[test]
    fn refuses_malformed_names() {
        let cases: &[(&str, NameError)] = &[
            ("acme/x", NameError::Shape),
            ("@acme", NameError::Shape),
            ("@acme/", NameError::Length("name")),
            ("@a/x", NameError::Length("scope")),
            ("@Acme/x", NameError::Charset("scope")),
            ("@acme/X", NameError::Charset("name")),
            ("@acme/a/b", NameError::Charset("name")),
            ("@acme/a:publish", NameError::Charset("name")),
            ("@acme/é", NameError::Charset("name")),
            ("@ac--me/x", NameError::Sequence("scope")),
            ("@-acme/x", NameError::Sequence("scope")),
            ("@acme/.x", NameError::Sequence("name")),
            ("@acme/_x", NameError::Sequence("name")),
            ("@acme/a..b", NameError::Sequence("name")),
            ("@acme/x.", NameError::Sequence("name")),
            ("@ruvector/x", NameError::Reserved("scope")),
            ("@acme/latest", NameError::Reserved("name")),
            ("@acme/node_modules", NameError::Reserved("name")),
        ];
        for (input, want) in cases {
            assert_eq!(PackageName::parse(input).unwrap_err(), *want, "{input}");
        }
        let too_long = format!("@{}/{}", "a".repeat(40), "b");
        assert!(PackageName::parse(&too_long).is_err());
        assert!(PackageName::parse(&format!("@acme/{}", "b".repeat(65))).is_err());
    }

    #[test]
    fn look_alike_and_brand_scopes_are_reserved() {
        for squat in [
            "ruvect0r",
            "ru-vector",
            "ruvector-ai",
            "ruvectorr",
            "rvf-official",
            "rvfhub",
            "cognitum-one",
            "c0gnitum",
            "anthropic-ai",
            "anthr0pic",
            "my-ruvnet",
            "0fficial",
            "adm1n",
            "r-uv",
            "pub1ic",
        ] {
            assert_eq!(
                Scope::parse(squat).unwrap_err(),
                NameError::Reserved("scope"),
                "{squat}"
            );
        }
        for ok in [
            "acme",
            "my-org",
            "openai",
            "data-team",
            "vectors",
            "ab",
            "acme-2",
        ] {
            assert!(Scope::parse(ok).is_ok(), "{ok}");
        }
    }

    #[test]
    fn skeleton_folds_confusables() {
        assert_eq!(scope_skeleton("acme"), scope_skeleton("acrne"));
        assert_eq!(scope_skeleton("acme"), scope_skeleton("ac-me"));
        assert_eq!(scope_skeleton("acme"), scope_skeleton("accme"));
        assert_eq!(scope_skeleton("w1dget"), scope_skeleton("vvidget"));
        assert_ne!(scope_skeleton("acme"), scope_skeleton("acne"));
    }

    #[test]
    fn serde_round_trips_and_validates() {
        let p = PackageName::parse("@acme/x").unwrap();
        let j = serde_json::to_string(&p).unwrap();
        assert_eq!(j, "\"@acme/x\"");
        assert_eq!(serde_json::from_str::<PackageName>(&j).unwrap(), p);
        assert!(serde_json::from_str::<PackageName>("\"@ACME/x\"").is_err());
    }
}
