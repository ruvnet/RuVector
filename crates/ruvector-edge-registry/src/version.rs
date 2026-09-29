//! Strict SemVer 2.0.0 versions: `MAJOR.MINOR.PATCH[-PRERELEASE]`.
//!
//! **Build metadata (`+...`) is refused.** SemVer gives `1.0.0+a` and
//! `1.0.0+b` equal precedence while they are different strings; accepting
//! both would let two different artifacts claim one immutable version.
//! Numeric components have no leading zeros and fit in `u64`; the whole
//! string is at most [`MAX_VERSION_LEN`] bytes. Ordering is SemVer
//! precedence (a pre-release sorts before its release).

use core::cmp::Ordering;
use core::fmt;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Longest accepted version string.
pub const MAX_VERSION_LEN: usize = 64;

/// Why a version was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum VersionError {
    /// Not `MAJOR.MINOR.PATCH[-PRE]`, or too long.
    #[error("version must be MAJOR.MINOR.PATCH[-PRERELEASE]")]
    Shape,
    /// A numeric identifier has a leading zero or overflows.
    #[error("numeric identifier is malformed")]
    Numeric,
    /// Build metadata is not accepted.
    #[error("build metadata is not accepted")]
    BuildMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Pre {
    Num(u64),
    Alpha(String),
}

impl Ord for Pre {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Pre::Num(a), Pre::Num(b)) => a.cmp(b),
            (Pre::Num(_), Pre::Alpha(_)) => Ordering::Less,
            (Pre::Alpha(_), Pre::Num(_)) => Ordering::Greater,
            (Pre::Alpha(a), Pre::Alpha(b)) => a.cmp(b),
        }
    }
}
impl PartialOrd for Pre {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A validated version.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Version {
    major: u64,
    minor: u64,
    patch: u64,
    pre: Vec<Pre>,
    text: String,
}

fn numeric(s: &str) -> Result<u64, VersionError> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(VersionError::Shape);
    }
    if s.len() > 1 && s.starts_with('0') {
        return Err(VersionError::Numeric);
    }
    s.parse().map_err(|_| VersionError::Numeric)
}

impl Version {
    /// Strict parse.
    pub fn parse(s: &str) -> Result<Self, VersionError> {
        if s.len() > MAX_VERSION_LEN {
            return Err(VersionError::Shape);
        }
        if s.contains('+') {
            return Err(VersionError::BuildMetadata);
        }
        let (core, pre) = match s.split_once('-') {
            Some((c, p)) => (c, Some(p)),
            None => (s, None),
        };
        let mut parts = core.split('.');
        let (Some(a), Some(b), Some(c), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Err(VersionError::Shape);
        };
        let (major, minor, patch) = (numeric(a)?, numeric(b)?, numeric(c)?);
        let mut ids = Vec::new();
        if let Some(pre) = pre {
            for id in pre.split('.') {
                let valid =
                    !id.is_empty() && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-');
                if !valid {
                    return Err(VersionError::Shape);
                }
                if id.bytes().all(|b| b.is_ascii_digit()) {
                    ids.push(Pre::Num(numeric(id)?));
                } else {
                    ids.push(Pre::Alpha(id.to_string()));
                }
            }
        }
        Ok(Version {
            major,
            minor,
            patch,
            pre: ids,
            text: s.to_string(),
        })
    }

    /// The canonical text (exactly what was parsed).
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// `true` for a pre-release (`1.0.0-rc.1`).
    pub fn is_prerelease(&self) -> bool {
        !self.pre.is_empty()
    }

    /// `(major, minor, patch)`.
    pub fn triple(&self) -> (u64, u64, u64) {
        (self.major, self.minor, self.patch)
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        self.triple().cmp(&other.triple()).then_with(|| {
            match (self.pre.is_empty(), other.pre.is_empty()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => self.pre.cmp(&other.pre),
            }
        })
    }
}
impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl Serialize for Version {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.text)
    }
}

impl<'de> Deserialize<'de> for Version {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Version::parse(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Version {
        Version::parse(s).unwrap()
    }

    #[test]
    fn precedence_follows_semver_spec_example() {
        let order = [
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-alpha.beta",
            "1.0.0-beta",
            "1.0.0-beta.2",
            "1.0.0-beta.11",
            "1.0.0-rc.1",
            "1.0.0",
            "1.0.1",
            "1.1.0",
            "2.0.0",
            "10.0.0",
        ];
        for w in order.windows(2) {
            assert!(v(w[0]) < v(w[1]), "{} < {}", w[0], w[1]);
        }
        assert!(v("1.0.0-rc.1").is_prerelease() && !v("1.0.0").is_prerelease());
    }

    #[test]
    fn refuses_malformed_versions() {
        let cases: &[(&str, VersionError)] = &[
            ("1.0", VersionError::Shape),
            ("1.0.0.0", VersionError::Shape),
            ("v1.0.0", VersionError::Shape),
            ("1.0.0-", VersionError::Shape),
            ("1.0.0-a..b", VersionError::Shape),
            ("1.0.0-a_b", VersionError::Shape),
            ("01.0.0", VersionError::Numeric),
            ("1.0.0-01", VersionError::Numeric),
            ("99999999999999999999.0.0", VersionError::Numeric),
            ("1.0.0+build.1", VersionError::BuildMetadata),
            ("", VersionError::Shape),
            (" 1.0.0", VersionError::Shape),
        ];
        for (s, want) in cases {
            assert_eq!(Version::parse(s).unwrap_err(), *want, "{s}");
        }
        assert!(Version::parse(&format!("1.0.0-{}", "a".repeat(64))).is_err());
    }

    #[test]
    fn serde_round_trips() {
        let x = v("1.2.3-rc.1");
        let j = serde_json::to_string(&x).unwrap();
        assert_eq!(serde_json::from_str::<Version>(&j).unwrap(), x);
        assert!(serde_json::from_str::<Version>("\"1.2\"").is_err());
    }
}
