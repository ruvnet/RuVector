//! Metadata and filters (ADR-351 §6.1 `filter_idx`, §7 `filter?: ≤8 clauses`).
//!
//! A collection declares up to [`MAX_FILTERABLE_KEYS`] metadata keys at
//! create time. Metadata is a JSON object of at most
//! [`MAX_METADATA_BYTES`] serialized bytes. A filter is a JSON object of at
//! most [`MAX_FILTER_CLAUSES`] clauses, each on a **declared** key (others are
//! `400 invalid_request`), ANDed together:
//!
//! - `"k": <scalar>` or `"k": {"$eq": <scalar>}`: equal;
//! - `"k": {"$ne": <scalar>}`: present and not equal;
//! - `"k": {"$in": [<scalar>, ...]}`: equal to one of ≤ [`MAX_IN_VALUES`].
//!
//! Scalars are strings (≤ [`MAX_FILTER_STRING`] bytes in a filter), numbers
//! or booleans; numbers compare as `f64`, so `1` equals `1.0`. A row without
//! the key never matches. A filter holds at most [`MAX_FILTER_VALUES`]
//! values in total, and its per-row cost is [`Filter::cost`].
//!
//! Resident rows keep only a compact form of their declared keys
//! ([`Compact`]), never a parsed metadata tree.

use crate::error::{ErrorCode, OpError};
use serde_json::{Map, Value as Json};

/// Declared filterable keys per collection.
pub const MAX_FILTERABLE_KEYS: usize = 8;
/// Clauses per filter.
pub const MAX_FILTER_CLAUSES: usize = 8;
/// Values in one `$in`.
pub const MAX_IN_VALUES: usize = 32;
/// Values across every clause of one filter.
pub const MAX_FILTER_VALUES: usize = 64;
/// Bytes of one string value in a filter.
pub const MAX_FILTER_STRING: usize = 256;
/// Serialized metadata bytes per vector (4 KiB).
pub const MAX_METADATA_BYTES: usize = 4 << 10;
/// Metadata key length.
pub const MAX_KEY_LEN: usize = 64;

/// Validate a declared key: `^[A-Za-z0-9_.-]{1,64}$`.
pub fn validate_key(k: &str) -> Result<(), OpError> {
    let ok = !k.is_empty()
        && k.len() <= MAX_KEY_LEN
        && k.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'));
    if ok {
        Ok(())
    } else {
        Err(OpError::invalid("invalid metadata key"))
    }
}

/// Validate a collection's `filterable_keys` (≤ 8, valid, no duplicates).
pub fn validate_filterable_keys(keys: &[String]) -> Result<(), OpError> {
    if keys.len() > MAX_FILTERABLE_KEYS {
        return Err(OpError::invalid("too many filterable keys"));
    }
    for (i, k) in keys.iter().enumerate() {
        validate_key(k)?;
        if keys[..i].contains(k) {
            return Err(OpError::invalid("duplicate filterable key"));
        }
    }
    Ok(())
}

/// Validate a metadata object, returning its serialized form.
pub fn validate_metadata(m: &Json) -> Result<String, OpError> {
    if !m.is_object() {
        return Err(OpError::invalid("metadata must be an object"));
    }
    let s = serde_json::to_string(m).map_err(|_| OpError::invalid("metadata"))?;
    if s.len() > MAX_METADATA_BYTES {
        return Err(OpError::new(
            ErrorCode::PayloadTooLarge,
            "metadata too large",
        ));
    }
    Ok(s)
}

/// Canonical text of a scalar for the `filter_idx` table, `None` for
/// non-scalars (not indexed).
pub fn index_value(v: &Json) -> Option<String> {
    match v {
        Json::String(s) => Some(format!("s:{s}")),
        Json::Number(n) => n.as_f64().map(|f| format!("n:{f}")),
        Json::Bool(b) => Some(format!("b:{b}")),
        _ => None,
    }
}

/// `filter_idx` rows `(key, value)` for a metadata object.
pub fn index_rows(meta: &Map<String, Json>, declared: &[String]) -> Vec<(String, String)> {
    declared
        .iter()
        .filter_map(|k| meta.get(k).and_then(index_value).map(|v| (k.clone(), v)))
        .collect()
}

/// A filterable scalar.
#[derive(Debug, Clone, PartialEq)]
pub enum Scalar {
    /// String.
    Str(Box<str>),
    /// Number, compared as `f64`.
    Num(f64),
    /// Boolean.
    Bool(bool),
}

impl Scalar {
    fn from_json(v: &Json) -> Option<Scalar> {
        match v {
            Json::String(s) => Some(Scalar::Str(s.as_str().into())),
            Json::Number(n) => n.as_f64().map(Scalar::Num),
            Json::Bool(b) => Some(Scalar::Bool(*b)),
            _ => None,
        }
    }

    /// Heap bytes held beyond the enum itself.
    pub fn heap_bytes(&self) -> usize {
        match self {
            Scalar::Str(s) => s.len(),
            _ => 0,
        }
    }
}

/// Compact resident form of a row's declared keys: `(key index, value)`.
pub type Compact = Box<[(u8, Scalar)]>;

/// The compact form of `meta` for the `declared` keys (non-scalars are
/// not filterable and are skipped).
pub fn compact(meta: Option<&Map<String, Json>>, declared: &[String]) -> Compact {
    let Some(m) = meta else {
        return Box::new([]);
    };
    declared
        .iter()
        .enumerate()
        .filter_map(|(i, k)| {
            let s = m.get(k).and_then(Scalar::from_json)?;
            Some((u8::try_from(i).ok()?, s))
        })
        .collect()
}

/// Resident bytes of a compact form.
pub fn compact_bytes(c: &[(u8, Scalar)]) -> usize {
    core::mem::size_of_val(c) + c.iter().map(|(_, s)| s.heap_bytes()).sum::<usize>()
}

#[derive(Debug, Clone, PartialEq)]
enum Clause {
    Eq(u8, Scalar),
    Ne(u8, Scalar),
    In(u8, Vec<Scalar>),
}

/// A validated filter.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Filter {
    clauses: Vec<Clause>,
    values: usize,
}

impl Filter {
    /// Parse `filter` against the collection's declared keys.
    pub fn parse(filter: &Json, declared: &[String]) -> Result<Filter, OpError> {
        let obj = filter
            .as_object()
            .ok_or(OpError::invalid("filter must be an object"))?;
        if obj.len() > MAX_FILTER_CLAUSES {
            return Err(OpError::invalid("too many filter clauses"));
        }
        let mut out = Filter::default();
        for (k, v) in obj {
            let idx = declared
                .iter()
                .position(|d| d == k)
                .and_then(|i| u8::try_from(i).ok())
                .ok_or(OpError::invalid("filter key not declared filterable"))?;
            let c = parse_clause(idx, v)?;
            out.values += match &c {
                Clause::In(_, vs) => vs.len(),
                _ => 1,
            };
            out.clauses.push(c);
        }
        if out.values > MAX_FILTER_VALUES {
            return Err(OpError::invalid("too many filter values"));
        }
        Ok(out)
    }

    /// `true` if the compact row satisfies every clause.
    pub fn matches(&self, row: &[(u8, Scalar)]) -> bool {
        let get = |k: u8| row.iter().find(|(i, _)| *i == k).map(|(_, s)| s);
        self.clauses.iter().all(|c| match c {
            Clause::Eq(k, v) => get(*k).is_some_and(|x| x == v),
            Clause::Ne(k, v) => get(*k).is_some_and(|x| x != v),
            Clause::In(k, vs) => get(*k).is_some_and(|x| vs.contains(x)),
        })
    }

    /// `true` for an empty filter.
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Work per scanned row: one distance plus one comparison per value.
    pub fn cost(&self) -> u64 {
        1 + self.values as u64
    }
}

fn scalar(v: &Json) -> Result<Scalar, OpError> {
    if matches!(v, Json::String(s) if s.len() > MAX_FILTER_STRING) {
        return Err(OpError::invalid("filter string too long"));
    }
    Scalar::from_json(v).ok_or(OpError::invalid("filter value must be a scalar"))
}

fn parse_clause(k: u8, v: &Json) -> Result<Clause, OpError> {
    let Some(op) = v.as_object() else {
        return Ok(Clause::Eq(k, scalar(v)?));
    };
    let mut it = op.iter();
    let (Some((name, arg)), None) = (it.next(), it.next()) else {
        return Err(OpError::invalid(
            "filter operator object needs exactly one operator",
        ));
    };
    match name.as_str() {
        "$eq" => Ok(Clause::Eq(k, scalar(arg)?)),
        "$ne" => Ok(Clause::Ne(k, scalar(arg)?)),
        "$in" => {
            let arr = arg
                .as_array()
                .ok_or(OpError::invalid("$in needs an array"))?;
            if arr.is_empty() || arr.len() > MAX_IN_VALUES {
                return Err(OpError::invalid("$in size out of range"));
            }
            let vals = arr.iter().map(scalar).collect::<Result<_, _>>()?;
            Ok(Clause::In(k, vals))
        }
        _ => Err(OpError::invalid("unknown filter operator")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn keys() -> Vec<String> {
        vec!["color".into(), "n".into(), "ok".into()]
    }

    fn c(v: Json) -> Compact {
        compact(v.as_object(), &keys())
    }

    #[test]
    fn equality_in_ne_and_numeric_normalisation() {
        let f = Filter::parse(&json!({"color": "red", "n": {"$in": [1, 2]}}), &keys()).unwrap();
        assert_eq!(f.cost(), 4);
        assert!(f.matches(&c(json!({"color": "red", "n": 2.0}))));
        assert!(!f.matches(&c(json!({"color": "red", "n": 3}))));
        assert!(!f.matches(&c(json!({"n": 1}))));
        assert!(!f.matches(&[]));
        let ne = Filter::parse(&json!({"ok": {"$ne": true}}), &keys()).unwrap();
        assert!(ne.matches(&c(json!({"ok": false}))));
        assert!(!ne.matches(&c(json!({}))));
        // Undeclared and non-scalar keys never enter the compact form.
        assert_eq!(c(json!({"secret": 1, "color": [1], "n": 5})).len(), 1);
    }

    #[test]
    fn rejects_undeclared_nonscalar_and_oversized() {
        assert!(Filter::parse(&json!({"secret": 1}), &keys()).is_err());
        assert!(Filter::parse(&json!({"color": [1]}), &keys()).is_err());
        assert!(Filter::parse(&json!({"color": {"$gt": 1}}), &keys()).is_err());
        assert!(Filter::parse(&json!({"color": {"$eq": 1, "$ne": 2}}), &keys()).is_err());
        assert!(Filter::parse(&json!([1]), &keys()).is_err());
        let nine: Vec<String> = (0..9).map(|i| format!("k{i}")).collect();
        assert!(validate_filterable_keys(&nine).is_err());
        assert!(validate_filterable_keys(&["a".into(), "a".into()]).is_err());
        let big = json!({"x": "y".repeat(MAX_METADATA_BYTES)});
        assert_eq!(
            validate_metadata(&big).unwrap_err().code,
            ErrorCode::PayloadTooLarge
        );
    }

    #[test]
    fn filter_cost_is_bounded() {
        let long = "x".repeat(MAX_FILTER_STRING + 1);
        assert!(Filter::parse(&json!({ "color": long }), &keys()).is_err());
        let ok = "x".repeat(MAX_FILTER_STRING);
        assert!(Filter::parse(&json!({ "color": ok }), &keys()).is_ok());
        let in32: Vec<u32> = (0..32).collect();
        let two = json!({"color": {"$in": in32}, "n": {"$in": in32}});
        assert_eq!(Filter::parse(&two, &keys()).unwrap().cost(), 65);
        let three = json!({"color": {"$in": in32}, "n": {"$in": in32}, "ok": true});
        assert!(Filter::parse(&three, &keys()).is_err());
    }
}
