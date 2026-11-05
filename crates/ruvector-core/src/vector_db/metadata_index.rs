//! In-memory inverted view of vector metadata, for the exact-match filters of [`SearchQuery`](crate::SearchQuery).
//!
//! `key -> canonical value -> ids`. A filtered query looks up one posting set per clause, which gives the exact match
//! count (the planner's selectivity) and an O(1) membership test (the traversal's predicate) without touching storage.
//! Scanning storage metadata per query instead would cost O(n) before the search even starts.
//!
//! Memory: one id string per (vector, metadata key), twice (posting set and per-id record, the latter so an overwrite
//! or delete can unlink the old values).

use crate::types::VectorId;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Default)]
pub(super) struct MetadataIndex {
    postings: HashMap<String, HashMap<String, HashSet<VectorId>>>,
    by_id: HashMap<VectorId, Vec<(String, String)>>,
}

impl MetadataIndex {
    /// Record `id`'s metadata. Mirrors storage: metadata on an entry replaces the old record, and an entry without
    /// metadata leaves the old record in place.
    pub(super) fn upsert(&mut self, id: &str, metadata: Option<&HashMap<String, Value>>) {
        let Some(metadata) = metadata else {
            return;
        };
        self.remove(id);
        let mut record = Vec::with_capacity(metadata.len());
        for (key, value) in metadata {
            let value = canonical(value);
            self.postings
                .entry(key.clone())
                .or_default()
                .entry(value.clone())
                .or_default()
                .insert(id.to_string());
            record.push((key.clone(), value));
        }
        self.by_id.insert(id.to_string(), record);
    }

    pub(super) fn remove(&mut self, id: &str) {
        let Some(record) = self.by_id.remove(id) else {
            return;
        };
        for (key, value) in record {
            if let Some(values) = self.postings.get_mut(&key) {
                if let Some(ids) = values.get_mut(&value) {
                    ids.remove(id);
                    if ids.is_empty() {
                        values.remove(&value);
                    }
                }
                if values.is_empty() {
                    self.postings.remove(&key);
                }
            }
        }
    }

    /// The posting set of every `key == value` clause, smallest first; `None` when some clause matches nothing.
    pub(super) fn clauses(
        &self,
        filter: &HashMap<String, Value>,
    ) -> Option<Vec<&HashSet<VectorId>>> {
        let mut sets = Vec::with_capacity(filter.len());
        for (key, value) in filter {
            sets.push(self.postings.get(key)?.get(&canonical(value))?);
        }
        sets.sort_by_key(|s| s.len());
        Some(sets)
    }
}

/// A string that is equal for two values exactly when the values are `==`: object keys sorted (map order does not
/// affect `Value` equality) and `-0.0` folded into `0.0`.
fn canonical(v: &Value) -> String {
    let mut out = String::new();
    write_canonical(v, &mut out);
    out
}

fn write_canonical(v: &Value, out: &mut String) {
    match v {
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            for (i, (key, value)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                write_canonical(value, out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        Value::Number(n) if n.is_f64() && n.as_f64() == Some(0.0) => out.push_str("0.0"),
        other => out.push_str(&other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn md(v: Value) -> HashMap<String, Value> {
        serde_json::from_value(v).unwrap()
    }

    fn ids(index: &MetadataIndex, filter: Value) -> Vec<String> {
        let Some(clauses) = index.clauses(&md(filter)) else {
            return vec![];
        };
        let mut out: Vec<String> = clauses[0]
            .iter()
            .filter(|id| clauses.iter().all(|s| s.contains(*id)))
            .cloned()
            .collect();
        out.sort();
        out
    }

    #[test]
    fn clauses_intersect_and_track_overwrites_and_deletes() {
        let mut ix = MetadataIndex::default();
        ix.upsert("a", Some(&md(json!({"c": "x", "n": 1}))));
        ix.upsert("b", Some(&md(json!({"c": "x", "n": 2}))));
        ix.upsert("c", Some(&md(json!({"c": "y", "n": 1}))));
        assert_eq!(ids(&ix, json!({"c": "x"})), vec!["a", "b"]);
        assert_eq!(ids(&ix, json!({"c": "x", "n": 1})), vec!["a"]);
        assert!(ids(&ix, json!({"c": "z"})).is_empty());
        assert!(ids(&ix, json!({"missing": 1})).is_empty());

        ix.upsert("a", Some(&md(json!({"c": "y"}))));
        assert_eq!(ids(&ix, json!({"c": "x"})), vec!["b"]);
        assert!(ids(&ix, json!({"c": "y", "n": 1})) == vec!["c"]);
        ix.upsert("a", None); // no metadata on the entry: storage keeps the old record, so do we
        assert_eq!(ids(&ix, json!({"c": "y"})), vec!["a", "c"]);
        ix.remove("c");
        assert_eq!(ids(&ix, json!({"c": "y"})), vec!["a"]);
        ix.remove("a");
        ix.remove("b");
        assert!(ix.postings.is_empty() && ix.by_id.is_empty());
    }

    #[test]
    fn canonical_form_agrees_with_value_equality() {
        assert_eq!(
            canonical(&json!({"b": 1, "a": [true, null]})),
            canonical(&json!({"a": [true, null], "b": 1}))
        );
        assert_eq!(canonical(&json!(-0.0)), canonical(&json!(0.0)));
        assert_ne!(canonical(&json!(1)), canonical(&json!(1.0)));
        assert_ne!(canonical(&json!("1")), canonical(&json!(1)));
        assert_ne!(
            canonical(&json!({"a": "b,c"})),
            canonical(&json!({"a": "b", "c": 1}))
        );
    }
}
