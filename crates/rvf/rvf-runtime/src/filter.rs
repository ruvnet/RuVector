//! Filter expression evaluation for metadata-based vector filtering.
//!
//! Filter expressions are boolean predicate trees evaluated against
//! per-vector metadata. The runtime selects a strategy (pre-filter,
//! intra-filter, or post-filter) based on estimated selectivity.

use crate::options::MetadataValue;

/// A filter expression for metadata-based vector filtering.
///
/// Leaf nodes compare a metadata field against a literal value.
/// Internal nodes combine sub-expressions with boolean logic.
#[derive(Clone, Debug)]
pub enum FilterExpr {
    /// field == value
    Eq(u16, FilterValue),
    /// field != value
    Ne(u16, FilterValue),
    /// field < value
    Lt(u16, FilterValue),
    /// field <= value
    Le(u16, FilterValue),
    /// field > value
    Gt(u16, FilterValue),
    /// field >= value
    Ge(u16, FilterValue),
    /// field in [values]
    In(u16, Vec<FilterValue>),
    /// field in [low, high)
    Range(u16, FilterValue, FilterValue),
    /// All sub-expressions must match.
    And(Vec<FilterExpr>),
    /// Any sub-expression must match.
    Or(Vec<FilterExpr>),
    /// Negate the sub-expression.
    Not(Box<FilterExpr>),
}

/// A typed value used in filter comparisons.
#[derive(Clone, Debug, PartialEq)]
pub enum FilterValue {
    U64(u64),
    I64(i64),
    F64(f64),
    String(String),
    Bool(bool),
}

impl FilterValue {
    /// Compare two filter values. Returns None if types are incompatible.
    fn partial_cmp_value(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (FilterValue::U64(a), FilterValue::U64(b)) => a.partial_cmp(b),
            (FilterValue::I64(a), FilterValue::I64(b)) => a.partial_cmp(b),
            (FilterValue::F64(a), FilterValue::F64(b)) => a.partial_cmp(b),
            (FilterValue::String(a), FilterValue::String(b)) => a.partial_cmp(b),
            (FilterValue::Bool(a), FilterValue::Bool(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

/// The `META_SEG` schema type a stored value declares for its field, or `None`
/// for values that constrain no type of their own.
///
/// Mirrors the mapping `build_metadata_schema` applies when it encodes a
/// generation; the discriminant is `rvf_types::metadata::MetadataType as u8`.
fn declared_type(value: &MetadataValue) -> Option<u8> {
    use rvf_types::metadata::MetadataType;
    Some(match value {
        MetadataValue::Null | MetadataValue::DeleteField => return None,
        MetadataValue::String(_) => MetadataType::String as u8,
        MetadataValue::Bytes(_) => MetadataType::Bytes as u8,
        MetadataValue::I64(_) => MetadataType::I64 as u8,
        MetadataValue::U64(_) => MetadataType::U64 as u8,
        MetadataValue::F64(_) => MetadataType::F64 as u8,
        MetadataValue::Bool(_) => MetadataType::Bool as u8,
    })
}

/// In-memory metadata store for filter evaluation.
/// Maps vector IDs to their complete durable metadata record.
#[derive(Clone)]
pub(crate) struct MetadataStore {
    entries: std::collections::BTreeMap<u64, std::collections::BTreeMap<u16, MetadataValue>>,
    /// How many live records declare each field id with each value type.
    ///
    /// A `META_SEG` declares one schema entry per field id, so a field carried
    /// by two types at once cannot be encoded as a full snapshot. Maintaining
    /// the counts as records are written makes that state detectable in
    /// `O(fields)` at ingest time instead of requiring a full rescan of every
    /// record on every commit (issue #772).
    field_types: std::collections::BTreeMap<u16, std::collections::BTreeMap<u8, usize>>,
}

impl MetadataStore {
    pub(crate) fn new() -> Self {
        Self {
            entries: std::collections::BTreeMap::new(),
            field_types: std::collections::BTreeMap::new(),
        }
    }

    /// Count `value` towards its field's live type set.
    fn declare(&mut self, field_id: u16, value: &MetadataValue) {
        let Some(value_type) = declared_type(value) else {
            return;
        };
        *self
            .field_types
            .entry(field_id)
            .or_default()
            .entry(value_type)
            .or_insert(0) += 1;
    }

    /// Discount a value that is no longer live, dropping the field's entry
    /// once nothing declares it any more.
    fn undeclare(&mut self, field_id: u16, value: &MetadataValue) {
        let Some(value_type) = declared_type(value) else {
            return;
        };
        let Some(counts) = self.field_types.get_mut(&field_id) else {
            return;
        };
        if let Some(count) = counts.get_mut(&value_type) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                counts.remove(&value_type);
            }
        }
        if counts.is_empty() {
            self.field_types.remove(&field_id);
        }
    }

    /// The lowest field id that two live records give different value types.
    ///
    /// `None` means the live records can be encoded as one full snapshot.
    pub(crate) fn conflicting_field(&self) -> Option<u16> {
        self.field_types
            .iter()
            .find(|(_, types)| types.len() > 1)
            .map(|(&field_id, _)| field_id)
    }

    /// Add metadata for a vector. `fields` are (field_id, value) pairs.
    pub(crate) fn insert(&mut self, vector_id: u64, fields: Vec<(u16, MetadataValue)>) {
        self.entries.entry(vector_id).or_default();
        for (field_id, value) in fields {
            let previous = {
                let record = self.entries.entry(vector_id).or_default();
                if matches!(value, MetadataValue::DeleteField) {
                    record.remove(&field_id)
                } else {
                    record.insert(field_id, value.clone())
                }
            };
            if let Some(previous) = previous {
                self.undeclare(field_id, &previous);
            }
            self.declare(field_id, &value);
        }
    }

    /// Get a field value for a vector.
    pub(crate) fn get_field(&self, vector_id: u64, field_id: u16) -> Option<FilterValue> {
        self.entries
            .get(&vector_id)?
            .get(&field_id)
            .and_then(metadata_value_to_filter_option)
    }

    pub(crate) fn get(&self, vector_id: u64) -> Option<Vec<(u16, MetadataValue)>> {
        self.entries.get(&vector_id).map(|fields| {
            fields
                .iter()
                .map(|(&id, value)| (id, value.clone()))
                .collect()
        })
    }

    /// Fields of one record in ascending `field_id` order, borrowed so callers
    /// that only compare values do not clone the record.
    pub(crate) fn fields(
        &self,
        vector_id: u64,
    ) -> Option<&std::collections::BTreeMap<u16, MetadataValue>> {
        self.entries.get(&vector_id)
    }

    /// Vector identifiers carrying a record, in ascending order.
    pub(crate) fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries.keys().copied()
    }

    /// True when no vector carries a record.
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every record whose vector identifier fails `keep`.
    pub(crate) fn retain_ids(&mut self, keep: impl Fn(u64) -> bool) {
        let dropped: Vec<u64> = self
            .entries
            .keys()
            .copied()
            .filter(|&vector_id| !keep(vector_id))
            .collect();
        self.remove_ids(&dropped);
    }

    pub(crate) fn decoded_size(&self, vector_id: u64) -> usize {
        self.entries.get(&vector_id).map_or(0, |fields| {
            fields.iter().fold(0usize, |size, (_, value)| {
                size.saturating_add(2).saturating_add(match value {
                    MetadataValue::Null | MetadataValue::DeleteField => 1,
                    MetadataValue::U64(_) | MetadataValue::I64(_) | MetadataValue::F64(_) => 9,
                    MetadataValue::Bool(_) => 2,
                    MetadataValue::String(value) => 5usize.saturating_add(value.len()),
                    MetadataValue::Bytes(value) => 5usize.saturating_add(value.len()),
                })
            })
        })
    }

    /// Remove all metadata for the given vector IDs.
    pub(crate) fn remove_ids(&mut self, ids: &[u64]) {
        for id in ids {
            let Some(record) = self.entries.remove(id) else {
                continue;
            };
            for (field_id, value) in record {
                self.undeclare(field_id, &value);
            }
        }
    }

    /// Return vector count tracked by the metadata store.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Evaluate a filter expression against a single vector's metadata.
pub(crate) fn evaluate(expr: &FilterExpr, vector_id: u64, meta: &MetadataStore) -> bool {
    match expr {
        FilterExpr::Eq(field_id, val) => meta
            .get_field(vector_id, *field_id)
            .map(|v| &v == val)
            .unwrap_or(false),
        FilterExpr::Ne(field_id, val) => meta
            .get_field(vector_id, *field_id)
            .map(|v| &v != val)
            .unwrap_or(true),
        FilterExpr::Lt(field_id, val) => meta
            .get_field(vector_id, *field_id)
            .and_then(|v| v.partial_cmp_value(val))
            .map(|ord| ord == std::cmp::Ordering::Less)
            .unwrap_or(false),
        FilterExpr::Le(field_id, val) => meta
            .get_field(vector_id, *field_id)
            .and_then(|v| v.partial_cmp_value(val))
            .map(|ord| ord != std::cmp::Ordering::Greater)
            .unwrap_or(false),
        FilterExpr::Gt(field_id, val) => meta
            .get_field(vector_id, *field_id)
            .and_then(|v| v.partial_cmp_value(val))
            .map(|ord| ord == std::cmp::Ordering::Greater)
            .unwrap_or(false),
        FilterExpr::Ge(field_id, val) => meta
            .get_field(vector_id, *field_id)
            .and_then(|v| v.partial_cmp_value(val))
            .map(|ord| ord != std::cmp::Ordering::Less)
            .unwrap_or(false),
        FilterExpr::In(field_id, vals) => meta
            .get_field(vector_id, *field_id)
            .map(|v| vals.contains(&v))
            .unwrap_or(false),
        FilterExpr::Range(field_id, low, high) => meta
            .get_field(vector_id, *field_id)
            .and_then(|v| {
                let ge_low = v
                    .partial_cmp_value(low)
                    .map(|o| o != std::cmp::Ordering::Less)?;
                let lt_high = v
                    .partial_cmp_value(high)
                    .map(|o| o == std::cmp::Ordering::Less)?;
                Some(ge_low && lt_high)
            })
            .unwrap_or(false),
        FilterExpr::And(exprs) => exprs.iter().all(|e| evaluate(e, vector_id, meta)),
        FilterExpr::Or(exprs) => exprs.iter().any(|e| evaluate(e, vector_id, meta)),
        FilterExpr::Not(expr) => !evaluate(expr, vector_id, meta),
    }
}

/// Convert a MetadataValue (options module) to a FilterValue for evaluation.
pub(crate) fn metadata_value_to_filter(mv: &MetadataValue) -> FilterValue {
    match mv {
        MetadataValue::U64(v) => FilterValue::U64(*v),
        MetadataValue::I64(v) => FilterValue::I64(*v),
        MetadataValue::F64(v) => FilterValue::F64(*v),
        MetadataValue::String(v) => FilterValue::String(v.clone()),
        MetadataValue::Bytes(_) => FilterValue::String(String::new()),
        MetadataValue::Bool(v) => FilterValue::Bool(*v),
        MetadataValue::Null | MetadataValue::DeleteField => FilterValue::String(String::new()),
    }
}

fn metadata_value_to_filter_option(mv: &MetadataValue) -> Option<FilterValue> {
    match mv {
        MetadataValue::Null | MetadataValue::DeleteField | MetadataValue::Bytes(_) => None,
        value => Some(metadata_value_to_filter(value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> MetadataStore {
        let mut store = MetadataStore::new();
        store.insert(
            0,
            vec![
                (0, MetadataValue::String("apple".into())),
                (1, MetadataValue::U64(100)),
            ],
        );
        store.insert(
            1,
            vec![
                (0, MetadataValue::String("banana".into())),
                (1, MetadataValue::U64(200)),
            ],
        );
        store.insert(
            2,
            vec![
                (0, MetadataValue::String("apple".into())),
                (1, MetadataValue::U64(300)),
            ],
        );
        store
    }

    #[test]
    fn filter_eq() {
        let store = make_store();
        let expr = FilterExpr::Eq(0, FilterValue::String("apple".into()));
        assert!(evaluate(&expr, 0, &store));
        assert!(!evaluate(&expr, 1, &store));
        assert!(evaluate(&expr, 2, &store));
    }

    #[test]
    fn filter_ne() {
        let store = make_store();
        let expr = FilterExpr::Ne(0, FilterValue::String("apple".into()));
        assert!(!evaluate(&expr, 0, &store));
        assert!(evaluate(&expr, 1, &store));
    }

    #[test]
    fn filter_range() {
        let store = make_store();
        let expr = FilterExpr::Range(1, FilterValue::U64(150), FilterValue::U64(250));
        assert!(!evaluate(&expr, 0, &store)); // 100 < 150
        assert!(evaluate(&expr, 1, &store)); // 200 in [150, 250)
        assert!(!evaluate(&expr, 2, &store)); // 300 >= 250
    }

    #[test]
    fn filter_and_or() {
        let store = make_store();
        let expr = FilterExpr::And(vec![
            FilterExpr::Eq(0, FilterValue::String("apple".into())),
            FilterExpr::Gt(1, FilterValue::U64(150)),
        ]);
        assert!(!evaluate(&expr, 0, &store)); // apple but 100 <= 150
        assert!(!evaluate(&expr, 1, &store)); // banana
        assert!(evaluate(&expr, 2, &store)); // apple and 300 > 150
    }

    #[test]
    fn filter_not() {
        let store = make_store();
        let expr = FilterExpr::Not(Box::new(FilterExpr::Eq(
            0,
            FilterValue::String("apple".into()),
        )));
        assert!(!evaluate(&expr, 0, &store));
        assert!(evaluate(&expr, 1, &store));
    }

    #[test]
    fn filter_in() {
        let store = make_store();
        let expr = FilterExpr::In(1, vec![FilterValue::U64(100), FilterValue::U64(300)]);
        assert!(evaluate(&expr, 0, &store));
        assert!(!evaluate(&expr, 1, &store));
        assert!(evaluate(&expr, 2, &store));
    }
}
