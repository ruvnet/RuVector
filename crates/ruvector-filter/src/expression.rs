use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Filter expression for querying vectors by payload
///
/// Serialized through [`FilterExpressionWire`]: logical operators use struct
/// variants on the wire (`{"type":"and","filters":[...]}`,
/// `{"type":"not","filter":{...}}`), while leaf variants keep their original
/// shape. Deriving serde directly on this internally tagged enum made the
/// recursive `And`/`Or`/`Not` newtype variants unserializable (serde cannot tag a
/// sequence) and sent rustc into unbounded `TaggedSerializer` nesting (E0275).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(into = "FilterExpressionWire", from = "FilterExpressionWire")]
pub enum FilterExpression {
    // Comparison operators
    Eq {
        field: String,
        value: Value,
    },
    Ne {
        field: String,
        value: Value,
    },
    Gt {
        field: String,
        value: Value,
    },
    Gte {
        field: String,
        value: Value,
    },
    Lt {
        field: String,
        value: Value,
    },
    Lte {
        field: String,
        value: Value,
    },

    // Range
    Range {
        field: String,
        gte: Option<Value>,
        lte: Option<Value>,
    },

    // Array operations
    In {
        field: String,
        values: Vec<Value>,
    },

    // Text matching
    Match {
        field: String,
        text: String,
    },

    // Geo operations (basic)
    GeoRadius {
        field: String,
        lat: f64,
        lon: f64,
        radius_m: f64,
    },
    GeoBoundingBox {
        field: String,
        top_left: (f64, f64),
        bottom_right: (f64, f64),
    },

    // Logical operators
    And {
        exprs: Vec<FilterExpression>,
    },
    Or {
        exprs: Vec<FilterExpression>,
    },
    Not {
        expr: Box<FilterExpression>,
    },

    // Existence check
    Exists {
        field: String,
    },
    IsNull {
        field: String,
    },
}

/// Serde representation of [`FilterExpression`]; see its docs.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum FilterExpressionWire {
    Eq {
        field: String,
        value: Value,
    },
    Ne {
        field: String,
        value: Value,
    },
    Gt {
        field: String,
        value: Value,
    },
    Gte {
        field: String,
        value: Value,
    },
    Lt {
        field: String,
        value: Value,
    },
    Lte {
        field: String,
        value: Value,
    },
    Range {
        field: String,
        gte: Option<Value>,
        lte: Option<Value>,
    },
    In {
        field: String,
        values: Vec<Value>,
    },
    Match {
        field: String,
        text: String,
    },
    GeoRadius {
        field: String,
        lat: f64,
        lon: f64,
        radius_m: f64,
    },
    GeoBoundingBox {
        field: String,
        top_left: (f64, f64),
        bottom_right: (f64, f64),
    },
    And {
        filters: Vec<FilterExpression>,
    },
    Or {
        filters: Vec<FilterExpression>,
    },
    Not {
        filter: Box<FilterExpression>,
    },
    Exists {
        field: String,
    },
    IsNull {
        field: String,
    },
}

impl From<FilterExpression> for FilterExpressionWire {
    fn from(e: FilterExpression) -> Self {
        use FilterExpression as F;
        match e {
            F::Eq { field, value } => Self::Eq { field, value },
            F::Ne { field, value } => Self::Ne { field, value },
            F::Gt { field, value } => Self::Gt { field, value },
            F::Gte { field, value } => Self::Gte { field, value },
            F::Lt { field, value } => Self::Lt { field, value },
            F::Lte { field, value } => Self::Lte { field, value },
            F::Range { field, gte, lte } => Self::Range { field, gte, lte },
            F::In { field, values } => Self::In { field, values },
            F::Match { field, text } => Self::Match { field, text },
            F::GeoRadius {
                field,
                lat,
                lon,
                radius_m,
            } => Self::GeoRadius {
                field,
                lat,
                lon,
                radius_m,
            },
            F::GeoBoundingBox {
                field,
                top_left,
                bottom_right,
            } => Self::GeoBoundingBox {
                field,
                top_left,
                bottom_right,
            },
            F::And(filters) => Self::And { filters },
            F::Or(filters) => Self::Or { filters },
            F::Not(filter) => Self::Not { filter },
            F::Exists { field } => Self::Exists { field },
            F::IsNull { field } => Self::IsNull { field },
        }
    }
}

impl From<FilterExpressionWire> for FilterExpression {
    fn from(w: FilterExpressionWire) -> Self {
        use FilterExpressionWire as W;
        match w {
            W::Eq { field, value } => Self::Eq { field, value },
            W::Ne { field, value } => Self::Ne { field, value },
            W::Gt { field, value } => Self::Gt { field, value },
            W::Gte { field, value } => Self::Gte { field, value },
            W::Lt { field, value } => Self::Lt { field, value },
            W::Lte { field, value } => Self::Lte { field, value },
            W::Range { field, gte, lte } => Self::Range { field, gte, lte },
            W::In { field, values } => Self::In { field, values },
            W::Match { field, text } => Self::Match { field, text },
            W::GeoRadius {
                field,
                lat,
                lon,
                radius_m,
            } => Self::GeoRadius {
                field,
                lat,
                lon,
                radius_m,
            },
            W::GeoBoundingBox {
                field,
                top_left,
                bottom_right,
            } => Self::GeoBoundingBox {
                field,
                top_left,
                bottom_right,
            },
            W::And { filters } => Self::And(filters),
            W::Or { filters } => Self::Or(filters),
            W::Not { filter } => Self::Not(filter),
            W::Exists { field } => Self::Exists { field },
            W::IsNull { field } => Self::IsNull { field },
        }
    }
}

impl FilterExpression {
    /// Create an equality filter
    pub fn eq(field: impl Into<String>, value: Value) -> Self {
        Self::Eq {
            field: field.into(),
            value,
        }
    }

    /// Create a not-equal filter
    pub fn ne(field: impl Into<String>, value: Value) -> Self {
        Self::Ne {
            field: field.into(),
            value,
        }
    }

    /// Create a greater-than filter
    pub fn gt(field: impl Into<String>, value: Value) -> Self {
        Self::Gt {
            field: field.into(),
            value,
        }
    }

    /// Create a greater-than-or-equal filter
    pub fn gte(field: impl Into<String>, value: Value) -> Self {
        Self::Gte {
            field: field.into(),
            value,
        }
    }

    /// Create a less-than filter
    pub fn lt(field: impl Into<String>, value: Value) -> Self {
        Self::Lt {
            field: field.into(),
            value,
        }
    }

    /// Create a less-than-or-equal filter
    pub fn lte(field: impl Into<String>, value: Value) -> Self {
        Self::Lte {
            field: field.into(),
            value,
        }
    }

    /// Create a range filter
    pub fn range(field: impl Into<String>, gte: Option<Value>, lte: Option<Value>) -> Self {
        Self::Range {
            field: field.into(),
            gte,
            lte,
        }
    }

    /// Create an IN filter
    pub fn in_values(field: impl Into<String>, values: Vec<Value>) -> Self {
        Self::In {
            field: field.into(),
            values,
        }
    }

    /// Create a text match filter
    pub fn match_text(field: impl Into<String>, text: impl Into<String>) -> Self {
        Self::Match {
            field: field.into(),
            text: text.into(),
        }
    }

    /// Create a geo radius filter
    pub fn geo_radius(field: impl Into<String>, lat: f64, lon: f64, radius_m: f64) -> Self {
        Self::GeoRadius {
            field: field.into(),
            lat,
            lon,
            radius_m,
        }
    }

    /// Create a geo bounding box filter
    pub fn geo_bounding_box(
        field: impl Into<String>,
        top_left: (f64, f64),
        bottom_right: (f64, f64),
    ) -> Self {
        Self::GeoBoundingBox {
            field: field.into(),
            top_left,
            bottom_right,
        }
    }

    /// Create an AND filter
    pub fn and(filters: Vec<FilterExpression>) -> Self {
        Self::And { exprs: filters }
    }

    /// Create an OR filter
    pub fn or(filters: Vec<FilterExpression>) -> Self {
        Self::Or { exprs: filters }
    }

    /// Create a NOT filter
    // Public API constructor mirrors `and`/`or`; not the `std::ops::Not` trait.
    #[allow(clippy::should_implement_trait)]
    pub fn not(filter: FilterExpression) -> Self {
        Self::Not {
            expr: Box::new(filter),
        }
    }

    /// Create an EXISTS filter
    pub fn exists(field: impl Into<String>) -> Self {
        Self::Exists {
            field: field.into(),
        }
    }

    /// Create an IS NULL filter
    pub fn is_null(field: impl Into<String>) -> Self {
        Self::IsNull {
            field: field.into(),
        }
    }

    /// Get all field names referenced in this expression
    pub fn get_fields(&self) -> Vec<String> {
        let mut fields = Vec::new();
        self.collect_fields(&mut fields);
        fields.sort();
        fields.dedup();
        fields
    }

    fn collect_fields(&self, fields: &mut Vec<String>) {
        match self {
            Self::Eq { field, .. }
            | Self::Ne { field, .. }
            | Self::Gt { field, .. }
            | Self::Gte { field, .. }
            | Self::Lt { field, .. }
            | Self::Lte { field, .. }
            | Self::Range { field, .. }
            | Self::In { field, .. }
            | Self::Match { field, .. }
            | Self::GeoRadius { field, .. }
            | Self::GeoBoundingBox { field, .. }
            | Self::Exists { field }
            | Self::IsNull { field } => {
                fields.push(field.clone());
            }
            Self::And { exprs } | Self::Or { exprs } => {
                for expr in exprs {
                    expr.collect_fields(fields);
                }
            }
            Self::Not { expr } => {
                expr.collect_fields(fields);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_filter_builders() {
        let filter = FilterExpression::eq("status", json!("active"));
        assert!(matches!(filter, FilterExpression::Eq { .. }));

        let filter = FilterExpression::and(vec![
            FilterExpression::eq("status", json!("active")),
            FilterExpression::gte("age", json!(18)),
        ]);
        assert!(matches!(filter, FilterExpression::And { .. }));
    }

    #[test]
    fn test_get_fields() {
        let filter = FilterExpression::and(vec![
            FilterExpression::eq("status", json!("active")),
            FilterExpression::or(vec![
                FilterExpression::gte("age", json!(18)),
                FilterExpression::lt("score", json!(100)),
            ]),
        ]);

        let fields = filter.get_fields();
        assert_eq!(fields, vec!["age", "score", "status"]);
    }

    #[test]
    fn test_serialization() {
        let filter = FilterExpression::eq("status", json!("active"));
        let json = serde_json::to_string(&filter).unwrap();
        let deserialized: FilterExpression = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, FilterExpression::Eq { .. }));
        // Leaf wire format is unchanged by the wire-enum indirection.
        assert_eq!(
            serde_json::to_value(&filter).unwrap(),
            json!({"type": "eq", "field": "status", "value": "active"})
        );
    }

    #[test]
    fn test_serialization_of_logical_operators() {
        let filter = FilterExpression::and(vec![
            FilterExpression::eq("status", json!("active")),
            FilterExpression::or(vec![
                FilterExpression::gte("age", json!(18)),
                FilterExpression::not(FilterExpression::exists("banned")),
            ]),
        ]);
        let value = serde_json::to_value(&filter).unwrap();
        assert_eq!(value["type"], "and");
        assert_eq!(value["filters"][1]["type"], "or");
        assert_eq!(value["filters"][1]["filters"][1]["type"], "not");
        assert_eq!(
            value["filters"][1]["filters"][1]["filter"]["type"],
            "exists"
        );

        let back: FilterExpression = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(&back).unwrap(), value);
        assert_eq!(back.get_fields(), vec!["age", "banned", "status"]);
    }

    #[test]
    fn test_logical_operator_round_trip() {
        let filter = FilterExpression::and(vec![
            FilterExpression::eq("status", json!("active")),
            FilterExpression::not(FilterExpression::or(vec![
                FilterExpression::lt("score", json!(10)),
                FilterExpression::exists("banned"),
            ])),
        ]);

        // Exact wire shape, per the `#[serde(tag = "type", rename_all = "snake_case")]`
        // enum definition: each variant is `{"type": <snake_case variant name>, ...struct fields}`.
        let expected = json!({
            "type": "and",
            "exprs": [
                {"type": "eq", "field": "status", "value": "active"},
                {"type": "not", "expr": {
                    "type": "or",
                    "exprs": [
                        {"type": "lt", "field": "score", "value": 10},
                        {"type": "exists", "field": "banned"},
                    ]
                }},
            ]
        });

        let actual = serde_json::to_value(&filter).unwrap();
        assert_eq!(actual, expected);

        let decoded: FilterExpression = serde_json::from_value(actual).unwrap();
        assert_eq!(serde_json::to_value(&decoded).unwrap(), expected);
    }
}
