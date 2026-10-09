//! Python <-> `ruvector_graph` value converters, plus the `GraphError` ->
//! `RuVectorError` mapper. Split out of `graph.rs` purely to keep that
//! file under this codebase's 500-line-per-file convention — nothing
//! here is Cypher-specific, it's the same kind of boundary-conversion
//! code `hnsw.rs` keeps inline because that file has room to.

use std::collections::HashMap;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyString};

use ruvector_graph::{Edge, GraphError, Node, PropertyValue};

use crate::error::RuVectorError;

/// Map a `ruvector_graph::GraphError` into a `PyErr` carrying the same
/// `RuVectorError` base class every other backend uses (see
/// `crates/ruvector-py/src/error.rs`'s `to_pyerr`/`to_pyerr_core`). Kept as
/// a local mapper rather than appended to `error.rs`: two sibling forks
/// (`gnn.rs`, `cluster.rs`/`sona.rs`) are adding their own `to_pyerr_*`
/// mappers in the same session, and three additions landing in the same
/// tail-of-file hunk of a shared file is a guaranteed merge conflict for
/// no real benefit — a private fn with the same one-line body here costs
/// nothing extra and touches zero shared files.
pub(super) fn to_pyerr_graph(err: GraphError) -> PyErr {
    RuVectorError::new_err(err.to_string())
}

/// Convert a Python value into `ruvector_graph::PropertyValue`.
///
/// Hand-rolled, mirroring `hnsw.rs`'s `py_to_json` line for line, rather
/// than round-tripping through `serde_json::Value`: `PropertyValue` derives
/// `serde::Serialize` with serde's default *externally tagged* enum
/// representation, so `serde_json::to_value(PropertyValue::Integer(5))`
/// produces `{"Integer": 5}`, not the bare JSON `5` a Python caller would
/// expect — going through `serde_json` would require either a custom
/// `Serialize` impl on `PropertyValue` (not ours to add; it lives in
/// `ruvector-graph`) or a second translation layer just to undo the
/// tagging. A direct Python <-> `PropertyValue` converter is simpler and is
/// explicitly sanctioned by the task brief over pulling in a new crate.
pub(super) fn py_to_property(value: &Bound<'_, PyAny>) -> PyResult<PropertyValue> {
    if value.is_none() {
        return Ok(PropertyValue::Null);
    }
    // Order matters: Python `bool` is a subclass of `int`, so the bool
    // check must run before the int check or every `True`/`False` would be
    // silently coerced into `PropertyValue::Integer(1)`/`Integer(0)`.
    if let Ok(b) = value.extract::<bool>() {
        return Ok(PropertyValue::Boolean(b));
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(PropertyValue::Integer(i));
    }
    if let Ok(f) = value.extract::<f64>() {
        return Ok(PropertyValue::Float(f));
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(PropertyValue::String(s));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let items: PyResult<Vec<PropertyValue>> = list.iter().map(|v| py_to_property(&v)).collect();
        return Ok(PropertyValue::Array(items?));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        return Ok(PropertyValue::Map(py_dict_to_properties(dict)?));
    }
    Err(PyTypeError::new_err(format!(
        "property values must be str/int/float/bool/None/list/dict, got {}",
        value.get_type().name()?
    )))
}

pub(super) fn py_dict_to_properties(
    dict: &Bound<'_, PyDict>,
) -> PyResult<HashMap<String, PropertyValue>> {
    let mut map = HashMap::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let key: String = k
            .extract()
            .map_err(|_| PyTypeError::new_err("property keys must be strings"))?;
        map.insert(key, py_to_property(&v)?);
    }
    Ok(map)
}

/// Convert a `PropertyValue` back into a Python object. `Array` and `List`
/// are the same JSON-facing shape (the Rust type keeps them as separate
/// variants; see `ruvector_graph::types::PropertyValue`'s doc comment) and
/// `FloatArray` — a contiguous `Vec<f32>` used for embeddings — becomes a
/// plain `list[float]` on the Python side, same as any other array.
pub(super) fn property_to_py<'py>(
    py: Python<'py>,
    value: &PropertyValue,
) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::IntoPyObjectExt;
    match value {
        PropertyValue::Null => Ok(py.None().into_bound(py)),
        PropertyValue::Boolean(b) => b.into_bound_py_any(py),
        PropertyValue::Integer(i) => i.into_bound_py_any(py),
        PropertyValue::Float(f) => f.into_bound_py_any(py),
        PropertyValue::String(s) => s.into_bound_py_any(py),
        PropertyValue::Array(items) | PropertyValue::List(items) => {
            let converted: PyResult<Vec<_>> = items.iter().map(|v| property_to_py(py, v)).collect();
            PyList::new(py, converted?)?.into_bound_py_any(py)
        }
        PropertyValue::FloatArray(items) => {
            let converted: Vec<f64> = items.iter().map(|f| *f as f64).collect();
            PyList::new(py, converted)?.into_bound_py_any(py)
        }
        PropertyValue::Map(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, property_to_py(py, v)?)?;
            }
            dict.into_bound_py_any(py)
        }
    }
}

pub(super) fn properties_to_py<'py>(
    py: Python<'py>,
    map: &HashMap<String, PropertyValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, property_to_py(py, v)?)?;
    }
    Ok(dict)
}

/// Parse a Python `labels` argument into the `Vec<String>` the real
/// `NodeBuilder::labels` takes. Explicitly rejects a bare `str`: Python
/// iterates a string one character at a time, so `labels="Person"` would
/// silently build six single-letter labels (`P`, `e`, `r`, ...) instead of
/// raising — the exact kind of boundary mistake `hnsw.rs` guards against
/// for its own arguments (see that module's C-contiguity checks).
pub(super) fn py_to_labels(value: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if value.cast::<PyString>().is_ok() {
        return Err(PyTypeError::new_err(
            "labels must be a sequence of strings, not a bare string \
             (a str iterates to one label per character)",
        ));
    }
    // A dict is iterable too (over its keys), so without this check a
    // dict would silently pass through `extract::<Vec<String>>` below and
    // turn its keys into labels — surprising in the same way a bare `str`
    // is, so it gets the same explicit rejection.
    if value.cast::<PyDict>().is_ok() {
        return Err(PyTypeError::new_err(
            "labels must be a sequence of strings, not a dict",
        ));
    }
    // Any sequence (list, tuple, ...) of strings — not just `PyList` — per
    // the `Sequence[str]` the stub promises in `_native.pyi`. `extract`'s
    // own `FromPyObject<Vec<String>>` impl walks any Python iterable and
    // extracts each item as a `String`, giving the same clear `TypeError`
    // on a non-string element without a separate hand-rolled loop.
    value
        .extract::<Vec<String>>()
        .map_err(|_| PyTypeError::new_err("labels must be a sequence of strings"))
}

/// `{id, labels, properties}` — the Python-facing shape of a `Node`.
pub(super) fn node_to_py<'py>(py: Python<'py>, node: &Node) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &node.id)?;
    let labels: Vec<String> = node.labels.iter().map(|l| l.name.clone()).collect();
    dict.set_item("labels", labels)?;
    dict.set_item("properties", properties_to_py(py, &node.properties)?)?;
    Ok(dict)
}

/// `{id, from, to, type, properties}` — the Python-facing shape of an `Edge`.
pub(super) fn edge_to_py<'py>(py: Python<'py>, edge: &Edge) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &edge.id)?;
    dict.set_item("from", &edge.from)?;
    dict.set_item("to", &edge.to)?;
    dict.set_item("type", &edge.edge_type)?;
    dict.set_item("properties", properties_to_py(py, &edge.properties)?)?;
    Ok(dict)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attach_py() {
        Python::initialize();
    }

    #[test]
    fn property_roundtrip_covers_every_json_shape() {
        attach_py();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(
                c"value = {\"a\": 1, \"b\": 1.5, \"c\": \"s\", \"d\": True, \"e\": None, \"f\": [1, 2, 3]}",
                Some(&globals),
                None,
            )
            .unwrap();
            let value = globals.get_item("value").unwrap().unwrap();
            let dict = value.cast::<PyDict>().unwrap();
            let props = py_dict_to_properties(dict).expect("dict should convert");
            assert_eq!(props.get("a"), Some(&PropertyValue::Integer(1)));
            assert_eq!(props.get("b"), Some(&PropertyValue::Float(1.5)));
            assert_eq!(
                props.get("c"),
                Some(&PropertyValue::String("s".to_string()))
            );
            assert_eq!(props.get("d"), Some(&PropertyValue::Boolean(true)));
            assert_eq!(props.get("e"), Some(&PropertyValue::Null));
            assert_eq!(
                props.get("f"),
                Some(&PropertyValue::Array(vec![
                    PropertyValue::Integer(1),
                    PropertyValue::Integer(2),
                    PropertyValue::Integer(3),
                ]))
            );

            let back = properties_to_py(py, &props).expect("properties should convert back");
            assert_eq!(
                back.get_item("a")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                1_i64
            );
        });
    }

    #[test]
    fn py_to_property_rejects_unsupported_type() {
        attach_py();
        Python::attach(|py| {
            let obj = py
                .import("builtins")
                .unwrap()
                .getattr("object")
                .unwrap()
                .call0()
                .unwrap();
            let err = py_to_property(&obj).unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn py_to_labels_rejects_bare_string() {
        attach_py();
        Python::attach(|py| {
            use pyo3::IntoPyObjectExt;
            let s = "Person".into_bound_py_any(py).unwrap();
            let err = py_to_labels(&s).unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn py_to_labels_accepts_list_of_strings() {
        attach_py();
        Python::attach(|py| {
            let list = PyList::new(py, ["Person", "Employee"]).unwrap();
            let labels = py_to_labels(list.as_any()).expect("list of strings should convert");
            assert_eq!(labels, vec!["Person".to_string(), "Employee".to_string()]);
        });
    }

    #[test]
    fn py_to_labels_accepts_tuple_of_strings() {
        attach_py();
        Python::attach(|py| {
            use pyo3::types::PyTuple;
            let tuple = PyTuple::new(py, ["Person", "Employee"]).unwrap();
            let labels = py_to_labels(tuple.as_any()).expect("tuple of strings should convert");
            assert_eq!(labels, vec!["Person".to_string(), "Employee".to_string()]);
        });
    }

    #[test]
    fn py_to_labels_rejects_dict() {
        attach_py();
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("Person", true).unwrap();
            let err = py_to_labels(dict.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn to_pyerr_graph_wraps_node_not_found() {
        attach_py();
        let err = to_pyerr_graph(GraphError::NodeNotFound("x".to_string()));
        Python::attach(|py| {
            assert!(err.is_instance_of::<RuVectorError>(py));
        });
    }
}
