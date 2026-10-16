//! Snapshot capture of a real GraphDB. Capture is NOT a database transaction.
use ruvector_graph::{GraphDB, PropertyValue};
use ruvector_language::{
    error::{Error, Result},
    types::{Edge, Node, Observation, Rule, Snapshot},
};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Capture while the application holds its own exclusive writer gate.
///
/// GraphDB permits concurrent writes through `&self`; borrowing it cannot establish
/// snapshot isolation. The caller MUST prevent every writer for this entire call.
/// Count checks detect some races, not all races. Hyperedges, multiple labels and
/// edge properties are rejected rather than silently discarded. v0.1 supports one
/// label per node. A FloatArray property named `embedding` becomes the vector while
/// remaining in the property map. Evidence is explicit and never inferred from
/// arbitrary node properties.
pub fn capture_quiescent(
    db: &GraphDB,
    revision: impl Into<String>,
    observations: Vec<Observation>,
    rules: Vec<Rule>,
) -> Result<Snapshot> {
    let revision = revision.into();
    let count = (db.node_count(), db.edge_count());
    if !crate::artifact::valid_name(&revision)
        || count
            .0
            .saturating_add(count.1)
            .saturating_add(observations.len())
            .saturating_add(rules.len())
            > 100_000
        || rules.len() > 10_000
        || db.hyperedge_count() != 0
    {
        return Err(Error::new(
            "CAPTURE",
            "unsupported graph shape, revision or size",
        ));
    }
    let mut work = 100_000usize;
    let mut nodes = Vec::with_capacity(count.0);
    db.try_for_each_node(|node| {
        if node.labels.len() != 1 {
            return Err(Error::new("CAPTURE", "exactly one label per node required"));
        }
        identity(&node.id, &mut work)?;
        identity(&node.labels[0].name, &mut work)?;
        let vector = match node.properties.get("embedding") {
            Some(PropertyValue::FloatArray(v))
                if v.len() <= 4096 && v.iter().all(|n| n.is_finite()) =>
            {
                Some(v.clone())
            }
            Some(_) => {
                return Err(Error::new(
                    "CAPTURE",
                    "embedding must be a bounded finite FloatArray",
                ))
            }
            None => None,
        };
        let mut properties = BTreeMap::new();
        for (key, value) in &node.properties {
            charge_text(key, &mut work)?;
            properties.insert(key.clone(), property(value, 0, &mut work)?);
        }
        nodes.push(Node {
            id: node.id.clone(),
            label: node.labels[0].name.clone(),
            properties,
            vector,
        });
        Ok(())
    })?;
    let mut edges = Vec::with_capacity(count.1);
    db.try_for_each_edge(|edge| {
        if !edge.properties.is_empty() {
            return Err(Error::new(
                "CAPTURE",
                "edge properties unsupported in v0.1 snapshot",
            ));
        }
        for value in [&edge.id, &edge.from, &edge.to, &edge.edge_type] {
            identity(value, &mut work)?;
        }
        edges.push(Edge {
            id: edge.id.clone(),
            source: edge.from.clone(),
            target: edge.to.clone(),
            relation: edge.edge_type.clone(),
        });
        Ok(())
    })?;
    if count != (nodes.len(), edges.len()) || count != (db.node_count(), db.edge_count()) {
        return Err(Error::new(
            "CAPTURE",
            "graph changed during capture; hold application writer gate",
        ));
    }
    nodes.sort_by(|a, b| a.id.cmp(&b.id));
    edges.sort_by(|a, b| a.id.cmp(&b.id));
    let mut snapshot = Snapshot {
        version: 1,
        revision,
        nodes,
        edges,
        observations,
        rules,
    };
    snapshot.observations.sort_by(|a, b| a.id.cmp(&b.id));
    snapshot.rules.sort_by(|a, b| a.id.cmp(&b.id));
    ruvector_language::runtime::validate_snapshot(&snapshot)?;
    Ok(snapshot)
}

fn charge(work: &mut usize, amount: usize) -> Result<()> {
    *work = work
        .checked_sub(amount)
        .ok_or_else(|| Error::new("CAPTURE", "property size limit exceeded"))?;
    Ok(())
}

fn charge_text(s: &str, work: &mut usize) -> Result<()> {
    charge(work, s.len().max(1))
}

fn identity(s: &str, work: &mut usize) -> Result<()> {
    if s.trim().is_empty() || s.len() > 4096 {
        return Err(Error::new(
            "CAPTURE",
            "identity must contain 1 through 4096 bytes",
        ));
    }
    charge_text(s, work)
}

fn property(value: &PropertyValue, depth: usize, work: &mut usize) -> Result<Value> {
    if depth > 16 {
        return Err(Error::new("CAPTURE", "property nesting exceeds 16"));
    }
    charge(work, 1)?;
    Ok(match value {
        PropertyValue::Null => Value::Null,
        PropertyValue::Boolean(v) => Value::Bool(*v),
        PropertyValue::Integer(v) => Value::Number((*v).into()),
        PropertyValue::Float(v) => Value::Number(
            serde_json::Number::from_f64(*v)
                .ok_or_else(|| Error::new("CAPTURE", "nonfinite float property"))?,
        ),
        PropertyValue::String(v) => {
            charge_text(v, work)?;
            Value::String(v.clone())
        }
        PropertyValue::Array(v) | PropertyValue::List(v) => {
            if v.len() > *work {
                return Err(Error::new("CAPTURE", "property array too large"));
            }
            Value::Array(
                v.iter()
                    .map(|x| property(x, depth + 1, work))
                    .collect::<Result<_>>()?,
            )
        }
        PropertyValue::FloatArray(v) => {
            charge(work, v.len())?;
            Value::Array(
                v.iter()
                    .map(|n| {
                        serde_json::Number::from_f64(*n as f64)
                            .map(Value::Number)
                            .ok_or_else(|| Error::new("CAPTURE", "nonfinite float array"))
                    })
                    .collect::<Result<_>>()?,
            )
        }
        PropertyValue::Map(v) => {
            if v.len() > *work {
                return Err(Error::new("CAPTURE", "property map too large"));
            }
            let mut sorted: Vec<_> = v.iter().collect();
            sorted.sort_by(|(a, _), (b, _)| a.cmp(b));
            let mut map = Map::new();
            for (key, val) in sorted {
                charge_text(key, work)?;
                map.insert(key.clone(), property(val, depth + 1, work)?);
            }
            Value::Object(map)
        }
    })
}
