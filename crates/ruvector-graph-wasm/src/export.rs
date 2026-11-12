//! `exportCypher`: the graph as one Cypher CREATE statement.
//!
//! Imported node IDs are fresh UUIDs, so an edge can only be recreated through a
//! variable bound earlier in the same statement. The export is therefore a single
//! CREATE on one line: every node as `(nK ...)`, then every edge between those
//! variables. `importCypher(text.split('\n'))` and `query(text)` both load it.
//!
//! Labels, types and keys are always backtick-quoted, and strings are quoted and
//! escaped, so stored text can never be read back as Cypher syntax.

use std::collections::HashMap;

use serde_json::Value as Json;

use crate::cypher_exec::Store;

/// Renders the nodes and edges in `st` as one CREATE statement. Empty graph: "".
pub(crate) fn export_cypher(st: &Store) -> String {
    let mut node_ids: Vec<&String> = st.nodes.keys().collect();
    node_ids.sort();
    let var: HashMap<&str, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();

    let mut patterns = Vec::with_capacity(st.nodes.len() + st.edges.len());
    for (i, id) in node_ids.iter().enumerate() {
        let node = &st.nodes[*id];
        let mut p = format!("(n{}", i);
        for label in &node.labels {
            p.push(':');
            p.push_str(&identifier(label));
        }
        p.push_str(&properties(&node.properties));
        p.push(')');
        patterns.push(p);
    }

    let mut edges: Vec<_> = st.edges.values().collect();
    edges.sort_by(|a, b| a.id.cmp(&b.id));
    for e in edges {
        // insert_edge and remove_node keep both endpoints present.
        let (Some(from), Some(to)) = (var.get(e.from.as_str()), var.get(e.to.as_str())) else {
            continue;
        };
        patterns.push(format!(
            "(n{})-[:{}{}]->(n{})",
            from,
            identifier(&e.edge_type),
            properties(&e.properties),
            to
        ));
    }

    if patterns.is_empty() {
        return String::new();
    }
    format!("CREATE {}", patterns.join(", "))
}

/// A backtick-quoted identifier. A backtick inside is doubled, which the lexer
/// reads as two identifiers, so such a name fails to import instead of escaping
/// its quotes.
fn identifier(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

fn properties(props: &HashMap<String, Json>) -> String {
    if props.is_empty() {
        return String::new();
    }
    let mut keys: Vec<&String> = props.keys().collect();
    keys.sort();
    let body: Vec<String> = keys
        .into_iter()
        .map(|k| format!("{}: {}", identifier(k), literal(&props[k])))
        .collect();
    format!(" {{{}}}", body.join(", "))
}

/// A Cypher literal that the lexer reads back as the same value.
fn literal(v: &Json) -> String {
    match v {
        Json::Null => "null".to_string(),
        Json::Bool(b) => b.to_string(),
        Json::Number(n) => match n.as_f64() {
            Some(f) if !(n.is_i64() || n.is_u64()) => float(f),
            _ => n.to_string(),
        },
        Json::String(s) => string(s),
        Json::Array(items) => format!(
            "[{}]",
            items.iter().map(literal).collect::<Vec<_>>().join(", ")
        ),
        Json::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, x)| format!("{}: {}", identifier(k), literal(x)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// A single-quoted string. Line breaks are escaped so the export stays on one line.
fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out.push('\'');
    out
}

/// The lexer only reads floats written as `digits.digits[e[+-]digits]`.
fn float(f: f64) -> String {
    let s = format!("{:?}", f);
    match s.find('e') {
        Some(pos) if !s[..pos].contains('.') => format!("{}.0{}", &s[..pos], &s[pos..]),
        _ => s,
    }
}

#[cfg(test)]
mod tests {
    use crate::GraphDB;
    use serde_json::{json, Value as Json};
    use std::collections::HashMap;

    type Props = HashMap<String, Json>;

    fn props(v: Json) -> Props {
        serde_json::from_value(v).unwrap()
    }

    /// Nodes as (labels, properties) and edges as (from props, type, props, to
    /// props), sorted, so two graphs with different IDs compare equal.
    fn shape(db: &GraphDB) -> (Vec<String>, Vec<String>) {
        let st = db.lock_store();
        let node = |id: &String| {
            let n = &st.nodes[id];
            format!("{:?}{}", n.labels, json!(n.properties))
        };
        let mut nodes: Vec<String> = st.nodes.keys().map(node).collect();
        let mut edges: Vec<String> = st
            .edges
            .values()
            .map(|e| {
                let p = json!(e.properties);
                format!("{} -{}{}-> {}", node(&e.from), e.edge_type, p, node(&e.to))
            })
            .collect();
        nodes.sort();
        edges.sort();
        (nodes, edges)
    }

    fn reimport(db: &GraphDB) -> GraphDB {
        let text = db.export_cypher();
        let copy = GraphDB::new(None).unwrap();
        for line in text.split('\n') {
            copy.execute_cypher(line)
                .unwrap_or_else(|e| panic!("import failed: {}\n{:?}", e, line));
        }
        copy
    }

    #[test]
    fn export_round_trips_nodes_edges_and_awkward_values() {
        let db = GraphDB::new(None).unwrap();
        {
            let mut st = db.lock_store();
            let a = st.insert_node(
                vec!["Person".into(), "MATCH".into()],
                props(json!({
                    "name": "line1\nline2\r\n\t\"q\" 'single' back\\slash \\' end\\",
                    "small": 1.5e-7, "huge": 1e300, "neg": -3, "pi": 3.25,
                    "list": [1, "a", null, [true]], "nested": {"k": {"z": 1}},
                    "embedding": [0.5, 0.25],
                })),
            );
            let b = st.insert_node(vec!["has space".into()], props(json!({"x-y": 1})));
            let c = st.insert_node(vec![], HashMap::new());
            st.insert_edge(a.clone(), b.clone(), "KNOWS".into(), props(json!({"w": 2})))
                .unwrap();
            st.insert_edge(b.clone(), c.clone(), "weird type".into(), HashMap::new())
                .unwrap();
            st.insert_edge(c.clone(), c, "SELF".into(), HashMap::new())
                .unwrap();
        }
        let copy = reimport(&db);
        assert_eq!(shape(&copy), shape(&db));
        assert_eq!(copy.lock_store().edges.len(), 3);
        // The embedding property is indexed again on import.
        let st = copy.lock_store();
        assert!(st
            .nodes
            .values()
            .any(|n| n.embedding == Some(vec![0.5, 0.25])));
    }

    #[test]
    fn export_round_trips_generated_strings() {
        let alphabet = [
            '\\', '\'', '"', 'a', '\n', '\r', '\t', '`', 'é', ' ', 'u', 'n',
        ];
        let mut seed: u64 = 0x2545_f491_4f6c_dd1d;
        let db = GraphDB::new(None).unwrap();
        for _ in 0..300 {
            let mut s = String::new();
            for _ in 0..(seed % 13) {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                s.push(alphabet[(seed >> 33) as usize % alphabet.len()]);
            }
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            db.lock_store()
                .insert_node(vec!["S".into()], props(json!({ "s": s })));
        }
        assert_eq!(shape(&reimport(&db)), shape(&db));
    }

    #[test]
    fn exported_names_cannot_inject_clauses() {
        let db = GraphDB::new(None).unwrap();
        db.lock_store().insert_node(
            vec!["X) CREATE (pwn:Evil".into()],
            props(json!({"k: 1}) CREATE (pwn2:Evil {z": "v'}) CREATE (pwn3:Evil {a:'"})),
        );
        let copy = reimport(&db);
        assert_eq!(shape(&copy), shape(&db));
        assert_eq!(copy.lock_store().nodes.len(), 1);
    }

    #[test]
    fn backtick_in_a_name_fails_import_without_creating_anything() {
        let db = GraphDB::new(None).unwrap();
        db.lock_store()
            .insert_node(vec!["a`) CREATE (pwn:Evil) //".into()], HashMap::new());
        let copy = GraphDB::new(None).unwrap();
        assert!(copy.execute_cypher(&db.export_cypher()).is_err());
        assert_eq!(copy.lock_store().nodes.len(), 0);
    }

    #[test]
    fn empty_graph_exports_nothing() {
        assert_eq!(GraphDB::new(None).unwrap().export_cypher(), "");
    }
}
