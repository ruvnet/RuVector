//! Node and edge removal.
//!
//! Every index that names a node or edge is cleaned here, so nothing keeps an ID
//! after the element it names is gone: the label and type indexes, both adjacency
//! lists of every incident edge, and the hyperedges the node belonged to.

use crate::cypher_exec::Store;

impl Store<'_> {
    /// Removes a node, its incident edges and the hyperedges that contain it.
    /// Returns false if the node does not exist.
    pub(crate) fn remove_node(&mut self, id: &str) -> bool {
        let Some(node) = self.nodes.remove(id) else {
            return false;
        };
        for label in &node.labels {
            if let Some(ids) = self.labels_index.get_mut(label) {
                ids.retain(|n| n != id);
                if ids.is_empty() {
                    self.labels_index.remove(label);
                }
            }
        }

        let mut incident = self.node_edges_out.remove(id).unwrap_or_default();
        incident.extend(self.node_edges_in.remove(id).unwrap_or_default());
        for eid in incident {
            // A self-loop is listed twice; the second removal is a no-op.
            self.remove_edge(&eid);
        }

        self.hyperedges
            .retain(|_, h| !h.nodes.iter().any(|n| n == id));
        self.hypergraph.remove_entity(&id.to_string(), true);
        true
    }

    /// Removes an edge from the edge map and from every index that lists it.
    /// Returns false if the edge does not exist.
    pub(crate) fn remove_edge(&mut self, id: &str) -> bool {
        let Some(edge) = self.edges.remove(id) else {
            return false;
        };
        if let Some(ids) = self.node_edges_out.get_mut(&edge.from) {
            ids.retain(|e| e != id);
        }
        if let Some(ids) = self.node_edges_in.get_mut(&edge.to) {
            ids.retain(|e| e != id);
        }
        if let Some(ids) = self.edge_types_index.get_mut(&edge.edge_type) {
            ids.retain(|e| e != id);
            if ids.is_empty() {
                self.edge_types_index.remove(&edge.edge_type);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use crate::GraphDB;
    use serde_json::json;
    use std::collections::HashMap;

    /// Asserts that every ID held by an index names an element that exists.
    fn assert_no_dangling(db: &GraphDB) {
        let st = db.lock_store();
        for ids in st.labels_index.values() {
            assert!(ids.iter().all(|n| st.nodes.contains_key(n)), "labels_index");
        }
        for ids in st.edge_types_index.values() {
            assert!(
                ids.iter().all(|e| st.edges.contains_key(e)),
                "edge_types_index"
            );
        }
        for (n, ids) in st.node_edges_out.iter().chain(st.node_edges_in.iter()) {
            assert!(st.nodes.contains_key(n), "adjacency key {}", n);
            assert!(
                ids.iter().all(|e| st.edges.contains_key(e)),
                "adjacency of {}",
                n
            );
        }
        for e in st.edges.values() {
            assert!(st.nodes.contains_key(&e.from) && st.nodes.contains_key(&e.to));
        }
        for h in st.hyperedges.values() {
            assert!(
                h.nodes.iter().all(|n| st.nodes.contains_key(n)),
                "hyperedge"
            );
        }
        assert_eq!(st.hypergraph.stats().total_hyperedges, st.hyperedges.len());
    }

    fn node(db: &GraphDB, label: &str) -> String {
        let props: HashMap<_, _> = [("embedding".to_string(), json!([1.0, 0.0]))].into();
        db.lock_store().insert_node(vec![label.to_string()], props)
    }

    fn edge(db: &GraphDB, from: &str, to: &str, ty: &str) -> String {
        db.lock_store()
            .insert_edge(from.into(), to.into(), ty.into(), HashMap::new())
            .unwrap()
    }

    #[test]
    fn delete_node_leaves_no_dangling_references() {
        let db = GraphDB::new(None).unwrap();
        let (a, b, c) = (node(&db, "A"), node(&db, "B"), node(&db, "C"));
        edge(&db, &a, &b, "R");
        edge(&db, &b, &c, "R");
        edge(&db, &b, &b, "SELF");
        let kept = edge(&db, &c, &a, "R");
        let h = db
            .create_hyperedge(
                vec![a.clone(), b.clone()],
                "ab".into(),
                Some(vec![1.0, 0.0]),
                None,
            )
            .unwrap();
        db.create_hyperedge(
            vec![a.clone(), c.clone()],
            "ac".into(),
            Some(vec![0.0, 1.0]),
            None,
        )
        .unwrap();

        assert!(db.delete_node(b.clone()));
        assert!(!db.delete_node(b.clone()));
        assert_no_dangling(&db);
        assert!(db.get_hyperedge(h).is_none());
        assert!(!db.lock_store().labels_index.contains_key("B"));
        assert!(!db.lock_store().edge_types_index.contains_key("SELF"));
        assert_eq!(db.lock_store().node_edges_out[&a], Vec::<String>::new());
        assert_eq!(db.lock_store().edge_types_index["R"], vec![kept]);
        assert_eq!(db.lock_store().hyperedges.len(), 1);
    }

    #[test]
    fn delete_edge_cleans_type_index() {
        let db = GraphDB::new(None).unwrap();
        let (a, b) = (node(&db, "A"), node(&db, "B"));
        let e = edge(&db, &a, &b, "R");
        assert!(db.delete_edge(e.clone()));
        assert!(!db.delete_edge(e));
        assert_no_dangling(&db);
        assert!(!db.lock_store().edge_types_index.contains_key("R"));
    }
}
