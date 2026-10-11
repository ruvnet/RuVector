use ruvector_graph::{GraphDB, NodeBuilder, PropertyValue};

#[test]
fn replacement_removes_old_label_and_property_membership() {
    let db = GraphDB::new();
    db.create_node(
        NodeBuilder::new()
            .id("same")
            .label("Old")
            .property("state", "old")
            .build(),
    )
    .unwrap();
    db.create_node(
        NodeBuilder::new()
            .id("same")
            .label("New")
            .property("state", "new")
            .build(),
    )
    .unwrap();
    assert!(db.get_nodes_by_label("Old").is_empty());
    assert!(db
        .get_nodes_by_property("state", &PropertyValue::from("old"))
        .is_empty());
    assert_eq!(db.get_nodes_by_label("New").len(), 1);
    assert_eq!(
        db.get_nodes_by_property("state", &PropertyValue::from("new"))
            .len(),
        1
    );
}

#[test]
fn deleting_replacement_leaves_no_index_ghosts() {
    let db = GraphDB::new();
    db.create_node(
        NodeBuilder::new()
            .id("same")
            .label("Old")
            .property("state", "old")
            .build(),
    )
    .unwrap();
    db.create_node(
        NodeBuilder::new()
            .id("same")
            .label("New")
            .property("state", "new")
            .build(),
    )
    .unwrap();
    db.delete_node("same").unwrap();
    assert!(db.node_ids_by_label("Old").is_empty());
    assert!(db.node_ids_by_label("New").is_empty());
}

#[test]
fn ordinary_creation_and_explicit_update_control() {
    let db = GraphDB::new();
    db.create_node(
        NodeBuilder::new()
            .id("one")
            .label("One")
            .property("state", "old")
            .build(),
    )
    .unwrap();
    db.create_node(NodeBuilder::new().id("two").label("Two").build())
        .unwrap();
    db.update_node("one", |node| {
        node.set_property("state", PropertyValue::from("new"))
    })
    .unwrap();
    assert_eq!(db.get_nodes_by_label("One").len(), 1);
    assert_eq!(db.get_nodes_by_label("Two").len(), 1);
    assert!(db
        .get_nodes_by_property("state", &PropertyValue::from("old"))
        .is_empty());
}

#[test]
fn edge_replacement_refreshes_type_and_adjacency() {
    use ruvector_graph::EdgeBuilder;
    let db = GraphDB::new();
    for id in ["a", "b", "c"] {
        db.create_node(NodeBuilder::new().id(id).build()).unwrap();
    }
    db.create_edge(
        EdgeBuilder::new("a".into(), "b".into(), "OLD")
            .id("same-edge")
            .build(),
    )
    .unwrap();
    db.create_edge(
        EdgeBuilder::new("c".into(), "b".into(), "NEW")
            .id("same-edge")
            .build(),
    )
    .unwrap();
    assert!(db.get_edges_by_type("OLD").is_empty());
    assert!(db.get_outgoing_edges(&"a".to_string()).is_empty());
    assert_eq!(db.get_edges_by_type("NEW").len(), 1);
    assert_eq!(db.get_outgoing_edges(&"c".to_string()).len(), 1);
}

#[test]
fn hyperedge_replacement_refreshes_membership() {
    use ruvector_graph::Hyperedge;
    let db = GraphDB::new();
    for id in ["a", "b", "c"] {
        db.create_node(NodeBuilder::new().id(id).build()).unwrap();
    }
    db.create_hyperedge(Hyperedge::with_id(
        "same-hyperedge".into(),
        vec!["a".into(), "b".into()],
        "OLD",
    ))
    .unwrap();
    db.create_hyperedge(Hyperedge::with_id(
        "same-hyperedge".into(),
        vec!["b".into(), "c".into()],
        "NEW",
    ))
    .unwrap();
    assert!(db.get_hyperedges_by_node(&"a".to_string()).is_empty());
    assert_eq!(db.get_hyperedges_by_node(&"c".to_string()).len(), 1);
}

#[test]
fn concurrent_replacements_leave_only_live_memberships() {
    use std::sync::{Arc, Barrier};
    let db = Arc::new(GraphDB::new());
    let barrier = Arc::new(Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|i| {
            let db = db.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                db.create_node(
                    NodeBuilder::new()
                        .id("same")
                        .label(format!("Label{i}"))
                        .property("version", i as i64)
                        .build(),
                )
                .unwrap();
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    let live = db.get_node("same").unwrap();
    for i in 0..8 {
        assert_eq!(
            db.get_nodes_by_label(&format!("Label{i}")).len(),
            usize::from(live.has_label(&format!("Label{i}")))
        );
        let expected = PropertyValue::from(i as i64);
        assert_eq!(
            db.get_nodes_by_property("version", &expected).len(),
            usize::from(live.get_property("version") == Some(&expected))
        );
    }
}

#[test]
fn invalid_relationship_replacement_preserves_existing_control() {
    use ruvector_graph::EdgeBuilder;
    let db = GraphDB::new();
    for id in ["a", "b"] {
        db.create_node(NodeBuilder::new().id(id).build()).unwrap();
    }
    db.create_edge(
        EdgeBuilder::new("a".into(), "b".into(), "OLD")
            .id("edge")
            .build(),
    )
    .unwrap();
    assert!(db
        .create_edge(
            EdgeBuilder::new("missing".into(), "b".into(), "NEW")
                .id("edge")
                .build()
        )
        .is_err());
    assert_eq!(db.get_edges_by_type("OLD").len(), 1);
    assert_eq!(db.get_outgoing_edges(&"a".to_string()).len(), 1);
}
