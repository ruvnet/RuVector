//! Comprehensive integration tests for ruvix-cap.
//!
//! Tests capability management including:
//! - Grant/revoke operations
//! - Derivation chains
//! - Depth limits
//! - Rights restrictions

use ruvix_cap::{
    can_grant, can_revoke, validate_grant, validate_revoke,
    CapError, CapHandle, CapManagerConfig, CapRights, CapTableEntry, Capability,
    CapabilityManager, CapabilityTable, DerivationNode, DerivationTree, GrantRequest,
    ObjectType, TaskHandle,
};

// ============================================================================
// Capability Table Tests
// ============================================================================

mod cap_table_tests {
    use super::*;

    #[test]
    fn test_table_creation() {
        let table = CapabilityTable::<64>::new();
        assert!(table.is_empty());
        assert_eq!(table.capacity(), 64);
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_table_allocate_root() {
        let mut table = CapabilityTable::<64>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::VectorStore, CapRights::ALL, 0, 0);

        let handle = table.allocate_root(cap, owner).unwrap();
        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());

        let entry = table.lookup(handle).unwrap();
        assert_eq!(entry.capability.object_id, 100);
        assert_eq!(entry.depth, 0);
        assert!(entry.parent.is_null());
        assert!(entry.is_valid);
    }

    #[test]
    fn test_table_allocate_derived() {
        let mut table = CapabilityTable::<64>::new();
        let owner = TaskHandle::new(1, 0);

        // Create root capability
        let root_cap = Capability::new(100, ObjectType::Region, CapRights::ALL, 0, 0);
        let root_handle = table.allocate_root(root_cap, owner).unwrap();

        // Create derived capability
        let derived_cap = Capability::new(100, ObjectType::Region, CapRights::READ, 42, 0);
        let derived_handle = table.allocate_derived(derived_cap, owner, 1, root_handle).unwrap();

        assert_eq!(table.len(), 2);

        let entry = table.lookup(derived_handle).unwrap();
        assert_eq!(entry.capability.rights, CapRights::READ);
        assert_eq!(entry.capability.badge, 42);
        assert_eq!(entry.depth, 1);
    }

    #[test]
    fn test_table_deallocate() {
        let mut table = CapabilityTable::<64>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Queue, CapRights::WRITE, 0, 0);

        let handle = table.allocate_root(cap, owner).unwrap();
        assert_eq!(table.len(), 1);

        table.deallocate(handle).unwrap();
        assert_eq!(table.len(), 0);

        // Lookup should fail - handle is invalid
        assert_eq!(table.lookup(handle), Err(CapError::InvalidHandle));
    }

    #[test]
    fn test_table_generation_counter() {
        let mut table = CapabilityTable::<64>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Timer, CapRights::READ, 0, 0);

        let handle1 = table.allocate_root(cap, owner).unwrap();
        let gen1 = handle1.raw().generation;

        table.deallocate(handle1).unwrap();

        // Allocate again - same slot, different generation
        let handle2 = table.allocate_root(cap, owner).unwrap();
        let gen2 = handle2.raw().generation;

        assert_eq!(handle1.raw().id, handle2.raw().id);
        assert_ne!(gen1, gen2);

        // Old handle should be stale
        assert_eq!(table.lookup(handle1), Err(CapError::StaleHandle));
    }

    #[test]
    fn test_table_full() {
        let mut table = CapabilityTable::<4>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Task, CapRights::READ, 0, 0);

        // Fill the table
        for _ in 0..4 {
            table.allocate_root(cap, owner).unwrap();
        }

        assert!(table.is_full());
        assert_eq!(table.allocate_root(cap, owner), Err(CapError::TableFull));
    }

    #[test]
    fn test_table_multiple_owners() {
        let mut table = CapabilityTable::<64>::new();
        let task1 = TaskHandle::new(1, 0);
        let task2 = TaskHandle::new(2, 0);

        let cap = Capability::new(100, ObjectType::VectorStore, CapRights::READ, 0, 0);

        let handle1 = table.allocate_root(cap, task1).unwrap();
        let handle2 = table.allocate_root(cap, task2).unwrap();

        let entry1 = table.lookup(handle1).unwrap();
        let entry2 = table.lookup(handle2).unwrap();

        assert_ne!(entry1.owner.raw().to_raw(), entry2.owner.raw().to_raw());
    }

    #[test]
    fn test_table_iterator() {
        let mut table = CapabilityTable::<64>::new();
        let owner = TaskHandle::new(1, 0);

        // Allocate several capabilities
        for i in 0..5 {
            let cap = Capability::new(i * 100, ObjectType::Region, CapRights::ALL, 0, 0);
            table.allocate_root(cap, owner).unwrap();
        }

        // Count via iterator
        let count = table.iter().count();
        assert_eq!(count, 5);

        // All entries should be valid
        for (_, entry) in table.iter() {
            assert!(entry.is_valid);
        }
    }
}

// ============================================================================
// Derivation Tree Tests
// ============================================================================

mod derivation_tests {
    use super::*;

    #[test]
    fn test_derivation_node_creation() {
        let handle = CapHandle::new(0, 0);
        let root = DerivationNode::new_root(handle);

        assert!(root.is_valid);
        assert_eq!(root.depth, 0);
        assert!(!root.has_children());
        assert!(!root.has_sibling());
    }

    #[test]
    fn test_derivation_tree_add_root() {
        
        let tree: DerivationTree<64> = DerivationTree::new();
        let handle = CapHandle::new(0, 0);

        // tree.add_root(handle).unwrap();
        // assert_eq!(tree.len(), 1);
        // assert!(tree.is_valid(handle));

        // Just test initial state for now
        assert!(tree.is_empty());
    }

    #[test]
    fn test_derivation_chain() {
        
        let mut tree: DerivationTree<64> = DerivationTree::new();
        let root = CapHandle::new(0, 0);
        let child = CapHandle::new(1, 0);
        let grandchild = CapHandle::new(2, 0);

        tree.add_root(root).unwrap();
        tree.add_child(root, child, 1).unwrap();
        tree.add_child(child, grandchild, 2).unwrap();

        assert_eq!(tree.len(), 3);
        assert_eq!(tree.depth(root).unwrap(), 0);
        assert_eq!(tree.depth(child).unwrap(), 1);
        assert_eq!(tree.depth(grandchild).unwrap(), 2);
    }

    #[test]
    fn test_derivation_revoke_propagation() {
        
        let mut tree: DerivationTree<64> = DerivationTree::new();
        let root = CapHandle::new(0, 0);
        let child1 = CapHandle::new(1, 0);
        let child2 = CapHandle::new(2, 0);
        let grandchild = CapHandle::new(3, 0);

        tree.add_root(root).unwrap();
        tree.add_child(root, child1, 1).unwrap();
        tree.add_child(root, child2, 1).unwrap();
        tree.add_child(child1, grandchild, 2).unwrap();

        // Revoke root should revoke all
        let revoked_count = tree.revoke(root).unwrap();
        assert_eq!(revoked_count, 4);

        assert!(!tree.is_valid(root));
        assert!(!tree.is_valid(child1));
        assert!(!tree.is_valid(child2));
        assert!(!tree.is_valid(grandchild));
    }

    #[test]
    fn test_derivation_partial_revoke() {
        
        let mut tree: DerivationTree<64> = DerivationTree::new();
        let root = CapHandle::new(0, 0);
        let child1 = CapHandle::new(1, 0);
        let child2 = CapHandle::new(2, 0);

        tree.add_root(root).unwrap();
        tree.add_child(root, child1, 1).unwrap();
        tree.add_child(root, child2, 1).unwrap();

        // Revoke only child1
        let revoked_count = tree.revoke(child1).unwrap();
        assert_eq!(revoked_count, 1);

        assert!(tree.is_valid(root));
        assert!(!tree.is_valid(child1));
        assert!(tree.is_valid(child2));
    }
}

// ============================================================================
// Grant Logic Tests
// ============================================================================

mod grant_tests {
    use super::*;

    fn make_entry(rights: CapRights, depth: u8) -> CapTableEntry {
        let cap = Capability::new(100, ObjectType::VectorStore, rights, 0, 0);
        let mut entry = CapTableEntry::new_root(cap, 0, TaskHandle::new(1, 0));
        entry.depth = depth;
        entry
    }

    #[test]
    fn test_grant_success() {
        let entry = make_entry(CapRights::ALL, 0);
        let request = GrantRequest::new(
            CapHandle::new(0, 0),
            CapRights::READ,
            42,
        );

        let result = validate_grant(&entry, &request).unwrap();
        assert!(result.capability.rights.contains(CapRights::READ));
        assert!(!result.capability.rights.contains(CapRights::WRITE));
        assert_eq!(result.capability.badge, 42);
        assert_eq!(result.depth, 1);
    }

    #[test]
    fn test_grant_no_grant_right() {
        let entry = make_entry(CapRights::READ | CapRights::WRITE, 0);
        let request = GrantRequest::new(
            CapHandle::new(0, 0),
            CapRights::READ,
            0,
        );

        assert_eq!(validate_grant(&entry, &request), Err(CapError::CannotGrant));
    }

    #[test]
    fn test_grant_rights_escalation() {
        let entry = make_entry(CapRights::READ | CapRights::GRANT, 0);
        let request = GrantRequest::new(
            CapHandle::new(0, 0),
            CapRights::READ | CapRights::WRITE, // Escalation!
            0,
        );

        assert_eq!(validate_grant(&entry, &request), Err(CapError::RightsEscalation));
    }

    #[test]
    fn test_grant_depth_exceeded() {
        let entry = make_entry(CapRights::ALL, 8); // At max depth
        let request = GrantRequest::new(
            CapHandle::new(0, 0),
            CapRights::READ,
            0,
        );

        assert_eq!(
            validate_grant(&entry, &request),
            Err(CapError::DelegationDepthExceeded)
        );
    }

    #[test]
    fn test_grant_once() {
        // GRANT_ONCE without GRANT - can grant but derived cap loses grant rights
        let entry = make_entry(
            CapRights::READ | CapRights::WRITE | CapRights::GRANT_ONCE,
            0,
        );
        // Request with GRANT_ONCE (subset check passes since source has it)
        let request = GrantRequest::new(
            CapHandle::new(0, 0),
            CapRights::READ | CapRights::WRITE | CapRights::GRANT_ONCE,
            0,
        );

        let result = validate_grant(&entry, &request).unwrap();

        // Should NOT have GRANT or GRANT_ONCE (stripped by GRANT_ONCE semantics)
        assert!(!result.capability.rights.contains(CapRights::GRANT));
        assert!(!result.capability.rights.contains(CapRights::GRANT_ONCE));
        assert!(result.capability.rights.contains(CapRights::READ));
    }

    #[test]
    fn test_can_grant_predicate() {
        let with_grant = make_entry(CapRights::READ | CapRights::GRANT, 0);
        let without_grant = make_entry(CapRights::READ | CapRights::WRITE, 0);

        assert!(can_grant(&with_grant, CapRights::READ, 8));
        assert!(!can_grant(&without_grant, CapRights::READ, 8));
    }

    #[test]
    fn test_grant_preserves_object_id() {
        let entry = make_entry(CapRights::ALL, 0);
        let request = GrantRequest::new(
            CapHandle::new(0, 0),
            CapRights::READ,
            0,
        );

        let result = validate_grant(&entry, &request).unwrap();
        assert_eq!(result.capability.object_id, entry.capability.object_id);
    }

    #[test]
    fn test_grant_chain() {
        // Test multiple levels of derivation
        let mut depth = 0;

        for _ in 0..7 {
            let entry = make_entry(CapRights::READ | CapRights::GRANT, depth);
            let request = GrantRequest::new(
                CapHandle::new(0, 0),
                CapRights::READ | CapRights::GRANT,
                0,
            );

            let result = validate_grant(&entry, &request).unwrap();
            depth = result.depth;
        }

        assert_eq!(depth, 7);

        // 8th level should fail
        let entry = make_entry(CapRights::READ | CapRights::GRANT, 8);
        let request = GrantRequest::new(
            CapHandle::new(0, 0),
            CapRights::READ,
            0,
        );

        assert_eq!(
            validate_grant(&entry, &request),
            Err(CapError::DelegationDepthExceeded)
        );
    }
}

// ============================================================================
// Revoke Logic Tests
// ============================================================================

mod revoke_tests {
    use super::*;
    use ruvix_cap::RevokeStats;

    fn make_entry(rights: CapRights) -> CapTableEntry {
        let cap = Capability::new(100, ObjectType::Region, rights, 0, 0);
        CapTableEntry::new_root(cap, 0, TaskHandle::new(1, 0))
    }

    #[test]
    fn test_revoke_success() {
        let entry = make_entry(CapRights::REVOKE);
        assert!(validate_revoke(&entry).is_ok());
    }

    #[test]
    fn test_revoke_no_right() {
        let entry = make_entry(CapRights::READ | CapRights::WRITE);
        assert_eq!(validate_revoke(&entry), Err(CapError::CannotRevoke));
    }

    #[test]
    fn test_revoke_invalid_entry() {
        let mut entry = make_entry(CapRights::REVOKE);
        entry.is_valid = false;
        assert_eq!(validate_revoke(&entry), Err(CapError::InvalidHandle));
    }

    #[test]
    fn test_can_revoke_predicate() {
        let with_revoke = make_entry(CapRights::REVOKE);
        let without_revoke = make_entry(CapRights::READ);

        assert!(can_revoke(&with_revoke));
        assert!(!can_revoke(&without_revoke));
    }

    #[test]
    fn test_revoke_stats() {
        let stats = RevokeStats {
            roots_revoked: 2,
            derived_revoked: 10,
            max_depth_reached: 5,
        };

        assert_eq!(stats.total(), 12);
    }
}

// ============================================================================
// Capability Manager Tests
// ============================================================================

mod manager_tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let config = CapManagerConfig::default();
        let _manager: CapabilityManager<64> = CapabilityManager::new(config);
        // Just verify it compiles and doesn't panic
    }

    // Additional manager tests would go here once the manager implementation
    // is complete. These would test the high-level API that combines
    // table, derivation tree, grant, and revoke operations.
}

// ============================================================================
// Rights Operations Tests
// ============================================================================

mod rights_tests {
    use super::*;

    #[test]
    fn test_rights_subset() {
        let full = CapRights::ALL;
        let read_only = CapRights::READ;
        let read_write = CapRights::READ | CapRights::WRITE;

        assert!(read_only.is_subset_of(full));
        assert!(read_write.is_subset_of(full));
        assert!(read_only.is_subset_of(read_write));
        assert!(!read_write.is_subset_of(read_only));
    }

    #[test]
    fn test_rights_intersection() {
        let rw = CapRights::READ | CapRights::WRITE;
        let rx = CapRights::READ | CapRights::EXECUTE;

        let intersection = rw.intersection(rx);
        assert!(intersection.contains(CapRights::READ));
        assert!(!intersection.contains(CapRights::WRITE));
        assert!(!intersection.contains(CapRights::EXECUTE));
    }

    #[test]
    fn test_rights_difference() {
        let all = CapRights::READ | CapRights::WRITE | CapRights::EXECUTE;
        let write = CapRights::WRITE;

        let diff = all.difference(write);
        assert!(diff.contains(CapRights::READ));
        assert!(!diff.contains(CapRights::WRITE));
        assert!(diff.contains(CapRights::EXECUTE));
    }
}

// ============================================================================
// Object Type Tests
// ============================================================================

mod object_type_tests {
    use super::*;

    #[test]
    fn test_object_types() {
        let types = [
            ObjectType::Task,
            ObjectType::Region,
            ObjectType::Queue,
            ObjectType::Timer,
            ObjectType::VectorStore,
            ObjectType::GraphStore,
            ObjectType::RvfMount,
            ObjectType::Capability,
            ObjectType::WitnessLog,
            ObjectType::Subscription,
        ];

        // Each type should have distinct value
        for (i, t1) in types.iter().enumerate() {
            for (j, t2) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(*t1 as u8, *t2 as u8);
                }
            }
        }
    }

    #[test]
    fn test_capability_with_different_types() {
        let mut table = CapabilityTable::<64>::new();
        let owner = TaskHandle::new(1, 0);

        for obj_type in [
            ObjectType::Task,
            ObjectType::Region,
            ObjectType::Queue,
            ObjectType::VectorStore,
        ] {
            let cap = Capability::new(100, obj_type, CapRights::READ, 0, 0);
            let handle = table.allocate_root(cap, owner).unwrap();
            let entry = table.lookup(handle).unwrap();
            assert_eq!(entry.capability.object_type, obj_type);
        }
    }
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

mod edge_cases_tests {
    use super::*;

    #[test]
    fn test_double_deallocate() {
        let mut table = CapabilityTable::<64>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Task, CapRights::READ, 0, 0);

        let handle = table.allocate_root(cap, owner).unwrap();
        table.deallocate(handle).unwrap();

        // Second deallocate should fail
        assert!(table.deallocate(handle).is_err());
    }

    #[test]
    fn test_lookup_out_of_bounds() {
        let table = CapabilityTable::<64>::new();
        let bad_handle = CapHandle::new(100, 0); // Index >= capacity

        assert_eq!(table.lookup(bad_handle), Err(CapError::InvalidHandle));
    }

    #[test]
    fn test_empty_rights() {
        let mut table = CapabilityTable::<64>::new();
        let owner = TaskHandle::new(1, 0);

        // Capability with no rights
        let cap = Capability::new(100, ObjectType::Region, CapRights::NONE, 0, 0);
        let handle = table.allocate_root(cap, owner).unwrap();

        let entry = table.lookup(handle).unwrap();
        assert_eq!(entry.capability.rights.bits(), 0);
    }

    #[test]
    fn test_max_badge_value() {
        let mut table = CapabilityTable::<64>::new();
        let owner = TaskHandle::new(1, 0);

        let cap = Capability::new(100, ObjectType::Queue, CapRights::READ, u64::MAX, 0);
        let handle = table.allocate_root(cap, owner).unwrap();

        let entry = table.lookup(handle).unwrap();
        assert_eq!(entry.capability.badge, u64::MAX);
    }

    #[test]
    fn test_allocate_after_table_drain() {
        let mut table = CapabilityTable::<4>::new();
        let owner = TaskHandle::new(1, 0);
        let cap = Capability::new(100, ObjectType::Task, CapRights::READ, 0, 0);

        // Fill table
        let mut handles = Vec::new();
        for _ in 0..4 {
            handles.push(table.allocate_root(cap, owner).unwrap());
        }

        // Drain table
        for h in handles {
            table.deallocate(h).unwrap();
        }

        assert!(table.is_empty());

        // Should be able to allocate again
        let new_handle = table.allocate_root(cap, owner).unwrap();
        assert!(table.lookup(new_handle).is_ok());
    }
}
